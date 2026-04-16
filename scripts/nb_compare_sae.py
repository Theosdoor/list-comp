#%% [markdown]
# # Compare SAE Models
# 
# Loads all SAE checkpoints from sae_models/ and compares key metrics.
# Outputs a markdown table for easy comparison.

#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import glob
import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime

from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

from src.utils.runtime import configure_runtime
from src.models.utils import load_model
from src.models.transformer import parse_model_name_safe
from src.data.datasets import get_dataset
from src.sae import identify_special_features
from src.sae.metrics import compute_sae_downstream_metrics

#%%
# --- Configuration ---
MODEL_NAME = '2layer_100dig_64d'
MODEL_CFG = parse_model_name_safe(MODEL_NAME)
SAE_FOLDER = 'results/sae_models/'
OUTPUT_FILE = f'sae_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
COMPUTE_RECON_ACC = True  # Toggle reconstruction accuracy computation

# Model config
D_MODEL = MODEL_CFG.d_model
N_LAYERS = MODEL_CFG.n_layers
N_DIGITS = MODEL_CFG.n_digits
LIST_LEN = 2
SEP_TOKEN_INDEX = 2

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

#%%
def load_sae(sae_path):
    """Load SAE checkpoint and return (sae, act_mean)."""
    checkpoint = torch.load(sae_path, map_location=DEVICE, weights_only=False)
    cfg = checkpoint.get("cfg", {})

    d_sae = cfg.get("dict_size", cfg.get("d_sae", 256))
    k = cfg.get("k", 4)

    sae = BatchTopKSAE(
        activation_dim=D_MODEL,
        dict_size=d_sae,
        k=k,
    ).to(DEVICE)
    sae.load_state_dict(checkpoint["state_dict"])

    act_mean = checkpoint["act_mean"].to(DEVICE)

    return sae, act_mean

#%%
def collect_activations(model, dataloader, sep_idx=2):
    """Collect SEP token activations and attention weights."""
    activations = []
    d1_all, d2_all = [], []
    alpha_d1_all, alpha_d2_all = [], []
    hook_name = "blocks.0.hook_resid_post"
    attn_hook = "blocks.0.attn.hook_attn_scores"
    
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Collecting activations", leave=False):
            inputs = inputs.to(DEVICE)
            d1_all.append(inputs[:, 0].cpu())
            d2_all.append(inputs[:, 1].cpu())
            
            _, cache = model.run_with_cache(
                inputs, 
                stop_at_layer=1,
                names_filter=[hook_name, attn_hook]
            )
            sep_acts = cache[hook_name][:, sep_idx, :]
            activations.append(sep_acts.cpu())
            
            # Extract attention weights from SEP to d1 and d2
            attn_scores = cache[attn_hook]  # [batch, n_heads, seq_len, seq_len]
            alpha_d1 = attn_scores[:, :, sep_idx, 0].mean(dim=1)  # Average over heads
            alpha_d2 = attn_scores[:, :, sep_idx, 1].mean(dim=1)
            alpha_d1_all.append(alpha_d1.cpu())
            alpha_d2_all.append(alpha_d2.cpu())
    
    return (
        torch.cat(activations),
        torch.cat(d1_all),
        torch.cat(d2_all),
        torch.cat(alpha_d1_all),
        torch.cat(alpha_d2_all)
    )

#%%
def evaluate_sae(sae, act_mean, sep_acts, d1_all, d2_all, n_digits, alpha_d1_all=None, alpha_d2_all=None, model=None, val_dl=None):
    """Compute metrics for a single SAE.

    If model and val_dl are provided and COMPUTE_RECON_ACC=True, computes downstream metrics.
    If alpha_d1_all and alpha_d2_all are provided, computes special features.
    """
    sae.eval()
    
    # Encode all activations
    sep_acts_centered = sep_acts.to(DEVICE) - act_mean
    with torch.no_grad():
        sae_acts = sae.encode(sep_acts_centered, use_threshold=True).cpu()

    d_sae = sae_acts.shape[1]
    
    # L0 Sparsity
    l0 = (sae_acts > 0).float().sum(dim=1).mean().item()
    
    # Dead features
    dead_mask = sae_acts.sum(dim=0) == 0
    n_dead = dead_mask.sum().item()
    dead_pct = 100 * n_dead / d_sae
    
    # Firing rates
    firing_rate = (sae_acts > 0).float().mean(dim=0)
    alive_rates = firing_rate[firing_rate > 0]
    min_firing = alive_rates.min().item() if len(alive_rates) > 0 else 0
    max_firing = firing_rate.max().item()
    mean_firing = alive_rates.mean().item() if len(alive_rates) > 0 else 0
    
    # Reconstruction error and explained variance
    with torch.no_grad():
        recon = sae.decode(sae_acts.to(DEVICE)).cpu()
        sep_acts_centered = sep_acts - act_mean.cpu()
        mse = ((sep_acts_centered - recon) ** 2).mean().item()
        orig_var = (sep_acts_centered ** 2).mean().item()
        explained_var = 1 - (mse / orig_var) if orig_var > 0 else 0
    
    # Top features analysis (most frequently firing)
    top_k_features = 5
    top_indices = torch.argsort(firing_rate, descending=True)[:top_k_features]
    
    top_features_info = []
    for feat_idx in top_indices:
        feat_idx = feat_idx.item()
        feat_acts = sae_acts[:, feat_idx].numpy()
        
        if feat_acts.sum() == 0:
            continue
        
        # Find which digit this feature is most selective for
        d1_selectivity = np.zeros(n_digits)
        d2_selectivity = np.zeros(n_digits)
        for digit in range(n_digits):
            d1_mask = d1_all.numpy() == digit
            d2_mask = d2_all.numpy() == digit
            if d1_mask.sum() > 0:
                d1_selectivity[digit] = feat_acts[d1_mask].mean()
            if d2_mask.sum() > 0:
                d2_selectivity[digit] = feat_acts[d2_mask].mean()
        
        best_d1 = d1_selectivity.argmax()
        best_d2 = d2_selectivity.argmax()
        is_d1 = d1_selectivity.max() > d2_selectivity.max()
        
        top_features_info.append({
            'idx': feat_idx,
            'firing_rate': firing_rate[feat_idx].item(),
            'position': 'D1' if is_d1 else 'D2',
            'best_digit': best_d1 if is_d1 else best_d2,
        })
    
    # Identify special features (attention-correlated)
    special_info = {}
    if alpha_d1_all is not None and alpha_d2_all is not None:
        special_results = identify_special_features(
            sae_acts_all=sae_acts,
            alpha_d1_all=alpha_d1_all,
            alpha_d2_all=alpha_d2_all,
            threshold=0.5
        )
        special_info = {
            'n_special_features': special_results['n_special_features'],
            'special_features_pct': 100 * special_results['n_special_features'] / d_sae,
            'max_correlation': special_results['max_correlation'],
            'mean_abs_correlation': special_results['mean_abs_correlation'],
            'special_features_list': special_results['special_features'],
        }
    
    # Compute downstream metrics (accuracy + CE) in a single two-pass loop
    acc_metrics = {}
    if COMPUTE_RECON_ACC and model is not None and val_dl is not None:
        print("  Computing downstream metrics...", end="", flush=True)
        downstream = compute_sae_downstream_metrics(
            model, sae, val_dl, act_mean,
            layer_idx=0, sep_idx=SEP_TOKEN_INDEX, device=DEVICE
        )
        acc_metrics = {
            'baseline_acc': downstream['baseline_acc'],
            'patched_task_acc': downstream['reconstruction_acc'],
            'acc_drop': downstream['accuracy_drop'],
            'baseline_ce': downstream['baseline_ce'],
            'patched_ce': downstream['patched_ce'],
            'ce_increase': downstream['ce_increase'],
        }
        print(" Done.")
    
    return {
        'l0': l0,
        'd_sae': d_sae,
        'k': sae.k.item(),
        'n_dead': n_dead,
        'dead_pct': dead_pct,
        'n_alive': d_sae - n_dead,
        'min_firing': min_firing,
        'max_firing': max_firing,
        'mean_firing': mean_firing,
        'mse': mse,
        'explained_var': explained_var,
        'top_features': top_features_info,
        **special_info,
        **acc_metrics,
    }

#%%
def generate_markdown_report(results, output_path):
    """Generate markdown comparison report."""
    if not results:
        return "No results to report."

    # Sort by k, then d_sae
    results = sorted(results, key=lambda x: (x['k'], x['d_sae']))

    has_ce = 'baseline_ce' in results[0]
    has_special = 'n_special_features' in results[0]

    lines = [
        "# SAE Sweep Comparison Report\n",
        f"Compared {len(results)} SAE models on {results[0]['n_samples']} samples (full train+val dataset).\n",
        "## Summary Table\n",
        "| Model | d_sae | k | L0 | Dead % | Exp Var | Baseline CE | Patched CE | CE Increase | N Special |",
        "|-------|-------|---|----|--------|---------|-------------|------------|-------------|-----------|",
    ]

    for r in results:
        ce_cols = (
            f" {r['baseline_ce']:.4f} | {r['patched_ce']:.4f} | {r['ce_increase']:.4f} |"
            if has_ce else " — | — | — |"
        )
        n_special = f" {r['n_special_features']} |" if has_special else " — |"
        lines.append(
            f"| {r['name']} | {r['d_sae']} | {r['k']} | {r['l0']:.2f} |"
            f" {r['dead_pct']:.1f}% | {r['explained_var']:.4f} |{ce_cols}{n_special}"
        )

    # ── Firing rate statistics ─────────────────────────────────────────────────
    lines.extend([
        "",
        "## Firing Rate Statistics\n",
        "| Model | Min Firing | Max Firing | Mean Firing |",
        "|-------|------------|------------|-------------|",
    ])
    for r in results:
        lines.append(
            f"| {r['name']} | {r['min_firing']:.4f} | {r['max_firing']:.4f} | {r['mean_firing']:.4f} |"
        )

    # ── Special features ───────────────────────────────────────────────────────
    if has_special:
        lines.extend([
            "",
            "## Special Features (Attention-Correlated)\n",
            "| Model | N Special | Special % | Max Corr | Mean Abs Corr |",
            "|-------|-----------|-----------|----------|---------------|",
        ])
        for r in results:
            lines.append(
                f"| {r['name']} | {r['n_special_features']} | {r['special_features_pct']:.1f}% | "
                f"{r['max_correlation']:.4f} | {r['mean_abs_correlation']:.4f} |"
            )

        lines.extend(["", "### Top Special Features by Model\n"])
        for r in results:
            if r.get('special_features_list'):
                top_special = sorted(
                    r['special_features_list'],
                    key=lambda x: abs(x['correlation']),
                    reverse=True
                )[:5]
                if top_special:
                    lines.append(f"\n**{r['name']}:**")
                    for feat in top_special:
                        lines.append(
                            f"- Feature {feat['feature_idx']}: {feat['type']}, "
                            f"corr={feat['correlation']:.4f}"
                        )

    lines.extend([
        "",
        "## Notes\n",
        "- **L0**: mean active features per token",
        "- **Dead %**: features that never fire on the full dataset (lower = better)",
        "- **Exp Var**: fraction of SEP-activation variance explained by SAE reconstruction (higher = better)",
        "- **Baseline / Patched CE**: output-token cross-entropy with original vs SAE-reconstructed activations",
        "- **CE Increase**: Patched CE − Baseline CE — primary faithfulness metric (lower = better)",
        "- **N Special**: features with |corr| > 0.5 with attention difference (alpha_d1 − alpha_d2)",
        "",
    ])

    report = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(report)

    return report

#%%
def main():
    print(f"Using device: {DEVICE}")
    
    # Setup runtime
    configure_runtime(
        list_len=LIST_LEN,
        seq_len=LIST_LEN * 2 + 1,
        vocab=N_DIGITS + 2,
        device=DEVICE
    )
    
    # Load base model
    model_path = f"models/{MODEL_NAME}.pt"
    model = load_model(
        model_path,
        n_layers=N_LAYERS,
        n_heads=1,
        d_model=D_MODEL,
        ln=False,
        use_bias=False,
        use_wv=False,
        use_wo=False
    )
    print(f"✓ Loaded base model from {model_path}")
    
    # Use full dataset (train+val) for exhaustive analysis — baseline accuracy is
    # also measured on this same set so the comparison is meaningful.
    train_ds, val_ds = get_dataset(
        list_len=LIST_LEN,
        n_digits=N_DIGITS,
    )
    full_dl = DataLoader(ConcatDataset([train_ds, val_ds]), batch_size=2048, shuffle=False)
    val_dl = full_dl  # alias kept for rest of script
    
    # Collect activations and attention weights once
    print("Collecting activations and attention weights...")
    sep_acts, d1_all, d2_all, alpha_d1_all, alpha_d2_all = collect_activations(model, val_dl, SEP_TOKEN_INDEX)
    n_samples = len(sep_acts)
    print(f"✓ Collected {n_samples} samples")
    
    # Find all SAE checkpoints
    sae_paths = sorted(glob.glob(os.path.join(SAE_FOLDER, "**/*.pt"), recursive=True))
    print(f"\nFound {len(sae_paths)} SAE checkpoints")
    
    # Evaluate each SAE
    results = []
    for sae_path in tqdm(sae_paths, desc="Evaluating SAEs"):
        name = os.path.basename(sae_path).replace('.pt', '')
        print(f"\nEvaluating: {name}")
        
        try:
            sae, act_mean = load_sae(sae_path)
            metrics = evaluate_sae(
                sae, act_mean, sep_acts, d1_all, d2_all, N_DIGITS,
                alpha_d1_all=alpha_d1_all, alpha_d2_all=alpha_d2_all,
                model=model, val_dl=val_dl
            )
            metrics['name'] = name
            metrics['n_samples'] = n_samples
            results.append(metrics)
            
            status = f"  L0: {metrics['l0']:.2f}, Dead: {metrics['n_dead']}/{metrics['d_sae']} ({metrics['dead_pct']:.1f}%)"
            if 'patched_task_acc' in metrics:
                status += f", Patched Task Acc: {metrics['patched_task_acc']:.4f}"
            print(status)
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Generate report
    if results:
        report = generate_markdown_report(results, OUTPUT_FILE)
        print(f"\n{'='*60}")
        print(f"✓ Report saved to {OUTPUT_FILE}")
        print(f"{'='*60}\n")
        print(report)
    else:
        print("\nNo SAE models found to compare.")

#%%
if __name__ == "__main__":
    main()
