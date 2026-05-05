#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import glob
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime

from src.utils.runtime import configure_runtime
from src.utils.nb_utils import load_sae as load_sae_from_nb_utils
from src.models.utils import load_model, infer_model_config
from src.models.transformer import parse_model_name_safe
from src.data.datasets import get_dataset
from src.sae import identify_special_features
from src.sae.metrics import compute_sae_downstream_metrics

#%%
# --- Configuration ---
DEFAULT_MODEL_PATH = 'models/2layer_100dig_64d.pt'
SAE_FOLDER = 'sae_checkpoints/'
OUTPUT_FOLDER = 'results/compare_sae/'
OUTPUT_FILE = f'{OUTPUT_FOLDER}sae_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
COMPUTE_RECON_ACC = True

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

#%%
def load_sae(sae_path):
    """Load SAE checkpoint and return (sae, act_mean, base_model_path).
    
    Uses centralized load_sae from nb_utils, which handles folders and .pt files.
    """
    sae_path = str(sae_path)
    
    # First, resolve to the actual .pt file if it's a folder
    if os.path.isdir(sae_path):
        pt_files = list(Path(sae_path).glob("*.pt"))
        if len(pt_files) == 0:
            raise ValueError(f"No .pt files found in SAE directory: {sae_path}")
        sae_path = str(pt_files[0])
    
    # Load checkpoint directly to extract base_model_path
    checkpoint = torch.load(sae_path, map_location=DEVICE, weights_only=False)
    cfg = checkpoint.get("cfg", {})
    d_model = cfg.get("d_model", cfg.get("activation_dim"))
    
    # Use centralized loader (this may warn if .pt was passed, which is fine here)
    sae, sae_cfg = load_sae_from_nb_utils(sae_path, d_model, device=DEVICE)
    
    act_mean = sae_cfg.get("act_mean", checkpoint["act_mean"].to(DEVICE))
    base_model_path = cfg.get("model_path", DEFAULT_MODEL_PATH)
    return sae, act_mean, base_model_path

#%%
def collect_activations(model, dataloader, sep_idx):
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
def evaluate_sae(sae, act_mean, sep_acts, d1_all, d2_all, n_digits, alpha_d1_all=None, alpha_d2_all=None, model=None, val_dl=None, sep_idx=2):
    """Compute metrics for a single SAE.

    If model and val_dl are provided and COMPUTE_RECON_ACC=True, computes downstream metrics.
    If alpha_d1_all and alpha_d2_all are provided, computes special features.
    """
    sae.eval()
    
    # Encode all activations
    sep_acts_centered = sep_acts.to(DEVICE) - act_mean
    with torch.no_grad():
        # JumpRelu SAEs don't support use_threshold parameter
        if hasattr(sae, 'k'):
            # BatchTopKSAE has k attribute
            sae_acts = sae.encode(sep_acts_centered, use_threshold=True).cpu()
        else:
            # JumpReluAutoEncoder and others
            sae_acts = sae.encode(sep_acts_centered).cpu()

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
            layer_idx=0, sep_idx=sep_idx, device=DEVICE
        )
        acc_metrics = {
            'baseline_acc': downstream['baseline_acc'],
            'patched_task_acc': downstream['reconstruction_acc'],
            'acc_drop': downstream['accuracy_drop'],
            'baseline_ce': downstream['baseline_ce'],
            'patched_ce': downstream['patched_ce'],
            'ce_increase': downstream['ce_increase'],
            'loss_recovered': downstream['loss_recovered'],
        }
        print(" Done.")
    
    # Get k value (available for BatchTopKSAE but not JumpReluAutoEncoder)
    k_val = sae.k.item() if hasattr(sae, 'k') else None
    
    return {
        'l0': l0,
        'd_sae': d_sae,
        'k': k_val,
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

    # Sort by d_sae, then actual L0
    results = sorted(results, key=lambda x: (x['d_sae'], x['l0']))

    has_ce = 'baseline_ce' in results[0]
    has_special = 'n_special_features' in results[0]

    lines = [
        "# SAE Sweep Comparison Report\n",
        f"Compared {len(results)} SAE models on {results[0]['n_samples']} samples (full train+val dataset).\n",
        "## Summary Table\n",
        "| Model | d_sae | Actual L0 | Dead % | Loss Recovered | Baseline CE | Patched CE | CE Increase | N Special |",
        "|-------|-------|-----------|--------|----------------|-------------|------------|-------------|-----------|",
    ]

    for r in results:
        lr_val = r.get('loss_recovered')
        lr_str = f"{lr_val:.4f}" if lr_val is not None else "—"
        ce_cols = (
            f" {r['baseline_ce']:.4f} | {r['patched_ce']:.4f} | {r['ce_increase']:.4f} |"
            if has_ce else " — | — | — |"
        )
        n_special = f" {r['n_special_features']} |" if has_special else " — |"
        lines.append(
            f"| {r['name']} | {r['d_sae']} | {r['l0']:.2f} |"
            f" {r['dead_pct']:.1f}% | {lr_str} |{ce_cols}{n_special}"
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
        "- **Actual L0**: mean active features per token (measured on full dataset)",
        "- **Dead %**: features that never fire on the full dataset (lower = better)",
        "- **Loss Recovered**: (H* − H0) / (Horig − H0); fraction of model loss recovered by SAE reconstruction vs zero ablation (higher = better; 1 = perfect, 0 = no better than zero ablation)",
        "- **Baseline / Patched CE**: output-token cross-entropy with original vs SAE-reconstructed activations",
        "- **CE Increase**: Patched CE − Baseline CE — raw faithfulness delta (lower = better)",
        "- **N Special**: features with |corr| > 0.5 with attention difference (alpha_d1 − alpha_d2)",
        "",
    ])

    report = "\n".join(lines)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)

    return report

#%%
def _load_base_model(model_path):
    """Load a transformer and return (model, model_cfg)."""
    cfg = infer_model_config(model_path)
    configure_runtime(
        list_len=cfg['list_len'],
        seq_len=cfg['list_len'] * 2 + 1,
        vocab=cfg['d_vocab'],
        device=DEVICE,
    )
    model = load_model(
        model_path,
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        d_model=cfg['d_model'],
        ln=cfg.get('use_ln', False),
        use_bias=cfg.get('use_bias', False),
        use_wv=cfg.get('use_wv', False),
        use_wo=cfg.get('use_wo', False),
    )
    return model, cfg


def main():
    print(f"Using device: {DEVICE}")

    # Find all SAE checkpoints and group by their base model
    sae_paths = sorted(glob.glob(os.path.join(SAE_FOLDER, "**/*.pt"), recursive=True))
    print(f"Found {len(sae_paths)} SAE checkpoints")

    # Peek at each checkpoint to read base_model_path from cfg
    groups = defaultdict(list)  # model_path -> [sae_path, ...]
    for sae_path in sae_paths:
        try:
            ck = torch.load(sae_path, map_location='cpu', weights_only=False)
            base = ck.get("cfg", {}).get("model_path", DEFAULT_MODEL_PATH)
        except Exception:
            base = DEFAULT_MODEL_PATH
        groups[base].append(sae_path)

    print(f"SAEs span {len(groups)} base model(s):")
    for mp, paths in groups.items():
        print(f"  {mp}  ({len(paths)} SAE(s))")

    results = []

    for model_path, group_sae_paths in groups.items():
        print(f"\n{'='*60}")
        print(f"Base model: {model_path}")
        print(f"{'='*60}")

        if not os.path.exists(model_path):
            print(f"  ✗ Model file not found, skipping {len(group_sae_paths)} SAE(s)")
            continue

        model, model_cfg = _load_base_model(model_path)
        print(f"✓ Loaded model")

        n_digits = model_cfg['d_vocab'] - 2
        list_len = model_cfg['list_len']
        sep_idx = list_len

        configure_runtime(
            list_len=list_len,
            seq_len=list_len * 2 + 1,
            vocab=model_cfg['d_vocab'],
            device=DEVICE,
        )

        train_ds, val_ds = get_dataset(list_len=list_len, n_digits=n_digits)
        full_dl = DataLoader(ConcatDataset([train_ds, val_ds]), batch_size=2048, shuffle=False)

        print("Collecting activations and attention weights...")
        sep_acts, d1_all, d2_all, alpha_d1_all, alpha_d2_all = collect_activations(
            model, full_dl, sep_idx
        )
        n_samples = len(sep_acts)
        print(f"✓ Collected {n_samples} samples")

        for sae_path in tqdm(group_sae_paths, desc="Evaluating SAEs"):
            name = os.path.basename(sae_path).replace('.pt', '')
            print(f"\nEvaluating: {name}")
            try:
                sae, act_mean, _ = load_sae(sae_path)
                metrics = evaluate_sae(
                    sae, act_mean, sep_acts, d1_all, d2_all, n_digits,
                    alpha_d1_all=alpha_d1_all, alpha_d2_all=alpha_d2_all,
                    model=model, val_dl=full_dl, sep_idx=sep_idx,
                )
                metrics['name'] = name
                metrics['n_samples'] = n_samples
                metrics['base_model'] = os.path.basename(model_path)
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
    import argparse
    parser = argparse.ArgumentParser(description="Compare SAE checkpoints")
    parser.add_argument("--sae_folders", type=str, nargs="+", default=None,
                        help="One or more folders to search for SAE checkpoints. If not provided, uses default SAE_FOLDER")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override base model for all SAEs (useful for SAEs trained before model_path was saved in cfg)")
    args = parser.parse_args()

    # Determine which folders to search
    sae_folders = args.sae_folders if args.sae_folders else [SAE_FOLDER]
    
    # Collect SAE paths from all specified folders
    all_sae_paths = []
    for folder in sae_folders:
        sae_paths = sorted(glob.glob(os.path.join(folder, "**/*.pt"), recursive=True))
        all_sae_paths.extend(sae_paths)
        print(f"Found {len(sae_paths)} SAEs in {folder}")
    
    print(f"Total {len(all_sae_paths)} SAE checkpoints across {len(sae_folders)} folder(s)")

    # Group by base model
    groups = defaultdict(list)
    for sae_path in all_sae_paths:
        try:
            ck = torch.load(sae_path, map_location='cpu', weights_only=False)
            base = ck.get("cfg", {}).get("model_path", DEFAULT_MODEL_PATH)
        except Exception:
            base = DEFAULT_MODEL_PATH
        groups[base].append(sae_path)

    print(f"SAEs span {len(groups)} base model(s):")
    for mp, paths in groups.items():
        print(f"  {mp}  ({len(paths)} SAE(s))")

    if args.model_path:
        DEFAULT_MODEL_PATH = args.model_path

    # Reimplement main loop with our collected paths
    print(f"\nUsing device: {DEVICE}")
    results = []

    for model_path, group_sae_paths in groups.items():
        print(f"\n{'='*60}")
        print(f"Base model: {model_path}")
        print(f"{'='*60}")

        if not os.path.exists(model_path):
            print(f"  ✗ Model file not found, skipping {len(group_sae_paths)} SAE(s)")
            continue

        model, model_cfg = _load_base_model(model_path)
        print(f"✓ Loaded model")

        n_digits = model_cfg['d_vocab'] - 2
        list_len = model_cfg['list_len']
        sep_idx = list_len

        configure_runtime(
            list_len=list_len,
            seq_len=list_len * 2 + 1,
            vocab=model_cfg['d_vocab'],
            device=DEVICE,
        )

        train_ds, val_ds = get_dataset(list_len=list_len, n_digits=n_digits)
        full_dl = DataLoader(ConcatDataset([train_ds, val_ds]), batch_size=2048, shuffle=False)

        print("Collecting activations and attention weights...")
        sep_acts, d1_all, d2_all, alpha_d1_all, alpha_d2_all = collect_activations(
            model, full_dl, sep_idx
        )
        n_samples = len(sep_acts)
        print(f"✓ Collected {n_samples} samples")

        for sae_path in tqdm(group_sae_paths, desc="Evaluating SAEs"):
            name = os.path.basename(sae_path).replace('.pt', '')
            print(f"\nEvaluating: {name}")
            try:
                sae, act_mean, _ = load_sae(sae_path)
                metrics = evaluate_sae(
                    sae, act_mean, sep_acts, d1_all, d2_all, n_digits,
                    alpha_d1_all=alpha_d1_all, alpha_d2_all=alpha_d2_all,
                    model=model, val_dl=full_dl, sep_idx=sep_idx,
                )
                metrics['name'] = name
                metrics['n_samples'] = n_samples
                metrics['base_model'] = os.path.basename(model_path)
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
