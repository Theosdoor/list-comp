#%% [markdown]
# # Compare SAE Models
# 
# Loads all SAE checkpoints from sae_models/ and compares key metrics.
# Outputs a markdown table for easy comparison.

#%%
import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm

from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

from model_scripts.model_utils import configure_runtime, load_model, parse_model_name_safe
from model_scripts.data import get_dataset

#%%
# --- Configuration ---
MODEL_NAME = '2layer_100dig_64d'
MODEL_CFG = parse_model_name_safe(MODEL_NAME)
SAE_FOLDER = 'results/sae_models/sweep_runs'  # Changed to sweep_runs
OUTPUT_FILE = 'sae_comparison.md'

# Model config
D_MODEL = MODEL_CFG.d_model
N_LAYERS = MODEL_CFG.n_layers
N_DIGITS = MODEL_CFG.n_digits
LIST_LEN = 2
SEP_TOKEN_INDEX = 2

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

#%%
def load_sae(sae_path):
    """Load SAE checkpoint and return (sae, act_mean, cfg, is_legacy).
    
    is_legacy=True means the SAE was trained with the old BatchTopK script
    that doesn't use a learned threshold.
    """
    checkpoint = torch.load(sae_path, map_location=DEVICE, weights_only=False)
    cfg = checkpoint.get("cfg", {})
    
    d_sae = cfg.get("dict_size", cfg.get("d_sae", 256))
    k = cfg.get("k", 4)
    
    sae = BatchTopKSAE(
        activation_dim=D_MODEL,
        dict_size=d_sae,
        k=k
    ).to(DEVICE)
    
    # Handle legacy format (old_sae uses W_enc/W_dec naming)
    old_state_dict = checkpoint["state_dict"]
    is_legacy = "W_enc" in old_state_dict
    
    if is_legacy:
        new_state_dict = {
            "encoder.weight": old_state_dict["W_enc"].T,
            "encoder.bias": old_state_dict["b_enc"],
            "decoder.weight": old_state_dict["W_dec"].T,
            "b_dec": old_state_dict["b_dec"],
        }
        sae.load_state_dict(new_state_dict, strict=False)
    else:
        sae.load_state_dict(old_state_dict)
    
    act_mean = checkpoint["act_mean"].to(DEVICE)
    
    return sae, act_mean, cfg, is_legacy

#%%
def collect_activations(model, dataloader, sep_idx=2):
    """Collect SEP token activations."""
    activations = []
    d1_all, d2_all = [], []
    hook_name = "blocks.0.hook_resid_post"
    
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Collecting activations", leave=False):
            inputs = inputs.to(DEVICE)
            d1_all.append(inputs[:, 0].cpu())
            d2_all.append(inputs[:, 1].cpu())
            
            _, cache = model.run_with_cache(
                inputs, 
                stop_at_layer=1,
                names_filter=hook_name
            )
            sep_acts = cache[hook_name][:, sep_idx, :]
            activations.append(sep_acts.cpu())
    
    return (
        torch.cat(activations),
        torch.cat(d1_all),
        torch.cat(d2_all)
    )

#%%
def evaluate_sae(sae, act_mean, sep_acts, d1_all, d2_all, n_digits, is_legacy=False):
    """Compute metrics for a single SAE.
    
    If is_legacy=True, uses batch-level TopK instead of threshold encoding.
    """
    sae.eval()
    
    # Encode all activations
    sep_acts_centered = sep_acts.to(DEVICE) - act_mean
    with torch.no_grad():
        if is_legacy:
            # Old SAE used batch-level TopK, not threshold
            # use_threshold=False triggers proper TopK behavior
            sae_acts = sae.encode(sep_acts_centered, use_threshold=False).cpu()
        else:
            sae_acts = sae.encode(sep_acts_centered, use_threshold=True).cpu()
    
    n_samples = sae_acts.shape[0]
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
    
    # Reconstruction error
    with torch.no_grad():
        recon = sae.decode(sae_acts.to(DEVICE)).cpu()
        mse = ((sep_acts - act_mean.cpu() - recon) ** 2).mean().item()
    
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
        'top_features': top_features_info,
    }

#%%
def generate_markdown_report(results, output_path):
    """Generate markdown comparison report."""
    if not results:
        return "No results to report."
    
    # Sort by top_k, then d_sae
    results = sorted(results, key=lambda x: (x['k'], x['d_sae']))
    
    lines = [
        "# SAE Sweep Comparison Report\n",
        f"Compared {len(results)} SAE models from sweep runs on {results[0]['n_samples']} samples.\n",
        "",
        "## Summary Table\n",
        "| Model | d_sae | k | L0 | Dead | Dead % | Alive | MSE |",
        "|-------|-------|---|----|----|--------|-------|-----|",
    ]
    
    for r in results:
        lines.append(
            f"| {r['name']} | {r['d_sae']} | {r['k']} | {r['l0']:.2f} | "
            f"{r['n_dead']} | {r['dead_pct']:.1f}% | {r['n_alive']} | {r['mse']:.4f} |"
        )
    
    lines.extend([
        "",
        "## Best Models by top_k\n",
    ])
    
    # Group by top_k and find best in each group
    from itertools import groupby
    for k, group in groupby(results, key=lambda x: x['k']):
        group_list = list(group)
        best_by_mse = min(group_list, key=lambda x: x['mse'])
        best_by_dead = min(group_list, key=lambda x: x['dead_pct'])
        
        lines.append(f"\n### top_k = {k}\n")
        lines.append(f"- Best reconstruction (MSE): **{best_by_mse['name']}** (MSE: {best_by_mse['mse']:.4f}, d_sae={best_by_mse['d_sae']})")
        lines.append(f"- Fewest dead features: **{best_by_dead['name']}** ({best_by_dead['dead_pct']:.1f}%, d_sae={best_by_dead['d_sae']})")
    
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
    
    lines.extend([
        "",
        "## Analysis\n",
        "- **L0**: Average number of active features per sample (lower = sparser)",
        "- **Dead %**: Percentage of features that never fire (lower = better utilization)",
        "- **MSE**: Mean squared reconstruction error (lower = better reconstruction)",
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
    
    # Get validation data
    val_ds, _ = get_dataset(
        list_len=LIST_LEN,
        n_digits=N_DIGITS,
        train_split=1.0,
        no_dupes=False
    )
    val_dl = DataLoader(val_ds, batch_size=2048, shuffle=False)
    
    # Collect activations once
    print("Collecting activations...")
    sep_acts, d1_all, d2_all = collect_activations(model, val_dl, SEP_TOKEN_INDEX)
    n_samples = len(sep_acts)
    print(f"✓ Collected {n_samples} samples")
    
    # Find all SAE checkpoints
    sae_paths = sorted(glob.glob(os.path.join(SAE_FOLDER, "*.pt")))
    print(f"\nFound {len(sae_paths)} SAE checkpoints")
    
    # Evaluate each SAE
    results = []
    for sae_path in sae_paths:
        name = os.path.basename(sae_path).replace('.pt', '')
        print(f"\nEvaluating: {name}")
        
        try:
            sae, act_mean, cfg, is_legacy = load_sae(sae_path)
            metrics = evaluate_sae(sae, act_mean, sep_acts, d1_all, d2_all, N_DIGITS, is_legacy=is_legacy)
            metrics['name'] = name
            metrics['n_samples'] = n_samples
            results.append(metrics)
            
            print(f"  L0: {metrics['l0']:.2f}, Dead: {metrics['n_dead']}/{metrics['d_sae']} ({metrics['dead_pct']:.1f}%)")
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
