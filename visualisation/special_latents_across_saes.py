#!/usr/bin/env python3
"""
Plot loss recovered vs actual L0 for SAEs, coloured by number of special latents.

Produces one PDF per SAE type (btk, jumprelu, matryoshka):
    {output_dir}/{sae_type}_loss_recovered.pdf

Loss recovered = (H* - H0) / (H_orig - H0)
    H_orig : baseline CE over output positions
    H*     : CE when SAE reconstruction is patched in at the SEP token
    H0     : CE when the SEP token activation is zero-ablated

Examples:
    python scripts/plot_sae_loss_recovered.py \\
        --sae_dirs sae_checkpoints/ \\
        --model_path models/2layer_100dig_64d.pt

    python scripts/plot_sae_loss_recovered.py \\
        --sae_dirs sae_checkpoints/btk sae_checkpoints/jumprelu \\
        --model_path models/2layer_100dig_64d.pt \\
        --alpha_diff_thresh 0.3 \\
        --output_dir results/plots \\
        --batch_size 1024
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.transformer import make_model
from src.models.utils import infer_model_config
from src.utils.runtime import configure_runtime
from src.data.datasets import get_dataset
from src.sae.loading import instantiate_sae_from_cfg
from src.sae.activation_collection import (
    collect_sae_activations,
    collect_attention_patterns,
    identify_special_features,
)
from src.sae.hooks import _encode_through_sae

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CATEGORY_ORDER = ["0", "1", "2", ">2"]


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_checkpoints(dirs: list[str]) -> list[Path]:
    checkpoints = []
    for d in dirs:
        root = Path(d)
        if root.is_file() and root.suffix == ".pt":
            checkpoints.append(root)
        elif root.is_dir():
            checkpoints.extend(sorted(root.rglob("*.pt")))
        else:
            print(f"  Warning: {d} is not a .pt file or directory, skipping.")
    return checkpoints


# ---------------------------------------------------------------------------
# Per-SAE evaluation
# ---------------------------------------------------------------------------

def compute_loss_recovered(
    model, sae, act_mean, val_dl, sep_idx: int, list_len: int, device: str
) -> tuple[float, float, float, float]:
    """
    Three-pass computation of loss recovered over output positions only.

    Returns:
        (loss_recovered, h_orig, h_star, h0)
    """
    hook_name = "blocks.0.hook_resid_post"

    def sae_patch_hook(activations, hook):
        activations = activations.clone()
        sep_acts = activations[:, sep_idx, :]
        reconstructed = _encode_through_sae(sep_acts, sae, act_mean, decode=True)
        activations[:, sep_idx, :] = reconstructed + act_mean.to(reconstructed.device)
        return activations

    def zero_ablate_hook(activations, hook):
        activations = activations.clone()
        activations[:, sep_idx, :] = 0.0
        return activations

    h_orig_sum = h_star_sum = h0_sum = 0.0
    n_tokens = 0

    with torch.no_grad():
        for inputs, targets in tqdm(val_dl, desc="    loss recovered", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            out_targets = targets[:, list_len + 1:]  # [batch, list_len]
            b, t = out_targets.shape

            orig_logits = model(inputs)[:, list_len + 1:]
            v = orig_logits.shape[-1]

            h_orig_sum += F.cross_entropy(
                orig_logits.reshape(b * t, v),
                out_targets.reshape(b * t),
                reduction="sum",
            ).item()

            star_logits = model.run_with_hooks(
                inputs, fwd_hooks=[(hook_name, sae_patch_hook)]
            )[:, list_len + 1:]
            h_star_sum += F.cross_entropy(
                star_logits.reshape(b * t, v),
                out_targets.reshape(b * t),
                reduction="sum",
            ).item()

            zero_logits = model.run_with_hooks(
                inputs, fwd_hooks=[(hook_name, zero_ablate_hook)]
            )[:, list_len + 1:]
            h0_sum += F.cross_entropy(
                zero_logits.reshape(b * t, v),
                out_targets.reshape(b * t),
                reduction="sum",
            ).item()

            n_tokens += b * t

    h_orig = h_orig_sum / n_tokens
    h_star = h_star_sum / n_tokens
    h0 = h0_sum / n_tokens

    denom = h_orig - h0
    loss_recovered = (h_star - h0) / denom if abs(denom) > 1e-8 else float("nan")

    return loss_recovered, h_orig, h_star, h0


def evaluate_sae(
    model,
    sae,
    act_mean,
    val_dl,
    sep_idx: int,
    list_len: int,
    alpha_diff_thresh: float,
    device: str,
) -> dict:
    """Run all evaluations for a single SAE checkpoint."""

    # Collect SAE latent activations (used for both L0 and special feature detection)
    _, _, sae_acts_all = collect_sae_activations(
        model, sae, val_dl, act_mean,
        layer_idx=0, sep_idx=sep_idx, device=device,
    )

    actual_l0 = (sae_acts_all > 0).float().sum(dim=1).mean().item()

    loss_recovered, h_orig, h_star, h0 = compute_loss_recovered(
        model, sae, act_mean, val_dl, sep_idx, list_len, device
    )

    alpha_d1_all, alpha_d2_all = collect_attention_patterns(
        model, val_dl, layer_idx=0, sep_idx=sep_idx, device=device,
    )
    special_info = identify_special_features(
        sae_acts_all, alpha_d1_all, alpha_d2_all, threshold=alpha_diff_thresh,
    )

    return {
        "actual_l0": actual_l0,
        "loss_recovered": loss_recovered,
        "h_orig": h_orig,
        "h_star": h_star,
        "h0": h0,
        "n_special": special_info["n_special_features"],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def bin_special(n: int) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return ">2"


def make_plot(df_type: pd.DataFrame, sae_type: str, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = dict(zip(CATEGORY_ORDER, sns.color_palette("muted", n_colors=len(CATEGORY_ORDER))))
    present = [c for c in CATEGORY_ORDER if c in df_type["n_special_bin"].values]

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.scatterplot(
        data=df_type,
        x="actual_l0",
        y="loss_recovered",
        hue="n_special_bin",
        hue_order=present,
        palette={c: palette[c] for c in present},
        s=60,
        alpha=0.8,
        ax=ax,
    )

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(0.0, color="grey", linestyle=":",  linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Actual L0 (mean active features per token)")
    ax.set_ylabel("Loss recovered")
    ax.set_title(f"{sae_type} SAE — loss recovered vs L0")
    ax.legend(title="Special latents", loc="lower right", framealpha=0.9)

    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot loss recovered vs actual L0 for SAE checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sae_dirs", nargs="+", required=True,
        help="Directories (or .pt files) to search for SAE checkpoints, recursively.",
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the transformer checkpoint.",
    )
    parser.add_argument(
        "--alpha_diff_thresh", type=float, default=0.5,
        help="Correlation threshold for identifying special latents (default: 0.5).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/sae_plots",
        help="Directory to save plots and metrics CSV (default: results/sae_plots).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512,
        help="Batch size for all forward passes (default: 512).",
    )
    args = parser.parse_args()

    # --- Model setup ---
    model_cfg = infer_model_config(args.model_path)
    n_layers  = model_cfg["n_layers"]
    n_heads   = model_cfg["n_heads"]
    d_model   = model_cfg["d_model"]
    n_digits  = model_cfg["d_vocab"] - 2
    list_len  = model_cfg["list_len"]
    sep_idx   = list_len

    configure_runtime(
        list_len=list_len,
        seq_len=list_len * 2 + 1,
        vocab=n_digits + 2,
        device=DEVICE,
    )

    model = make_model(
        n_layers=n_layers, n_heads=n_heads, d_model=d_model,
        ln=model_cfg.get("use_ln", False),
        use_bias=model_cfg.get("use_bias", False),
        use_wv=model_cfg.get("use_wv", False),
        use_wo=model_cfg.get("use_wo", False),
    )
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"✓ Loaded transformer from {args.model_path}  (d_model={d_model}, n_digits={n_digits}, list_len={list_len})")

    # --- Dataloader ---
    _, val_ds = get_dataset(list_len=list_len, n_digits=n_digits, no_dupes=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"✓ Validation set: {len(val_ds):,} samples")

    # --- Discover checkpoints ---
    checkpoints = find_checkpoints(args.sae_dirs)
    if not checkpoints:
        print("No .pt checkpoints found. Exiting.")
        sys.exit(1)
    print(f"\nFound {len(checkpoints)} checkpoint(s)\n")

    # --- Evaluate ---
    records = []
    for ckpt_path in checkpoints:
        print(f"[{len(records)+1}/{len(checkpoints)}] {ckpt_path.name}")
        try:
            ckpt     = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            cfg      = ckpt.get("cfg", {})
            sae_type = cfg.get("sae_type", "btk")

            sae = instantiate_sae_from_cfg(cfg, d_model, DEVICE)
            sae.load_state_dict(ckpt["state_dict"])
            sae.eval()

            act_mean = ckpt.get("act_mean", torch.zeros(d_model)).to(DEVICE)

            metrics = evaluate_sae(
                model, sae, act_mean, val_dl,
                sep_idx, list_len, args.alpha_diff_thresh, DEVICE,
            )

            records.append({
                "checkpoint": ckpt_path.name,
                "sae_type":   sae_type,
                "d_sae":      cfg.get("dict_size", cfg.get("d_sae")),
                **metrics,
            })
            print(
                f"  L0={metrics['actual_l0']:.2f}  "
                f"loss_recovered={metrics['loss_recovered']:.4f}  "
                f"n_special={metrics['n_special']}  "
                f"(H_orig={metrics['h_orig']:.4f}, H*={metrics['h_star']:.4f}, H0={metrics['h0']:.4f})"
            )

        except Exception as exc:
            print(f"  ✗ Failed: {exc}")
            import traceback
            traceback.print_exc()

    if not records:
        print("\nNo SAEs evaluated successfully. Exiting.")
        sys.exit(1)

    # --- Build DataFrame ---
    df = pd.DataFrame(records)
    df["n_special_bin"] = pd.Categorical(
        df["n_special"].apply(bin_special),
        categories=CATEGORY_ORDER,
        ordered=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "sae_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Metrics saved to {csv_path}")

    # --- Plots ---
    print("\nGenerating plots...")
    for sae_type, df_type in df.groupby("sae_type"):
        n = len(df_type)
        if n < 2:
            print(f"  ⚠ Skipping {sae_type}: only {n} checkpoint(s), need at least 2 for a meaningful plot.")
            continue
        plot_path = output_dir / f"{sae_type}_loss_recovered.pdf"
        make_plot(df_type.copy(), sae_type, plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()