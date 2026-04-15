"""
Crossover Analysis Script

Runs the crossover analysis pipeline on all inputs and saves results to disk.
Designed to run on GPU via SLURM job submission.

Pass --report to additionally generate a failure-reason markdown report after
the pipeline completes (reusing already-loaded model/SAE objects).
"""
import os
import sys
import argparse
import torch
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.nb_utils import setup_notebook, load_transformer_model, load_sae
from src.data.datasets import get_dataset
from src.sae import (
    collect_sae_activations,
    get_xovers_df,
    get_output_swap_bounds,
    swap_outputs,
    build_merged,
    generate_example_visuals,
    generate_markdown,
)


def parse_args():
    p = argparse.ArgumentParser(description="Run crossover analysis pipeline")
    p.add_argument("--model", default="2layer_100dig_64d",
                   help="Model name (without .pt extension)")
    p.add_argument("--sae", default="sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d.pt",
                   help="SAE checkpoint path relative to results/sae_models/")
    p.add_argument("--feature", type=int, default=30, dest="feature_idx",
                   help="SAE feature index to analyse (default: 30)")
    p.add_argument("--results-dir", default="results/xover",
                   help="Directory to write CSV outputs (default: results/xover)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for crossover search (default: 64)")
    p.add_argument("--report", action="store_true",
                   help="Generate a failure-reason markdown report after the pipeline")
    return p.parse_args()


def run_pipeline(args):
    """Run the crossover pipeline and return (results_dir, loaded objects for reporting)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sae_tag = Path(args.sae).stem
    results_dir = Path(args.results_dir) / sae_tag
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CROSSOVER ANALYSIS - GPU JOB")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model:  {args.model}")
    print(f"SAE:    {args.sae}")
    print(f"Feature: {args.feature_idx}")
    print(f"Batch size: {args.batch_size}")
    print(f"Results directory: {results_dir}")
    print("=" * 60)

    _ = setup_notebook(seed=42)

    # [1] Load models
    print("\n[1/6] Loading models...")
    model, model_cfg = load_transformer_model(args.model, device=device)
    d_model = model_cfg['d_model']
    n_digits = model_cfg['n_digits']
    list_len = model_cfg['list_len']
    sep_idx = model_cfg['sep_token_index']

    sae, sae_cfg = load_sae(args.sae, d_model, device=device)

    sae_path = os.path.join('results/sae_models', args.sae)
    sae_checkpoint = torch.load(sae_path, map_location=device, weights_only=False)
    act_mean = sae_checkpoint["act_mean"].to(device)

    # [2] Load dataset
    print("\n[2/6] Loading dataset...")
    train_ds, val_ds = get_dataset(
        n_digits=n_digits, list_len=list_len, no_dupes=False, train_dupes_only=False
    )
    all_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])
    print(f"Total inputs: {len(all_ds)}")

    # [3] Collect SAE activations
    print("\n[3/6] Collecting SAE activations...")
    from torch.utils.data import DataLoader
    all_dl = DataLoader(all_ds, batch_size=128, shuffle=False)

    d1_all, d2_all, sae_acts_all = collect_sae_activations(
        model=model, sae=sae, val_dl=all_dl, act_mean=act_mean,
        layer_idx=0, sep_idx=sep_idx, device=device,
    )

    # [4] Find crossovers
    print(f"\n[4/6] Finding crossovers (feature {args.feature_idx})...")
    xovers_df = get_xovers_df(
        model=model, sae=sae, act_mean=act_mean,
        feature_idx=args.feature_idx,
        d1_all=d1_all, d2_all=d2_all, sae_acts_all=sae_acts_all,
        dataset=all_ds, layer_idx=0, sep_idx=sep_idx, n_digits=n_digits,
        batch_size=args.batch_size, device=device,
    )

    xovers_path = results_dir / f"xovers_feat{args.feature_idx}.csv"
    xovers_df.to_csv(xovers_path, index=False)
    print(f"Saved crossovers to {xovers_path}")
    print(f"  Total inputs: {len(xovers_df)}")
    print(f"  With feature firing: {(xovers_df['feat_orig'] > 0).sum()}")
    print(f"  With crossovers: {((xovers_df['n_o1_xover'] > 0) | (xovers_df['n_o2_xover'] > 0)).sum()}")

    # [5] Get swap bounds
    print(f"\n[5/6] Identifying swap zones...")
    swap_bounds_df = get_output_swap_bounds(xovers_df)

    swap_bounds_path = results_dir / f"swap_bounds_feat{args.feature_idx}.csv"
    swap_bounds_df.to_csv(swap_bounds_path, index=False)
    print(f"Saved swap bounds to {swap_bounds_path}")
    valid_swaps = swap_bounds_df['failure_reason'].isna().sum()
    print(f"  Valid swap zones: {valid_swaps}")
    print(f"  Failed: {len(swap_bounds_df) - valid_swaps}")

    # [6] Verify swaps
    print(f"\n[6/6] Verifying output swaps...")
    swap_results_df = swap_outputs(
        model=model, sae=sae, act_mean=act_mean,
        feature_idx=args.feature_idx,
        swap_bounds_df=swap_bounds_df,
        d1_all=d1_all, d2_all=d2_all, sae_acts_all=sae_acts_all,
        dataset=all_ds, layer_idx=0, sep_idx=sep_idx, n_digits=n_digits,
        device=device,
    )

    swap_results_path = results_dir / f"swap_results_feat{args.feature_idx}.csv"
    swap_results_df.to_csv(swap_results_path, index=False)
    print(f"Saved swap results to {swap_results_path}")
    total = len(swap_results_df)
    swapped = swap_results_df['swapped'].sum()
    print(f"  Successfully swapped: {swapped}/{total} ({swapped/total*100:.1f}%)")

    print("\n" + "=" * 60)
    print("CROSSOVER ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to {results_dir}:")
    print(f"  - {xovers_path.name}")
    print(f"  - {swap_bounds_path.name}")
    print(f"  - {swap_results_path.name}")

    context = dict(
        model=model, sae=sae, act_mean=act_mean,
        d1_all=d1_all, d2_all=d2_all, sae_acts_all=sae_acts_all,
        all_ds=all_ds, all_dl=all_dl,
        xovers_df=xovers_df, swap_bounds_df=swap_bounds_df,
        n_digits=n_digits, list_len=list_len, device=device,
        results_dir=results_dir,
    )
    return context


def run_report(args, context):
    """Generate the failure-reason markdown report using already-loaded objects."""
    feature_idx = args.feature_idx
    results_dir = context["results_dir"]
    output_path = results_dir / f"failure_analysis_feat{feature_idx}.md"
    plots_dir = results_dir / "plots"

    print("\n" + "=" * 60)
    print("GENERATING FAILURE-REASON REPORT")
    print("=" * 60)

    print("Building merged dataset with correctness labels...")
    merged = build_merged(
        context["xovers_df"], context["swap_bounds_df"],
        context["model"], context["all_dl"],
        context["n_digits"], context["list_len"], context["device"],
    )

    present_reasons = merged["failure_reason"].value_counts().index.tolist()
    from src.sae.reporting import FAILURE_ORDER
    ordered = [r for r in FAILURE_ORDER if r in present_reasons]
    ordered += [r for r in present_reasons if r not in FAILURE_ORDER]

    visuals = {}
    for reason in ordered:
        g = merged[merged["failure_reason"] == reason]
        if len(g) == 0:
            continue
        print(f"Generating visuals for '{reason}' ({len(g)} samples)...")
        try:
            plot_rel, crossover_md = generate_example_visuals(
                g, reason,
                context["model"], context["sae"], context["act_mean"],
                context["d1_all"], context["d2_all"], context["sae_acts_all"],
                context["all_ds"],
                feature_idx, plots_dir,
                context["n_digits"], context["list_len"],
            )
            visuals[reason] = (plot_rel, crossover_md)
        except Exception as exc:
            print(f"  Warning: visual generation failed for '{reason}': {exc}")
            visuals[reason] = (None, None)

    print("Rendering markdown...")
    md = generate_markdown(merged, feature_idx, visuals=visuals)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md)
    print(f"Report written to {output_path}")


def main():
    args = parse_args()
    context = run_pipeline(args)
    if args.report:
        run_report(args, context)


if __name__ == "__main__":
    main()
