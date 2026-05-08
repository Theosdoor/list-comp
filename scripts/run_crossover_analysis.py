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
from src.models.utils import infer_model_config, load_model as _load_model_direct
from src.utils.runtime import configure_runtime
from src.data.datasets import get_dataset
from src.sae import (
    collect_sae_activations,
    collect_attention_patterns,
    identify_special_features,
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
                   help="Model name (without .pt extension), resolved from models/")
    p.add_argument("--model_path", default=None,
                   help="Full path to model checkpoint (overrides --model; infers config from weights)")
    p.add_argument("--sae", default="sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d.pt",
                   help="SAE checkpoint filename (relative to sae_checkpoints/) or full path")
    p.add_argument("--feature", type=str, default="auto", dest="feature_idx",
                   help="SAE feature index to analyse ('auto' to detect, or an integer to override)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Correlation threshold for special feature detection (default: 0.5)")
    p.add_argument("--max-features", type=int, default=2,
                   help="Max features to run the pipeline on in auto mode (default: 2)")
    p.add_argument("--results-dir", default="results/xover",
                   help="Directory to write CSV outputs (default: results/xover)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for crossover search (default: 64)")
    p.add_argument("--report", action="store_true",
                   help="Generate a failure-reason markdown report after the pipeline")
    return p.parse_args()


def _build_special_features_table(found_sorted, firing_rates, max_features, threshold, sae_tag):
    """Build markdown lines for the special-features summary table."""
    n_found = len(found_sorted)
    n_run = min(n_found, max_features)
    lines = [
        f"# Special Features — {sae_tag}",
        "",
        f"Threshold: {threshold} | Found: {n_found} | Running top {n_run} (--max-features)",
        "",
        "| Feature | Type | Correlation | Firing Rate |",
        "|---------|------|-------------|-------------|",
    ]
    for feat in found_sorted:
        idx = feat["feature_idx"]
        corr = feat["correlation"]
        ftype = feat["type"]
        fr = firing_rates[idx].item()
        sign = "+" if corr >= 0 else ""
        lines.append(f"| {idx} | {ftype} | {sign}{corr:.4f} | {fr:.4f} |")
    if n_found > max_features:
        lines.append("")
        lines.append(f"_Note: only top {n_run} features were run through the pipeline._")
    return lines


def run_pipeline(args):
    """Run the crossover pipeline and return list of per-feature context dicts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_name = Path(args.sae).name
    sae_tag = sae_name[:-3] if sae_name.endswith('.pt') else sae_name
    sae_results_dir = Path(args.results_dir) / sae_tag
    sae_results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CROSSOVER ANALYSIS - GPU JOB")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model:  {args.model_path or args.model}")
    print(f"SAE:    {args.sae}")
    print(f"Feature: {args.feature_idx}")
    if args.feature_idx == "auto":
        print(f"Threshold: {args.threshold}")
        print(f"Max features: {args.max_features}")
    print(f"Batch size: {args.batch_size}")
    print(f"Results directory: {sae_results_dir}")
    print("=" * 60)

    _ = setup_notebook(seed=42)

    # [1/7] Load models
    print("\n[1/7] Loading models...")
    if args.model_path:
        raw_cfg = infer_model_config(args.model_path, device=device)
        d_model = raw_cfg["d_model"]
        n_digits = raw_cfg["d_vocab"] - 2
        list_len = raw_cfg["list_len"]
        sep_idx = list_len
        configure_runtime(list_len=list_len, seq_len=list_len * 2 + 1,
                          vocab=raw_cfg["d_vocab"], device=device)
        model = _load_model_direct(
            args.model_path,
            n_layers=raw_cfg["n_layers"], n_heads=raw_cfg["n_heads"], d_model=d_model,
            ln=raw_cfg.get("use_ln", False), use_bias=raw_cfg.get("use_bias", False),
            use_wv=raw_cfg.get("use_wv", False), use_wo=raw_cfg.get("use_wo", False),
        )
        print(f"✓ Loaded model from {args.model_path}")
    else:
        model, model_cfg = load_transformer_model(args.model, device=device)
        d_model = model_cfg["d_model"]
        n_digits = model_cfg["n_digits"]
        list_len = model_cfg["list_len"]
        sep_idx = model_cfg["sep_token_index"]

    # Resolve SAE path (can be folder or .pt file)
    sae_path = args.sae if os.path.sep in args.sae or os.path.isabs(args.sae) \
        else os.path.join("sae_checkpoints", args.sae)
    
    # load_sae now handles both folder and .pt file paths
    sae, sae_cfg = load_sae(sae_path, d_model, device=device)
    
    # Extract act_mean from config (or load from checkpoint if needed)
    act_mean = sae_cfg.get("act_mean")
    if act_mean is None:
        # Fallback: if act_mean wasn't returned, load directly from checkpoint
        sae_checkpoint = torch.load(
            str(Path(sae_path) if Path(sae_path).suffix == ".pt" else list(Path(sae_path).glob("*.pt"))[0]),
            map_location=device, weights_only=False
        )
        act_mean = sae_checkpoint["act_mean"].to(device)

    # [2/7] Load dataset
    print("\n[2/7] Loading dataset...")
    train_ds, val_ds = get_dataset(
        n_digits=n_digits, list_len=list_len, no_dupes=False, train_dupes_only=False
    )
    all_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])
    print(f"Total inputs: {len(all_ds)}")

    # [3/7] Collect SAE activations
    print("\n[3/7] Collecting SAE activations...")
    from torch.utils.data import DataLoader
    all_dl = DataLoader(all_ds, batch_size=128, shuffle=False)

    d1_all, d2_all, sae_acts_all = collect_sae_activations(
        model=model, sae=sae, val_dl=all_dl, act_mean=act_mean,
        layer_idx=0, sep_idx=sep_idx, device=device,
    )

    # [4/7] Detect special features or resolve override
    if args.feature_idx == "auto":
        print("\n[4/7] Collecting attention patterns and detecting special features...")
        alpha_d1_all, alpha_d2_all = collect_attention_patterns(
            model=model, val_dl=all_dl, layer_idx=0, sep_idx=sep_idx, device=device,
        )
        special_results = identify_special_features(
            sae_acts_all=sae_acts_all,
            alpha_d1_all=alpha_d1_all,
            alpha_d2_all=alpha_d2_all,
            threshold=args.threshold,
        )
        found_sorted = sorted(
            special_results["special_features"],
            key=lambda x: abs(x["correlation"]),
            reverse=True,
        )

        if not found_sorted:
            print(f"\nWARNING: No special features found above threshold {args.threshold} for {sae_tag}.")
            print("Consider lowering --threshold or inspecting the SAE with compare_sae.py.")
            print("Exiting.")
            return []

        firing_rates = (sae_acts_all > 0).float().mean(dim=0)
        table_lines = _build_special_features_table(
            found_sorted, firing_rates, args.max_features, args.threshold, sae_tag
        )
        table_md = "\n".join(table_lines)
        print("\n" + table_md)

        sf_path = sae_results_dir / "special_features.md"
        sf_path.write_text(table_md)
        print(f"\nSaved special features summary to {sf_path}")

        feature_list = [f["feature_idx"] for f in found_sorted[: args.max_features]]
    else:
        print(f"\n[4/7] Feature override: using feature {args.feature_idx}, skipping detection.")
        feature_list = [int(args.feature_idx)]

    # Per-feature loop: steps 5-7
    all_contexts = []
    for feature_idx in feature_list:
        feat_results_dir = sae_results_dir / str(feature_idx)
        feat_results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─' * 60}")
        print(f"Feature {feature_idx}")
        print(f"{'─' * 60}")

        # [5/7] Find crossovers
        print(f"\n[5/7] Finding crossovers (feature {feature_idx})...")
        xovers_df = get_xovers_df(
            model=model, sae=sae, act_mean=act_mean,
            feature_idx=feature_idx,
            d1_all=d1_all, d2_all=d2_all, sae_acts_all=sae_acts_all,
            dataset=all_ds, layer_idx=0, sep_idx=sep_idx, n_digits=n_digits,
            batch_size=args.batch_size, device=device,
        )

        # Reclassify feat_zero as dead_latent if feature never fires
        is_dead_latent = (sae_acts_all[:, feature_idx] > 0).sum() == 0
        if is_dead_latent:
            xovers_df = xovers_df.copy()
            xovers_df.loc[
                xovers_df['o1_failure_reason'] == 'feat_zero', 'o1_failure_reason'
            ] = 'dead_latent'
            print(f"Feature {feature_idx} is a dead latent (never activates); reclassified feat_zero entries upstream.")

        xovers_path = feat_results_dir / f"xovers_feat{feature_idx}.csv"
        xovers_df.to_csv(xovers_path, index=False)
        print(f"Saved crossovers to {xovers_path}")
        print(f"  Total inputs: {len(xovers_df)}")
        print(f"  With feature firing: {(xovers_df['feat_orig'] > 0).sum()}")
        print(f"  With crossovers: {((xovers_df['n_o1_xover'] > 0) | (xovers_df['n_o2_xover'] > 0)).sum()}")

        # [6/7] Get swap bounds
        print(f"\n[6/7] Identifying swap zones...")
        swap_bounds_df = get_output_swap_bounds(xovers_df)

        swap_bounds_path = feat_results_dir / f"swap_bounds_feat{feature_idx}.csv"
        swap_bounds_df.to_csv(swap_bounds_path, index=False)
        print(f"Saved swap bounds to {swap_bounds_path}")
        valid_swaps = swap_bounds_df["failure_reason"].isna().sum()
        print(f"  Valid swap zones: {valid_swaps}")
        print(f"  Failed: {len(swap_bounds_df) - valid_swaps}")

        # [7/7] Verify swaps
        print(f"\n[7/7] Verifying output swaps...")
        swap_results_df = swap_outputs(
            model=model, sae=sae, act_mean=act_mean,
            feature_idx=feature_idx,
            swap_bounds_df=swap_bounds_df,
            d1_all=d1_all, d2_all=d2_all, sae_acts_all=sae_acts_all,
            dataset=all_ds, layer_idx=0, sep_idx=sep_idx, n_digits=n_digits,
            device=device,
        )

        swap_results_path = feat_results_dir / f"swap_results_feat{feature_idx}.csv"
        swap_results_df.to_csv(swap_results_path, index=False)
        print(f"Saved swap results to {swap_results_path}")
        total = len(swap_results_df)
        swapped = swap_results_df["swapped"].sum()
        print(f"  Successfully swapped: {swapped}/{total} ({swapped/total*100:.1f}%)")

        all_contexts.append(dict(
            model=model, sae=sae, act_mean=act_mean,
            d1_all=d1_all, d2_all=d2_all, sae_acts_all=sae_acts_all,
            all_ds=all_ds, all_dl=all_dl,
            xovers_df=xovers_df, swap_bounds_df=swap_bounds_df,
            n_digits=n_digits, list_len=list_len, device=device,
            results_dir=feat_results_dir,
            feature_idx=feature_idx,
        ))

    print("\n" + "=" * 60)
    print("CROSSOVER ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved under {sae_results_dir}")
    for ctx in all_contexts:
        print(f"  Feature {ctx['feature_idx']}: {ctx['results_dir']}")

    return all_contexts


def run_report(args, context):
    """Generate the failure-reason markdown report using already-loaded objects."""
    feature_idx = context["feature_idx"]
    results_dir = context["results_dir"]
    output_path = results_dir / f"failure_analysis_feat{feature_idx}.md"
    plots_dir = results_dir / "plots"

    print("\n" + "=" * 60)
    print(f"GENERATING FAILURE-REASON REPORT (feature {feature_idx})")
    print("=" * 60)

    print("Building merged dataset with correctness labels...")
    merged = build_merged(
        context["xovers_df"], context["swap_bounds_df"],
        context["model"], context["all_dl"],
        context["n_digits"], context["list_len"], context["device"],
    )

    from src.sae.reporting import FAILURE_ORDER, SUMMARY_ONLY

    present_reasons = merged["failure_reason"].value_counts().index.tolist()
    ordered = [r for r in FAILURE_ORDER if r in present_reasons]
    ordered += [r for r in present_reasons if r not in FAILURE_ORDER]

    visuals = {}
    for reason in ordered:
        if reason in SUMMARY_ONLY:
            continue
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
    contexts = run_pipeline(args)
    if args.report:
        for ctx in contexts:
            run_report(args, ctx)


if __name__ == "__main__":
    main()
