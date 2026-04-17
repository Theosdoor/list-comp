from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import wandb

WANDB_ENTITY = "theo-farrell99-durham-university"
WANDB_PROJECT = "order-by-scale"
CACHE_PATH = Path(__file__).resolve().parent.parent / "results" / "2layer_sweep_cache.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "grid_search"

CONFIG_KEYS = ["d_model", "n_heads", "use_ln", "use_bias", "use_wv", "use_wo", "use_mlp"]
FLAG_KEYS = ["use_ln", "use_bias", "use_wv", "use_wo", "use_mlp"]


def _coerce_int(value: object, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n", ""}:
            return False
    return bool(value)


def fetch_runs(sweep_ids: list[str]) -> pd.DataFrame:
    api = wandb.Api()
    rows = []
    t0 = time.time()
    for sweep_idx, sweep_id in enumerate(sweep_ids, 1):
        print(f"[{sweep_idx}/{len(sweep_ids)}] Fetching sweep {sweep_id} ...", flush=True)
        sweep = api.sweep(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}")
        all_runs = list(sweep.runs)
        n_total = len(all_runs)
        print(f"  Sweep {sweep_id}: {n_total} runs found", flush=True)
        n_finished = n_skipped = n_no_acc = 0
        for i, run in enumerate(all_runs, 1):
            if i % 20 == 0 or i == n_total:
                elapsed = time.time() - t0
                print(f"  [{i}/{n_total}] elapsed={elapsed:.0f}s  accepted={n_finished}  skipped={n_skipped}  no_acc={n_no_acc}", flush=True)
            if run.state != "finished":
                n_skipped += 1
                continue
            val_accuracy = run.summary.get("final/val_accuracy")
            if val_accuracy is None:
                n_no_acc += 1
                continue
            n_finished += 1
            cfg = run.config or {}
            d_model = _coerce_int(cfg.get("d_model", 64), 64)
            n_heads = _coerce_int(cfg.get("n_heads", 1), 1)
            seed = _coerce_int(cfg.get("seed", -1), -1)
            rows.append(
                {
                    "sweep_id": sweep_id,
                    "run_name": run.name,
                    "run_id": run.id,
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "use_ln": _coerce_bool(cfg.get("use_ln", False), default=False),
                    "use_bias": _coerce_bool(cfg.get("use_bias", False), default=False),
                    "use_wv": _coerce_bool(cfg.get("use_wv", False), default=False),
                    "use_wo": _coerce_bool(cfg.get("use_wo", False), default=False),
                    "use_mlp": _coerce_bool(cfg.get("use_mlp", False), default=False),
                    "seed": seed,
                    "val_accuracy": pd.to_numeric(val_accuracy, errors="coerce"),
                }
            )
        print(f"  Done sweep {sweep_id}: {n_finished} accepted, {n_skipped} non-finished, {n_no_acc} missing acc", flush=True)

    print(f"Fetch complete: {len(rows)} total rows from {len(sweep_ids)} sweeps  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(
            "No completed runs with final/val_accuracy found for sweep ids: "
            + ", ".join(sweep_ids)
        )
    df["val_accuracy"] = pd.to_numeric(df["val_accuracy"], errors="coerce")
    df = df.dropna(subset=["val_accuracy"])
    if df.empty:
        raise SystemExit(
            "No valid val_accuracy values after numeric conversion for sweep ids: "
            + ", ".join(sweep_ids)
        )
    return df


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(CONFIG_KEYS, dropna=False)["val_accuracy"]
    stats = grouped.agg(
        mean="mean",
        std="std",
        max="max",
        min="min",
        median="median",
        n_seeds="count",
    )
    stats["std"] = stats["std"].fillna(0.0)
    return stats.reset_index()


def tf(value: bool) -> str:
    return "T" if value else "F"


def fmt_acc(val: float, std: float = 0.0, bold: bool = False) -> str:
    formatted = f"{val:.4f} $\\pm$ {std:.4f}"
    return f"\\textbf{{{formatted}}}" if bold else formatted


def _is_fffff(row: pd.Series) -> bool:
    return all(not row[key] for key in FLAG_KEYS)


def _top3_threshold(means: pd.Series) -> float:
    top3 = means.nlargest(3)
    return float(top3.iloc[-1]) if len(top3) == 3 else float("-inf")


def make_flags_table(stats: pd.DataFrame) -> str:
    subset = stats[stats["d_model"] == 64].copy()
    subset = subset.sort_values(FLAG_KEYS)
    threshold = _top3_threshold(subset["mean"])

    lines = [
        "\\begin{tabular}{c c c c c c l c}",
        "\\toprule",
        "$d_{model}$ & LN & Bias & $W_V$ & $W_O$ & MLP & Accuracy & Max \\\\",
        "\\midrule",
    ]
    for _, row in subset.iterrows():
        mean_str = fmt_acc(row["mean"], bold=row["mean"] >= threshold)
        lines.append(
            f"{int(row['d_model'])} & {tf(row['use_ln'])} & {tf(row['use_bias'])} & "
            f"{tf(row['use_wv'])} & {tf(row['use_wo'])} & {tf(row['use_mlp'])} & "
            f"{mean_str} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _make_single_var_table(
    stats: pd.DataFrame,
    group_col: str,
    header_label: str,
    extra_mask: "pd.Series | None" = None,
    ascending: bool = True,
) -> str:
    fffff = stats.apply(_is_fffff, axis=1)
    subset = stats[fffff if extra_mask is None else fffff & extra_mask].copy()
    subset = subset.sort_values(group_col, ascending=ascending)
    lines = [
        "\\begin{tabular}{c c c c c c}",
        f"{header_label} & mean & max & min & median & n \\\\",
        "\\midrule",
    ]
    for _, row in subset.iterrows():
        mean_str = fmt_acc(row["mean"], bold=row["mean"] >= BOLD_THRESHOLD)
        lines.append(
            f"{int(row[group_col])} & {mean_str} & {fmt_acc(row['max'])} & "
            f"{fmt_acc(row['min'])} & {fmt_acc(row['median'])} & {int(row['n_seeds'])} \\\\"
        )
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def make_dmodel_table(stats: pd.DataFrame) -> str:
    # Exclude d_model=64 since it's already represented in the flags block.
    return _make_single_var_table(
        stats, "d_model", "d\\_model",
        extra_mask=stats["d_model"] != 64,
        ascending=False,
    )


def _row_latex(row: pd.Series, bold_mean: bool = False, bold_max: bool = False) -> str:
    max_str = f"\\textbf{{{row['max']:.4f}}}" if bold_max else f"{row['max']:.4f}"
    return (
        f"{int(row['d_model'])} & {tf(row['use_ln'])} & {tf(row['use_bias'])} & "
        f"{tf(row['use_wv'])} & {tf(row['use_wo'])} & {tf(row['use_mlp'])} & "
        f"{fmt_acc(row['mean'], std=row['std'], bold=bold_mean)} & "
        f"{max_str} \\\\"
    )


def make_compact_table(stats: pd.DataFrame) -> str:
    fffff_mask = stats.apply(_is_fffff, axis=1)

    baseline = stats[(stats["d_model"] == 64) & fffff_mask].copy()
    flags_sort = ["use_mlp"] + [k for k in FLAG_KEYS if k != "use_mlp"]
    flags_block = stats[(stats["d_model"] == 64) & ~fffff_mask].copy().sort_values(flags_sort)
    dmodel_block = stats[fffff_mask & (stats["d_model"] != 64)].copy().sort_values(
        "d_model", ascending=False
    )

    flags_best_mean = flags_block["mean"].max()
    flags_best_max = flags_block["max"].max()
    dmodel_best_mean = dmodel_block["mean"].max()
    dmodel_best_max = dmodel_block["max"].max()

    lines = [
        "\\begin{tabular}{c c c c c c l c}",
        "\\toprule",
        "$d_{model}$ & LN & Bias & $W_V$ & $W_O$ & MLP & Accuracy & Max \\\\",
        "\\midrule",
    ]
    for _, row in baseline.iterrows():
        lines.append(_row_latex(row, bold_mean=False, bold_max=False))

    lines.append("\\midrule")
    for _, row in flags_block.iterrows():
        lines.append(_row_latex(row,
            bold_mean=row["mean"] == flags_best_mean,
            bold_max=row["max"] == flags_best_max,
        ))

    lines.append("\\midrule")
    for _, row in dmodel_block.iterrows():
        lines.append(_row_latex(row,
            bold_mean=row["mean"] == dmodel_best_mean,
            bold_max=row["max"] == dmodel_best_max,
        ))

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def make_nheads_table(stats: pd.DataFrame) -> str:
    return _make_single_var_table(
        stats, "n_heads", "n\\_heads",
        extra_mask=stats["d_model"] == 64,
    )


def print_top3_fffff(df: pd.DataFrame) -> None:
    mask = (df["d_model"] == 64)
    for key in FLAG_KEYS:
        mask &= ~df[key]
    top3 = df[mask].sort_values("val_accuracy", ascending=False).head(3)
    print("Top 3 fffff models (models/2_layer_sweep/)")
    for _, row in top3.iterrows():
        run_name = str(row["run_name"])
        filename = run_name if run_name.endswith(".pt") else f"{run_name}_acc{row['val_accuracy']:.4f}.pt"
        print(
            f"{filename}   (seed={int(row['seed'])}, acc={row['val_accuracy']:.4f})"
        )


def _load_or_fetch(all_sweep_ids: list[str], use_cache: bool) -> pd.DataFrame:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    cache_df = pd.DataFrame()
    if use_cache and CACHE_PATH.exists():
        cache_df = pd.read_csv(CACHE_PATH)

    cached_ids = set(cache_df["sweep_id"].unique()) if not cache_df.empty else set()
    missing_ids = [sid for sid in all_sweep_ids if sid not in cached_ids]

    if not missing_ids:
        print(f"All {len(all_sweep_ids)} sweeps loaded from cache ({CACHE_PATH})", flush=True)
        return cache_df

    if cached_ids & set(all_sweep_ids):
        print(f"Cache hit for {sorted(cached_ids & set(all_sweep_ids))}; fetching missing: {missing_ids}", flush=True)
    else:
        print(f"No cache found; fetching {len(missing_ids)} sweeps from WandB ...", flush=True)

    for sweep_id in missing_ids:
        new_df = fetch_runs([sweep_id])
        cache_df = pd.concat([cache_df, new_df], ignore_index=True)
        cache_df.to_csv(CACHE_PATH, index=False)
        print(f"  Saved cache after sweep {sweep_id} ({len(cache_df)} total rows)", flush=True)

    return cache_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 2-layer sweep result tables.")
    parser.add_argument("--flags-sweep-ids", nargs="+", required=True)
    parser.add_argument("--dmodel-sweep-ids", nargs="+", required=True)
    parser.add_argument("--nheads-sweep-ids", nargs="+", default=[])
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    all_sweep_ids = list(dict.fromkeys(
        args.flags_sweep_ids + args.dmodel_sweep_ids + args.nheads_sweep_ids
    ))
    print(f"Sweeps to process: {all_sweep_ids}", flush=True)
    df = _load_or_fetch(all_sweep_ids, use_cache=not args.no_cache)
    df = df[df["sweep_id"].isin(all_sweep_ids)].copy()
    print(f"Loaded {len(df)} rows; computing stats ...", flush=True)
    stats = compute_stats(df)
    print(f"Stats computed: {len(stats)} config combinations\n", flush=True)

    table = make_compact_table(stats)
    print(table)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep_tag = "_".join(all_sweep_ids)
    out_path = OUTPUT_DIR / f"2layer_table_{sweep_tag}.tex"
    out_path.write_text(table + "\n")
    print(f"\nSaved to {out_path}", flush=True)

    print_top3_fffff(df)


if __name__ == "__main__":
    main()
