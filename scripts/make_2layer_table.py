from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import wandb

WANDB_ENTITY = "theo-farrell99-durham-university"
WANDB_PROJECT = "order-by-scale"
CACHE_PATH = Path(__file__).resolve().parent.parent / "results" / "2layer_sweep_cache.csv"
BOLD_THRESHOLD = 0.90

CONFIG_KEYS = ["d_model", "use_ln", "use_bias", "use_wv", "use_wo", "use_mlp"]
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
    for sweep_id in sweep_ids:
        sweep = api.sweep(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}")
        for run in sweep.runs:
            if run.state != "finished":
                continue
            val_accuracy = run.summary.get("final/val_accuracy")
            if val_accuracy is None:
                continue
            cfg = run.config or {}
            d_model = _coerce_int(cfg.get("d_model", 64), 64)
            seed = _coerce_int(cfg.get("seed", -1), -1)
            rows.append(
                {
                    "sweep_id": sweep_id,
                    "run_name": run.name,
                    "run_id": run.id,
                    "d_model": d_model,
                    "use_ln": _coerce_bool(cfg.get("use_ln", False), default=False),
                    "use_bias": _coerce_bool(cfg.get("use_bias", False), default=False),
                    "use_wv": _coerce_bool(cfg.get("use_wv", False), default=False),
                    "use_wo": _coerce_bool(cfg.get("use_wo", False), default=False),
                    "use_mlp": _coerce_bool(cfg.get("use_mlp", False), default=False),
                    "seed": seed,
                    "val_accuracy": pd.to_numeric(val_accuracy, errors="coerce"),
                }
            )

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
        max="max",
        min="min",
        median="median",
        n_seeds="count",
    )
    return stats.reset_index()


def tf(value: bool) -> str:
    return "T" if value else "F"


def fmt_acc(val: float, bold: bool = False) -> str:
    formatted = f"{val:.4f}"
    return f"\\textbf{{{formatted}}}" if bold else formatted


def _is_fffff(row: pd.Series) -> bool:
    return all(not row[key] for key in FLAG_KEYS)


def make_flags_table(stats: pd.DataFrame) -> str:
    subset = stats[stats["d_model"] == 64].copy()
    subset = subset.sort_values(FLAG_KEYS)

    lines = [
        "\\begin{tabular}{c c c c c c c c c c c}",
        "d\\_model & ln & bias & wv & wo & mlp & mean & max & min & median & n \\\\",
        "\\midrule",
    ]
    for _, row in subset.iterrows():
        mean_str = fmt_acc(row["mean"], bold=row["mean"] >= BOLD_THRESHOLD)
        line = (
            f"{int(row['d_model'])} & {tf(row['use_ln'])} & {tf(row['use_bias'])} & "
            f"{tf(row['use_wv'])} & {tf(row['use_wo'])} & {tf(row['use_mlp'])} & "
            f"{mean_str} & {fmt_acc(row['max'])} & {fmt_acc(row['min'])} & "
            f"{fmt_acc(row['median'])} & {int(row['n_seeds'])} \\\\"
        )
        lines.append(line)
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def make_dmodel_table(stats: pd.DataFrame) -> str:
    subset = stats[stats.apply(_is_fffff, axis=1)].copy()
    # Exclude d_model=64 since it's already represented in the flags block.
    subset = subset[subset["d_model"] != 64].sort_values("d_model", ascending=False)

    lines = [
        "\\begin{tabular}{c c c c c c}",
        "d\\_model & mean & max & min & median & n \\\\",
        "\\midrule",
    ]
    for _, row in subset.iterrows():
        mean_str = fmt_acc(row["mean"], bold=row["mean"] >= BOLD_THRESHOLD)
        line = (
            f"{int(row['d_model'])} & {mean_str} & {fmt_acc(row['max'])} & "
            f"{fmt_acc(row['min'])} & {fmt_acc(row['median'])} & {int(row['n_seeds'])} \\\\"
        )
        lines.append(line)
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def make_compact_table(stats: pd.DataFrame) -> str:
    flags_block = stats[stats["d_model"] == 64].copy().sort_values(FLAG_KEYS)
    dmodel_block = stats[stats.apply(_is_fffff, axis=1)].copy()
    dmodel_block = dmodel_block[dmodel_block["d_model"] != 64].sort_values(
        "d_model", ascending=False
    )

    lines = [
        "\\begin{tabular}{c c c c c c c}",
        "d\\_model & ln & bias & wv & wo & mlp & mean \\\\",
        "\\midrule",
    ]
    for _, row in flags_block.iterrows():
        mean_str = fmt_acc(row["mean"], bold=row["mean"] >= BOLD_THRESHOLD)
        lines.append(
            f"{int(row['d_model'])} & {tf(row['use_ln'])} & {tf(row['use_bias'])} & "
            f"{tf(row['use_wv'])} & {tf(row['use_wo'])} & {tf(row['use_mlp'])} & "
            f"{mean_str} \\\\"
        )

    lines.append("\\midrule")
    for _, row in dmodel_block.iterrows():
        mean_str = fmt_acc(row["mean"], bold=row["mean"] >= BOLD_THRESHOLD)
        lines.append(
            f"{int(row['d_model'])} & {tf(row['use_ln'])} & {tf(row['use_bias'])} & "
            f"{tf(row['use_wv'])} & {tf(row['use_wo'])} & {tf(row['use_mlp'])} & "
            f"{mean_str} \\\\"
        )
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def print_top3_fffff(df: pd.DataFrame) -> None:
    mask = (df["d_model"] == 64)
    for key in FLAG_KEYS:
        mask &= ~df[key]
    top3 = df[mask].sort_values("val_accuracy", ascending=False).head(3)
    print("Top 3 fffff models (models/2_layer_sweep/)")
    for _, row in top3.iterrows():
        run_name = str(row["run_name"])
        filename = run_name if run_name.endswith(".pt") else f"{run_name}.pt"
        print(
            f"{filename}   (seed={int(row['seed'])}, acc={row['val_accuracy']:.4f})"
        )


def _load_or_fetch(all_sweep_ids: list[str], use_cache: bool) -> pd.DataFrame:
    if use_cache and CACHE_PATH.exists():
        cache_df = pd.read_csv(CACHE_PATH)
        cached_ids = set(cache_df.get("sweep_id", []))
        missing_ids = [sweep_id for sweep_id in all_sweep_ids if sweep_id not in cached_ids]
        if missing_ids:
            new_df = fetch_runs(missing_ids)
            cache_df = pd.concat([cache_df, new_df], ignore_index=True)
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            cache_df.to_csv(CACHE_PATH, index=False)
        return cache_df

    df = fetch_runs(all_sweep_ids)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 2-layer sweep result tables.")
    parser.add_argument("--flags-sweep-ids", nargs="+", required=True)
    parser.add_argument("--dmodel-sweep-ids", nargs="+", required=True)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    all_sweep_ids = list(dict.fromkeys(args.flags_sweep_ids + args.dmodel_sweep_ids))
    df = _load_or_fetch(all_sweep_ids, use_cache=not args.no_cache)
    df = df[df["sweep_id"].isin(all_sweep_ids)].copy()
    stats = compute_stats(df)

    print("BLOCK 1: FLAGS TABLE")
    print(make_flags_table(stats))
    print("BLOCK 2: DMODEL TABLE")
    print(make_dmodel_table(stats))
    print("COMPACT TABLE")
    print(make_compact_table(stats))
    print_top3_fffff(df)


if __name__ == "__main__":
    main()
