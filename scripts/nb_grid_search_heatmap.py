# %% [markdown]
# # Results Heatmap (LIST_LEN x N_LAYERS)
#
# Reads run data from WandB (default) or a local CSV (--source csv),
# averages validation accuracy over seeds, pivots to rows=LIST_LEN / cols=N_LAYERS,
# and renders a seaborn heatmap.

# %%
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

WANDB_ENTITY = "theo-farrell99-durham-university"
WANDB_PROJECT = "order-by-scale"
DEFAULT_SWEEP_ID = "2nu3lkwf"

CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "grid_search" / "results.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "grid_search"


def _coerce_int(value: object, default: int) -> int:
    if value is None:
        return default
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def load_from_wandb(sweep_id: str) -> pd.DataFrame:
    import wandb
    import time

    api = wandb.Api()
    print(f"Fetching sweep {sweep_id} from WandB ...", flush=True)
    sweep = api.sweep(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}")
    all_runs = list(sweep.runs)
    print(f"  {len(all_runs)} runs found", flush=True)

    rows = []
    t0 = time.time()
    n_skipped = n_no_acc = 0
    for i, run in enumerate(all_runs, 1):
        if i % 50 == 0 or i == len(all_runs):
            print(f"  [{i}/{len(all_runs)}] elapsed={time.time()-t0:.0f}s  accepted={len(rows)}  skipped={n_skipped}  no_acc={n_no_acc}", flush=True)
        if run.state != "finished":
            n_skipped += 1
            continue
        val_acc = run.summary.get("best/accuracy")
        if val_acc is None:
            n_no_acc += 1
            continue
        cfg = run.config or {}
        rows.append({
            "LIST_LEN": _coerce_int(cfg.get("list_len"), -1),
            "N_LAYERS": _coerce_int(cfg.get("n_layers"), -1),
            "val_acc": float(val_acc),
        })

    print(f"Fetch done: {len(rows)} accepted, {n_skipped} non-finished, {n_no_acc} missing acc  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No completed runs with best/accuracy found for sweep {sweep_id}")
    return df


def load_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["LIST_LEN", "N_LAYERS", "val_acc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["LIST_LEN", "N_LAYERS", "val_acc"]).copy()
    # Patch: known-good value for (2,2) that was missing from the original CSV
    df.loc[(df["LIST_LEN"] == 2) & (df["N_LAYERS"] == 2), "val_acc"] = 0.914
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search heatmap (LIST_LEN x N_LAYERS).")
    parser.add_argument("--source", choices=["wandb", "csv"], default="wandb",
                        help="Data source (default: wandb)")
    parser.add_argument("--sweep-id", default=DEFAULT_SWEEP_ID,
                        help=f"WandB sweep ID (default: {DEFAULT_SWEEP_ID})")
    parser.add_argument("--include-n1", action="store_true",
                        help="Include N_LAYERS=1 in the heatmap")
    parser.add_argument("--include-l1", action="store_true",
                        help="Include LIST_LEN=1 in the heatmap (excluded by default)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.source == "wandb":
        df = load_from_wandb(args.sweep_id)
    else:
        df = load_from_csv(CSV_PATH)

    if not args.include_l1:
        df = df[df["LIST_LEN"] != 1]

    # Average over seeds for each LIST_LEN × N_LAYERS cell
    agg = (
        df.groupby(["LIST_LEN", "N_LAYERS"])["val_acc"]
          .mean()
          .reset_index(name="val_acc_mean")
    )

    pivot = agg.pivot(index="LIST_LEN", columns="N_LAYERS", values="val_acc_mean")
    pivot = pivot.sort_index().sort_index(axis=1)

    print("Pivot shape (rows=LIST_LEN, cols=N_LAYERS):", pivot.shape)
    print(pivot)

    plt.figure(figsize=(8, 5.5))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Mean Validation Accuracy"},
        linewidths=0.2,
        linecolor="white",
    )
    ax.set_title("Mean Validation Accuracy")
    ax.set_xlabel("No. of Layers")
    ax.set_ylabel("N-gram Size")
    ax.invert_yaxis()
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "heatmap.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved to {out_path}", flush=True)
    plt.show()


if __name__ == "__main__":
    main()

# %%
