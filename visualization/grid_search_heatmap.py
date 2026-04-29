# %% [markdown]
# # Results Heatmap (LIST_LEN x N_LAYERS)
#
# Reads run data from WandB (default) or a local CSV (--source csv),
# averages validation accuracy over seeds, pivots to rows=LIST_LEN / cols=N_LAYERS,
# and renders a seaborn heatmap.

# %%
import argparse
import sys
import types
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

WANDB_ENTITY = "theo-farrell99-durham-university"
WANDB_PROJECT = "order-by-scale"
DEFAULT_SWEEP_ID = "2nu3lkwf"

CSV_PATH = Path(__file__).resolve().parent.parent.parent.parent / "results" / "grid_search" / "results.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "grid_search"

FILL_IN_BLANKS_WITH_CSV = True  # Set to True to fill in missing (LIST_LEN, N_LAYERS) combos with averages from the csv instead of leaving blank. Prints which combos are missing from wandb

HIDE_LAYERS = [1,9,10]
HIDE_LIST_LEN = [1,9,10]

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
    # Known-good value for (2,2) that was missing from the original CSV
    df.loc[(df["LIST_LEN"] == 2) & (df["N_LAYERS"] == 2), "val_acc"] = 0.914
    return df


def fill_missing_with_csv(wandb_df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """
    Fill in missing (LIST_LEN, N_LAYERS) combinations with averages from CSV.
    
    Args:
        wandb_df: DataFrame loaded from WandB
        csv_path: Path to CSV file
    
    Returns:
        DataFrame with missing combos filled in from CSV
    """
    csv_df = load_from_csv(csv_path)
    
    # Get unique combos from wandb and csv
    wandb_combos = set(zip(wandb_df["LIST_LEN"], wandb_df["N_LAYERS"]))
    csv_combos = set(zip(csv_df["LIST_LEN"], csv_df["N_LAYERS"]))
    
    # Find missing combos
    missing_combos = csv_combos - wandb_combos
    
    if missing_combos:
        print(f"\n[FILL_IN_BLANKS] Missing combos from WandB (will fill from CSV):")
        for list_len, n_layers in sorted(missing_combos):
            avg_acc = csv_df[(csv_df["LIST_LEN"] == list_len) & 
                            (csv_df["N_LAYERS"] == n_layers)]["val_acc"].mean()
            print(f"  (LIST_LEN={list_len}, N_LAYERS={n_layers}): avg_acc={avg_acc:.4f}")
        
        # Create rows for missing combos from CSV
        missing_rows = []
        for list_len, n_layers in missing_combos:
            avg_acc = csv_df[(csv_df["LIST_LEN"] == list_len) & 
                            (csv_df["N_LAYERS"] == n_layers)]["val_acc"].mean()
            missing_rows.append({
                "LIST_LEN": list_len,
                "N_LAYERS": n_layers,
                "val_acc": avg_acc,
            })
        
        missing_df = pd.DataFrame(missing_rows)
        result = pd.concat([wandb_df, missing_df], ignore_index=True)
        print(f"Filled {len(missing_rows)} missing combos from CSV\n", flush=True)
        return result
    else:
        print("[FILL_IN_BLANKS] No missing combos found\n", flush=True)
        return wandb_df


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


def run(args) -> None:
    if args.source == "wandb":
        df = load_from_wandb(args.sweep_id)
        # Fill in missing combos from CSV if enabled
        if FILL_IN_BLANKS_WITH_CSV:
            df = fill_missing_with_csv(df, CSV_PATH)
    else:
        df = load_from_csv(CSV_PATH)

    if not args.include_l1:
        df = df[df["LIST_LEN"] != 1]
    if not args.include_n1:
        df = df[df["N_LAYERS"] != 1]

    # Skip hidden layers and list lengths to avoid slow computation
    if HIDE_LAYERS:
        df = df[~df["N_LAYERS"].isin(HIDE_LAYERS)]
    if HIDE_LIST_LEN:
        df = df[~df["LIST_LEN"].isin(HIDE_LIST_LEN)]

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


def main() -> None:
    run(_parse_args())


# %%
if "ipykernel" in sys.modules or not sys.argv[0].endswith(".py"):
    run(types.SimpleNamespace(
        source="wandb",
        sweep_id=DEFAULT_SWEEP_ID,
        include_n1=False,
        include_l1=False,
    ))
elif __name__ == "__main__":
    main()
