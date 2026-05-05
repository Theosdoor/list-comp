#!/usr/bin/env python3
"""
Inspect what's stored in local SAE checkpoint files.

Usage:
    python scripts/inspect_saes.py path/to/sae_models/
    python scripts/inspect_saes.py sae_checkpoints/ sae_checkpoints/sweep_runs/
    python scripts/inspect_saes.py path/to/single_checkpoint.pt
"""

import sys
import json
from pathlib import Path
import torch


def inspect_checkpoint(path: Path) -> None:
    print(f"\n{'='*70}")
    print(f"  {path}")
    print(f"{'='*70}")

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return

    if not isinstance(ckpt, dict):
        print(f"  Top-level type: {type(ckpt)} (not a dict)")
        return

    print(f"  Top-level keys: {sorted(ckpt.keys())}")
    print()

    # --- cfg ---
    cfg = ckpt.get("cfg", {})
    if cfg:
        print("  cfg:")
        for k, v in sorted(cfg.items()):
            print(f"    {k}: {v}")
    else:
        print("  cfg: (absent)")
    print()

    # --- scalar metrics stored at top level ---
    SCALAR_KEYS = [
        "final_loss", "final_l0", "avg_l0",
        "reconstruction_mse", "explained_variance",
        "loss_recovered",
        "patched_acc", "baseline_acc", "accuracy_drop",
        "n_special_features", "max_attn_correlation",
        "dead_features", "dead_features_pct",
    ]
    found_scalars = {k: ckpt[k] for k in SCALAR_KEYS if k in ckpt}
    if found_scalars:
        print("  Stored metrics:")
        for k, v in found_scalars.items():
            print(f"    {k}: {v}")
    else:
        print("  Stored metrics: (none of the expected keys found)")
    print()

    # --- other non-tensor keys ---
    other_keys = [
        k for k in ckpt
        if k not in ("cfg", "state_dict", "act_mean") and k not in SCALAR_KEYS
    ]
    if other_keys:
        print("  Other keys (non-state-dict):")
        for k in sorted(other_keys):
            v = ckpt[k]
            if isinstance(v, torch.Tensor):
                print(f"    {k}: Tensor {tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"    {k}: {type(v).__name__} = {v}")
        print()

    # --- act_mean ---
    if "act_mean" in ckpt:
        am = ckpt["act_mean"]
        print(f"  act_mean: Tensor {tuple(am.shape)} dtype={am.dtype}")
    else:
        print("  act_mean: (absent)")
    print()

    # --- state_dict summary ---
    sd = ckpt.get("state_dict", {})
    if sd:
        print("  state_dict tensors:")
        for k, v in sd.items():
            print(f"    {k}: {tuple(v.shape)} dtype={v.dtype}")
    else:
        print("  state_dict: (absent)")


def find_checkpoints(paths: list[str]) -> list[Path]:
    checkpoints = []
    for p in paths:
        root = Path(p)
        if root.is_file() and root.suffix == ".pt":
            checkpoints.append(root)
        elif root.is_dir():
            found = sorted(root.rglob("*.pt"))
            checkpoints.extend(found)
        else:
            print(f"Warning: {p} is not a .pt file or directory, skipping.")
    return checkpoints


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    checkpoints = find_checkpoints(sys.argv[1:])

    if not checkpoints:
        print("No .pt files found in the given path(s).")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoint(s).")

    for path in checkpoints:
        inspect_checkpoint(path)

    print(f"\n{'='*70}")
    print(f"Done. Inspected {len(checkpoints)} checkpoint(s).")


if __name__ == "__main__":
    main()