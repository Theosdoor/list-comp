#!/usr/bin/env python3
"""
Log experiment runs to EXPERIMENTS.md for reproducibility.

Usage:
    python scripts/log_experiment.py --title "SAE sweep k=4" \\
        --command "wandb sweep sweep_configs/sweep.yaml" \\
        --outputs "results/sae_models/" \\
        --notes "Testing BatchTopK with d_sae=100-256"
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a run entry to EXPERIMENTS.md.")
    parser.add_argument("--title", required=True, help="Short experiment title.")
    parser.add_argument("--command", required=True, help="Command used to run the experiment.")
    parser.add_argument(
        "--outputs",
        nargs="*",
        default=[],
        help="Key output directories/files (space-separated).",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[],
        help="Config file paths used (space-separated).",
    )
    parser.add_argument(
        "--results",
        nargs="*",
        default=[],
        help="Key results or metrics (space-separated; each becomes a bullet).",
    )
    parser.add_argument(
        "--notes",
        nargs="*",
        default=[],
        help="Optional freeform notes (space-separated; each becomes a bullet).",
    )
    parser.add_argument(
        "--log_path",
        default="EXPERIMENTS.md",
        help="Markdown file to append to (default: EXPERIMENTS.md).",
    )
    parser.add_argument(
        "--wandb_url",
        default=None,
        help="W&B run or sweep URL (optional).",
    )
    return parser.parse_args(argv)


def git_commit(repo_root: Path) -> str | None:
    """Get current git commit hash."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
    except Exception:
        return None
    return out or None


def git_dirty(repo_root: Path) -> bool | None:
    """Check if git working directory has uncommitted changes."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root), text=True)
    except Exception:
        return None
    return bool(out.strip())


def ensure_log_header(path: Path) -> None:
    """Create EXPERIMENTS.md header if file doesn't exist."""
    if path.exists():
        return
    path.write_text(
        "# Experiment Log\n\n"
        "This file tracks all experiments for reproducibility. Each entry includes:\n"
        "- Command used to run the experiment\n"
        "- Output locations (models, SAEs, results)\n"
        "- Key results and metrics\n"
        "- Git commit hash (when available)\n\n"
        "**Usage**: Append entries via `python scripts/log_experiment.py` or manually.\n\n"
        "---\n\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    log_path = (repo_root / args.log_path).resolve() if not Path(args.log_path).is_absolute() else Path(args.log_path)
    ensure_log_header(log_path)

    timestamp = datetime.now().isoformat(timespec="seconds")
    commit = git_commit(repo_root)
    dirty = git_dirty(repo_root)

    lines = []
    lines.append(f"## {timestamp} — {args.title}\n\n")
    lines.append(f"**Command**: `{args.command}`\n\n")
    
    if args.outputs:
        lines.append("**Outputs**:\n")
        for output in args.outputs:
            lines.append(f"- `{output}`\n")
        lines.append("\n")
    
    if args.configs:
        lines.append("**Configs**: " + ", ".join(f"`{c}`" for c in args.configs) + "\n\n")
    
    if args.results:
        lines.append("**Results**:\n")
        for result in args.results:
            lines.append(f"- {result}\n")
        lines.append("\n")
    
    if args.wandb_url:
        lines.append(f"**W&B**: {args.wandb_url}\n\n")
    
    if args.notes:
        lines.append("**Notes**:\n")
        for note in args.notes:
            lines.append(f"- {note}\n")
        lines.append("\n")
    
    if commit is not None:
        suffix = " (dirty)" if dirty else ""
        lines.append(f"<details><summary>Git commit</summary>\n\n`{commit}`{suffix}\n</details>\n\n")
    
    lines.append("---\n\n")

    with log_path.open("a", encoding="utf-8") as f:
        f.writelines(lines)
    
    print(f"✓ Logged experiment to {log_path}")


if __name__ == "__main__":
    main()
