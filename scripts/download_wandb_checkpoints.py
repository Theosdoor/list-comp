#!/usr/bin/env python3
"""
Download checkpoints from WandB projects and runs.

Examples:
    # Download all SAE checkpoints from a project
    python scripts/download_wandb_checkpoints.py \\
        --entity theo-farrell99-durham-university \\
        --project orderbyscale_sae_sweep \\
        --artifact_type sae_model \\
        --output_dir results/sae_models/

    # Download transformer checkpoints from specific runs
    python scripts/download_wandb_checkpoints.py \\
        --entity theo-farrell99-durham-university \\
        --project order-by-scale \\
        --artifact_type model \\
        --runs run1 run2 run3 \\
        --output_dir models/

    # Download all artifacts from a project
    python scripts/download_wandb_checkpoints.py \\
        --entity theo-farrell99-durham-university \\
        --project orderbyscale_sae_sweep \\
        --output_dir checkpoints/
"""

import argparse
import os
from pathlib import Path
from typing import Optional, List
import wandb
from tqdm import tqdm


def download_artifacts_from_project(
    entity: str,
    project: str,
    output_dir: str,
    artifact_type: Optional[str] = None,
    run_ids: Optional[List[str]] = None,
    artifact_name_pattern: Optional[str] = None,
) -> None:
    """
    Download artifacts from a WandB project.
    
    Args:
        entity: WandB entity (user or organization)
        project: WandB project name
        output_dir: Local directory to save checkpoints
        artifact_type: Filter by artifact type (e.g., 'model', 'sae_model'). If None, download all.
        run_ids: List of specific run IDs to download from. If None, download from all runs.
        artifact_name_pattern: Optional substring to filter artifact names (e.g., 'final' or '.pt')
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    
    # Get the project
    try:
        wandb_project = api.project(f"{entity}/{project}")
        print(f"✓ Connected to {entity}/{project}")
    except Exception as e:
        print(f"✗ Could not access project {entity}/{project}: {e}")
        return
    
    # Fetch runs
    filters = {}
    if run_ids:
        filters["$or"] = [{"id": rid} for rid in run_ids]
    
    try:
        runs = api.runs(f"{entity}/{project}", filters=filters)
        runs_list = list(runs)
        print(f"✓ Found {len(runs_list)} run(s)")
    except Exception as e:
        print(f"✗ Could not fetch runs: {e}")
        return
    
    if not runs_list:
        print("No runs found matching the criteria.")
        return
    
    downloaded_count = 0
    skipped_count = 0
    
    for run in tqdm(runs_list, desc="Processing runs"):
        try:
            artifacts = api.artifacts(type_filter=artifact_type, per_page=100)
            # Filter to this run's artifacts
            run_artifacts = [a for a in artifacts if a.logged_by() == run.id]
        except Exception:
            # Fallback: iterate through artifact versions
            run_artifacts = []
            try:
                for artifact_collection in api.artifact_types(f"{entity}/{project}"):
                    for artifact in artifact_collection.artifacts(per_page=100):
                        if artifact.logged_by() == run.id:
                            run_artifacts.append(artifact)
            except Exception:
                continue
        
        for artifact in run_artifacts:
            # Filter by type if specified
            if artifact_type and artifact.type != artifact_type:
                continue
            
            # Filter by name pattern if specified
            if artifact_name_pattern and artifact_name_pattern not in artifact.name:
                continue
            
            # Download the artifact
            try:
                artifact_dir = artifact.download(root=output_dir)
                
                # Find the actual checkpoint files (.pt, .pth, etc.)
                checkpoint_files = list(Path(artifact_dir).glob("**/*.pt")) + \
                                  list(Path(artifact_dir).glob("**/*.pth")) + \
                                  list(Path(artifact_dir).glob("**/*.pkl"))
                
                if checkpoint_files:
                    for ckpt in checkpoint_files:
                        # Organize by artifact name
                        dest = Path(output_dir) / artifact.name / ckpt.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if not dest.exists():
                            ckpt.rename(dest)
                        downloaded_count += 1
                else:
                    # If no .pt files, just keep the directory structure
                    downloaded_count += 1
                
            except Exception as e:
                print(f"  ✗ Failed to download {artifact.name}: {e}")
                skipped_count += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Download complete!")
    print(f"  Downloaded: {downloaded_count} artifact(s)")
    print(f"  Skipped: {skipped_count} artifact(s)")
    print(f"  Saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download checkpoints from WandB projects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--entity", type=str, required=True,
                        help="WandB entity (user or organization)")
    parser.add_argument("--project", type=str, required=True,
                        help="WandB project name")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Local directory to save checkpoints")
    parser.add_argument("--artifact_type", type=str, default=None,
                        help="Filter by artifact type (e.g., 'model', 'sae_model')")
    parser.add_argument("--runs", type=str, nargs="+", default=None,
                        help="Specific run IDs to download from (space-separated)")
    parser.add_argument("--name_filter", type=str, default=None,
                        help="Filter artifacts by name substring (e.g., 'final')")
    
    args = parser.parse_args()
    
    download_artifacts_from_project(
        entity=args.entity,
        project=args.project,
        output_dir=args.output_dir,
        artifact_type=args.artifact_type,
        run_ids=args.runs,
        artifact_name_pattern=args.name_filter,
    )


if __name__ == "__main__":
    main()
