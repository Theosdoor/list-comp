import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import wandb
from tqdm.auto import tqdm

# Initialize API once globally
api = wandb.Api()

def download_run_artifact(run, save_dir):
    """Worker function to fetch and download a single run's artifact."""
    try:
        # Fetch artifacts for this specific run
        artifacts = run.logged_artifacts()
        model_artifact = next((art for art in artifacts if art.type == "model"), None)
        
        if not model_artifact:
            return False, f"Skipped {run.name}: No model artifact found."

        # Isolate downloads in run-specific directories to prevent filename collisions
        run_save_dir = save_dir / run.name
        run_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the artifact
        model_artifact.download(root=str(run_save_dir))
        return True, run.name
        
    except Exception as e:
        return False, f"Error on {run.name}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Download all SAE models from a W&B sweep concurrently.")
    parser.add_argument("sweep_id", type=str, 
                        help="The W&B sweep ID (e.g., k2bsjr0n)")
    parser.add_argument("--project", type=str, default="theo-farrell99-durham-university/orderbyscale_sae_sweep", 
                        help="The W&B project path formatted as 'entity/project'")
    parser.add_argument("--save_dir", type=str, default="sae_checkpoints", 
                        help="Root directory to save the checkpoints")
    parser.add_argument("--workers", type=int, default=15, 
                        help="Maximum number of concurrent downloads (default: 15)")
    
    args = parser.parse_args()
    save_folder_name = f"sweep_{args.sweep_id}"
    save_path = Path(args.save_dir) / save_folder_name

    print(f"Fetching run metadata for sweep '{args.sweep_id}' from W&B server...")
    
    # Fetch the list of runs
    runs = list(api.runs(
        path=args.project,
        filters={"sweep": args.sweep_id}
    ))
    
    total_runs = len(runs)
    if total_runs == 0:
        print("No runs found for this sweep ID and project combination. Check your inputs.")
        return

    print(f"Found {total_runs} runs. Starting parallel downloads with {args.workers} workers...\n")
    
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Pass the specific save directory to the worker function
        futures = {executor.submit(download_run_artifact, run, save_path): run for run in runs}
        
        for future in tqdm(as_completed(futures), total=total_runs, desc="Downloading SAEs", unit="model"):
            success, result_msg = future.result()
            
            if success:
                success_count += 1
            else:
                fail_count += 1
                tqdm.write(result_msg)

    print(f"\nFinished! Downloaded {success_count} models. ({fail_count} failed/skipped).")

if __name__ == "__main__":
    main()