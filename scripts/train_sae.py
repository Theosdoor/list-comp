"""
Train SAE on SEP token activations.

Standalone:  python scripts/train_sae.py --sae_type btk --d_sae 150 --top_k 4
W&B sweep:   python scripts/train_sae.py --wandb  (config injected via wandb.config)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import gc
import os
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm

from dictionary_learning.trainers import BatchTopKTrainer
from dictionary_learning.trainers.jumprelu import JumpReluTrainer
from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKTrainer

from src.models.transformer import make_model
from src.models.utils import infer_model_config, count_params
from src.utils.runtime import configure_runtime
from src.data.datasets import get_dataset
from src.sae import (
    collect_sae_activations,
    compute_reconstruction_metrics,
    collect_attention_patterns,
    identify_special_features,
    compute_sae_downstream_metrics,
)


_DEFAULT_MODEL_PATH = 'models/2layer_100dig_64d.pt'

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# These are set by _init_model_globals() before training begins
MODEL_NAME = None
MODEL_PATH = None
MODEL_CFG = None
N_LAYERS = N_HEADS = D_MODEL = VOCAB = N_DIGITS = LIST_LEN = SEP_TOKEN_INDEX = None


def _init_model_globals(model_path: str):
    global MODEL_NAME, MODEL_PATH, MODEL_CFG
    global N_LAYERS, N_HEADS, D_MODEL, VOCAB, N_DIGITS, LIST_LEN, SEP_TOKEN_INDEX
    MODEL_PATH = model_path
    MODEL_NAME = Path(model_path).stem
    MODEL_CFG = infer_model_config(model_path)
    N_LAYERS = MODEL_CFG['n_layers']
    N_HEADS = MODEL_CFG['n_heads']
    D_MODEL = MODEL_CFG['d_model']
    VOCAB = MODEL_CFG['d_vocab']
    N_DIGITS = VOCAB - 2
    LIST_LEN = MODEL_CFG['list_len']
    SEP_TOKEN_INDEX = LIST_LEN


# ---------------------------------------------------------------------------
# Trainer registry
# ---------------------------------------------------------------------------

def _make_btk_trainer(cfg, activation_dim, device):
    trainer = BatchTopKTrainer(
        steps=cfg.n_steps, activation_dim=activation_dim, dict_size=cfg.d_sae,
        k=cfg.top_k, layer=0, lm_name="custom", lr=cfg.lr,
        warmup_steps=cfg.warmup_steps, seed=cfg.seed, device=device,
    )
    run_name = f"btk_sae_d{cfg.d_sae}_k{cfg.top_k}_lr{cfg.lr}_seed{cfg.seed}"
    extra_cfg = {"sae_type": "btk", "k": cfg.top_k}
    return trainer, run_name, extra_cfg


def _make_jumprelu_trainer(cfg, activation_dim, device):
    trainer = JumpReluTrainer(
        steps=cfg.n_steps, activation_dim=activation_dim, dict_size=cfg.d_sae,
        layer=0, lm_name="custom", lr=cfg.lr, warmup_steps=cfg.warmup_steps,
        sparsity_penalty=cfg.sparsity_penalty, target_l0=cfg.target_l0,
        seed=cfg.seed, device=device,
    )
    run_name = f"jumprelu_sae_d{cfg.d_sae}_tl0{cfg.target_l0}_sp{cfg.sparsity_penalty}_lr{cfg.lr}_seed{cfg.seed}"
    extra_cfg = {
        "sae_type": "jumprelu",
        "target_l0": cfg.target_l0,
        "sparsity_penalty": cfg.sparsity_penalty,
    }
    return trainer, run_name, extra_cfg


def _make_matryoshka_trainer(cfg, activation_dim, device):
    n_groups = cfg.n_groups
    frac = 1.0 / n_groups
    group_fractions = [frac] * (n_groups - 1) + [1.0 - frac * (n_groups - 1)]
    trainer = MatryoshkaBatchTopKTrainer(
        steps=cfg.n_steps, activation_dim=activation_dim, dict_size=cfg.d_sae,
        k=cfg.top_k, layer=0, lm_name="custom", lr=cfg.lr,
        warmup_steps=cfg.warmup_steps, group_fractions=group_fractions,
        seed=cfg.seed, device=device,
    )
    run_name = f"matryoshka_sae_d{cfg.d_sae}_k{cfg.top_k}_ng{n_groups}_lr{cfg.lr}_seed{cfg.seed}"
    extra_cfg = {
        "sae_type": "matryoshka", "k": cfg.top_k, "n_groups": n_groups,
        "group_fractions": group_fractions, "group_sizes": trainer.group_sizes,
    }
    return trainer, run_name, extra_cfg


TRAINER_REGISTRY = {
    "btk":        _make_btk_trainer,
    "jumprelu":   _make_jumprelu_trainer,
    "matryoshka": _make_matryoshka_trainer,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_sep_activations(model, dataloader, layer_idx=0, sep_idx=2, max_acts=100_000):
    """Extract SEP token activations from the residual stream after a given layer."""
    activations = []
    count = 0
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    print(f"Collecting activations from {hook_name} at SEP token (idx {sep_idx})...")
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Collecting"):
            inputs = inputs.to(model.cfg.device)
            _, cache = model.run_with_cache(
                inputs, stop_at_layer=layer_idx + 1, names_filter=hook_name
            )
            sep_acts = cache[hook_name][:, sep_idx, :]
            activations.append(sep_acts.cpu())
            count += sep_acts.shape[0]
            if count >= max_acts:
                break
    return torch.cat(activations, dim=0)


def _load_model_and_acts():
    configure_runtime(list_len=LIST_LEN, seq_len=LIST_LEN * 2 + 1, vocab=N_DIGITS + 2, device=DEVICE)
    model = make_model(
        n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL,
        ln=MODEL_CFG.get('use_ln', False),
        use_bias=MODEL_CFG.get('use_bias', False),
        use_wv=MODEL_CFG.get('use_wv', False),
        use_wo=MODEL_CFG.get('use_wo', False),
    )
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)  # Explicitly move model to device after loading
    print(f"✓ Loaded model from {MODEL_PATH}")

    # Load FULL dataset (train + val) for SAE training to cover entire input space
    # This ensures the SAE sees all digit combinations during training, not just 80% of them.
    train_ds, val_ds = get_dataset(
        list_len=LIST_LEN, n_digits=N_DIGITS, mask_tok=N_DIGITS, sep_tok=N_DIGITS + 1,
        no_dupes=False
    )
    full_ds = ConcatDataset([train_ds, val_ds])
    train_dl = DataLoader(full_ds, batch_size=512, shuffle=True)

    all_acts = get_sep_activations(model, train_dl, layer_idx=0, sep_idx=SEP_TOKEN_INDEX).to(DEVICE)
    act_mean = all_acts.mean(0)  # Computed after .to(DEVICE) to ensure act_mean is on correct device
    print(f"Collected {len(all_acts)} activations, shape: {all_acts.shape}")
    return model, act_mean, all_acts - act_mean


def _log_wandb_eval_metrics(model, sae, act_mean, d_sae):
    """Compute post-training eval metrics and write to wandb.summary."""
    import wandb
    print("\nGenerating analysis metrics...")
    try:
        _train_ds, _val_ds = get_dataset(list_len=LIST_LEN, n_digits=N_DIGITS, no_dupes=False)
        analysis_dl = DataLoader(ConcatDataset([_train_ds, _val_ds]), batch_size=2048, shuffle=False)

        _, _, sae_acts_all = collect_sae_activations(
            model, sae, analysis_dl, act_mean,
            layer_idx=0, sep_idx=SEP_TOKEN_INDEX, device=DEVICE,
        )
        l0 = (sae_acts_all > 0).float().sum(dim=1).mean()
        dead_features = (sae_acts_all.sum(dim=0) == 0).sum().item()
        wandb.summary["avg_l0"] = l0.item()
        wandb.summary["dead_features_pct"] = 100 * dead_features / d_sae

        try:
            recon = compute_reconstruction_metrics(
                model, sae, analysis_dl, act_mean,
                layer_idx=0, sep_idx=SEP_TOKEN_INDEX, device=DEVICE,
            )
            wandb.summary["explained_variance"] = recon["explained_variance"]
        except Exception as e:
            print(f"    ⚠ Reconstruction metrics failed: {e}")
            wandb.summary["explained_variance"] = None

        try:
            downstream = compute_sae_downstream_metrics(
                model, sae, analysis_dl, act_mean,
                layer_idx=0, sep_idx=SEP_TOKEN_INDEX, device=DEVICE,
            )
            wandb.summary["baseline_ce"] = downstream["baseline_ce"]
            wandb.summary["patched_ce"] = downstream["patched_ce"]
            wandb.summary["ce_increase"] = downstream["ce_increase"]
        except Exception as e:
            print(f"    ⚠ Downstream CE metrics failed: {e}")
            wandb.summary["baseline_ce"] = None
            wandb.summary["patched_ce"] = None
            wandb.summary["ce_increase"] = None

        try:
            alpha_d1_all, alpha_d2_all = collect_attention_patterns(
                model, analysis_dl, layer_idx=0, sep_idx=SEP_TOKEN_INDEX, device=DEVICE,
            )
            special_info = identify_special_features(
                sae_acts_all, alpha_d1_all, alpha_d2_all, threshold=0.5
            )
            wandb.summary["n_special_features"] = special_info["n_special_features"]
        except Exception as e:
            print(f"    ⚠ Special features failed: {e}")
            wandb.summary["n_special_features"] = None

        print("✓ Logged all analysis metrics to W&B")
    except Exception as e:
        print(f"⚠ Error during metric logging: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------

def _train(cfg, use_wandb: bool, save_folder: str, model_path: str = None):
    """Train an SAE given a config object (argparse.Namespace or wandb.Config)."""
    import wandb as _wandb

    _init_model_globals(model_path or _DEFAULT_MODEL_PATH)

    sae_type = cfg.sae_type
    if sae_type not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown sae_type '{sae_type}'. Known: {list(TRAINER_REGISTRY)}")

    torch.manual_seed(cfg.seed)
    # Initialize all variables to None for safe cleanup in finally block
    model = None
    trainer = None
    act_mean = None
    all_acts_centered = None
    sae_dl = None
    iter_dl = None
    try:
        model, act_mean, all_acts_centered = _load_model_and_acts()

        sae_dl = DataLoader(all_acts_centered, batch_size=cfg.batch_size, shuffle=True)
        trainer, run_name, extra_cfg = TRAINER_REGISTRY[sae_type](cfg, D_MODEL, DEVICE)

        if use_wandb:
            _wandb.run.name = run_name
            
            # Log model metrics (use summary for static metadata to avoid incrementing step counter)
            total_params, trainable_params = count_params(trainer.ae)
            full_dataset_size = N_DIGITS ** LIST_LEN
            sampled = full_dataset_size > len(all_acts_centered)
            
            _wandb.summary.update({
                "model/params_total": total_params,
                "model/params_trainable": trainable_params,
                "model/d_model": D_MODEL,
                "model/model_name": MODEL_NAME,
                "data/full_dataset_size": full_dataset_size,
                "data/n_total": len(all_acts_centered),
                "data/sampled": sampled,
                "data/batch_size": cfg.batch_size,
                "config/warmup_steps": cfg.warmup_steps,
                "config/n_steps": cfg.n_steps,
            })

        is_tty = sys.stdout.isatty()
        log_interval = 100 if is_tty else 1000

        print(f"\n{'='*60}")
        print(f"Run:      {run_name}")
        print(f"Model:    {MODEL_PATH}")
        print(f"Steps:    {cfg.n_steps}  |  LR: {cfg.lr}  |  d_sae: {cfg.d_sae}  |  device: {DEVICE}")
        print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        sys.stdout.flush()

        def cycle(it):
            while True:
                yield from it

        iter_dl = cycle(sae_dl)
        pbar = tqdm(range(cfg.n_steps), disable=not is_tty, desc="Training", unit="step")
        loss = 0.0
        t_start = datetime.now()
        best_loss = float('inf')
        best_step = 0

        for step in pbar:
            batch_acts = next(iter_dl)
            loss = trainer.update(step, batch_acts)
            if step % log_interval == 0:
                log_info = trainer.get_logging_parameters()
                if 'effective_l0' in log_info:
                    # NOTE: trainer.effective_l0 is hardcoded to k (the target sparsity), not actual sparsity.
                    # It's logged for reference but is constant throughout training.
                    # Real sparsity (avg num nonzero features per token) is computed in _log_wandb_eval_metrics
                    # from actual activations after training completes.
                    effective_l0 = log_info['effective_l0']
                else:
                    with torch.no_grad():
                        f = trainer.ae.encode(batch_acts.to(DEVICE))
                        effective_l0 = (f > 0).float().sum(dim=-1).mean().item()
                
                pct_complete = 100 * (step+1) / cfg.n_steps
                
                # Track best metrics
                if loss < best_loss:
                    best_loss = loss
                    best_step = step
                
                if use_wandb:
                    # Get actual LR from optimizer (accounts for warmup schedule)
                    actual_lr = trainer.optimizer.param_groups[0]['lr']
                    _wandb.log({
                        "train/loss": loss,
                        "train/l0": effective_l0,
                        "train/lr": actual_lr,
                        "train/pct_complete": pct_complete,
                    }, step=step)
                elapsed = (datetime.now() - t_start).total_seconds()
                eta_s = (elapsed / max(step, 1)) * (cfg.n_steps - step)
                eta_str = f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s"
                if is_tty:
                    pbar.set_postfix({"loss": f"{loss:.4f}", "L0": f"{effective_l0:.1f}", "ETA": eta_str})
                else:
                    ts = datetime.now().strftime('%H:%M:%S')
                    print(f"[{ts}] step {step:>6}/{cfg.n_steps}  ({pct_complete:4.1f}%)  loss={loss:.4f}  L0={effective_l0:.1f}  ETA={eta_str}")
                    sys.stdout.flush()

        log_info = trainer.get_logging_parameters()
        final_l0 = log_info.get('effective_l0', 0)
        total_time = (datetime.now() - t_start).total_seconds()
        elapsed_min = total_time / 60
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training complete — "
              f"final_loss={loss:.4f}  final_L0={final_l0:.1f}  "
              f"time={int(total_time // 60)}m{int(total_time % 60):02d}s")
        sys.stdout.flush()

        if use_wandb:
            _wandb.summary["final/loss"] = loss
            _wandb.summary["final/l0"] = final_l0
            _wandb.summary["best/loss"] = best_loss
            _wandb.summary["best/step"] = best_step
            _wandb.summary["final/elapsed_min"] = elapsed_min
            _wandb.summary["final/step"] = cfg.n_steps
            _log_wandb_eval_metrics(model, trainer.ae.to(DEVICE), act_mean, cfg.d_sae)

        # Save checkpoint
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"{run_name}_{MODEL_NAME}.pt")
        torch.save({
            "state_dict": trainer.ae.state_dict(),
            "cfg": {
                "activation_dim": D_MODEL, "dict_size": cfg.d_sae,
                "d_model": D_MODEL, "d_sae": cfg.d_sae,
                "lr": cfg.lr, "seed": cfg.seed,
                "model_path": MODEL_PATH,
                **extra_cfg,
            },
            "act_mean": act_mean.cpu(),
            "final_loss": loss,
            "final_l0": final_l0,
        }, save_path)
        print(f"\n✓ SAE saved to {save_path}")

        if use_wandb:
            artifact = _wandb.Artifact(
                name=run_name.replace("_", "-"), type="model",
                metadata={
                    "sae_type": sae_type, "d_sae": cfg.d_sae,
                    "lr": cfg.lr, "seed": cfg.seed,
                    "final_loss": loss, "final_l0": final_l0,
                    **extra_cfg,
                },
            )
            artifact.add_file(save_path)
            _wandb.log_artifact(artifact)
    finally:
        # Clean up GPU memory and close wandb run (must be in finally to guarantee execution)
        # Delete all references to GPU-held tensors to ensure memory is freed
        del model, trainer, act_mean, all_acts_centered, sae_dl, iter_dl
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if use_wandb:
            _wandb.finish()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def train_sae_sweep():
    """Entry point for W&B sweep agent (called by sweep_sae.py wrapper)."""
    import wandb
    run = wandb.init()
    cfg = wandb.config
    sweep_id = run.sweep_id or "standalone"
    _train(cfg, use_wandb=True, save_folder=f"results/sae_models/sweep_{sweep_id}")


def main():
    parser = argparse.ArgumentParser(description="Train SAE on SEP token activations")
    parser.add_argument("--wandb", action="store_true",
                        help="W&B sweep mode: call wandb.init() and read config from wandb.config")
    parser.add_argument("--sae_type", type=str, default="btk", choices=["btk", "jumprelu", "matryoshka"])
    parser.add_argument("--d_sae", type=int, default=150)
    parser.add_argument("--top_k", type=int, default=4, help="Active features per token (btk/matryoshka)")
    parser.add_argument("--target_l0", type=float, default=4.0, help="Target sparsity (jumprelu)")
    parser.add_argument("--sparsity_penalty", type=float, default=1.0, help="Sparsity penalty (jumprelu)")
    parser.add_argument("--n_groups", type=int, default=4, help="Nested groups (matryoshka)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=50_000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_folder", type=str, default="results/sae_models")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to transformer checkpoint (default: models/2layer_100dig_64d.pt)")
    args = parser.parse_args()

    if args.wandb:
        train_sae_sweep()
    else:
        _train(args, use_wandb=False, save_folder=args.save_folder, model_path=args.model_path)


if __name__ == "__main__":
    main()
