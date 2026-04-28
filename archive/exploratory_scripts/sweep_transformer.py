"""
W&B Sweep Script for Transformer Architecture Grid Search

Generalised from sweep_2layer.py. Reads list_len and n_layers from wandb.config
(both default to 2), so all existing 2-layer sweep YAMLs work without changes.
"""

from pathlib import Path
import sys
import time
import random

import gc

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.datasets import get_dataset
from src.models.train import train
from src.models.transformer import make_model
from src.models.utils import count_params, save_model
from src.utils.runtime import configure_runtime


WANDB_PROJECT = "order-by-scale"
N_DIGITS = 100
DEV = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

LR = 1e-3
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 2048
VAL_BATCH_SIZE = 4096
MAX_STEPS = 50_000
EARLY_STOP_ACC = 0.999


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sweep_transformer():
    load_dotenv()
    run = wandb.init(
        settings=wandb.Settings(init_timeout=300),
        project=WANDB_PROJECT,
        config={
            "max_steps": MAX_STEPS,
            "early_stop_acc": EARLY_STOP_ACC,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "val_batch_size": VAL_BATCH_SIZE,
            "n_digits": N_DIGITS,
            "list_len": 2,
            "n_layers": 2,
            "n_heads": 1,
        },
    )
    print(f"[wandb] run_id={run.id} name={run.name} url={run.url}")

    # Flush any leftover memory from previous runs (including crashed ones)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    config = wandb.config

    list_len = config.list_len
    n_layers  = config.n_layers
    seed      = config.seed

    d_model   = config.d_model
    n_heads   = config.n_heads
    use_ln    = config.use_ln
    use_bias  = config.use_bias
    use_wv    = config.use_wv
    use_wo    = config.use_wo
    use_mlp   = config.use_mlp

    seq_len = list_len * 2 + 1
    vocab   = N_DIGITS + 2

    def flag(value: bool) -> str:
        return "T" if value else "F"

    run.name = (
        f"d{d_model}_L{list_len}_N{n_layers}_h{n_heads}_"
        f"ln{flag(use_ln)}_bias{flag(use_bias)}_"
        f"wv{flag(use_wv)}_wo{flag(use_wo)}_mlp{flag(use_mlp)}_s{seed}"
    )
    
    # list_len=1 gives only 100 sequences (one per digit). After 80/20 split, val digits
    # are fully OOD — their embeddings are never updated during training — so val acc is
    # always 0 regardless of architecture. Skip rather than log misleading zeros.
    if list_len == 1:
        wandb.log({"skip_reason": "list_len=1_ood_val"})
        wandb.finish()
        return

    print(
        f"[model] list_len={list_len}, n_layers={n_layers}, d_model={d_model}, "
        f"n_heads={n_heads}, seq_len={seq_len}, vocab={vocab}, "
        f"ln={use_ln}, bias={use_bias}, wv={use_wv}, wo={use_wo}, mlp={use_mlp}, seed={seed}"
    )

    configure_runtime(
        list_len=list_len,
        seq_len=seq_len,
        vocab=vocab,
        device=DEV,
        seed=0,
    )

    # Dataset seed is fixed (independent of config.seed) so all 30 model seeds
    # for a given (list_len, n_layers) cell train on identical data. This isolates
    # model-initialisation variance from data-sampling variance.
    train_ds, val_ds = get_dataset(
        list_len=list_len,
        n_digits=N_DIGITS,
        train_split=0.8,
        mask_tok=N_DIGITS,
        sep_tok=N_DIGITS + 1,
        seed=0,
    )

    full_dataset_size = N_DIGITS ** list_len
    sampled = full_dataset_size > len(train_ds) + len(val_ds)
    print(
        f"[dataset] list_len={list_len}, full_size={full_dataset_size:,}, "
        f"train={len(train_ds):,}, val={len(val_ds):,}"
        + (" (sampled)" if sampled else " (full enumeration)")
    )

    set_seeds(seed)

    train_batch_size = min(TRAIN_BATCH_SIZE, len(train_ds))
    val_batch_size = min(VAL_BATCH_SIZE, len(val_ds))
    _pin = DEV == "cuda"
    train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, drop_last=True,
                          pin_memory=_pin)
    val_dl = DataLoader(val_ds, batch_size=val_batch_size, drop_last=False,
                        pin_memory=_pin)

    model = None
    try:
        model = make_model(
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            ln=use_ln,
            use_bias=use_bias,
            use_wv=use_wv,
            use_wo=use_wo,
            attn_only=not use_mlp,
        ).to(DEV)

        total_params, trainable_params = count_params(model)
        wandb.summary.update({
            "model/params_total": total_params,
            "model/params_trainable": trainable_params,
            "data/n_train": len(train_ds),
            "data/n_val": len(val_ds),
            "data/full_dataset_size": full_dataset_size,
            "data/sampled": sampled,
        })

        print(
            f"[sweep] list_len={list_len}, n_layers={n_layers}, d_model={d_model}, "
            f"n_heads={n_heads}, ln={use_ln}, bias={use_bias}, wv={use_wv}, "
            f"wo={use_wo}, mlp={use_mlp}, seed={seed}, params={total_params}"
        )

        t0 = time.time()
        best_acc = train(
            model,
            train_dl,
            val_dl,
            max_steps=MAX_STEPS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            list_len=list_len,
            vocab=vocab,
            device=DEV,
            early_stop_acc=EARLY_STOP_ACC,
            use_wandb=True,
            show_progress=False,
        )
        elapsed_min = (time.time() - t0) / 60
        print(f"[sweep] done: best_acc={best_acc:.4f}, elapsed={elapsed_min:.1f}min")

        wandb.log({"final/val_accuracy": best_acc, "final/elapsed_min": elapsed_min})
        wandb.summary["final/val_accuracy"] = best_acc
        wandb.summary["final/elapsed_min"] = elapsed_min

        if (not use_ln and not use_bias and not use_wv and not use_wo and not use_mlp and d_model == 64):
            base_dir = Path(__file__).resolve().parents[3] / "models" / "2_layer_sweep"
            model_path = base_dir / f"{run.name}_acc{best_acc:.4f}.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            save_model(model, str(model_path))
    finally:
        del model, train_dl, val_dl, train_ds, val_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        wandb.finish()


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        import types
        _cfg = types.SimpleNamespace(
            list_len=2, n_layers=2, d_model=64, n_heads=1,
            use_ln=False, use_bias=False, use_wv=False, use_wo=False,
            use_mlp=False, seed=0,
        )
        configure_runtime(list_len=_cfg.list_len, seq_len=_cfg.list_len * 2 + 1,
                          vocab=N_DIGITS + 2, device=DEV, seed=0)
        train_ds, val_ds = get_dataset(
            list_len=_cfg.list_len, n_digits=N_DIGITS, train_split=0.8,
            mask_tok=N_DIGITS, sep_tok=N_DIGITS + 1, seed=0,
        )
        set_seeds(_cfg.seed)
        _pin = DEV == "cuda"
        _train_dl = DataLoader(train_ds, batch_size=min(TRAIN_BATCH_SIZE, len(train_ds)),
                               shuffle=True, drop_last=True, pin_memory=_pin)
        _val_dl = DataLoader(val_ds, batch_size=min(VAL_BATCH_SIZE, len(val_ds)),
                             drop_last=False, pin_memory=_pin)
        _model = make_model(n_layers=_cfg.n_layers, n_heads=_cfg.n_heads,
                            d_model=_cfg.d_model, ln=_cfg.use_ln,
                            use_bias=_cfg.use_bias, use_wv=_cfg.use_wv,
                            use_wo=_cfg.use_wo, attn_only=not _cfg.use_mlp).to(DEV)
        total, _ = count_params(_model)
        print(f"[test] device={DEV}, d_model={_cfg.d_model}, params={total}")
        acc = train(_model, _train_dl, _val_dl, max_steps=300, lr=LR,
                    weight_decay=WEIGHT_DECAY, list_len=_cfg.list_len,
                    vocab=N_DIGITS + 2, device=DEV, early_stop_acc=EARLY_STOP_ACC)
        print(f"[test] PASSED — best_acc={acc:.4f}")
    else:
        sweep_transformer()
