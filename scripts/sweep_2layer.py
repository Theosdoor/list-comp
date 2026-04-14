"""
W&B Sweep Script for 2-Layer Architecture Grid Search
"""

from pathlib import Path

import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
from dotenv import load_dotenv

from src.data.datasets import get_dataset
from src.models.transformer import make_model
from src.models.utils import accuracy, save_model
from src.utils.runtime import configure_runtime


WANDB_PROJECT = "order-by-scale"
LIST_LEN = 2
N_DIGITS = 100
VOCAB = N_DIGITS + 2
SEQ_LEN = LIST_LEN * 2 + 1
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# cuDNN auto-tunes kernels for fixed input shapes — free win for grid sweeps
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

LR = 1e-3
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 2048
VAL_BATCH_SIZE = 4096
MAX_STEPS = 100_000
EARLY_STOP_ACC = 0.999


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cycle(dl):
    while True:
        for batch in dl:
            yield batch


def train(model, train_dl, val_dl, max_steps: int, early_stop_acc: float) -> float:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    use_amp = DEV == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    iter_dl = _cycle(train_dl)
    best_acc = 0.0
    model.train()

    for step in range(1, max_steps + 1):
        inputs, targets = next(iter_dl)
        inputs = inputs.to(DEV, non_blocking=True)
        targets = targets.to(DEV, non_blocking=True)

        with torch.autocast(device_type=DEV, dtype=torch.float16, enabled=use_amp):
            logits = model(inputs)[:, LIST_LEN + 1 :]
            loss = criterion(
                logits.reshape(-1, VOCAB),
                targets[:, LIST_LEN + 1 :].reshape(-1),
            )

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step % 100 == 0:
            val_acc = accuracy(model, val_dl, list_len=LIST_LEN, device=DEV)
            best_acc = max(best_acc, val_acc)
            if wandb.run is not None:
                wandb.log({
                    "val/accuracy": val_acc,
                    "train/loss": loss.item(),
                    "step": step,
                })
            if val_acc >= early_stop_acc:
                break
            model.train()

    return best_acc


def sweep_2layer():
    load_dotenv()
    run = wandb.init(project=WANDB_PROJECT)
    config = wandb.config

    d_model = config.get("d_model", 64)
    use_ln = config.use_ln
    use_bias = config.use_bias
    use_wv = config.use_wv
    use_wo = config.use_wo
    use_mlp = config.use_mlp
    seed = config.seed

    def flag(value: bool) -> str:
        return "T" if value else "F"

    run.name = (
        f"d{d_model}_ln{flag(use_ln)}_bias{flag(use_bias)}_"
        f"wv{flag(use_wv)}_wo{flag(use_wo)}_mlp{flag(use_mlp)}_s{seed}"
    )

    configure_runtime(
        list_len=LIST_LEN,
        seq_len=SEQ_LEN,
        vocab=VOCAB,
        device=DEV,
        seed=0,
    )

    train_ds, val_ds = get_dataset(
        list_len=LIST_LEN,
        n_digits=N_DIGITS,
        train_split=0.8,
        mask_tok=N_DIGITS,
        sep_tok=N_DIGITS + 1,
        seed=0,
    )

    set_seeds(seed)

    train_batch_size = min(TRAIN_BATCH_SIZE, len(train_ds))
    val_batch_size = min(VAL_BATCH_SIZE, len(val_ds))
    _pin = DEV == "cuda"
    train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, drop_last=True,
                          pin_memory=_pin, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=val_batch_size, drop_last=False,
                        pin_memory=_pin, num_workers=2)

    model = make_model(
        n_layers=2,
        n_heads=1,
        d_model=d_model,
        ln=use_ln,
        use_bias=use_bias,
        use_wv=use_wv,
        use_wo=use_wo,
        attn_only=not use_mlp,
    ).to(DEV)

    best_acc = train(
        model,
        train_dl,
        val_dl,
        max_steps=MAX_STEPS,
        early_stop_acc=EARLY_STOP_ACC,
    )

    wandb.log({"final/val_accuracy": best_acc})
    wandb.summary["final/val_accuracy"] = best_acc

    if (not use_ln and not use_bias and not use_wv and not use_wo and not use_mlp and d_model == 64):
        base_dir = Path(__file__).resolve().parents[1] / "models" / "2_layer_sweep"
        model_path = base_dir / f"{run.name}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, str(model_path))

    wandb.finish()


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        # Smoke-test: run a single config for 300 steps without touching W&B
        import types
        _cfg = types.SimpleNamespace(
            d_model=64, use_ln=False, use_bias=False,
            use_wv=False, use_wo=False, use_mlp=False, seed=0,
        )

        configure_runtime(list_len=LIST_LEN, seq_len=SEQ_LEN, vocab=VOCAB, device=DEV, seed=0)
        train_ds, val_ds = get_dataset(
            list_len=LIST_LEN, n_digits=N_DIGITS, train_split=0.8,
            mask_tok=N_DIGITS, sep_tok=N_DIGITS + 1, seed=0,
        )
        set_seeds(_cfg.seed)
        _pin = DEV == "cuda"
        _train_dl = DataLoader(train_ds, batch_size=min(TRAIN_BATCH_SIZE, len(train_ds)),
                               shuffle=True, drop_last=True, pin_memory=_pin, num_workers=2)
        _val_dl = DataLoader(val_ds, batch_size=min(VAL_BATCH_SIZE, len(val_ds)),
                             drop_last=False, pin_memory=_pin, num_workers=2)
        _model = make_model(n_layers=2, n_heads=1, d_model=_cfg.d_model,
                            ln=_cfg.use_ln, use_bias=_cfg.use_bias,
                            use_wv=_cfg.use_wv, use_wo=_cfg.use_wo,
                            attn_only=not _cfg.use_mlp).to(DEV)
        print(f"[test] device={DEV}, d_model={_cfg.d_model}")
        acc = train(_model, _train_dl, _val_dl, max_steps=300, early_stop_acc=EARLY_STOP_ACC)
        print(f"[test] PASSED — best_acc={acc:.4f}")
    else:
        sweep_2layer()
