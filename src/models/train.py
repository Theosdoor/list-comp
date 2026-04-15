"""
Shared training loop for list-comparison transformers.

Both scripts/train_model.py and scripts/sweep_2layer.py import from here
so training logic stays in one place.
"""

import itertools
import math

import torch
from tqdm.auto import tqdm


def train(
    model,
    train_dl,
    val_dl,
    *,
    # Core
    max_steps: int = 100_000,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    # Context needed for output-slice loss
    list_len: int = 2,
    vocab: int,
    device: str = "cpu",
    # Stopping
    early_stop_acc: float = 1.0,
    patience: int | None = None,
    patience_threshold: float = 0.9,
    # LR schedule
    use_lr_scheduler: bool = False,
    warmup_steps: int = 1_000,
    # Regularisation
    max_grad_norm: float | None = None,
    # Checkpointing
    checkpoint_every: int | None = None,
    checkpoint_path: str | None = None,
    # Logging
    use_wandb: bool = False,
    show_progress: bool = True,
) -> float:
    """Train *model* in-place, restoring the best checkpoint at the end.

    Returns
    -------
    float
        Best validation accuracy seen during training.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    scheduler = None
    if use_lr_scheduler:
        def _lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    dl = itertools.cycle(train_dl)
    pbar = tqdm(range(1, max_steps + 1), desc="Training", disable=not show_progress)

    best_acc = 0.0
    best_state: dict | None = None
    steps_without_improvement = 0

    from src.models.utils import accuracy as _accuracy  # local import avoids circular

    for step in pbar:
        inputs, targets = next(dl)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        model.train()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            logits = model(inputs)[:, list_len + 1:]
            loss = criterion(
                logits.reshape(-1, vocab),
                targets[:, list_len + 1:].reshape(-1),
            )

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if max_grad_norm is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(opt)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        if step % 100 == 0:
            val_acc = _accuracy(model, val_dl, list_len=list_len, device=device)
            current_lr = opt.param_groups[0]["lr"]

            # Track best and save state
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            if use_wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            "val/accuracy": val_acc,
                            "train/loss": loss.item(),
                            "train/lr": current_lr,
                            "train/pct_complete": step / max_steps,
                            "step": step,
                        })
                except ImportError:
                    pass

            if show_progress:
                postfix = {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{val_acc:.2%}",
                    "best": f"{best_acc:.2%}",
                }
                if scheduler is not None:
                    postfix["lr"] = f"{current_lr:.2e}"
                pbar.set_postfix(postfix)

            # Periodic checkpoint
            if checkpoint_every and checkpoint_path and step % checkpoint_every == 0:
                from src.models.utils import save_model
                save_model(model, checkpoint_path)

            # Target-accuracy early stop
            if val_acc >= early_stop_acc:
                print(f"Early stop at step {step}: acc {val_acc:.2%} >= {early_stop_acc:.2%}")
                break

            # Patience-based early stop (only above threshold)
            if (
                patience
                and steps_without_improvement >= patience
                and val_acc >= patience_threshold
            ):
                print(
                    f"Early stop at step {step}: no improvement for {patience} evals "
                    f"(acc={val_acc:.2%}, best={best_acc:.2%})"
                )
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return best_acc
