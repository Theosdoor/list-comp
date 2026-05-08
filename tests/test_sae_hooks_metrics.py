import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.sae.hooks import (
    _patch_sep_reconstruction,
    make_batched_sae_patch_hook,
    make_dynamic_sae_patch_hook,
    make_sae_patch_hook,
    make_zero_sep_hook,
)
from src.sae.metrics import _accumulate_output_ce, compute_sae_downstream_metrics
from src.utils.runtime import configure_runtime


def test_patch_sep_reconstruction_clones_and_adds_mean_on_activation_device():
    activations = torch.zeros(2, 4, 3)
    recon = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    act_mean = torch.tensor([0.5, 1.5, 2.5])

    patched = _patch_sep_reconstruction(activations, recon, act_mean, sep_idx=2)

    assert patched is not activations
    assert torch.equal(activations, torch.zeros_like(activations))
    assert torch.equal(patched[:, 2, :], recon + act_mean.to(activations.device))
    assert torch.equal(patched[:, 0, :], activations[:, 0, :])


def test_public_sae_patch_hooks_share_sep_patch_behavior():
    activations = torch.zeros(2, 4, 3)
    recon = torch.ones(2, 3)
    act_mean = torch.tensor([0.5, 1.5, 2.5])

    for hook in (
        make_sae_patch_hook(recon, act_mean, sep_idx=1),
        make_batched_sae_patch_hook(recon, act_mean, sep_idx=1),
    ):
        patched = hook(activations, hook=None)
        assert torch.equal(patched[:, 1, :], recon + act_mean.to(activations.device))
        assert torch.equal(activations, torch.zeros_like(activations))


def test_dynamic_sae_patch_hook_reconstructs_sep_then_adds_mean():
    class AddOneSAE:
        def encode(self, activations, use_threshold=True):
            assert use_threshold is True
            return activations + 1

        def decode(self, sae_z):
            return sae_z * 2

    activations = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    act_mean = torch.tensor([0.5, 1.5, 2.5])
    hook = make_dynamic_sae_patch_hook(AddOneSAE(), act_mean, sep_idx=2)

    patched = hook(activations, hook=None)

    sep_centered = activations[:, 2, :] - act_mean
    expected = (sep_centered + 1) * 2 + act_mean
    assert torch.equal(patched[:, 2, :], expected)
    assert torch.equal(patched[:, 0, :], activations[:, 0, :])


def test_zero_sep_hook_clones_and_only_zeroes_sep_position():
    activations = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    hook = make_zero_sep_hook(sep_idx=2)

    patched = hook(activations, hook=None)

    assert patched is not activations
    assert torch.equal(patched[:, 2, :], torch.zeros_like(patched[:, 2, :]))
    assert torch.equal(patched[:, 0, :], activations[:, 0, :])
    assert torch.equal(activations[:, 2, :], torch.tensor([[6.0, 7.0, 8.0], [18.0, 19.0, 20.0]]))


def test_accumulate_output_ce_matches_cross_entropy_sum_and_accuracy():
    logits = torch.tensor(
        [
            [[4.0, 0.0, 1.0], [0.0, 3.0, 1.0]],
            [[0.0, 2.0, 5.0], [1.0, 4.0, 0.0]],
        ]
    )
    targets = torch.tensor([[0, 2], [2, 1]])

    ce_sum, correct, token_count = _accumulate_output_ce(logits, targets)

    expected_ce = F.cross_entropy(logits.reshape(4, 3), targets.reshape(4), reduction="sum")
    assert ce_sum == pytest.approx(expected_ce.item())
    assert correct == 3
    assert token_count == 4


def test_compute_sae_downstream_metrics_preserves_public_keys_and_semantics():
    configure_runtime(list_len=1, seq_len=3, vocab=3, device="cpu")

    class IdentitySAE:
        def encode(self, activations, use_threshold=True):
            return activations

        def decode(self, sae_z):
            return sae_z

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, inputs):
            return self._logits(inputs, mode="baseline")

        def run_with_hooks(self, inputs, fwd_hooks):
            _, hook_fn = fwd_hooks[0]
            activations = torch.ones(len(inputs), 3, 2)
            patched = hook_fn(activations, hook=None)
            mode = "zero" if torch.equal(patched[:, 1, :], torch.zeros_like(patched[:, 1, :])) else "patched"
            return self._logits(inputs, mode=mode)

        def _logits(self, inputs, mode):
            logits = torch.zeros(len(inputs), 3, 3)
            values = {
                "baseline": torch.tensor([4.0, 0.0, 0.0]),
                "patched": torch.tensor([2.0, 0.0, 0.0]),
                "zero": torch.tensor([0.0, 4.0, 0.0]),
            }[mode]
            logits[:, 2, :] = values
            return logits

    inputs = torch.tensor([[0, 2, 2], [1, 2, 2]])
    targets = torch.tensor([[0, 2, 0], [1, 2, 0]])
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    metrics = compute_sae_downstream_metrics(
        FakeModel(),
        IdentitySAE(),
        dataloader,
        act_mean=torch.zeros(2),
        layer_idx=0,
        sep_idx=1,
        device="cpu",
    )

    assert set(metrics) == {
        "baseline_acc",
        "reconstruction_acc",
        "accuracy_drop",
        "total_samples",
        "total_tokens",
        "h_orig",
        "h_star",
        "h0",
        "zero_ce",
        "loss_recovered",
    }
    assert metrics["baseline_acc"] == 1.0
    assert metrics["reconstruction_acc"] == 1.0
    assert metrics["accuracy_drop"] == 0.0
    assert metrics["total_samples"] == 2
    assert metrics["total_tokens"] == 2
    assert metrics["zero_ce"] == metrics["h0"]
    assert metrics["h_orig"] < metrics["h_star"] < metrics["h0"]
    expected_loss_recovered = (metrics["h_star"] - metrics["h0"]) / (
        metrics["h_orig"] - metrics["h0"]
    )
    assert metrics["loss_recovered"] == pytest.approx(expected_loss_recovered)
