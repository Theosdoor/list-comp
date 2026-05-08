from types import SimpleNamespace

import pytest
import torch

from src.sae import loading
from src.utils import nb_utils


def test_load_transformer_model_uses_parsed_list_len(monkeypatch, tmp_path):
    runtime_calls = []

    monkeypatch.setattr(
        nb_utils,
        "parse_model_name_safe",
        lambda name: SimpleNamespace(
            n_digits=11,
            d_model=32,
            n_layers=3,
            list_len=4,
        ),
    )
    monkeypatch.setattr(
        nb_utils,
        "configure_runtime",
        lambda **kwargs: runtime_calls.append(kwargs),
    )
    monkeypatch.setattr(nb_utils, "_load_model", lambda *args, **kwargs: "model")

    model, config = nb_utils.load_transformer_model(
        "L3_H1_D32_V11_Len4",
        device="cpu",
        models_dir=tmp_path,
        n_heads=2,
    )

    assert model == "model"
    assert runtime_calls == [
        {
            "list_len": 4,
            "seq_len": 9,
            "vocab": 13,
            "device": "cpu",
        }
    ]
    assert config["list_len"] == 4
    assert config["sep_token_index"] == 4


def test_load_sae_checkpoint_normalizes_legacy_btk_and_moves_act_mean(
    monkeypatch, tmp_path
):
    loaded_state_dicts = []

    class FakeSAE:
        def load_state_dict(self, state_dict):
            loaded_state_dicts.append(state_dict)

    monkeypatch.setattr(
        loading,
        "instantiate_sae_from_cfg",
        lambda cfg, d_model, device: FakeSAE(),
    )

    checkpoint = {
        "cfg": {"sae_type": "btk", "dict_size": 3, "k": 1},
        "state_dict": {
            "W_enc": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "b_enc": torch.tensor([0.1, 0.2, 0.3]),
            "W_dec": torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
            "b_dec": torch.tensor([0.4, 0.5]),
        },
        "act_mean": torch.tensor([1.0, 2.0]),
    }
    path = tmp_path / "sae.pt"
    path.write_bytes(b"unused")
    monkeypatch.setattr(
        loading.torch,
        "load",
        lambda loaded_path, map_location, weights_only: checkpoint,
    )

    result = loading.load_sae_checkpoint(path, d_model=2, device="cpu")

    assert result["sae"].__class__ is FakeSAE
    assert torch.equal(
        loaded_state_dicts[0]["encoder.weight"],
        checkpoint["state_dict"]["W_enc"].T,
    )
    assert torch.equal(result["act_mean"], checkpoint["act_mean"].to("cpu"))
    assert result["config"]["dict_size"] == 3
    assert result["config"]["d_sae"] == 3
    assert result["config"]["activation_dim"] == 2
    assert result["checkpoint"] is checkpoint


def test_load_sae_checkpoint_infers_activation_dim_from_act_mean_when_d_model_missing(
    monkeypatch, tmp_path
):
    instantiated = []

    class FakeSAE:
        def load_state_dict(self, state_dict):
            pass

    monkeypatch.setattr(
        loading,
        "instantiate_sae_from_cfg",
        lambda cfg, d_model, device: instantiated.append(d_model) or FakeSAE(),
    )

    checkpoint = {
        "cfg": {"sae_type": "btk", "dict_size": 3, "k": 1},
        "state_dict": {
            "encoder.weight": torch.zeros(3, 5),
            "encoder.bias": torch.zeros(3),
            "decoder.weight": torch.zeros(3, 5),
            "decoder.bias": torch.zeros(5),
        },
        "act_mean": torch.zeros(5),
    }
    path = tmp_path / "sae.pt"
    path.write_bytes(b"unused")
    monkeypatch.setattr(
        loading.torch,
        "load",
        lambda loaded_path, map_location, weights_only: checkpoint,
    )

    result = loading.load_sae_checkpoint(path, d_model=None, device="cpu")

    assert instantiated == [5]
    assert result["config"]["activation_dim"] == 5


def test_nb_utils_load_sae_delegates_to_canonical_loader(monkeypatch, tmp_path):
    path = tmp_path / "sae.pt"
    path.write_bytes(b"unused")
    canonical = {
        "sae": "sae",
        "config": {"dict_size": 5, "d_sae": 5, "act_mean": torch.zeros(2)},
    }
    calls = []

    monkeypatch.setattr(
        nb_utils,
        "load_sae_checkpoint",
        lambda sae_path, d_model, device: calls.append((sae_path, d_model, device))
        or canonical,
    )

    sae, config = nb_utils.load_sae(path, d_model=2, device="cpu")

    assert calls == [(path, 2, "cpu")]
    assert sae == "sae"
    assert config is canonical["config"]


def test_load_sae_from_local_delegates_to_canonical_loader(monkeypatch, tmp_path):
    calls = []
    canonical = {
        "sae": "sae",
        "act_mean": torch.zeros(2),
        "config": {"d_sae": 5, "sae_type": "btk"},
        "checkpoint": {"cfg": {"sae_type": "btk"}, "final_loss": 0.1},
    }

    monkeypatch.setattr(
        loading,
        "load_sae_checkpoint",
        lambda sae_path, d_model, device: calls.append((sae_path, d_model, device))
        or canonical,
    )

    result = loading.load_sae_from_local(
        "sae.pt",
        d_model=2,
        device="cpu",
        sae_dir=tmp_path,
    )

    assert calls == [(str(tmp_path / "sae.pt"), 2, "cpu")]
    assert result is canonical


def test_wandb_loader_uses_canonical_checkpoint_loader(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "downloaded.pt"
    checkpoint_path.write_bytes(b"unused")
    calls = []

    class FakeArtifact:
        def download(self, root):
            return str(tmp_path)

    class FakeRun:
        config = {
            "sae_type": "btk",
            "d_sae": 7,
            "top_k": 2,
            "seed": 123,
        }

        def use_artifact(self, name):
            return FakeArtifact()

    class FakeApi:
        def run(self, name):
            return FakeRun()

    monkeypatch.setattr(loading.wandb, "Api", lambda: FakeApi())
    monkeypatch.setattr(
        loading,
        "load_sae_checkpoint",
        lambda sae_path, d_model, device: calls.append((sae_path, d_model, device))
        or {
            "sae": "sae",
            "act_mean": torch.zeros(4),
            "config": {"activation_dim": 4},
            "checkpoint": {"state_dict": {}},
        },
    )

    result = loading.load_sae_from_wandb_run(
        "run-id",
        project="entity/project",
        download_dir=str(tmp_path / "downloads"),
        device="cpu",
    )

    assert calls == [(str(checkpoint_path), None, "cpu")]
    assert result["sae"] == "sae"
    assert result["run_config"] == FakeRun.config


def test_wandb_loader_requires_explicit_project_when_default_is_anonymized():
    with pytest.raises(ValueError, match="project must be provided"):
        loading.load_sae_from_wandb_run("run-id")


def test_compare_sweep_runs_requires_explicit_project_when_default_is_anonymized():
    with pytest.raises(ValueError, match="project must be provided"):
        loading.compare_sweep_runs()
