import itertools
import re
import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.make_2layer_table as make_2layer_table  # noqa: E402
from scripts.make_2layer_table import (  # noqa: E402
    compute_stats,
    fetch_runs,
    fmt_acc,
    make_compact_table,
    make_dmodel_table,
    make_flags_table,
    print_top3_fffff,
    tf,
)


@pytest.fixture
def mock_df():
    rows = []
    run_counter = 0
    sweep_flags = "flags_sweep"
    sweep_dmodel = "dmodel_sweep"

    for flags in itertools.product([False, True], repeat=5):
        for seed in range(3):
            run_counter += 1
            val_accuracy = 0.82 + 0.01 * sum(flags) + 0.001 * seed
            rows.append(
                {
                    "sweep_id": sweep_flags,
                    "run_name": f"flags_{''.join('T' if f else 'F' for f in flags)}_seed{seed}",
                    "run_id": f"run_{run_counter}",
                    "d_model": 64,
                    "use_ln": flags[0],
                    "use_bias": flags[1],
                    "use_wv": flags[2],
                    "use_wo": flags[3],
                    "use_mlp": flags[4],
                    "seed": seed,
                    "val_accuracy": val_accuracy,
                }
            )

    for d_model in [8, 32, 128]:
        for seed in range(3):
            run_counter += 1
            val_accuracy = 0.75 + 0.001 * seed + (d_model / 5120)
            rows.append(
                {
                    "sweep_id": sweep_dmodel,
                    "run_name": f"model_{d_model}_seed{seed}",
                    "run_id": f"run_{run_counter}",
                    "d_model": d_model,
                    "use_ln": False,
                    "use_bias": False,
                    "use_wv": False,
                    "use_wo": False,
                    "use_mlp": False,
                    "seed": seed,
                    "val_accuracy": val_accuracy,
                }
            )

    return pd.DataFrame(rows)


def test_tf():
    assert tf(True) == "T"
    assert tf(False) == "F"


def test_fmt_acc_plain_and_bold():
    assert fmt_acc(0.123456) == "0.1235"
    assert fmt_acc(0.9, bold=True) == "\\textbf{0.9000}"


def test_compute_stats_shapes_and_bounds(mock_df):
    stats = compute_stats(mock_df)
    required_columns = {"mean", "max", "min", "median", "n_seeds"}
    assert len(stats) == 35
    assert required_columns.issubset(stats.columns)
    assert (stats["n_seeds"] == 3).all()
    assert stats["mean"].between(mock_df["val_accuracy"].min(), mock_df["val_accuracy"].max()).all()


def test_make_flags_table_only_dmodel_64(mock_df):
    stats = compute_stats(mock_df)
    table = make_flags_table(stats)
    data_lines = [line for line in table.splitlines() if line.strip().startswith("64 &")]
    assert len(data_lines) == 32
    assert all(line.startswith("64 &") for line in data_lines)


def test_make_dmodel_table_excludes_64(mock_df):
    stats = compute_stats(mock_df)
    table = make_dmodel_table(stats)
    data_lines = [line for line in table.splitlines() if line.strip() and line[0].isdigit()]
    d_models = {int(line.split("&")[0].strip()) for line in data_lines}
    assert 64 not in d_models
    assert d_models == {8, 32, 128}


def test_make_compact_table_has_midrule(mock_df):
    stats = compute_stats(mock_df)
    table = make_compact_table(stats)
    assert "\\begin{tabular}" in table
    assert "\\midrule" in table
    assert "\\end{tabular}" in table


def test_print_top3_fffff_outputs_three_lines(mock_df, capsys):
    print_top3_fffff(mock_df)
    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().splitlines() if line.strip()]
    assert len(lines) == 4
    assert "models/2_layer_sweep" in lines[0]
    pattern = re.compile(r".+\.pt\s+\(seed=-?\d+, acc=\d\.\d{4}\)")
    assert all(pattern.match(line) for line in lines[1:])


def test_fetch_runs_normalizes_defaults_and_types(monkeypatch):
    class DummyRun:
        def __init__(self, state, summary, config, name, run_id):
            self.state = state
            self.summary = summary
            self.config = config
            self.name = name
            self.id = run_id

    class DummySweep:
        def __init__(self, runs):
            self.runs = runs

    class DummyApi:
        def __init__(self, sweeps):
            self.sweeps = sweeps

        def sweep(self, path):
            sweep_id = path.split("/")[-1]
            return self.sweeps[sweep_id]

    runs = [
        DummyRun(
            state="finished",
            summary={"final/val_accuracy": "0.9123"},
            config={
                "d_model": "128",
                "use_ln": "true",
                "use_bias": 1,
                "use_wv": 0,
                "use_wo": None,
                "use_mlp": False,
                "seed": "7",
            },
            name="run_a",
            run_id="run_a_id",
        ),
        DummyRun(
            state="finished",
            summary={"final/val_accuracy": 0.845},
            config={},
            name="run_b",
            run_id="run_b_id",
        ),
    ]

    sweeps = {"sweep_1": DummySweep(runs)}
    monkeypatch.setattr(make_2layer_table.wandb, "Api", lambda: DummyApi(sweeps))

    df = fetch_runs(["sweep_1"])
    row_a = df.loc[df["run_id"] == "run_a_id"].iloc[0]
    assert row_a["d_model"] == 128
    assert bool(row_a["use_ln"]) is True
    assert bool(row_a["use_bias"]) is True
    assert bool(row_a["use_wv"]) is False
    assert bool(row_a["use_wo"]) is False
    assert bool(row_a["use_mlp"]) is False
    assert row_a["seed"] == 7
    assert isinstance(row_a["val_accuracy"], float)

    row_b = df.loc[df["run_id"] == "run_b_id"].iloc[0]
    assert row_b["d_model"] == 64
    assert bool(row_b["use_ln"]) is False
    assert bool(row_b["use_bias"]) is False
    assert bool(row_b["use_wv"]) is False
    assert bool(row_b["use_wo"]) is False
    assert bool(row_b["use_mlp"]) is False
    assert row_b["seed"] == -1


def test_fetch_runs_raises_when_no_valid_val_accuracy(monkeypatch):
    class DummyRun:
        def __init__(self, state, summary, config, name, run_id):
            self.state = state
            self.summary = summary
            self.config = config
            self.name = name
            self.id = run_id

    class DummySweep:
        def __init__(self, runs):
            self.runs = runs

    class DummyApi:
        def __init__(self, sweeps):
            self.sweeps = sweeps

        def sweep(self, path):
            sweep_id = path.split("/")[-1]
            return self.sweeps[sweep_id]

    runs = [
        DummyRun(
            state="finished",
            summary={"final/val_accuracy": "not-a-number"},
            config={},
            name="run_bad",
            run_id="run_bad_id",
        )
    ]
    sweeps = {"sweep_2": DummySweep(runs)}
    monkeypatch.setattr(make_2layer_table.wandb, "Api", lambda: DummyApi(sweeps))

    with pytest.raises(SystemExit, match="No valid val_accuracy"):
        fetch_runs(["sweep_2"])
