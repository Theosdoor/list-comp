import itertools
import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_2layer_table import (  # noqa: E402
    compute_stats,
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
                    "run_name": f"flags_{''.join('T' if f else 'F' for f in flags)}_seed{seed}.pt",
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
                    "run_name": f"model_{d_model}_seed{seed}.pt",
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
    assert len(lines) == 3
    assert all(".pt" in line and "acc=" in line for line in lines)
