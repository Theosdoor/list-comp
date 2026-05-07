import importlib.util
from pathlib import Path

import pandas as pd


def _load_plot_sae_sweep_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_sae_sweep.py"
    spec = importlib.util.spec_from_file_location("plot_sae_sweep", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_latex_table_bolds_all_tied_display_values(tmp_path):
    plot_sae_sweep = _load_plot_sae_sweep_module()
    agg = pd.DataFrame(
        [
            {
                "l0": 3,
                "d_sae": 128,
                "n_runs": 10,
                "loss_recovered_mean": 0.9948,
                "loss_recovered_std": 0.0100,
                "dead_pct_mean": 14.1,
                "dead_pct_std": 1.0,
                "n_special_mean": 1.0,
                "n_special_std": 0.0,
            },
            {
                "l0": 3,
                "d_sae": 192,
                "n_runs": 10,
                "loss_recovered_mean": 0.99964,
                "loss_recovered_std": 0.0001,
                "dead_pct_mean": 20.0,
                "dead_pct_std": 1.0,
                "n_special_mean": 1.0,
                "n_special_std": 0.0,
            },
            {
                "l0": 3,
                "d_sae": 256,
                "n_runs": 10,
                "loss_recovered_mean": 0.99963,
                "loss_recovered_std": 0.0001,
                "dead_pct_mean": 30.0,
                "dead_pct_std": 1.0,
                "n_special_mean": 1.0,
                "n_special_std": 0.0,
            },
            {
                "l0": 3,
                "d_sae": 320,
                "n_runs": 10,
                "loss_recovered_mean": 0.99962,
                "loss_recovered_std": 0.0001,
                "dead_pct_mean": 40.0,
                "dead_pct_std": 1.0,
                "n_special_mean": 1.0,
                "n_special_std": 0.0,
            },
        ]
    )

    plot_sae_sweep.write_latex_table(
        agg,
        tmp_path,
        no_errors=False,
        exclude_runs_col=False,
        exclude_special_col=True,
    )

    table_text = (tmp_path / "sae_sweep_table.tex").read_text()
    assert table_text.count(r"\mathbf{0.9996 \pm 0.0001}") == 3


def test_latex_table_keeps_plus_minus_when_std_is_zero(tmp_path):
    plot_sae_sweep = _load_plot_sae_sweep_module()
    agg = pd.DataFrame(
        [
            {
                "l0": 4,
                "d_sae": 128,
                "n_runs": 17,
                "loss_recovered_mean": 0.9996,
                "loss_recovered_std": 0.0,
                "dead_pct_mean": 1.1,
                "dead_pct_std": 0.9,
                "n_special_mean": 1.9,
                "n_special_std": 0.2,
            }
        ]
    )

    plot_sae_sweep.write_latex_table(
        agg,
        tmp_path,
        no_errors=False,
        exclude_runs_col=False,
        exclude_special_col=True,
    )

    table_text = (tmp_path / "sae_sweep_table.tex").read_text()
    assert r"\mathbf{0.9996 \pm 0.0000}" in table_text
