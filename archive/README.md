# Archive

This directory contains exploration artifacts, intermediate results, and superseded code from the development process. It's kept for reference and reproducibility but not essential to the paper.

## Contents

### `notebooks/`
Rough Jupyter notebooks and their Python equivalents from exploratory phases (Jan–Apr 2026). These contain:
- Attention pattern analysis
- Feature activation studies
- SAE evaluation experiments
- Miscellaneous hypothesis testing

**Status:** Exploratory. Not required for paper reproduction.

### `exploratory_scripts/`
Python scripts used during development but not part of the core paper pipeline:
- `nb_grid_search_heatmap.py` — Grid search visualization (LIST_LEN × N_LAYERS heatmap)
- `sweep_transformer.py` — General W&B sweep script for hyperparameter sweeps (exploratory architecture search)

**Status:** Exploratory/superseded. Reference only.

### `sweeps/`
WandB sweep configuration files used during exploratory hyperparameter searches:
- `2layer_flags.yaml`, `2layer_dmodel.yaml`, `2layer_nheads.yaml` — Architecture sweep configs (2-layer models)
- `btksae_sweep.yaml` — SAE hyperparameter sweep (batch top-k)
- Other experimental sweep configs

**Status:** Exploratory. Sweep IDs and results are logged in `EXPERIMENTS.md`.

## For Paper Reproducibility

You do **not** need anything in this directory. See the main README.md and `visualization/` for figure reproduction and `scripts/` for core model/SAE training.

## Re-activating Exploration Code

If you want to re-use code from this archive:
1. Move the script back to `scripts/` or `visualization/`, or run it directly from the archive
2. Verify sys.path imports still work (scripts have been updated for archive location)
3. Check timestamps in notebook names (format: `nb_YYYY_MM_DD`) to understand experimental timeline
4. For sweep configs, reference `EXPERIMENTS.md` for the corresponding sweep ID and results
