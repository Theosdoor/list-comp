# Visualization & Figure Generation

Scripts to reproduce figures and tables from the paper.

## Core Figure Scripts

### `make_2layer_table.py`
Generates Table 1 (2-layer architecture sweep results) from WandB sweeps.

**Usage:**
```bash
python3 make_2layer_table.py --flags-sweep-ids <id1> --dmodel-sweep-ids <id2> --nheads-sweep-ids <id3>
```

**Output:** Markdown and LaTeX tables saved to `results/`

### `nb_compare_sae.py`
Compares multiple SAE checkpoints and generates a summary table of metrics (L0, MSE, reconstruction quality).

**Usage:**
```bash
python3 nb_compare_sae.py
```

**Output:** Comparison table (markdown format) to stdout

### `plot_sae_sweep.py`
Plots SAE hyperparameter sweep results aggregated by (L0, d_sae) across seeds. Generates markdown + LaTeX tables and scatter plots.

**Usage:**
```bash
python3 plot_sae_sweep.py
python3 plot_sae_sweep.py --report path/to/sae_comparison.md
python3 plot_sae_sweep.py --output-dir report/figures
```

**Output:** PNG plots and tables saved to `results/` or `report/`

## Analysis & Mechanistic Interpretation

### `nb_model_interp.py`
Attention pattern and residual-stream analysis. Examines attention flow through the SEP bottleneck and token-to-token dependencies.

**Usage:** Jupyter notebook or `python3 nb_model_interp.py`

### `nb_sae_feat_analysis.py`
Single SAE feature analysis: heatmaps, firing rate histograms, digit distribution, and correlation with attention patterns.

**Usage:** Jupyter notebook or `python3 nb_sae_feat_analysis.py`

### `special_latents_across_saes.py`
Correlates special feature count with SAE performance metrics. Produces swarm plots and Spearman correlations.

**Usage:**
```bash
python3 special_latents_across_saes.py
```

**Output:** Plots saved to `results/special_latent_analysis/`

## Running All Figures

For complete figure reproduction:
1. Train baseline model (see `scripts/train_model.py`)
2. Train SAE (see `scripts/train_sae.py`)
3. Run visualization scripts in this order:
   - `python3 make_2layer_table.py ...` (architecture sweep table)
   - `python3 nb_compare_sae.py` (SAE comparison)
   - `python3 plot_sae_sweep.py` (SAE sweep plots)
   - `python3 special_latents_across_saes.py` (special features analysis)
   - `python3 nb_model_interp.py` (mechanistic interpretation)
   - `python3 nb_sae_feat_analysis.py` (feature analysis)
4. Results populate `results/` and can be inserted into paper

## Future: Jupyter Notebooks

Some scripts in this directory are being converted to Jupyter notebooks for better interactivity. Check for `.ipynb` versions.
