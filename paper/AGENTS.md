# AGENTS.md

This file provides guidance to LLM coding agents when working with code in this repository.

Key: **must** follow paper formatting instructions at `paper/formatting_instructions.tex`.

## Companion Code Repositories

The experiments described in this thesis live in two separate repos:

| Repo | Purpose |
|------|---------|
| Main experiment repo | RQ1 + RQ2: model training, SAE grid search, feature analysis, steering pipeline. Has `src/`, `scripts/`, `notebooks/`, `sweeps/`, `results/`, `EXPERIMENTS.md` |
| Prior-work repo | Older/simpler code for the previously published workshop paper. Has `train.py`, `interp.ipynb`, `src/` |

`list-comp` is the authoritative source for all experiment results in this thesis. Key files from its CLAUDE.md:

| File | Role |
|------|------|
| `src/models/transformer.py` | Model construction, custom attention mask |
| `src/data/datasets.py` | Dataset generation (`get_dataset()`) |
| `src/utils/nb_utils.py` | `load_transformer_model()` — default model loader |
| `src/sae/steering.py` | Feature steering: `get_xovers_df`, `swap_outputs` |
| `src/sae/activation_collection.py` | `identify_special_features()` — finds features with \|r\| > threshold vs α\_diff |

**Canonical model**: `models/2layer_100dig_64d.pt`

**Canonical SAE**: k = 3, d\_sae = 128. The existing checkpoint `sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d.pt` (d\_sae = 100) predates this decision — a d\_sae = 128 run is the intended reported configuration.

---

## Notes Directory

`notes/` contains researcher notes that are the ground truth for experiment results:

| File | Contents |
|------|---------|
| `notes/btk_sae_notes.md` | Raw experiment notes on SAE grid search, feature heatmaps, F30 steering analysis — **do not overwrite** |
| `notes/REPORT_NOTES.md` | High-level notes on which experiments to include per section |
| `notes/EVALUATION.md` | Current paper evaluation: list of TODOs, incomplete sections, stale content |

When resolving ambiguity about a numerical result (e.g. which k/d\_sae was best), check `notes/btk_sae_notes.md` first.

---

## Build System

```bash
make -C paper          # Build PDF → paper/build/main.pdf
make -C paper clean    # Remove all build artefacts
```

- Uses `latexmk` under the hood for automated compilation.
- **Always run the build after editing any `.tex` file** to catch LaTeX errors early.
- VS Code with LaTeX Workshop is configured to use `make` automatically.
- Output is `paper/build/main.pdf`.

### Build troubleshooting (Biber cache)

If compilation suddenly fails at the `biber` step with errors resembling `Unicode::UCD: failed to find unicore/version`, this is usually a corrupted local Biber PAR cache, not a LaTeX source error.

```bash
rm -rf "$(biber --cache)"
make -C paper
```

Notes:
- The first run after a Biber failure can still report a stale `latexmk` summary (`biber ... gave an error in previous invocation`). Re-run `make -C paper` once.
- Ensure `references.bib` has unique citation keys, because duplicate keys can also cause Biber failures.

---

## Label Conventions

| Prefix | Usage | Example |
|--------|-------|---------|
| `s:` | Sections / appendices | `\label{s:method}`, `\label{s:grid-search}` |
| `ss:` | Subsections | `\label{ss:results-circuits}` |
| `tab:` | Tables | `\label{tab:mask}`, `\label{tab:model_config}` |
| `fig:` | Figures | `\label{fig:attn-flow}`, `\label{fig:logit-diff}` |
| `eq:` | Equations | `\label{eq:o1_logit_composition}` |

- Use `\eqref{}` for equations, `\ref{}` for everything else.
- Citations: use `\citep{}`, `\citet{}`, or `\textcite{}` (biblatex natbib-compatible).

---

## Writing Conventions

- **No `\emph{...}`** in generated prose — the author decides when to introduce emphasis for new terms; do not add it in completions.
- **No em-dashes** (`---` or `—`) — use commas, parentheses, or split sentences.

---

## Key Notation (use consistently throughout the paper)

| Symbol | Meaning |
|--------|---------|
| $d_1, d_2$ | Input digit tokens (values 0–99) |
| $s$ | SEP token |
| $o_1, o_2$ | Output position tokens |
| $\boldsymbol{r}_z^{L_i}$ | Residual stream at position $z$ after layer $i$ |
| $\alpha_{x \to y}$ | Attention probability from token $x$ to token $y$ in layer 1 |
| $\beta_{x \to y}$ | Attention probability from token $x$ to token $y$ in layer 2 |
| $\alpha_\text{diff}$ | $\alpha_{s \to d_1} - \alpha_{s \to d_2}$ (key discriminator) |
| F30 | The "special" SAE feature strongly correlated with $\alpha_\text{diff}$ |

---

## Research Context (for informed editing)

### Task Definition
The toy transformer maps the sequence `[d₁, d₂, SEP, MASK, MASK]` → `[d₁, d₂, SEP, o₁, o₂]`, where $d_1, d_2 \in [0, 99]$ and outputs are sorted (i.e. the smaller digit goes to $o_1$, larger to $o_2$).

### Model Architecture
- 2-layer attention-only transformer (no MLP, no LayerNorm, no bias)
- Value and output matrices frozen to identity to isolate the composition mechanism
- Achieves 92.2% validation accuracy (vs 100% for 3-layer baseline)
- Trained on 80% of 10,000 possible (d₁, d₂) input pairs

### SAE Experiments
- Trained BatchTopK SAEs with grid search over k ∈ {1,2,3,4,5} and d\_sae ∈ {50,100,200,400}
- Best SAEs: k=3 or k=4 with d\_sae=100 (86–89% reconstructed accuracy)
- k=1 and k=2 are too compressed; d\_sae=50 is smaller than d\_model=64 (sanity check: always worse)
- Feature heatmaps are 100×100 grids: cell (i,j) = feature activation magnitude for input (d₁=i, d₂=j)

### Key Findings
1. Most SAE features show "cross" patterns in their heatmaps — they act as digit detectors (activating when a specific digit appears in either $d_1$ or $d_2$ position)
2. "Special features" (1–2 per SAE, identifiable by |correlation with $\alpha_\text{diff}$| > 0.5) have qualitatively different heatmaps
3. F30 (in the k=3, d=100 SAE) correlates strongly with output ordering: scaling it up/down can swap the predicted output order
4. Feature steering with F30 succeeds on ~60% of inputs; ~30% of inputs never activate F30 at all
5. These findings challenge the binary-feature view: SAE latent activations have meaningful *magnitudes*, not just on/off states

### What "Graded Latent Activations" Means
The paper's novel contribution is showing that activation magnitudes in SAEs encode ordinal information. This contradicts the "linear representation hypothesis" and suggests that mechanistic interpretability methods that treat features as binary should be revisited.

---

## Figures in Use

All figures live in `paper/figures/`. Do not rename files without updating `\includegraphics` references.

| File | Used in | Description |
|------|---------|-------------|
| `attn-flow.png` | results | Attention flow diagram |
| `attn-hist.png` | results | Attention weight histogram |
| `confusion-band.png` | results | Accuracy / confusion band |
| `lin-sep.png` | results | Linear separability plot |
| `logit-diff.png` | results | Logit difference plot |
| `sae_heatmap.png` | saes | Feature activation heatmap |
| `steering_feat30_success.png` | saes/results | Feature steering result |


---

## Things to Avoid

- Do not change margin, spacing, or column settings anywhere in `main.tex`.
- Do not include math or `\cite` in `sections/abstract.tex`.
- Do not add `\emph` in generated prose (the author decides when to use it); do not use em-dashes.
- Do not overwrite `notes/btk_sae_notes.md` — it contains the researcher's raw experiment notes and is informational only.
- Do not create new section files without adding the corresponding `\input` to `main.tex`.
