# AGENTS.md

This file provides guidance to LLM coding agents working in this repository.

Key:
- Use the project virtualenv for Python execution from the repository root:
  `../.venv/bin/python`, `../.venv/bin/pytest`, or activate `../.venv` first.
- Ensure any subagents also use the project virtualenv.
- Follow the paper formatting instructions and examples in
  `example_paper/example_paper.tex`.
- Do not change margin, spacing, or column settings anywhere in `main.tex`.
- Do not include math or citations in `sections/abstract.tex`.
- Do not add `\emph{...}` in generated prose.
- Do not use em dashes, use commas, parentheses, or split sentences.

## Repository Context

The parent repository studies mechanistic behavior in small attention-only
transformers on a list-copy task.

- Canonical sequence format in code is `[d1, d2, SEP, o1, o2]`.
- Token IDs are conventionally `MASK = n_digits` and `SEP = n_digits + 1`.
- The output slice is `[:, list_len + 1:]`.
- `src/models/utils.py::accuracy()` is per-token accuracy, so each output token
  contributes independently.

When editing the paper, keep manuscript claims aligned with the current local
experiment evidence. If task prose, notes, and code conventions disagree, check
the source files and local reports before broadening or rewriting claims.

## Companion Code Repository

The experiments described in this paper live in the parent repository:

| File | Role |
|------|------|
| `../src/models/transformer.py` | Model construction and custom attention mask |
| `../src/data/datasets.py` | Dataset generation via `get_dataset()` |
| `../src/utils/nb_utils.py` | `load_transformer_model()`, the default model loader |
| `../src/sae/steering.py` | Feature steering via `get_xovers_df` and `swap_outputs` |
| `../src/sae/activation_collection.py` | `identify_special_features()` |
| `../src/sae/loading.py` | SAE loading and checkpoint selection |
| `../src/sae/reporting.py` | Failure-reason classification and markdown reports |

Common baseline files from the parent repository:
- Base model: `../models/2layer_100dig_64d.pt`.
- Common SAE: `../sae_checkpoints/sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d.pt`.

## Core Architecture and Data Flow

- `../src/data/datasets.py::get_dataset()` builds all `n_digits^list_len`
  combinations and returns `(train_ds, val_ds)` with default `train_split=0.8`.
- `../src/models/transformer.py` defines custom attention masks
  (`build_attention_mask`, `attach_custom_mask`) implementing task-specific
  routing.
- `../src/utils/runtime.py::configure_runtime()` sets global `_RUNTIME` values
  used across model and utility code. Many helpers assert these are configured.
- `../src/utils/nb_utils.py::load_transformer_model()` configures runtime and
  returns `(model, model_cfg)`. Use it as the default loader in analysis code.

## Attention Mask Architecture

Two masks are built in `build_attention_mask()` and applied via hooks in
`attach_custom_mask()`:

- `mask_bias_l0` in layer 0: output tokens (`o1`, `o2`) can only self-attend;
  SEP reads input digits; outputs are further zeroed via the `_zero_o_rows`
  pattern hook.
- `mask_bias` in layers 1+: output tokens read from SEP and causally prior
  outputs; input tokens are blocked from reading outputs.

This enforces the SEP compression bottleneck:
`inputs -> SEP (layer 0) -> outputs (layers 1+)`.

## SAE Conventions

- SAE checkpoints in `../sae_checkpoints/` include `state_dict`, `cfg`, and
  `act_mean`.
- Always load and pass `act_mean` when collecting or patching activations.
- Use `load_sae(sae_path, d_model)` from `../src/utils/nb_utils.py`; it
  delegates to `../src/sae/loading.py` and handles legacy and new state dict
  formats.
- SAE classes are instantiated in
  `../src/sae/loading.py::instantiate_sae_from_cfg()`. Supported `sae_type`
  values are `btk`, `jumprelu`, and `matryoshka`.
- For feature steering and crossover work, use `../src/sae/steering.py`:
  `get_xovers_df`, `get_output_swap_bounds`, and `swap_outputs`.
- Special features are identified via `identify_special_features` in
  `../src/sae/activation_collection.py`: features whose activation correlates
  strongly with `alpha_d1 - alpha_d2`. Run `collect_attention_patterns` first
  to obtain `alpha_d1_all` and `alpha_d2_all`.
- For checkpoint selection, use
  `select_checkpoints(paths, use_best=False)` from `../src/sae/loading.py`.
  The default keeps only final checkpoints. Use `use_best=True` to prefer
  best-validation-loss variants when available.

## Dataset Evaluation Standard

- Transformer model accuracy should use the held-out validation split from
  `get_dataset()` with the default `train_split=0.8`.
- SAE evaluation and exhaustive activation scans should use the full input
  space via `ConcatDataset([train_ds, val_ds])`.
- Do not evaluate models with `train_split=1.0`; this mixes train data into
  evaluation and inflates reported accuracy.
- Do not silently swap the model-vs-SAE dataset conventions.

## Canonical Workflows

Run these from the parent repository unless noted otherwise:

- Environment: `uv sync`, then `source .venv/bin/activate`.
- Train SAE:
  `.venv/bin/python scripts/train_sae.py --sae_type btk --d_sae ... --top_k ... --n_steps ...`.
- Run crossover pipeline:
  `.venv/bin/python scripts/run_crossover_analysis.py [--feature auto] [--threshold 0.5] [--max-features 2] [--report]`.
- SAE sweep comparison:
  `.venv/bin/python scripts/compare_sae.py [--best]`.
- SAE sweep plotting:
  `.venv/bin/python scripts/plot_sae_sweep.py --report <results/compare_sae/sae_comparison_*.md>`.
- Run tests: `.venv/bin/pytest tests/`.

There is no currently tracked `scripts/train_model.py` entry point. Shared
training logic lives in `../src/models/train.py`, so inspect or restore the
intended wrapper before documenting or running model-training commands.

When running experiments, append a concise entry to the experiment log with the
command, output paths, and headline metrics. The tracked historical log is
`../archive/EXPERIMENTS.md`; create or use a root `../EXPERIMENTS.md` only if
the project owner restores that convention.

## Paper Build System

From this `paper/` directory:

```bash
make          # Build PDF -> build/main.pdf
make clean    # Remove build artifacts
```

From the parent repository:

```bash
make -C paper
make -C paper clean
```

- Uses `latexmk` under the hood for automated compilation.
- Always run the build after editing any `.tex` file.
- If `latexmk` says "Nothing to do" after source edits, force a rebuild with
  `latexmk -g`.
- Output is `build/main.pdf`.

### Build Troubleshooting

If compilation fails at the `biber` step with an error resembling
`Unicode::UCD: failed to find unicore/version`, this is usually a corrupted
local Biber PAR cache, not a LaTeX source error.

```bash
rm -rf "$(biber --cache)"
make
```

The first run after a Biber failure can still report a stale `latexmk` summary.
Run `make` once more. Also ensure `ref.bib` has unique citation keys, because
duplicate keys can cause Biber failures.

## Paper Writing Conventions

- Keep the abstract as a single paragraph.
- Keep broad framing out of the abstract when evidence is limited to the toy
  setting. Move broader implications to Discussion.
- Prefer shorter, clearer sentences when prose gets comma-heavy.
- When editing the introduction, explain the purpose of the method section and
  why the reader should care, not only the section roadmap.
- If the author has revised wording, preserve that wording as the baseline.
- Avoid calling selected evaluation checkpoints "cherry-picked"; prefer neutral
  phrasing such as "selected high-performing checkpoints chosen to span
  architecture, dictionary size, and sparsity."
- When steering results have many `No Activation` cases, report conditional
  success rather than only global success.

## Label Conventions

| Prefix | Usage | Example |
|--------|-------|---------|
| `s:` | Sections and appendices | `\label{s:method}` |
| `ss:` | Subsections | `\label{ss:results-circuits}` |
| `tab:` | Tables | `\label{tab:mask}` |
| `fig:` | Figures | `\label{fig:attn-flow}` |
| `eq:` | Equations | `\label{eq:o1_logit_composition}` |

- Use `\eqref{}` for equations.
- Use `\ref{}` for everything else.
- Citations should use `\citep{}`, `\citet{}`, or `\textcite{}`.

## Key Notation

| Symbol | Meaning |
|--------|---------|
| `$d_1, d_2$` | Input digit tokens, values 0 to 99 |
| `$s$` | SEP token |
| `$o_1, o_2$` | Output position tokens |
| `$\boldsymbol{r}_z^{L_i}$` | Residual stream at position `z` after layer `i` |
| `$\alpha_{x \to y}$` | Attention probability from token `x` to token `y` in layer 1 |
| `$\beta_{x \to y}$` | Attention probability from token `x` to token `y` in layer 2 |
| `$\alpha_\text{diff}$` | `$\alpha_{s \to d_1} - \alpha_{s \to d_2}$` |
| F30 | The special SAE feature strongly correlated with `$\alpha_\text{diff}$` |

## Research Context

The paper's current research context concerns graded latent activations in SAEs.
When resolving ambiguity about numerical results, check local notes and reports
first, then the parent repository results.

- The model is a 2-layer attention-only transformer with no MLP, no LayerNorm,
  and no bias.
- Value and output matrices are frozen to identity to isolate the composition
  mechanism.
- The model was trained on 80 percent of the 10,000 possible input pairs.
- BatchTopK SAEs were trained across a grid of `k` and `d_sae` values.
- Feature heatmaps are 100 by 100 grids where cell `(i, j)` is feature
  activation magnitude for input `(d_1=i, d_2=j)`.
- Most SAE features show cross patterns in their heatmaps and act like digit
  detectors.
- Special features, one or two per SAE, are identifiable by correlation with
  `\alpha_\text{diff}` and have qualitatively different heatmaps.
- F30 in the k=3, d=100 SAE correlates strongly with output ordering; scaling
  it up or down can swap predicted output order.
- Feature steering with F30 succeeds on a subset of inputs; many failures are
  `No Activation`, so conditional success should be reported where relevant.

## Figures in Use

All figures live in `figures/`. Do not rename files without updating
`\includegraphics` references.

| File | Description |
|------|-------------|
| `attn-flow.png` | Attention flow diagram |
| `attn-hist.png` | Attention weight histogram |
| `confusion-band.png` | Accuracy or confusion band |
| `lin-sep.png` | Linear separability plot |
| `logit-diff.png` | Logit difference plot |
| `sae_heatmap.png` | Feature activation heatmap |
| `steering_feat30_success.png` | Feature steering result |

## Things to Avoid

- Do not overwrite raw researcher notes if a `notes/` directory is restored.
- Do not create new section files without adding the corresponding `\input` to
  `main.tex`.
- Do not assume one saved-model filename format; existing naming appears in
  both `2layer_100dig_64d.pt` and timestamped forms.
- Do not include private dependency sources in blind-review materials.
