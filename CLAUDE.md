# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Key - don't forget to use the .venv for python execution. Also ensure all subagents also use rtk

## Scope and Task Shape
- This repo studies mechanistic behavior in small attention-only transformers on a list-copy task.
- Canonical sequence format is `[d1, d2, SEP, o1, o2]` where outputs must copy inputs (not sort).
- Token IDs are conventionally `MASK = n_digits` and `SEP = n_digits + 1`; output slice is `[:, list_len + 1:]`.

## Core Architecture and Data Flow
- `src/data/datasets.py::get_dataset()` builds all `n_digits^list_len` combinations and returns `(train_ds, val_ds)` with default `train_split=0.8`.
- `src/models/transformer.py` defines custom attention masks (`build_attention_mask`, `attach_custom_mask`) implementing task-specific routing.
- `src/utils/runtime.py::configure_runtime()` sets global `_RUNTIME` values used across model/util code; many helpers assert these are configured.
- `src/utils/nb_utils.py::load_transformer_model()` configures runtime and returns `(model, model_cfg)`; use this as the default loader in analysis code.
- `src/models/utils.py::accuracy()` is per-token accuracy (each output token contributes independently).

## SAE Conventions
- SAE checkpoints in `results/sae_models/` include `state_dict`, `cfg`, and `act_mean`.
- Always load and pass `act_mean` when collecting/patching activations (see `scripts/run_crossover_analysis.py`).
- For feature steering/crossover work, main entry points are in `src/sae/steering.py`: `get_xovers_df`, `get_output_swap_bounds`, `swap_outputs`.

## Canonical Workflows
- Environment: `uv sync` then `source .venv/bin/activate`.
- Train model: `python3 scripts/train_model.py ...` (supports retries until `--min-acc`; saves to `models/`).
- Train SAE: `python3 scripts/train_sae.py --d_sae ... --top_k ... --n_steps ...`.
- Run crossover pipeline: `python3 scripts/run_crossover_analysis.py` (writes CSVs to `results/xover/`).
- Cluster/GPU workflow is captured in `submit_job.sh` (sync env, activate `.venv`, run analysis scripts).

## Project-Specific Patterns
- Prefer imports from `src.utils.nb_utils` and `src.sae` in notebooks/scripts to stay consistent with existing analysis flow.
- Default analyses use full data via `ConcatDataset([train_ds, val_ds])` when exhaustively scanning input space.
- Do not evaluate with `train_split=1.0`; this mixes train data into evaluation and inflates reported accuracy.
- Existing saved-model naming appears in two styles (`2layer_100dig_64d.pt` and timestamped `L*_H*_D*_V*..._acc*.pt`); do not assume one format only.

## Current Baselines and Files
- Common base model: `models/2layer_100dig_64d.pt`.
- Common SAE for feature-30 analysis: `results/sae_models/sae_d100_k3_lr0.0003_seed44_2layer_100dig_64d.pt`.
- Key reference files: `src/data/datasets.py`, `src/models/transformer.py`, `src/models/utils.py`, `src/utils/nb_utils.py`, `src/sae/steering.py`.

## Reproducibility Requirement
- When running experiments, append a concise entry to `EXPERIMENTS.md` with command, output paths, and headline metrics.

## Attention Mask Architecture
Two masks are built in `build_attention_mask()` and applied via hooks in `attach_custom_mask()`:
- **`mask_bias_l0`** (layer 0): output tokens (`o1`, `o2`) can only self-attend; SEP reads input digits; outputs are further zeroed via `_zero_o_rows` pattern hook.
- **`mask_bias`** (layers 1+): output tokens read from SEP and causally prior outputs; input tokens are blocked from reading outputs.

This enforces the SEP compression bottleneck: information must flow `inputs → SEP (layer 0) → outputs (layers 1+)`.

## SAE Loading Details
- SAE class: `dictionary_learning.trainers.batch_top_k.BatchTopKSAE(activation_dim, dict_size, k)`.
- Use `load_sae(sae_name, d_model)` from `src/utils/nb_utils.py` — handles both legacy (`W_enc`/`b_enc`/`W_dec`/`b_dec`) and new state dict formats automatically.
- Checkpoints contain `state_dict`, `cfg`, and `act_mean`; always retrieve and pass `act_mean` when patching activations.

## Model Config Inference
`src/models/utils.py::infer_model_config(path)` can auto-infer `d_model`, `n_layers`, `n_heads`, `list_len`, `use_wv`, `use_wo`, etc. directly from a checkpoint — useful when loading models whose names don't encode all parameters.

## Development Environment
No formal test suite exists. To verify changes, run a training smoke-test or the crossover pipeline on the baseline model.

<!-- rtk-instructions v2 -->
# RTK (Rust Token Killer) - Token-Optimized Commands

## Golden Rule

**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.

**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# ❌ Wrong
git add . && git commit -m "msg" && git push

# ✅ Correct
rtk git add . && rtk git commit -m "msg" && rtk git push
```

## RTK Commands by Workflow

### Build & Compile (80-90% savings)
```bash
rtk cargo build         # Cargo build output
rtk cargo check         # Cargo check output
rtk cargo clippy        # Clippy warnings grouped by file (80%)
rtk tsc                 # TypeScript errors grouped by file/code (83%)
rtk lint                # ESLint/Biome violations grouped (84%)
rtk prettier --check    # Files needing format only (70%)
rtk next build          # Next.js build with route metrics (87%)
```

### Test (90-99% savings)
```bash
rtk cargo test          # Cargo test failures only (90%)
rtk vitest run          # Vitest failures only (99.5%)
rtk playwright test     # Playwright failures only (94%)
rtk test <cmd>          # Generic test wrapper - failures only
```

### Git (59-80% savings)
```bash
rtk git status          # Compact status
rtk git log             # Compact log (works with all git flags)
rtk git diff            # Compact diff (80%)
rtk git show            # Compact show (80%)
rtk git add             # Ultra-compact confirmations (59%)
rtk git commit          # Ultra-compact confirmations (59%)
rtk git push            # Ultra-compact confirmations
rtk git pull            # Ultra-compact confirmations
rtk git branch          # Compact branch list
rtk git fetch           # Compact fetch
rtk git stash           # Compact stash
rtk git worktree        # Compact worktree
```

Note: Git passthrough works for ALL subcommands, even those not explicitly listed.

### GitHub (26-87% savings)
```bash
rtk gh pr view <num>    # Compact PR view (87%)
rtk gh pr checks        # Compact PR checks (79%)
rtk gh run list         # Compact workflow runs (82%)
rtk gh issue list       # Compact issue list (80%)
rtk gh api              # Compact API responses (26%)
```

### JavaScript/TypeScript Tooling (70-90% savings)
```bash
rtk pnpm list           # Compact dependency tree (70%)
rtk pnpm outdated       # Compact outdated packages (80%)
rtk pnpm install        # Compact install output (90%)
rtk npm run <script>    # Compact npm script output
rtk npx <cmd>           # Compact npx command output
rtk prisma              # Prisma without ASCII art (88%)
```

### Files & Search (60-75% savings)
```bash
rtk ls <path>           # Tree format, compact (65%)
rtk read <file>         # Code reading with filtering (60%)
rtk grep <pattern>      # Search grouped by file (75%)
rtk find <pattern>      # Find grouped by directory (70%)
```

### Analysis & Debug (70-90% savings)
```bash
rtk err <cmd>           # Filter errors only from any command
rtk log <file>          # Deduplicated logs with counts
rtk json <file>         # JSON structure without values
rtk deps                # Dependency overview
rtk env                 # Environment variables compact
rtk summary <cmd>       # Smart summary of command output
rtk diff                # Ultra-compact diffs
```

### Infrastructure (85% savings)
```bash
rtk docker ps           # Compact container list
rtk docker images       # Compact image list
rtk docker logs <c>     # Deduplicated logs
rtk kubectl get         # Compact resource list
rtk kubectl logs        # Deduplicated pod logs
```

### Network (65-70% savings)
```bash
rtk curl <url>          # Compact HTTP responses (70%)
rtk wget <url>          # Compact download output (65%)
```

### Meta Commands
```bash
rtk gain                # View token savings statistics
rtk gain --history      # View command history with savings
rtk discover            # Analyze Claude Code sessions for missed RTK usage
rtk proxy <cmd>         # Run command without filtering (for debugging)
rtk init                # Add RTK instructions to CLAUDE.md
rtk init --global       # Add RTK to ~/.claude/CLAUDE.md
```

## Token Savings Overview

| Category | Commands | Typical Savings |
|----------|----------|-----------------|
| Tests | vitest, playwright, cargo test | 90-99% |
| Build | next, tsc, lint, prettier | 70-87% |
| Git | status, log, diff, add, commit | 59-80% |
| GitHub | gh pr, gh run, gh issue | 26-87% |
| Package Managers | pnpm, npm, npx | 70-90% |
| Files | ls, read, grep, find | 60-75% |
| Infrastructure | docker, kubectl | 85% |
| Network | curl, wget | 65-70% |

Overall average: **60-90% token reduction** on common development operations.
<!-- /rtk-instructions -->