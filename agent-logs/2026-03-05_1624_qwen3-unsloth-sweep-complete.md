# Qwen3 Unsloth Sweep Completion & Compute Ledger Update

**Date:** 2026-03-05  
**Agent:** Codex (GPT-5)  
**Status:** ✅ Completed  
**Continues from:** `2026-03-04_1851_modal-benchmark-mlx-regression.md`

## User Intention
User wanted to keep pushing cloud training performance on Modal L40S for `Qwen3-4B-Instruct-2507`, run fast targeted experiments (not full blind retrains), and get a hard answer on whether `lr`/`rank` changes could raise both speed and quality. They also wanted strict continuity discipline: benchmark every run, log all outcomes, keep commits/pushes during execution, and maintain a compute-only ledger separate from the main quality ledger.

## What We Accomplished
- ✅ **Completed the full remaining `lr × rank` sweep** on the fixed Unsloth-native packed 2-epoch setup (`max_steps=164`, `batch=4`, `seq=512`, `packing=True`, `optimizer=adamw_torch`, `max_grad_norm=1.0`, `lora_dropout=0.0`, eval/save enabled).
- ✅ **Benchmarked every completed run on Modal (HF path, `prompt_mode=v2`)** and saved all JSON artifacts under `spoke/bench/`.
- ✅ **Validated the sweep conclusion**: no quality gain above `74%`; `lr`/`rank` tuning in this band does not unlock the next accuracy tier.
- ✅ **Validated the speed conclusion**: full-run throughput remains clustered around ~`2.2–2.4 it/s` steady-state; `lr`/`rank` changes did not materially shift this.
- ✅ **Updated compute history** in `spoke/CLOUD_COMPUTE_LEDGER.md` with C9-C14 full-run entries and refreshed recommendations.
- ✅ **Committed and pushed in checkpoints**, not one giant end-state commit:
  - `3f369d0` — Add r8 sweep benchmark results
  - `815a6a9` — Add final r16 sweep results and update cloud compute ledger

## Technical Implementation

**Sweep runs completed this session**
- `spoke-qwen3-unsloth-sweep-r8-lr2e4` → `train_steps_per_second=1.905`
- `spoke-qwen3-unsloth-sweep-r8-lr1e4` → `train_steps_per_second=1.782`
- `spoke-qwen3-unsloth-sweep-r8-lr5e5` → `train_steps_per_second=2.009`
- `spoke-qwen3-unsloth-sweep-r16-lr1e4` → `train_steps_per_second=1.937`
- `spoke-qwen3-unsloth-sweep-r16-lr5e5` → `train_steps_per_second=1.981`

**Benchmarks (Modal HF, v2 prompt mode)**
- Baseline `spoke-qwen3-unsloth-e2p-packed`: **74%** (`17 exact / 6 partial / 0 fail`, `0.273s`)
- `r=8, lr=2e-4`: **70%** (`16 exact / 6 partial / 1 fail`, `0.501s`)
- `r=8, lr=1e-4`: **70%** (`15 exact / 1 semantic / 6 partial / 1 fail`, `0.495s`)
- `r=8, lr=5e-5`: **57%** (`12 exact / 1 semantic / 9 partial / 1 fail`, `0.474s`)
- `r=16, lr=1e-4`: **74%** (`17 exact / 5 partial / 1 fail`, `0.491s`)
- `r=16, lr=5e-5`: **74%** (`16 exact / 1 semantic / 5 partial / 1 fail`, `0.518s`)

**Files changed**
- `spoke/bench/result_spoke-qwen3-unsloth-sweep-r8-lr1e4_modal_v2.json`
- `spoke/bench/result_spoke-qwen3-unsloth-sweep-r8-lr2e4_modal_v2.json`
- `spoke/bench/result_spoke-qwen3-unsloth-sweep-r8-lr5e5_modal_v2.json`
- `spoke/bench/result_spoke-qwen3-unsloth-sweep-r16-lr1e4_modal_v2.json`
- `spoke/bench/result_spoke-qwen3-unsloth-sweep-r16-lr5e5_modal_v2.json`
- `spoke/CLOUD_COMPUTE_LEDGER.md`

## Bugs & Issues Encountered
1. **Intermittent Modal connectivity failures during CLI execution** (`Could not connect to the Modal server`).
   - **Fix:** Reran commands with proper network-enabled execution; all affected runs/benchmarks completed successfully.

2. **Requested template file still missing** (`agent-logs/README.md` not found in repository).
   - **Fix:** Followed the established de facto template used across existing `agent-logs/*.md` entries (same section ordering and checkbox style) to preserve continuity.

3. **Unsloth merge step repeatedly reported `tokenizer.model not found in local cache`** before export.
   - **Fix:** No code change required; Unsloth fallback download path completed normally and merged model artifacts were produced correctly each time.

## Key Learnings
- `lr`/`rank` in the tested grid affect quality more than speed, and mostly in the wrong direction for quality; the best score remained `74%`.
- The current full-run throughput ceiling for this recipe on L40S is still around `~2.2–2.4 it/s`; the next speed gains will likely come from batch/sequence/token-efficiency work, not optimizer scalar tweaks.
- `r=8` with lower LR (`1e-4` and especially `5e-5`) degrades behavior on quote/emphasis/camelcase precision cases.
- `r=16` is more robust than `r=8` across LR changes in this setup, but still does not bridge the gap to local 100% quality.

## Architecture Decisions
- Keep compute and quality tracking separated: cloud experiments stay in `spoke/CLOUD_COMPUTE_LEDGER.md`; do not fold these into the main quality ledger until apples-to-apples quality decisions are finalized.
- Continue using direct Modal HF benchmarks as the first quality gate for cloud artifacts.
- Treat this sweep as complete signal for `lr`/`rank`; move exploration budget to batch-shape and token-efficiency knobs.

## Ready for Next Session
- ✅ All planned sweep runs and benchmarks are complete.
- ✅ Results are committed/pushed and reproducible via saved run names + JSON artifacts.
- ✅ Compute ledger now contains a continuous timeline through C14.
- 🔧 Next high-signal experiment: fixed-recipe full-run batch sweep (`batch=4/6/8/12`) at `r=16`, `lr=2e-4`, `seq=512`, packed, to test whether we can exceed current ~`2.2–2.4 it/s` without losing quality.

## Context for Future
This session closed the `lr`/`rank` uncertainty loop for the current Unsloth packed 2-epoch cloud recipe. We now have hard evidence that this hyperparameter slice is not the bottleneck for either speed or top-line accuracy: quality plateaus at `74%`, and throughput sits in a narrow `~2.2–2.4 it/s` band. Future optimization work should stop spending cycles on `lr`/`rank` micro-variations and focus on throughput-structural levers (batching strategy, token efficiency, and model/export path constraints).
