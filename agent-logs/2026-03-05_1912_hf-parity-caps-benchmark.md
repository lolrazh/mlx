# HF Parity Run Completion & CAPS Benchmark Alignment

**Date:** 2026-03-05  
**Agent:** Codex (GPT-5)  
**Status:** ✅ Completed  
**Continues from:** `2026-03-05_1624_qwen3-unsloth-sweep-complete.md`

## User Intention
User wanted a strict, practical parity checkpoint on cloud for `Qwen3-4B-Instruct-2507`: update the benchmark expectation format to CAPS (not markdown emphasis), finish the true HF parity training run, benchmark it in the same Modal path used for other cloud artifacts, and preserve an auditable checkpoint with commit/push continuity.

## What We Accomplished
- ✅ **Updated benchmark expectations to CAPS in both benchmark sets** for the two emphasis targets:
  - `id=14`: `It's SURPRISINGLY fast.`
  - `id=21`: `This is absolutely CRITICAL for the launch.`
- ✅ **Completed the full HF parity training run on Modal** (`spoke-qwen3-hf-parity-v1`) with merged bf16 export.
- ✅ **Ran Modal benchmark on the completed HF parity artifact** using `prompt_mode=v2`.
- ✅ **Recorded a materially improved cloud quality checkpoint**: **96%** (`22 exact / 1 partial / 0 fail`, avg latency `0.24s`).
- ✅ **Committed and pushed all related changes** in `ccfe8e7` to `main`.

## Technical Implementation

**Training run**
- Script: `spoke/cloud/train_hf.py`
- Run name: `spoke-qwen3-hf-parity-v1`
- Modal app: `ap-zu4xqMaImjOhSzAOg3N1ge`
- W&B run: `yoj2nu2c` (`spoke-qwen3-hf-parity-v1`)
- Final train metrics:
  - `train_runtime`: `601.3249s`
  - `train_steps_per_second`: `3.326`
  - `train_samples_per_second`: `13.304`
  - `epoch`: `6.67`

**Benchmark run**
- Command path: `spoke/cloud/benchmark.py --run-name spoke-qwen3-hf-parity-v1 --prompt-mode v2`
- Modal app: `ap-xce4vV4ViSt5ya1IIhp9Le`
- Result artifact: `spoke/bench/result_spoke-qwen3-hf-parity-v1_modal_v2.json`
- Score: `96%` (`22 exact / 0 semantic / 1 partial / 0 fail`)
- Avg latency: `0.241s`
- Single miss (`partial`):
  - `id=21` expected `This is absolutely CRITICAL for the launch.`
  - got `This is CRITICAL for the launch.`

**Files changed in commit `ccfe8e7`**
- `spoke/bench/test_set.json`
- `spoke/bench/test_set_v3.json`
- `spoke/cloud/train.py`
- `spoke/cloud/train_hf.py`
- `spoke/bench/result_spoke-qwen3-hf-parity-v1_modal_v2.json`

## Bugs & Issues Encountered
1. **Benchmark expectation mismatch for emphasis style** (markdown bold vs CAPS) risked invalidating parity comparisons.
   - **Fix:** Updated both benchmark sets (`test_set.json`, `test_set_v3.json`) to enforce CAPS outputs for the two emphasis targets.

2. **Potential parity mask mismatch in cloud trainer** for short rows in batch collation.
   - **Fix:** Added one extra padded target-token label in `spoke/cloud/train.py` (`batch_labels[row, length] = 0` when row length is below batch max) to mirror MLX mask semantics.

3. **Template reference file still missing** (`agent-logs/README.md` not found at requested path).
   - **Fix:** Followed the established de facto agent-log structure already used in existing `agent-logs/*.md` files.

## Key Learnings
- The pure HF+PEFT cloud path now closes most of the historical gap: this checkpoint reached `96%`, far above prior Unsloth sweep plateau scores (`~74%`).
- Remaining failure mode is precision retention in phrasing (`absolutely` dropped), not catastrophic instruction-following failure.
- Aligning benchmark target formatting (CAPS vs markdown emphasis) is necessary before making any parity claims between training stacks.

## Architecture Decisions
- Keep a separate HF parity trainer (`spoke/cloud/train_hf.py`) to avoid Unsloth-specific training/export behavior while debugging quality parity.
- Keep benchmark scoring on the same Modal HF path (`spoke/cloud/benchmark.py`, `prompt_mode=v2`) for consistent cloud-to-cloud comparability.
- Preserve incremental commits/pushes as checkpoints instead of batching multiple experiment stages into a single opaque change.

## Ready for Next Session
- ✅ HF parity baseline is now reproducible (`train_hf.py` + run name + benchmark artifact).
- ✅ CAPS benchmark target normalization is in place across both test sets.
- ✅ Commit/push checkpoint is complete (`ccfe8e7` on `main`).
- 🔧 Next highest-signal step: run side-by-side output diff against the local 100% MLX model for the remaining miss classes (especially emphasis + phrase retention).

## Context for Future
This session established a stronger cloud reference point: with the HF parity path and corrected benchmark targets, cloud reached `96%` on the same evaluation style instead of the earlier low-70s Unsloth sweep plateau. The remaining gap to `100%` is now narrow and specific, which makes targeted ablations (prompt framing, loss masking details, and exact-generation defaults) more valuable than broad hyperparameter sweeps.
