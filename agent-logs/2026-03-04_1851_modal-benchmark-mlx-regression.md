# Modal Benchmark Confirmed MLX Regression

**Date:** 2026-03-04
**Agent:** Codex (GPT-5)
**Status:** ✅ Completed
**Continues from:** `2026-03-04_0417_qwen35-modal-fastpath-repair.md`

## User Intention
User wanted to understand why the strict cloud Qwen3 parity run was still showing only 35% after conversion, verify whether the failure was happening in cloud training or later in the pipeline, preserve the results in the project ledger, and maintain continuity with a detailed agent log for future debugging.

## What We Accomplished
- ✅ **Benchmarked the merged cloud model directly on Modal before MLX conversion** — Added a dedicated Modal-side inference benchmark at `spoke/cloud/benchmark.py`.
- ✅ **Verified the cloud model quality in native HF/Transformers** — The strict parity cloud model scored **87%** (`20 exact / 3 partial / 0 fail`) on Modal at `0.28s` average latency.
- ✅ **Identified the real failure boundary** — The same model still scores **35%** after MLX conversion (`5 exact / 3 semantic / 11 partial / 4 fail`, `2.38s`), so the major regression is in the MLX conversion and/or MLX inference path, not primarily in cloud training.
- ✅ **Fixed a false-negative bug in the new benchmark** — The first Modal benchmark pass incorrectly scored obviously-correct outputs as partial because it decoded with `skip_special_tokens=False`, leaving trailing `<|im_end|>` tokens in the text.
- ✅ **Trimmed the benchmark container to avoid pointless build tax** — Replaced the overkill `unsloth` image path with a lean PyTorch runtime for inference-only benchmarking.
- ✅ **Updated the main training ledger** — Rewrote the cloud entries in `spoke/LEDGER.md` to reflect the original confounded run, the strict parity rerun, and the new pre-MLX Modal benchmark result.
- ✅ **Committed the work** — Saved the benchmark + ledger updates in commit `7eb7dc8` (`Benchmark cloud Qwen3 model on Modal`).

## Technical Implementation

**Primary files changed:**
- `spoke/cloud/benchmark.py`
- `spoke/LEDGER.md`
- `spoke/bench/result_qwen3-t2-cloud-mlx_v2.json`
- `spoke/bench/result_spoke-qwen3-t2-cloud_modal_v2.json`

**New benchmark path (`spoke/cloud/benchmark.py`):**
- Runs on Modal against `/output/<run_name>/merged`
- Loads the merged bf16 model with `AutoTokenizer` + `AutoModelForCausalLM`
- Uses the same v2 system prompt, scoring rubric, and category-by-category reporting style as the local benchmark
- Generates greedily on GPU (`L40S`) and returns a local JSON artifact under `spoke/bench/`

**Container optimization:**
- Initial attempt used a CUDA base + `pip install unsloth`
- This pulled in a huge dependency tree and wasted time building an inference image
- Final image uses:
```bash
pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
```
- Then installs only:
```bash
transformers==4.51.3
accelerate==1.2.1
sentencepiece
safetensors
```

**Modal benchmark results (strict cloud parity model, pre-MLX):**
- Run name: `spoke-qwen3-t2-cloud`
- Modal app: `ap-MVfqry6XmAfK2PXcQPnzMA`
- Result artifact: `spoke/bench/result_spoke-qwen3-t2-cloud_modal_v2.json`
- Accuracy: `87%`
- Breakdown: `20 exact / 0 semantic / 3 partial / 0 fail`
- Avg latency: `0.28s`

**MLX benchmark comparison (same model, post-conversion):**
- Result artifact: `spoke/bench/result_qwen3-t2-cloud-mlx_v2.json`
- Accuracy: `35%`
- Breakdown: `5 exact / 3 semantic / 11 partial / 4 fail`
- Avg latency: `2.38s`

**Three remaining misses on Modal HF (all partial, not catastrophic):**
1. `Celero -> Silerio` instead of `Silero`
2. `Kibbeh Nayeh` not replaced with `Kibbinay`
3. Misplaced quote scope in the `"Gluten-free"` example

## Bugs & Issues Encountered

1. **The first Modal benchmark path was bloated and slow** — The initial image installed `unsloth`, which triggered a large dependency install and delayed feedback.
   - **Fix:** Switched to a lean inference-only PyTorch runtime.

2. **The first benchmark result was falsely reported as 0%** — Correct outputs were carrying literal `<|im_end|>` tokens because the script decoded with `skip_special_tokens=False`.
   - **Fix:** Changed decoding to `skip_special_tokens=True`, then reran the benchmark.

3. **The old ledger conclusion was no longer accurate** — The project notes still blamed the 35% cloud result primarily on packing, response-only masking, and dropout mismatches.
   - **Fix:** Updated `spoke/LEDGER.md` to separate the original confounded fast-path run, the strict parity rerun, and the new direct Modal HF benchmark that proves the large regression is downstream in MLX.

4. **`agent-logs/README.md` is still missing** — The user requested the template in that file, but the file does not exist in this repo.
   - **Fix:** Followed the established structure used by the existing `agent-logs/*.md` files for continuity.

## Key Learnings
- **The cloud trainer is not the main source of the 35% collapse.** The strict parity cloud model is materially better before conversion than after conversion.
- **The MLX path is now the primary suspect.** The quality drop from `87%` to `35%` happens after the merged bf16 model leaves the HF/Transformers path.
- **The cloud parity model is close, but not identical, to the local 100% MLX run.** The `87%` score is only three partial misses, all on precision-sensitive edge cases.
- **Packing was a real confound, but it was not the full explanation.** Disabling packing and restoring local-like training behavior did not close the gap to 100%.
- **A small benchmark bug can completely distort conclusions.** Special-token leakage turned an initially useful benchmark into a fake `0%` failure until the decode path was corrected.

## Architecture Decisions
- **Benchmark before converting again** — Instead of guessing whether training or conversion was bad, we inserted a direct HF/Transformers benchmark at the Modal artifact boundary.
- **Keep the benchmark lightweight** — Inference-only containers should not inherit the heavy Unsloth training dependency tree.
- **Preserve both cloud runs in the ledger** — The original fast-path run remains documented as a confounded experiment, while the strict parity rerun carries the stronger root-cause signal.

## Ready for Next Session
- ✅ **A reusable Modal-side benchmark exists** — `spoke/cloud/benchmark.py` can now score any merged cloud model before MLX conversion.
- ✅ **The ledger now reflects the real failure boundary** — `spoke/LEDGER.md` records that pre-MLX quality is `87%` and post-MLX quality is `35%`.
- ✅ **The benchmark evidence is committed** — Commit `7eb7dc8` contains the new benchmark tooling and ledger update.
- 🔧 **Next likely step:** isolate exactly what `mlx_lm.convert` or MLX generation is changing (config metadata, tokenizer behavior, or decoding semantics).
- 🔧 **Secondary follow-up:** compare the HF merged model and the MLX-converted model prompt-by-prompt to pinpoint where outputs diverge.

## Supplementary Insights (from Claude session)

**Agent:** Claude (Opus 4.6)
**Context:** Parallel sessions debugging the same cloud pipeline across multiple runs.

### 1. Unsloth `save_pretrained_merged` config mangling — probable cause of 87% → 35%

The merged bf16 `config.json` exported by Unsloth differs from the original HuggingFace Qwen3-4B config in several ways that can break `mlx_lm.convert` or MLX inference:

- **`rope_theta` nested inside `rope_parameters`** instead of top-level. mlx-lm expects `rope_theta` at the top level — without it, `ModelArgs.__init__()` throws `missing required positional argument: 'rope_theta'`. We patched this manually but other consumers of the config may still read the wrong location.
- **`bos_token_id` stripped** (set to `null` instead of original value).
- **`eos_token_id` changed from list to int** (e.g., `151645` instead of `[151645]`).
- **`use_cache: false`** added (original doesn't set this).
- **Extra fields added**: `unsloth_fixed: true`, `unsloth_version: "2026.3.3"`.

These config differences are the highest-signal suspect for the conversion regression. A proper fix would restore the original HF config fields before running `mlx_lm.convert`, rather than patching individual fields.

### 2. `modal volume get` can silently corrupt safetensors

During our first download attempt, `modal volume get` errored on `.cache already exists` but left safetensor files that appeared complete by file size. Loading them with `safetensors.safe_open()` revealed `SafetensorError: incomplete metadata, file not fully covered`. The model produced `!!!!!!!!` on every output until we re-downloaded with `--force`.

**Lesson:** Always verify safetensors with `safe_open()` after Modal downloads. Use `--force` flag by default.

### 3. Qwen3.5-4B is a dead end on MLX

Before the Qwen3-4B parity experiments, we ran a full Qwen3.5-4B cloud training (spoke-qwen35-t2, 2000 steps). The merged model:
- Required mlx-lm upgrade from 0.30.5 → 0.30.7 (PR #869 added qwen3_5 support)
- Had an empty `tokenizer_config.json` (0 bytes) — Unsloth bug, fixed by copying from HuggingFace
- **Benchmarked at 0% accuracy** — model generates incoherent garbage

The hybrid DeltaNet+attention architecture (24 linear_attention + 8 full_attention layers) is fundamentally broken in MLX inference despite conversion completing without errors. Abandoned.

### 4. Practical conclusion: cloud not worth it for this model size

Qwen3-4B with 1201 examples trains in ~3.5 hours on M4 24GB and produces 100% accuracy locally. The cloud pipeline adds Modal setup, Unsloth quirks, config patching, download corruption risk, conversion debugging — and still lands at 35% (or 87% pre-conversion vs 100% local). Cloud GPU training pays off for 7B+ models or massive datasets, not for 4B with 1K examples.

### 5. Actionable next debug step (if revisiting cloud path)

Diff the full Unsloth-exported `config.json` against the original `Qwen/Qwen3-4B-Instruct` config from HuggingFace. Restore ALL original fields before `mlx_lm.convert`. The rope_theta fix alone recovered the model from crashing to producing output, but other mangled fields may cause the remaining 52-point quality regression.

## Context for Future
This session changed the working diagnosis. The team originally suspected that cloud training quality collapsed because the first cloud run was not apples-to-apples with the local MLX training recipe. That was only part of the story. After the strict parity rerun still produced a 35% MLX benchmark, the new direct Modal HF benchmark showed that the merged bf16 cloud model is actually much healthier at 87%. The largest remaining gap is therefore not "Modal + Unsloth is bad" but "the MLX export/inference path is degrading the model." Future debugging should treat conversion and inference parity as the main frontier, not cloud training hyperparameters.

**Final team consensus:** Local MLX fine-tuning remains the proven path. Cloud is shelved unless model size or data scale demands it.
