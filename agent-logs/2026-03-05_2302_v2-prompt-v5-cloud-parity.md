# V2 Prompt Override on V5 Data Restored 100% Cloud Benchmark

**Date:** 2026-03-05  
**Agent:** Codex (GPT-5)  
**Status:** ✅ Completed  
**Continues from:** `2026-03-05_1912_hf-parity-caps-benchmark.md`

## User Intention
User wanted practical, cost-aware quality recovery on cloud finetuning: verify whether prompt framing (v2 vs v3) was causing the quality gap, keep `seq_length=512`, run a full cloud train with the same v5 dataset, benchmark immediately, and keep commit/push continuity while selecting the checkpoint that actually wins benchmark quality (not just eval loss).

## What We Accomplished
- ✅ **Validated the `v2`/`v3` confusion source:** these are benchmark/training prompt modes, not different test sets.
- ✅ **Confirmed data reality before another spend-heavy run:** `spoke-training-data:/train.jsonl` has `1287` rows and currently contains v5-style expanded system prompt text.
- ✅ **Checked sequence-length tradeoff on real v5 token lengths** (Qwen tokenizer):
  - `>256`: `116/1287` (`9.0%`) would truncate
  - `>512`: `0/1287`
  - Decision kept as requested: **`max_seq_length=512`**.
- ✅ **Added training-time system prompt override support** in `spoke/cloud/train_hf.py` (`system_prompt_mode=as_is|v2|v3`) so prompt ablations do not require rebuilding dataset files.
- ✅ **Ran full Modal HF+PEFT training with forced v2 prompt** on v5 data:
  - Run: `spoke-qwen3-hf-v5-v2prompt-v1-20260305-2247`
  - W&B: `9vhjvms9`
  - Trainer-selected best checkpoint by eval loss: `checkpoint-600`
- ✅ **Benchmarked merged best-checkpoint artifact (v2 prompt mode):** **87%** (`19 exact / 1 semantic / 3 partial / 0 fail`).
- ✅ **Merged and benchmarked `checkpoint-1200` from the same run:** **100%** (`23 exact / 0 partial / 0 fail`) on the same 23-example v3 test set with `prompt_mode=v2`.
- ✅ **Committed and pushed code/results** for this run (`5f62e70` on `main`).

## Technical Implementation

**Code changes**
- File: `spoke/cloud/train_hf.py`
- Added:
  - `V2_SYSTEM_PROMPT` constant
  - `V3_SYSTEM_PROMPT` constant
  - New arg `system_prompt_mode: str = "as_is"` (train + local entrypoint)
  - Validation + override path that rewrites only the first system message in each chat example at runtime.
- Output logging now prints active system prompt mode for reproducibility.

**Training run (forced v2 prompt, v5 data)**
- Script: `spoke/cloud/train_hf.py`
- Command profile:
  - model `Qwen/Qwen3-4B-Instruct-2507`
  - `max_steps=1200`, `lr=1e-5`, `batch=4`, `grad_accum=1`
  - `rank=8`, `alpha=16`, `lora_dropout=0.05`
  - `max_seq_length=512`, `optimizer=adam`, `max_grad_norm=1.0`
  - `eval_steps=50`, `save_steps=100`, `system_prompt_mode=v2`
- Modal app: `ap-KuG8qEca8g781dXQInNECu`
- W&B run: `https://wandb.ai/spoke/spoke/runs/9vhjvms9`
- Final train metrics:
  - `train_runtime`: `488.5567s`
  - `train_steps_per_second`: `2.456`
  - `train_loss`: `0.2302`
  - `epoch`: `3.74`
- Trainer-selected checkpoint: `/output/spoke-qwen3-hf-v5-v2prompt-v1-20260305-2247/checkpoint-600`

**Benchmark runs**
1. Merged artifact from trainer-selected checkpoint (`checkpoint-600`)
   - Command: `spoke/cloud/benchmark.py --run-name spoke-qwen3-hf-v5-v2prompt-v1-20260305-2247 --prompt-mode v2`
   - Modal app: `ap-ngMhCnrlvBeIH02JIGWnFg`
   - Result: `87%` (`19 exact / 1 semantic / 3 partial / 0 fail`)
   - Output file: `spoke/bench/result_spoke-qwen3-hf-v5-v2prompt-v1-20260305-2247_modal_v2.json`

2. Manual merge of `checkpoint-1200` + benchmark
   - Merge script: `spoke/cloud/merge_adapter_checkpoint.py`
   - Merge app: `ap-wifeL8nYBmvGZqppShB0zw`
   - Benchmark app: `ap-ERu3UhEkZg1emKvf6AH205`
   - Result: **`100%`** (`23 exact / 0 semantic / 0 partial / 0 fail`)
   - Output file: `spoke/bench/result_spoke-qwen3-hf-v5-v2prompt-v1-20260305-2247-ckpt1200_modal_v2.json`

**Related continuity commits in this segment**
- `9e810ac` — Added Modal benchmark result snapshots
- `b8f7226` — Added v2 benchmark for latest HF v5 v3prompt run
- `e83d552` — Added adapter checkpoint merge utility + ckpt1200 benchmark results
- `5f62e70` — Added system-prompt override + v2-prompt v5 benchmark results

## Bugs & Issues Encountered
1. **Prompt-mode naming confusion (`v2` vs `v3`)**
   - `v2`/`v3` were interpreted as possible test-set versions.
   - **Fix:** Explicitly verified and documented: they are prompt templates; test set stayed `test_set_v3.json`.

2. **Trainer “best checkpoint by eval_loss” did not match benchmark winner**
   - `checkpoint-600` was selected automatically but scored `87%`, while `checkpoint-1200` scored `100%`.
   - **Fix:** Added/used checkpoint merge utility and benchmarked downstream checkpoint directly.

3. **Requested template file still absent**
   - `agent-logs/README.md` was not present at the requested location.
   - **Fix:** Followed the established structure used by existing recent logs in `agent-logs/`.

## Key Learnings
- Prompt framing materially impacts this stack: with v5 data, forcing training prompt to **v2** recovered full benchmark quality when using the right checkpoint.
- This pipeline now has repeated evidence that **eval loss alone is not a reliable checkpoint selector** for final task accuracy.
- Keeping `seq_length=512` remains the safe choice for v5 (`0%` truncation); dropping to `256` would cut `9%` of training examples.

## Architecture Decisions
- Added runtime system-prompt override in trainer instead of mutating data files, enabling cheap controlled ablations (`as_is`, `v2`, `v3`).
- Continued using pure HF+PEFT cloud path for parity debugging.
- Treated benchmark-on-checkpoint as the source of truth for promotion, not trainer auto-selected best checkpoint.

## Ready for Next Session
- ✅ v5 + forced v2 training profile is now reproducible and code-pinned.
- ✅ A cloud run achieved **100%** again (`checkpoint-1200`) on the canonical 23-example test set with `prompt_mode=v2`.
- ✅ Commits are pushed and traceable for handoff continuity.
- 🔧 Next high-value step: automate post-train checkpoint sweep (`600/800/1000/1200`) and auto-promote the best benchmark checkpoint.

## Context for Future
This session resolved the immediate “why did it regress?” loop without another blind hyperparameter sweep. The main unlock was separating prompt effects from dataset effects and then selecting by benchmark outcome. With v5 data unchanged and `seq=512`, forcing the v2 system prompt plus selecting `checkpoint-1200` produced a clean **100%** cloud benchmark result, confirming the cloud stack can match local-quality targets when checkpoint selection and prompt policy are aligned.
