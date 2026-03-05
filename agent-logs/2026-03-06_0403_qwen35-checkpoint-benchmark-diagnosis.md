# Qwen3.5 Checkpoint Benchmark Diagnosis (1000 vs 1500)

**Date:** 2026-03-06  
**Agent:** Codex (GPT-5)  
**Status:** ✅ Completed  
**Continues from:** `2026-03-06_0257_qwen35-textonly-cloud-probes.md`

## User Intention
User wanted immediate, trustworthy checkpoint comparison after a completed Qwen3.5 cloud run: benchmark both `checkpoint-1000` and `checkpoint-1500` on Modal, explain any quality gap clearly, and identify whether failures came from training quality vs inference/export bugs before spending more money.

## What We Accomplished
- ✅ **Verified run artifacts exist on Modal volume** for `spoke-qwen35-4b-hf-v5-v2prompt-1500-20260306-0625` (`checkpoint-500/1000/1500`, `adapter`, `merged`) and for merged `checkpoint-1000` run (`...-ckpt1000/merged`).
- ✅ **Ran requested benchmarks on `checkpoint-1000` and `checkpoint-1500`** across both suites (`core23` + `broad58`) using `prompt_mode=v2`.
- ✅ **Captured and inspected raw benchmark JSON outputs** to validate that `checkpoint-1000` failures were caused by trailing generation artifacts, not just visible formatting in terminal logs.
- ✅ **Diagnosed a concrete stop-token config mismatch** between merged artifacts (`ckpt1000` vs final `1500`) that explains the runaway generation behavior.
- ✅ **Explained the `96%` result on final checkpoint precisely**: one exact-match miss (`Kibbinay` expected vs `Kibinaay` produced), not an evaluation bug.

## Technical Implementation
**Run and merge targets**
- Training run: `spoke-qwen35-4b-hf-v5-v2prompt-1500-20260306-0625`
- Checkpoint merge target benchmarked: `spoke-qwen35-4b-hf-v5-v2prompt-1500-20260306-0625-ckpt1000`

**Benchmark results**
- `checkpoint-1000` + `core23`: `0%` (`exact=0, semantic=0, partial=23, fail=0`, avg latency `6.03s`)
- `checkpoint-1000` + `broad58`: `0%` (`exact=0, semantic=0, partial=56, fail=2`, avg latency `6.21s`)
- `checkpoint-1500` + `core23`: `96%` (`exact=22, semantic=0, partial=1, fail=0`, avg latency `0.38s`)
- `checkpoint-1500` + `broad58`: `71%` (`exact=39, semantic=2, partial=15, fail=2`, avg latency `0.49s`)

**Result artifacts**
- `spoke/bench/result_spoke-qwen35-4b-hf-v5-v2prompt-1500-20260306-0625-ckpt1000_modal_v2_test_set_v3.json`
- `spoke/bench/result_spoke-qwen35-4b-hf-v5-v2prompt-1500-20260306-0625-ckpt1000_modal_v2_test_set_evals.json`
- `spoke/bench/result_spoke-qwen35-4b-hf-v5-v2prompt-1500-20260306-0625_modal_v2_test_set_v3.json`
- `spoke/bench/result_spoke-qwen35-4b-hf-v5-v2prompt-1500-20260306-0625_modal_v2_test_set_evals.json`

## Bugs & Issues Encountered
1. **`checkpoint-1000` output corruption during generation**
   - Symptom: outputs looked initially correct, then appended long trailing artifacts (`\n!\n!\n...`) and occasional leaked chat-role text (`user`, `assistant`) causing all rows to score non-exact.
   - **Fix/Diagnosis:** extracted and inspected benchmark JSON `repr(...)` values to prove artifacts were in actual model output, not terminal rendering.

2. **Stop-token mismatch between merged artifacts**
   - `ckpt1000/merged/generation_config.json`: `eos_token_id=248044`
   - `1500/merged/generation_config.json`: `eos_token_id=[248046,248044]`, `pad_token_id=248044`
   - **Impact:** `ckpt1000` failed to stop cleanly and drifted into garbage continuation.
   - **Fix direction:** normalize generation config at merge/export time so intermediate checkpoint merges preserve the same EOS/PAD semantics as final merged artifact.

3. **`agent-logs/README.md` missing in repo path**
   - Requested template file still not present under `agent-logs/`.
   - **Fix:** followed the established de facto structure used across existing `agent-logs/*.md` files for continuity.

## Key Learnings
- `checkpoint-1000` was not a true quality read; it was primarily an inference/export config failure.
- Final `checkpoint-1500` behaves normally and quickly (no artifact rows), which isolates the issue to checkpoint merge/config handling rather than the entire training run.
- The `96%` on `core23` is a real single-example miss (`Kibbinay` vs `Kibinaay`), so remaining gap is now targeted lexical precision, not infrastructure instability.

## Architecture Decisions
- Keep evaluating this run from the final merged artifact as the quality reference until checkpoint-merge EOS/PAD normalization is made deterministic.
- Treat intermediate checkpoint benchmarks as invalid if artifact signatures appear (`!\n` loops, role leakage), and require raw JSON inspection before drawing model-quality conclusions.

## Ready for Next Session
- ✅ Benchmarks requested by the user are complete for both checkpoints and both suites.
- ✅ Root cause category is identified: checkpoint merge/inference stop-token config mismatch for `checkpoint-1000`.
- 🔧 Next change should be a deterministic merge/export patch that copies or reconstructs final-generation EOS/PAD behavior for intermediate checkpoints before benchmarking them.

## Context for Future
This session prevents a costly wrong conclusion. `checkpoint-1000` appeared catastrophic (`0%`) but was mostly an inference-stop configuration failure after merge; `checkpoint-1500` confirms training itself is viable (`96% core23`). Future checkpoint selection experiments must first enforce generation config parity; otherwise quality comparisons will be contaminated by export artifacts instead of model learning signal.
