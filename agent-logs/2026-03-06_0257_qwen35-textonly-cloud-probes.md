# Qwen3.5 Text-Only Cloud Probe Recovery

**Date:** 2026-03-06  
**Agent:** Codex (GPT-5)  
**Status:** ✅ Completed  
**Continues from:** `2026-03-05_2302_v2-prompt-v5-cloud-parity.md`

## User Intention
User wanted to stop burning money on broken cloud runs and get a real answer on whether Qwen3.5 (2B/4B) can be trained usefully on Modal for this task. The practical goal was to fix the current blocker, run low-cost validation probes end-to-end, then capture outcomes in the ledgers and session log for future decisions.

## What We Accomplished
- ✅ **Fixed the active cloud blocker for Qwen3.5 loading** - upgraded Modal HF runtime and implemented text-only loading for Qwen3.5 models.
- ✅ **Ran base cloud benchmarks for both Qwen3.5 sizes** - 2B (`9%`) and 4B (`13%`) on `core23` with `prompt_mode=v2`.
- ✅ **Ran cost-capped 50-step smoke finetunes for both sizes** - exported merged bf16 artifacts and benchmarked immediately.
- ✅ **Measured post-smoke quality deltas** - 2B improved to `22%`; 4B improved to `30%` on `core23`.
- ✅ **Updated both ledgers with this experiment set** - compute ledger and main experiment ledger now include the new runs, metrics, and decision implications.
- ✅ **Saved benchmark artifacts for traceability** - four new JSON result files under `spoke/bench/`.

## Technical Implementation
Key code changes were applied to force a clean text-only path for Qwen3.5 in both training and benchmarking.

**Files Modified:**
- `spoke/cloud/train_hf.py` - upgraded to `transformers==5.2.0`; added `Qwen3_5ForCausalLM` + `text_config` loading path for `model_type=qwen3_5`.
- `spoke/cloud/benchmark.py` - upgraded to `transformers==5.2.0`; added matching Qwen3.5 text-only model load path.
- `spoke/CLOUD_COMPUTE_LEDGER.md` - added C21-C24 runs, updated stack versions/findings/recommendations.
- `spoke/LEDGER.md` - added Qwen3.5 base/smoke benchmarks, cloud training status updates, queue and findings updates.
- `spoke/bench/result_Qwen-Qwen3.5-2B_modal_v2_test_set_v3.json` - new base benchmark artifact.
- `spoke/bench/result_Qwen-Qwen3.5-4B_modal_v2_test_set_v3.json` - new base benchmark artifact.
- `spoke/bench/result_spoke-qwen35-2b-hf-smoke50-20260306_modal_v2_test_set_v3.json` - new smoke benchmark artifact.
- `spoke/bench/result_spoke-qwen35-4b-hf-smoke50-20260306_modal_v2_test_set_v3.json` - new smoke benchmark artifact.

## Bugs & Issues Encountered
1. **`model type qwen3_5 not recognized` in Modal benchmark path** - base Qwen3.5 cloud benchmarks failed immediately.
   - **Fix:** upgraded cloud image to `transformers==5.2.0` and loaded Qwen3.5 via `Qwen3_5ForCausalLM` with `model_config.text_config`.
2. **Qwen3.5 fast-path kernels not available in HF image** - runtime fell back to torch path (`flash-linear-attention`/`causal-conv1d` warning).
   - **Workaround:** proceeded with smoke probes for quality signal; documented that this path is valid for compatibility but not speed-optimized.
3. **`agent-logs/README.md` missing in this repo path**
   - **Fix:** followed the established logging template and structure already used in this repo, consistent with prior sessions.

## Key Learnings
- **Compatibility is fixed, quality is not.** Qwen3.5 now loads/trains in cloud HF text-only mode, but core23 quality remains low (`30%` best at 50 steps).
- **4B outperformed 2B in early smoke quality** (`30%` vs `22%`) and also ran faster in reported steps/s on this specific setup (`1.035` vs `0.777`).
- **This is now a solved engineering-path problem, not a mystery failure.** The remaining gap is recipe/model suitability, not broken infrastructure.

## Architecture Decisions
- **Used pure HF+PEFT path for Qwen3.5 probes** to avoid Unsloth-specific export/runtime ambiguity.
- **Ran base benchmarks before smoke finetunes** to quantify true delta per dollar.
- **Capped experiments at 50 steps** to control spend while validating direction.

## Ready for Next Session
- ✅ **Qwen3.5 cloud path is operational and reproducible** with the new text-only loader.
- ✅ **Ledger continuity is restored** with explicit run IDs, metrics, and benchmark outputs.
- 🔧 **Next decision point is strategic:** either define a stronger Qwen3.5 training recipe (longer run + targeted data/objective changes) or prioritize Qwen3 where parity and quality are already proven.

## Context for Future
This session removed uncertainty about whether Qwen3.5 cloud training was fundamentally broken. It is now technically functional, but under the current recipe it is not quality-competitive with the established Qwen3 path, so further Qwen3.5 spend should be gated behind a concrete quality-improvement hypothesis.
