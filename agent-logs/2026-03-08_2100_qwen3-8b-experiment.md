# Qwen3-8B Experiment: Bigger Model = Worse for Copy-Heavy Editing

**Date:** 2026-03-08
**Agent:** Claude Opus 4.6
**Status:** Completed

## User Intention
User wanted to test whether doubling model capacity (Qwen3-8B vs Qwen3-4B) would improve accuracy on the v5 test set. Initially considered Qwen3.5-9B but switched to Qwen3-8B to keep the same architecture family for a clean comparison.

## What We Accomplished
- Verified Qwen3-8B exists as instruct model on HuggingFace (6.4M downloads, `conversational` tag)
- Killed an accidentally-launched Qwen3.5-9B run on Modal before it consumed GPU time
- Trained Qwen3-8B with identical recipe to T4-v5split baseline (lr=1e-5, r=8, alpha=16, dropout=0.05, adam, 2000 steps)
- Benchmarked both step 1000 (auto-merged) and step 2000 (manual merge) on v5 test set
- Results: 72% (step 1000) and 73% (step 2000) — both 10 pts worse than 4B baseline (82%)
- Updated LEDGER with Qwen3-8B-T1 experiment, benchmark rows, finding #92
- Updated experiment summary table

## Technical Implementation

**Training:** 13 min on Modal L40S, 2.56 it/s. Eval loss bottomed at 0.107 (step 1000), rose to 0.175 by step 2000.

**Merge script bug:** `merge_adapter_checkpoint.py` defaults to `Qwen/Qwen3-4B-Instruct-2507` as base model. Must pass `--model-name Qwen/Qwen3-8B` explicitly for non-4B models. Shape mismatch error: `torch.Size([8, 4096])` vs `torch.Size([8, 2560])`.

**Key failure mode:** 8 hard fails, ALL in quote category. The 8B model outputs literal "quote-unquote" text:
- Expected: `His "investment" strategy...`
- Got: `His quote-unquote investment strategy...`

The 4B model converts these correctly with 0 fails.

**Files Modified:**
- `spoke/LEDGER.md` — Added Qwen3-8B-T1 entry, benchmark rows, finding #92
- `spoke/bench/result_spoke-qwen3-8b-v5split-2k-20260308_modal_v2_test_set_v5.json` — Step 1000 results
- `spoke/bench/result_spoke-qwen3-8b-v5split-2k-20260308-ckpt2000_modal_v2_test_set_v5.json` — Step 2000 results

## Bugs & Issues Encountered
1. **Merge script wrong base model** — Defaults to 4B, crashes with shape mismatch on 8B adapters.
   - **Fix:** Pass `--model-name Qwen/Qwen3-8B` explicitly.
2. **Background task ID lost** — The step 2000 merge task ID couldn't be retrieved via TaskOutput (known issue). Re-ran in foreground.

## Key Learnings
- **Larger models have stronger base priors that resist mechanical transformations.** The 8B model "knows" that "quote-unquote" is valid English text. The 4B model is more malleable — it learns from examples that "quote-unquote X" should become `"X"` without overthinking.
- **This is consistent with the DRES paper (Teleki et al.):** Phi-4 over-deletes because reasoning models bias toward semantic abstraction. Same principle applies here — more capacity = more "intelligence" = worse at literal text manipulation.
- **4B is the sweet spot for Spoke.** Enough capacity to learn all editing rules from ~1000 examples, not so much that it resists them.

## Context for Future
Model capacity exploration is now complete for Qwen3. The 4B model is confirmed as the right size — smaller (1.7B) is too weak (74%), larger (8B) is too "smart" (73%). The accuracy gap is a data problem, not a model problem. Next work should focus on v6 data generation targeting weak categories: at-symbol (60%), multi (30%), spell (67%).
