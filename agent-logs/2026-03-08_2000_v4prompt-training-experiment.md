# V4 Prompt Training Experiment: "Train Rich, Deploy Lean" Disproved

**Date:** 2026-03-08
**Agent:** Claude Opus 4.6
**Status:** Completed
**Continues from:** `2026-03-08_v5split-2k-prompt-experiments.md` (prompt engineering dead end at inference time)

## User Intention
After discovering that inference-time prompt engineering was a dead end (finding #90), the user wanted to test the opposite approach: training with a richer prompt (v4, 121 tokens with explicit quote/at-symbol/multi rules) while keeping the lean v2 prompt at deployment. The hypothesis was that explicit rules during training would act as supervision signal, helping the model learn weak categories better without the inference-time interference problem. Additionally, the user wanted to verify whether the richer prompt at inference would help when the model was trained with it (matched prompt condition).

## What We Accomplished
- Updated `build_split.py` SYSTEM_PROMPT to v4 (121 tokens with quote, at-symbol, multi rules)
- Rebuilt v5 stratified split with v4 prompt in all training examples
- Uploaded to Modal and trained Qwen3 4B for 2000 steps (same recipe as T4-v5split)
- Benchmarked step 2000 with v4 inference prompt: **79%** (-3 pts vs baseline)
- Benchmarked step 2000 with v2 inference prompt: **73%** (-9 pts vs baseline)
- Reverted `build_split.py` back to v2 prompt (confirmed as correct)
- Updated LEDGER: T5-v4prompt experiment entry, benchmark rows, finding #91
- Committed all results (3 commits total this session)

## Technical Implementation

**Experiment design:**
- Control: T4-v5split (v2 train, v2 inference) = 82%
- Test A: T5-v4prompt (v4 train, v4 inference) = 79%
- Test B: T5-v4prompt (v4 train, v2 inference) = 73%

**V4 prompt (121 tokens):**
```
v2 base (83 tok) +
"Quote-unquote wraps nearest word(s). Quote...end quote wraps everything between."
"At-symbol: insert @, drop instruction."
"Multiple directives: execute all in order. Last conflicting directive wins."
```

**Training config:** Identical to T4-v5split — Qwen3 4B, lr=1e-5, r=8, alpha=16, dropout=0.05, adam, max_seq=256, 2000 steps on Modal L40S.

**Files Modified:**
- `spoke/data/v5/build_split.py` — Temporarily changed SYSTEM_PROMPT to v4, then reverted to v2
- `spoke/LEDGER.md` — Added T5-v4prompt experiment entry, benchmark rows, finding #91, updated header
- `spoke/bench/result_spoke-qwen3-4b-v4prompt-2k-20260308_modal_v4_test_set_v5.json` — New benchmark result
- `spoke/bench/result_spoke-qwen3-4b-v4prompt-2k-20260308_modal_v2_test_set_v5.json` — New benchmark result

## Bugs & Issues Encountered
1. **Modal heartbeat DNS failure at step 1947/2000** — Local client lost DNS resolution, training output showed `socket.gaierror: [Errno 8] nodename nor servname provided`. Training completed on remote GPU (all 2000 steps + merged model saved).
   - **Fix:** No fix needed — checked `modal volume ls` to confirm output was saved. Local network hiccup only.

## Key Learnings
- **"Train rich, deploy lean" does NOT work for copy-heavy editing tasks.** This principle assumes the model needs to learn new behaviors from rules. For ASR post-processing, the correct behavior is fully encoded in input-output pairs. Adding explicit rules competes with data-driven learning and creates prompt dependency.
- **Prompt dependency is asymmetric and catastrophic.** v4 train + v4 inference lost only 3 pts, but v4 train + v2 inference lost 9 pts. The model learned to rely on seeing "Quote-unquote wraps nearest word(s)" — without it, quotes output as literal "quote-unquote" text (4 hard fails vs 0-1 normally). The dependency penalty is 3x larger than the training benefit.
- **"Train lean, deploy lean" is the correct strategy for Spoke.** v2 prompt (83 tokens) contains just enough context to activate the LoRA's editing circuits. The data teaches what to do; the prompt just says "be an ASR cleaner."

## Architecture Decisions
- **Reverted build_split.py to v2 prompt** — Confirmed v2 is the correct training prompt. Any future data rebuilds should use v2.
- **Finding #91 closes the prompt investigation** — Both inference-time (#90) and training-time (#91) prompt enrichment have been tested and disproved. The prompt axis is fully explored.

## Ready for Next Session
- Current best: T4-v5split step 2000, v2 prompt = **82%** on v5 test (131 examples)
- Prompt axis fully explored (v2, v3, v4 at both training and inference time)
- Only remaining lever: **more/better training data** for weak categories

## Context for Future
The prompt investigation is now complete. Two findings (#90, #91) conclusively show that neither inference-time nor training-time prompt enrichment improves accuracy. The v2 prompt (83 tokens) is optimal at both training and inference. All future accuracy gains must come from v6 data generation targeting weak categories: at-symbol (60%), multi (30%), spell (67%), emoji (80%). The 131-example v5 test set provides reliable category-level feedback for measuring data improvements.
