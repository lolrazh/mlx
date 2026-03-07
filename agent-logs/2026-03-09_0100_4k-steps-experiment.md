# 4000 Steps Experiment: Extended Training Hurts V2, Recovers V4

**Date:** 2026-03-09
**Agent:** Claude Opus 4.6
**Status:** Completed
**Continues from:** `2026-03-08_2100_qwen3-8b-experiment.md`

## User Intention
User wanted to test whether doubling training steps from 2000 to 4000 on the winning Qwen3-4B recipe would push accuracy past the 82% baseline on the v5 test set (131 examples). Both v4-trained and v2-trained models were benchmarked at steps 3000 and 4000.

## What We Accomplished
- Discovered the previous 4k run (spoke-qwen3-4b-v5split-4k-20260308) was trained on v4 prompt data — Modal volume had stale data from the v4prompt experiment
- Rebuilt v5 data locally with correct v2 prompt, re-uploaded to Modal
- Benchmarked v4-trained 4k run at step 3000 (81%) and step 4000 (82%)
- Trained new 4k run on correct v2 data (spoke-qwen3-4b-v5split-4k-v2-20260309)
- Merged and benchmarked step 3000 (81%) and step 4000 (80%)
- Updated LEDGER with T5-v4prompt-4k, T6-v2-4k entries, benchmark rows, finding #93

## Technical Implementation

**V4-trained 4k results (v4 inference prompt):**
| Step | Accuracy | Exact | Semantic | Partial | Fail |
|------|----------|-------|----------|---------|------|
| 2000 | 79% | — | — | — | — |
| 3000 | 81% | 99 | 7 | 23 | 2 |
| 4000 | 82% | 100 | 7 | 22 | 2 |

**V2-trained 4k results (v2 inference prompt):**
| Step | Accuracy | Exact | Semantic | Partial | Fail |
|------|----------|-------|----------|---------|------|
| 1000 | 79% | — | — | — | — |
| 2000 | 82% | 103 | 4 | 24 | 0 |
| 3000 | 81% | 102 | 4 | 25 | 0 |
| 4000 | 80% | 102 | 3 | 26 | 0 |

**Training config:** Identical to T4-v5split — Qwen3 4B, lr=1e-5, r=8, alpha=16, dropout=0.05, adam, max_seq=256, v2 prompt, cloud Modal L40S HF+PEFT. 19 min, 15.3 epochs.

**Files Modified:**
- `spoke/LEDGER.md` — Added T5-v4prompt-4k and T6-v2-4k experiment entries, benchmark rows, finding #93
- `spoke/data/v5/train.jsonl` — Rebuilt with v2 prompt (was accidentally v4)
- `spoke/data/v5/valid.jsonl` — Rebuilt
- `spoke/data/v5/test.jsonl` — Rebuilt
- `spoke/bench/test_set_v5.json` — Rebuilt
- `spoke/data/v5/valid_categorized.json` — Rebuilt
- `spoke/bench/result_spoke-qwen3-4b-v5split-4k-20260308-ckpt3000_modal_v4_test_set_v5.json` — V4-trained step 3000
- `spoke/bench/result_spoke-qwen3-4b-v5split-4k-20260308-ckpt4000_modal_v4_test_set_v5.json` — V4-trained step 4000
- `spoke/bench/result_spoke-qwen3-4b-v5split-4k-v2-20260309-ckpt3000_modal_v2_test_set_v5.json` — V2-trained step 3000
- `spoke/bench/result_spoke-qwen3-4b-v5split-4k-v2-20260309-ckpt4000_modal_v2_test_set_v5.json` — V2-trained step 4000

## Bugs & Issues Encountered
1. **Stale data on Modal volume** — `build_split.py` was reverted from v4→v2 locally, but the JSONL files were never regenerated, and the Modal volume still had v4 data from the previous upload. The first 4k run unknowingly trained on v4 data.
   - **Fix:** Rebuilt JSONL with `python spoke/data/v5/build_split.py`, re-uploaded with `python spoke/cloud/upload_data.py`. Verified file timestamps and sizes match.
2. **Wrong CLI flag** — Used `--save-every` instead of `--save-steps` for the train_hf.py script.
   - **Fix:** Checked error message, used correct `--save-steps 1000`.
3. **Background task ID lost** — TaskOutput returned "No task found with ID" for background training task. Known issue.
   - **Fix:** Read output file directly via `tail`.

## Key Learnings
- **Data pipeline has 3 layers of state**: `build_split.py` → `*.jsonl` files → Modal volume. Changing one doesn't propagate to the others. Always verify the full chain.
- **V2-trained models peak at ~7.7 epochs (2000 steps on 1046 examples).** Beyond that, over-memorization degrades generalization monotonically.
- **V4-trained models benefit from extended training** — the extra prompt rules may act as a regularizer, giving the model more diverse signals to learn from in later epochs. But they only recover to v2's peak, never exceeding it.
- **Zero hard fails at all v2-trained checkpoints** — the model becomes more conservative (more partials) rather than hallucinating. This is the right failure mode for production.
- **The accuracy ceiling is data-limited.** Training duration (2k vs 4k), prompt (v2 vs v4), and model size (4B vs 8B) all converge around 80-82%. Only more/better training data can push higher.

## Context for Future
The training duration axis is now fully explored. Combined with prompt (findings #90, #91) and model size (finding #92), three of four optimization axes are exhausted. The only remaining lever is v6 data generation targeting weak categories: at-symbol (60%), multi (30%), spell (67%), emoji (80%). User's Modal credits are depleted ($1.38 → ~$0). Future cloud runs will need credits replenished.
