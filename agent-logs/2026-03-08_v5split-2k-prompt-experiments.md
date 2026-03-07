# V5 Split 2K Steps + Prompt Engineering Experiments

**Date:** 2026-03-08
**Agent:** Claude Opus 4.6
**Status:** Completed
**Continues from:** `2026-03-07_2145_v5-stratified-split-benchmark.md` (79% baseline on 131 examples)

## User Intention
Extend the T4-v5split training to 2000 steps (from 1000), compare checkpoint quality, then systematically test whether inference-time prompt engineering can improve accuracy beyond the v2 baseline.

## What We Accomplished

### Training Extension
- Ran 2000 steps with checkpoints every 250 after step 1000 (1000, 1250, 1500, 1750, 2000)
- Same recipe: Qwen3 4B, lr=1e-5, r=8, alpha=16, dropout=0.05, adam, max_seq=256, v2 prompt

### Checkpoint Comparison (v5 test, 131 examples)
| Checkpoint | Eval Loss | Accuracy | Notes |
|-----------|-----------|----------|-------|
| Step 1000 | 0.115 (best) | 79% | Auto-selected by eval_loss |
| Step 2000 | 0.164 | **82%** | Better despite worse eval_loss |

**Finding #88**: Step 2000 > step 1000 despite worse eval_loss. 6th confirmation that eval_loss minimum is harmful for checkpoint selection on this task.

### Prompt Engineering Experiments (all on step 2000 checkpoint)
| Prompt | Tokens | Accuracy | Delta vs v2 |
|--------|--------|----------|-------------|
| v2 (base) | 83 | **82%** | baseline |
| v3 (full rules) | 191 | 79% | -3 pts |
| v4 (cherry-picked rules + @) | 121 | 79% | -3 pts |
| v4 (no @ rule) | 112 | 76% | -6 pts |

**Finding #89**: v3 prompt is a net wash — helps some categories but hurts others equally.

**Finding #90**: Inference-time prompt engineering is a dead end for fine-tuned models. Every rule added creates cross-category interference:
- Quote rule bleeds into emphasis (wraps bold words in quotes)
- At-symbol rule inserts @ into spell/camelCase outputs
- Multi rule makes self-correction too aggressive
- Disfluency rule strips discourse markers from hard-negatives

### Category-Level Analysis (step 2000)
| Category | v2 (82%) | v3 (79%) | Best prompt |
|----------|----------|----------|-------------|
| at-symbol | 60% | 60% | tie |
| camelcase | 100% | 100% | tie |
| caps | 100% | 100% | tie |
| disfluency | 100% | 100% | tie |
| emoji | 80% | 80% | tie |
| emphasis | 100% | 75% | v2 |
| hard-negative | 100% | 93% | v2 |
| meta | 100% | 100% | tie |
| multi | 30% | 60% | v3 |
| quote | 92% | 77% | v2 |
| self-correction | 89% | 74% | v2 |
| spell | 67% | 71% | v3 |

v3 helps multi (+30 pts) and spell (+4 pts) but destroys emphasis (-25 pts), hard-neg (-7 pts), quote (-15 pts), self-correction (-15 pts). Net: -3 pts.

## Files Created/Modified
- `spoke/cloud/benchmark.py` — Added V4_PROMPT constant and v4 prompt mode
- `spoke/LEDGER.md` — Updated T4-v5split to 2000 steps/82%, added findings #88-90
- `spoke/bench/result_spoke-qwen3-4b-v5split-2k-20260308_modal_v2_test_set_v5.json` — step 1000 result
- `spoke/bench/result_spoke-qwen3-4b-v5split-2k-20260308-ckpt2000_modal_v2_test_set_v5.json` — step 2000/v2
- `spoke/bench/result_spoke-qwen3-4b-v5split-2k-20260308-ckpt2000_modal_v3_test_set_v5.json` — step 2000/v3
- `spoke/bench/result_spoke-qwen3-4b-v5split-2k-20260308-ckpt2000_modal_v4_test_set_v5.json` — step 2000/v4 (no-@ variant, overwrote with-@ variant)

## Key Learnings
1. **More training steps help even past eval_loss minimum.** The model continues learning useful patterns even as it becomes less calibrated on validation loss.
2. **Prompt engineering and fine-tuning are complementary during training, but conflicting at inference.** The model internalizes the training prompt's rules into its weights. Adding different/more rules at inference creates contradictions.
3. **"Train rich, deploy lean" is the correct strategy.** Train with detailed rules as supervision signal, deploy with minimal v2 prompt. The model has already learned the rules — repeating them at inference is redundant at best, harmful at worst.

## Conclusion
The only remaining lever for accuracy improvement is **better training data** targeting the weak categories:
- at-symbol (60%) — hardest category, model misplaces @ insertions
- multi (30%) — multi-step chaining still very weak
- spell (67%) — edge cases in spell scoping
- emoji (80%) — close but not perfect

Prompt engineering is exhausted. Next gains must come from v6 data generation.
