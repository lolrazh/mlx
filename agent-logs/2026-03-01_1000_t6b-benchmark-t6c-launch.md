# T6b/T6c Benchmarks: AdamW Ablation Conclusion

**Date:** 2026-03-01
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed

## User Intention
User wanted to complete the T6b benchmark pipeline, then extend AdamW training to 800 iters (T6c) to see if it could match T4's 74% accuracy and determine once and for all whether AdamW improves 6-bit quantization resilience.

## What We Accomplished
- ✅ **T6b bf16 benchmark** — 70% (15 exact + 1 semantic). 4 pts below T4.
- ✅ **T6b 6-bit benchmark** — 57% (12 exact + 1 semantic). 13% quant loss.
- ✅ **T6c training** — resumed from T6b iter 500, ran +300 more (800 total). Val loss bottomed at 0.190 @600 then plateaued ~0.204.
- ✅ **T6c bf16 benchmark** — **74%** (15 exact + 2 semantic) at iter 800. **Matches T4!**
- ✅ **T6c 6-bit benchmark** — 61% (12 exact + 2 semantic). 13% quant loss. **Worse than T4's 65%.**
- ✅ **Conclusive finding: AdamW loses on 6-bit deploy.** Same bf16 accuracy, but 2.7x more training AND 4% worse 6-bit.
- ✅ **Ledger fully updated** — T6c rows, findings 18-20, experiment queue revised.

## Technical Implementation

### Full AdamW Comparison Table
| Run | bf16 | 6-bit | Quant Loss | Val Loss | Iters |
|-----|------|-------|------------|----------|-------|
| T4 (adam) | **74%** | **65%** | 9% | 0.174 | 300 |
| T6 (adamw) | 43% | 43% | 0% | 0.231 | 200 |
| T6b (adamw) | 70% | 57% | 13% | 0.200 | 500 |
| T6c (adamw) | **74%** | 61% | 13% | 0.205 | 800 |

### T6c Val Loss Curve (resume iters → total iters)
| Resume | Total | Val Loss | Note |
|--------|-------|----------|------|
| 1 | 501 | 0.200 | Matches T6b end |
| 50 | 550 | 0.219 | Optimizer cold start |
| 100 | 600 | **0.190** | New low (artifact?) |
| 150 | 650 | 0.201 | Bounce |
| 200 | 700 | 0.204 | Plateau |
| 250 | 750 | 0.204 | Plateau |
| 300 | 800 | 0.205 | Plateau |

### T6c Failure Analysis
bf16→6-bit lost 3 examples (same pattern as T6b):
- at-symbol: bf16 ✓ → 6-bit △ (understood intent but gave instructions instead)
- caps: bf16 ✓ → 6-bit △ (added exclamation mark)
- quote-endquote #10: bf16 ✓ → 6-bit △ (inserted "quote" prefix)

### Val loss vs accuracy paradox
- T6c iter 600 (val loss 0.190, "best"): 65% bf16
- T6c iter 800 (val loss 0.205, "worse"): 74% bf16
- **Val loss on 20 examples is too noisy for reliable accuracy prediction.**

**Files Modified:**
- `spoke/LEDGER.md` — T6c training row, 4 benchmark rows, findings 18-20, experiment queue
- `spoke/config.yaml` — Updated for T6c (iters=300, adapter_path=adapters-t6c)

**Files Created:**
- `spoke/bench/result_qwen3-4b-bf16+lora_v2.json` — T6c bf16 benchmark
- `spoke/bench/result_model-t6c-6bit_v2.json` — T6c 6-bit benchmark
- `spoke/model-t6b-fused/` — Fused T6b model
- `spoke/model-t6b-6bit/` — T6b 6-bit quantized
- `spoke/model-t6c-fused/` — Fused T6c model
- `spoke/model-t6c-6bit/` — T6c 6-bit quantized
- `spoke/adapters-t6c/` — T6c adapters (100/200/300 checkpoints)
- `spoke/adapters-t6c-best/` — T6c best checkpoint (iter 100 = total 600)
- `agent-logs/2026-03-01_1000_t6b-benchmark-t6c-launch.md` — This log

## Bugs & Issues Encountered
1. **Benchmark expects directory, not file** — `--adapter-path` must point to a directory containing `adapter_config.json` + `adapters.safetensors`, not a single `.safetensors` file. Created `adapters-t6c-best/` directory as workaround.
2. **Optimizer state reset on resume** — `--resume-adapter-file` loads weights but resets AdamW momentum/velocity. Causes ~50-iter warm-up with temporarily higher val loss (0.200 → 0.219 → 0.190 recovery).
3. **Val loss misleading** — T6c's best val loss checkpoint (0.190 @600) scored 65%, while worse val loss (0.205 @800) scored 74%. Small validation set (20 examples) makes val loss unreliable.

## Key Learnings
- **AdamW is worse for 6-bit deployment** — consistent 13% quant loss vs adam's 9%. Weight decay may distribute information more evenly, making more weights sensitive to quantization noise.
- **AdamW converges 2.7x slower** — 800 iters to match adam's 300-iter accuracy. For small datasets (447 examples), adam's willingness to "commit harder" actually helps.
- **Val loss is unreliable with 20 validation examples** — differences of 0.01-0.02 are noise. Must benchmark accuracy directly. Val loss only useful for tracking gross overfitting.
- **Optimizer state matters for resume** — momentum/velocity not saved with adapters. Consider this when designing resumed runs.
- **Zero quant loss at low accuracy is meaningless** — statistical artifact. Test quant loss at >65% accuracy for meaningful results.

## Architecture Decisions
- **T4 (adam) confirmed as best for deploy** — same bf16, better 6-bit, 3x faster training. AdamW exploration complete.
- **Experiment queue refocused on T4 as baseline** — T7 (cosine LR), T9 (QLoRA), T10 (mask_prompt) now branch from T4, not AdamW.
- **Data improvements > optimizer changes** — the ablation phase tested DoRA, AdamW, DoRA+AdamW. None beat T4's deploy config. Next frontier is data quality/quantity.

## Ready for Next Session
- ✅ **AdamW ablation complete** — conclusive: adam wins for 6-bit deploy
- ✅ **T4 confirmed as best deploy model** — 74% bf16, 65% 6-bit, 300 iters, 3.1 GB
- ✅ **Ledger fully updated** with all T6/T6b/T6c results and findings
- 🔧 **Next moves**: T7 (cosine LR), T9 (QLoRA memory), T10 (mask_prompt), or data expansion (650-750 examples)

## Context for Future
The AdamW ablation is now conclusive after T6→T6b→T6c. AdamW matches adam's bf16 accuracy but requires 2.7x more training and produces worse 6-bit models. DoRA was already eliminated (OOM + 18s latency). The optimizer axis of the ablation is fully explored — adam is the winner for this task. The biggest remaining improvements are likely on the data axis (more examples, especially for XML, email, code-aware categories that still fail) or inference-time (better prompts). Building on `2026-03-01_0830_ablation-dora-adamw.md`.
