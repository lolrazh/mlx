# T4 Training Run, V2 Baselines & Experiment Ledger

**Date:** 2026-03-01
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed

## User Intention
User wanted to isolate the effect of the new v2 training data (447 examples with targeted fixes) by keeping all hyperparameters the same as T2, training a new model (T4), benchmarking it, and establishing fair zero-shot baselines on the new 23-example v2 test set. Also wanted a comprehensive experiment ledger as a single source of truth for all past and planned runs. Session included data review (with XML tag fixes), Muon optimizer research, and the full train→benchmark→fuse→quantize pipeline.

## What We Accomplished
- ✅ **Created experiment ledger** (`spoke/LEDGER.md`) — baselines, training runs, benchmarks, quant impact, persistent failures, data leakage, experiment queue (T5-T10), key findings
- ✅ **Reviewed training data v2** (3 rounds) — caught broken XML patterns (`<SENSITIVE>SENSITIVE</SENSITIVE>`), user fixed them, verified 447 train / 20 valid / 23 test
- ✅ **Trained T4** — LoRA r=8, v2 data, 300 iters (200 initial + 100 resumed). Best val loss 0.174 @300. Overfits ~350, OOM @450.
- ✅ **Benchmarked T4** — bf16+v2: **74%**, bf16+generic: 61%, 6-bit+v2: **65%**
- ✅ **Fused & quantized T4** — `spoke/model-t4-6bit/` (3.1 GB)
- ✅ **V2 test set baselines** — generic v1: **13%** (floor), v2 prompt: **35%**. Confirms v2 test set is harder than v1.
- ✅ **Researched Muon optimizer** — built into MLX, but designed for full-rank matrices. Low priority for LoRA.
- ✅ **Updated ledger with test set version boundary** — split benchmarks into v1/v2 sections with comparability warning

## Technical Implementation

### T4 Training
```bash
# Initial 200 iters
caffeinate -i mlx_lm.lora -c spoke/config.yaml

# Extended to find overfitting point (resumed from iter 200)
caffeinate -i mlx_lm.lora -c spoke/config.yaml \
  --resume-adapter-file spoke/adapters-t4/0000200_adapters.safetensors
```

Val loss curve: 1→2.534, 50→0.408, 100→0.272, 150→0.256, 200→0.231, 250→0.228, **300→0.174★**, 350→0.194, 400→0.198, 450→OOM

### Baseline Comparison (v2 test set, 23 examples)
| Config | Accuracy | Delta vs floor |
|--------|----------|----------------|
| Zero-shot, generic v1 (B9) | **13%** | — (floor) |
| Zero-shot, v2 prompt (B10) | **35%** | +22 pts |
| T4 fine-tuned, bf16 + v2 | **74%** | +61 pts |
| T4 fine-tuned, 6-bit + v2 | **65%** | +52 pts |

### Key T4 Results
- Self-correction #3 (Cloudflare Workers compound): **FIXED** with both prompts
- Quote-endquote #6 (multi-word scope): **Scope FIXED**, period placement still wrong
- 6-bit quant regression: 9% loss (74%→65%), worse than T2's zero loss

**Files Modified:**
- `spoke/LEDGER.md` — Created and updated multiple times (baselines, training runs, benchmarks, key findings)
- `spoke/config.yaml` — Updated iters (300), steps_per_eval (50), save_every (100), adapter_path (adapters-t4)
- `MEMORY.md` — Updated data counts, baselines, T4 results

**Files Created:**
- `spoke/bench/result_qwen3-4b-bf16+lora_v2.json` — T4 bf16 v2 benchmark
- `spoke/bench/result_qwen3-4b-bf16+lora_generic.json` — T4 bf16 generic benchmark
- `spoke/bench/result_model-t4-6bit_v2.json` — T4 6-bit v2 benchmark
- `spoke/bench/result_qwen3-4b_v2.json` — B10 zero-shot v2 baseline
- `spoke/bench/result_qwen3-4b_generic.json` — B9 zero-shot generic baseline
- `spoke/adapters-t4/` — T4 adapter checkpoints
- `spoke/model-t4-6bit/` — 6-bit quantized T4 model (3.1 GB)

## Bugs & Issues Encountered
1. **OOM crash at iter ~450** — Metal GPU memory exhaustion (exit code 134, SIGABRT). Peak was 14 GB on 24 GB M4.
   - **Workaround:** Best checkpoint (iter 300) already saved. Not blocking.
2. **`--resume-adapter-file` resets iter counter** — Loading pretrained adapters restarts from iter 1. `iters: 600` = 600 MORE, not 600 total.
   - **Workaround:** Noted to user. Actually gave more data points for overfitting curve. Set config to `iters: 300` (the total that matters).
3. **Broken XML training examples** — `<SENSITIVE>SENSITIVE</SENSITIVE>` where tag name = content = nonsensical.
   - **Fix:** User rewrote pattern B examples. Re-review confirmed fix (437→447 examples).

## Key Learnings
- **V2 test set is genuinely harder** — floor drops from 25%→13%, v2 prompt from 50%→35%. New categories (XML, email, hard negatives, code-aware) expose real model gaps.
- **Fine-tuning gain is +61 points** on v2 test (13%→74% bf16). Larger than the +50 we measured on v1.
- **6-bit quant regression is model-specific** — T2 had zero loss, T4 lost 9%. The v2-trained model may have tighter decision boundaries on edge cases (quote-endquote, code-aware) that quantization disrupts.
- **Muon optimizer** — built into MLX (`optim.Muon`), uses Newton-Schulz orthogonalization. Designed for full-rank matrices, unclear benefit for low-rank LoRA adapters. "LoRA meets Riemannion" paper (July 2025) addresses this.
- **Overfitting onset earlier with v2 data** — ~iter 350 vs ~iter 500 on v1 data. Possibly due to harder/more diverse examples requiring more careful learning.

## Architecture Decisions
- **Isolated data effect before changing hyperparameters** — T4 uses identical config to T2, only data changed. This confirms the v2 data improved persistent failures.
- **Ablation study design** — T5-T10 each change one variable at a time from T4 as baseline. Methodical approach over "try everything at once."

## Ready for Next Session
- ✅ **T4 complete** — best on v2 test set (74% bf16, 65% 6-bit)
- ✅ **Fair baselines established** — B9 (13% floor) and B10 (35% v2) on v2 test set
- ✅ **Ledger fully up to date** — all runs, benchmarks, and findings documented
- 🔧 **T5-T10 ablation runs queued** — DoRA, AdamW, cosine LR, full stack, QLoRA, mask_prompt:false
- 🔧 **6-bit quant regression** — worth investigating whether DoRA/AdamW improve quantization resilience

## Context for Future
T4 validates that data quality is the primary driver — targeted examples for persistent failures worked. The v2 test set (23 examples) is now the standard benchmark, with fair baselines at 13% (floor) and 35% (v2 prompt). Next high-impact moves are the ablation runs (T5-T8) to find which hyperparameter changes improve on T4's 74%. The 6-bit regression (9% loss) is a new concern — T2 had zero quant loss, so something about the v2-trained model is more quant-sensitive. Building on `2026-03-01_0400_benchmark-audit-research.md` and `2026-03-01_0500_dataset-v2-quality-pass.md`.
