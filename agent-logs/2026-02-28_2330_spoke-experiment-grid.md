# Spoke Experiment Grid: Iters, Quantization, Model Exploration

**Date:** 2026-02-28
**Agent:** Claude Opus 4.6
**Status:** 🔄 Ongoing

## User Intention
User wanted to systematically explore the fine-tuning parameter space for Spoke (ASR post-processing LLM). After the initial 1000-iter training run showed overfitting, the goal was to test shorter training (200 iters), different quantization levels (4/6/8-bit), and begin evaluating alternative models (Llama 3.2 1B). Also wanted to try training without multi-step examples to see if complex examples confuse the model on simpler tasks. Building on `2026-02-28_2300_spoke-finetune-pipeline.md`.

## What We Accomplished
- ✅ **200-iter training run** - Qwen3-4B LoRA, val loss still improving at iter 200 (0.161), no overfitting
- ✅ **Quantization sweep (4/6/8-bit)** - From 200-iter fused model, benchmarked all three levels with generic + spoke prompts
- ✅ **Identified 6-bit as sweet spot** - 83% spoke accuracy (matches bf16), 3.1 GB size, 0.82s latency
- ✅ **Set up task tracking system** - 6 tasks in todo list for the full experiment grid
- ✅ **Gitignore cleanup** - Added spoke/fused/ and spoke/model/ to .gitignore (large transient artifacts)
- ✅ **Organized adapter checkpoints** - Archived run1 (1000-iter) checkpoints, clean directory for run2
- ⚠️ **Llama 3.2 1B research** - Found mlx-community models (bf16/8bit/4bit), not yet benchmarked
- ⚠️ **No-multi-step experiment** - Planned (task #8), not yet executed

## Technical Implementation

### 200-iter Training
Config changes in `spoke/config.yaml`:
- `iters: 200` (from 1000), `steps_per_eval: 50` (from 100), `save_every: 200`
- ~1.7 epochs over 472 examples (each example seen <2 times)
- Val loss curve: 3.301 -> 0.298 -> 0.195 -> 0.185 -> **0.161** (still improving!)
- Train loss at iter 200: 0.127 (healthy gap with val loss = no overfitting)
- Used `caffeinate -i` to prevent Mac sleep during training

### Quantization Sweep (200-iter model)
| Quant | Size | Generic | Spoke | Latency (generic) |
|-------|------|---------|-------|--------------------|
| bf16+LoRA | 7.5 GB | 58% | **83%** | 1.70s |
| 8-bit | 4.0 GB | 58% | **83%** | 0.97s |
| 6-bit | 3.1 GB | 58% | **83%** | 0.82s |
| 4-bit | 2.1 GB | 58% | 75% | 0.68s |

### 200-iter vs 600-iter Comparison
| Config | Generic | Spoke |
|--------|---------|-------|
| 600-iter bf16+LoRA | **75%** | **83%** |
| 600-iter 4-bit | 67% | 75% |
| 200-iter bf16+LoRA | 58% | **83%** |
| 200-iter 6-bit | 58% | **83%** |
| 200-iter 4-bit | 58% | 75% |

**Files Modified:**
- `spoke/config.yaml` - Updated iters 1000->200, steps_per_eval 100->50
- `.gitignore` - Added spoke/fused/ and spoke/model/
- `spoke/adapters/run1_1000iters/` - Archived old checkpoints
- `spoke/adapters/iter200_run2/` - New 200-iter checkpoint
- `spoke/model-4bit/`, `spoke/model-6bit/`, `spoke/model-8bit/` - Quantized models
- `spoke/bench/result_*.json` - Multiple new benchmark result files

## Bugs & Issues Encountered
1. **Local model path treated as HuggingFace repo** - `mlx_lm.load("spoke/model-4bit")` tried to fetch from HuggingFace
   - **Fix:** Use relative path with directory separator: `../../spoke/model-4bit` (needs at least one `/` to be recognized as local)

## Key Learnings
- **6-bit quantization preserves accuracy** - For Qwen3-4B at 200 iters, 6-bit and 8-bit both maintain 83% spoke accuracy (same as bf16). 4-bit is where quality drops. This was consistent across both 200-iter and 600-iter runs.
- **200 iters is enough with spoke prompts** - With the full spoke dynamic prompt, 200-iter model matches 600-iter at 83%. The gap only shows on generic prompts (58% vs 75%), meaning the extra training helps internalize rules but isn't needed if prompts provide them.
- **Val loss still improving at 200** - Suggests we could push to 300-400 iters without overfitting. The 1000-iter run showed overfitting started around iter 500.
- **Same 2 persistent failures everywhere** - self-correction #3 and quote-endquote #6 fail across ALL model configs. These are data quality issues, not model/quantization issues.
- **`mlx_lm.load()` path detection** - Paths without `/` are treated as HuggingFace repo IDs. Local paths need explicit directory separators.

## Architecture Decisions
- **Sweep from 200-iter, not 600-iter** - 200-iter model has lower risk of overfitting damage under quantization. Results confirmed: 200-iter 6-bit matches 600-iter bf16 on spoke prompts.
- **Keep val set (8 examples)** - User suggested merging val into train, but val set is what detected overfitting in run 1. Worth keeping for training diagnostics.
- **Organized checkpoints by run** - `run1_1000iters/`, `iter200_run2/` subdirectories prevent confusion when comparing across experiments.

## Ready for Next Session
- ✅ **200-iter 6-bit model** at `spoke/model-6bit/` - Best deployment candidate (83%, 3.1 GB, 0.82s)
- ✅ **Llama 3.2 1B models identified** - `mlx-community/Llama-3.2-1B-Instruct-{bf16,8bit,4bit}` ready to benchmark
- ✅ **Task list active** with remaining experiments:
  - #8: Train without multi-step examples (blocked by #7, now unblocked)
  - #9: Zero-shot benchmark Llama 3.2 1B
  - #10: Fine-tune Llama 3.2 1B if viable
  - #11: Update results.html with full comparison

## Remaining Task Board
| # | Task | Status |
|---|------|--------|
| 6 | Quant sweep (4/6/8-bit) | ✅ Done |
| 7 | Train Qwen3-4B @ 200 iters | ✅ Done |
| 8 | Train without multi-step examples | Pending |
| 9 | Zero-shot Llama 3.2 1B | Pending |
| 10 | Fine-tune Llama 3.2 1B | Pending (blocked by #9) |
| 11 | Update results.html dashboard | Pending (blocked by all) |

## Context for Future
This session established that 6-bit quantization at 200 iters is the sweet spot for Qwen3-4B deployment (83% accuracy, 3.1 GB). The next sessions should focus on: (1) testing Llama 3.2 1B as a smaller alternative, (2) trying training without multi-step examples to see if it helps simpler tasks, and (3) generating targeted training data for the 2 persistent failure categories (self-correction context preservation, quote-endquote scoping).
