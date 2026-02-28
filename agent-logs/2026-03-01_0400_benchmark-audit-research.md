# Benchmark Audit, Data Leakage Discovery, & Training Research

**Date:** 2026-03-01
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed

## User Intention
User wanted to understand why accuracy plateaued at 83% despite extensive hyperparameter tuning, prepare for the next training iteration with updated data (v2 prompt, multi-command removed, targeted fixes), and research alternative training methods (QLoRA, DoRA, DPO, etc.) before committing to the next training run. Session evolved into a critical audit of benchmark methodology when data leakage was discovered in the spoke prompt baselines.

## What We Accomplished

### Benchmark Audit & Data Leakage Discovery
- ✅ **Identified data leakage in spoke benchmarks** - 4/11 test examples are exact copies of few-shot examples in `spoke/bench/prompts.py`. The "58% spoke baseline" was actually 38% on clean examples.
- ✅ **Established clean baselines** - Generic v1: 25%, V2: 50%, Fine-tuned+generic: 75%. These are the only honest numbers.
- ✅ **Added evaluation discipline rules to CLAUDE.md** - Terminology (zero-shot vs few-shot vs fine-tuned), leakage warnings
- ✅ **Updated project memory** with corrected baselines and leakage documentation

### V2 Prompt Benchmarking
- ✅ **Added v2 prompt mode to run_benchmark.py** - New `--prompt-mode v2` option
- ✅ **Benchmarked v2 on zero-shot 4-bit** - 50% accuracy (doubled from generic's 25%)
- ✅ **Benchmarked v2 on zero-shot bf16** - 36% (bf16 more verbose at zero-shot)
- ✅ **Benchmarked v2 on fine-tuned r=8 iter400** - 58% (training/inference mismatch penalty)
- ✅ **Key finding**: V2 on fine-tuned model fixed quote-endquote #6 scope (multi-word!) but period placement wrong

### Persistent Failure Analysis
- ✅ **Documented 2 persistent failures** in detail with root cause analysis
- ✅ **Self-correction #3**: Over-aggressive clause replacement (training data teaches "replace whole clause")
- ✅ **Quote-endquote #6**: Wrong quote scope (model applies quote-unquote pattern instead of quote...end quote)
- ✅ **Created datagen brief** at `spoke/data/DATAGEN_BRIEF.md` for the datagen agent

### Training Research (4 Parallel Sonnet Agents)
- ✅ **Agent 1 - MLX-LM capabilities**: Found DoRA, QLoRA (automatic), 5 optimizers, LR schedules, gradient checkpointing, LoRA key targeting — all built into our mlx_lm 0.30.7
- ✅ **Agent 2 - QLoRA**: Auto-activates with quantized base model. Memory ~9GB → ~4-5GB. Quality loss negligible for narrow tasks.
- ✅ **Agent 3 - Alternative paradigms**: DoRA (rank 1 easy win), NEFTune (anti-overfitting), DPO (after SFT plateau). Skip: ORPO, IA3, prefix tuning, BitFit.
- ✅ **Agent 4 - Data optimization**: Quote-endquote only 12 examples (2.4% of data!), punctuation inconsistency in training data, hard negatives missing, `mask_prompt: false` worth testing. Target 650-750 total examples.

### Data Verification
- ✅ **Confirmed datagen agent updated data** - Commit `8e63a9f`: 408 train + 16 valid + 11 test, v2 prompt, multi-command removed, targeted fixes added

## Technical Implementation

### Clean Benchmark Results (All Honest Numbers)

| Config | Prompt | Accuracy |
|--------|--------|----------|
| Base model, no training | generic v1 | **25%** |
| Base model, no training | v2 | **50%** |
| Fine-tuned r=8 iter400 | generic v1 | **75%** |
| Fine-tuned r=8 iter400 | v2 (mismatched) | 58% |
| ~~Base model~~ | ~~spoke (few-shot)~~ | ~~58%~~ → 38% clean (LEAKED) |
| ~~Fine-tuned r=8 iter400~~ | ~~spoke (few-shot)~~ | ~~83%~~ → 75% clean (LEAKED) |

### Proposed Next Training Config
```yaml
fine_tune_type: dora          # was: lora
optimizer: adamw              # was: adam (default, no weight decay)
optimizer_config:
  adamw:
    weight_decay: 0.01
lr_schedule:
  name: cosine_decay
  warmup: 50
  arguments: [1e-5, 200, 1e-7]
iters: 200
steps_per_eval: 50
save_every: 100
```

### MLX-LM Training Features (Newly Discovered)

| Feature | Config | Notes |
|---------|--------|-------|
| DoRA | `fine_tune_type: dora` | Better at low ranks, 1-line change |
| QLoRA | Use quantized model path | Automatic, halves memory |
| AdamW | `optimizer: adamw` | Weight decay prevents drift |
| Cosine decay + warmup | `lr_schedule:` block | Avoids early instability |
| Gradient checkpointing | `grad_checkpoint: true` | 30-50% less memory |
| LoRA key targeting | `keys: [...]` in lora_parameters | Target specific layers |
| Gradient accumulation | `grad_accumulation_steps: N` | Simulate larger batch |

**Files Modified:**
- `spoke/bench/run_benchmark.py` - Added v2 prompt mode + CLI option
- `spoke/data/DATAGEN_BRIEF.md` - Created datagen instructions for other agent
- `CLAUDE.md` - Added evaluation discipline section (zero-shot/few-shot terminology, leakage warnings)
- `agent-logs/2026-03-01_0030_lora-rank-experiments.md` - Updated with failure analysis and data leakage finding

**Files Created:**
- `spoke/bench/result_qwen3-4b_v2.json` - V2 zero-shot 4-bit benchmark
- `spoke/bench/result_qwen3-4b-bf16_v2.json` - V2 zero-shot bf16 benchmark
- `spoke/bench/result_qwen3-4b-bf16+lora_v2.json` - V2 fine-tuned benchmark

## Bugs & Issues Encountered
1. **Data leakage in spoke benchmarks** - 4/11 test examples are exact copies of few-shot examples in `prompts.py`. Both came from the same `evals.csv` pool. Inflated reported accuracy by ~20%.
   - **Fix:** Added warnings to CLAUDE.md and memory. Clean baselines use generic/v2 prompts only. Spoke-mode benchmarks should not be cited without noting leakage.
2. **Terminology confusion (zero-shot vs few-shot)** - Was calling spoke prompt results "zero-shot" when they included few-shot examples.
   - **Fix:** Added precise definitions to CLAUDE.md. Zero-shot = no training + no examples in prompt. Few-shot = examples in prompt. Fine-tuned = model trained.

## Key Learnings
- **Data leakage is easy to miss in small test sets** - With only 12 test examples, 4 leaked examples inflated accuracy from 38% to 58%. Always cross-check test set against any prompt examples.
- **V2 prompt is a strong zero-shot baseline** - 50% accuracy with no few-shot, nearly matching the leaked spoke baseline (58%). The prompt alone does heavy lifting.
- **Fine-tuned model doesn't need few-shot prompts** - Scores 75% with just generic v1 prompt. Fine-tuning bakes in what few-shot was teaching. Simplifies deployment.
- **Training/inference prompt mismatch costs ~17%** - Fine-tuned on v1, benchmarked with v2 = 58% vs 75% with v1. Always match.
- **MLX-LM has many unused features** - DoRA, 5 optimizers, LR schedules, gradient checkpointing, key targeting — all built in, zero code changes.
- **QLoRA is automatic in mlx_lm** - Just point `model:` at a quantized model. No flag needed.
- **Quote-endquote has only 12 training examples** - 2.4% of dataset, smallest category, persistent failure. This is the #1 data fix needed.
- **Punctuation inconsistency in quote training data** - Half put period inside quotes, half outside. Model learns both, picks randomly.
- **`mask_prompt: false` might help for short outputs** - Research shows more gradient signal when prompt isn't masked, especially with ~15-token outputs.

## Architecture Decisions
- **V2 prompt as the standard** - Simpler than spoke dynamic prompt, no leakage risk, nearly as effective. Train with v2, deploy with v2.
- **DoRA over LoRA for next run** - Better at low ranks (r=8), already implemented, zero-cost to test.
- **AdamW + cosine decay** - Standard best practice we've been missing. Weight decay + warmup + decay is free improvement.
- **Target 650-750 total examples** - Research says 400 is the floor for narrow tasks, 600-750 is the sweet spot, >1000 is diminishing returns.
- **Stack improvements** - Combine data fixes + DoRA + AdamW + LR schedule in one training run rather than testing each independently.

## Ready for Next Session
- ✅ **Training data v2 ready** - 408 train / 16 valid / 11 test with v2 prompt (commit `8e63a9f`)
- ✅ **Proposed config improvements documented** - DoRA, AdamW, cosine decay + warmup
- ✅ **Datagen brief ready** at `spoke/data/DATAGEN_BRIEF.md`
- ✅ **Research compiled** from 4 parallel agents covering all major training paradigms
- 🔧 **Data gaps still open** - Need 30-40 more quote-endquote examples, 15-20 hard negatives, punctuation standardization
- 🔧 **QLoRA experiment** - Worth trying to validate memory savings with no quality loss
- 🔧 **mask_prompt: false experiment** - Quick A/B test, could improve gradient signal

## Remaining Task Board
| # | Task | Priority | Status |
|---|------|----------|--------|
| 1 | Add 30-40 quote-endquote examples | **HIGH** | Pending — datagen agent |
| 2 | Add 15-20 hard negatives (trigger words as natural speech) | HIGH | Pending — datagen agent |
| 3 | Standardize punctuation in quote training data | HIGH | Pending — datagen agent |
| 4 | Train with DoRA + AdamW + cosine decay on current data | HIGH | Ready to go |
| 5 | Try mask_prompt: false experiment | Medium | Quick A/B test |
| 6 | QLoRA experiment (4-bit base training) | Medium | Validate memory savings |
| 7 | Update results.html dashboard | Low | Pending |

## Context for Future
This session was primarily an audit and research session. The biggest discovery was data leakage in the spoke benchmarks — our "58% baseline" was actually 38% on clean data, and our "83% fine-tuned" was 75% clean. This reframes the entire experiment history: fine-tuning gained +50 points (25%→75%), not +25 as previously thought. The v2 prompt alone accounts for half that gain (25%→50%).

The research phase identified several free config improvements (DoRA, AdamW, cosine decay) and confirmed that data quality is the primary bottleneck. The next high-impact move is: (1) get more quote-endquote examples + hard negatives from datagen agent, (2) train with improved config on updated data, (3) benchmark with clean methodology. Building on `2026-03-01_0030_lora-rank-experiments.md`.
