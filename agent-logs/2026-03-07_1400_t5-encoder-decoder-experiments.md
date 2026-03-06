# T5 Encoder-Decoder Experiments

**Date:** 2026-03-07
**Agent:** Claude Opus 4.6
**Status:** 🔄 Ongoing (T5Gemma 2 next)

## User Intention
User wanted to test encoder-decoder architecture (T5) as an alternative to decoder-only models (Qwen3, Llama) for ASR post-processing. The hypothesis was that encoder-decoder is architecturally better for copy-heavy text editing (finding #25 in LEDGER). After initial Flan-T5 results, pivoted to T5Gemma 2 which has a modern 256K vocab (solves the emoji blocker).

## What We Accomplished
- ✅ **Created `spoke/cloud/train_t5.py`** — Modal cloud training script for T5/Flan-T5 encoder-decoder models (Seq2SeqTrainer, prefix format, full FT + LoRA support)
- ✅ **Modified `spoke/cloud/benchmark.py`** — Added `t5` prompt mode, `--checkpoint` flag, fixed no-chat-template crash for T5 tokenizers
- ✅ **Trained 4 T5 runs** — Flan-T5-base and Flan-T5-large at lr=3e-4 (v1) and lr=1e-5 (v2)
- ✅ **Benchmarked all runs** — 8 benchmark runs total (merged + checkpoints)
- ✅ **Zero-shot baseline** — Flan-T5-base = 17% on v3 test
- ✅ **Researched FlanEC hyperparams** — lr=5e-5, AdamW, batch=16, linear scheduler with 10% warmup
- ✅ **Updated LEDGER.md** — Training runs, benchmarks, model comparison, 7 new key findings (#66-72)

## Training Runs & Results

| Model | LR | Steps | eval_loss (best) | Accuracy | Latency |
|-------|-----|-------|-----------------|----------|---------|
| Zero-shot Flan-T5-base | — | — | — | **17%** | 0.13s |
| Flan-T5-base v1 | 3e-4 | 200/2000 | 0.249 @200 | **48%/57%** | 0.12/0.28s |
| Flan-T5-large v1 | 3e-4 | 200/2000 | 0.157 @200 | **70%/70%** | 0.22s |
| Flan-T5-base v2 | 1e-5 | 2000 | 1.152 @2000 | **17%** | 0.14s |
| Flan-T5-large v2 | 1e-5 | 2000 | 0.772 @2000 | **30%** | 0.22s |

## Bugs & Issues Encountered
1. **Benchmark crash on T5 tokenizer** — `enforce_no_thinking_chat_template()` called `apply_chat_template()` on T5 which has no chat template. Transformers 5.3.0 raises `ValueError`.
   - **Fix:** Added `if not template: return tokenizer` early return.
2. **Benchmark CLI validation error** — Passed both `--run-name` and `--model-name` which triggers mutual exclusivity check.
   - **Fix:** Use only `--run-name` for fine-tuned models.
3. **Checkpoint 1000 not found** — `save_total_limit=3` pruned intermediate checkpoints.
   - **Workaround:** Only benchmark merged (last step) and available checkpoints.

## Key Learnings
- **T5's 32K SentencePiece vocab can't generate emoji** — hard 13% ceiling (3/23 tests always fail). No amount of training fixes this.
- **Over-memorization pattern confirmed for encoder-decoder** — T5-base step 2000 (val_loss 0.574) was +9 pts better than step 200 (val_loss 0.249).
- **lr=1e-5 is way too low for full fine-tuning** — optimal for LoRA (0.2-0.4% params) but negligible per-param update for full FT (100% params). T5-base: 57% (3e-4) vs 17% (1e-5).
- **FlanEC paper (arXiv:2501.12979) used lr=5e-5** for full FT on exact same task (ASR error correction). This is the proven sweet spot between our overfitting 3e-4 and useless 1e-5.
- **Encoder-decoder doesn't outperform decoder-only (so far)** — T5-large 70% vs Qwen3 100%. But vocab limitation (emoji) and suboptimal LR were confounds.

## Architecture Decisions
- **Prefix format for T5**: "Correct this transcription: {input}" -> "{output}". No chat templates needed.
- **Full fine-tuning over LoRA for T5** — 220M-783M models are small enough, and full FT gives the model more freedom to adapt.
- **Pivoting to T5Gemma 2** — solves vocab blocker (256K Gemma tokenizer), modern architecture (Gemma 3 based), works with AutoModelForSeq2SeqLM + AutoProcessor.

## Ready for Next Session
- ✅ **`train_t5.py` tested and working** — can be adapted for T5Gemma 2
- ✅ **Benchmark infra supports enc-dec** — `prompt_mode="t5"` works
- 🔧 **T5Gemma 2 training script** — needs new script or major modifications (different tokenizer, model class, input format)
- 🔧 **Use FlanEC hyperparams** — lr=5e-5, AdamW, linear scheduler, 10% warmup

## Context for Future
T5 experiments confirmed encoder-decoder works for the task but Flan-T5's ancient 32K vocab is a hard blocker. T5Gemma 2 (1B-1B, ~1.7B total) with Gemma's 256K tokenizer is the next step — it removes the emoji ceiling and brings a modern architecture. Use FlanEC's proven recipe: lr=5e-5, AdamW, batch=16, linear with 10% warmup. The training script needs adaptation for T5Gemma 2's AutoProcessor and potentially different input formatting.
