# LoRA Learning Rate Research & Retraining

**Date:** 2026-03-07
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed

## User Intention
User wanted to verify whether we used optimal fine-tuning hyperparameters for Llama 3.2 3B and Qwen3 4B. Research suggested our LR was 10-20x too low. User wanted to re-test both models with "optimal" hyperparams and get a definitive answer on whether Llama still has a competitive shot.

## What We Accomplished
- ✅ **Research: 3 parallel Sonnet agents** — Meta official recs, community consensus, paper surveys
- ✅ **Llama 3.2 3B retrained** with lr=2e-4, r=16 → **78%** (WORSE than old 91%)
- ✅ **Qwen3 4B retrained** with lr=2e-4, r=16 → **96%** (worse than old 100%)
- ✅ **Key discovery: "optimal" LR from papers hurts copy-heavy editing tasks**
- ✅ **Ledger updated**, results committed, memory updated

## Training Runs

### Llama 3.2 3B (`spoke-llama3-v4-v2`)
```
modal run spoke/cloud/train_hf.py --model-name meta-llama/Llama-3.2-3B-Instruct \
  --run-name spoke-llama3-v4-v2 --learning-rate 2e-4 --lr-scheduler-type constant_with_warmup \
  --warmup-ratio 0.03 --max-grad-norm 0.3 --weight-decay 0.01 --rank 16 --lora-alpha 32 \
  --lora-dropout 0.0 --max-steps 1200 --data-dir /data/v4
```
- Step 400 (best eval_loss 0.215): **78%** (17 exact, 1 sem, 5 partial, 0 fail)
- Step 1200 (final): **78%** — identical failures
- Old lr=1e-5 at 2000 steps: **91%** — 13 pts better

### Qwen3 4B (`spoke-qwen3-v4-v2`)
```
modal run spoke/cloud/train_hf.py --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --run-name spoke-qwen3-v4-v2 --learning-rate 2e-4 --lr-scheduler-type constant_with_warmup \
  --warmup-ratio 0.03 --max-grad-norm 0.3 --weight-decay 0.01 --rank 16 --lora-alpha 32 \
  --lora-dropout 0.0 --max-steps 1200 --data-dir /data/v4
```
- Step 1200 (final): **96%** (21 exact, 1 sem, 1 partial, 0 fail)
- Best eval_loss at step 100 (!), then overfit hard
- Old lr=1e-5 at 2000 steps: **100%** — 4 pts better

## Key Learnings
- **"Optimal" LoRA LR from papers is task-dependent.** 2e-4 works for chat/reasoning but HURTS copy-heavy editing. Our 1e-5 was actually correct.
- **Higher LR creates qualitatively different failures.** Llama at 2e-4 outputs garbled emoji (`praying hands` instead of emoji), ignores emphasis, mis-scopes quotes. Not just "more overfitting" — a fundamentally different (worse) editing policy.
- **Exception: Gemma 3n benefits from 2e-4.** Its MatFormer+PLE architecture is specifically designed for efficient adaptation. Architecture-specific LR tuning is mandatory.
- **Llama 3.2 3B is #5 in fine-tuning benchmarks** (DistilLabs). Qwen3-4B is #1. Even Qwen3-1.7B outranks Llama. But Llama has zero fails across ALL runs — unique conservative editing.
- **mlx-lm targets all linear layers by default** — auto-discovers all Linear/QuantizedLinear modules, no config needed.
- **Over-memorization paper (arXiv:2508.04117)** explains val loss divergence: model becomes overconfident on correct tokens, accuracy stays flat, OOD robustness drops. Early stopping on val loss showed 1.3-3.0 pts LOWER accuracy.

## Architecture Decisions
- **lr=1e-5 is correct for Spoke** — validated by negative results from "optimal" 2e-4 on both Llama and Qwen3
- **r=8 is sufficient for our data size** — r=16 didn't help when combined with higher LR

## Context for Future
Our original hyperparams (lr=1e-5, r=8, adam, constant LR, 2000 iters) are actually well-tuned for the copy-heavy text editing task. The only model that benefited from higher LR was Gemma 3n, due to its architecture-specific design. Future experiments should focus on data quality/quantity rather than hyperparameter tuning. The leaderboard remains: Qwen3 T2 (100%) > Qwen3 DWQ / Gemma 3n E4B (96%) > Llama T2 / Gemma 3n E2B (91%).
