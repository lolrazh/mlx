# Agent Log: Gemma 3n E4B v5 Experiments

**Date:** 2026-03-07
**Status:** Completed

## User Intention
User wanted to test Gemma 3n E4B with v5 data and 2000 steps (up from v4/1200), with seq_length 512. Hypothesis: v5 targeted data + more training could push Gemma 3n closer to Qwen3's 100% core / 69% broad.

## What We Accomplished
- Investigated why only Qwen3 was doing well — discovered most models got fewer optimization passes
- Confirmed cloud ≈ local parity was already validated (finding #50, #58)
- Trained Gemma 3n E4B on v5 data, 2000 steps, Google hyperparams (lr=2e-4, constant_with_warmup, warmup=0.03, max_grad_norm=0.3, wd=0.01, r=16), seq_length 512
- Benchmarked merged model (best checkpoint = epoch 1.2 via load_best_model_at_end)
- Ran broad58 eval

## Training Runs & Results

| Model | Data | Steps | eval_loss (best) | Core23 | Broad58 | Latency |
|-------|------|-------|-----------------|--------|---------|---------|
| Gemma 3n E4B v4 (prev) | v4 (1201) | 1200 | 0.643 @500 | **96%** | **59%** | 1.20s |
| Gemma 3n E4B v5 (best ckpt, epoch 1.2) | v5 (1287) | 2000 | 0.595 @100 | **83%** | **64%** | 0.62s |
| Gemma 3n E4B v5 (step 2000) | v5 (1287) | 2000 | 0.595 @100 | **83%** | **64%** | 0.57s |

## Bugs & Issues
1. **Modal `--detach` required** — without it, local CLI timeout kills the remote training. First run died at step 173. Re-launched with `modal run --detach`.
2. **`load_best_model_at_end` selected epoch 1.2** — barely trained model. eval_loss was noisy (0.59-0.91 oscillation over 24.7 epochs). Step 2000 benchmark pending via merge script.
3. **Checkpoint config.json incomplete** — adapter checkpoints don't have `model_type` key, can't be loaded directly by AutoConfig. Must use merge script to combine with base model.

## Key Learnings
- **V5 interference confirmed on 3rd model**: Gemma 3n E4B 96% → 83% core (-13 pts). Only Qwen3 4B immune.
- **V5 helps broad even while hurting core**: 59% → 64% broad (+5 pts). Disfluency 0/4 → 1/4.
- **Trade-off is bad**: -13 core for +5 broad. Better to improve data quality (v6) than add harder examples (v5).
- **24.7 epochs is massive overtraining for Gemma 3n** with lr=2e-4. Previous run (1200 steps) = ~12 epochs, already plateaued.
- **Always use `modal run --detach`** for training runs > 10 minutes.

## T5Gemma 2 EOS Fix (earlier this session)
Also completed T5Gemma 2 1B-1B experiments:
- Discovered Gemma tokenizer doesn't append EOS to labels → 0% accuracy (degenerate repetition)
- Fix: explicitly append eos_token_id → 70% accuracy
- Phase C (encoder-decoder) concluded: 70% ceiling, 30 pts behind decoder-only

## Step 2000 Results
Step 2000 merged and benchmarked — **identical** to best checkpoint:
- Core23: 83% (19/23), 0.57s — same 4 failures (Kadaai, Kibbin Nayeh, 2x emphasis rewrites)
- Broad58: 64% (37/58), 0.67s — same 21 partials

Model converges very early (~epoch 1.2) and additional training neither helps nor hurts. The v5 interference is a data problem, not a training duration problem.

## Context for Future
- V5 interference confirmed at both early and late checkpoints — it's the data, not training duration
- Next logical test: Gemma 3n E4B on v4 data + 2000 steps (keep 96% core, test if longer training helps broad)
