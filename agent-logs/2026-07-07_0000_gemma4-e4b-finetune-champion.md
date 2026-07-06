# Gemma 4 E4B Fine-Tune: New Accuracy Champion (via a 3-part load-bug fix)

**Date:** 2026-07-07 (session spanned 2026-07-06 evening → 07-07)
**Agent:** Claude Opus 4.8 (1M context)
**Status:** Completed — new accuracy champion trained + benchmarked; checkpoint sweep in progress

## User Intention
User flagged Gemma 4 E4B-it as a bake-off omission ("did we even explore hyperparameters? maybe we should've done research into how to fine-tune that specific model because it is different from Qwen"). Goal: benchmark E4B zero-shot, then fine-tune it with the proper Gemma-family recipe and see if it challenges the Qwen3-4B champion (100 core / 74 broad / 82 v5) at comparable size (~2 GB).

## What We Accomplished
- **Zero-shot sweep** (B49–B55): 5 prompt modes × core23 + broad58 on `mlx-community/gemma-4-E4B-it-qat-4bit`.
- **Enabled Gemma 4 fine-tuning** across the Modal pipeline (`train_hf.py`, `benchmark.py`, `merge_adapter_checkpoint.py`) after a genuinely tricky 3-part load bug.
- **Trained the full run** (`spoke-gemma4-e4b-v5-2k-20260706`, v5-split, 2000 steps) → **new accuracy champion**.
- **Benchmarked** vs the Qwen champion on all 3 test sets.
- Updated LEDGER (finding #106, champion line, benchmark rows) + 6 bite-sized commits.

## Zero-shot results (mlx 4-bit, local M4)

| Prompt | core23 | broad58 | note |
|--------|--------|---------|------|
| mini | 17% | — | terse prompt collapses small model |
| generic | 22% | — | |
| v2 | 39% | 34% | 22 pts below the 26B on v2 |
| v3 | 57% | — | |
| **spoke-full** | **61%** | **45%** | best; 44-pt prompt swing = capacity signature |

Folds 20 pts vs the 26B MoE on broad (45 vs 66). Zero-shot ≠ fine-tune potential (finding #63) — the fine-tune is the real test.

## The 3-part fine-tune bug (why Gemma 4 "is different")
Gemma 4's multimodal checkpoint nests the text decoder under `model.language_model.*`. Each fix was caught by a ~$0.50 smoke:

1. **Text-only load is impossible.** Neither `AutoModelForCausalLM(config=text_config)` nor the dedicated `Gemma4ForCausalLM.from_pretrained` remaps the `language_model.` prefix in transformers 5.5.2 → every decoder weight MISSING → random init → loss frozen at ln(vocab)≈12.9, grad_norm ~6000. Fix: load the **FULL `Gemma4ForConditionalGeneration`** (keys match → 0 MISSING). gemma3n never hit this — it stores text weights flat.
2. **torch 2.6 too old.** peft 0.19.0's `cast_adapter_dtype` references `torch.float8_e8m0fnu` (added torch 2.7). Fix: isolated `gemma4_image` = debian_slim + `torch==2.8.*` + transformers 5.5.2 + peft 0.19.0 (accelerate resolved 1.14.0). Shared `standard_image` stays at torch 2.6.
3. **Adapter surface.** `target_modules=None` gave PEFT's q/v-only default (4.5M params). Fix: a **`language_model`-scoped regex** `.*language_model.*\.(q_proj|...|down_proj)` restores the proven 7-module recipe (34.9M, 0.44%) AND excludes the vision/audio `Gemma4ClippableLinear` towers (an explicit list would walk them and crash).

**Recipe (our validated Gemma-family one, NOT the blog's lr=1e-4):** lr=2e-4, constant_with_warmup, warmup=0.03, max_grad_norm=0.3, wd=0.01, r=16, α=16, dropout=0.05, adam, 2000 steps, v5-split (FULL data, with emoji — 1046 rows, 106 emoji).

**Ops lessons (cost 2 relaunches):** two overlapping `modal run` clients on the same App name collide (`ConflictError APP_STATE_STOPPED`); piping `modal run` through `head` SIGPIPE-kills it; flaky local network cancels attached runs. Rule: one clean client, no head, no overlap, retry loops.

## Fine-tune result — NEW ACCURACY CHAMPION (best ckpt 800, v2 greedy)

| test set | Gemma 4 E4B | Qwen champion | Δ | hard fails |
|----------|-------------|---------------|-----|-----------|
| **broad58** (58) | **79.3%** | 74.1% | **+5.2** | **0 vs 2** |
| **v5-131** (131) | **86.3%** | 81.7% | **+4.6** | 0 vs 0 |
| core23 (23) | 95.7% | 100% | −4.3 (~1 ex) | 0 vs 0 |
| latency (L40S) | 0.9–1.1s | 0.24–0.27s | 3–4× slower | — |

- **0 hard fails across all 231 test examples.** Wins the two statistically-weighty sets, loses ~1 example on the smallest/easiest.
- Gen-4 PLE is a **+15 broad jump** over the gemma3n E4B sibling (79 vs 64) on IDENTICAL data/recipe — architecture, not recipe.
- Temp-parity confirmed non-issue (champion identical greedy vs t0.6 on v5, 81.7 both).
- Converged early (best eval_loss 0.073 @ step 800), overfit after (eval 0.07→0.16 by step 2000).

## Caveats (in the ledger)
1. **3–4× slower on L40S** (M4 local latency after MLX conversion is unmeasured — do not over-weight the cloud number).
2. **Deployment unproven:** merged artifact is a full multimodal HF checkpoint; shipping locally needs a text-decoder → MLX 4-bit conversion (mlx-lm gemma4 support unvalidated for our checkpoint).
3. **Single-seed:** +4/+5 margins are ~3–5 examples — real but a 2–3 seed confirm would harden it.

## Dead ends / don't-benchmark
- `spoke-gemma4-e4b-smoke`, `-smoke2` — random-init garbage from the pre-fix load bug.
- `-smoke3c` — the good smoke (q/v-only, proved the pipeline).

## Next steps
- **In progress:** checkpoint sweep — merging + benchmarking step 2000 (finding #93: best-eval-loss ≠ best-benchmark; step 2000 may beat step 800).
- Queued: 2–3 seed confirm; validate MLX conversion + measure real M4 latency; v6 data on this stronger base.
- Separate: emoji-ablation-confirm (Qwen noemoji retrain) is still valid but NOT implied by this result — Gemma trained on FULL data.

## Commits (bite-sized, chronological)
1. E4B zero-shot sweep results
2. train_hf.py gemma4 enablement
3. benchmark.py gemma4 support
4. merge_adapter_checkpoint.py gemma4 support
5. Fine-tune benchmark results (new champion)
6. LEDGER (finding #106 + champion line)
7. This log
