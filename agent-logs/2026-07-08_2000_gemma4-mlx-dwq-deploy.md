# Gemma 4 E4B → MLX: Deployment CLOSED + DWQ 4-bit is the Deploy Pick

**Date:** 2026-07-08 (continues the same-day port session)
**Agent:** Claude Opus 4.8 (1M context)
**Status:** DONE — champion ported, quantized, benchmarked, DWQ-optimized. Deploy = DWQ 4-bit (3.9 GB, 100/75.9/84.7, ~0.79s / 0.62s cached). See LEDGER finding #108.

## The arc (full detail in LEDGER #108)
1. **Conversion trivial** — mlx_lm 0.31.3 has native `gemma4`; `sanitize()` drops vision/audio towers → `mlx_lm convert` eats the merged bf16 directly.
2. **The model was secretly THINKING** — Gemma 4's chat template opens a hidden `<|channel>thought` channel; the LoRA-tuned model still slips into it on hard examples under MLX greedy decode, tanking broad58. **Fix: 2-token logit bias (ban `<|think|>`=98, `<|channel>`=100)** → broad58 63.8→82.8, = cloud bf16 exactly. Patched into `run_benchmark.py` (self-gating).
3. **Quantization was NOT the thinking cause** — bf16-in-MLX thinks too. The whole gap was the serving path.
4. **Quant sweep (all + no-think bias):** 4-bit 95.7/69.0/83.2 (real broad cliff), 5-bit 95.7/82.8/84.0, 6-bit=8-bit 100/82.8/86.3, mixed_4_6 95.7/74.1/84.0.

## DWQ 4-bit — the winner (this session's payoff)
Distilled Weight Quantization: optimizes the 4-bit scales/biases to match the bf16 teacher on **task-specific calibration** (256 spoke examples, v2 prompt). Validation loss **0.335 → 0.039**.

| model | size | core | broad | v5 | latency |
|---|---|---|---|---|---|
| plain 4-bit | 3.9 GB | 95.7 | 69.0 | 83.2 | ~0.79s |
| mixed_4_6 | 4.1 GB | 95.7 | 74.1 | 84.0 | 0.80s |
| **DWQ 4-bit** | **3.9 GB** | **100.0** | **75.9** | **84.7** | **0.79s / 0.62s cached** |
| 6-bit | 5.7 GB | 100 | 82.8 | 86.3 | ~0.93s |
| cloud bf16 | 15.9 GB | 95.7 | 82.8 | 88.5 | (L40S) |

**DWQ 4-bit DOMINATES mixed_4_6 on every axis** (smaller + more accurate + marginally faster) → new deploy pick, mixed_4_6 deleted. Still −6.9 broad58 vs 6-bit, but this was a QUICK DWQ (256 of 1046 samples, one short pass) — broad58 75.9 is a floor. **More calibration (512–1024 samples, longer train) should push it toward 6-bit's 82.8 at the same 3.9 GB.**

## DWQ gotchas (cost several failed runs — worth keeping)
1. **`python -m mlx_lm.quant.dwq` does NOTHING** — the module has `def main()` but no `if __name__=="__main__"` block, so `-m` imports and exits 0 silently. Use the console script **`mlx_lm.dwq`** (entry_points → `:main`). Same for `mlx_lm.awq`.
2. **Two-phase target coupling** — `--target-dir` targets are baked to a specific `(batch_size, max_seq_length)`. Phase 2 MUST use the SAME `--batch-size` and `--max-seq-length` as Phase 1, or `loss_fn` throws `broadcast_shapes (1,160) vs (4,128)`. Recompute targets if you change either.
3. **Phase 2 OOMs** (`[METAL] Insufficient Memory`) at batch 4 / no grad-ckpt — trains 233 M scale/bias params + Adam + activations. **Fix: `--grad-checkpoint` + small batch + `--max-seq-length 256`** (our seqs are ~125 tok). Two-phase avoids teacher+student co-residence (peak = max(15 GB teacher pass, ~5 GB student)) — but the training pass still needs the memory knobs.
4. **Local calibration data** — HF dir-loader chokes if the dir has mismatched-schema jsonls; make a clean dir with ONLY `train.jsonl` (messages format).
5. **Don't pass args via a shell `$CFG` var** — word-splitting failed → argparse "unrecognized arguments" (exit 2). Inline them.

## Deployment config (the recipe)
- Model: `spoke/models/g4e4b-champion-mlx-dwq4-g64` (3.9 GB, gitignored)
- Prompt: v2 (trained-on), greedy
- **REQUIRED at inference: no-think logit bias** (ban token 98 + 100) — without it broad58 drops ~17 pts
- Prompt-cache the fixed v2 system prompt (~15% latency); real deploy = `mlx_lm.server` warm cache

## NEXT (both need a clear machine — bf16 pass overflows under load)
1. **Bigger DWQ** (1024 samples + longer train) → chase 6-bit's broad58 at 3.9 GB.
2. **Speculative decoding** → the <0.5s lever (we're at 0.62s cached). Needs a Gemma-4-E2B draft model (shared 262144 vocab), 4-bit.
3. llama.cpp-vs-MLX still BLOCKED (Gemma 4 PLE unimplemented, issue #22243).
