# LFM2.5-1.2B Training, Model Exploration & Tooling Setup

**Date:** 2026-03-02
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed
**Building on:** 2026-03-02_0030_lfm2-83-percent.md

## User Intention
Continue Phase B model exploration by training and benchmarking LFM2.5-1.2B (the smallest LFM2 variant) and baselining Llama 3.2 3B. The broader goal is answering "what's the best model for Spoke?" by testing across architectures (transformer vs hybrid conv+attention) and scales (1.2B to 4B). Also: set up live training visualization, audit the experiment queue, and research T5Gemma encoder-decoder viability.

## What We Accomplished
- ✅ **Llama 3.2 3B zero-shot baseline (B16)** — 26% accuracy, 1.55s latency. Lands between Qwen3-4B (35%) and LFM2 (9%).
- ✅ **LFM2.5-1.2B LoRA training (LFM2.5-T1)** — 1000 iters, all 16 layers, 5.5M trainable params. Val loss plateaued at ~0.51 from iter 400. No double-descent.
- ✅ **LFM2.5-T1 benchmark** — **70% bf16, 0.63s latency, 0 fails.** 9% → 70% (+61 pts). 4.2x faster than Qwen3. Phone-deployable at ~1.2 GB.
- ✅ **wandb integration** — `pip install wandb`, logged in, added `report_to: wandb` to config.yaml. Live loss curve dashboards for all future training runs.
- ✅ **T5Gemma viability research** — **BLOCKED.** mlx-lm has 111 model architectures, all decoder-only. Zero encoder-decoder support. No T5Gemma on mlx-community. Phase C requires upstream changes.
- ✅ **rsLoRA research** — Explained mechanics, confirmed mlx-lm uses raw scale (not alpha/r internally). rsLoRA is a config-only change: r=16, scale=4.0.
- ✅ **LEDGER updated** — LFM2.5-T1 results, B16 baseline, findings #38-40, model comparison table, queue reprioritized.
- ✅ **MEMORY.md updated** — Removed stale refs, added wandb, LFM2 results, T5Gemma blocker, rsLoRA mechanics, new lessons.

## Technical Implementation

### LFM2.5-1.2B Training
- Config: r=8, scale=2.0, adam, lr=1e-5, batch=4, 1000 iters, all 16 layers
- 5.554M trainable / 1170.341M total (0.475%)
- Peak memory: 6.5 GB (half of LFM2-2.6B's 13.3 GB)
- Training speed: ~0.45 it/sec
- Val loss curve: 2.279 → 0.859 (iter 50) → 0.506 (iter 550, best) → 0.530 (iter 1000, final)
- Plateaued at iter 400 with no second drop (unlike 2.6B's double-descent)
- Iter 1000 checkpoint scored higher than iter 500 (70% vs 65%) despite worse val loss — confirms val loss unreliability

### Key Benchmark Comparison
| Model | Params | Zero-shot | Fine-tuned | Latency | Peak Mem |
|-------|--------|-----------|------------|---------|----------|
| Qwen3-4B (T11) | 4B | 35% | 83% | 2.67s | ~14 GB |
| LFM2-2.6B (T1b) | 2.6B | 9% | 83% | 1.66s | 13.3 GB |
| LFM2.5-1.2B (T1) | 1.2B | 9% | 70% | 0.63s | 6.5 GB |
| Llama 3.2 3B | 3B | 26% | — | — | — |

**Files Modified:**
- `spoke/config.yaml` — Added `report_to: wandb`, pointed at LFM2.5-1.2B
- `spoke/LEDGER.md` — LFM2.5-T1 results, B16 baseline, findings #38-40, model comparison table, queue updates
- `spoke/bench/run_benchmark.py` — Added `lfm2.5-1.2b-bf16` and `llama3.2-3b` model entries (previous session)
- `spoke/bench/result_llama3.2-3b_v2.json` — New baseline result
- `spoke/bench/result_lfm2.5-1.2b-bf16+lora_v2.json` — LFM2.5-T1 benchmark result

**Files Created:**
- `spoke/adapters-lfm2.5-t1/` — Training checkpoints (100-1000 every 100 iters)
- `spoke/adapters-lfm2.5-t1-best/` — Best checkpoint snapshot (iter 500, val loss 0.512)

## Bugs & Issues Encountered
1. **Laptop hung running training + model download simultaneously** — LFM2.5-T1 training (6.5 GB peak) + Llama 3.2 3B download (6 GB) overwhelmed 24 GB M4.
   - **Fix:** Killed both tasks, cleaned up partial training data, ran sequentially (one at a time).
2. **Previous background tasks already dead after hang** — `TaskStop` returned "No task found" since processes died when laptop hung.
   - **Fix:** Checked for partial artifacts manually (adapters dir, HF cache), cleaned up. Llama download had actually completed before the hang.
3. **Iter 1000 beat iter 500 despite worse val loss** — 70% vs 65% accuracy, but val loss 0.530 vs 0.512.
   - **Lesson:** Val loss is unreliable with 20 validation examples (finding #19 confirmed again). Always benchmark accuracy directly.

## Key Learnings
- **mlx-lm has built-in wandb callback** — Just `report_to: wandb` in config.yaml. Also supports `swanlab`. No custom code needed.
- **mlx-lm scale parameter is the raw multiplier** — Not alpha. No internal `alpha/r` computation. rsLoRA = just change the config value (r=16, scale=4.0 for alpha=16).
- **`python -m mlx_lm.lora` is deprecated** — Use `mlx_lm.lora` directly.
- **mlx-lm has zero encoder-decoder support** — 111 model files, all decoder-only. T5Gemma evaluation is blocked. mlx-examples has a standalone T5 but it's not integrated into the fine-tuning pipeline.
- **T5Gemma sizing: 4B-4B = ~8B total** (too much for 24 GB), 1B-1B = ~2B total (would fit). Moot since blocked.
- **Don't run training + large downloads simultaneously on 24 GB** — Memory pressure from both can hang the system.
- **LFM2.5-1.2B has no double-descent** — Unlike the 2.6B which showed plateau→second-drop. Smaller model lacks capacity for a second learning phase.

## Architecture Decisions
- **Sequential execution over parallel** — After the laptop hang, switched to running Llama baseline first (quick, model already cached), then LFM2.5 training. Safer on 24 GB.
- **wandb over custom plotting** — mlx-lm has native wandb support. Zero code, free account, live dashboards. Better than building a custom script.
- **Reprioritized experiment queue** — Moved T11-ext (2000 iters) and rsLoRA to HIGH priority over Q1 (mixed-bit quant). Rationale: nail the model choice first, quantize the winner later.
- **T-enc marked BLOCKED** — Rather than attempting a custom T5Gemma port, acknowledged the blocker and moved on. Better use of time on achievable experiments.

## Ready for Next Session
- ✅ **wandb configured** — Next training run will have live loss curves at wandb.ai
- ✅ **LFM2.5-T1 benchmarked** — 70% established, best checkpoint saved
- ✅ **Llama 3.2 3B baselined** — 26% zero-shot, model cached (~6 GB)
- 🔧 **T11-ext (Qwen3-4B 2000 iters)** — Highest priority next experiment. T11 only trained 300 iters (~2.4 epochs). Config needs updating back to Qwen3-4B.
- 🔧 **rsLoRA experiment** — Config change only (r=16, scale=4.0). May unlock r=16 benefit.
- 🔧 **Llama 3.2 3B fine-tune** — Baselined but not trained. Could be interesting comparison.

## Context for Future
Phase B model exploration is nearly complete. We now have a clear picture: LFM2-2.6B ties Qwen3-4B at 83% (faster), LFM2.5-1.2B hits 70% (much faster, phone-sized). The next frontier is squeezing more from Qwen3-4B (T11 only saw 2.4 epochs — likely undertrained) and testing rsLoRA. Phase C (encoder-decoder) is blocked by mlx-lm limitations. The biggest remaining lever per the research doc is "better data from a stronger teacher" (3-5K distilled examples).

## Commits This Session
- `15086dd` — Add wandb live dashboards, Llama 3.2 3B baseline (26%), LFM2.5-T1 config
- `12ce713` — LFM2.5-T1: 70% bf16 on 1.2B model (0.63s, 4.2x faster than Qwen3)
