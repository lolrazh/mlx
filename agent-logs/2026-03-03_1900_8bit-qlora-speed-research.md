# 8-bit QLoRA, Training Speed Research, and Cloud Pipeline

**Date:** 2026-03-03
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed
**Continues from:** `2026-03-03_1530_llama3-t2-training.md`

## User Intention
After completing the DWQ deploy model (96%, 2.1 GB) and Llama3-T2 (91%), user wanted to speed up the iteration cycle. The initial hypothesis was that 8-bit QLoRA would be faster than bf16 LoRA — this turned out to be wrong (8-bit is actually slower due to dequant overhead). This led to a deep research sprint on MLX training optimizations and cloud fine-tuning options, uncovering several actionable speedups and the discovery that Muon optimizer is already available in mlx-lm.

## What We Accomplished
- ✅ **8-bit QLoRA benchmark** — 96% accuracy at iter 800 (killed early). Confirmed 8-bit ceiling is 96% vs bf16's 100%.
- ✅ **8-bit zero-shot validation** — 8-bit = bf16 at zero-shot (both 22%). No precision loss at zero-shot level.
- ✅ **MLX speed research** — Comprehensive analysis of local training optimizations. Key finding: M4 is memory-bandwidth-bound at 120 GB/s. Realistic ceiling is ~3-3.5x via stacking optimizations.
- ✅ **Cloud fine-tuning research** — Full pricing/workflow comparison of Modal, Fireworks AI, DigitalOcean, Baseten, Vast.ai, RunPod.
- ✅ **Muon optimizer discovery** — Already wired up in mlx-lm 0.30.5 as `optimizer: muon`. Just a config change. Needs higher LR (~0.02 range).
- ✅ **Data token length analysis** — v4 data max is 227 tokens, median 132. max_seq_length=512 was wasting 56% compute on padding.
- ✅ **Speed test config prepared** — bf16 + max_seq_length=256 + num_layers=16 + new adapter path. Expected ~3x speedup.
- ✅ **Quantization format ELI5** — Explained Int4 vs NF4 vs GPTQ vs DWQ differences (user asked about Qwen 3.5 GPTQ-Int4 announcement).
- ✅ **Ledger updated** — Finding #48 (8-bit QLoRA slower than bf16), B19 baseline, Muon queue entry corrected, completed queue updated.
- ✅ **Memory updated** — All MLX speed insights, cloud provider details, Muon availability saved.

## Technical Implementation

**8-bit QLoRA training (killed at ~iter 850):**
```
Starting val loss: 6.110 (vs bf16's 2.843 — 2x higher due to dequant noise)
Final val loss: 0.276 @iter 800
Peak memory: 16.4 GB (vs 18.6 GB bf16 — only 12% savings)
Speed: ~0.12 it/sec (vs ~2 it/sec bf16 — 17x SLOWER)
Accuracy: 96% (21 exact, 1 semantic, 1 partial)
```

**Token length analysis:**
```
Full sequence: Min=106, Max=227, Mean=134, Median=132
  Examples > 256: 0/1201 (NONE exceed 256)
  Examples > 192: 1/1201
Assistant tokens only (with mask_prompt): Mean=13, Median=12
```

**Speed test config (committed, not yet launched):**
```yaml
model: mlx-community/Qwen3-4B-Instruct-2507-bf16  # back to bf16
max_seq_length: 256   # was 512 (56% wasted)
num_layers: 16        # was 36 (test if top-half suffices)
adapter_path: ./spoke/adapters-qwen3-speed
```

**MLX version status:**
- Installed: mlx 0.30.6, mlx-lm 0.30.5
- Latest: mlx 0.31.0, mlx-lm 0.30.7
- Muon optimizer: present in `mlx.optimizers.Muon` and wired up in `mlx_lm/lora.py`

**Files Modified:**
- `spoke/config.yaml` — Updated to speed test config (bf16, 256 seq, 16 layers)
- `spoke/LEDGER.md` — Finding #48, B19 baseline, Muon queue corrected, 8-bit QLoRA completed
- `spoke/bench/result_Qwen3-4B-Instruct-2507-8bit+lora_v2.json` — 8-bit QLoRA benchmark (96%, new)
- `MEMORY.md` — MLX speed insights, cloud provider details, Muon availability

## Bugs & Issues Encountered
1. **Benchmark script expects directory, not file for adapter path** — Same issue as Llama3-T2 iter 700. `run_benchmark.py --adapter-path` needs a directory with `adapter_config.json`.
   - **Fix:** Created `/tmp/qwen3-8bit-iter800/` with checkpoint renamed to `adapters.safetensors` + copied `adapter_config.json`.
2. **Benchmark `--test-set v3` doesn't work** — Script expects full path, not shorthand.
   - **Fix:** Used `--test-set spoke/bench/test_set_v3.json` instead.

## Key Learnings
- **8-bit QLoRA is SLOWER than bf16 LoRA for training.** Every matmul requires dequantizing 8-bit→bf16 first. Confirmed by MLX maintainer Awni Hannun: "If you have enough RAM for regular LoRA it can be faster." QLoRA is a memory optimization, NOT a speed optimization.
- **max_seq_length waste is massive for short tasks.** With median 132 tokens and max 227, setting 512 means 60-70% of every batch is padding that still flows through attention + MLP. Cutting to 256 is a free ~1.5x speedup.
- **Muon is already in mlx-lm.** `optimizer: muon` in config.yaml works. Needs much higher LR (~0.02 vs 1e-5). Uses Newton-Schulz orthogonalization on momentum. Safe for LoRA (all 2D matrices). Research claims 2x compute efficiency and better quant robustness.
- **MLX training is memory-bandwidth-bound on M4 (120 GB/s).** No software optimization can bypass this. M4 Pro=273, M4 Max=546 GB/s. The bandwidth gap is the real 2-4.6x multiplier.
- **`mx.compile` is already active in mlx-lm's trainer.** The training step function is compiled. No additional flags to enable.
- **Cloud GPU fine-tuning is pennies per run.** Vast.ai A100 + Unsloth = $0.02-0.04/run. Modal L40S = $0.55/run. The hybrid workflow (cloud train → fuse → mlx_lm.convert → DWQ locally) is fully supported.
- **Int4 = 4-bit integer.** GPTQ-Int4 (GPU ecosystem) and our DWQ 4-bit (MLX) are different algorithms solving the same problem. Both are learned quantization, both crush naive RTN.

## Architecture Decisions
- **Killed 8-bit training at iter 800** — Val loss had plateaued at ~0.28, and 800 iter checkpoint gives enough data to compare. No need to burn another 30 min for marginal improvement.
- **Speed test uses 16 layers, not 8** — T11 (16 layers, v3 data) hit 83%. With v4's stronger data, 16 layers should be enough for 96%+. Going to 8 would be too aggressive given we haven't tested 16 on v4 yet.
- **Muon experiment planned for YOLO run** — User chose to combine speed optimizations + Muon in one run rather than isolating variables. Pragmatic: if it works, great. If accuracy drops, we can untangle which change caused it.

## Ready for Next Session
- ✅ **Speed test config committed** — `spoke/config.yaml` ready to launch with bf16 + 256 seq + 16 layers
- ✅ **Muon available** — Just add `optimizer: muon` and adjust LR (~0.02)
- ✅ **8-bit QLoRA data point logged** — 96% at 800 iters, slower than bf16. Case closed.
- ✅ **Cloud pipeline researched** — Modal + Unsloth is the recommended path. User has credits on Fireworks ($100), DigitalOcean ($200), Baseten ($500).
- 🔧 **Launch speed test** — User said "don't launch yet" — wants to also add Muon to the config first
- 🔧 **Upgrade mlx/mlx-lm** — 0.31.0 / 0.30.7 available. Do after current experiments.

## Context for Future
The project pivoted from "test 8-bit for faster training" to "optimize the entire training pipeline." Key insight: bf16 is already faster than 8-bit QLoRA on hardware that fits both. The real speedups come from reducing waste (shorter sequences, fewer layers) and better optimizers (Muon). Cloud GPU is the escape hatch for true 10x+ speedup at pennies per run. Next immediate step: launch bf16 + short seq + 16 layers + Muon as a combined YOLO test to see how fast training can go while maintaining accuracy.
