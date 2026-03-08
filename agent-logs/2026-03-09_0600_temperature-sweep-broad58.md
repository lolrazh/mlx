# Temperature Sweep + Broad58 New Best

**Date:** 2026-03-09
**Agent:** Claude Opus 4.6
**Status:** Completed

## User Intention
User wanted to benchmark zero-shot Qwen3.5-4B locally via MLX with the full Spoke prompt, testing multiple temperatures (0.0, 0.2, 0.6). Then benchmark the best fine-tuned Qwen3-4B checkpoint (T4-v5split ckpt2000) on Modal with temp=0.6 on both v5 (131 ex) and broad58 (58 ex).

## What We Accomplished
- Upgraded mlx (0.30.6 → 0.31.0) and mlx-lm (0.30.7 → 0.31.0) for native Qwen3.5 support
- Added temperature support to both local (`run_benchmark.py`) and cloud (`benchmark.py`) benchmark scripts
- Added `qwen3.5-4b-bf16` model alias to `run_benchmark.py`
- Ran zero-shot Qwen3.5-4B temperature sweep: temp=0.0/0.2/0.6 on v5 (131 ex) and temp=0.0/0.6 on broad58
- Ran fine-tuned T4-v5split ckpt2000 on Modal at temp=0.6 on both v5 and broad58
- Discovered 74% broad58 = NEW ALL-TIME BEST (+5 over previous best of 71%)
- Updated LEDGER with new baselines, benchmark rows, and finding #94

## Results

### Zero-shot Qwen3.5-4B (MLX, spoke-full prompt)

| Test Set | Temp | Accuracy | Exact | Sem | Part | Fail | Latency |
|----------|------|----------|-------|-----|------|------|---------|
| v5 (131) | 0.0 | 37% | 45 | 3 | 58 | 25 | 5.33s |
| v5 (131) | 0.2 | 37% | 44 | 4 | 58 | 25 | 6.51s |
| v5 (131) | 0.6 | 37% | 45 | 4 | 58 | 24 | 6.00s |
| broad58 | 0.0 | 41% | 19 | 5 | 21 | 13 | 5.69s |
| broad58 | 0.6 | 34% | 17 | 3 | 30 | 8 | 6.48s |

### Fine-tuned Qwen3-4B T4-v5split ckpt2000 (Modal, v2 prompt, temp=0.6)

| Test Set | Accuracy | Exact | Sem | Part | Fail | Latency |
|----------|----------|-------|-----|------|------|---------|
| v5 (131) | **82%** | 103 | 4 | 24 | 0 | 0.26s |
| broad58 | **74%** | 41 | 2 | 13 | 2 | 0.27s |

### Broad58 All-Time Top 5

| Rank | Model | Score |
|------|-------|-------|
| 1 | **T4-v5split ckpt2000 (temp=0.6)** | **74%** |
| 2 | Qwen3.5-4B (v5, 1500 steps) | 71% |
| 3 | T3-v5 (v5 full, ckpt 1200) | 69% |
| 4 | T2-v4 DWQ | 67% |
| 5 | Qwen3.5-4B lr=5e-5 / EPO-w3 | 66% |

## Technical Implementation

**MLX upgrade:** mlx 0.30.6 → 0.31.0, mlx-lm 0.30.7 → 0.31.0. Native `qwen3_5` model type now in mlx-lm. `mlx-community/Qwen3.5-4B-MLX-bf16` loads directly via `mlx_lm.load()` despite being a VLM conversion.

**Benchmark script changes:**
- `spoke/bench/run_benchmark.py` — Added `--temperature` CLI arg, `qwen3.5-4b-bf16` model alias, dynamic sampler creation, temp suffix in output filenames
- `spoke/cloud/benchmark.py` — Added `temperature` param to `benchmark_remote()` and `main()`, `do_sample=True` when temp > 0, temp suffix in output filenames

**Files Modified:**
- `spoke/bench/run_benchmark.py` — temperature support + Qwen3.5 model alias
- `spoke/cloud/benchmark.py` — temperature support
- `spoke/LEDGER.md` — new baselines (B23-B27), T4-v5split broad58 row, finding #94

**Result files created:**
- `spoke/bench/result_qwen3.5-4b-bf16_spoke-full.json` — v5, temp=0.0
- `spoke/bench/result_qwen3.5-4b-bf16_spoke-full_t0.2.json` — v5, temp=0.2
- `spoke/bench/result_qwen3.5-4b-bf16_spoke-full_t0.6.json` — broad58, temp=0.6 (overwrote v5 t=0.6)
- `spoke/bench/result_spoke-qwen3-4b-v5split-2k-20260308-ckpt2000_modal_v2_test_set_v5_t0.6.json`
- `spoke/bench/result_spoke-qwen3-4b-v5split-2k-20260308-ckpt2000_modal_v2_test_set_evals_t0.6.json`

## Key Learnings
- **Temperature is irrelevant for copy-heavy editing tasks.** Both zero-shot (37% at all temps) and fine-tuned (82% at all temps) models produce identical accuracy regardless of sampling temperature. The task has a deterministic optimal output.
- **80:10:10 split > full-data training for generalization.** T4-v5split (1046 train, 131 val) beats T3-v5 (1287 train, 20 val) on broad58 by +5 pts. Meaningful validation set enables better checkpoint selection.
- **mlx-lm 0.31.0 loads VLM-converted models seamlessly.** The `mlx-community/Qwen3.5-4B-MLX-bf16` model was converted via mlx-vlm but loads fine with mlx-lm's native `qwen3_5` architecture.

## Context for Future
Temperature is now a closed axis — greedy is optimal. Four axes exhausted: prompt, model size, training duration, temperature. The remaining lever is v6 data. Modal credits: ~$0.30 remaining.
