# DWQ Quantization: 4-bit Matches 6-bit at 33% Less Size

**Date:** 2026-03-03
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed

## User Intention
Building on the v4 training session (100% bf16), user wanted to find the best deploy-ready quantized model. After naive quantization showed the usual accuracy/size tradeoffs, user wanted to explore learned quantization methods (DWQ, dynamic quant) from mlx-lm docs they'd found. The deeper goal was understanding quantization from first principles — not just running commands but building mental models of WHY these methods work.

## What We Accomplished
- ✅ **Naive quantization sweep** — 6-bit (96%, 3.1 GB), mixed 4/6 (91%, 2.2 GB), 4-bit (87%, 2.1 GB) on fused T2-v4 model
- ✅ **Ad-hoc testing exposed naive quant weakness** — 6-bit=3/5, mixed 4/6=2/5 on novel inputs (vs bf16=5/5). Test set flatters quantized models.
- ✅ **Dynamic quant attempted** — SSL cert fix, then OOM (exit 137), then with --grad-checkpoint survived but ~220s/layer. Too slow for 4B on M4. Killed.
- ✅ **DWQ 4-bit: 96% at 2.1 GB** — Matches naive 6-bit accuracy at 33% less size. 5/5 on ad-hoc tests. 0.88s latency (fastest model ever). New deploy target.
- ✅ **First-principles ELI5 of quantization** — RTN vs DWQ explained with ruler/translator analogies, math of scales/zero-points, thought experiments
- ✅ **Llama3-T2 config prepped** — Config updated for Llama 3.2 3B on v4 data, ready to launch tomorrow

## Technical Implementation

**Fuse pipeline:**
```bash
mlx_lm.fuse --model mlx-community/Qwen3-4B-Instruct-2507-bf16 \
  --adapter-path spoke/adapters-qwen3-t2 --de-quantize \
  --mlx-path spoke/models/qwen3-t2-v4-fused
```

**DWQ command (the winner):**
```bash
mlx_lm.dwq --model spoke/models/qwen3-t2-v4-fused --bits 4 \
  --mlx-path spoke/models/qwen3-t2-v4-dwq4 --max-seq-length 512 \
  --batch-size 1 --grad-checkpoint --num-samples 512 \
  --data-path spoke/data/v4
```

**DWQ training stats:** 512 iters, ~42 min, 14.1 GB peak memory. Val loss: 0.102 → 0.018 → 0.019. 125.9M trainable params (3.1% — quant scales/biases only).

**Final deploy model comparison:**

| Model | Size | Accuracy | Ad-hoc | Latency |
|-------|------|----------|--------|---------|
| bf16 | 7.5 GB | 100% | 5/5 | 1.82s |
| naive 6-bit | 3.1 GB | 96% | 3/5 | 0.94s |
| **DWQ 4-bit** | **2.1 GB** | **96%** | **5/5** | **0.88s** |
| naive 4-bit | 2.1 GB | 87% | — | 0.78s |

**Files Modified:**
- `spoke/config.yaml` — Updated for Llama3-T2 (model, layers=28, adapter path)
- `spoke/LEDGER.md` — Added DWQ results, finding #46, updated quant impact section, marked learned quant done in queue
- `spoke/bench/result_qwen3-t2-v4-dwq4_v2.json` — DWQ benchmark results (new)
- `spoke/models/qwen3-t2-v4-fused/` — Fused bf16 model (7.5 GB, kept for DWQ input)
- `spoke/models/qwen3-t2-v4-6bit/` — Naive 6-bit (3.1 GB, kept)
- `spoke/models/qwen3-t2-v4-dwq4/` — DWQ 4-bit deploy model (2.1 GB, new)
- `spoke/models/qwen3-t2-v4-4bit/` — DELETED (freed disk space)
- `spoke/models/qwen3-t2-v4-mixed46/` — DELETED (freed disk space)

## Bugs & Issues Encountered
1. **SSL cert error on dynamic_quant** — `ssl.SSLCertVerificationError` when downloading calibration model
   - **Fix:** `SSL_CERT_FILE=.venv/lib/python3.11/site-packages/certifi/cacert.pem`
2. **Dynamic quant OOM (exit 137)** — Killed by system on first attempt
   - **Fix:** Added `--grad-checkpoint`, survived but 220s/layer — impractical. Abandoned in favor of DWQ.
3. **DWQ disk full (Errno 28)** — Default calibration dataset (allenai/tulu-3-sft-mixture, 939K examples) filled disk
   - **Fix:** (1) Deleted unneeded model dirs (freed 3.1 → 8.4 GB), (2) Used `--data-path spoke/data/v4` for local calibration data
4. **Dynamic quant disk cleanup needed** — Partial download of dynquant model left behind
   - **Fix:** Deleted `spoke/models/qwen3-t2-v4-dynquant/`

## Key Learnings
- **DWQ recovers +9 pts over naive 4-bit at identical file size.** The "teacher-student" approach moves quantization grids to protect critical weights — exactly the "ruler sliding" analogy.
- **Ad-hoc tests > benchmark for evaluating quant quality.** Naive 6-bit and DWQ 4-bit both score 96% on benchmark, but 3/5 vs 5/5 on novel inputs. Benchmarks flatter naive quant; real-world use exposes it.
- **Task-specific calibration data is key for DWQ.** Using our own v4 ASR data instead of generic text means DWQ optimizes for our specific task's weight sensitivities.
- **`mlx_lm.dwq` defaults to downloading a 939K-example dataset** — always use `--data-path` to point at local data. Saves disk and gives task-specific calibration.
- **`mlx_lm.dynamic_quant` is impractical for 4B models on M4 24GB** — ~220s/layer, 56 layers = 3+ hours. DWQ is much faster (~42 min total).
- **M4 Air (fanless) thermal concerns after 48 hours of continuous training** — recommended overnight rest before next run.

## Architecture Decisions
- **DWQ over dynamic quant** — Dynamic quant was attempted first (per-layer bit allocation) but too slow. DWQ (distilled weight quantization) was faster and more effective — fine-tunes the interpretation of fixed 4-bit integers rather than choosing bits per layer.
- **Calibration on our own data** — Used v4 training data instead of generic dataset. Task-specific optimization > generic calibration.
- **Deleted naive 4-bit and mixed 4/6 models** — DWQ 4-bit strictly dominates both. Freed disk space for future experiments.
- **Llama3-T2 next, not Muon** — Llama is a config change + known pipeline. Muon requires custom optimizer code + integration. Lower risk, faster iteration.

## Ready for Next Session
- ✅ **DWQ 4-bit deploy model** — `spoke/models/qwen3-t2-v4-dwq4/` (2.1 GB, 96%, 0.88s)
- ✅ **Llama3-T2 config ready** — `spoke/config.yaml` updated, launch with `caffeinate -dims mlx_lm.lora --config spoke/config.yaml`
- ✅ **DWQ pipeline proven** — Same commands work for Llama after fuse
- 🔧 **Run Llama3-T2 training** — ~2.5 hours on M4
- 🔧 **Fuse + DWQ Llama3-T2** — If accuracy is good, DWQ'd Llama could be ~1.6 GB (smaller than Qwen3's 2.1 GB)

## Context for Future
DWQ 4-bit is the new deploy target, replacing naive 6-bit — same accuracy, 33% smaller, fastest latency. The quantization problem is effectively solved for Qwen3. Next session trains Llama 3.2 3B on v4 data to see if a smaller architecture can match Qwen3's 100% — if so, the DWQ'd Llama deploy model would be the smallest and fastest option yet. Muon optimizer remains in the queue for a future session (potential to reduce quant loss at the training stage rather than post-hoc).
