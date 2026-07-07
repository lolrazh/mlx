# Gemma 4 E4B → MLX Port: Conversion Validated, 4-bit Breaks Behavior (quant sweep PAUSED on machine memory)

**Date:** 2026-07-08
**Agent:** Claude Opus 4.8 (1M context)
**Status:** PAUSED mid-quant-sweep — a memory-heavy external process + full disk wedged the GPU (watchdog timeouts). Resume when memory frees.

## User Intention
Port the accuracy-champion Gemma 4 E4B checkpoint (`spoke-g4e4b-hp-lplus2-ckpt900`, best = 95.7 core / 82.8 broad / 88.5 v5 bf16) to MLX, quantized, for local M4 deployment — closing THE deployment gate every prior log ends on. Go easy on the MacBook's memory.

## What's PROVEN (the port fundamentally works)
1. **MLX has native Gemma 4 support** — `mlx_lm 0.31.3` ships `gemma4.py` + `gemma4_text.py`. `gemma4.sanitize()` drops the vision/audio towers at load time and rewrites `language_model.*` keys, so `mlx_lm.convert` eats the FULL multimodal merged checkpoint directly. No from-scratch reimplementation (unlike the Moonshine port).
2. **Conversion works**: merged bf16 (15.88 GB, 2076 tensors) → MLX 4-bit = **4.501 bits/weight, 3.9 GB**. Clean `mlx_lm convert` run.
3. **bf16 MLX inference is CORRECT**: on a 2-example smoke (v2 prompt, `enable_thinking=False`) the bf16 model produced clean, correct output ("...feature 🔥", "Notify the @ops team...") — no thinking channel. Conversion + runtime + tokenizer all good.

## THE KEY FINDING — uniform 4-bit PTQ breaks a LEARNED behavior (behavioral cliff, not graceful decay)
- 4-bit core23 (v2, local run_benchmark) = **13%** — and EVERY output falls into `<|channel>thought` (the base model's Harmony-style thinking channel).
- The LoRA fine-tune taught this thinking-capable base to SKIP the thought channel and answer directly. **bf16 obeys that; 4-bit quantization noise reverts it to the base "think first" prior.** Isolation test: identical prompt/runtime/greedy, bf16 = clean, 4-bit = thinks. The ONLY variable is quantization → quantization is the cause.
- So: **conversion is validated; the open question is purely PRECISION** — find the least-aggressive quant that preserves the no-think behavior (8-bit likely near-lossless; mirrors Google's own QAT choice of 8-bit for the MLP block; `mlx_lm convert --quant-predicate {mixed_3_6,mixed_4_6}` is a one-flag mixed-precision lever if a uniform width doesn't land).

## What BLOCKED the sweep (machine state, not the approach)
- **Disk is 99% full (~5 GB free)** → no swap headroom (swap 5.6/7.2 GB used) → Metal GPU command buffers stall on memory → `kIOGPUCommandBufferCallbackErrorTimeout` (uncatchable C++ abort).
- A **memory-heavy external process** was running (user flagged it). bf16 (15 GB) local inference is at the M4 memory edge even idle; in-memory quantization holds bf16 mmap + quantized arrays simultaneously → tips it over.
- Two-way squeeze: in-memory quant → memory thrash → GPU timeout; on-disk quant → can't write (8-bit ≈ 8 GB won't fit in 5 GB free). **Both blocked by the full disk.**

## Cloud/infra gotchas hit + fixed (worth keeping)
- `mlx==0.31.3` has NO Linux wheel (Linux tops at 0.31.2); and `mlx-lm`'s `mlx` dep is macOS-gated so pip skips it on Linux → must `pip install mlx==0.31.2` EXPLICITLY. Even then the Linux mlx wheel is a thin wrapper (`libmlx.so cannot open` / `mlx.__file__ is None`) → **MLX conversion on Modal Linux is a dead end.** Convert locally.
- `modal volume get <vol> <dir> <dest>` on a DIRECTORY concatenates all files into one blob (starts with config.json, 15.88 GB, `Invalid json header length`). **Download the safetensors PER-FILE** (`.../merged/model.safetensors`) — per-file get is clean. (Parallels the subfolder note in download_model.py.)

## Artifacts / state
- `spoke/models/g4e4b-champion-bf16/` — full merged bf16 HF checkpoint (15.88 GB, valid, gitignored). KEEP to resume (re-download is 15.9 GB).
- `spoke/cloud/extract_text_decoder.py` — committed (fe6f58b). Modal-side sanitize; inspect mode proved towers are only ~1 GB of 15.9 GB (PLE embeddings dominate) → not worth extracting for size.
- 4-bit + 8-bit MLX models were created then DELETED (4-bit = dead-end; 8-bit = partial, disk-full write fail).
- Sweep script: `scratchpad/quant_sweep.py` (in-memory, disk-free; skips bf16 which times out).

## NEXT (resume when memory/disk free)
1. **Free disk** (~25-30 GB) so an 8-bit model (~8 GB) can be written AND swap has room. This is the real unblock.
2. Re-run quant sweep 8/6/5-bit on core23 → find the precision floor that kills the think-cliff.
3. Convert the winner to disk, benchmark broad58 + v5-131 for the real deployment numbers + **measure true M4 latency** (cloud L40S 0.9-1.1s is not representative).
4. Grader-parity caveat: local run_benchmark grader is simpler than cloud benchmark.py; compare quant-vs-quant locally, anchor to cloud bf16 95.7/82.8/88.5 loosely.
5. Update LEDGER (finding #108?) once numbers land.
