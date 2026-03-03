# Qwen3.5 Modal Fast-Path Repair and Relaunch

**Date:** 2026-03-04
**Agent:** Codex (GPT-5)
**Status:** ✅ Completed (probe passed, full run launched in detached mode)
**Continues from:** `2026-03-04_0100_modal-unsloth-cloud-pipeline.md`

## User Intention
User wanted to salvage the broken Qwen3.5-4B Modal + Unsloth cloud setup after a failed, slow run on L40S, get the training path onto the actual fast kernels, and relaunch a real run with materially better throughput than the previous ~0.3 it/sec failure.

## What We Accomplished
- ✅ **Replaced the bad container strategy** — Removed the `unsloth/unsloth` interactive image path after confirming it booted Jupyter/SSH/Ollama services under `supervisord`, which is wrong for Modal workers.
- ✅ **Rebuilt the training image on a clean CUDA base** — Switched `spoke/cloud/train.py` to `nvidia/cuda:12.4.1-devel-ubuntu22.04` with Python 3.11, `unsloth`, `wandb`, and pinned `trl==0.22.2`.
- ✅ **Enabled the missing Qwen3.5 fast-path deps** — Installed `flash-linear-attention` and successfully built `causal-conv1d` from source for L40S after constraining the build.
- ✅ **Removed self-inflicted slow-path config** — Changed LoRA dropout from `0.05` to `0.0`, which restored Unsloth's fast patch path.
- ✅ **Fixed dataset preprocessing overhead** — Forced `train_on_responses_only(..., num_proc=1)` to stop the 21-worker masking pass that was wasting time and throwing timeouts.
- ✅ **Kept the cloud run structurally safe** — Timeout remains `10800`, default `max_seq_length` remains `256`, and eval/save cadence remains configurable.
- ✅ **Validated the repaired path with a short probe** — `spoke-qwen35-probe4` completed 50 steps and exported a merged bf16 model to the `spoke-output` volume.
- ✅ **Launched the real training run** — Started `spoke-qwen35-t2` for 2000 steps in detached mode on Modal.

## Technical Implementation

**Primary file changed:**
- `spoke/cloud/train.py`

**Image strategy:**
- Base image: `nvidia/cuda:12.4.1-devel-ubuntu22.04`
- Added Python via Modal `add_python="3.11"`
- Installed `unsloth`, `wandb`, `flash-linear-attention`
- Built `causal-conv1d` with:
```bash
CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST=8.9 python -m pip install --no-build-isolation --no-deps causal-conv1d
```

**Training config changes:**
- `timeout=10800`
- `max_seq_length=256`
- `lora_dropout=0.0`
- `eval_steps=200` (default, can disable)
- `save_steps=500` (default, can disable)
- `train_on_responses_only(..., num_proc=1)`

**Probe run (`spoke-qwen35-probe4`):**
- 50 steps completed successfully
- `train_runtime`: `132.94s`
- Trainer-reported `train_steps_per_second`: `0.376`
- This average is skewed by a huge first-step warmup (`~104.8s`)
- After warmup, steady-state training settled around `~1.8-2.1 it/sec`
- Merged bf16 model exported to `/output/spoke-qwen35-probe4/merged`

**Full run launched:**
- Run name: `spoke-qwen35-t2`
- Steps: `2000`
- Modal app id: `ap-CUgOeTcbuqgfHsrcro6txv`

## Bugs & Issues Encountered

1. **`unsloth/unsloth` image booted irrelevant services** — The container started Jupyter, SSH, and Ollama under `supervisord`.
   - **Fix:** Abandoned that image and returned to a clean CUDA build image.

2. **Temporary token-count instrumentation broke the Qwen3.5 processor** — Calling the processor directly on plain text triggered the vision/image path and crashed.
   - **Fix:** Removed the token-count diagnostic entirely.

3. **`flash-linear-attention` alone was insufficient** — Unsloth still reported "fast path is not available" after only adding FLA.
   - **Fix:** Added `causal-conv1d` as well; both are required for this model family.

4. **Naive `causal-conv1d` build failed on Modal** — The default build fell back to source, picked the wrong compiler path, and tried compiling many GPU architectures.
   - **Fix:** Forced `gcc/g++`, disabled build isolation, and constrained `TORCH_CUDA_ARCH_LIST=8.9` for the L40S.

5. **`train_on_responses_only` used too many worker processes** — Default behavior spawned `num_proc=21`, which added unnecessary overhead and raised `multiprocess.context.TimeoutError`.
   - **Fix:** Set `num_proc=1`.

6. **LoRA dropout blocked the fast path** — Unsloth explicitly warned that dropout `0.05` disables its fastest patching path.
   - **Fix:** Set `lora_dropout=0.0`.

## Key Learnings
- **The original failure was not just "cloud is slow."** The setup was on the wrong image, with missing CUDA extensions and a slow-path LoRA config.
- **Qwen3.5-4B on Unsloth needs both `flash-linear-attention` and `causal-conv1d`.** Installing one without the other is not enough.
- **For CUDA extension builds on Modal, architecture scoping matters.** Letting `causal-conv1d` compile for every architecture is wasteful and brittle; targeting `sm_89` for L40S is the correct move.
- **The first training step is an outlier.** Warmup/compilation makes the headline average look much worse than the steady-state loop.
- **This repaired setup is much better, but it is not 15 it/sec.** The realistic steady-state observed here was roughly `~2 it/sec`, which is a major improvement over `0.3 it/sec` but nowhere near the earlier guess.

## Architecture Decisions
- **Probe before full rerun** — We used repeated short 50-step probes to verify the image, fast-path warnings, and steady-state throughput before spending another long run.
- **Keep the proven training recipe otherwise stable** — The optimizer stayed `adamw_torch` with `weight_decay=0.0`, rank stayed `8`, alpha stayed `16`, and the EOS/chat template fix remained unchanged.
- **Detach the full run** — The 2000-step job was started with `modal run --detach` so the local session could exit without killing the remote worker.

## Ready for Next Session
- ✅ **Repaired cloud image path is committed locally** — `spoke/cloud/train.py` contains the working build path and training defaults.
- ✅ **A successful probe artifact exists** — `spoke-qwen35-probe4` produced a merged bf16 export on the output volume.
- ✅ **Full run is in progress** — `spoke-qwen35-t2` is running remotely under app `ap-CUgOeTcbuqgfHsrcro6txv`.
- 🔧 **Next likely step:** monitor logs, confirm the first eval at step 200, and verify whether the steady-state remains near the probe's post-warmup rate.
- 🔧 **If speed still feels too low:** inspect gradient accumulation and eval cadence before changing model architecture again.

## Context for Future
This session converted the cloud pipeline from "structurally broken and misleadingly benchmarked" into a working, reproducible training path. The core repair was not a single flag; it was getting onto the correct container base, restoring the actual Qwen3.5 fast-path dependencies, removing the dropout-induced slow path, and stripping out wasteful multiprocessing in the masking pass. The resulting setup is no longer catastrophically slow, but the honest throughput expectation is closer to ~2 it/sec after warmup than to the earlier 15 it/sec claim. The current 2000-step run is live and should be treated as the first serious benchmark of the repaired pipeline.
