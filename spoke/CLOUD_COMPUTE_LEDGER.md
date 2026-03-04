# Spoke Cloud Compute Ledger

> Single source of truth for Modal + Unsloth throughput experiments.
> Last updated: 2026-03-04 (Qwen3.5 speed probes logged; text-only Qwen3 probe pending.)

## How to Read This

- **Reported `train_steps_per_second` is not the real speed signal.** It includes the huge first-step compile/warmup and will understate steady-state throughput.
- **Steady-state it/sec** below means the observed late-step rate after warmup (roughly after step 20 on 50-step probes).
- **Train samples/sec** comes from TRL/W&B. It can rise even when raw step rate falls if the batch size is larger.
- **Probe runs** are 50-step runs with eval/save disabled unless noted.
- **Current stack** = Modal L40S (48 GB), `spoke/cloud/train.py`, clean CUDA base image, `unsloth`, `flash-linear-attention`, `causal-conv1d`, `trl==0.22.2`.

---

## Environment Baseline

- **GPU:** Modal `L40S` (48 GB)
- **Dataset:** v4 (`1201` train / `20` valid)
- **Precision:** bf16 LoRA (no QLoRA)
- **Default LoRA:** `r=8`, `alpha=16`, `dropout=0.0`
- **Important current findings:**
  - `FA2 = False` in all successful runs so far
  - `flash-linear-attention` + `causal-conv1d` are present
  - `Sample packing skipped (processor-based model detected)` on Qwen3.5
  - Qwen3.5 path is loading as `Qwen3VLProcessor` and includes `model.visual.*` weights

---

## Experiment Log

| Run | Date | Model | Seq | Batch | Accum | Grad Ckpt | Status | Reported Steps/s | Steady-State it/s | Notes |
|-----|------|-------|-----|-------|-------|-----------|--------|------------------|-------------------|-------|
| **C0** | 03-04 | `unsloth/Qwen3.5-4B` | 512 | 4 | 2 (effective) | On | **FAILED** | ~0.33 | ~0.3 | Original broken cloud run. Timed out at `3600s` around step `1114/2000`. Missing fast CUDA path and structurally wrong setup. |
| **C1** | 03-04 | `unsloth/Qwen3.5-4B` | 256 | 4 | 1 | On | **CANCELLED** | — | — | Tried `unsloth/unsloth` image. Booted Jupyter/SSH/Ollama under `supervisord`. Wrong image for Modal workers. |
| **C2** | 03-04 | `unsloth/Qwen3.5-4B` | 256 | 4 | 1 | On | **PASS** | `0.376` | `~1.8-2.1` | `spoke-qwen35-probe4`. First clean success after fixing image, adding `flash-linear-attention` + `causal-conv1d`, setting `lora_dropout=0.0`, and forcing `num_proc=1` in masking. First step ~`104.8s`. |
| **C3** | 03-04 | `unsloth/Qwen3.5-4B` | 512 | 8 | 1 | Off | **CANCELLED** | — | — | Tried source-building `flash-attn` in the image to force FA2. Wheel compile ran for 20+ minutes. Bad probe strategy. Need prebuilt wheel or cached image instead. |
| **C4** | 03-04 | `unsloth/Qwen3.5-4B` | 512 | 8 | 1 | Off | **PASS** | `0.398` | `~3.8-4.1` | `spoke-qwen35-speed-probe-b8-nofa2`. Best raw step rate so far. First step ~`106.7s`. `FA2` still false. Packing flag had no effect because the model loaded as processor-based. |
| **C5** | 03-04 | `unsloth/Qwen3.5-4B` | 512 | 12 | 1 | Off | **PASS** | `0.361` | `~2.9-3.1` | `spoke-qwen35-speed-probe-b12-noexport`. Lower raw step rate than C4, but `train_samples_per_second` improved to `4.336` (vs `3.184` on batch 8). First step ~`113.5s`. |
| **C6** | 03-04 | `unsloth/Qwen3.5-4B` | 512 | 4 | 1 | Off | **COMPLETED** | — | — | Full run `spoke-qwen35-t2` completed per user. Detailed throughput not captured in this ledger yet. |

---

## What Actually Helped

1. **Fixing the image** mattered more than anything else. Dropping the interactive `unsloth/unsloth` image and going back to a clean CUDA image stopped the Jupyter/SSH/Ollama garbage and restored control over the runtime.
2. **Installing the Qwen3.5 fast-path deps** mattered. `flash-linear-attention` alone was not enough; `causal-conv1d` was also needed for the Qwen3.5 hybrid path.
3. **Setting `lora_dropout=0.0`** mattered. Unsloth's fast patch path was blocked when dropout was nonzero.
4. **Turning off gradient checkpointing** helped on L40S because this run has enough VRAM headroom to trade memory for less recompute.
5. **Making `gradient_accumulation_steps=1` explicit** removed hidden micro-step overhead and made the comparisons sane.
6. **Raising batch size from `4` to `8`** produced the biggest clean speed jump in observed post-warmup step rate.
7. **Skipping merged export on probes** cuts wasted wall-clock on short runs. Export was adding ~25-30 seconds after the 50-step train loop.

## What Did Not Help

1. **Source-building `flash-attn` inside probe startup** is too slow. It blocked iteration before training even started.
2. **`packing=True` on current Qwen3.5 runs** did nothing. Unsloth explicitly said packing was skipped because the model loaded through a processor-based path.
3. **Bigger batch is not automatically better on raw it/sec.** `batch=12` improved work per second but reduced raw step rate compared with `batch=8`.

## Root Causes of Remaining Waste

1. **This Qwen3.5 path is not a clean text-only LM path.** The logs show `Qwen3VLProcessor` plus `model.visual.*` weights.
2. **FlashAttention2 is still off.** Successful runs still print `FA2 = False`.
3. **The first step is dominated by compile/warmup.** Short probes always look worse in averaged trainer metrics than the actual late-step rate.
4. **Masking density is low.** Only `14/128` active labels in the sample check (~`10.9%`), so a lot of tokens are still paying forward-pass cost without contributing to loss.

---

## Current Best Speed Profiles

### Best raw step rate

- **Run:** C4
- **Config:** `seq=512`, `batch=8`, `accum=1`, `grad_ckpt=off`, no eval, no save
- **Observed steady-state:** `~3.8-4.1 it/sec`
- **Why it wins:** Better GPU saturation than batch 4 without the per-step slowdown seen at batch 12

### Best total work per second (so far)

- **Run:** C5
- **Config:** `seq=512`, `batch=12`, `accum=1`, `grad_ckpt=off`, no eval, no save, no export
- **Observed steady-state:** `~2.9-3.1 it/sec`
- **Reported train samples/sec:** `4.336`
- **Why it matters:** Lower steps/sec than C4, but more examples processed per step

---

## Next High-Signal Experiments

1. **Try a text-only Qwen3 checkpoint** (`Qwen3-4B-Instruct-2507`) to avoid the processor/VLM overhead and to see whether sample packing becomes available.
2. **Probe `batch=16` on the best text-only path** if batch 8 succeeds cleanly, to find the real L40S headroom.
3. **Enable FA2 the right way** via a prebuilt wheel or cached image, not by rebuilding `flash-attn` during every probe.
4. **Reduce prompt-token waste** if training speed remains the constraint. The current masking density shows a lot of non-loss-bearing tokens.
