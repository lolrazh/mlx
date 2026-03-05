# Spoke Cloud Compute Ledger

> Single source of truth for Modal + Unsloth throughput experiments.
> Last updated: 2026-03-05 (Added no-thinking parity + ultra-quality full runs and matched benchmark artifacts.)

## How to Read This

- **Reported `train_steps_per_second` is not the real speed signal.** It includes the huge first-step compile/warmup and will understate steady-state throughput.
- **Steady-state it/sec** below means the observed late-step rate after warmup (roughly after step 20 on 50-step probes).
- **Train samples/sec** comes from TRL/W&B. It can rise even when raw step rate falls if the batch size is larger.
- **Probe runs** are 50-step runs with eval/save disabled unless noted.
- **This file is compute-only.** Quality/benchmark outcomes should be tracked separately.
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
  - Full 164-step packed Qwen3 runs cluster around `~2.2-2.4 it/s` steady-state with this stack
  - Changing `rank`/`learning_rate` moved quality but did not materially move speed at fixed batch/seq
  - No-thinking full runs with unpacked parity data path reached `~3.8-4.1 it/s`, but quality remained below local 100% (`74%` parity, `83%` ultra)

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
| **C6** | 03-04 | `unsloth/Qwen3.5-4B` | 512 | 4 | 1 | Off | **COMPLETED** | — | — | Full run `spoke-qwen35-t2` completed. It served as a workflow proof, but not as a clean throughput datapoint; the short probes remain the reliable compute reference. |
| **C7** | 03-04 | `unsloth/Qwen3-4B-Instruct-2507` | 512 | 8 | 1 | Off | **PASS** | `1.133` | `~1.35-1.45` | `spoke-qwen3-text-speed-probe-b8`. Text-only path. `Qwen2Tokenizer`, not processor-based. Packing enabled, examples collapsed `1201 -> 327`, masking density jumped to `42.3%`, first step only ~`7.4s`, `train_samples_per_second = 9.066`. |
| **C8** | 03-04 | `unsloth/Qwen3-4B-Instruct-2507` | 512 | 16 | 1 | Off | **PASS** | `0.667` | `~0.72-0.88` | `spoke-qwen3-text-speed-probe-b16`. Still fits cleanly. Raw steps/sec fell versus C7, but `train_samples_per_second` improved again to `10.669`. First step ~`7.6s`. |
| **C9** | 03-05 | `unsloth/Qwen3-4B-Instruct-2507` | 512 | 4 | 1 | Off | **COMPLETED** | — | `~2.2-2.4` | `spoke-qwen3-unsloth-e2p-packed` baseline full run (`164` steps, packed, export on). Benchmark artifact: `result_spoke-qwen3-unsloth-e2p-packed_modal_v2.json`. |
| **C10** | 03-05 | `unsloth/Qwen3-4B-Instruct-2507` | 512 | 4 | 1 | Off | **COMPLETED** | `1.905` | `~2.2-2.4` | `spoke-qwen3-unsloth-sweep-r8-lr2e4` (`r=8`, `lr=2e-4`). |
| **C11** | 03-05 | `unsloth/Qwen3-4B-Instruct-2507` | 512 | 4 | 1 | Off | **COMPLETED** | `1.782` | `~2.1-2.3` | `spoke-qwen3-unsloth-sweep-r8-lr1e4` (`r=8`, `lr=1e-4`). |
| **C12** | 03-05 | `unsloth/Qwen3-4B-Instruct-2507` | 512 | 4 | 1 | Off | **COMPLETED** | `2.009` | `~2.2-2.4` | `spoke-qwen3-unsloth-sweep-r8-lr5e5` (`r=8`, `lr=5e-5`). Highest reported `train_steps_per_second` in this full-run sweep. |
| **C13** | 03-05 | `unsloth/Qwen3-4B-Instruct-2507` | 512 | 4 | 1 | Off | **COMPLETED** | `1.937` | `~2.2-2.4` | `spoke-qwen3-unsloth-sweep-r16-lr1e4` (`r=16`, `lr=1e-4`). |
| **C14** | 03-05 | `unsloth/Qwen3-4B-Instruct-2507` | 512 | 4 | 1 | Off | **COMPLETED** | `1.981` | `~2.2-2.4` | `spoke-qwen3-unsloth-sweep-r16-lr5e5` (`r=16`, `lr=5e-5`). |
| **C15** | 03-05 | `unsloth/Qwen3-4B-Instruct-2507` | 512 | 4 | 1 | Off | **COMPLETED** | — | `~3.8-4.1` | `spoke-qwen3-parity-nothink-v1`. Hard no-thinking formatting guard, `packing=False`, parity trainer, Adam, `max_grad_norm=0.0`. Benchmark: `result_spoke-qwen3-parity-nothink-v1_modal_v2.json` (`74%`). |
| **C16** | 03-05 | `unsloth/Qwen3-4B-Instruct-2507` | 512 | 4 | 1 | Off | **COMPLETED** | — | `~3.8-4.1` | `spoke-qwen3-ultra-quality-nothink-v1`. Hard no-thinking + ultra profile (`2500` steps, `r=32`, `alpha=64`, `dropout=0.05`, `rsLoRA=True`, `packing=False`). Benchmark: `result_spoke-qwen3-ultra-quality-nothink-v1_modal_v2.json` (`83%`). |

---

## What Actually Helped

1. **Fixing the image** mattered more than anything else. Dropping the interactive `unsloth/unsloth` image and going back to a clean CUDA image stopped the Jupyter/SSH/Ollama garbage and restored control over the runtime.
2. **Installing the Qwen3.5 fast-path deps** mattered. `flash-linear-attention` alone was not enough; `causal-conv1d` was also needed for the Qwen3.5 hybrid path.
3. **Setting `lora_dropout=0.0`** mattered. Unsloth's fast patch path was blocked when dropout was nonzero.
4. **Turning off gradient checkpointing** helped on L40S because this run has enough VRAM headroom to trade memory for less recompute.
5. **Making `gradient_accumulation_steps=1` explicit** removed hidden micro-step overhead and made the comparisons sane.
6. **Raising batch size from `4` to `8`** produced the biggest clean speed jump in observed post-warmup step rate.
7. **Skipping merged export on probes** cuts wasted wall-clock on short runs. Export was adding ~25-30 seconds after the 50-step train loop.
8. **Switching to text-only Qwen3 is the biggest structural win so far.** It removes the processor/VLM path, enables packing, slashes first-step warmup from ~`100s` to ~`7s`, and more than doubles `train_samples_per_second` versus the best Qwen3.5 probe.
9. **Run-type alignment matters.** Comparing full 164-step packed runs to full runs (not 50-step probes) gave stable throughput around `~2.2-2.4 it/s` and removed most noise from first-step warmup.

## What Did Not Help

1. **Source-building `flash-attn` inside probe startup** is too slow. It blocked iteration before training even started.
2. **`packing=True` on current Qwen3.5 runs** did nothing. Unsloth explicitly said packing was skipped because the model loaded through a processor-based path.
3. **Bigger batch is not automatically better on raw it/sec.** `batch=12` improved work per second but reduced raw step rate compared with `batch=8`.
4. **Raw `it/sec` is a bad cross-model comparison once packing changes.** The text-only Qwen3 probes do fewer steps per second than Qwen3.5 batch-8, but each step carries far more useful tokens and much less warmup overhead.
5. **A completed long run is not automatically a useful speed datapoint.** The 2000-step Qwen3.5 run proved the pipeline could finish, but the short probes are still the only clean apples-to-apples throughput reference in this file.
6. **Tuning `learning_rate` and LoRA `rank` for speed did not pay off.** In the 03-05 full-run sweep (batch 4, seq 512, packed), throughput stayed in essentially the same band.
7. **Forcing no-thinking did not close the quality gap by itself.** The no-thinking parity/ultra runs landed at `74%` and `83%`, matching earlier configuration-driven behavior bands.

## Root Causes of Remaining Waste

1. **This Qwen3.5 path is not a clean text-only LM path.** The logs show `Qwen3VLProcessor` plus `model.visual.*` weights.
2. **FlashAttention2 is still off.** Successful runs still print `FA2 = False`.
3. **The first step is dominated by compile/warmup.** Short probes always look worse in averaged trainer metrics than the actual late-step rate.
4. **Masking density is low.** Only `14/128` active labels in the sample check (~`10.9%`), so a lot of tokens are still paying forward-pass cost without contributing to loss.
5. **The Qwen3.5 checkpoint itself is part of the waste.** As long as it loads as a processor/VLM path, it blocks packing and keeps the warmup/throughput profile worse than the text-only Qwen3 path.

---

## Current Best Speed Profiles

### Best raw step rate (same-model comparison only)

- **Run:** C4
- **Config:** `seq=512`, `batch=8`, `accum=1`, `grad_ckpt=off`, no eval, no save
- **Observed steady-state:** `~3.8-4.1 it/sec`
- **Why it wins:** Better GPU saturation than batch 4 without the per-step slowdown seen at batch 12

### Best total work per second (so far)

- **Run:** C8
- **Config:** `Qwen3-4B text-only`, `seq=512`, `batch=16`, `accum=1`, `grad_ckpt=off`, no eval, no save, no export
- **Observed steady-state:** `~0.72-0.88 it/sec`
- **Reported train samples/sec:** `10.669`
- **Why it matters:** Much lower raw step rate than C4, but far more useful work per second because packing is active and the warmup cost is tiny.

### Best balanced probe profile right now

- **Run:** C7
- **Config:** `Qwen3-4B text-only`, `seq=512`, `batch=8`, `accum=1`, `grad_ckpt=off`, no eval, no save, no export
- **Observed steady-state:** `~1.35-1.45 it/sec`
- **Reported train samples/sec:** `9.066`
- **Why it wins:** Clean text-only path, packing enabled, low warmup, and better responsiveness than batch 16 while still massively outperforming Qwen3.5 on useful throughput.

### Best full-run profile (current training recipe)

- **Packed recipe runs:** C9-C14 family (`164` steps, packed, export on)
- **Observed steady-state:** generally `~2.2-2.4 it/sec`
- **Reported train steps/s range:** `1.782` to `2.009`
- **No-thinking parity/ultra runs:** C15-C16 (`packing=False`, export on) ran around `~3.8-4.1 it/sec`, but this is not directly comparable to packed runs because effective tokens-per-step differ.

---

## Final Recommendation

1. **Use `unsloth/Qwen3-4B-Instruct-2507` as the default cloud training model.**
2. **Treat `unsloth/Qwen3.5-4B` as a lower-priority compute path** unless a separate, explicit debugging effort makes it competitive again.
3. **Optimize from the text-only baseline** (batch size, FA2 via cached image, prompt-token efficiency) rather than spending more cycles on the Qwen3.5 VLM-style path.

---

## Next High-Signal Experiments

1. **Use text-only Qwen3 as the new speed baseline.** The Qwen3.5 VLM-style path is now clearly the inferior compute path for this task.
2. **Run a full-run batch sweep (`batch=4/6/8/12`) on text-only Qwen3** at fixed `r=16`, `lr=2e-4`, `seq=512`, packed, to find the next real speed ceiling for quality-comparable runs.
3. **Enable FA2 the right way** via a prebuilt wheel or cached image, not by rebuilding `flash-attn` during every probe.
4. **Reduce prompt-token waste** if training speed remains the constraint. The current Qwen3 text-only path is much better, but trimming repeated non-loss-bearing prompt tokens still helps.
