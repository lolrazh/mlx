# Spoke Cloud Compute Ledger

> Single source of truth for Modal cloud throughput experiments (legacy Unsloth + current HF+PEFT).
> Last updated: 2026-03-06 (Added Qwen3.5 HF text-only compatibility + smoke matrix on Modal; reconciled with latest benchmark outputs.)

## How to Read This

- **Reported `train_steps_per_second` is not the real speed signal.** It includes the huge first-step compile/warmup and will understate steady-state throughput.
- **Steady-state it/sec** below means the observed late-step rate after warmup (roughly after step 20 on 50-step probes).
- **Train samples/sec** comes from TRL/W&B. It can rise even when raw step rate falls if the batch size is larger.
- **Probe runs** are 50-step runs with eval/save disabled unless noted.
- **This file is compute-only.** Quality/benchmark outcomes should be tracked separately.
- **Current default stack** = Modal L40S (48 GB), `spoke/cloud/train_hf.py`, clean CUDA base image, `transformers==5.2.0`, `peft==0.14.0` (legacy Unsloth runs retained for historical comparison).

---

## Environment Baseline

- **GPU:** Modal `L40S` (48 GB)
- **Dataset snapshots:**
  - v4 (`1201` train / `20` valid) for historical Unsloth sweep runs (`C0-C17`)
  - v5 (`train.jsonl=1287` rows) for recent HF+PEFT parity and SmolLM3 runs (`C18+`)
- **Precision:** bf16 LoRA (no QLoRA)
- **Current default trainer stack (`C18+`):** `spoke/cloud/train_hf.py` (PyTorch `2.6.0` CUDA `12.4`, `transformers==5.2.0`, `peft==0.14.0`)
- **Legacy trainer stack (`C0-C17`):** `spoke/cloud/train.py` (`unsloth`, `flash-linear-attention`, `causal-conv1d`, `trl==0.22.2`)
- **Important current findings:**
  - HF+PEFT parity run `spoke-qwen3-hf-parity-v1` reported `train_steps_per_second=3.326` at `2000` steps (`2026-03-05_1912` log).
  - HF+PEFT v5 forced-v2 run `spoke-qwen3-hf-v5-v2prompt-v1-20260305-2247` reported `train_steps_per_second=2.456` at `1200` steps (`2026-03-05_2302` log).
  - On that same run, eval-loss best (`checkpoint-600`) was not benchmark-best (`checkpoint-1200`), so checkpoint promotion cannot rely on eval loss alone.
  - SmolLM3 cloud run (`spoke-smollm3-v5-v2prompt-v1-20260305-2358`) has benchmark deltas logged, but train throughput was not captured in agent logs yet.
  - Qwen3.5 base-model benchmarking on Modal now works under HF via `Qwen3_5ForCausalLM` + `text_config` (Transformers 5.2.0); this removed the previous `model type qwen3_5 not recognized` blocker.
  - Historical Unsloth packed Qwen3 full runs remain clustered around `~2.2-2.4 it/s`; unpacked no-thinking profile stayed around `~3.8-4.1 it/s`.

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
| **C17** | 03-05 | `unsloth/Qwen3-4B-Instruct-2507` | 512 | 4 | 1 | Off | **COMPLETED** | `3.568` | `~3.8-4.1` | `spoke-qwen3-parity-nothink-templatefix-v1`. Enforced no-thinking at tokenizer template level (no output stripping fallback). Benchmark: `result_spoke-qwen3-parity-nothink-templatefix-v1_modal_v2.json` (`74%`). |
| **C18** | 03-05 | `Qwen/Qwen3-4B-Instruct-2507` | 512 | 4 | 1 | On | **COMPLETED** | `3.326` | — | `spoke-qwen3-hf-parity-v1` (pure HF+PEFT parity stack, 2000 steps, epoch `6.67`). Benchmarked at `96%` on core23 (`result_spoke-qwen3-hf-parity-v1_modal_v2.json`). |
| **C19** | 03-05 | `Qwen/Qwen3-4B-Instruct-2507` | 512 | 4 | 1 | On | **COMPLETED** | `2.456` | — | `spoke-qwen3-hf-v5-v2prompt-v1-20260305-2247` (v5 data + forced v2 prompt, eval/save `50/100`). Eval-loss best was `checkpoint-600` (`87%`), but `checkpoint-1200` hit `100%` core23 after manual merge benchmark. |
| **C20** | 03-05 | `HuggingFaceTB/SmolLM3-3B` | 512 | 4 | 1 | On | **COMPLETED** | — | — | `spoke-smollm3-v5-v2prompt-v1-20260305-2358` (best checkpoint by eval loss = `800`). Benchmark sidecar: core23 `87%` for both merged-best and `checkpoint-1200`; broad58 improved `48.3% -> 53.5%` at `checkpoint-1200`. |
| **C21** | 03-06 | `Qwen/Qwen3.5-2B` | 512 | — | — | — | **PASS** | — | — | Base-model cloud benchmark after HF fix. `Qwen3_5ForCausalLM` text-only load path validated. core23 `9%` (`result_Qwen-Qwen3.5-2B_modal_v2_test_set_v3.json`). |
| **C22** | 03-06 | `Qwen/Qwen3.5-4B` | 512 | — | — | — | **PASS** | — | — | Base-model cloud benchmark after HF fix. core23 `13%` (`result_Qwen-Qwen3.5-4B_modal_v2_test_set_v3.json`). |
| **C23** | 03-06 | `Qwen/Qwen3.5-2B` | 512 | 4 | 1 | On | **COMPLETED** | `0.777` | `~0.75-0.90` | 50-step HF smoke (`spoke-qwen35-2b-hf-smoke50-20260306`). Runtime `64.38s`, train loss `0.744`. core23 `22%` (`result_spoke-qwen35-2b-hf-smoke50-20260306_modal_v2_test_set_v3.json`). |
| **C24** | 03-06 | `Qwen/Qwen3.5-4B` | 512 | 4 | 1 | On | **COMPLETED** | `1.035` | `~1.00-1.15` | 50-step HF smoke (`spoke-qwen35-4b-hf-smoke50-20260306`). Runtime `48.31s`, train loss `0.599`. core23 `30%` (`result_spoke-qwen35-4b-hf-smoke50-20260306_modal_v2_test_set_v3.json`). |

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
10. **Moving to pure HF+PEFT (`train_hf.py`) removed Unsloth conversion drift** and gave reproducible merged exports for cloud parity.
11. **Prompt-policy override at training time (`system_prompt_mode`) prevented dataset churn** while running prompt ablations (`as_is`, `v2`, `v3`) on the same uploaded v5 rows.
12. **Benchmarking non-best checkpoints paid off.** On C19, eval-loss picked `checkpoint-600` (`87%`) while `checkpoint-1200` reached `100%` on the same core23 benchmark.
13. **Upgrading HF runtime to Transformers 5.2.0 unlocked Qwen3.5 support** and made cloud base/smoke probes possible without falling back to Unsloth.

## What Did Not Help

1. **Source-building `flash-attn` inside probe startup** is too slow. It blocked iteration before training even started.
2. **`packing=True` on current Qwen3.5 runs** did nothing. Unsloth explicitly said packing was skipped because the model loaded through a processor-based path.
3. **Bigger batch is not automatically better on raw it/sec.** `batch=12` improved work per second but reduced raw step rate compared with `batch=8`.
4. **Raw `it/sec` is a bad cross-model comparison once packing changes.** The text-only Qwen3 probes do fewer steps per second than Qwen3.5 batch-8, but each step carries far more useful tokens and much less warmup overhead.
5. **A completed long run is not automatically a useful speed datapoint.** The 2000-step Qwen3.5 run proved the pipeline could finish, but the short probes are still the only clean apples-to-apples throughput reference in this file.
6. **Tuning `learning_rate` and LoRA `rank` for speed did not pay off.** In the 03-05 full-run sweep (batch 4, seq 512, packed), throughput stayed in essentially the same band.
7. **Forcing no-thinking did not close the quality gap by itself.** The no-thinking parity/ultra runs landed at `74%` and `83%`, matching earlier configuration-driven behavior bands.
8. **Template-level no-thinking enforcement also did not move parity quality.** After replacing the tokenizer chat template with a no-thinking template and hard-failing on `<think>` in prompts, parity remained `74%`.
9. **Promoting checkpoints by eval loss alone did not match benchmark reality.** C19 is the concrete failure case (`checkpoint-600` vs `checkpoint-1200`).
10. **Missing throughput capture on some runs (e.g., SmolLM3) made comparisons weaker.** This is a logging/process gap, not a model limitation.
11. **Qwen3.5 short smokes did not produce usable quality.** Even after fixing loader/runtime issues, 50-step runs only reached `22%` (2B) and `30%` (4B) on core23.

## Root Causes of Remaining Waste

1. **Legacy Unsloth Qwen3.5 path was not clean text-only LM.** It routed through processor/VLM handling and blocked useful packing behavior for text-only SFT comparisons.
2. **FlashAttention2 is still off.** Successful runs still print `FA2 = False`.
3. **The first step is dominated by compile/warmup.** Short probes always look worse in averaged trainer metrics than the actual late-step rate.
4. **Masking density is low.** Only `14/128` active labels in the sample check (~`10.9%`), so a lot of tokens are still paying forward-pass cost without contributing to loss.
5. **Qwen3.5 remains low-leverage for this task on the current recipe.** HF text-only loading now works, but early quality remains far below the Qwen3 baseline path, so additional spend here has weak ROI.
6. **HF runs with frequent eval/save trade throughput for observability.** C19 (`eval_steps=50`, `save_steps=100`) reports lower steps/s than C18.
7. **Checkpoint benchmarking is still partially manual.** Until this is automated, late-checkpoint wins can be missed or found too late.
8. **Compute ledger instrumentation is inconsistent across runs.** We need standardized capture for `train_steps_per_second`, runtime, and checkpoint-level benchmark deltas.

---

## Current Best Speed Profiles

### Best raw step rate (legacy Unsloth probe)

- **Run:** C4
- **Config:** `seq=512`, `batch=8`, `accum=1`, `grad_ckpt=off`, no eval, no save
- **Observed steady-state:** `~3.8-4.1 it/sec`
- **Why it wins:** Better GPU saturation than batch 4 without the per-step slowdown seen at batch 12

### Best reported full-run speed (current HF+PEFT stack)

- **Run:** C18
- **Config:** `Qwen/Qwen3-4B-Instruct-2507`, `seq=512`, `batch=4`, `accum=1`, `grad_ckpt=on`, `2000` steps
- **Reported train steps/s:** `3.326`
- **Why it matters:** This is the highest logged full-run throughput on the current parity stack (from `2026-03-05_1912` log).

### Best quality-aligned full run with known throughput (current stack)

- **Run:** C19 + checkpoint sweep
- **Config:** `Qwen/Qwen3-4B-Instruct-2507`, v5 data, `system_prompt_mode=v2`, `seq=512`, `batch=4`, `grad_ckpt=on`, `1200` steps
- **Reported train steps/s:** `2.456`
- **Quality sidecar:** merged best-checkpoint (`checkpoint-600`) scored `87%`, while merged `checkpoint-1200` scored `100%` on core23.

### Historical full-run Unsloth packed profile

- **Packed recipe runs:** C9-C14 family (`164` steps, packed, export on)
- **Observed steady-state:** generally `~2.2-2.4 it/sec`
- **Reported train steps/s range:** `1.782` to `2.009`
- **No-thinking parity/ultra runs:** C15-C16 (`packing=False`, export on) ran around `~3.8-4.1 it/sec`, but this is not directly comparable to packed runs because effective tokens-per-step differ.

### SmolLM3 checkpoint behavior (latest cloud run)

- **Run:** C20
- **Config:** `HuggingFaceTB/SmolLM3-3B`, `seq=512`, `batch=4`, `1200` steps, v5 + v2 prompt
- **Throughput note:** train steps/s was not captured in logs yet.
- **Checkpoint sidecar:** core23 stayed `87%` at both merged-best and `checkpoint-1200`; broad58 improved from `48.3%` to `53.5%` at `checkpoint-1200`.

### Qwen3.5 text-only smoke profile (latest cloud run)

- **Runs:** C23 (2B), C24 (4B)
- **Config:** `seq=512`, `batch=4`, `accum=1`, `grad_ckpt=on`, `50` steps, v5 data, forced v2 prompt
- **Reported train steps/s:** `0.777` (2B), `1.035` (4B)
- **Quality sidecar:** core23 moved from base `9% -> 22%` (2B) and `13% -> 30%` (4B) after 50 steps; still far from usable parity.

---

## Final Recommendation

1. **Use pure HF+PEFT (`spoke/cloud/train_hf.py`) with official `Qwen/Qwen3-4B-Instruct-2507` as the default cloud quality stack.**
2. **Treat Unsloth runs as compute probes and fallback experiments**, not the primary parity path.
3. **Always benchmark at least two late checkpoints before promotion** (`best_by_eval` + final checkpoint minimum).
4. **Standardize metrics capture per run** (train steps/s, runtime, checkpoint benchmark scores) to keep this ledger comparable.
5. **De-prioritize Qwen3.5 spend until a stronger training recipe is defined.** Loader/runtime blockers are fixed, but current quality slope is weak versus Qwen3 at the same cost envelope.

---

## Next High-Signal Experiments

1. **Add an HF compute probe mode** (`eval=0`, `save=0`, `export_merged=False`, `50` steps) and run `batch=4/6/8` on Qwen3 for apples-to-apples throughput curves.
2. **Automate post-train checkpoint sweeps** (`600/800/1000/1200` where present) and auto-append core23 + broad58 results.
3. **Backfill missing SmolLM3 throughput metrics** from run summaries so C20 is fully comparable in this ledger.
4. **Only continue FA2 work with prebuilt/cached images**; avoid source builds inside worker startup.
