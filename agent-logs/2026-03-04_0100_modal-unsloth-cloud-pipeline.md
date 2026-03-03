# Modal + Unsloth Cloud Training Pipeline for Qwen3.5-4B

**Date:** 2026-03-04
**Agent:** Claude Opus 4.6
**Status:** 🔄 Ongoing (pipeline working, Qwen35-T1 training in progress)

## User Intention
User wanted to break free from the M4 24GB local training bottleneck (~2 it/sec, hours per run) by building a cloud GPU training pipeline. The goal: train Qwen3.5-4B (the latest Qwen model) on Modal's L40S GPU using Unsloth for optimized LoRA, matching the exact T2-v4 hyperparameters that achieved 100% accuracy on Qwen3-4B. The full pipeline needed to be end-to-end: upload data → train on cloud → download merged model → convert to MLX locally → benchmark.

## What We Accomplished
- ✅ **Cloud training pipeline** — Three scripts: `spoke/cloud/upload_data.py`, `spoke/cloud/train.py`, `spoke/cloud/download_model.py`
- ✅ **Data uploaded to Modal** — 1201 train + 20 valid + 23 test examples on `spoke-training-data` volume
- ✅ **Unsloth EOS_TOKEN bug resolved** — Root-caused and fixed the `<EOS_TOKEN>` placeholder crash after ~6 failed Modal runs
- ✅ **Qwen35-T1 training launched** — Running on Modal L40S with T2-v4 hyperparams (r=8, lr=1e-5, 2000 steps, adam, v4 data)
- ✅ **All learnings documented** — MEMORY.md (12 Unsloth lessons), LEDGER.md (cloud training section + queue updates)
- ⚠️ **Training results pending** — Run is in progress; download + MLX convert + benchmark not yet done

## Technical Implementation

**Architecture:** Modal serverless GPU → Unsloth optimized LoRA → merged bf16 export → download → `mlx_lm.convert` → `mlx_lm.dwq` locally

**Modal setup:**
- 3 Volumes: `spoke-model-cache` (HF cache), `spoke-training-data` (JSONL), `spoke-output` (merged models)
- 1 Secret: `wandb-secret` (WANDB_API_KEY)
- GPU: L40S (48 GB VRAM) — Qwen3.5-4B bf16 LoRA uses ~10 GB
- Image: `debian_slim` + `unsloth` + `wandb` + `trl==0.22.2` (pinned)

**Key code pattern (the working EOS fix):**
```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Qwen3.5-4B", ...)
tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")  # Fixes <EOS_TOKEN>
```

**Training config (matches T2-v4):**
- LoRA r=8, alpha=16, dropout=0.05, all target modules
- `adamw_torch` with `weight_decay=0.0` (= plain Adam)
- `lr_scheduler_type="constant"`, lr=1e-5, batch=4, 2000 steps
- `train_on_responses_only` with ChatML markers
- WandB logging enabled

**Files Modified:**
- `spoke/cloud/train.py` — Created, then 3 rounds of fixes (EOS, optimizer, columns)
- `spoke/cloud/upload_data.py` — Created, fixed double-slash path bug
- `spoke/cloud/download_model.py` — Created (untested, pending training completion)
- `spoke/LEDGER.md` — Added cloud training section, updated queue

## Bugs & Issues Encountered

1. **`git` not found in `debian_slim`** — Modal's base image has no git
   - **Fix:** Added `.apt_install("git")` to image definition

2. **pip dependency loop with `unsloth[cu124-ampere-torch250]`** — Old torch/xformers pins conflicted
   - **Fix:** Use plain `pip install "unsloth"` (no extras), let it resolve deps

3. **`SFTConfig` rejects `max_seq_length`** — Removed in TRL 0.24+
   - **Fix:** Only set `max_seq_length` in `FastLanguageModel.from_pretrained`, not SFTConfig

4. **`paged_adam_32bit` not a valid OptimizerNames** — Not in transformers enum
   - **Fix:** Use `adamw_torch` with `weight_decay=0.0` (mathematically identical to Adam)

5. **`<EOS_TOKEN>` not in vocabulary** (THE BIG ONE) — Unsloth replaces eos_token with placeholder string that isn't in the tokenizer vocab. TRL's SFTTrainer validates this on init.
   - **Root cause:** Unsloth's `FastLanguageModel.from_pretrained` monkey-patches tokenizer loading, replacing the real EOS (`<|im_end|>`) with `<EOS_TOKEN>` placeholder
   - **Fix:** Call `get_chat_template(tokenizer, chat_template="qwen3-instruct")` after loading — this remaps the EOS back to the correct token. Also pin `trl==0.22.2` (all Unsloth notebooks use this exact version)
   - **What didn't work:** Loading tokenizer before Unsloth import, manually setting eos_token, extracting inner tokenizer from Processor, different TRL versions

6. **Data collator crash on nested `messages` column** — `messages` is a list of dicts (role/content), can't be converted to tensors for batching
   - **Fix:** `dataset.remove_columns([non-text cols])` after formatting with `apply_chat_template`

7. **`causal_conv1d` build failure** — Requires nvcc (CUDA compiler) not in `debian_slim`
   - **Fix:** Skip it — optional performance dep for Qwen3.5 hybrid layers. Model works without it.

8. **Double-slash in upload paths** — `REMOTE_DIR="/"` + `f"/{f}"` = `//train.jsonl`
   - **Fix:** Hardcode `f"/{f}"` directly

## Key Learnings

- **Unsloth's `get_chat_template` is mandatory, not optional** — The Qwen3-4B-Instruct notebook includes this step; we initially skipped it thinking it was just formatting. It's actually the EOS token fix.
- **Pin TRL exactly** — `trl<0.24.0` resolves to 0.24.0 (latest), not 0.23.x. Unsloth notebooks use `pip install --no-deps trl==0.22.2` for a reason. The `--no-deps` prevents TRL from pulling conflicting transformers versions.
- **Qwen3.5-4B is a VLM** — Despite being used for text, it's architecturally a Vision-Language Model with hybrid Gated DeltaNet + standard attention layers (32 layers). The only official Unsloth notebook is the Vision one (`FastVisionModel`). For text-only fine-tuning, use `FastLanguageModel` + `get_chat_template` (from the Qwen3 text notebook pattern).
- **Use `tokenizer=tokenizer` not `processing_class=tokenizer`** — Unsloth's pinned TRL 0.22.2 expects the old kwarg name. Newer TRL deprecated it.
- **Always read the actual Unsloth notebooks** — The docs page gives a high-level overview but misses critical setup steps (like `get_chat_template`). The Colab notebooks on GitHub are the ground truth.
- **Modal image builds are slow (~3 min) and expensive** — Each failed run burns compute. Do thorough research before each attempt, not rapid-fire trial and error.

## Architecture Decisions

- **`FastLanguageModel` over `FastVisionModel`** — Qwen3.5-4B is a VLM, but we're doing text-only fine-tuning. The docs recommend `FastLanguageModel` for text, and the Vision path has very different SFTConfig (custom data collator, skip_prepare_dataset). Text path is simpler and matches our existing data format.
- **Merged bf16 export, not raw adapters** — PEFT adapter format differs from mlx-lm (different key naming, transposed tensors). Exporting merged bf16 via `save_pretrained_merged` is the robust path — then `mlx_lm.convert` works directly.
- **`adamw_torch` with wd=0 instead of `adamw_8bit`** — L40S has 48 GB VRAM, no need for 8-bit optimizer states. Full precision Adam matches our T2-v4 local config exactly. Our experiments showed Adam > AdamW for downstream quantization.
- **Skip `causal_conv1d` and `flash-linear-attention`** — These optimize Qwen3.5's hybrid layers but require nvcc to build. The model works without them (slower kernels but functional). Not worth switching to a CUDA devel base image for a single training run.

## Ready for Next Session
- ✅ **Cloud pipeline is working** — Can run new training experiments with `modal run spoke/cloud/train.py --run-name <name> --max-steps N --learning-rate X`
- 🔧 **Download + convert + benchmark** — When Qwen35-T1 finishes: `python spoke/cloud/download_model.py --run-name spoke-qwen35-t1` → `mlx_lm.convert` → benchmark
- 🔧 **Update LEDGER with results** — Fill in accuracy, latency, and comparison to Qwen3-4B T2-v4 (100%)

## Context for Future
This cloud pipeline unblocks experimentation that was impractical locally (hours per run on M4). The key question Qwen35-T1 will answer: does the newer Qwen3.5-4B architecture match or beat Qwen3-4B's 100% accuracy? If yes, it may also quantize better (hybrid architecture). The pipeline is reusable for any future model on Unsloth — just change `--model-name`. Next high-value experiments: rsLoRA (r=16), expanded test set (50+ examples), or trying Qwen3.5's larger variants (27B with QLoRA) on the same pipeline.
