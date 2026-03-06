# Architecture Research: Gemma 3n Rerun + T5 Exploration

**Date:** 2026-03-06
**Agent:** Claude Opus 4.6
**Status:** 🔄 Ongoing

## User Intention
User was frustrated that no model meets both >90% accuracy AND <2 GB deploy size. Qwen3-4B DWQ = 96% but 2.1 GB; Llama 3B = 87% bf16. Wanted a deep analysis of what's possible: alternative architectures (MoE, diffusion LMs, encoder-decoder), data augmentation, model pruning, and specific next steps. The goal is to find a path to high accuracy at small size, possibly by changing the fundamental approach rather than just trying more decoder-only LLMs.

## What We Accomplished

- ✅ **Deep architecture analysis** — Evaluated MoE, diffusion LMs, encoder-decoder (T5), knowledge distillation, custom architectures, Phi-4, Gemma 3n E4B as options. Ranked by practicality.
- ✅ **DRES paper research** — Found and analyzed "DRES: Benchmarking LLMs for Disfluency Removal" (Teleki et al., Sept 2025, arXiv:2509.20321). Key finding: Phi-4 over-deletes (skip it), Llama-3B gets biggest fine-tuning gain (+23.46 pts), Qwen under-deletes (conservative = good for our use case).
- ✅ **T5/encoder-decoder discovery** — Found that `mlx-examples/t5/` supports T5 inference on MLX (T5-small through T5-11B + Flan-T5). This was previously thought blocked ("mlx-lm has zero encoder-decoder support") but mlx-examples is separate from mlx-lm. T5-base at 4-bit = ~55-70 MB, Flan-T5-large at 4-bit = ~190-240 MB.
- ✅ **T5 ASR correction research** — Found FlanEC (Jan 2025): Flan-T5-base reduces ASR WER 11.8%->10.3%. Also found Jan 2026 paper showing small decoder-only models are 3x worse than encoder-decoder at grammar correction on copy-heavy tasks.
- ✅ **Gemma 3n hyperparameter research** — Found Google's official QLoRA guide recommends lr=2e-4 (our failed run used 1e-5, 20x too low), constant_with_warmup scheduler, warmup_ratio=0.03, max_grad_norm=0.3, weight_decay=0.01, lora_r=16.
- ✅ **Added hyperparameter flags to train_hf.py** — `lr_scheduler_type`, `warmup_ratio`, `weight_decay` now configurable (were hardcoded).
- ✅ **Launched Gemma 3n E2B v2 run** — `spoke-gemma3n-e2b-v4-v2` on Modal with corrected hyperparams. Training confirmed running at ~2.3 it/s.
- ✅ **Created NEXT_RUNS.md** — Documented both paths (Gemma 3n + T5) with full hyperparams, commands, risk assessments.
- ✅ **Updated auto-memory** — Added DRES findings, Gemma 3n hyperparams, T5 MLX support, cloud pipeline updates.
- ✅ **Analyzed Llama v4 benchmark errors** — Confirmed emphasis bold/caps is NOT an issue (v4 data = all CAPS, v3 test = all CAPS). Llama's 4 errors: spell garbling, quote truncation, over-emphasis, emoji positioning.

## Technical Implementation

### train_hf.py changes (3 new params)
```python
# Added to both train() and main() signatures:
lr_scheduler_type: str = "constant",
warmup_ratio: float = 0.0,
weight_decay: float = 0.0,

# Wired into TrainingArguments:
lr_scheduler_type=lr_scheduler_type,
warmup_ratio=warmup_ratio,
weight_decay=weight_decay,
```

### Gemma 3n E2B v2 run config
- lr=2e-4, constant_with_warmup, warmup_ratio=0.03, max_grad_norm=0.3
- weight_decay=0.01, lora_r=16, lora_alpha=16, lora_dropout=0.05
- 1200 steps (~4 epochs), v4 data, v2 system prompt, adamw optimizer
- Early signals: loss=6.532, grad_norm=737 (clipped to 0.3), lr ramping during warmup

**Files Modified:**
- `spoke/cloud/train_hf.py` — Added lr_scheduler_type, warmup_ratio, weight_decay params
- `spoke/NEXT_RUNS.md` — NEW: temporary planning doc for Gemma 3n + T5 runs

**Files Created:**
- `spoke/NEXT_RUNS.md` — Full plan for Path A (Gemma 3n) and Path B (T5)

## Bugs & Issues Encountered
1. **No code bugs this session** — All changes were additive parameter wiring.
2. **Local Llama-T2 used old test set** — The `result_llama3.2-3b+lora_v2.json` benchmark used a test set with `**bold**` emphasis ideals, while v3 test uses CAPS. This explains the 91% vs 87% discrepancy partially (different test sets, not just cloud vs local).

## Key Learnings
- **`mlx-examples/t5/` exists and works** — T5 encoder-decoder inference on MLX is possible via standalone scripts (t5.py, hf_t5.py). Inference only, no training. This was missed in earlier research because we only looked at mlx-lm.
- **T5Gemma != T5** — T5Gemma is Google's newer encoder-decoder model that mlx-lm doesn't support. But the original T5/Flan-T5 runs fine via mlx-examples. Also llama.cpp supports T5 GGUF (PR #8055, use Q5_K_M+).
- **Gemma 3n tokenizer supports enable_thinking=False** — Surprising since it's not a thinking model. The code's try/except path handled this correctly.
- **Google recommends constant scheduler (not cosine) for Gemma QLoRA** — With warmup. This differs from common practice.
- **DRES R4: Avoid reasoning models for text editing** — Phi-4, o4-mini over-delete. Saved us from wasting a cloud run on Phi-4.
- **Encoder-decoder models are 2.25x faster at inference** for copy-heavy tasks (Oct 2025 paper). T5Gemma paper shows +7 pts instruction-tuning improvement over decoder-only at same scale.
- **Small decoder-only models are 3x worse** at grammar correction than encoder-decoder (Jan 2026 paper). Copy-heavy tasks specifically suffer.
- **FlanEC models exist on HuggingFace** — `morenolq/flanec-base-cd`, `morenolq/flanec-large-cd`. Pre-trained for ASR error correction.

## Architecture Decisions
- **Gemma 3n first, T5 second** — Gemma 3n is a quick re-run of an existing pipeline with corrected hyperparams. T5 requires a new training script. Do the easy thing first.
- **1200 steps for Gemma (not 2000)** — With lr=2e-4 (20x higher than previous), model converges faster. Google recommends 3 epochs (~900 steps). We do 4 epochs (1200) for buffer. Checkpoints every 200.
- **constant_with_warmup over cosine** — Google's own recommendation for Gemma QLoRA. Warmup handles initial instability, constant LR after.
- **T5 cloud training over local** — T5-base (220M) could train locally on MPS, but MPS has known precision bugs. Cloud is ~$0.10 and more reliable.
- **Skip Phi-4** — DRES paper confirms reasoning models over-delete on text editing tasks. Not worth a run.

## Ready for Next Session
- ✅ **Gemma 3n E2B v2 training running** — Monitor wandb at spoke/spoke, run `spoke-gemma3n-e2b-v4-v2`. Will need benchmarking after completion.
- 🔧 **T5 training script needed** — `spoke/cloud/train_t5.py` with HF Seq2SeqTrainer. Data format conversion from chat JSONL to seq2seq pairs. Plan in NEXT_RUNS.md.
- 🔧 **T5 MLX inference integration** — Need to test `mlx-examples/t5/` with a fine-tuned model. May need adaptation.
- 🔧 **Benchmark Gemma 3n** — After training completes, run `spoke/cloud/benchmark.py` on best and last checkpoints.
- 🔧 **Delete NEXT_RUNS.md** — After both paths are evaluated.

## Context for Future
This session marked a strategic pivot from "try more decoder-only LLMs" to exploring fundamentally different architectures. T5/encoder-decoder is the most promising new direction — architecturally ideal for copy-heavy text editing, 10-30x smaller than current models. Gemma 3n with corrected hyperparams is the quick-win check. The DRES paper validated our approach and eliminated Phi-4 as an option. Next session should benchmark the Gemma 3n result and build the T5 training pipeline.
