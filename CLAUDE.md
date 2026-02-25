# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MLX learning and model porting project on Apple Silicon (M4, 24 GB unified memory). Started as an educational repo, now transitioning into a real project: porting Moonshine ASR to MLX.

## Setup

- Python 3.11 virtualenv at `.venv/`
- Key packages: `mlx` (0.30.6), `mlx-lm` (0.30.7), `numpy`, `transformers`
- Activate: `source .venv/bin/activate`
- MNIST data in `/tmp/` as gzip files (pre-downloaded)
- HuggingFace model cache at `~/.cache/huggingface/hub/`

## Running

```bash
source .venv/bin/activate
python 1-mnist/03_train.py            # MNIST training
python 2-gpt/05_train.py              # Train character-level GPT on Shakespeare
python 2-gpt/07_break_things.py       # Ablation experiments
python 3-real-models/01_inspect_gpt2.py  # Inspect real GPT-2 weights
```

Scripts run from project root or from within their lesson directory.

## Structure

### Completed learning exercises
- `1-mnist/` — MLP digit classifier. Data loading → model → training loop (SGD, cross_entropy).
- `2-gpt/` — Character-level GPT on Shakespeare. Full transformer from scratch (embeddings, multi-head attention, MLP, residual connections, layer norm). Includes training, generation, and ablation experiments.
- `3-real-models/` — Inspecting real HuggingFace models. GPT-2 weight inspection, 4-bit quantization conversion.

### Active project (next)
**Moonshine ASR → MLX port.** Porting `UsefulSensors/moonshine-streaming-medium` (245M params) to run natively in MLX. Non-streaming single-pass mode for best accuracy.

Phases:
1. Baseline: Whisper Large v3 via mlx-whisper + WER eval pipeline (LibriSpeech, Common Voice)
2. Port: Reimplement Moonshine architecture in MLX (preprocessor → encoder → adapter → decoder), write weight sanitize function
3. Optimize: Quantization experiments (8-bit, 4-bit), KV cache, memory profiling, throughput tuning

Reference implementations: `mlx-examples/whisper/`, `mlx-lm/models/gpt2.py`

## Teaching / Communication Style

- Explain concepts with analogies before showing code
- Walk through code chunk by chunk — don't dump big files without explanation
- The user has visual/node-based ML experience (Grasshopper) and knows 3b1b concepts
- The user has now built models from scratch (MNIST MLP, GPT) and can read production model code
- Keep it conversational — pause for questions between steps

## Key Concepts the User Knows

- Arrays, shapes, dtypes, matmul
- nn.Module, Linear layers, embeddings (token + positional)
- Multi-head attention, transformer blocks, residual connections
- Training loops: loss functions (cross_entropy), optimizers (SGD, Adam), `nn.value_and_grad`
- Model inspection: safetensors, config.json, weight name mapping
- Quantization basics: float16 → 4-bit, size/speed tradeoffs
- `mlx_lm.convert` pipeline

## Hardware

- Apple M4, 24 GB unified memory, 120 GB/s memory bandwidth
- Whisper Large v3 float16: ~3.1 GB (fits easily)
- Moonshine streaming medium float16: ~0.49 GB
