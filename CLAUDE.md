# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Educational ML project for learning to port models to MLX on Apple Silicon (M4, 24 GB). Lessons are numbered directories, each with step-by-step Python scripts.

## Setup

- Python 3.11 virtualenv at `.venv/`
- Key packages: `mlx` (0.30.6), `mlx-lm` (0.30.7), `numpy`, `transformers`
- MNIST data lives in `/tmp/` as gzip files (pre-downloaded)

## Running

```bash
source .venv/bin/activate
python 1-mnist/03_train.py      # run from project root
python 2-gpt/01_look_at_data.py
```

Scripts are meant to run from the project root or from within their lesson directory.

## Structure

- `1-mnist/` — MLP digit classifier. Complete (data → model → training).
- `2-gpt/` — Character-level GPT on Shakespeare. In progress.
- Each lesson has numbered scripts (`01_`, `02_`, ...) that build on each other.
- `data.py` files handle data loading, step scripts handle the teaching.

## Teaching Style

This is a learning repo. The user learns interactively in conversation — code is explained chunk by chunk as it's written. Keep scripts self-contained with clear comments, but the real teaching happens in chat. Don't dump big files without explanation.

- Explain concepts with analogies before showing code.
- Walk through code chunk by chunk — don't just run it and say "see?"
- The user has visual/node-based ML experience (built an MNIST MLP in Grasshopper).
- The user knows 3b1b concepts (weights, matmul, attention, backprop) but is coding models for the first time.

## Progress

- `1-mnist/` — Complete. Data loading, MLP model, training loop with SGD.
- `2-gpt/` — In progress. Data + tokenizer done (01), embeddings done (02). Next: attention.
