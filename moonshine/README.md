# Moonshine ASR → MLX Port

Porting [UsefulSensors/moonshine-streaming-medium](https://huggingface.co/UsefulSensors/moonshine-streaming-medium) (245M params) to run natively on Apple Silicon via MLX.

## Benchmark Results

| Model | Precision | Dataset | Samples | WER | RTF | Model Size |
|---|---|---|---|---|---|---|
| Whisper Large v3 (MLX) | fp16 | LS test-clean | 250 (shuffled) | 3.02% | 0.34x | 3.08 GB |
| Moonshine Medium (PyTorch+MPS) | fp32 | LS test-clean | 250 (shuffled) | 2.19% | 0.11x | 1.06 GB |
| Moonshine Medium (MLX port) | fp16 | LS test-clean | *pending* | — | — | ~0.53 GB |

*RTF = Real-Time Factor (lower is faster; <1.0 = faster than real-time). All runs on M4 24GB, seed=42.*
*Whisper and Moonshine PyTorch baselines establish targets for the MLX port.*

## Quick Start

```bash
# Activate virtualenv
source .venv/bin/activate

# Quick test (10 samples from LibriSpeech test-clean)
python -m moonshine.baselines.whisper_baseline

# Full evaluation on LibriSpeech
python -m moonshine.baselines.whisper_baseline --full

# Specific dataset with sample limit
python -m moonshine.baselines.whisper_baseline --dataset librispeech-other --max-samples 50
```

## Project Structure

```
moonshine/
├── eval/
│   ├── datasets.py      — download + load LibriSpeech test sets
│   ├── wer.py           — WER computation with text normalization
│   └── run_eval.py      — model-agnostic eval runner
├── baselines/
│   └── whisper_baseline.py  — Whisper Large v3 baseline
├── results/             — JSON result files (auto-created)
└── README.md
```

## Phases

1. **Baseline** (current): Whisper Large v3 via mlx-whisper + WER eval pipeline
2. **Port**: Reimplement Moonshine architecture in MLX
3. **Optimize**: Quantization, KV cache, memory profiling
