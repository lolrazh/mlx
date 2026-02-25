# Moonshine ASR → MLX Port

Porting [UsefulSensors/moonshine-streaming-medium](https://huggingface.co/UsefulSensors/moonshine-streaming-medium) (245M params) to run natively on Apple Silicon via MLX.

## Benchmark Results

| Model | Dataset | WER | RTF | Peak RAM |
|---|---|---|---|---|
| Whisper Large v3 | LS test-clean | *pending* | — | — |
| Whisper Large v3 | LS test-other | *pending* | — | — |
| Whisper Large v3 | CV English test | *pending* | — | — |

*Results will be filled in after running baselines.*

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
│   ├── datasets.py      — download + load LibriSpeech and Common Voice
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
