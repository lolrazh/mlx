"""Parakeet TDT V2 baseline evaluation (4-bit quantized).

Loads a LOCAL 4-bit MLX checkpoint of NVIDIA's Parakeet TDT 0.6B v2, built by
moonshine/recipes/parakeet/quantize_v2_8bit.py --bits 4 (nn.quantize group_size=64,
bits=4). Only nn.Linear/nn.Embedding are quantized; Conv1d/BatchNorm/LSTM stay bf16.

Cold load in this process, so the eval's RAM number reflects the quantized model.

Usage:
    python -m moonshine.baselines.parakeet_v2_4bit_baseline                    # quick test (10 samples)
    python -m moonshine.baselines.parakeet_v2_4bit_baseline --full             # full LibriSpeech test-clean
    python -m moonshine.baselines.parakeet_v2_4bit_baseline --max-samples 250  # custom sample count
"""

import argparse
import json

import mlx.core as mx
import numpy as np

from moonshine.eval.run_eval import evaluate
from moonshine.eval.datasets import LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER

MODEL_DIR = "moonshine/models/parakeet-tdt-0.6b-v2-4bit"
GROUP_SIZE = 64
BITS = 4


def make_transcribe_fn(model_dir=MODEL_DIR):
    """Create a transcribe function compatible with the eval runner."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from parakeet_mlx.utils import from_config
    from parakeet_mlx.audio import get_logmel

    print(f"Loading Parakeet TDT V2 4-bit from {model_dir}...")
    cfg = json.load(open(f"{model_dir}/config.json"))
    model = from_config(cfg)
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)
    model.load_weights(f"{model_dir}/model.safetensors")

    # Match from_pretrained: cast non-quantized fp32 params to bf16.
    # Leave integer (packed quantized) arrays untouched.
    cw = dict(tree_flatten(model.parameters()))
    cw = [
        (k, v.astype(mx.bfloat16) if v.dtype == mx.float32 else v)
        for k, v in cw.items()
    ]
    model.update(tree_unflatten(cw))
    model.eval()

    # Count params
    params = tree_flatten(model.parameters())
    total = sum(v.size for _, v in params)
    print(f"Loaded: {total/1e6:.1f}M params (4-bit quantized linears + bf16, MLX)")

    # Warm-up: one dummy inference
    dummy_mel = get_logmel(mx.zeros((16000,)), model.preprocessor_config)
    _ = model.generate(dummy_mel)
    print("Warm-up complete.")

    def transcribe(audio_array, sample_rate):
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)

        # Convert numpy audio → MLX mel spectrogram → transcribe
        audio_mx = mx.array(audio_array)
        mel = get_logmel(audio_mx, model.preprocessor_config)
        result = model.generate(mel)[0]

        # Join tokens into text, strip leading/trailing whitespace
        text = "".join(t.text for t in result.tokens).strip()
        return text

    return transcribe


def main():
    parser = argparse.ArgumentParser(description="Parakeet TDT V2 (4-bit) evaluation")
    parser.add_argument(
        "--dataset",
        choices=["librispeech-clean", "librispeech-other", "all"],
        default="librispeech-clean",
        help="Which dataset to evaluate on (default: librispeech-clean)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full evaluation on all LibriSpeech splits",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling (on by default when using --max-samples)",
    )
    args = parser.parse_args()

    # Quick test mode: default to 10 samples if nothing specified
    if args.max_samples is None and not args.full:
        print("Tip: Running with --max-samples 10 for a quick test.")
        print("     Use --full for complete evaluation, or --max-samples N for custom.\n")
        args.max_samples = 10

    shuffle = (args.max_samples is not None) and (not args.no_shuffle)

    transcribe_fn = make_transcribe_fn()

    if args.full:
        datasets = [LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER]
        args.max_samples = None
        shuffle = False
    elif args.dataset == "all":
        datasets = [LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER]
    else:
        datasets = [args.dataset]

    results = []
    for ds in datasets:
        result = evaluate(
            transcribe_fn=transcribe_fn,
            dataset_name=ds,
            model_name="parakeet-tdt-v2-4bit",
            max_samples=args.max_samples,
            shuffle=shuffle,
        )
        results.append(result)

    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  Summary: Parakeet TDT V2 4-bit (MLX)")
        print(f"{'=' * 70}")
        print(f"  {'Dataset':<25} {'WER':>8} {'RTF':>8} {'Peak RAM':>10}")
        print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 10}")
        for r in results:
            print(
                f"  {r['dataset']:<25} "
                f"{r['wer']:>7.2%} "
                f"{r['real_time_factor']:>7.2f}x "
                f"{r['peak_memory_gb']:>8.2f} GB"
            )
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
