"""Moonshine Medium MLX baseline evaluation.

Native MLX port of Moonshine ASR, running entirely on Apple Silicon
without PyTorch.

Usage:
    python -m moonshine.baselines.mlx_baseline                    # quick test (10 samples)
    python -m moonshine.baselines.mlx_baseline --full             # full LibriSpeech test-clean
    python -m moonshine.baselines.mlx_baseline --max-samples 250  # custom sample count
"""

import argparse

import mlx.core as mx
import numpy as np
from transformers import PreTrainedTokenizerFast

from moonshine.eval.run_eval import evaluate
from moonshine.eval.datasets import LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER
from moonshine.model import load_model, generate, pad_audio

MODEL = "UsefulSensors/moonshine-streaming-medium"


def make_transcribe_fn(model_path=MODEL, dtype=mx.float32):
    """Create a transcribe function compatible with the eval runner."""
    from huggingface_hub import snapshot_download

    # Download model if needed, get local path
    local_path = snapshot_download(model_path)

    dtype_name = "fp16" if dtype == mx.float16 else "fp32"
    print(f"Loading Moonshine MLX ({dtype_name}) from {local_path}...")
    model, args = load_model(local_path, dtype=dtype)

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(local_path)

    # Count params
    import mlx.utils
    params = mlx.utils.tree_flatten(model.parameters())
    total = sum(v.size for _, v in params)
    print(f"Loaded: {total/1e6:.1f}M params ({dtype_name}, MLX)")

    # Warm-up: one dummy inference to compile MLX graphs
    dummy = mx.zeros((1, 16000))
    _ = generate(model, dummy, max_tokens=1)
    print("Warm-up complete.")

    def transcribe(audio_array, sample_rate):
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)

        audio = pad_audio(audio_array)
        token_ids = generate(model, audio, max_tokens=512)
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return text.strip()

    return transcribe


def main():
    parser = argparse.ArgumentParser(description="Moonshine Medium MLX evaluation")
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
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Model weight dtype (default: fp32)",
    )
    args = parser.parse_args()

    # Quick test mode: default to 10 samples if nothing specified
    if args.max_samples is None and not args.full:
        print("Tip: Running with --max-samples 10 for a quick test.")
        print("     Use --full for complete evaluation, or --max-samples N for custom.\n")
        args.max_samples = 10

    shuffle = (args.max_samples is not None) and (not args.no_shuffle)

    dtype = mx.float16 if args.dtype == "fp16" else mx.float32
    transcribe_fn = make_transcribe_fn(dtype=dtype)

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
        dtype_suffix = f"-{args.dtype}" if args.dtype != "fp32" else ""
        result = evaluate(
            transcribe_fn=transcribe_fn,
            dataset_name=ds,
            model_name=f"moonshine-medium-mlx{dtype_suffix}",
            max_samples=args.max_samples,
            shuffle=shuffle,
        )
        results.append(result)

    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  Summary: Moonshine Medium (MLX)")
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
