"""Whisper Large v3 baseline evaluation.

Runs Whisper through our eval pipeline to establish baseline WER numbers.
These are the numbers we'll compare Moonshine against.

Usage:
    python -m moonshine.baselines.whisper_baseline                    # quick test (10 samples)
    python -m moonshine.baselines.whisper_baseline --full             # full LibriSpeech test-clean
    python -m moonshine.baselines.whisper_baseline --dataset all      # all datasets
    python -m moonshine.baselines.whisper_baseline --max-samples 50   # custom sample count
"""

import argparse
import numpy as np
import mlx_whisper

from moonshine.eval.run_eval import evaluate
from moonshine.eval.datasets import LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER, COMMON_VOICE_EN

MODEL = "mlx-community/whisper-large-v3-mlx"


def make_transcribe_fn(model_path=MODEL):
    """Create a transcribe function compatible with the eval runner.

    mlx_whisper.transcribe() expects a file path or numpy array.
    We pass the raw float32 audio array directly.
    """
    def transcribe(audio_array, sample_rate):
        # mlx_whisper expects float32 numpy array
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)

        result = mlx_whisper.transcribe(
            audio_array,
            path_or_hf_repo=model_path,
            language="en",
        )
        return result["text"].strip()

    return transcribe


def main():
    parser = argparse.ArgumentParser(description="Whisper Large v3 baseline evaluation")
    parser.add_argument(
        "--dataset",
        choices=["librispeech-clean", "librispeech-other", "common-voice-en", "all"],
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
    args = parser.parse_args()

    # Quick test mode: default to 10 samples if nothing specified
    if args.max_samples is None and not args.full:
        print("Tip: Running with --max-samples 10 for a quick test.")
        print("     Use --full for complete evaluation, or --max-samples N for custom.\n")
        args.max_samples = 10

    transcribe_fn = make_transcribe_fn()

    if args.full:
        # Full eval on both LibriSpeech splits
        datasets = [LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER]
        args.max_samples = None
    elif args.dataset == "all":
        datasets = [LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER, COMMON_VOICE_EN]
    else:
        datasets = [args.dataset]

    results = []
    for ds in datasets:
        result = evaluate(
            transcribe_fn=transcribe_fn,
            dataset_name=ds,
            model_name="whisper-large-v3",
            max_samples=args.max_samples,
        )
        results.append(result)

    # Print comparison table if multiple datasets
    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  Summary: Whisper Large v3 Baseline")
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
