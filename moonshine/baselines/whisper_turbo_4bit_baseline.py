"""Whisper large-v3 turbo 4-bit baseline (the model the Spoke app ships).

This is the CONTROL for the Cohere / Granite ASR comparison. It points
mlx_whisper at the turbo 4-bit repo, which is byte-identical to the spokedotso
fork that ships in the app.

Usage:
    python -m moonshine.baselines.whisper_turbo_4bit_baseline                    # quick test (10 samples)
    python -m moonshine.baselines.whisper_turbo_4bit_baseline --max-samples 250  # custom sample count
    python -m moonshine.baselines.whisper_turbo_4bit_baseline --full             # full LibriSpeech splits
"""

import argparse
import os
import numpy as np
import mlx_whisper

from moonshine.eval.run_eval import evaluate
from moonshine.eval.datasets import LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER

MODEL = "mlx-community/whisper-large-v3-turbo-4bit"
MODEL_NAME = "whisper-turbo-4bit"


def _resolve_model_path(repo):
    """Return a local model dir mlx_whisper can load.

    mlx_whisper 0.4.3's load_model looks for `weights.safetensors` / `weights.npz`,
    but this repo ships `model.safetensors`. We snapshot-download the repo and
    expose a `weights.safetensors` symlink so the existing loader picks it up.
    The weights are byte-identical to the shipped repo.
    """
    from huggingface_hub import snapshot_download

    snap = snapshot_download(repo)
    src = os.path.join(snap, "model.safetensors")
    dst = os.path.join(snap, "weights.safetensors")
    if os.path.exists(src) and not os.path.exists(dst):
        try:
            os.symlink(src, dst)
        except OSError:
            import shutil
            shutil.copy(src, dst)
    return snap


def make_transcribe_fn(model_path=MODEL):
    """Create a transcribe function compatible with the eval runner."""
    model_path = _resolve_model_path(model_path)

    def transcribe(audio_array, sample_rate):
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)
        else:
            audio_array = audio_array.astype(np.float32)

        result = mlx_whisper.transcribe(
            audio_array,
            path_or_hf_repo=model_path,
            language="en",
        )
        return result["text"].strip()

    return transcribe


def main():
    parser = argparse.ArgumentParser(description="Whisper turbo 4-bit baseline evaluation")
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
            model_name=MODEL_NAME,
            max_samples=args.max_samples,
            shuffle=shuffle,
        )
        results.append(result)

    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  Summary: Whisper large-v3 turbo (MLX, 4-bit)")
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
