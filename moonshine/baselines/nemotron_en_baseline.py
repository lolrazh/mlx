"""Nemotron Speech Streaming EN 0.6B baseline (English-only, via nemotron-asr-mlx).

Evaluates nvidia/nemotron-speech-streaming-en-0.6b through the dedicated MLX port
(pip nemotron-asr-mlx, weights dboris/nemotron-asr-mlx, bf16). This is the
English-only sibling of nemotron-3.5-asr-streaming-0.6b: same cache-aware
streaming FastConformer-RNNT recipe, no language-prompt conditioning. Greedy
decode to stay comparable with the other baselines.

Usage:
    python -m moonshine.baselines.nemotron_en_baseline --max-samples 250
"""

import argparse

import numpy as np

from moonshine.eval.run_eval import evaluate
from moonshine.eval.datasets import LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER

REPO = "dboris/nemotron-asr-mlx"


def make_transcribe_fn():
    from nemotron_asr_mlx import from_pretrained

    print(f"Loading {REPO} (bf16)...")
    model = from_pretrained(REPO)

    _ = model.transcribe(np.zeros(16000, dtype=np.float32))
    print("Warm-up complete.")

    def transcribe(audio_array, sample_rate):
        audio_array = np.asarray(audio_array, dtype=np.float32)
        return model.transcribe(audio_array).text.strip()

    return transcribe


def main():
    parser = argparse.ArgumentParser(description="Nemotron EN streaming evaluation")
    parser.add_argument(
        "--dataset",
        choices=["librispeech-clean", "librispeech-other", "all"],
        default="librispeech-clean",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    if args.max_samples is None and not args.full:
        print("Tip: Running with --max-samples 10 for a quick test.")
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

    for ds in datasets:
        evaluate(
            transcribe_fn=transcribe_fn,
            dataset_name=ds,
            model_name="nemotron-speech-en-bf16",
            max_samples=args.max_samples,
            shuffle=shuffle,
        )


if __name__ == "__main__":
    main()
