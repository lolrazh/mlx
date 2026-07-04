"""Granite 4.0 1B Speech baseline evaluation.

Uses mlx-audio's MLX implementation of IBM Granite Speech (granite_speech).
Conformer audio encoder + Granite LLM decoder, 4-bit quantized.

Requires mlx-audio >= 0.4.0 (the 0.3.1 STT registry has no granite_speech type).

Granite is a speech-LM that needs a transcription prompt. mlx-audio's default
prompt ("can you transcribe the speech into a written format?") is applied when
we pass NO `language` argument. Passing `language=...` instead builds a
"Translate the speech to <lang>." prompt, which is the wrong task — so we do
NOT pass language and rely on the default transcription prompt.

Usage:
    python -m moonshine.baselines.granite_baseline                    # quick test (10 samples)
    python -m moonshine.baselines.granite_baseline --full             # full LibriSpeech test-clean
    python -m moonshine.baselines.granite_baseline --max-samples 250  # custom sample count
"""

import argparse

import numpy as np

from moonshine.eval.run_eval import evaluate
from moonshine.eval.datasets import LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER

MODEL = "mlx-community/granite-4.0-1b-speech-4bit"
MODEL_NAME = "granite-4.0-1b-speech-4bit"

# Explicit transcription prompt (matches mlx-audio's default for granite_speech).
TRANSCRIBE_PROMPT = "can you transcribe the speech into a written format?"


def make_transcribe_fn(model_path=MODEL):
    """Create a transcribe function compatible with the eval runner."""
    from mlx_audio.stt.utils import load_model

    print(f"Loading Granite Speech from {model_path}...")
    model = load_model(model_path)

    # Count params
    import mlx.utils
    params = mlx.utils.tree_flatten(model.parameters())
    total = sum(v.size for _, v in params)

    from collections import Counter
    dtype_counts = Counter(str(v.dtype) for _, v in params)
    dtype_str = ", ".join(f"{dt}" for dt in dtype_counts)
    print(f"Loaded: {total/1e6:.1f}M params ({dtype_str}, MLX)")

    # Warm-up: one dummy inference (transcription prompt, no language=translate).
    dummy = np.zeros(16000, dtype=np.float32)
    _ = model.generate(dummy, prompt=TRANSCRIBE_PROMPT, max_tokens=1)
    print("Warm-up complete.")

    def transcribe(audio_array, sample_rate):
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)
        else:
            audio_array = audio_array.astype(np.float32)

        # NOTE: do NOT pass language= (that triggers translation prompt).
        result = model.generate(audio_array, prompt=TRANSCRIBE_PROMPT)
        return result.text.strip()

    return transcribe


def main():
    parser = argparse.ArgumentParser(description="Granite 4.0 1B Speech evaluation")
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
        print(f"  Summary: Granite 4.0 1B Speech (MLX, 4-bit)")
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
