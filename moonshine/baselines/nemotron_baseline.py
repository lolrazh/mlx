"""Nemotron 3.5 ASR streaming baseline evaluation (MLX, any local checkpoint).

Evaluates mlx-audio's nemotron_asr port (cache-aware streaming FastConformer-RNNT,
nvidia/nemotron-3.5-asr-streaming-0.6b) on the standard harness. The --lookahead
flag selects att_context_size, i.e. the streaming latency operating point; per the
port, offline decode is token-identical to streamed decode at the native chunk
size, so these WERs are the true streaming-quality numbers.

Usage:
    python -m moonshine.baselines.nemotron_baseline                      # 10 samples, bf16
    python -m moonshine.baselines.nemotron_baseline --max-samples 250
    python -m moonshine.baselines.nemotron_baseline --bits 6 --max-samples 250
    python -m moonshine.baselines.nemotron_baseline --lookahead 320 --max-samples 250
"""

import argparse

import mlx.core as mx
import numpy as np

from moonshine.eval.run_eval import evaluate
from moonshine.eval.datasets import LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER

BASE_DIR = "moonshine/models/nemotron-3.5-asr-streaming-0.6b"
LOOKAHEAD_TO_ACS = {80: [56, 0], 320: [56, 3], 560: [56, 6], 1120: [56, 13]}


def make_transcribe_fn(model_dir, lookahead, language="en-US"):
    from mlx_audio.stt.utils import load_model

    acs = LOOKAHEAD_TO_ACS[lookahead]
    print(f"Loading {model_dir} (lookahead {lookahead}ms, acs={acs})...")
    model = load_model(model_dir)

    # Warm-up
    _ = model.generate(mx.zeros(16000), language=language, att_context_size=acs)
    print("Warm-up complete.")

    def transcribe(audio_array, sample_rate):
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)
        result = model.generate(
            mx.array(audio_array.astype(np.float32)),
            language=language,
            att_context_size=acs,
        )
        return result.text.strip()

    return transcribe


def main():
    parser = argparse.ArgumentParser(description="Nemotron 3.5 ASR evaluation")
    parser.add_argument(
        "--dataset",
        choices=["librispeech-clean", "librispeech-other", "all"],
        default="librispeech-clean",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--bits", type=int, default=None, choices=[4, 5, 6, 8],
        help="Use the N-bit checkpoint (default: bf16)",
    )
    parser.add_argument(
        "--lookahead", type=int, default=1120, choices=sorted(LOOKAHEAD_TO_ACS),
        help="Streaming latency operating point in ms (default: 1120, best WER)",
    )
    args = parser.parse_args()

    if args.max_samples is None and not args.full:
        print("Tip: Running with --max-samples 10 for a quick test.")
        print("     Use --full for complete evaluation, or --max-samples N for custom.\n")
        args.max_samples = 10

    shuffle = (args.max_samples is not None) and (not args.no_shuffle)

    model_dir = BASE_DIR if args.bits is None else f"{BASE_DIR}-{args.bits}bit"
    tag = "bf16" if args.bits is None else f"{args.bits}bit"
    model_name = f"nemotron-3.5-asr-{tag}-la{args.lookahead}"

    transcribe_fn = make_transcribe_fn(model_dir, args.lookahead)

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
            model_name=model_name,
            max_samples=args.max_samples,
            shuffle=shuffle,
        )


if __name__ == "__main__":
    main()
