"""Granite Speech 4.1 2B NAR baseline evaluation.

Uses mlx-audio's MLX implementation of IBM Granite Speech 4.1 NAR
(granite_speech_nar): Conformer CTC encoder + projector + bidirectional
Granite editor, single-pass non-autoregressive.

The NAR model's generate() accepts a numpy/mlx array directly (see
_load_waveform) and needs NO prompt or language arg.

Usage:
    python -m moonshine.baselines.granite_nar_baseline                    # quick test (10 samples)
    python -m moonshine.baselines.granite_nar_baseline --max-samples 250
    python -m moonshine.baselines.granite_nar_baseline --model <path> --model-name <name>
"""

import argparse

import numpy as np

from moonshine.eval.run_eval import evaluate
from moonshine.eval.datasets import LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER

MODEL = "mlx-community/granite-speech-4.1-2b-nar-mlx-5bit"
MODEL_NAME = "granite-speech-4.1-2b-nar-5bit"


def _load_model(model_path, full_quant=False):
    """Load a NAR model. If full_quant=True, override the stock
    `editor.`-only quant predicate so encoder+projector+editor layers that were
    saved quantized (have `.scales`) all get rebuilt as QuantizedLinear."""
    from mlx_audio.stt.utils import load_model
    from mlx_audio.stt.models.granite_speech_nar.granite_speech_nar import Model

    if not full_quant:
        return load_model(model_path)

    orig = Model.model_quant_predicate
    # Return True everywhere; apply_quantization's get_class_predicate then falls
    # through to the `f"{p}.scales" in weights` test, so exactly the layers that
    # were saved quantized get rebuilt as QuantizedLinear.
    Model.model_quant_predicate = lambda self, p, m: True
    try:
        return load_model(model_path)
    finally:
        Model.model_quant_predicate = orig


def make_transcribe_fn(model_path=MODEL, smoke_print=0, full_quant=False):
    """Create a transcribe function compatible with the eval runner."""
    print(f"Loading Granite Speech NAR from {model_path}...")
    model = _load_model(model_path, full_quant=full_quant)

    import mlx.utils
    from collections import Counter
    params = mlx.utils.tree_flatten(model.parameters())
    total = sum(v.size for _, v in params)
    dtype_counts = Counter(str(v.dtype) for _, v in params)
    print(f"Loaded: {total/1e6:.1f}M params ({', '.join(dtype_counts)}, MLX)")

    # Warm-up
    dummy = np.zeros(16000, dtype=np.float32)
    _ = model.generate(dummy)
    print("Warm-up complete.")

    state = {"printed": 0}

    def transcribe(audio_array, sample_rate):
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)
        else:
            audio_array = audio_array.astype(np.float32)
        result = model.generate(audio_array)
        text = result.text.strip()
        if state["printed"] < smoke_print:
            print(f"  [HYP {state['printed']}] {text!r}")
            state["printed"] += 1
        return text

    return transcribe


def main():
    parser = argparse.ArgumentParser(description="Granite Speech 4.1 NAR evaluation")
    parser.add_argument("--dataset", choices=["librispeech-clean", "librispeech-other", "all"], default="librispeech-clean")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--smoke-print", type=int, default=0, help="Print first N (hyp) transcripts")
    parser.add_argument("--full-quant", action="store_true", help="Rebuild ALL saved-quantized layers (encoder+projector+editor) as QuantizedLinear")
    args = parser.parse_args()

    if args.max_samples is None and not args.full:
        args.max_samples = 10

    shuffle = (args.max_samples is not None) and (not args.no_shuffle)

    transcribe_fn = make_transcribe_fn(args.model, smoke_print=args.smoke_print, full_quant=args.full_quant)

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
            model_name=args.model_name,
            max_samples=args.max_samples,
            shuffle=shuffle,
        )


if __name__ == "__main__":
    main()
