"""Granite Speech 4.1 2B (AR) baseline evaluation.

Loads via mlx-audio's `granite_speech` loader (the 4.0 AR port also loads the
4.1-2b AR checkpoint). Needs the transcription PROMPT, not a language arg.

Usage:
    python -m moonshine.baselines.granite_ar_baseline --model <path> --model-name <name> --max-samples 250
"""

import argparse
import numpy as np

from moonshine.eval.run_eval import evaluate
from moonshine.eval.datasets import LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER

MODEL = "ibm-granite/granite-speech-4.1-2b"
MODEL_NAME = "granite-4.1-2b-ar"
TRANSCRIBE_PROMPT = "can you transcribe the speech into a written format?"


def _load_model(model_path, full_quant=False):
    from mlx_audio.stt.utils import load_model
    from mlx_audio.stt.models.granite_speech.granite_speech import Model
    if not full_quant:
        return load_model(model_path)
    orig = Model.model_quant_predicate
    # Stock AR predicate excludes encoder/projector. Return True everywhere so
    # apply_quantization falls through to the `f"{p}.scales" in weights` test and
    # rebuilds exactly the saved-quantized layers (encoder+projector+LM).
    Model.model_quant_predicate = lambda self, p, m: True
    try:
        return load_model(model_path)
    finally:
        Model.model_quant_predicate = orig


def make_transcribe_fn(model_path=MODEL, smoke_print=0, full_quant=False):
    print(f"Loading Granite Speech AR from {model_path}...")
    model = _load_model(model_path, full_quant=full_quant)

    import mlx.utils
    from collections import Counter
    params = mlx.utils.tree_flatten(model.parameters())
    total = sum(v.size for _, v in params)
    print(f"Loaded: {total/1e6:.1f}M params ({', '.join(Counter(str(v.dtype) for _, v in params))}, MLX)")

    dummy = np.zeros(16000, dtype=np.float32)
    _ = model.generate(dummy, prompt=TRANSCRIBE_PROMPT, max_tokens=1)
    print("Warm-up complete.")

    state = {"printed": 0}

    def transcribe(audio_array, sample_rate):
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)
        else:
            audio_array = audio_array.astype(np.float32)
        result = model.generate(audio_array, prompt=TRANSCRIBE_PROMPT)
        text = result.text.strip()
        if state["printed"] < smoke_print:
            print(f"  [HYP {state['printed']}] {text!r}")
            state["printed"] += 1
        return text

    return transcribe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["librispeech-clean", "librispeech-other", "all"], default="librispeech-clean")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--smoke-print", type=int, default=0)
    parser.add_argument("--full-quant", action="store_true")
    args = parser.parse_args()

    shuffle = (args.max_samples is not None) and (not args.no_shuffle)
    transcribe_fn = make_transcribe_fn(args.model, smoke_print=args.smoke_print, full_quant=args.full_quant)

    datasets = [LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        evaluate(transcribe_fn=transcribe_fn, dataset_name=ds, model_name=args.model_name,
                 max_samples=args.max_samples, shuffle=shuffle)


if __name__ == "__main__":
    main()
