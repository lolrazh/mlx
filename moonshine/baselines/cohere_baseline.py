"""Cohere Transcribe (03-2026) baseline — WORKING 4-bit MLX via mlx-speech.

============================================================================
IMPORTANT: This baseline requires the DEDICATED venv `.venv-cohere`, NOT the
main `.venv`. Cohere's MLX checkpoints are loaded with `mlx-speech`
(https://github.com/appautomaton/mlx-speech), which has its own CohereAsr
graph and requires Python >= 3.13 + mlx >= 0.31. The main `.venv` (mlx-audio)
CANNOT load these checkpoints correctly — it silently drops the decoder
weights and produces token-salad garbage.

Run with:
    .venv-cohere/bin/python -m moonshine.baselines.cohere_baseline               # 10-sample smoke test
    .venv-cohere/bin/python -m moonshine.baselines.cohere_baseline --max-samples 250
    .venv-cohere/bin/python moonshine/baselines/cohere_baseline.py --max-samples 250
============================================================================

THE WORKING LOAD RECIPE
-----------------------
    from mlx_speech.generation import CohereAsrModel
    model = CohereAsrModel.from_path(MODEL_DIR)          # dir w/ model.safetensors + config.json
    result = model.transcribe(audio16k, sample_rate=16000, language="en")
    text = result.text.strip()

`from_path` wants the directory that directly contains `model.safetensors` and
`config.json` (NOT a parent). The mlx-community 8-bit keeps those under an
`mlx-int8/` subfolder, so point at <snapshot>/mlx-int8.

WHY WE BUILD OUR OWN 4-bit
--------------------------
beshkenadze/cohere-transcribe-03-2026-mlx-4bit (and its -fp16) are NOT in
mlx-speech's layout: they use a fused-qkv / Cohere `sub_layer` decoder graph
and a different subsampling conv decomposition (key_map.json is from their own
converter). mlx-speech's CohereAsr graph uses split q/k/v projections,
`encoder_attn`, `relative_k_proj`, `pw_convs`, etc. Loading beshkenadze's 4-bit
fails strict alignment (795 checkpoint-only / 660 model-only / 145 shape
mismatches) and there is no native-layout 4-bit published on HF
(appautomaton/cohere-asr-mlx and mlx-community both only publish 8-bit).

So we produce a NATIVE-layout 4-bit (group_size 64, affine) by loading the
native 8-bit, dequantizing it, and re-quantizing to 4-bit with mlx-speech's own
quantizer. The result loads cleanly via `from_path`, transcribes correct
English, and peaks well under 2 GB (~1.7 GB). It is built once and cached at
MODEL_4BIT_DIR.
"""

import argparse
import os
import shutil

import numpy as np

from moonshine.eval.run_eval import evaluate
from moonshine.eval.datasets import LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER

# Native 8-bit source (mlx-speech layout, `format: mlx`), used as the quant source.
MODEL_8BIT = "mlx-community/cohere-transcribe-03-2026-mlx-8bit"
# Where we cache the built native-layout 4-bit checkpoint.
MODEL_4BIT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "cohere-transcribe-mlx-4bit"
)
MODEL_NAME = "cohere-transcribe-4bit"


def _resolve_8bit_dir():
    """8-bit repo keeps its files under `mlx-int8/`; return that directory."""
    from huggingface_hub import snapshot_download

    return os.path.join(snapshot_download(MODEL_8BIT), "mlx-int8")


def build_4bit_checkpoint(out_dir=MODEL_4BIT_DIR, force=False):
    """Build a native-layout 4-bit checkpoint by re-quantizing the 8-bit.

    Dequantize the native 8-bit weights, then re-quantize to 4-bit
    (group_size 64, affine) using mlx-speech's own quantizer, and save in
    mlx-speech's native checkpoint format.
    """
    if os.path.exists(os.path.join(out_dir, "model.safetensors")) and not force:
        return out_dir

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from mlx_speech.models.cohere_asr import CohereAsrForConditionalGeneration
    from mlx_speech.models.cohere_asr.checkpoint import (
        load_cohere_asr_checkpoint,
        get_quantization_config,
        quantize_cohere_asr_model,
        load_checkpoint_into_model,
        save_cohere_asr_model,
        QuantizationConfig,
    )

    src8 = _resolve_8bit_dir()
    print(f"Building native 4-bit from 8-bit source: {src8}")

    ckpt = load_cohere_asr_checkpoint(src8)
    model = CohereAsrForConditionalGeneration(ckpt.config)
    q8 = get_quantization_config(ckpt.config)
    quantize_cohere_asr_model(model, q8, state_dict=ckpt.state_dict)
    load_checkpoint_into_model(model, ckpt, strict=True)
    model.eval()
    mx.eval(model.parameters())

    def dequantize_module(m):
        for name, child in list(m.children().items()):
            if isinstance(child, nn.QuantizedLinear):
                w = mx.dequantize(
                    child.weight, child.scales, child.biases,
                    child.group_size, child.bits, getattr(child, "mode", "affine"),
                )
                out_f, in_f = w.shape
                lin = nn.Linear(in_f, out_f, bias=("bias" in child))
                lin.weight = w
                if "bias" in child:
                    lin.bias = child.bias
                setattr(m, name, lin)
            elif isinstance(child, nn.QuantizedEmbedding):
                w = mx.dequantize(
                    child.weight, child.scales, child.biases,
                    child.group_size, child.bits, getattr(child, "mode", "affine"),
                )
                n, d = w.shape
                emb = nn.Embedding(n, d)
                emb.weight = w
                setattr(m, name, emb)
            elif isinstance(child, nn.Module):
                dequantize_module(child)
            elif isinstance(child, (list, tuple)):
                for c in child:
                    if isinstance(c, nn.Module):
                        dequantize_module(c)

    dequantize_module(model)
    mx.eval(model.parameters())

    q4 = QuantizationConfig(bits=4, group_size=64, mode="affine")
    quantize_cohere_asr_model(model, q4, state_dict=None)
    mx.eval(model.parameters())

    save_cohere_asr_model(model, out_dir, config=ckpt.config, quantization=q4)
    for fn in os.listdir(src8):
        if fn in ("model.safetensors", "config.json"):
            continue
        shutil.copy(os.path.join(src8, fn), os.path.join(out_dir, fn))

    size_gb = os.path.getsize(os.path.join(out_dir, "model.safetensors")) / 1e9
    print(f"Built 4-bit checkpoint at {out_dir} ({size_gb:.2f} GB on disk)")
    return out_dir


def make_transcribe_fn(model_dir=MODEL_4BIT_DIR):
    """Create a transcribe function compatible with the eval runner."""
    import mlx.core as mx
    from mlx_speech.generation import CohereAsrModel

    model_dir = build_4bit_checkpoint(model_dir)
    print(f"Loading Cohere Transcribe (4-bit) from {model_dir}...")
    mx.reset_peak_memory()
    model = CohereAsrModel.from_path(model_dir)
    mx.eval(model.model.parameters())
    print(f"Loaded; peak after load: {mx.get_peak_memory() / 1024**3:.2f} GB")

    # Warm-up: one dummy inference.
    dummy = np.zeros(16000, dtype=np.float32)
    _ = model.transcribe(dummy, sample_rate=16000, language="en", max_new_tokens=1)
    print("Warm-up complete.")

    def transcribe(audio_array, sample_rate):
        audio_array = np.asarray(audio_array, dtype=np.float32)
        # mlx-speech's transcribe() refuses non-16k audio, so resample if needed.
        if sample_rate != 16000:
            import librosa

            audio_array = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=16000
            ).astype(np.float32)
            sample_rate = 16000
        result = model.transcribe(audio_array, sample_rate=sample_rate, language="en")
        return result.text.strip()

    return transcribe


def main():
    parser = argparse.ArgumentParser(description="Cohere Transcribe 4-bit (MLX) evaluation")
    parser.add_argument(
        "--dataset",
        choices=["librispeech-clean", "librispeech-other", "all"],
        default="librispeech-clean",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build the 4-bit checkpoint and exit (no eval).",
    )
    args = parser.parse_args()

    if args.build_only:
        build_4bit_checkpoint()
        return

    if args.max_samples is None and not args.full:
        print("Tip: Running with --max-samples 10 for a quick test.")
        print("     Use --full or --max-samples N for a real benchmark.\n")
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
        print("  Summary: Cohere Transcribe (MLX, 4-bit)")
        print(f"{'=' * 70}")
        print(f"  {'Dataset':<25} {'WER':>8} {'RTF':>8} {'Peak RAM':>10}")
        print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 10}")
        for r in results:
            ram = r.get("mlx_peak_memory_gb", r["peak_memory_gb"])
            print(
                f"  {r['dataset']:<25} "
                f"{r['wer']:>7.2%} "
                f"{r['real_time_factor']:>7.2f}x "
                f"{ram:>8.2f} GB"
            )
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
