"""Moonshine Medium baseline evaluation (PyTorch / transformers).

Runs the original Moonshine model through our eval pipeline to establish
WER numbers we must match with our MLX port.

Usage:
    python -m moonshine.baselines.moonshine_baseline                    # quick test (10 samples)
    python -m moonshine.baselines.moonshine_baseline --full             # full LibriSpeech test-clean
    python -m moonshine.baselines.moonshine_baseline --max-samples 250  # custom sample count
"""

import argparse
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from moonshine.eval.run_eval import evaluate
from moonshine.eval.datasets import LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER

MODEL = "UsefulSensors/moonshine-streaming-medium"


def make_transcribe_fn(model_path=MODEL, dtype=torch.float32, device=None):
    """Create a transcribe function compatible with the eval runner.

    Uses MPS (Metal GPU) by default on Apple Silicon for ~10x speedup over CPU.
    Defaults to fp32 because MPS has precision issues with fp16 on some inputs.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Loading Moonshine from {model_path} ({dtype}, {device})...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, dtype=dtype).to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Loaded: {param_count:.1f}M params, {model_bytes / 1e9:.2f} GB on {device}")

    def transcribe(audio_array, sample_rate):
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)

        inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        text = processor.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0]
        return text.strip()

    return transcribe


def main():
    parser = argparse.ArgumentParser(description="Moonshine Medium baseline evaluation")
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
        "--cpu",
        action="store_true",
        help="Force CPU instead of MPS GPU",
    )
    args = parser.parse_args()

    # Quick test mode: default to 10 samples if nothing specified
    if args.max_samples is None and not args.full:
        print("Tip: Running with --max-samples 10 for a quick test.")
        print("     Use --full for complete evaluation, or --max-samples N for custom.\n")
        args.max_samples = 10

    shuffle = (args.max_samples is not None) and (not args.no_shuffle)
    device = torch.device("cpu") if args.cpu else None

    transcribe_fn = make_transcribe_fn(device=device)

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
            model_name="moonshine-medium-pt",
            max_samples=args.max_samples,
            shuffle=shuffle,
        )
        results.append(result)

    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  Summary: Moonshine Medium Baseline (PyTorch)")
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
