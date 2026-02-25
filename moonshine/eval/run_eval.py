"""Model-agnostic ASR evaluation runner.

Runs any transcribe_fn through a dataset and computes WER + timing metrics.
Designed to be reused across Whisper baseline, Moonshine, and any future models.
"""

import json
import os
import time
import tracemalloc
import numpy as np

from moonshine.eval.datasets import load_samples, dataset_info
from moonshine.eval.wer import compute_wer, compute_wer_per_sample, print_summary


def evaluate(
    transcribe_fn,
    dataset_name,
    model_name="unknown",
    max_samples=None,
    output_dir="moonshine/results",
):
    """Run evaluation of a transcription function on a dataset.

    Args:
        transcribe_fn: Callable(audio_array, sample_rate) -> str
        dataset_name: Dataset key from the registry.
        model_name: Name for logging and result files.
        max_samples: Limit number of samples (None = all).
        output_dir: Where to save JSON results.

    Returns:
        dict with WER metrics, timing, and metadata.
    """
    info = dataset_info(dataset_name)
    print(f"\nEvaluating '{model_name}' on {info['name']} ({info['hf_path']}, {info['split']})")
    print(f"Max samples: {max_samples or 'all'}\n")

    references = []
    hypotheses = []
    total_audio_seconds = 0.0
    total_inference_seconds = 0.0

    tracemalloc.start()

    for i, (audio, sr, ref_text) in enumerate(load_samples(dataset_name, max_samples)):
        audio_duration = len(audio) / sr
        total_audio_seconds += audio_duration

        start = time.perf_counter()
        hypothesis = transcribe_fn(audio, sr)
        elapsed = time.perf_counter() - start
        total_inference_seconds += elapsed

        references.append(ref_text)
        hypotheses.append(hypothesis)

        if (i + 1) % 100 == 0:
            current_wer = compute_wer(references, hypotheses)["wer"]
            rtf = total_inference_seconds / total_audio_seconds if total_audio_seconds > 0 else 0
            print(
                f"  [{i+1}] running WER: {current_wer:.2%} | "
                f"RTF: {rtf:.2f}x | "
                f"audio: {total_audio_seconds:.0f}s processed"
            )

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_gb = peak_memory / (1024**3)

    # Compute final metrics
    aggregate = compute_wer(references, hypotheses)
    per_sample = compute_wer_per_sample(references, hypotheses)
    sample_wers = [s["wer"] for s in per_sample]
    median_wer = float(np.median(sample_wers)) if sample_wers else 0.0

    rtf = total_inference_seconds / total_audio_seconds if total_audio_seconds > 0 else 0

    results = {
        "model": model_name,
        "dataset": dataset_name,
        "dataset_info": info,
        "num_samples": aggregate["num_samples"],
        "wer": aggregate["wer"],
        "median_wer": median_wer,
        "substitutions": aggregate["substitutions"],
        "insertions": aggregate["insertions"],
        "deletions": aggregate["deletions"],
        "total_words": aggregate["total_words"],
        "total_audio_seconds": round(total_audio_seconds, 1),
        "total_inference_seconds": round(total_inference_seconds, 1),
        "real_time_factor": round(rtf, 3),
        "peak_memory_gb": round(peak_memory_gb, 2),
    }

    # Print summary
    print_summary(aggregate, f"{model_name} / {dataset_name}")
    print(f"  Median sample WER:  {median_wer:.2%}")
    print(f"  Audio processed:    {total_audio_seconds:.1f}s")
    print(f"  Inference time:     {total_inference_seconds:.1f}s")
    print(f"  Real-time factor:   {rtf:.2f}x")
    print(f"  Peak memory:        {peak_memory_gb:.2f} GB")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name}_{dataset_name}.json".replace(" ", "_").replace("/", "_")
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {filepath}")

    return results
