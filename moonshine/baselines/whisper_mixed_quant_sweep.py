"""Run a small recipe sweep for Whisper Large v3 mixed quantization."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx

from moonshine.baselines.whisper_mixed_quant_baseline import (
    DEFAULT_SWEEP_RECIPES,
    build_quantized_model,
    load_recipe_config,
    make_transcribe_fn,
)
from moonshine.eval.datasets import LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER
from moonshine.eval.run_eval import evaluate
from mlx_whisper.transcribe import ModelHolder


def _clear_mlx_cache():
    clear_cache = getattr(mx, "clear_cache", None)
    if clear_cache is None:
        metal = getattr(mx, "metal", None)
        clear_cache = getattr(metal, "clear_cache", None) if metal else None
    if clear_cache is not None:
        clear_cache()


def _resolve_recipe_specs(args) -> list[tuple[str | None, str | None]]:
    specs: list[tuple[str | None, str | None]] = []
    for recipe in args.recipe:
        specs.append((recipe, None))
    for recipe_file in args.recipe_file:
        specs.append((None, recipe_file))
    if not specs:
        specs.extend((recipe, None) for recipe in DEFAULT_SWEEP_RECIPES)
    return specs


def main():
    parser = argparse.ArgumentParser(description="Sweep mixed-quant Whisper recipes")
    parser.add_argument("--model-path", default="mlx-community/whisper-large-v3-mlx")
    parser.add_argument(
        "--recipe",
        action="append",
        default=[],
        help="Built-in recipe name. Repeat to include multiple.",
    )
    parser.add_argument(
        "--recipe-file",
        action="append",
        default=[],
        help="JSON recipe path. Repeat to include multiple.",
    )
    parser.add_argument(
        "--dataset",
        choices=["librispeech-clean", "librispeech-other", "all"],
        default="librispeech-clean",
    )
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--output-json",
        default="moonshine/results/whisper-mixed-quant-sweep.json",
        help="Where to save the aggregated sweep summary",
    )
    args = parser.parse_args()

    datasets = (
        [LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER]
        if args.dataset == "all"
        else [args.dataset]
    )
    shuffle = not args.no_shuffle
    summary_rows = []

    for recipe, recipe_file in _resolve_recipe_specs(args):
        recipe_name, recipe_config = load_recipe_config(recipe, recipe_file)
        print(f"\n{'#' * 72}")
        print(f"Recipe sweep item: {recipe_name}")
        print(f"{'#' * 72}")

        model = build_quantized_model(args.model_path, recipe_name, recipe_config)
        transcribe_fn = make_transcribe_fn(model)

        for dataset_name in datasets:
            result = evaluate(
                transcribe_fn=transcribe_fn,
                dataset_name=dataset_name,
                model_name=f"whisper-large-v3-{recipe_name}",
                max_samples=args.max_samples,
                shuffle=shuffle,
            )
            summary_rows.append(
                {
                    "recipe": recipe_name,
                    "dataset": dataset_name,
                    "wer": result["wer"],
                    "rtf": result["real_time_factor"],
                    "mlx_peak_memory_gb": result.get("mlx_peak_memory_gb"),
                    "mlx_active_memory_gb_end": result.get("mlx_active_memory_gb_end"),
                }
            )

        ModelHolder.model = None
        ModelHolder.model_path = None
        del transcribe_fn
        del model
        gc.collect()
        _clear_mlx_cache()

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\n{'=' * 72}")
    print("Sweep Summary")
    print(f"{'=' * 72}")
    print(f"  {'Recipe':<28} {'Dataset':<20} {'WER':>8} {'RTF':>8} {'MLX Peak':>10}")
    print(f"  {'-' * 28} {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 10}")
    for row in summary_rows:
        peak = row['mlx_peak_memory_gb']
        peak_text = f"{peak:.2f} GB" if peak is not None else "n/a"
        print(
            f"  {row['recipe']:<28} "
            f"{row['dataset']:<20} "
            f"{row['wer']:>7.2%} "
            f"{row['rtf']:>7.2f}x "
            f"{peak_text:>10}"
        )
    print(f"{'=' * 72}")
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
