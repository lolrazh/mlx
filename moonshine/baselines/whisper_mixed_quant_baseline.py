"""Whisper Large v3 mixed-quantization experiments.

Builds an in-memory mixed-quant Whisper model with MLX and runs it through the
existing ASR eval pipeline. This avoids patching `mlx_whisper` itself while we
figure out which quantization layout is worth keeping.

Usage:
    python -m moonshine.baselines.whisper_mixed_quant_baseline --recipe decoder-mixed-v1
    python -m moonshine.baselines.whisper_mixed_quant_baseline --recipe decoder-uniform-4 --max-samples 50
    python -m moonshine.baselines.whisper_mixed_quant_baseline --recipe all-linear-mixed-v1 --dataset all
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx_whisper
import numpy as np
from mlx.utils import tree_flatten
from mlx_whisper.load_models import load_model
from mlx_whisper.transcribe import ModelHolder

from moonshine.eval.datasets import LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER
from moonshine.eval.run_eval import evaluate

MODEL = "mlx-community/whisper-large-v3-mlx"
IN_MEMORY_MODEL_PATH = "__in_memory_mixed_quant_whisper__"


@dataclass(frozen=True)
class QuantSpec:
    bits: int
    group_size: int = 64
    mode: str = "affine"

    def as_dict(self) -> dict:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
        }


BUILTIN_RECIPE_CONFIGS = {
    "decoder-uniform-4": {
        "rules": [
            {
                "match": {"prefix": "decoder.", "module_types": ["Linear"]},
                "quant": {"bits": 4, "group_size": 64, "mode": "affine"},
            }
        ]
    },
    "decoder-uniform-6": {
        "rules": [
            {
                "match": {"prefix": "decoder.", "module_types": ["Linear"]},
                "quant": {"bits": 6, "group_size": 64, "mode": "affine"},
            }
        ]
    },
    "decoder-mixed-v1": {
        "rules": [
            {
                "match": {
                    "prefix": "decoder.blocks.",
                    "contains": ".cross_attn.",
                    "module_types": ["Linear"],
                },
                "quant": {"bits": 6, "group_size": 64, "mode": "affine"},
            },
            {
                "match": {
                    "prefix": "decoder.blocks.",
                    "contains_any": [".attn.", ".mlp"],
                    "block_lt": 24,
                    "module_types": ["Linear"],
                },
                "quant": {"bits": 4, "group_size": 64, "mode": "affine"},
            },
            {
                "match": {
                    "prefix": "decoder.blocks.",
                    "contains_any": [".attn.", ".mlp"],
                    "block_gte": 24,
                    "module_types": ["Linear"],
                },
                "quant": {"bits": 6, "group_size": 64, "mode": "affine"},
            },
        ]
    },
    "all-linear-mixed-v1": {
        "rules": [
            {
                "match": {"prefix": "encoder.", "module_types": ["Linear"]},
                "quant": {"bits": 6, "group_size": 64, "mode": "affine"},
            },
            {
                "match": {
                    "prefix": "decoder.blocks.",
                    "contains": ".cross_attn.",
                    "module_types": ["Linear"],
                },
                "quant": {"bits": 6, "group_size": 64, "mode": "affine"},
            },
            {
                "match": {
                    "prefix": "decoder.blocks.",
                    "contains_any": [".attn.", ".mlp"],
                    "block_lt": 24,
                    "module_types": ["Linear"],
                },
                "quant": {"bits": 4, "group_size": 64, "mode": "affine"},
            },
            {
                "match": {
                    "prefix": "decoder.blocks.",
                    "contains_any": [".attn.", ".mlp"],
                    "block_gte": 24,
                    "module_types": ["Linear"],
                },
                "quant": {"bits": 6, "group_size": 64, "mode": "affine"},
            },
        ]
    },
}

DEFAULT_SWEEP_RECIPES = [
    "decoder-uniform-4",
    "decoder-uniform-6",
    "decoder-mixed-v1",
    "all-linear-mixed-v1",
]


def _parse_block_index(path: str) -> int | None:
    parts = path.split(".")
    if len(parts) < 3 or parts[1] != "blocks":
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def _is_decoder_linear(path: str, module: nn.Module) -> bool:
    return path.startswith("decoder.") and isinstance(module, nn.Linear)


def _is_encoder_linear(path: str, module: nn.Module) -> bool:
    return path.startswith("encoder.") and isinstance(module, nn.Linear)


def _is_decoder_cross_attn(path: str) -> bool:
    return ".cross_attn." in path


def _is_decoder_self_attn(path: str) -> bool:
    return ".attn." in path and ".cross_attn." not in path


def _is_decoder_mlp(path: str) -> bool:
    return path.endswith(".mlp1") or path.endswith(".mlp2")


def _active_memory_gb() -> float | None:
    get_active = getattr(mx, "get_active_memory", None)
    if get_active is None:
        metal = getattr(mx, "metal", None)
        get_active = getattr(metal, "get_active_memory", None) if metal else None
    if get_active is None:
        return None
    return get_active() / (1024**3)


def _peak_memory_gb() -> float | None:
    get_peak = getattr(mx, "get_peak_memory", None)
    if get_peak is None:
        metal = getattr(mx, "metal", None)
        get_peak = getattr(metal, "get_peak_memory", None) if metal else None
    if get_peak is None:
        return None
    return get_peak() / (1024**3)


def _normalize_quant_spec(spec: dict) -> dict:
    quant = QuantSpec(
        bits=int(spec["bits"]),
        group_size=int(spec.get("group_size", 64)),
        mode=spec.get("mode", "affine"),
    )
    return quant.as_dict()


def _module_type_name(module: nn.Module) -> str:
    return type(module).__name__


def _rule_matches(path: str, module: nn.Module, match: dict) -> bool:
    prefix = match.get("prefix")
    if prefix and not path.startswith(prefix):
        return False

    suffix = match.get("suffix")
    if suffix and not path.endswith(suffix):
        return False

    contains = match.get("contains")
    if contains and contains not in path:
        return False

    contains_any = match.get("contains_any")
    if contains_any and not any(part in path for part in contains_any):
        return False

    excludes = match.get("excludes")
    if excludes and any(part in path for part in excludes):
        return False

    module_types = match.get("module_types")
    if module_types and _module_type_name(module) not in module_types:
        return False

    block_idx = _parse_block_index(path)
    if "block_lt" in match and (block_idx is None or block_idx >= int(match["block_lt"])):
        return False
    if "block_lte" in match and (block_idx is None or block_idx > int(match["block_lte"])):
        return False
    if "block_gt" in match and (block_idx is None or block_idx <= int(match["block_gt"])):
        return False
    if "block_gte" in match and (block_idx is None or block_idx < int(match["block_gte"])):
        return False

    return True


def load_recipe_config(recipe: str | None = None, recipe_file: str | None = None) -> tuple[str, dict]:
    if recipe_file:
        recipe_path = Path(recipe_file)
        with recipe_path.open("r") as f:
            config = json.load(f)
        recipe_name = config.get("name", recipe_path.stem)
        return recipe_name, config

    if recipe is None:
        raise ValueError("Either recipe or recipe_file must be provided")
    if recipe not in BUILTIN_RECIPE_CONFIGS:
        raise ValueError(f"Unknown recipe: {recipe}")
    return recipe, BUILTIN_RECIPE_CONFIGS[recipe]


def recipe_predicate(recipe_config: dict) -> Callable[[str, nn.Module], bool | dict]:
    rules = recipe_config.get("rules", [])

    def predicate(path: str, module: nn.Module):
        for rule in rules:
            if _rule_matches(path, module, rule.get("match", {})):
                return _normalize_quant_spec(rule["quant"])
        return False

    return predicate


def _summarize_model_size(model: nn.Module) -> float:
    total_bytes = sum(arr.nbytes for _, arr in tree_flatten(model.parameters()))
    return total_bytes / (1024**3)


def _summarize_recipe_targets(model: nn.Module, predicate) -> tuple[Counter, Counter]:
    module_counts = Counter()
    parameter_counts = Counter()

    for path, module in model.named_modules():
        spec = predicate(path, module)
        if not spec:
            continue

        if spec is True:
            label = "default"
        else:
            label = f"{spec['bits']}bit-g{spec['group_size']}"

        module_counts[label] += 1

        weight = getattr(module, "weight", None)
        if weight is not None:
            parameter_counts[label] += int(np.prod(weight.shape))

    return module_counts, parameter_counts


def build_quantized_model(model_path: str, recipe_name: str, recipe_config: dict) -> nn.Module:
    model = load_model(model_path, dtype=mx.float16)
    base_size_gb = _summarize_model_size(model)

    predicate = recipe_predicate(recipe_config)
    module_counts, parameter_counts = _summarize_recipe_targets(model, predicate)

    print(f"Loaded base model: {model_path}")
    print(f"Base parameter storage: {base_size_gb:.2f} GB")
    print(f"Recipe: {recipe_name}")
    for label in sorted(module_counts):
        params_m = parameter_counts[label] / 1_000_000
        print(f"  {label:<10} {module_counts[label]:>3} modules | {params_m:>7.1f}M weight params")

    nn.quantize(model, class_predicate=predicate)
    mx.eval(model.parameters())

    quantized_size_gb = _summarize_model_size(model)
    print(f"Quantized parameter storage: {quantized_size_gb:.2f} GB")
    print(f"Storage reduction: {base_size_gb / quantized_size_gb:.2f}x\n")
    active_gb = _active_memory_gb()
    peak_gb = _peak_memory_gb()
    if active_gb is not None:
        print(f"Active MLX memory after load: {active_gb:.2f} GB")
    if peak_gb is not None:
        print(f"Peak MLX memory after load:   {peak_gb:.2f} GB")
    if active_gb is not None or peak_gb is not None:
        print()
    return model


def make_transcribe_fn(model: nn.Module):
    ModelHolder.model = model
    ModelHolder.model_path = IN_MEMORY_MODEL_PATH

    def transcribe(audio_array, sample_rate):
        del sample_rate

        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)

        # Reassert the cached model each call so mlx_whisper never tries to
        # resolve the sentinel path through the hub.
        ModelHolder.model = model
        ModelHolder.model_path = IN_MEMORY_MODEL_PATH

        result = mlx_whisper.transcribe(
            audio_array,
            path_or_hf_repo=IN_MEMORY_MODEL_PATH,
            language="en",
        )
        return result["text"].strip()

    return transcribe


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper Large v3 with mixed quantization recipes",
    )
    parser.add_argument(
        "--recipe",
        default="decoder-mixed-v1",
        help="Built-in quantization layout to apply in memory",
    )
    parser.add_argument(
        "--recipe-file",
        help="Path to a JSON recipe file. Overrides --recipe when provided.",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL,
        help="HF repo or local converted Whisper model path",
    )
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
        help="Run full evaluation on both LibriSpeech splits",
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
    recipe_name, recipe_config = load_recipe_config(args.recipe, args.recipe_file)

    if args.full:
        datasets = [LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER]
        args.max_samples = None
    elif args.dataset == "all":
        datasets = [LIBRISPEECH_CLEAN, LIBRISPEECH_OTHER]
    else:
        datasets = [args.dataset]

    model = build_quantized_model(args.model_path, recipe_name, recipe_config)
    transcribe_fn = make_transcribe_fn(model)

    results = []
    for dataset_name in datasets:
        result = evaluate(
            transcribe_fn=transcribe_fn,
            dataset_name=dataset_name,
            model_name=f"whisper-large-v3-{recipe_name}",
            max_samples=args.max_samples,
            shuffle=shuffle,
        )
        results.append(result)

    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  Summary: Whisper Large v3 Mixed Quant ({recipe_name})")
        print(f"{'=' * 70}")
        print(f"  {'Dataset':<25} {'WER':>8} {'RTF':>8} {'MLX Peak':>10}")
        print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 10}")
        for result in results:
            print(
                f"  {result['dataset']:<25} "
                f"{result['wer']:>7.2%} "
                f"{result['real_time_factor']:>7.2f}x "
                f"{result.get('mlx_peak_memory_gb', 0.0):>8.2f} GB"
            )
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
