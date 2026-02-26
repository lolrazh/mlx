"""Weight loading and sanitization for Moonshine.

Handles the mapping from HuggingFace safetensors format to MLX model parameters:
  1. Strip `model.` prefix (present on encoder/decoder weights but not proj_out)
  2. Remap conv paths: embedder.conv{1,2}.weight → embedder.conv{1,2}.conv.weight
  3. Swap Conv1d axes from PyTorch (out, in, kernel) to MLX (out, kernel, in)

Note: MLX and PyTorch both store Linear weights as (out_features, in_features),
so NO transpose is needed for 2D weight matrices.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import ModelArgs
from .moonshine import Moonshine


def sanitize(weights: dict) -> dict:
    """Transform HuggingFace weights to match MLX model parameter names and shapes."""
    sanitized = {}

    for key, value in weights.items():
        # 1. Strip `model.` prefix (most keys have it, proj_out doesn't)
        new_key = key.removeprefix("model.")

        # 2. Remap conv paths: conv1.weight → conv1.conv.weight
        for conv_name in ("conv1", "conv2"):
            if f"embedder.{conv_name}.weight" in new_key:
                new_key = new_key.replace(
                    f"embedder.{conv_name}.weight",
                    f"embedder.{conv_name}.conv.weight",
                )
            elif f"embedder.{conv_name}.bias" in new_key:
                new_key = new_key.replace(
                    f"embedder.{conv_name}.bias",
                    f"embedder.{conv_name}.conv.bias",
                )

        # 3. Only Conv1d needs axis swapping
        if len(value.shape) == 3:
            # Conv1d weight: PyTorch (out, in, kernel) → MLX (out, kernel, in)
            value = mx.swapaxes(value, 1, 2)

        # All other weights (Linear, Embedding, LayerNorm, biases, scalars)
        # have the same layout between PyTorch and MLX — no transform needed.

        sanitized[new_key] = value

    return sanitized


def load_model(model_path: str, dtype: mx.Dtype = mx.float32) -> tuple[Moonshine, ModelArgs]:
    """Load a Moonshine model from a HuggingFace model directory.

    Args:
        model_path: Path to the HF model directory (containing config.json
                     and model.safetensors)
        dtype: Weight dtype (mx.float32 or mx.float16)

    Returns:
        (model, args) tuple
    """
    model_path = Path(model_path)

    # Load config
    with open(model_path / "config.json") as f:
        raw_config = json.load(f)
    args = ModelArgs.from_dict(raw_config)

    # Build model
    model = Moonshine(args)

    # Load and sanitize weights
    weights = mx.load(str(model_path / "model.safetensors"))
    weights = sanitize(weights)

    # Cast weights to target dtype
    if dtype != mx.float32:
        weights = {k: v.astype(dtype) if v.dtype == mx.float32 else v for k, v in weights.items()}

    # Load into model
    model.load_weights(list(weights.items()))

    return model, args
