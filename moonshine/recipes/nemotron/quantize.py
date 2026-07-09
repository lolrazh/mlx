"""Quantize Nemotron 3.5 ASR streaming 0.6B to N-bit (group_size=64) MLX.

Unlike parakeet-mlx, mlx-audio's nemotron_asr loader natively reads quantized
checkpoints (a "quantization" block in config.json), so the output loads with the
stock ``mlx_audio.stt.utils.load_model``. Mirrors the predicate of
``mlx_audio.stt.models.nemotron_asr.convert._quantize``: only modules with
``to_quantized`` (Linear/Embedding) and group-size-divisible last dims are
quantized; conv/norm/LSTM stay bf16.

Usage:
    python -m moonshine.recipes.nemotron.quantize             # 6-bit (default)
    python -m moonshine.recipes.nemotron.quantize --bits 4
"""

import argparse
import json
import os
import shutil

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import quantize_model

from mlx_audio.stt.models.nemotron_asr.nemotron_asr import Model, ModelConfig

SRC_DIR = "moonshine/models/nemotron-3.5-asr-streaming-0.6b"
GROUP_SIZE = 64
BITS = 6


def main(bits=BITS, group_size=GROUP_SIZE, src_dir=SRC_DIR):
    out_dir = f"{src_dir}-{bits}bit"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {src_dir} (bf16)...")
    config = json.load(open(os.path.join(src_dir, "config.json")))
    model = Model(ModelConfig.from_dict(config))
    model.load_weights(os.path.join(src_dir, "model.safetensors"))

    def predicate(_path, module) -> bool:
        return (
            hasattr(module, "to_quantized")
            and hasattr(module, "weight")
            and module.weight.shape[-1] % group_size == 0
        )

    print(f"Quantizing: group_size={group_size}, bits={bits} ...")
    model, q_config = quantize_model(
        model, config, group_size, bits, quant_predicate=predicate
    )
    q_weights = dict(tree_flatten(model.parameters()))

    mx.save_safetensors(os.path.join(out_dir, "model.safetensors"), q_weights)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(q_config, f, ensure_ascii=False, indent=2)
    for extra in ("tokenizer.model", "vocab.txt"):
        src = os.path.join(src_dir, extra)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out_dir, extra))

    size = os.path.getsize(os.path.join(out_dir, "model.safetensors"))
    print(f"Done. {out_dir}: {size / 1e9:.3f} GB weights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Nemotron 3.5 ASR")
    parser.add_argument("--bits", type=int, default=BITS, choices=[2, 3, 4, 5, 6, 8])
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--src-dir", default=SRC_DIR)
    args = parser.parse_args()
    main(bits=args.bits, group_size=args.group_size, src_dir=args.src_dir)
