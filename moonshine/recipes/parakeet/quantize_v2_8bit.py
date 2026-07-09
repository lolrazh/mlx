"""Quantize Parakeet TDT 0.6B v2 to 8-bit (group_size=64) and save an MLX checkpoint.

parakeet-mlx's from_pretrained/from_config has no quantized-checkpoint loading, so we
build one ourselves. nn.quantize only swaps nn.Linear/nn.Embedding layers; Conv1d,
BatchNorm, and the hand-rolled LSTM in rnnt.py stay unquantized (expected; matches the
animaslabs/parakeet-tdt-0.6b-v3-mlx-8bit recipe: -q 8 -g 64). Result is a modest size
reduction, not 4x.

Usage:
    python -m moonshine.recipes.parakeet.quantize_v2_8bit
"""

import json
import os
import shutil

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from huggingface_hub import snapshot_download

from parakeet_mlx import from_pretrained

REPO = "mlx-community/parakeet-tdt-0.6b-v2"
GROUP_SIZE = 64
BITS = 8
OUT_DIR = "moonshine/models/parakeet-tdt-0.6b-v2-8bit"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {REPO} (bf16 after cast)...")
    model = from_pretrained(REPO)

    print(f"Quantizing in place: group_size={GROUP_SIZE}, bits={BITS} ...")
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)

    flat = dict(mlx.utils.tree_flatten(model.parameters()))
    n_params = sum(v.size for v in flat.values())
    print(f"Flattened {len(flat)} arrays, {n_params/1e6:.1f}M params")

    out_weights = os.path.join(OUT_DIR, "model.safetensors")
    mx.save_safetensors(out_weights, flat)
    print(f"Saved weights -> {out_weights}")

    # Copy config.json from the HF snapshot cache unchanged (keep the schema
    # parakeet-mlx's from_config expects intact).
    snap = snapshot_download(REPO, allow_patterns=["config.json"])
    src_config = os.path.join(snap, "config.json")
    dst_config = os.path.join(OUT_DIR, "config.json")
    shutil.copyfile(src_config, dst_config)
    print(f"Copied config -> {dst_config}")

    # Provenance note as a separate README so we don't touch config.json schema.
    readme = os.path.join(OUT_DIR, "README.md")
    with open(readme, "w") as f:
        f.write(
            f"# {REPO} quantized 8-bit\n\n"
            f"Built with parakeet_mlx.from_pretrained -> nn.quantize"
            f"(group_size={GROUP_SIZE}, bits={BITS}).\n"
            f"Only nn.Linear/nn.Embedding layers are quantized; Conv1d/BatchNorm/LSTM"
            f" remain bf16.\n\n"
            f"Load: from_config(json.load(config.json)) -> "
            f"nn.quantize(model, {GROUP_SIZE}, {BITS}) -> model.load_weights(...).\n"
        )
    print(f"Wrote provenance -> {readme}")

    size_bytes = os.path.getsize(out_weights)
    print(f"\nDone. Checkpoint size: {size_bytes/1e9:.3f} GB ({size_bytes} bytes)")


if __name__ == "__main__":
    main()
