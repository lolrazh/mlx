"""Convert IBM Granite Speech 4.1 2B (AR) to a TRUE full-quantization MLX model.

mlx-audio's `granite_speech` loader (the 4.0 AR port) also loads the 4.1-2b AR
checkpoint (same model_type=granite_speech). The stock conversions only
quantize the language model; here we quantize ALL Linears (encoder + projector
+ language_model) uniformly so the encoder no longer stays bf16.

Strategy: load the bf16 model, quantize every Linear/Embedding to target bits
(conv/norm auto-skipped), save weights + patched config (quantization.bits) +
tokenizer/aux files.

Usage:
    python -m moonshine.baselines.granite_ar_convert --bits 4 --out <dir>
    python -m moonshine.baselines.granite_ar_convert --bits 6 --out <dir>
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

SRC = "ibm-granite/granite-speech-4.1-2b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mlx_audio.stt.utils import load_model
    from mlx_audio.utils import get_model_path

    print(f"Loading source model {args.src} (bf16) ...")
    model = load_model(args.src)
    mx.eval(model.parameters())

    def class_predicate(p, m):
        if not hasattr(m, "to_quantized"):
            return False
        if hasattr(m, "weight") and m.weight.shape[-1] % args.group_size != 0:
            print(f"  skip (group_size) {p}: {m.weight.shape}")
            return False
        return True

    print(f"Quantizing ALL Linears/Embeddings to {args.bits}-bit (group_size {args.group_size}) ...")
    nn.quantize(model, group_size=args.group_size, bits=args.bits,
                class_predicate=class_predicate)
    mx.eval(model.parameters())

    n_q = sum(1 for _, m in model.named_modules() if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)))
    n_lin = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Linear))
    print(f"After quantize: {n_q} quantized modules, {n_lin} plain Linear remaining")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import mlx.utils
    flat = dict(mlx.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), flat)
    print(f"Saved weights -> {out/'model.safetensors'} ({len(flat)} tensors)")

    src_dir = Path(get_model_path(args.src))
    cfg = json.load(open(src_dir / "config.json"))
    q = {"group_size": args.group_size, "bits": args.bits}
    cfg["quantization"] = q
    cfg["quantization_config"] = q
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print(f"Wrote config.json (quantization bits={args.bits})")

    for f in os.listdir(src_dir):
        if f == "config.json" or f.endswith(".safetensors") or f.endswith(".npz") or f.endswith(".index.json"):
            continue
        s = src_dir / f
        if s.is_file():
            shutil.copy2(s, out / f)
    print(f"Copied tokenizer/aux files into {out}")
    print("DONE")


if __name__ == "__main__":
    main()
