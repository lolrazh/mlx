"""Convert Granite Speech 4.1 NAR to a TRUE full-quantization MLX model.

The stock mlx-community 5-bit NAR only quantizes the EDITOR; the conformer
encoder and projector stay bf16, so peak RAM is ~3.3 GB. To get under 2 GB we
must quantize the encoder + projector + editor Linears uniformly.

Strategy:
  1. Load the existing 5-bit MLX NAR (editor=5bit quantized, encoder/projector=bf16).
  2. Dequantize every QuantizedLinear back to a plain bf16 nn.Linear.
  3. Re-quantize ALL Linears (encoder + projector + editor) uniformly to the
     target bits/group_size. Conv1d / RMSNorm / LayerNorm / BatchNorm have no
     `to_quantized` and are skipped automatically.
  4. Save weights (safetensors) + config (quantization.bits = target) + the
     tokenizer / processor files so mlx-audio can reload it.

Usage:
    python -m moonshine.baselines.granite_nar_convert --bits 4 --out <dir>
    python -m moonshine.baselines.granite_nar_convert --bits 6 --out <dir>
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

SRC = "mlx-community/granite-speech-4.1-2b-nar-mlx-5bit"


def _dequantize_module(model: nn.Module) -> None:
    """In-place: replace every QuantizedLinear with a plain bf16 nn.Linear."""

    def replace(module: nn.Module):
        for name, child in list(module.children().items()):
            if isinstance(child, nn.QuantizedLinear):
                w = mx.dequantize(
                    child.weight, child.scales, child.biases,
                    group_size=child.group_size, bits=child.bits, mode=child.mode,
                ).astype(mx.bfloat16)
                out_dims, in_dims = w.shape
                lin = nn.Linear(in_dims, out_dims, bias=hasattr(child, "bias"))
                lin.weight = w
                if hasattr(child, "bias"):
                    lin.bias = child.bias.astype(mx.bfloat16)
                module[name] = lin
            elif isinstance(child, nn.QuantizedEmbedding):
                w = mx.dequantize(
                    child.weight, child.scales, child.biases,
                    group_size=child.group_size, bits=child.bits, mode=child.mode,
                ).astype(mx.bfloat16)
                num_emb, dims = w.shape
                emb = nn.Embedding(num_emb, dims)
                emb.weight = w
                module[name] = emb
            elif isinstance(child, nn.Module):
                replace(child)
            elif isinstance(child, (list, tuple)):
                for sub in child:
                    if isinstance(sub, nn.Module):
                        replace(sub)

    replace(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mlx_audio.stt.utils import load_model
    from mlx_audio.utils import get_model_path

    print(f"Loading source model {args.src} ...")
    model = load_model(args.src)

    print("Dequantizing existing QuantizedLinear layers to bf16 ...")
    _dequantize_module(model)
    mx.eval(model.parameters())

    # Re-quantize ALL Linears uniformly (encoder + projector + editor).
    def class_predicate(p, m):
        if not hasattr(m, "to_quantized"):
            return False
        if hasattr(m, "weight") and m.weight.shape[-1] % args.group_size != 0:
            print(f"  skip (group_size) {p}: {m.weight.shape}")
            return False
        return True

    print(f"Quantizing ALL Linears to {args.bits}-bit (group_size {args.group_size}) ...")
    nn.quantize(model, group_size=args.group_size, bits=args.bits,
                class_predicate=class_predicate)
    mx.eval(model.parameters())

    # Count quantized vs not
    n_q = sum(1 for _, m in _named_modules(model) if isinstance(m, nn.QuantizedLinear))
    n_lin = sum(1 for _, m in _named_modules(model) if isinstance(m, nn.Linear))
    print(f"After quantize: {n_q} QuantizedLinear, {n_lin} plain Linear remaining")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Save weights
    import mlx.utils
    flat = dict(mlx.utils.tree_flatten(model.parameters()))
    weights_path = out / "model.safetensors"
    mx.save_safetensors(str(weights_path), flat)
    print(f"Saved weights -> {weights_path} ({len(flat)} tensors)")

    # Copy + patch config from source snapshot
    src_dir = Path(get_model_path(args.src))
    cfg = json.load(open(src_dir / "config.json"))
    q = {"group_size": args.group_size, "bits": args.bits}
    cfg["quantization"] = q
    cfg["quantization_config"] = q
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print(f"Wrote config.json (quantization bits={args.bits})")

    # Copy tokenizer / processor / aux files (everything except weights+config)
    for f in os.listdir(src_dir):
        if f in ("config.json",) or f.endswith(".safetensors") or f.endswith(".npz"):
            continue
        s = src_dir / f
        if s.is_file():
            shutil.copy2(s, out / f)
    print(f"Copied tokenizer/aux files into {out}")
    print("DONE")


def _named_modules(model):
    return model.named_modules()


if __name__ == "__main__":
    main()
