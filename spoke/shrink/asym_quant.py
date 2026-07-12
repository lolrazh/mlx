#!/usr/bin/env python3
"""Asymmetric quantization: embeddings at low bits, body at 4-bit g64.

Usage: python asym_quant.py <teacher_dir> <out_dir> <embed_bits> <embed_group_size>
"""
import sys
from mlx_lm.utils import load, save, quantize_model

teacher = sys.argv[1]
out = sys.argv[2]
embed_bits = int(sys.argv[3])
embed_gs = int(sys.argv[4])

print(f"[asym] teacher={teacher} out={out} embed_bits={embed_bits} embed_gs={embed_gs}")

model, tokenizer, config = load(teacher, lazy=True, return_config=True)

def pred(path, module):
    # embed_tokens and embed_tokens_per_layer both matched by substring
    if "embed_tokens" in path:
        return {"bits": embed_bits, "group_size": embed_gs, "mode": "affine"}
    return True  # body -> global 4-bit g64

model, config = quantize_model(
    model, config, group_size=64, bits=4, mode="affine", quant_predicate=pred
)

# Report what the predicate captured
q = config["quantization"]
print("[asym] global quant:", {k: q[k] for k in ("bits", "group_size", "mode")})
overrides = {k: v for k, v in q.items() if isinstance(v, dict)}
print("[asym] per-path overrides:")
for k, v in overrides.items():
    print("   ", k, "->", v)

save(out, teacher, model, tokenizer, config)
print("[asym] saved ->", out)
