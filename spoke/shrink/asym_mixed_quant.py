#!/usr/bin/env python3
"""Mixed_3_6 body + 2-bit g32 embeddings quantization.

Reuses mlx-lm's own `mixed_3_6` sensitive-layer selection for the transformer
body, then overrides the two embedding tensors (`embed_tokens`,
`embed_tokens_per_layer`) to 2-bit group-32.

Usage: python asym_mixed_quant.py <teacher_dir> <out_dir> [recipe] [body_group_size] [emb_bits] [emb_group_size]
"""
import sys
from mlx_lm.utils import load, save, quantize_model
from mlx_lm.convert import mixed_quant_predicate_builder

teacher = sys.argv[1]
out = sys.argv[2]
recipe = sys.argv[3] if len(sys.argv) > 3 else "mixed_3_6"
body_gs = int(sys.argv[4]) if len(sys.argv) > 4 else 64
emb_bits = int(sys.argv[5]) if len(sys.argv) > 5 else 2
emb_gs = int(sys.argv[6]) if len(sys.argv) > 6 else 32

print(f"[mixed] teacher={teacher} out={out} recipe={recipe} body_gs={body_gs} "
      f"emb_bits={emb_bits} emb_gs={emb_gs}")

model, tokenizer, config = load(teacher, lazy=True, return_config=True)

# mlx-lm's exact mixed_3_6 body predicate (first/last eighth + every-third
# middle layer -> v_proj/down_proj @ high bits; lm_head @ high bits; else low).
base_pred = mixed_quant_predicate_builder(recipe, model, group_size=body_gs)


def pred(path, module):
    # Embedding tables -> 2-bit g32 (overrides the body recipe for these).
    if "embed_tokens" in path:
        return {"bits": emb_bits, "group_size": emb_gs, "mode": "affine"}
    # Everything else: exact mlx-lm mixed recipe decision.
    return base_pred(path, module)


# group_size/bits here are only defaults; the predicate returns a dict for
# every quantizable module so these are effectively overridden per-path.
model, config = quantize_model(
    model, config, group_size=body_gs, bits=3, mode="affine", quant_predicate=pred
)

# Report what the predicate captured.
q = config["quantization"]
print("[mixed] global quant:", {k: q[k] for k in ("bits", "group_size", "mode")})
overrides = {k: v for k, v in q.items() if isinstance(v, dict)}
from collections import Counter
bit_hist = Counter((v["bits"], v["group_size"]) for v in overrides.values())
print(f"[mixed] {len(overrides)} per-path overrides; (bits,gs) histogram:")
for k, n in sorted(bit_hist.items()):
    print(f"    bits={k[0]} gs={k[1]}: {n} tensors")
for k, v in overrides.items():
    if "embed" in k:
        print("    EMB", k, "->", v)

save(out, teacher, model, tokenizer, config)
print("[mixed] saved ->", out)
