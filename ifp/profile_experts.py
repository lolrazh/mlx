#!/usr/bin/env python3
"""Profile expert routing of Gemma 4 26B A4B on Spoke inputs (ifp-lite step 1).

Taps every MoE router in the model during generation and records which
experts fire, per layer, separately for prefill (prompt) and decode
(generation). The output histogram answers: how concentrated is Spoke
traffic across the 30x128 expert grid, i.e. how much of the model could
a domain-pruned variant drop?

Usage:
    python ifp/profile_experts.py                          # broad58, v2 prompt
    python ifp/profile_experts.py --test-set spoke/bench/test_set_v3.json
    python ifp/profile_experts.py --test-set spoke/data/v5/train.jsonl --limit 200
    python ifp/profile_experts.py --prompt-mode spoke-full --max-tokens 128
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate
from mlx_lm.models import gemma4_text

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "spoke" / "bench"))
from run_benchmark import (  # noqa: E402
    GENERIC_PROMPT,
    MINI_PROMPT,
    SPOKE_FULL_PROMPT,
    V2_PROMPT,
    V3_PROMPT,
)

PROMPTS = {
    "generic": GENERIC_PROMPT,
    "mini": MINI_PROMPT,
    "v2": V2_PROMPT,
    "v3": V3_PROMPT,
    "spoke-full": SPOKE_FULL_PROMPT,
}

DEFAULT_MODEL = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"


class RouterTap:
    """Class-level patch on the model's router that records routing decisions.

    Supports two MoE families:
    - gemma4:     dedicated Router module returning (indices, weights)
    - qwen3.5/3.6: routing inline in Qwen3NextSparseMoeBlock.__call__ — we let
      the block run, then recompute the gate top-k (one tiny extra matmul,
      deterministic so bit-identical to what the block used)
    """

    def __init__(self):
        self.layer_of = {}      # id(router or moe block) -> layer index
        self.records = []       # (layer_idx, top_k_indices, top_k_weights) as lazy mx arrays
        self.enabled = False

    def install(self, model):
        layers = getattr(model, "layers", None)
        if layers is None:
            layers = model.language_model.layers
        tap = self

        if getattr(layers[0], "router", None) is not None or any(
            getattr(l, "router", None) is not None for l in layers
        ):
            for i, layer in enumerate(layers):
                router = getattr(layer, "router", None)
                if router is not None:
                    self.layer_of[id(router)] = i

            orig_call = gemma4_text.Router.__call__

            def tapped_router(router_self, x):
                idx, w = orig_call(router_self, x)
                if tap.enabled:
                    layer = tap.layer_of.get(id(router_self))
                    if layer is not None:
                        tap.records.append((layer, idx, w))
                return idx, w

            gemma4_text.Router.__call__ = tapped_router
            return len(self.layer_of)

        # Qwen MoE blocks (qwen3_moe and qwen3_next/3.5/3.6 share the same
        # gate/top_k/norm_topk_prob interface) — patch whichever class the
        # model's MoE layers actually use.
        moe_classes = []
        try:
            from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock
            moe_classes.append(Qwen3NextSparseMoeBlock)
        except ImportError:
            pass
        try:
            from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock
            moe_classes.append(Qwen3MoeSparseMoeBlock)
        except ImportError:
            pass

        block_cls = None
        for i, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if any(isinstance(mlp, c) for c in moe_classes):
                self.layer_of[id(mlp)] = i
                block_cls = type(mlp)
        if not self.layer_of:
            raise RuntimeError("No routers found — is this a MoE checkpoint?")

        orig_moe_call = block_cls.__call__

        def tapped_moe(blk, x):
            y = orig_moe_call(blk, x)
            if tap.enabled:
                layer = tap.layer_of.get(id(blk))
                if layer is not None:
                    gates = mx.softmax(blk.gate(x), axis=-1, precise=True)
                    k = blk.top_k
                    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                    scores = mx.take_along_axis(gates, inds, axis=-1)
                    if blk.norm_topk_prob:
                        scores = scores / scores.sum(axis=-1, keepdims=True)
                    tap.records.append((layer, inds, scores))
            return y

        block_cls.__call__ = tapped_moe
        return len(self.layer_of)

    def drain(self):
        """Materialize and return recorded routing, then clear."""
        out = []
        for layer, idx, w in self.records:
            # numpy can't read bfloat16 buffers — cast on the MLX side first
            out.append((layer, np.array(idx.astype(mx.int32)),
                        np.array(w.astype(mx.float32))))
        self.records = []
        return out


def load_examples(path: Path):
    """Return list of {input, ideal, id, category} from .json test sets or v5 .jsonl."""
    examples = []
    if path.suffix == ".jsonl":
        with open(path) as f:
            for i, line in enumerate(f):
                msgs = json.loads(line)["messages"]
                by_role = {m["role"]: m["content"] for m in msgs}
                examples.append({
                    "id": f"{path.stem}-{i}",
                    "category": None,
                    "input": by_role["user"],
                    "ideal": by_role.get("assistant", ""),
                })
    else:
        with open(path) as f:
            for ex in json.load(f):
                examples.append({
                    "id": ex.get("id"),
                    "category": ex.get("category"),
                    "input": ex["input"],
                    "ideal": ex.get("ideal", ""),
                })
    return examples


def build_prompt(tokenizer, system_prompt: str, user_input: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        # Some Gemma templates reject a system role — fold it into the user turn.
        merged = [{"role": "user", "content": f"{system_prompt}\n\n{user_input}"}]
        return tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )


def coverage_curve(counts_layer: np.ndarray, thresholds=(0.5, 0.9, 0.99)):
    """Smallest number of experts covering each fraction of routed assignments."""
    total = counts_layer.sum()
    if total == 0:
        return {str(t): 0 for t in thresholds}
    sorted_counts = np.sort(counts_layer)[::-1]
    cum = np.cumsum(sorted_counts) / total
    return {str(t): int(np.searchsorted(cum, t) + 1) for t in thresholds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--test-set", default=str(REPO_ROOT / "spoke/bench/test_set_evals.json"))
    parser.add_argument("--prompt-mode", default="v2", choices=sorted(PROMPTS))
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    test_path = Path(args.test_set)
    examples = load_examples(test_path)
    if args.limit:
        examples = examples[: args.limit]
    system_prompt = PROMPTS[args.prompt_mode]

    print(f"Loading {args.model} ...")
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"  loaded in {time.time() - t0:.1f}s")

    def cfg_get(c, *names):
        for n in names:
            v = c.get(n) if isinstance(c, dict) else getattr(c, n, None)
            if v:
                return v
        return None

    cfg = getattr(model, "args", None)
    text_cfg = getattr(cfg, "text_config", None) or cfg
    num_experts = cfg_get(text_cfg, "num_experts")
    top_k = cfg_get(text_cfg, "top_k_experts", "num_experts_per_tok")

    tap = RouterTap()
    n_layers = tap.install(model)
    print(f"  tapped {n_layers} routers ({num_experts} experts each, top-{top_k})")

    # Accumulators: assignments = (token, slot) routing events.
    prefill_counts = np.zeros((n_layers, num_experts), dtype=np.int64)
    decode_counts = np.zeros((n_layers, num_experts), dtype=np.int64)
    weight_mass = np.zeros((n_layers, num_experts), dtype=np.float64)
    per_example_fired = []  # example -> layer -> sorted expert list (decode+prefill union)
    union_curve = []        # cumulative unique (layer, expert) pairs after each example
    global_fired = set()

    total_prompt_toks = total_gen_toks = 0
    t_start = time.time()

    for n, ex in enumerate(examples):
        prompt = build_prompt(tokenizer, system_prompt, ex["input"])
        tap.records = []
        tap.enabled = True
        text = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens)
        tap.enabled = False

        fired = [set() for _ in range(n_layers)]
        for layer, idx, w in tap.drain():
            flat_idx = idx.reshape(-1)
            is_prefill = idx.shape[-2] > 1
            target = prefill_counts if is_prefill else decode_counts
            np.add.at(target[layer], flat_idx, 1)
            np.add.at(weight_mass[layer], flat_idx, w.reshape(-1))
            fired[layer].update(int(e) for e in np.unique(flat_idx))
            if layer == 0:  # count tokens once, not once per layer
                if is_prefill:
                    total_prompt_toks += idx.shape[-2]
                else:
                    total_gen_toks += 1

        for layer, s in enumerate(fired):
            global_fired.update((layer, e) for e in s)
        union_curve.append(len(global_fired))
        per_example_fired.append({
            "id": ex["id"],
            "category": ex["category"],
            "fired": {str(l): sorted(s) for l, s in enumerate(fired)},
            "output": text,
        })
        el = time.time() - t_start
        print(f"  [{n + 1}/{len(examples)}] unique(layer,expert)={len(global_fired)}"
              f"/{n_layers * num_experts}  ({el:.0f}s elapsed)")

    # Summary
    all_counts = prefill_counts + decode_counts
    print(f"\n=== Expert usage summary ({len(examples)} examples, "
          f"prompt={args.prompt_mode}) ===")
    print(f"  total routed assignments: prefill={prefill_counts.sum():,} "
          f"decode={decode_counts.sum():,}")
    print(f"  global unique experts fired: {len(global_fired)}/{n_layers * num_experts} "
          f"({len(global_fired) / (n_layers * num_experts):.1%})")

    per_layer = []
    for l in range(n_layers):
        cov_all = coverage_curve(all_counts[l])
        cov_dec = coverage_curve(decode_counts[l])
        uniq = int((all_counts[l] > 0).sum())
        per_layer.append({"layer": l, "unique": uniq,
                          "coverage_all": cov_all, "coverage_decode": cov_dec})
        print(f"  L{l:>2}: unique={uniq:>3}  "
              f"50%={cov_all['0.5']:>3}  90%={cov_all['0.9']:>3}  99%={cov_all['0.99']:>3}"
              f"   (decode 90%={cov_dec['0.9']:>3})")

    med90 = int(np.median([p["coverage_all"]["0.9"] for p in per_layer]))
    print(f"\n  median experts for 90% of routing mass: {med90}/{num_experts} per layer")

    model_slug = args.model.rstrip("/").split("/")[-1].lower()
    out_path = Path(args.out) if args.out else (
        Path(__file__).parent
        / f"profile_{model_slug}_{test_path.stem}_{args.prompt_mode}.json"
    )
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "prompt_mode": args.prompt_mode,
            "test_set": test_path.name,
            "n_examples": len(examples),
            "max_tokens": args.max_tokens,
            "n_layers": n_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "prompt_tokens": total_prompt_toks,
            "decode_tokens": total_gen_toks,
            "prefill_counts": prefill_counts.tolist(),
            "decode_counts": decode_counts.tolist(),
            "weight_mass": weight_mass.tolist(),
            "union_curve": union_curve,
            "per_layer_summary": per_layer,
            "per_example_fired": per_example_fired,
        }, f)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
