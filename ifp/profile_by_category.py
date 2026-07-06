#!/usr/bin/env python3
"""Per-category / per-example expert-routing profiler (ifp-lite granular).

The first-pass profiler (profile_experts.py) stored only GLOBAL expert counts
plus per-example fired *sets*. But a single Spoke example fires ~73% of all
experts, so fired/not-fired is not discriminative — the signal lives in the
mass distribution. This script stores, for every example, its full per-layer
decode-mass vector (n_layers x n_experts), keyed by category, so all downstream
questions can be answered offline:

  - Do examples of the same category (emoji, spell-replace, quote...) route to
    a distinct set of experts?
  - Is within-category routing more similar than across-category?
  - Which experts are the "signature" of each command type?

We use DECODE mass (the generated output), not prefill: the system prompt is
identical across every example in a prompt-mode, so prefill routing is
dominated by the shared prompt and washes out the per-input signal.

Usage:
    python ifp/profile_by_category.py --model mlx-community/gemma-4-26B-A4B-it-qat-4bit
    python ifp/profile_by_category.py --model ... --prompt-mode spoke-full
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from mlx_lm import load
from mlx_lm.generate import generate

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from profile_experts import (  # noqa: E402
    PROMPTS,
    RouterTap,
    build_prompt,
    load_examples,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/gemma-4-26B-A4B-it-qat-4bit")
    parser.add_argument("--test-set", default=str(HERE.parent / "spoke/bench/test_set_evals.json"))
    parser.add_argument("--prompt-mode", default="v2", choices=sorted(PROMPTS))
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    test_path = Path(args.test_set)
    examples = load_examples(test_path)
    system_prompt = PROMPTS[args.prompt_mode]

    print(f"Loading {args.model} ...", flush=True)
    model, tokenizer = load(args.model)
    tap = RouterTap()
    n_layers = tap.install(model)
    # infer expert count from the first router's projection
    num_experts = None

    per_example = []  # {id, category, output, decode_counts (nL x nE)}
    t_start = time.time()

    for n, ex in enumerate(examples):
        prompt = build_prompt(tokenizer, system_prompt, ex["input"])
        tap.records = []
        tap.enabled = True
        text = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens)
        tap.enabled = False

        drained = tap.drain()
        if num_experts is None:
            num_experts = int(max(idx.max() for _, idx, _ in drained)) + 1
        dec = np.zeros((n_layers, num_experts), dtype=np.int32)
        for layer, idx, w in drained:
            is_prefill = idx.shape[-2] > 1
            if is_prefill:
                continue  # decode only — see module docstring
            np.add.at(dec[layer], idx.reshape(-1), 1)

        per_example.append({
            "id": ex["id"],
            "category": ex["category"],
            "output": text,
            "decode_counts": dec.tolist(),
        })
        print(f"  [{n+1}/{len(examples)}] {ex['category'] or '?':<16} "
              f"decode_toks={int(dec.sum()//8)}  ({time.time()-t_start:.0f}s)", flush=True)

    model_slug = args.model.rstrip("/").split("/")[-1].lower()
    out_path = Path(args.out) if args.out else (
        HERE / f"bycat_{model_slug}_{test_path.stem}_{args.prompt_mode}.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "prompt_mode": args.prompt_mode,
            "test_set": test_path.name,
            "n_layers": n_layers,
            "num_experts": num_experts,
            "per_example": per_example,
        }, f)
    print(f"  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
