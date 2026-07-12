#!/usr/bin/env python3
"""Sweep PLD (n,k) configs against plain greedy on broad58, single model load."""
import json, time, sys, argparse
from pathlib import Path
import mlx.core as mx
import mlx_lm
BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR))
from pld_bench import generate_tokens, no_think_bias_ids, summarize
from run_benchmark import build_prompt


def run_config(model, tokenizer, model_path, test_set, ban_ids, n, k, use_pld):
    # warm
    wp = tokenizer.encode(build_prompt(tokenizer, "test", model_path, prompt_mode="v2"))
    generate_tokens(model, tokenizer, wp, max_tokens=8, n=n, k=(k if use_pld else 0), ban_ids=ban_ids)
    rows = []
    for ex in test_set:
        pids = tokenizer.encode(build_prompt(tokenizer, ex["input"], model_path, prompt_mode="v2"))
        t0 = time.perf_counter()
        toks, stats = generate_tokens(model, tokenizer, pids, max_tokens=256,
                                      n=n, k=(k if use_pld else 0), ban_ids=ban_ids)
        dt = (time.perf_counter() - t0) * 1000
        rows.append({"id": ex["id"], "tokens": toks, "ms": dt, "stats": stats})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="spoke/models/g4e4b-champion-mlx-dwq4-g64")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    test_set = json.load(open(BENCH_DIR / "test_set_evals.json"))
    print(f"Loaded {len(test_set)} examples; model={args.model}")
    model, tokenizer = mlx_lm.load(args.model)
    ban = no_think_bias_ids(tokenizer)

    configs = [("greedy", 0, 0, False),
               ("pld_n2_k1", 2, 1, True),
               ("pld_n2_k2", 2, 2, True),
               ("pld_n2_k3", 2, 3, True),
               ("pld_n2_k4", 2, 4, True),
               ("pld_n3_k3", 3, 3, True),
               ("pld_n3_k4", 3, 4, True)]

    greedy_tokens = None
    table = []
    for label, n, k, use_pld in configs:
        rows = run_config(model, tokenizer, args.model, test_set, ban, n, k, use_pld)
        s = summarize([{"ms": r["ms"], "stats": r["stats"]} for r in rows], label)
        if label == "greedy":
            greedy_tokens = {r["id"]: r["tokens"] for r in rows}
            s["identical"] = True
            s["mismatches"] = []
        else:
            mism = [r["id"] for r in rows if r["tokens"] != greedy_tokens[r["id"]]]
            s["identical"] = len(mism) == 0
            s["mismatches"] = mism
        table.append(s)
        print(f"  {label:12} mean={s['mean_ms']:7.1f} med={s['median_ms']:7.1f} p90={s['p90_ms']:7.1f} "
              f"acc/step={s['accepted_per_step']:.2f} tps={s['tokens_per_step']:.2f} identical={s['identical']}")

    g = table[0]["mean_ms"]
    gm = table[0]["median_ms"]
    print("\n== SPEEDUP vs greedy ==")
    for s in table:
        print(f"  {s['label']:12} mean_speedup={g/s['mean_ms']:.2f}x  median_speedup={gm/s['median_ms']:.2f}x  <500ms_mean={s['mean_ms']<500}")

    if args.out:
        json.dump({"model": args.model, "n_examples": len(test_set), "table": table},
                  open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
