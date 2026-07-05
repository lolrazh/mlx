#!/usr/bin/env python3
"""Benchmark Apple's on-device Foundation Model (macOS 26+) on Spoke test sets.

Drives spoke/bench/apple_fm_shim.swift (compile first — see that file's header)
and scores with the same functions as run_benchmark.py.

Usage:
    python spoke/bench/apple_fm_benchmark.py --shim /path/to/apple_fm_shim
    python spoke/bench/apple_fm_benchmark.py --shim ... --test-set spoke/bench/test_set_evals.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR))
from run_benchmark import (  # noqa: E402
    GENERIC_PROMPT,
    MINI_PROMPT,
    SPOKE_FULL_PROMPT,
    V2_PROMPT,
    V3_PROMPT,
    clean_output,
    score_output,
)

PROMPTS = {
    "generic": GENERIC_PROMPT,
    "mini": MINI_PROMPT,
    "v2": V2_PROMPT,
    "v3": V3_PROMPT,
    "spoke-full": SPOKE_FULL_PROMPT,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shim", required=True, help="Path to compiled apple_fm_shim binary")
    parser.add_argument("--test-set", default=str(BENCH_DIR / "test_set_v3.json"))
    parser.add_argument("--model-name", default="apple-fm-ondevice")
    parser.add_argument("--prompt-mode", default="v2", choices=sorted(PROMPTS))
    args = parser.parse_args()

    system_prompt = PROMPTS[args.prompt_mode]
    test_path = Path(args.test_set)
    with open(test_path) as f:
        test_data = json.load(f)

    payload = "\n".join(
        json.dumps({"system": system_prompt, "input": ex["input"]}) for ex in test_data
    )
    proc = subprocess.run(
        [args.shim], input=payload, capture_output=True, text=True, timeout=3600
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)

    outputs = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    if len(outputs) != len(test_data):
        print(f"WARNING: {len(outputs)} outputs for {len(test_data)} inputs", file=sys.stderr)

    results, scores, total_s = [], [], 0.0
    for ex, out in zip(test_data, outputs):
        raw = out["output"]
        cleaned = clean_output(raw, args.model_name)
        refused = raw.startswith("I apologize") or raw.startswith("I'm sorry") or "<<ERROR" in raw
        score = "fail" if refused else score_output(cleaned, ex["ideal"])
        scores.append(score)
        total_s += out["seconds"]
        icon = {"exact": "✓", "semantic": "~", "partial": "△", "fail": "✗"}[score]
        print(f"  [{icon}] {ex.get('category', '?'):<16} {out['seconds']:.2f}s"
              + ("  (REFUSED)" if refused else ""))
        if score != "exact":
            print(f"       expected: {ex['ideal']}")
            print(f"       got:      {cleaned[:160]}")
        results.append({
            "id": ex.get("id"),
            "category": ex.get("category"),
            "input": ex["input"],
            "ideal": ex["ideal"],
            "output": cleaned,
            "raw_output": raw,
            "refused": refused,
            "score": score,
            "gen_time_s": out["seconds"],
        })

    n = len(scores)
    exact, sem = scores.count("exact"), scores.count("semantic")
    partial, fail = scores.count("partial"), scores.count("fail")
    refusals = sum(1 for r in results if r["refused"])
    accuracy = (exact + sem) / n if n else 0.0
    print(f"\n  Accuracy: {accuracy:.0%}  (exact={exact} semantic={sem} partial={partial} fail={fail}, refusals={refusals})")
    print(f"  Avg latency: {total_s / n:.2f}s")

    out_path = BENCH_DIR / f"result_{args.model_name}_{args.prompt_mode}_{test_path.stem}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model_name,
            "prompt_mode": args.prompt_mode,
            "test_set": test_path.name,
            "n": n,
            "accuracy": accuracy,
            "exact": exact, "semantic": sem, "partial": partial, "fail": fail,
            "refusals": refusals,
            "avg_latency_s": total_s / n if n else None,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
