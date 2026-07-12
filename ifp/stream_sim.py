#!/usr/bin/env python3
"""Offline streaming-cache simulation from saved IFP routing traces.

Question: if we hold the top-K experts PER LAYER resident (a static hot-set,
chosen by observed decode firing counts) and stream the cold 128-K from disk
on their rare hits, what fraction of decode expert-accesses hit RAM, and how
much expert memory do we free?

Uses decode_counts (firing COUNTS = disk-access events), which is the right
metric for streaming latency (each firing of a non-cached expert = 1 SSD read).
Mass-weighting (softmax importance) matters for pruning quality, not streaming.
"""
import json
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "ifp/profile_gemma4-26b-a4b_test_set_evals_v2.json"
d = json.load(open(path))
dec = np.array(d["decode_counts"], dtype=np.float64)   # [L, E]
pre = np.array(d["prefill_counts"], dtype=np.float64)
L, E = dec.shape
top_k = d["top_k"]
print(f"model={d['model']}")
print(f"layers={L}  experts/layer={E}  top_k={top_k}  decode_tokens={d['decode_tokens']}")
print(f"total decode expert-accesses = {int(dec.sum()):,}  (= decode_tokens * L * top_k)\n")

def hit_curve(counts):
    """For each K, fraction of total firings covered by each layer's top-K experts."""
    total = counts.sum()
    # sort each layer's counts descending, cumulative-sum across experts, then
    # sum over layers at each K -> global coverage if we cache top-K PER LAYER.
    srt = np.sort(counts, axis=1)[:, ::-1]          # [L,E] desc per layer
    cum = np.cumsum(srt, axis=1)                     # [L,E] cumulative per layer
    per_K = cum.sum(axis=0) / total                  # [E] global hit-rate at cache size K (1..E)
    return per_K

hit_dec = hit_curve(dec)

print("Static hot-set cache (top-K experts per layer), DECODE:")
print(f"{'K/layer':>8} {'%experts kept':>13} {'RAM freed':>10} {'cache HIT%':>10} {'disk MISS%':>10}")
for K in [8, 16, 24, 32, 48, 56, 64, 80, 96, 112, 128]:
    keep_frac = K / E
    freed = (E - K) / E
    hit = hit_dec[K-1] * 100
    print(f"{K:>8} {keep_frac*100:>12.1f}% {freed*100:>9.1f}% {hit:>9.1f}% {100-hit:>9.1f}%")

# find K for target hit rates
print("\nCache size needed for target decode hit-rate:")
for tgt in [0.90, 0.95, 0.99, 0.999]:
    K = int(np.searchsorted(hit_dec, tgt) + 1)
    print(f"  hit>={tgt*100:.1f}%  ->  K={K}/{E} per layer  ({K/E*100:.0f}% of experts resident, {(E-K)/E*100:.0f}% streamed)")

# cross-example working set (locality): from per_example_fired
pef = d.get("per_example_fired")
if pef:
    # each entry: list over layers of fired-expert lists (union decode+prefill)
    ex_sizes = []
    for ex in pef:
        layers = ex["fired"] if isinstance(ex, dict) and "fired" in ex else ex
        if isinstance(layers, dict):
            layers = list(layers.values())
        sizes = [len(l) for l in layers]
        ex_sizes.append(np.mean(sizes))
    ex_sizes = np.array(ex_sizes)
    print(f"\nPer-example working set (avg experts fired per layer, decode+prefill union):")
    print(f"  median {np.median(ex_sizes):.0f}/{E}, min {ex_sizes.min():.0f}, max {ex_sizes.max():.0f}")
    print(f"  -> a SINGLE example already needs ~{np.median(ex_sizes)/E*100:.0f}% of experts (incl. prefill).")
