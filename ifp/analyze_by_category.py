#!/usr/bin/env python3
"""Analyze per-category / per-example expert routing from profile_by_category output.

Answers three questions:
  1. Signature — which experts most distinguish each category from the global mean?
  2. Discriminability — are category routing-distributions distinct (cosine dist)?
  3. Locality — do same-category examples route more similarly than random pairs?

Usage:
    python ifp/analyze_by_category.py ifp/bycat_gemma4-26b-a4b_test_set_evals_v2.json
"""

import json
import sys
from collections import defaultdict

import numpy as np


def main():
    path = sys.argv[1]
    d = json.load(open(path))
    nL, nE = d["n_layers"], d["num_experts"]
    ex = d["per_example"]

    # Per-example flattened decode distribution (normalized to a probability vector)
    vecs, cats = [], []
    for e in ex:
        c = np.array(e["decode_counts"], dtype=np.float64).reshape(-1)  # nL*nE
        s = c.sum()
        vecs.append(c / s if s else c)
        cats.append(e["category"] or "?")
    vecs = np.array(vecs)
    cats = np.array(cats)

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na and nb else 0.0

    # --- populated categories only (>=3 examples) ---
    from collections import Counter
    counts = Counter(cats)
    pop = sorted([c for c, n in counts.items() if n >= 3], key=lambda c: -counts[c])

    print(f"Model: {d['model']}  prompt={d['prompt_mode']}  ({len(ex)} examples)")
    print(f"Grid: {nL} layers x {nE} experts = {nL*nE} routing slots\n")

    # Category mean distributions
    cat_mean = {c: vecs[cats == c].mean(axis=0) for c in pop}
    global_mean = vecs.mean(axis=0)

    # --- Q2: discriminability — cosine SIMILARITY between category means ---
    print("=== Category routing-distribution cosine similarity (1.0=identical) ===")
    print("     " + " ".join(f"{c[:6]:>6}" for c in pop))
    for a in pop:
        row = " ".join(f"{cos(cat_mean[a], cat_mean[b]):>6.2f}" for b in pop)
        print(f"{a[:10]:>10} {row}")
    off = [cos(cat_mean[a], cat_mean[b]) for i, a in enumerate(pop) for b in pop[i+1:]]
    print(f"\n  mean off-diagonal similarity: {np.mean(off):.3f} "
          f"(min {np.min(off):.3f}, max {np.max(off):.3f})")

    # --- Q3: locality — within vs across category example similarity ---
    within, across = [], []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            s = cos(vecs[i], vecs[j])
            (within if cats[i] == cats[j] else across).append(s)
    print(f"\n=== Per-example routing similarity ===")
    print(f"  within-category pairs:  mean {np.mean(within):.3f}  (n={len(within)})")
    print(f"  across-category pairs:  mean {np.mean(across):.3f}  (n={len(across)})")
    print(f"  separation (within - across): {np.mean(within)-np.mean(across):+.3f}")

    # --- Q1: signature experts per category (largest positive lift over global) ---
    print(f"\n=== Top signature experts per category (layer.expert : lift over global) ===")
    for c in pop:
        lift = cat_mean[c] - global_mean
        top = np.argsort(lift)[::-1][:5]
        sig = "  ".join(f"L{t//nE}.E{t%nE}({lift[t]*1000:+.1f})" for t in top)
        print(f"  {c:<16} {sig}")


if __name__ == "__main__":
    main()
