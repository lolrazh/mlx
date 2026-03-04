#!/usr/bin/env python3
"""Convert Spoke Evals CSV to benchmark-compatible JSON test set."""

import csv
import json
import re
import sys
from pathlib import Path


def categorize(inp: str, ideal: str) -> str:
    """Heuristically assign a category based on input/ideal patterns."""
    inp_l = inp.lower()

    # Multi-step: contains multiple command types
    signals = 0
    if "spell" in inp_l or re.search(r"\b[A-Z]-[A-Z]-[A-Z]", inp): signals += 1
    if any(w in inp_l for w in ["sorry", "wait no", "scratch that", "actually"]): signals += 1
    if "at symbol" in inp_l or "tag" in inp_l and "at" in inp_l: signals += 1
    if "quote" in inp_l: signals += 1
    if "emoji" in inp_l: signals += 1
    if "caps" in inp_l or "lowercase" in inp_l: signals += 1
    if "emphasize" in inp_l or "emphasis" in inp_l: signals += 1
    if signals >= 2:
        return "multi"

    # Emoji
    if "emoji" in inp_l:
        return "emoji"

    # Spell-replace
    if re.search(r"spell\s+(that|it|this)", inp_l) or re.search(r"\b[A-Z]-[A-Z]-[A-Z]", inp):
        return "spell-replace"

    # Quote-endquote (explicit "end quote")
    if "end quote" in inp_l and "quote-unquote" not in inp_l:
        return "quote-endquote"

    # Quote-unquote
    if "quote-unquote" in inp_l:
        return "quote-unquote"
    if "quote" in inp_l and "end quote" not in inp_l:
        return "quote-unquote"

    # Self-correction
    if any(w in inp_l for w in ["wait no, sorry", "scratch that", "actually,", "actually "]):
        # Check if the ideal is shorter (something was removed)
        if len(ideal) < len(inp) * 0.95:
            return "self-correction"

    # At-symbol
    if "at symbol" in inp_l or ("tag" in inp_l and "@" in ideal):
        return "at-symbol"

    # Caps / lowercase
    if "all caps" in inp_l or "in caps" in inp_l or "use caps" in inp_l:
        return "caps"
    if "lowercase" in inp_l:
        return "caps"

    # Emphasis
    if "emphasize" in inp_l or "emphasis" in inp_l:
        return "emphasis"

    # CamelCase (check if ideal has camelCase that input doesn't)
    if re.search(r"[a-z][A-Z]", ideal) and not re.search(r"[a-z][A-Z]", inp):
        return "camelcase"

    # Passthrough (input ≈ ideal)
    if inp.strip().rstrip(".") == ideal.strip().rstrip("."):
        return "passthrough"

    # Disfluency (filler removal, punctuation cleanup)
    if len(ideal) < len(inp) * 0.98:
        return "disfluency"

    return "passthrough"


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Spoke - Evals.csv")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("spoke/bench/test_set_evals.json")

    seen = set()
    examples = []
    idx = 1

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inp = row["input"].strip()
            ideal = row["ideal"].strip()

            # Deduplicate by input
            if inp in seen:
                continue
            seen.add(inp)

            cat = categorize(inp, ideal)
            examples.append({
                "id": idx,
                "category": cat,
                "input": inp,
                "ideal": ideal,
            })
            idx += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    # Summary
    from collections import Counter
    cats = Counter(ex["category"] for ex in examples)
    print(f"Converted {len(examples)} unique examples -> {out_path}")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:<20} {count}")


if __name__ == "__main__":
    main()
