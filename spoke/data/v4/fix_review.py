#!/usr/bin/env python3
"""Apply all 38 review fixes to v4 raw data.

Fixes identified by 5 parallel review agents:
  - 34 deletions (broken examples)
  - 4 in-place fixes (salvageable examples)

Idempotent: safe to re-run.
Run: python spoke/data/v4/fix_review.py
"""

import json
from pathlib import Path

RAW = Path(__file__).parent / "raw"


def load(name):
    with open(RAW / name) as f:
        return json.load(f)


def save(name, data):
    with open(RAW / name, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    total_deleted = 0
    total_fixed = 0

    # ── 1. spell-simple.json (35 → 32) ──
    data = load("spell-simple.json")
    if len(data) == 35:
        data[33]["ideal"] = "The language is TypeScript with strict mode enabled."
        total_fixed += 1
        for i in sorted([28, 31, 32], reverse=True):
            del data[i]
        total_deleted += 3
        save("spell-simple.json", data)
    print(f"  spell-simple: {len(data)} examples")

    # ── 2. spell-corrective.json (45 → 43) ──
    data = load("spell-corrective.json")
    if len(data) == 45:
        for i in sorted([1, 8], reverse=True):
            del data[i]
        total_deleted += 2
        save("spell-corrective.json", data)
    print(f"  spell-corrective: {len(data)} examples")

    # ── 3. compound-quote.json (25, fix 1) ──
    data = load("compound-quote.json")
    if "broken ready" in data[9]["ideal"]:
        data[9]["ideal"] = 'She called it "broken."'
        total_fixed += 1
        save("compound-quote.json", data)
    print(f"  compound-quote: {len(data)} examples")

    # ── 4. compound-3plus.json (30 → 15) ──
    data = load("compound-3plus.json")
    if len(data) == 30:
        bad = [0, 2, 6, 7, 10, 11, 12, 14, 15, 19, 20, 23, 25, 26, 28]
        for i in sorted(bad, reverse=True):
            del data[i]
        total_deleted += 15
        save("compound-3plus.json", data)
    print(f"  compound-3plus: {len(data)} examples")

    # ── 5. caps.json (30 → 29) ──
    data = load("caps.json")
    if len(data) == 30:
        data[5]["ideal"] = "CLICK THE SUBMIT BUTTON."
        total_fixed += 1
        del data[28]
        total_deleted += 1
        save("caps.json", data)
    print(f"  caps: {len(data)} examples")

    # ── 6. emoji.json (40, fix 1) ──
    data = load("emoji.json")
    if any("😭" in ex["ideal"] for ex in data):
        for ex in data:
            if "crying emoji" in ex["input"] and "😭" in ex["ideal"]:
                ex["ideal"] = ex["ideal"].replace("😭", "😢")
                total_fixed += 1
                break
        save("emoji.json", data)
    print(f"  emoji: {len(data)} examples")

    # ── 7. hn-disfluency.json (110 → 100) ──
    data = load("hn-disfluency.json")
    if len(data) == 110:
        # Delete all "No no" correction patterns (keep "I completely understand" — pure emphasis)
        bad = [i for i, ex in enumerate(data)
               if ex["input"].startswith("No no,")
               and "I completely understand" not in ex["input"]]
        print(f"  hn-disfluency: found {len(bad)} 'No no' correction patterns to delete")
        assert len(bad) == 10, f"Expected 10, got {len(bad)}"
        for i in sorted(bad, reverse=True):
            del data[i]
        total_deleted += 10
        save("hn-disfluency.json", data)
    print(f"  hn-disfluency: {len(data)} examples")

    # ── 8. hn-symbols.json (45 → 43) ──
    data = load("hn-symbols.json")
    if len(data) == 45:
        for i in sorted([18, 36], reverse=True):
            del data[i]
        total_deleted += 2
        save("hn-symbols.json", data)
    print(f"  hn-symbols: {len(data)} examples")

    # ── 9. hn-casing.json (35 → 34) ──
    data = load("hn-casing.json")
    if len(data) == 35:
        del data[29]
        total_deleted += 1
        save("hn-casing.json", data)
    print(f"  hn-casing: {len(data)} examples")

    # ── Summary ──
    print(f"\nApplied: {total_deleted} deleted + {total_fixed} fixed = {total_deleted + total_fixed} changes")


if __name__ == "__main__":
    main()
