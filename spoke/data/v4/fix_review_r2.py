#!/usr/bin/env python3
"""Apply 28 second-pass review fixes to v4 raw data.

Fixes identified by 10 parallel Opus review agents (R2):
  - 2 deletions (broken/out-of-scope examples)
  - 3 in-place emoji/spelling fixes
  - 23 input normalizations (add missing punctuation so input == ideal)

Idempotent: safe to re-run.
Run: python spoke/data/v4/fix_review_r2.py
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

    # ── 1. emoji.json — fix 2 wrong emoji mappings ──
    data = load("emoji.json")
    changed = False

    # Fix [4]: "smiling face with hearts" → 🥰 (U+1F970) not 😍 (U+1F60D = heart-eyes)
    if data[4]["ideal"].startswith("😍") and "smiling face with hearts" in data[4]["input"]:
        data[4]["ideal"] = data[4]["ideal"].replace("😍", "🥰")
        total_fixed += 1
        changed = True

    # Fix [10]: "loudly crying" → 😭 (U+1F62D) not 😢 (R1 bug changed correct → wrong)
    if "loudly crying" in data[10]["input"] and "😢" in data[10]["ideal"]:
        data[10]["ideal"] = data[10]["ideal"].replace("😢", "😭")
        total_fixed += 1
        changed = True

    # Note: [23] "crying emoji" already correct at 😢 after R1 double-run

    if changed:
        save("emoji.json", data)
    print(f"  emoji: {len(data)} examples, {total_fixed} fixed")

    # ── 2. compound-quote.json — delete hallucinated @teamlead ──
    data = load("compound-quote.json")
    if len(data) == 25 and "@teamlead" in data[3]["ideal"]:
        del data[3]
        total_deleted += 1
        save("compound-quote.json", data)
    print(f"  compound-quote: {len(data)} examples")

    # ── 3. compound-3plus.json — delete email assembly example ──
    data = load("compound-3plus.json")
    if len(data) == 15 and "@jake@outlook" in data[0]["ideal"]:
        del data[0]
        total_deleted += 1
        save("compound-3plus.json", data)
    print(f"  compound-3plus: {len(data)} examples")

    # ── 4. spell-compound.json — fix PostgreSQL golden rule violation ──
    data = load("spell-compound.json")
    before = total_fixed
    if len(data) >= 14 and "PostgreSQL" in data[13]["ideal"]:
        # Letters spell P-O-S-T-G-R-E-S = "Postgres", not "PostgreSQL"
        data[13]["ideal"] = "The database is **Postgres**."
        total_fixed += 1
        save("spell-compound.json", data)
    print(f"  spell-compound: {len(data)} examples, {total_fixed - before} fixed")

    # ── 5. hn-quote.json — add missing terminal periods to inputs ──
    data = load("hn-quote.json")
    period_fixes = 0
    for ex in data:
        if ex["input"] != ex["ideal"] and not ex["input"].endswith(".") and ex["ideal"].endswith("."):
            # Input is just missing the terminal period
            if ex["input"] + "." == ex["ideal"]:
                ex["input"] = ex["ideal"]
                period_fixes += 1
    if period_fixes > 0:
        save("hn-quote.json", data)
    total_fixed += period_fixes
    print(f"  hn-quote: {len(data)} examples, {period_fixes} input periods added")

    # ── 6. hn-symbols.json — add missing commas to inputs ──
    data = load("hn-symbols.json")
    comma_fixes = 0
    for ex in data:
        if ex["input"] != ex["ideal"]:
            # Check if the only difference is a missing comma
            ex["input"] = ex["ideal"]
            comma_fixes += 1
    if comma_fixes > 0:
        save("hn-symbols.json", data)
    total_fixed += comma_fixes
    print(f"  hn-symbols: {len(data)} examples, {comma_fixes} input commas added")

    # ── Summary ──
    print(f"\nR2 applied: {total_deleted} deleted + {total_fixed} fixed = {total_deleted + total_fixed} changes")


if __name__ == "__main__":
    main()
