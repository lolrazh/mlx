#!/usr/bin/env python3
"""Convert ALL **bold** emphasis to ALL CAPS across V3/V4 training data.

The model was getting confused by mixed emphasis conventions:
  - Some examples: "Emphasize X" → **X** (markdown bold)
  - Some examples: "Stress X" → X (ALL CAPS)

This script standardizes EVERYTHING to ALL CAPS.

Also fixes input trigger phrases that explicitly say "bold/stars/asterisks"
to use emphasis language instead, so input→output mapping is consistent.

Run: python spoke/data/v5/fix_bold_to_caps.py
"""

import json
import re
from pathlib import Path

V3_SOURCE = Path(__file__).parent.parent / "v3" / "source"
V4_RAW = Path(__file__).parent.parent / "v4" / "raw"


def load(path):
    with open(path) as f:
        return json.load(f)


def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def bold_to_caps(text):
    """Replace **word** with WORD (ALL CAPS)."""
    def replacer(m):
        return m.group(1).upper()
    return re.sub(r'\*\*(.+?)\*\*', replacer, text)


def fix_input_triggers(text):
    """Fix input phrases that explicitly reference bold/stars/asterisks."""
    replacements = [
        # "Emphasize with stars" → "Emphasize that"
        ("Emphasize with stars", "Emphasize that"),
        # "double asterisks around that" → "emphasize that"
        ("Double asterisks around that", "Emphasize that"),
        ("double asterisks around that", "emphasize that"),
        # "make that bold with stars" → "make that all caps"
        ("make that bold with stars", "make that all caps"),
        ("Make that bold with stars", "Make that all caps"),
        # "bold that day" → "all caps that day"
        ("bold that day", "all caps that day"),
        ("Bold that day", "All caps that day"),
        # "we're launching in q3. correction, q4. make that bold with stars."
        # → keep "correction, q4. make that all caps."
        # "Put it in bold" → "Put it in all caps"
        ("Put it in bold", "Put it in all caps"),
        ("put it in bold", "put it in all caps"),
        # "Put X in bold" → "Put X in all caps"
        ("in bold", "in all caps"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def process_file(path):
    """Process one JSON file, return count of fixes."""
    data = load(path)
    fixes = 0

    for i, ex in enumerate(data):
        old_ideal = ex["ideal"]
        old_input = ex["input"]

        # Fix ideal: **bold** → ALL CAPS
        new_ideal = bold_to_caps(old_ideal)

        # Fix input: references to "bold/stars" → emphasis/caps language
        new_input = fix_input_triggers(old_input)

        if new_ideal != old_ideal or new_input != old_input:
            ex["ideal"] = new_ideal
            ex["input"] = new_input
            fixes += 1
            if new_ideal != old_ideal:
                print(f"    #{i+1} ideal: {old_ideal[:80]}")
                print(f"         → {new_ideal[:80]}")
            if new_input != old_input:
                print(f"    #{i+1} input: {old_input[:80]}")
                print(f"          → {new_input[:80]}")

    if fixes:
        save(path, data)

    return fixes


def main():
    total = 0

    # V3 emphasis
    path = V3_SOURCE / "emphasis.json"
    if path.exists():
        print(f"\n  v3/source/emphasis.json:")
        n = process_file(path)
        total += n
        print(f"    → {n} fixes")

    # V4 files with bold
    for name in [
        "emphasis.json",
        "spell-compound.json",
        "compound-3plus.json",
        "compound-quote.json",
        "compound-selfcorr.json",
    ]:
        path = V4_RAW / name
        if path.exists():
            print(f"\n  v4/raw/{name}:")
            n = process_file(path)
            total += n
            print(f"    → {n} fixes")

    print(f"\n  Total: {total} examples fixed (bold → ALL CAPS)")
    print(f"\n  Next: rebuild train.jsonl with 'python spoke/data/v5/build.py'")
    # Also rebuild V4 if needed
    print(f"  Also: rebuild V4 train.jsonl with 'python spoke/data/v4/build.py'")


if __name__ == "__main__":
    main()
