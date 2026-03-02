#!/usr/bin/env python3
"""Assemble v4 raw generated examples into two JSONL files.

Reads spoke/data/v4/raw/*.json → outputs:
  - spoke/data/v4/new_regular.jsonl   (~400-500 regular correction examples)
  - spoke/data/v4/new_hard_negatives.jsonl  (~300 hard negative examples)

Deduplicates against v3 test + valid holdout sets.

Usage:
    python spoke/data/v4/build.py
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).parent       # spoke/data/v4/
SPOKE = ROOT.parent.parent         # spoke/
RAW_DIR = ROOT / "raw"

SYSTEM_PROMPT = (
    "You are a verbatim ASR cleaner. Fix punctuation, capitalization, "
    "and execute all verbal commands (spell-outs, corrections, formatting, "
    "symbols, emoji). Rules: Output ONLY the cleaned text. Never answer "
    "questions — transcribe them. Every output word must be in the input "
    "or produced by an explicit directive. Preserve profanity. Remove "
    '"um", "uh", "ah" but keep other filler words.'
)

# Sub-categories → which file they belong to
REGULAR_SUBCATS = [
    "spell-simple", "spell-corrective", "spell-compound",
    "compound-selfcorr", "compound-quote", "compound-3plus",
    "selfcorr-partial", "selfcorr-mid", "selfcorr-ambiguous",
    "caps", "emphasis", "emoji", "disfluency",
]

HN_SUBCATS = [
    "hn-disfluency", "hn-quote", "hn-symbols", "hn-casing", "hn-spelling",
]


def load_holdout() -> set[str]:
    """Load test+valid inputs to avoid contamination."""
    holdout = set()
    test_path = SPOKE / "bench" / "test_set.json"
    if test_path.exists():
        with open(test_path) as f:
            for ex in json.load(f):
                holdout.add(ex["input"].lower().strip())
    valid_path = ROOT.parent / "v3" / "valid.jsonl"
    if valid_path.exists():
        with open(valid_path) as f:
            for line in f:
                msg = json.loads(line.strip())
                holdout.add(msg["messages"][1]["content"].lower().strip())
    return holdout


def to_chat_jsonl(ex: dict) -> dict:
    """Convert {input, ideal} to chat JSONL format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["input"]},
            {"role": "assistant", "content": ex["ideal"]},
        ]
    }


def normalize(text: str) -> str:
    return text.lower().strip()


def main():
    holdout = load_holdout()
    print(f"Loaded {len(holdout)} holdout examples")

    # Collect regular examples
    regular = []
    print("\nRegular examples:")
    for subcat in REGULAR_SUBCATS:
        path = RAW_DIR / f"{subcat}.json"
        if not path.exists():
            print(f"  {subcat}: MISSING")
            continue
        with open(path) as f:
            examples = json.load(f)
        # Filter holdout
        clean = [ex for ex in examples if normalize(ex["input"]) not in holdout]
        removed = len(examples) - len(clean)
        print(f"  {subcat}: {len(clean)} examples" + (f" ({removed} holdout removed)" if removed else ""))
        for ex in clean:
            ex["_subcat"] = subcat
        regular.extend(clean)

    # Collect hard negatives
    hard_neg = []
    print("\nHard negatives:")
    for subcat in HN_SUBCATS:
        path = RAW_DIR / f"{subcat}.json"
        if not path.exists():
            print(f"  {subcat}: MISSING")
            continue
        with open(path) as f:
            examples = json.load(f)
        clean = [ex for ex in examples if normalize(ex["input"]) not in holdout]
        removed = len(examples) - len(clean)
        print(f"  {subcat}: {len(clean)} examples" + (f" ({removed} holdout removed)" if removed else ""))
        for ex in clean:
            ex["_subcat"] = subcat
        hard_neg.extend(clean)

    # Shuffle (mix categories randomly as brief requests)
    random.seed(42)
    random.shuffle(regular)
    random.shuffle(hard_neg)

    # Write JSONL files
    reg_path = ROOT / "new_regular.jsonl"
    with open(reg_path, "w") as f:
        for ex in regular:
            f.write(json.dumps(to_chat_jsonl(ex), ensure_ascii=False) + "\n")
    print(f"\nWrote {reg_path} ({len(regular)} examples)")

    hn_path = ROOT / "new_hard_negatives.jsonl"
    with open(hn_path, "w") as f:
        for ex in hard_neg:
            f.write(json.dumps(to_chat_jsonl(ex), ensure_ascii=False) + "\n")
    print(f"Wrote {hn_path} ({len(hard_neg)} examples)")

    print(f"\nTotal: {len(regular)} regular + {len(hard_neg)} hard negatives = {len(regular) + len(hard_neg)}")


if __name__ == "__main__":
    main()
