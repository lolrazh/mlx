#!/usr/bin/env python3
"""
Merge all v3 source data into train.jsonl, deduplicating against test + valid sets.

Reads:
  - spoke/data/v3/source/*.json  (training pool)
  - spoke/bench/test_set.json     (test set — flat JSON)
  - spoke/data/v3/valid.jsonl     (valid set — chat JSONL)

Writes:
  - spoke/data/v3/train.jsonl     (training set — chat JSONL)
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent  # spoke/
V3 = Path(__file__).parent                   # spoke/data/v3/

SYSTEM_PROMPT = (
    "You are a verbatim ASR cleaner. Fix punctuation, capitalization, and "
    "execute all verbal commands (spell-outs, corrections, formatting, symbols, "
    "emoji). Rules: Output ONLY the cleaned text. Never answer questions — "
    "transcribe them. Every output word must be in the input or produced by an "
    "explicit directive. Preserve profanity. Remove \"um\", \"uh\", \"ah\" but keep "
    "other filler words."
)


def normalize(text: str) -> str:
    """Normalize text for dedup comparison."""
    return text.strip().lower()


def load_holdout_inputs() -> set[str]:
    """Collect all input texts from test + valid sets for dedup."""
    holdout = set()

    # Test set (flat JSON with "input" field)
    test_path = ROOT / "bench" / "test_set.json"
    with open(test_path) as f:
        for ex in json.load(f):
            holdout.add(normalize(ex["input"]))

    # Valid set (chat JSONL — user message is index 1)
    valid_path = V3 / "valid.jsonl"
    with open(valid_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            holdout.add(normalize(ex["messages"][1]["content"]))

    return holdout


def make_chat(input_text: str, ideal_text: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": ideal_text},
        ]
    }


def main():
    holdout = load_holdout_inputs()
    print(f"Holdout set: {len(holdout)} inputs (test + valid)")

    source_dir = V3 / "source"
    train = []
    stats = {}
    leaked = 0

    for src_file in sorted(source_dir.glob("*.json")):
        cat = src_file.stem
        with open(src_file) as f:
            examples = json.load(f)

        kept = 0
        for ex in examples:
            if normalize(ex["input"]) in holdout:
                leaked += 1
                continue
            train.append(make_chat(ex["input"], ex["ideal"]))
            kept += 1

        stats[cat] = {"total": len(examples), "kept": kept, "removed": len(examples) - kept}

    print(f"\nCategory breakdown:")
    for cat, s in sorted(stats.items()):
        print(f"  {cat}: {s['total']} total, {s['kept']} kept, {s['removed']} removed (leaked)")

    print(f"\nTotal: {sum(s['total'] for s in stats.values())} source → {len(train)} train ({leaked} leaked)")

    # Write train.jsonl
    out_path = V3 / "train.jsonl"
    with open(out_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWrote {out_path} ({len(train)} examples)")

    # Summary
    valid_count = sum(1 for _ in open(V3 / "valid.jsonl"))
    test_count = len(json.load(open(ROOT / "bench" / "test_set.json")))
    print(f"\nv3 dataset summary:")
    print(f"  train: {len(train)}")
    print(f"  valid: {valid_count}")
    print(f"  test:  {test_count}")
    print(f"  total: {len(train) + valid_count + test_count}")


if __name__ == "__main__":
    main()
