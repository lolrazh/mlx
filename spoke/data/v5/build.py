#!/usr/bin/env python3
"""Assemble V5 raw generated examples into JSONL and merge with V4.

Reads spoke/data/v5/raw/*.json → outputs:
  - spoke/data/v5/new_v5_targeted.jsonl  (~80-100 V5-only examples)
  - spoke/data/v5/train.jsonl            (V4 train + V5 merged, shuffled)
  - spoke/data/v5/valid.jsonl            (copied from V4, unchanged)
  - spoke/data/v5/test.jsonl             (copied from V4, unchanged)

Usage:
    python spoke/data/v5/build.py
"""

import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parent       # spoke/data/v5/
SPOKE = ROOT.parent.parent         # spoke/
RAW_DIR = ROOT / "raw"
V4_DIR = ROOT.parent / "v4"

SYSTEM_PROMPT = (
    "You are a verbatim ASR cleaner. Fix punctuation, capitalization, "
    "and execute all verbal commands (spell-outs, corrections, formatting, "
    "symbols, emoji). Rules: Output ONLY the cleaned text. Never answer "
    "questions — transcribe them. Every output word must be in the input "
    "or produced by an explicit directive. Preserve profanity. Remove "
    '"um", "uh", "ah" but keep other filler words.'
)

# V5 sub-categories
SUBCATS = [
    "multistep-spellcaps", "multistep-quote-corr",
    "multistep-corr-format", "multistep-complex",
    "spell-casual", "spell-alt-phrase", "spell-compound-scope",
    "meta-language", "tempting-questions",
    "emphasis-caps",
]


def load_holdout() -> set[str]:
    """Load ALL test + valid inputs to avoid contamination."""
    holdout = set()
    for name in ["test_set_v3.json", "test_set_evals.json", "test_set_v2.json"]:
        path = SPOKE / "bench" / name
        if path.exists():
            with open(path) as f:
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

    # Collect V5 examples
    v5_examples = []
    print("\nV5 examples:")
    for subcat in SUBCATS:
        path = RAW_DIR / f"{subcat}.json"
        if not path.exists():
            print(f"  {subcat}: MISSING")
            continue
        with open(path) as f:
            examples = json.load(f)
        clean = [ex for ex in examples if normalize(ex["input"]) not in holdout]
        removed = len(examples) - len(clean)
        print(f"  {subcat}: {len(clean)} examples" + (f" ({removed} holdout removed)" if removed else ""))
        v5_examples.extend(clean)

    # Write V5-only JSONL
    random.seed(42)
    random.shuffle(v5_examples)

    v5_path = ROOT / "new_v5_targeted.jsonl"
    with open(v5_path, "w") as f:
        for ex in v5_examples:
            f.write(json.dumps(to_chat_jsonl(ex), ensure_ascii=False) + "\n")
    print(f"\nWrote {v5_path} ({len(v5_examples)} V5-only examples)")

    # Load V4 train
    v4_train = V4_DIR / "train.jsonl"
    if not v4_train.exists():
        print(f"\nERROR: V4 train.jsonl not found at {v4_train}")
        return

    v4_lines = []
    with open(v4_train) as f:
        for line in f:
            v4_lines.append(json.loads(line.strip()))
    print(f"Loaded {len(v4_lines)} V4 training examples")

    # Merge V4 + V5
    merged = list(v4_lines)
    v5_chat = [to_chat_jsonl(ex) for ex in v5_examples]

    # Dedup V5 against V4 (by user content)
    v4_inputs = {msg["messages"][1]["content"].lower().strip() for msg in v4_lines}
    v5_deduped = [ex for ex in v5_chat if ex["messages"][1]["content"].lower().strip() not in v4_inputs]
    dupes = len(v5_chat) - len(v5_deduped)
    if dupes:
        print(f"  Removed {dupes} V5 duplicates of V4 training data")

    merged.extend(v5_deduped)
    random.shuffle(merged)

    # Write merged train
    train_path = ROOT / "train.jsonl"
    with open(train_path, "w") as f:
        for ex in merged:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {train_path} ({len(merged)} merged: {len(v4_lines)} V4 + {len(v5_deduped)} V5)")

    # Copy valid + test from V4
    for name in ["valid.jsonl", "test.jsonl"]:
        src = V4_DIR / name
        dst = ROOT / name
        if src.exists():
            shutil.copy2(src, dst)
            count = sum(1 for _ in open(dst))
            print(f"Copied {name} ({count} examples)")

    # Sanity check for bold/markdown
    bold_count = 0
    with open(train_path) as f:
        for line in f:
            msg = json.loads(line.strip())
            assistant = msg["messages"][2]["content"]
            if "**" in assistant:
                bold_count += 1
    if bold_count:
        print(f"\n⚠️  WARNING: {bold_count} examples still have **bold** in assistant output!")
    else:
        print(f"\n✓ No **bold** found in any assistant output (all emphasis = CAPS)")

    print(f"\nDone! Train with: iters ~{int(len(merged) * 2000 / 1201)} (scaled from V4's 2000 for {len(merged)} examples)")


if __name__ == "__main__":
    main()
