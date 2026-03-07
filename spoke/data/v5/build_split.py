#!/usr/bin/env python3
"""Rebuild V5 dataset with stratified 80:10:10 train/valid/test split.

Pools ALL source data from v3/source/, v4/raw/, v5/raw/ with category labels,
maps sub-categories to broad categories, deduplicates, then does a stratified
split so every broad category is represented in all three sets.

Usage:
    python spoke/data/v5/build_split.py
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).parent          # spoke/data/v5/
DATA = ROOT.parent                    # spoke/data/

SYSTEM_PROMPT = (
    "You are a verbatim ASR cleaner. Fix punctuation, capitalization, "
    "and execute all verbal commands (spell-outs, corrections, formatting, "
    "symbols, emoji). Rules: Output ONLY the cleaned text. Never answer "
    "questions — transcribe them. Every output word must be in the input "
    "or produced by an explicit directive. Preserve profanity. Remove "
    '"um", "uh", "ah" but keep other filler words.'
)

# Sub-category -> broad category mapping
CATEGORY_MAP = {
    # v3/source/
    "spell-replace": "spell",
    "self-correction": "self-correction",
    "quote-endquote": "quote",
    "quote-unquote": "quote",
    "at-symbol": "at-symbol",
    "caps": "caps",
    "emphasis": "emphasis",
    "emoji": "emoji",
    "camelcase": "camelcase",
    # v4/raw/ regular
    "spell-simple": "spell",
    "spell-corrective": "spell",
    "spell-compound": "spell",
    "compound-selfcorr": "multi",
    "compound-quote": "multi",
    "compound-3plus": "multi",
    "selfcorr-partial": "self-correction",
    "selfcorr-mid": "self-correction",
    "selfcorr-ambiguous": "self-correction",
    "caps": "caps",
    "emphasis": "emphasis",
    "emoji": "emoji",
    "disfluency": "disfluency",
    # v4/raw/ hard negatives
    "hn-disfluency": "hard-negative",
    "hn-quote": "hard-negative",
    "hn-symbols": "hard-negative",
    "hn-casing": "hard-negative",
    "hn-spelling": "hard-negative",
    # v5/raw/
    "multistep-spellcaps": "multi",
    "multistep-quote-corr": "multi",
    "multistep-corr-format": "multi",
    "multistep-complex": "multi",
    "spell-casual": "spell",
    "spell-alt-phrase": "spell",
    "spell-compound-scope": "spell",
    "meta-language": "meta",
    "tempting-questions": "meta",
    "emphasis-caps": "emphasis",
}

SOURCES = [
    ("v3/source", DATA / "v3" / "source"),
    ("v4/raw", DATA / "v4" / "raw"),
    ("v5/raw", DATA / "v5" / "raw"),
]


def to_chat(ex: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["input"]},
            {"role": "assistant", "content": ex["ideal"]},
        ]
    }


def main():
    random.seed(42)

    # 1. Pool all examples with category labels
    all_examples = []  # list of (broad_cat, subcat, source, {input, ideal})
    seen_inputs = set()

    for source_name, source_dir in SOURCES:
        if not source_dir.exists():
            print(f"WARNING: {source_dir} not found, skipping")
            continue
        for path in sorted(source_dir.glob("*.json")):
            subcat = path.stem
            broad = CATEGORY_MAP.get(subcat)
            if broad is None:
                print(f"  WARNING: unknown subcat '{subcat}' in {source_name}, skipping")
                continue
            with open(path) as f:
                examples = json.load(f)
            for ex in examples:
                key = ex["input"].lower().strip()
                if key in seen_inputs:
                    continue  # dedup
                seen_inputs.add(key)
                all_examples.append((broad, subcat, source_name, ex))

    print(f"Pooled {len(all_examples)} unique examples\n")

    # 2. Group by broad category
    by_cat: dict[str, list] = {}
    for broad, subcat, source, ex in all_examples:
        by_cat.setdefault(broad, []).append((subcat, source, ex))

    print("Broad category counts:")
    for cat in sorted(by_cat):
        print(f"  {cat}: {len(by_cat[cat])}")

    # 3. Stratified 80:10:10 split
    train, valid, test = [], [], []

    print("\nSplit allocation:")
    for cat in sorted(by_cat):
        items = by_cat[cat]
        random.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * 0.10))
        n_valid = max(1, round(n * 0.10))
        n_train = n - n_test - n_valid

        cat_test = items[:n_test]
        cat_valid = items[n_test:n_test + n_valid]
        cat_train = items[n_test + n_valid:]

        print(f"  {cat}: {n_train} train / {n_valid} valid / {n_test} test (total {n})")

        for subcat, source, ex in cat_train:
            train.append({"_cat": cat, "_subcat": subcat, **to_chat(ex)})
        for subcat, source, ex in cat_valid:
            valid.append({"_cat": cat, "_subcat": subcat, **to_chat(ex)})
        for subcat, source, ex in cat_test:
            test.append({"_cat": cat, "_subcat": subcat, **to_chat(ex)})

    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)

    # 4. Write JSONL (strip internal metadata for training files)
    def write_jsonl(path, data, keep_meta=False):
        with open(path, "w") as f:
            for row in data:
                if keep_meta:
                    out = row
                else:
                    out = {"messages": row["messages"]}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    train_path = ROOT / "train.jsonl"
    valid_path = ROOT / "valid.jsonl"
    test_path = ROOT / "test.jsonl"

    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)
    write_jsonl(test_path, test)

    # Also write a categorized test set for benchmarking (same format as test_set_v3.json)
    bench_test = []
    for row in test:
        bench_test.append({
            "id": len(bench_test) + 1,
            "category": row["_cat"],
            "input": row["messages"][1]["content"],
            "ideal": row["messages"][2]["content"],
        })
    bench_path = ROOT.parent.parent / "bench" / "test_set_v5.json"
    with open(bench_path, "w") as f:
        json.dump(bench_test, f, indent=2, ensure_ascii=False)

    # Also write a categorized valid set for reference
    bench_valid = []
    for row in valid:
        bench_valid.append({
            "id": len(bench_valid) + 1,
            "category": row["_cat"],
            "input": row["messages"][1]["content"],
            "ideal": row["messages"][2]["content"],
        })
    valid_ref_path = ROOT / "valid_categorized.json"
    with open(valid_ref_path, "w") as f:
        json.dump(bench_valid, f, indent=2, ensure_ascii=False)

    print(f"\nFinal split:")
    print(f"  Train: {len(train)} ({len(train)/len(all_examples)*100:.1f}%)")
    print(f"  Valid: {len(valid)} ({len(valid)/len(all_examples)*100:.1f}%)")
    print(f"  Test:  {len(test)} ({len(test)/len(all_examples)*100:.1f}%)")
    print(f"\nFiles written:")
    print(f"  {train_path}")
    print(f"  {valid_path}")
    print(f"  {test_path}")
    print(f"  {bench_path} (benchmark format)")
    print(f"  {valid_ref_path} (categorized reference)")

    # Category breakdown in test set
    print(f"\nTest set by category:")
    test_cats: dict[str, int] = {}
    for row in test:
        test_cats[row["_cat"]] = test_cats.get(row["_cat"], 0) + 1
    for cat in sorted(test_cats):
        print(f"  {cat}: {test_cats[cat]}")


if __name__ == "__main__":
    main()
