"""
Merge all per-category JSON files into train/valid/test JSONL for mlx-lm.

Output:
  spoke/data/final/train.jsonl  — training examples (shuffled, seed=42)
  spoke/data/final/valid.jsonl  — ~16 examples (2 per category, stratified)
  spoke/data/final/test.jsonl   — 11 examples (sacred, from bench/test_set.json)

Each line:
  {"messages": [
    {"role": "system",    "content": SYSTEM_PROMPT},
    {"role": "user",      "content": "<raw transcript>"},
    {"role": "assistant", "content": "<cleaned output>"}
  ]}
"""

import json
import random
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).parent.parent.parent
FINAL_DIR  = Path(__file__).parent / "final"
TEST_SET   = REPO_ROOT / "spoke" / "bench" / "test_set.json"

# ── System prompt v2 (consistent across all training examples) ─
SYSTEM_PROMPT = (
    "You are a verbatim ASR cleaner. Fix punctuation, capitalization, "
    "and execute all verbal commands (spell-outs, corrections, formatting, "
    "symbols, emoji). Rules: Output ONLY the cleaned text. Never answer "
    "questions — transcribe them. Every output word must be in the input "
    "or produced by an explicit directive. Preserve profanity. "
    'Remove "um", "uh", "ah" but keep other filler words.'
)

# ── Generated categories (in order) ────────────────────────────
CATEGORIES = [
    "spell-replace",
    "self-correction",
    "quote-unquote",
    "quote-endquote",
    "formatting",
    "email",
    "emoji",
    "code-aware",
]


def to_chat(input_text: str, ideal_text: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": input_text},
            {"role": "assistant", "content": ideal_text},
        ]
    }


def write_jsonl(path: Path, examples: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(examples):>4} examples → {path.relative_to(REPO_ROOT)}")


def main() -> None:
    # ── 1. Load all generated categories ───────────────────────
    all_by_category: dict[str, list[dict]] = {}
    for cat in CATEGORIES:
        path = FINAL_DIR / f"{cat}.json"
        examples = json.loads(path.read_text(encoding="utf-8"))
        all_by_category[cat] = examples
        print(f"  Loaded {len(examples):>3} examples from {cat}.json")

    # ── 2. Build valid set: 2 examples per category ────────────
    # Pick indices 5 and 15 for most categories. For small categories
    # (quote-endquote=12), fall back to indices 3 and 8.
    VALID_INDICES = [5, 15]
    valid_examples: list[dict] = []
    valid_keys: set[tuple[str, str]] = set()

    for cat in CATEGORIES:
        cat_data = all_by_category[cat]
        for idx in VALID_INDICES:
            safe_idx = min(idx, len(cat_data) - 1)
            ex = cat_data[safe_idx]
            key = (ex["input"], ex["ideal"])
            if key not in valid_keys:  # avoid dupe if safe_idx clamped
                valid_examples.append(to_chat(ex["input"], ex["ideal"]))
                valid_keys.add(key)

    # ── 3. Build train set: everything except valid picks ──────
    train_examples: list[dict] = []
    for cat in CATEGORIES:
        for ex in all_by_category[cat]:
            key = (ex["input"], ex["ideal"])
            if key not in valid_keys:
                train_examples.append(to_chat(ex["input"], ex["ideal"]))

    # Shuffle for training (avoids category blocks in gradient updates)
    random.seed(42)
    random.shuffle(train_examples)

    # ── 4. Build test set from sacred bench examples ───────────
    test_raw = json.loads(TEST_SET.read_text(encoding="utf-8"))
    test_examples = [to_chat(ex["input"], ex["ideal"]) for ex in test_raw]

    # ── 5. Write all three files ────────────────────────────────
    print("\nWriting splits:")
    write_jsonl(FINAL_DIR / "train.jsonl", train_examples)
    write_jsonl(FINAL_DIR / "valid.jsonl", valid_examples)
    write_jsonl(FINAL_DIR / "test.jsonl",  test_examples)

    total = len(train_examples) + len(valid_examples) + len(test_examples)
    print(f"\nTotal: {total} examples  ({len(train_examples)} train + {len(valid_examples)} valid + {len(test_examples)} test)")
    print(f"System prompt: \"{SYSTEM_PROMPT}\"")


if __name__ == "__main__":
    main()
