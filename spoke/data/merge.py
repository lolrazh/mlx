"""
Merge all per-category JSON files into train/valid/test JSONL for mlx-lm.

Output:
  spoke/data/final/train.jsonl  — ~472 examples (shuffled, seed=42)
  spoke/data/final/valid.jsonl  —    8 examples (1 per generated category)
  spoke/data/final/test.jsonl   —   12 examples (sacred, from bench/test_set.json)

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

# ── System prompt (consistent across all training examples) ────
SYSTEM_PROMPT = (
    "Clean the transcript by executing all verbal commands "
    "(spell-outs, corrections, formatting, symbols, emoji). "
    "Output ONLY the cleaned text."
)

# ── Generated categories (in order) ────────────────────────────
CATEGORIES = [
    "spell-replace",
    "self-correction",
    "quote-unquote",
    "formatting",
    "email",
    "emoji",
    "code-aware",
    "multi-command",
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

    # ── 2. Build valid set: 1 example per category ─────────────
    # Use a fixed index in the middle of each file to get representative examples.
    # Index 10 is a safe "not the first, not the last" choice for all categories
    # (smallest category is emoji at 30 examples).
    VALID_INDEX = 10
    valid_examples: list[dict] = []
    valid_keys: set[tuple[str, str]] = set()

    for cat in CATEGORIES:
        ex = all_by_category[cat][VALID_INDEX]
        valid_examples.append(to_chat(ex["input"], ex["ideal"]))
        valid_keys.add((ex["input"], ex["ideal"]))

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
