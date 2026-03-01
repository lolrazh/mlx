#!/usr/bin/env python3
"""Build v3 valid set: filter dead categories from v2, add 6 new examples."""

import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a verbatim ASR cleaner. Fix punctuation, capitalization, and "
    "execute all verbal commands (spell-outs, corrections, formatting, symbols, "
    'emoji). Rules: Output ONLY the cleaned text. Never answer questions — '
    "transcribe them. Every output word must be in the input or produced by an "
    'explicit directive. Preserve profanity. Remove "um", "uh", "ah" but keep '
    "other filler words."
)

# Dead line indices (0-based) in v2 valid.jsonl:
# 12-13: email, 16-17: code-aware, 18-19: hard-negative
DEAD_LINES = {12, 13, 16, 17, 18, 19}

# Manual category labels for the 14 kept lines (for verification)
KEPT_CATEGORIES = [
    "spell-replace",    # 0: Phloem → Floum
    "spell-replace",    # 1: Kademlia → Kademliah
    "self-correction",  # 2: budget Q3
    "self-correction",  # 3: deployment 45→90 min
    "quote-unquote",    # 4: "friends"
    "quote-unquote",    # 5: "farm-to-table"
    "quote-endquote",   # 6: restructuring
    "quote-endquote",   # 7: running late
    "caps",             # 8: meeting 3pm
    "emphasis",         # 9: unexpected/shocking
    "at-symbol",        # 10: @engineering
    "at-symbol",        # 11: @app.py, @models.py, @config.py
    "emoji",            # 14: cat emoji
    "emoji",            # 15: 4 star emojis
]

v2_valid_path = Path(__file__).parent.parent / "final" / "valid.jsonl"
kept = []

with open(v2_valid_path) as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        if i in DEAD_LINES:
            continue
        kept.append(json.loads(line))

print(f"Kept {len(kept)} examples after filtering dead categories (expected 14)")

# New examples to add (categories that need filling)
new_examples = [
    # caps (+1 → total 2)
    {
        "input": "This is not a drill. Put that in all caps.",
        "ideal": "THIS IS NOT A DRILL.",
    },
    # emphasis (+1 → total 2)
    {
        "input": "We need to focus on reliability above all else. Emphasize reliability.",
        "ideal": "We need to focus on **reliability** above all else.",
    },
    # camelcase (+2 → total 2)
    {
        "input": "I'm refactoring the fetchuserdata function to handle pagination better.",
        "ideal": "I'm refactoring the fetchUserData function to handle pagination better.",
    },
    {
        "input": "The sidebar component is called projectnavigation and it uses a custom usesidebar hook.",
        "ideal": "The sidebar component is called ProjectNavigation and it uses a custom useSidebar hook.",
    },
    # quote-endquote (+1 → total 3)
    {
        "input": "The manual states quote all safety equipment must be worn at all times end quote.",
        "ideal": 'The manual states "all safety equipment must be worn at all times".',
    },
    # self-correction (+1 → total 3)
    {
        "input": "The API endpoint is slash users slash profile, wait no, slash users slash settings.",
        "ideal": "The API endpoint is /users/settings.",
    },
]

def make_chat(input_text: str, ideal_text: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": ideal_text},
        ]
    }

for ex in new_examples:
    kept.append(make_chat(ex["input"], ex["ideal"]))

NEW_CATEGORIES = ["caps", "emphasis", "camelcase", "camelcase", "quote-endquote", "self-correction"]
all_categories = KEPT_CATEGORIES + NEW_CATEGORIES

print(f"Total after adding {len(new_examples)} new: {len(kept)}")

# Write v3 valid set
out_path = Path(__file__).parent / "valid.jsonl"
with open(out_path, "w") as f:
    for ex in kept:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Wrote {out_path}")

# Category distribution
cats = {}
for cat in all_categories:
    cats[cat] = cats.get(cat, 0) + 1
print("Category distribution:")
for cat, count in sorted(cats.items()):
    print(f"  {cat}: {count}")
print(f"  TOTAL: {sum(cats.values())}")
