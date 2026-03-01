"""Merge v2 source data + Kimi-generated data into combined v3 source files.

v3/source/ has the filtered v2 data.
v3/kimi/ has the newly generated Kimi data.
This script merges them, deduplicates, and saves back to v3/source/.
"""

import json
from pathlib import Path

SOURCE_DIR = Path(__file__).parent / "source"
KIMI_DIR = Path(__file__).parent / "kimi"

CATEGORIES = [
    "spell-replace", "self-correction", "quote-unquote", "quote-endquote",
    "at-symbol", "caps", "emphasis", "emoji", "camelcase",
]


def is_duplicate(a: dict, b: dict, threshold: float = 0.85) -> bool:
    """Check word overlap between two examples."""
    a_words = set(a["input"].lower().split())
    b_words = set(b["input"].lower().split())
    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words) / min(len(a_words), len(b_words))
    return overlap > threshold


def main():
    print(f"\n{'=' * 60}")
    print(f"  Merging v2 source + Kimi data → v3 combined")
    print(f"{'=' * 60}\n")

    total = 0
    for cat in CATEGORIES:
        source_file = SOURCE_DIR / f"{cat}.json"
        kimi_file = KIMI_DIR / f"{cat}.json"

        # Load v2 source
        v2_data = []
        if source_file.exists():
            v2_data = json.loads(source_file.read_text())

        # Load Kimi new
        kimi_data = []
        if kimi_file.exists():
            kimi_data = json.loads(kimi_file.read_text())

        # Dedup Kimi against v2
        added = 0
        dupes = 0
        combined = list(v2_data)
        for item in kimi_data:
            if any(is_duplicate(item, existing) for existing in combined):
                dupes += 1
            else:
                combined.append(item)
                added += 1

        # Save combined
        source_file.write_text(json.dumps(combined, indent=2, ensure_ascii=False))

        kimi_str = f"+{added}" if added else "  0"
        dupe_str = f"({dupes} dupes)" if dupes else ""
        print(f"  {cat:20s}: {len(v2_data):3d} v2 {kimi_str:>4s} kimi = {len(combined):3d} total  {dupe_str}")
        total += len(combined)

    print(f"\n  {'TOTAL':20s}: {total:3d}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
