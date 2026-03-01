"""Split v2 data into v3 categories (trigger-matched only).

Reads per-category JSONs from spoke/data/final/.
Splits formatting → caps, emphasis, at-symbol (drops XML).
Filters code-aware → camelcase only.
Copies kept categories as-is.
Outputs per-category JSONs to spoke/data/v3/source/.
"""

import json
import re
from pathlib import Path

V2_DIR = Path(__file__).parent.parent / "final"
V3_DIR = Path(__file__).parent / "source"
V3_DIR.mkdir(parents=True, exist_ok=True)


def has_trigger(text: str, triggers: list[str]) -> bool:
    lower = text.lower()
    return any(t.lower() in lower for t in triggers)


# ─── Classify formatting.json ────────────────────────────────────

def classify_formatting(item: dict) -> str | None:
    """Classify a formatting example into: caps, emphasis, at-symbol, or None (XML/drop)."""
    inp = item["input"]
    ideal = item["ideal"]

    # XML detection — if ideal has XML tags, drop it
    if re.search(r'<[a-zA-Z]', ideal):
        return None  # XML → remove

    # At-symbol detection
    is_at = ("@" in ideal and
             has_trigger(inp, ["at symbol", "at sign", "put an at", "add an at",
                               "tag ", "with an at"]))

    # Emphasis detection (bold markers)
    is_bold = "**" in ideal

    # Caps/lowercase detection
    is_caps = has_trigger(inp, ["all caps", "uppercase", "lowercase", "lower case",
                                "in caps", "capitalize", "make it all"])

    # Stress/emphasis via CAPS (no ** but uppercase words used for emphasis)
    is_stress = has_trigger(inp, ["stress", "emphasis on", "emphasis"])

    # For combined: at-symbol is primary if present
    if is_at:
        return "at-symbol"
    if is_bold:
        return "emphasis"
    if is_stress:
        return "emphasis"
    if is_caps:
        return "caps"

    # Excitement-only (no other trigger) — goes to caps since it's punctuation transform
    if has_trigger(inp, ["show excitement", "make it excited"]):
        return "caps"

    print(f"  UNCLASSIFIED: {inp[:80]}...")
    return None


# ─── Classify code-aware.json ────────────────────────────────────

def is_camelcase_example(item: dict) -> bool:
    """Check if example involves camelCase/PascalCase restoration."""
    inp = item["input"]
    ideal = item["ideal"]

    # Look for camelCase pattern in ideal: lowercase followed by uppercase mid-word
    camel_pattern = r'[a-z][A-Z]'
    ideal_has_camel = bool(re.search(camel_pattern, ideal))

    # PascalCase: word starting with uppercase that was lowercase in input
    # Check for known PascalCase patterns
    pascal_words = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', ideal)

    # Check for React hooks (use* pattern)
    has_hooks = bool(re.search(r'\buse[A-Z]\w+', ideal))

    return ideal_has_camel or bool(pascal_words) or has_hooks


# ─── Main ─────────────────────────────────────────────────────────

def main():
    results = {}

    # 1. Direct copy categories
    for cat in ["spell-replace", "self-correction", "quote-unquote",
                "quote-endquote", "emoji"]:
        src = V2_DIR / f"{cat}.json"
        if src.exists():
            data = json.loads(src.read_text())
            dst = V3_DIR / f"{cat}.json"
            dst.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            results[cat] = len(data)
        else:
            print(f"  WARNING: {src} not found")
            results[cat] = 0

    # 2. Split formatting.json
    fmt_data = json.loads((V2_DIR / "formatting.json").read_text())
    buckets = {"caps": [], "emphasis": [], "at-symbol": []}
    dropped_xml = []

    for i, item in enumerate(fmt_data):
        cat = classify_formatting(item)
        if cat is None:
            dropped_xml.append(i)
        elif cat in buckets:
            buckets[cat].append(item)

    for cat, items in buckets.items():
        dst = V3_DIR / f"{cat}.json"
        dst.write_text(json.dumps(items, indent=2, ensure_ascii=False))
        results[cat] = len(items)

    print(f"\n  formatting.json split: caps={len(buckets['caps'])}, "
          f"emphasis={len(buckets['emphasis'])}, at-symbol={len(buckets['at-symbol'])}, "
          f"XML dropped={len(dropped_xml)}")
    print(f"  Dropped XML indices: {dropped_xml}")

    # 3. Filter code-aware.json → camelcase
    code_data = json.loads((V2_DIR / "code-aware.json").read_text())
    camelcase = []
    dropped_code = []

    for i, item in enumerate(code_data):
        if is_camelcase_example(item):
            camelcase.append(item)
        else:
            dropped_code.append(i)

    dst = V3_DIR / "camelcase.json"
    dst.write_text(json.dumps(camelcase, indent=2, ensure_ascii=False))
    results["camelcase"] = len(camelcase)
    print(f"\n  code-aware.json filtered: camelcase={len(camelcase)}, "
          f"dropped={len(dropped_code)}")
    print(f"  Dropped indices: {dropped_code}")

    # 4. Summary
    print(f"\n{'=' * 60}")
    print(f"  V3 Source Data Summary")
    print(f"{'=' * 60}")
    total = 0
    targets = {
        "spell-replace": 50, "self-correction": 70, "quote-unquote": 50,
        "quote-endquote": 65, "at-symbol": 30, "caps": 25,
        "emphasis": 20, "emoji": 45, "camelcase": 30,
    }
    for cat in ["spell-replace", "self-correction", "quote-unquote",
                "quote-endquote", "at-symbol", "caps", "emphasis",
                "emoji", "camelcase"]:
        count = results.get(cat, 0)
        target = targets.get(cat, "?")
        gap = max(0, target - count) if isinstance(target, int) else "?"
        total += count
        print(f"  {cat:20s}: {count:3d}  (target ~{target}, need ~{gap})")
    print(f"  {'─' * 40}")
    print(f"  {'TOTAL':20s}: {total:3d}  (target ~500)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
