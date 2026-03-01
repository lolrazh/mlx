#!/usr/bin/env python3
"""Generate additional training examples for weak v3 categories via Kimi K2.5.

Saves new examples to spoke/data/v3/kimi-boost/{category}.json,
then merges them into v3/source/ and re-runs merge.py.

Usage:
    python spoke/data/v3/boost.py
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Add parent dir so we can import validate
sys.path.insert(0, str(Path(__file__).parent.parent))
from validate import VALIDATORS

ROOT = Path(__file__).parent  # spoke/data/v3/
PROMPTS_DIR = ROOT.parent / "prompts"
BOOST_DIR = ROOT / "kimi-boost"
BOOST_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT.parent.parent / ".env")

# Categories to boost and how many new examples to generate
TARGETS = {
    "emphasis": 15,
    "caps": 15,
    "camelcase": 15,
    "at-symbol": 15,
    "emoji": 5,
}


def get_client() -> OpenAI:
    api_key = os.getenv("BASETEN_API_KEY")
    if not api_key:
        print("Error: Set BASETEN_API_KEY in spoke/.env")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url="https://inference.baseten.co/v1")


def call_kimi(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model="moonshotai/Kimi-K2.5",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.9,
        top_p=0.95,
    )
    return response.choices[0].message.content


def parse_json(text: str) -> list[dict]:
    if not text:
        return []
    text = text.strip()
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return []


def is_duplicate(new: dict, existing: list[dict], threshold: float = 0.8) -> bool:
    new_words = set(new["input"].lower().split())
    for ex in existing:
        ex_words = set(ex["input"].lower().split())
        if not new_words or not ex_words:
            continue
        overlap = len(new_words & ex_words) / min(len(new_words), len(ex_words))
        if overlap > threshold:
            return True
    return False


def generate_category(client: OpenAI, category: str, target: int):
    print(f"\n{'='*60}")
    print(f"Generating {target} new {category} examples")
    print(f"{'='*60}")

    # Load existing v3 source for dedup
    source_file = ROOT / "source" / f"{category}.json"
    existing = json.load(open(source_file)) if source_file.exists() else []
    print(f"  Existing: {len(existing)} examples in v3 source")

    # Load prompt
    prompt_file = PROMPTS_DIR / f"{category}.md"
    if not prompt_file.exists():
        print(f"  ERROR: No prompt at {prompt_file}")
        return []
    base_prompt = prompt_file.read_text()

    all_new = []
    batch = 0

    while len(all_new) < target:
        batch += 1
        remaining = target - len(all_new)
        print(f"\n  Batch {batch} (need {remaining} more)...")

        # Build prompt with diversity hints
        prompt = base_prompt
        if batch > 1 or existing:
            prompt += f"\n\nIMPORTANT: This is batch {batch}. "
            prompt += f"I already have {len(existing) + len(all_new)} examples. "
            prompt += "Generate examples that are DIFFERENT from what you've seen. "
            prompt += "Vary topics, sentence structures, and command phrasings.\n"

            # Show last few to avoid dupes
            recent = (existing + all_new)[-3:]
            prompt += "\nExamples I ALREADY HAVE (don't repeat similar ones):\n"
            for s in recent:
                prompt += f'- Input: "{s["input"][:80]}"\n'

        # Call API
        try:
            raw = call_kimi(client, prompt)
        except Exception as e:
            print(f"  API error: {e}")
            continue

        examples = parse_json(raw)
        if not examples:
            print(f"  Could not parse response")
            continue

        print(f"  Got {len(examples)} raw examples")

        # Validate + dedup
        passed = 0
        for ex in examples:
            if not isinstance(ex, dict) or "input" not in ex or "ideal" not in ex:
                continue

            # Validate using per-category validator
            validator = VALIDATORS.get(category)
            if validator:
                ok, errors = validator(ex)
                if not ok:
                    continue

            # Dedup against existing + already collected
            if is_duplicate(ex, existing + all_new):
                continue

            all_new.append(ex)
            passed += 1

            if len(all_new) >= target:
                break

        print(f"  Passed validation: {passed}")

    # Save to kimi-boost/
    out_file = BOOST_DIR / f"{category}.json"
    with open(out_file, "w") as f:
        json.dump(all_new, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(all_new)} new examples to {out_file}")

    return all_new


def main():
    client = get_client()
    total = 0

    for category, target in TARGETS.items():
        new = generate_category(client, category, target)
        total += len(new)

    print(f"\n{'='*60}")
    print(f"DONE: Generated {total} new examples across {len(TARGETS)} categories")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Review: check spoke/data/v3/kimi-boost/*.json")
    print(f"  2. Merge into source: update spoke/data/v3/source/*.json")
    print(f"  3. Re-run: python spoke/data/v3/merge.py")


if __name__ == "__main__":
    main()
