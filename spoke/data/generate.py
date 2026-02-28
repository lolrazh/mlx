"""Automated data generation pipeline for Spoke.

Usage:
    python spoke/data/generate.py <category> [--target 80] [--batch-size 10] [--max-fix-attempts 2]

Pipeline per batch:
    1. Call Kimi K2.5 with generation prompt + seeds
    2. Validate response with category-specific checks
    3. Auto-fix flagged examples (call API again with error descriptions)
    4. Re-validate fixes, drop persistent failures
    5. Accumulate passed examples until target reached
"""

import json
import os
import re
import sys
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from validate import validate, VALIDATORS

# ─── Config ─────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent.parent / ".env")

PROMPTS_DIR = Path(__file__).parent / "prompts"
RAW_DIR = Path(__file__).parent / "raw"
VALIDATED_DIR = Path(__file__).parent / "validated"
FINAL_DIR = Path(__file__).parent / "final"

for d in [RAW_DIR, VALIDATED_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_client() -> OpenAI:
    api_key = os.getenv("BASETEN_API_KEY")
    if not api_key or api_key == "your_key_here":
        print("Error: Set BASETEN_API_KEY in spoke/.env")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url="https://inference.baseten.co/v1")


# ─── API Call ───────────────────────────────────────────────────────

def call_kimi(client: OpenAI, prompt: str, max_tokens: int = 4000) -> str:
    """Call Kimi K2.5 and return the full response text."""
    response = client.chat.completions.create(
        model="moonshotai/Kimi-K2.5",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.9,  # some creativity for diversity
        top_p=0.95,
    )
    return response.choices[0].message.content


def parse_json_response(text: str) -> list[dict]:
    """Extract JSON array from LLM response, handling markdown wrapping."""
    if text is None:
        print("  Warning: API returned null response")
        return []
    # Try direct parse first
    text = text.strip()
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try extracting from markdown code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the array within the text
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    print(f"  Warning: Could not parse JSON from response")
    print(f"  Response preview: {text[:200]}...")
    return []


# ─── Prompts ────────────────────────────────────────────────────────

def load_gen_prompt(category: str) -> str:
    """Load the generation prompt for a category."""
    prompt_file = PROMPTS_DIR / f"{category}.md"
    if not prompt_file.exists():
        print(f"Error: No prompt found at {prompt_file}")
        print(f"Create it first, then run this script.")
        sys.exit(1)
    return prompt_file.read_text()


def build_gen_prompt(category: str, batch_num: int, existing: list[dict]) -> str:
    """Build generation prompt with diversity hints for subsequent batches."""
    base_prompt = load_gen_prompt(category)

    if batch_num == 0 and not existing:
        return base_prompt

    # For subsequent batches, add diversity instructions
    diversity_note = f"\n\nIMPORTANT: This is batch {batch_num + 1}. "
    diversity_note += f"I already have {len(existing)} examples. "
    diversity_note += "Generate examples that are DIFFERENT from what you've seen. "
    diversity_note += "Vary topics, sentence structures, and command phrasings. "
    diversity_note += "Try unusual or edge-case scenarios.\n"

    # Show a few existing examples to avoid duplicates
    if existing:
        samples = existing[-3:]  # show last 3 to avoid
        diversity_note += "\nExamples I ALREADY HAVE (don't repeat similar ones):\n"
        for s in samples:
            diversity_note += f'- Input: "{s["input"][:80]}..."\n'

    return base_prompt + diversity_note


def build_fix_prompt(category: str, flagged: list[dict]) -> str:
    """Build a fix prompt for flagged examples."""
    # Category-specific fix instructions
    fix_instructions = {
        "spell-replace": (
            "IMPORTANT CONTEXT: In these examples, ASR (speech-to-text) MISHEARS a word. "
            "The user then spells out the CORRECT version letter by letter. "
            "So the word in the input must be DIFFERENT from what the letters spell out.\n\n"
            "For SAME-WORD errors: change the word in the input to a plausible ASR misheard "
            "version (something that sounds similar but is spelled wrong).\n"
            "For 'still contains instruction' errors: remove all spelling meta-commentary "
            "from the ideal output — it should read as natural clean text."
        ),
        "self-correction": (
            "In these examples, the speaker corrects themselves mid-sentence. "
            "The ideal output should contain ONLY the corrected version, with the "
            "mistaken part and correction trigger ('sorry', 'wait no', etc.) removed.\n\n"
            "For 'no trigger' errors: add a natural self-correction trigger phrase.\n"
            "For 'ideal longer than input' errors: make sure the correction removes text, "
            "not adds it."
        ),
        "quote-unquote": (
            "In these examples, the speaker says 'quote-unquote' or 'quote...end quote' "
            "to indicate quoted text. The ideal output should have actual quotation marks "
            "around the quoted content, with the verbal triggers removed."
        ),
        "at-symbol": (
            "In these examples, the speaker says 'at symbol' or 'tag X with an at symbol' "
            "to insert @. The ideal should have @ inserted and the verbal instruction removed."
        ),
        "email": (
            "In these examples, the speaker dictates an email address verbally "
            "('dot com', 'at gmail'). The ideal should have a properly formatted email."
        ),
        "formatting": (
            "In these examples, the speaker requests formatting (caps, bold, emphasis, "
            "lowercase). The ideal should have the formatting applied and the instruction removed."
        ),
        "emoji": (
            "In these examples, the speaker names an emoji verbally. "
            "The ideal should contain the actual emoji character(s), not the word."
        ),
        "code-aware": (
            "In these examples, ASR mis-transcribes tech terms, filenames, or code identifiers. "
            "The ideal should have the correct technical formatting."
        ),
        "multi-command": (
            "These examples combine 2+ operations (spelling, correction, quoting, @, formatting, "
            "emoji). ALL operations must be executed correctly in the ideal output."
        ),
    }

    prompt = "These are training examples for an ASR post-processing model. Each has a specific error.\n\n"
    prompt += fix_instructions.get(category, "") + "\n\n"
    prompt += "Fix each example below. Change ONLY what's needed to fix the described error.\n\n---\n\n"

    for i, item in enumerate(flagged):
        prompt += f"Example {i+1}:\n"
        prompt += f'Input: "{item["input"]}"\n'
        prompt += f'Ideal: "{item["ideal"]}"\n'
        for err in item["errors"]:
            prompt += f"Error: {err}\n"
        prompt += "\n"

    prompt += "---\n\nOutput the fixed examples as a JSON array: "
    prompt += '[{"input": "...", "ideal": "..."}, ...]\n'

    return prompt


# ─── Dedup ──────────────────────────────────────────────────────────

def is_duplicate(new: dict, existing: list[dict], threshold: float = 0.8) -> bool:
    """Check if a new example is too similar to existing ones (simple word overlap)."""
    new_words = set(new["input"].lower().split())
    for ex in existing:
        ex_words = set(ex["input"].lower().split())
        if not new_words or not ex_words:
            continue
        overlap = len(new_words & ex_words) / min(len(new_words), len(ex_words))
        if overlap > threshold:
            return True
    return False


# ─── Main Pipeline ──────────────────────────────────────────────────

def run_pipeline(category: str, target: int, batch_size: int, max_fix_attempts: int):
    """Run the full generation pipeline for a category."""
    if category not in VALIDATORS:
        print(f"Unknown category '{category}'. Available: {list(VALIDATORS.keys())}")
        sys.exit(1)

    client = get_client()
    all_passed = []
    all_dropped = []
    batch_num = 0
    total_generated = 0
    total_api_calls = 0

    print(f"\n{'=' * 60}")
    print(f"  Spoke Data Generation: {category}")
    print(f"  Target: {target} examples, batch size: {batch_size}")
    print(f"{'=' * 60}\n")

    while len(all_passed) < target:
        remaining = target - len(all_passed)
        current_batch_size = min(batch_size, remaining + 5)  # generate a few extra to account for drops

        print(f"── Batch {batch_num + 1} ({len(all_passed)}/{target} collected) ──")

        # 1. Generate
        print(f"  Generating {current_batch_size} examples...")
        gen_prompt = build_gen_prompt(category, batch_num, all_passed)
        # Override the "Generate 10" in the base prompt with actual batch size
        gen_prompt = re.sub(
            r'Generate \d+ new pairs',
            f'Generate {current_batch_size} new pairs',
            gen_prompt
        )

        try:
            raw_response = call_kimi(client, gen_prompt)
            total_api_calls += 1
        except Exception as e:
            print(f"  API error: {e}")
            print(f"  Waiting 10s before retry...")
            time.sleep(10)
            continue

        batch = parse_json_response(raw_response)
        if not batch:
            print(f"  Failed to parse response, retrying...")
            continue

        total_generated += len(batch)
        print(f"  Got {len(batch)} examples")

        # 2. Dedup against existing
        deduped = []
        for item in batch:
            if not is_duplicate(item, all_passed):
                deduped.append(item)
        if len(deduped) < len(batch):
            print(f"  Removed {len(batch) - len(deduped)} duplicates")
        batch = deduped

        # 3. Validate
        result = validate(category, batch)
        passed = result["passed"]
        flagged = result["flagged"]
        print(f"  Validated: {len(passed)} passed, {len(flagged)} flagged")

        # 4. Fix flagged (up to max_fix_attempts)
        for attempt in range(max_fix_attempts):
            if not flagged:
                break

            print(f"  Fix attempt {attempt + 1}/{max_fix_attempts} for {len(flagged)} flagged...")
            fix_prompt = build_fix_prompt(category, flagged)

            try:
                fix_response = call_kimi(client, fix_prompt)
                total_api_calls += 1
            except Exception as e:
                print(f"  Fix API error: {e}")
                break

            fixed = parse_json_response(fix_response)
            if not fixed:
                print(f"  Failed to parse fix response")
                break

            # Re-validate fixed examples
            fix_result = validate(category, fixed)
            newly_passed = fix_result["passed"]
            still_flagged = fix_result["flagged"]
            print(f"  After fix: {len(newly_passed)} passed, {len(still_flagged)} still flagged")

            passed.extend(newly_passed)
            flagged = still_flagged

        # Drop persistent failures
        if flagged:
            print(f"  Dropping {len(flagged)} unfixable examples")
            all_dropped.extend(flagged)

        # 5. Accumulate (with final dedup check)
        for item in passed:
            clean = {"input": item["input"], "ideal": item["ideal"]}
            if not is_duplicate(clean, all_passed):
                all_passed.append(clean)

        print(f"  Running total: {len(all_passed)}/{target}\n")

        # Incremental save — don't lose progress on crash
        final_file = FINAL_DIR / f"{category}.json"
        final_file.write_text(json.dumps(all_passed[:target], indent=2, ensure_ascii=False))

        batch_num += 1

        # Safety: don't loop forever
        if batch_num > target // batch_size * 3:
            print(f"  Too many batches ({batch_num}), stopping.")
            break

        # Brief pause between batches to avoid rate limits
        if len(all_passed) < target:
            time.sleep(2)

    # ─── Save Results ───────────────────────────────────────────────

    final_file = FINAL_DIR / f"{category}.json"
    final_file.write_text(json.dumps(all_passed[:target], indent=2, ensure_ascii=False))

    if all_dropped:
        dropped_file = VALIDATED_DIR / f"{category}_dropped.json"
        dropped_file.write_text(json.dumps(all_dropped, indent=2, ensure_ascii=False))

    # Also save all raw batches for reference
    raw_file = RAW_DIR / f"{category}_all.json"
    raw_file.write_text(json.dumps(all_passed, indent=2, ensure_ascii=False))

    print(f"{'=' * 60}")
    print(f"  DONE: {category}")
    print(f"  Collected: {len(all_passed[:target])}/{target}")
    print(f"  Generated total: {total_generated}")
    print(f"  Dropped: {len(all_dropped)}")
    print(f"  API calls: {total_api_calls}")
    print(f"  Batches: {batch_num}")
    print(f"  Saved → {final_file}")
    print(f"{'=' * 60}\n")


# ─── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate training data for Spoke")
    parser.add_argument("category", help=f"Category: {', '.join(VALIDATORS.keys())}")
    parser.add_argument("--target", type=int, default=80, help="Target number of examples (default: 80)")
    parser.add_argument("--batch-size", type=int, default=10, help="Examples per API call (default: 10)")
    parser.add_argument("--max-fix-attempts", type=int, default=2, help="Max fix retries per batch (default: 2)")
    args = parser.parse_args()

    run_pipeline(args.category, args.target, args.batch_size, args.max_fix_attempts)


if __name__ == "__main__":
    main()
