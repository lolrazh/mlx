"""Validate synthetic training data for Spoke.

Usage:
    python spoke/data/validate.py <category> [--file path/to/raw.json]

Each category has a validator that checks common failure modes.
Outputs a report of passed/flagged examples with error descriptions.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

# ─── Helpers ────────────────────────────────────────────────────────

def extract_spelled_letters(text: str) -> list[str]:
    """Find all X-X-X-X letter sequences in text. Returns list of assembled words."""
    # Match sequences like S-I-L-E-R-O (single chars separated by hyphens)
    pattern = r'\b([A-Za-z](?:-[A-Za-z]){2,})\b'
    matches = re.findall(pattern, text)
    return ["".join(m.split("-")) for m in matches]


def has_trigger(text: str, triggers: list[str]) -> bool:
    """Check if any trigger phrase appears in text (case-insensitive)."""
    lower = text.lower()
    return any(t.lower() in lower for t in triggers)


def count_pattern(text: str, pattern: str) -> int:
    """Count regex pattern matches in text."""
    return len(re.findall(pattern, text))


def has_emoji(text: str) -> bool:
    """Check if text contains emoji characters."""
    for ch in text:
        if unicodedata.category(ch) in ("So", "Sk") or ord(ch) > 0x1F000:
            return True
    return False


# ─── Validators ─────────────────────────────────────────────────────
# Each validator returns (passed: bool, errors: list[str])

def validate_spell_replace(pair: dict) -> tuple[bool, list[str]]:
    inp, ideal = pair["input"], pair["ideal"]
    errors = []

    # 1. Extract spelled letters from input
    spelled_words = extract_spelled_letters(inp)
    if not spelled_words:
        errors.append("No X-X-X letter pattern found in input")
        return False, errors

    # The FIRST spelled word is the primary correction target.
    # Additional spelled words may be contrasts ("not B-R-I-X like...") — skip those.
    primary_word = spelled_words[0]

    # 2. Check that the primary spelled word appears in ideal output
    if primary_word.lower() not in ideal.lower():
        errors.append(f"Spelled word '{primary_word}' not found in ideal output")

    # 3. Key check: the word being replaced should DIFFER from the spelled word
    # Strategy: check if the spelled word already appears verbatim in the input
    # (excluding the letter patterns themselves)
    spell_pattern = r'[A-Za-z](?:-[A-Za-z]){2,}'
    input_without_spelling = re.sub(spell_pattern, '___', inp)

    # Also strip common instruction phrases that may contain the correct word
    # e.g., "Can you spell Massaman M-A-S-S-A-M-A-N" — "Massaman" here is part
    # of the instruction, not the ASR-misheard word in the content.
    instruction_patterns = [
        r'(?:can you |could you )?spell (?:that |it |this )?(?:as )?(\w+)\s+___',
        r'(?:it\'?s |that\'?s )?spelled\s+___',
        r'spell\s+(\w+)\s+(?:as\s+)?___',
    ]
    input_content_only = input_without_spelling
    for pat in instruction_patterns:
        input_content_only = re.sub(pat, '___', input_content_only, flags=re.IGNORECASE)

    if re.search(r'\b' + re.escape(primary_word) + r'\b', input_content_only, re.IGNORECASE):
        errors.append(
            f"SAME-WORD: '{primary_word}' already appears correctly in input — "
            f"no ASR error to correct"
        )

    # 4. Check that the spelling instruction is removed from ideal
    if re.search(spell_pattern, ideal):
        errors.append("Ideal output still contains X-X-X letter pattern")

    spell_triggers = ["spell that", "spell it", "spell this", "spelled", "it's spelled",
                      "can you spell", "spell as"]
    if has_trigger(ideal, spell_triggers):
        errors.append("Ideal output still contains spelling instruction")

    return len(errors) == 0, errors


def validate_self_correction(pair: dict) -> tuple[bool, list[str]]:
    inp, ideal = pair["input"], pair["ideal"]
    errors = []

    # 1. Check for correction trigger words
    triggers = ["wait no", "sorry", "scratch that", "actually", "not ", "I meant",
                "I mean", "let me rephrase", "no no", "you know,"]
    if not has_trigger(inp, triggers):
        errors.append("No self-correction trigger found in input")

    # 2. Output should generally be shorter (correction removes text)
    # Allow some slack — sometimes the correction replaces with something longer
    if len(ideal) > len(inp):
        errors.append(f"Ideal ({len(ideal)} chars) is longer than input ({len(inp)} chars) — unusual for correction")

    # 3. Output should differ from input
    if inp.strip() == ideal.strip():
        errors.append("Input and ideal are identical — no correction happened")

    return len(errors) == 0, errors


def validate_quote_unquote(pair: dict) -> tuple[bool, list[str]]:
    inp, ideal = pair["input"], pair["ideal"]
    errors = []

    # 1. Input should have quote trigger
    triggers = ["quote-unquote", "quote unquote", "end quote", "in quotes"]
    if not has_trigger(inp, triggers):
        errors.append("No quote trigger found in input")

    # 2. Ideal should contain actual quotation marks
    quote_chars = ['"', '\u201c', '\u201d']  # straight and curly quotes
    if not any(q in ideal for q in quote_chars):
        errors.append("No quotation marks found in ideal output")

    # 3. Quote trigger should be removed from ideal
    if has_trigger(ideal, ["quote-unquote", "quote unquote"]):
        # Exception: meta-discussion about quote-unquote (passthrough case)
        errors.append("Ideal still contains 'quote-unquote' trigger")

    return len(errors) == 0, errors


def validate_at_symbol(pair: dict) -> tuple[bool, list[str]]:
    inp, ideal = pair["input"], pair["ideal"]
    errors = []

    # 1. Input should have @ trigger
    triggers = ["at symbol", "at sign", "tag ", "add an at"]
    if not has_trigger(inp, triggers):
        errors.append("No @-symbol trigger found in input")

    # 2. Ideal should contain @
    if "@" not in ideal:
        errors.append("No @ symbol found in ideal output")

    # 3. @ trigger instruction should be removed from ideal
    if has_trigger(ideal, ["at symbol", "at sign", "add an at"]):
        errors.append("Ideal still contains @-symbol instruction")

    return len(errors) == 0, errors


def validate_email(pair: dict) -> tuple[bool, list[str]]:
    inp, ideal = pair["input"], pair["ideal"]
    errors = []

    # 1. Ideal should contain an email address
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    if not re.search(email_pattern, ideal):
        errors.append("No valid email address found in ideal output")

    # 2. Input should have dictated email components
    triggers = ["dot com", "dot org", "dot net", "at gmail", "at yahoo",
                "at outlook", "dot co", "at sign"]
    if not has_trigger(inp, triggers) and "@" not in inp:
        errors.append("No dictated email components found in input")

    return len(errors) == 0, errors


def validate_formatting(pair: dict) -> tuple[bool, list[str]]:
    inp, ideal = pair["input"], pair["ideal"]
    errors = []

    # 1. Check for formatting trigger
    triggers = ["in caps", "all caps", "uppercase", "lowercase", "emphasize",
                "bold", "emphasis on", "in capital", "show excitement"]
    if not has_trigger(inp, triggers):
        errors.append("No formatting trigger found in input")

    # 2. Check that formatting was applied
    has_caps = any(c.isupper() for c in ideal) and ideal != ideal.lower()
    has_bold = "**" in ideal
    is_lowercase = ideal == ideal.lower()

    # Check specific formatting types
    if has_trigger(inp, ["all caps", "in caps", "uppercase", "in capital"]):
        # At least some portion should be uppercase in ideal
        upper_ratio = sum(1 for c in ideal if c.isupper()) / max(len(ideal), 1)
        if upper_ratio < 0.3:
            errors.append("Caps requested but ideal has low uppercase ratio")
    elif has_trigger(inp, ["lowercase"]):
        if not is_lowercase:
            errors.append("Lowercase requested but ideal contains uppercase")
    elif has_trigger(inp, ["emphasize", "bold", "emphasis"]):
        if "**" not in ideal and not any(w.isupper() for w in ideal.split()):
            errors.append("Emphasis requested but no bold markers or caps found")

    # 3. Formatting instruction should be removed from ideal
    if has_trigger(ideal, triggers):
        errors.append("Ideal still contains formatting instruction")

    return len(errors) == 0, errors


def validate_emoji(pair: dict) -> tuple[bool, list[str]]:
    inp, ideal = pair["input"], pair["ideal"]
    errors = []

    # 1. Input should mention emoji or emoji name
    triggers = ["emoji", "hearts", "heart", "crying", "fire", "thumbs",
                "smiley", "smile", "laugh", "sad", "happy", "angry", "love"]
    if not has_trigger(inp, triggers):
        errors.append("No emoji reference found in input")

    # 2. Ideal should contain actual emoji
    if not has_emoji(ideal):
        errors.append("No emoji found in ideal output")

    # 3. Input should NOT contain emoji (it's verbal)
    if has_emoji(inp):
        errors.append("Input already contains emoji — should be verbal description only")

    return len(errors) == 0, errors


def validate_code_aware(pair: dict) -> tuple[bool, list[str]]:
    inp, ideal = pair["input"], pair["ideal"]
    errors = []

    # Lightweight checks — this category mostly needs LLM judgment
    if inp.strip() == ideal.strip():
        errors.append("Input and ideal are identical — no transformation happened")

    return len(errors) == 0, errors


def validate_multi_command(pair: dict) -> tuple[bool, list[str]]:
    inp, ideal = pair["input"], pair["ideal"]
    errors = []

    # Multi-command should have at least 2 different operation types
    ops_found = 0
    if extract_spelled_letters(inp):
        ops_found += 1
    if has_trigger(inp, ["sorry", "wait no", "scratch that", "actually"]):
        ops_found += 1
    if has_trigger(inp, ["quote-unquote", "end quote"]):
        ops_found += 1
    if has_trigger(inp, ["at symbol", "at sign"]):
        ops_found += 1
    if has_trigger(inp, ["in caps", "lowercase", "emphasize", "bold"]):
        ops_found += 1
    if has_trigger(inp, ["emoji"]):
        ops_found += 1

    if ops_found < 2:
        errors.append(f"Only {ops_found} operation type(s) detected — multi-command needs 2+")

    if inp.strip() == ideal.strip():
        errors.append("Input and ideal are identical")

    return len(errors) == 0, errors


# ─── Registry ───────────────────────────────────────────────────────

VALIDATORS = {
    "spell-replace": validate_spell_replace,
    "self-correction": validate_self_correction,
    "quote-unquote": validate_quote_unquote,
    "at-symbol": validate_at_symbol,
    "email": validate_email,
    "formatting": validate_formatting,
    "emoji": validate_emoji,
    "code-aware": validate_code_aware,
    "multi-command": validate_multi_command,
}


# ─── Main ───────────────────────────────────────────────────────────

def validate(category: str, data: list[dict]) -> dict:
    """Run validation on a list of (input, ideal) pairs.

    Returns dict with 'passed', 'flagged', and 'summary'.
    """
    validator = VALIDATORS.get(category)
    if not validator:
        print(f"No validator for category '{category}'. Available: {list(VALIDATORS.keys())}")
        sys.exit(1)

    passed = []
    flagged = []

    for i, pair in enumerate(data):
        ok, errors = validator(pair)
        entry = {**pair, "index": i}
        if ok:
            passed.append(entry)
        else:
            entry["errors"] = errors
            flagged.append(entry)

    return {"passed": passed, "flagged": flagged}


def main():
    if len(sys.argv) < 2:
        print("Usage: python spoke/data/validate.py <category> [--file path.json]")
        print(f"Categories: {', '.join(VALIDATORS.keys())}")
        sys.exit(1)

    category = sys.argv[1]

    # Find input file
    file_path = None
    if "--file" in sys.argv:
        idx = sys.argv.index("--file")
        file_path = Path(sys.argv[idx + 1])
    else:
        # Default: spoke/data/raw/<category>.json
        file_path = Path(f"spoke/data/raw/{category}.json")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        print(f"Either place raw JSON at {file_path} or use --file <path>")
        sys.exit(1)

    data = json.loads(file_path.read_text())

    result = validate(category, data)
    n_pass = len(result["passed"])
    n_flag = len(result["flagged"])
    total = n_pass + n_flag

    # Print report
    print(f"\n{'=' * 60}")
    print(f"  Validation: {category}")
    print(f"  {n_pass}/{total} passed, {n_flag} flagged")
    print(f"{'=' * 60}\n")

    if result["flagged"]:
        for item in result["flagged"]:
            print(f"  #{item['index']+1} FLAGGED")
            print(f"     Input: {item['input'][:80]}...")
            print(f"     Ideal: {item['ideal'][:80]}...")
            for err in item["errors"]:
                print(f"     ⚠ {err}")
            print()

    if result["passed"]:
        print(f"  ✓ {n_pass} examples passed all checks\n")

    # Save results
    out_dir = Path("spoke/data/validated")
    out_dir.mkdir(parents=True, exist_ok=True)

    if result["passed"]:
        passed_file = out_dir / f"{category}_passed.json"
        passed_file.write_text(json.dumps(result["passed"], indent=2, ensure_ascii=False))
        print(f"  Saved passed → {passed_file}")

    if result["flagged"]:
        flagged_file = out_dir / f"{category}_flagged.json"
        flagged_file.write_text(json.dumps(result["flagged"], indent=2, ensure_ascii=False))
        print(f"  Saved flagged → {flagged_file}")

    print()


if __name__ == "__main__":
    main()
