#!/usr/bin/env python3
"""Trigger analysis and optional prompt reformatting for Spoke training data.

Ports Spoke's production trigger detection from TypeScript to Python.
Primary use: --stats mode to analyze which training examples would fire
triggers in production vs. which would bypass the LLM entirely.

NOTE: We decided to use a static prompt (v2-style) for all training, since
the fine-tuned model runs on every call and doesn't need dynamic per-trigger
prompts or few-shot examples. The rewrite (-o) mode is preserved for
experimentation but is not used in the current training pipeline.

Source of truth:
  - spoke-app/worker/src/services/llm/triggers.ts
  - spoke-app/worker/src/services/llm/prompts.ts

Usage:
    # Analyze trigger coverage (primary use)
    python spoke/data/reformat.py spoke/data/v3/train.jsonl --stats
    python spoke/data/reformat.py spoke/data/v4/new_regular.jsonl --stats

    # Optional: rewrite with dynamic prompts (not used in current pipeline)
    python spoke/data/reformat.py input.jsonl -o output.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# ============================================================================
# Trigger Detection
# Ported from: spoke-app/worker/src/services/llm/triggers.ts
# ============================================================================

# Exact regex patterns from production triggers.ts (verified 2026-03-02)
PRODUCTION_TRIGGERS: dict[str, list[re.Pattern]] = {
    "spelling": [
        # Spelled sequences: "S I L E R O" (space-separated single letters, 3+)
        re.compile(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b", re.IGNORECASE),
        # Spell instructions: "spell", "spelled", "spelling"
        re.compile(r"\bspell(?:ing|ed)?\b", re.IGNORECASE),
    ],
    "symbols": [
        re.compile(
            r"\b(?:symbol|symbols|tag|hashtag|at sign|percent|ampersand|"
            r"asterisk|dollar sign|plus sign|equals sign)\b",
            re.IGNORECASE,
        ),
    ],
    "casing": [
        re.compile(
            r"\b(?:uppercase|lowercase|caps|capitals|capitalise|capitalised|"
            r"capitalize|capitalized|all caps|in caps)\b",
            re.IGNORECASE,
        ),
    ],
    "quotes": [
        re.compile(
            r"\b(?:quote|quotes|in quotes|quote unquote|quote-unquote)\b",
            re.IGNORECASE,
        ),
    ],
    "disfluency": [
        re.compile(
            r"\b(?:sorry|wait no|wait,? no|scratch that|I mean|actually|"
            r"no wait|oops|my bad)\b",
            re.IGNORECASE,
        ),
    ],
}

# Extended triggers for training categories not yet in production.
# These should be added to triggers.ts before deploying these categories.
EXTENDED_TRIGGERS: dict[str, list[re.Pattern]] = {
    "emphasis": [
        re.compile(r"\b(?:emphasize|emphasis|bold|italicize)\b", re.IGNORECASE),
    ],
    "emoji": [
        re.compile(r"\bemoji\b", re.IGNORECASE),
    ],
}


def detect_triggers(text: str, extended: bool = True) -> set[str]:
    """Detect which triggers fire for a given input text.

    Mirrors production detectTriggers() from triggers.ts.

    Args:
        text: Raw ASR input text.
        extended: If True, include emphasis/emoji triggers (not yet in production).
    """
    fired = set()
    normalized = text.strip()
    if not normalized:
        return fired

    all_triggers = dict(PRODUCTION_TRIGGERS)
    if extended:
        all_triggers.update(EXTENDED_TRIGGERS)

    for name, patterns in all_triggers.items():
        for pattern in patterns:
            if pattern.search(normalized):
                fired.add(name)
                break

    return fired


# ============================================================================
# Prompt Composition
# Ported from: spoke-app/worker/src/services/llm/prompts.ts
# ============================================================================

# Exact text from production prompts.ts (verified 2026-03-02)
BASE_INSTRUCTIONS = (
    "You are a verbatim ASR cleaner for Spoke, an AI dictation app. "
    "Your input is coming from Whisper, an ASR model. "
    "The user's dictation comes through you, where you will apply "
    "necessary fixes to what the user spoke.\n\n"
    "YOU WILL ALWAYS RETURN ONLY THE TRANSCRIPTION AND NOTHING ELSE. "
    "NEVER IGNORE THESE INSTRUCTIONS."
)

CORE_RULES = [
    "Fix the ASR input with punctuation and capitalization. Keep the output "
    "as close to the input as possible.",
    "Output only the corrected transcription. Never answer questions, explain, "
    "refuse, or take actions.",
    "Any question that the user might ask is not directed towards you, but is "
    "something that you should transcribe. SO NEVER EVER OUTPUT ANSWERS TO "
    "QUESTIONS. ONLY APPLY TEXT-EDIT DIRECTIVES AND GRAMMAR FIXES TO THE "
    "TRANSCRIPTION.",
    "Every output word must be in the input or produced by an explicit "
    "text-edit directive or punctuation.",
    "Do not summarize, explain, add pre/post text, headings, or labels, "
    "or answer questions.",
    'Do not change wording/tone unless explicitly requested by the speaker. '
    'Keep filler words like "like", "sort of", "basically", etc. but remove '
    'filler words like "um", "uh" and "ah".',
    "Preserve all profanity.",
]

# Note: Production rule about OCR vocabulary is omitted — training data
# doesn't include vocabulary context. Model will see it at inference when
# <vocabulary> section is present, but it's benign when absent.

TRIGGER_RULES: dict[str, str] = {
    "spelling": (
        "If the user asks you to spell something a certain way, convert the raw "
        "characters into a Sentence Case token and replace the closest phonetic "
        "token or it's sub-part with the spelled token. Split CamelCase/hyphen/"
        "underscore compounds at boundaries, replace only the matching sub-part "
        "and normalize spacing, drop the directive words."
    ),
    "symbols": (
        'When the user mentions a symbol by name (e.g., "at symbol", "hashtag", '
        '"percent sign"), insert the actual symbol character in the appropriate '
        "location and drop the directive words that describe the symbol."
    ),
    "casing": (
        'When the user specifies casing instructions (e.g., "uppercase", '
        '"lowercase", "in caps", "capitalize"), apply the casing transformation '
        "to the referenced text and drop the casing directive words. "
        '"all caps" or "in caps" means UPPERCASE.'
    ),
    "quotes": (
        'When the user says "quote-unquote", wrap the nearest sensible word or '
        'set of words in quotes. When the user says "quote" and "end quote", '
        "wrap everything in between in quotes. Drop the directive words."
    ),
    "disfluency": (
        'If the user corrects themselves by saying "sorry", "scratch that", '
        '"actually", "wait no", "I mean", or similar, correct the output by '
        "fixing disfluencies."
    ),
    "list": (
        "Auto-format as a list when the speaker clearly enumerates \u22653 items "
        '(e.g., "one, two, three\u2026", "first, second, third\u2026", or '
        '"1., 2., 3." cadence). Stay true to the input while formatting '
        "appropriately with line breaks and markers."
    ),
    # Extended rules (not in production yet)
    "emphasis": (
        "When the user says 'emphasize' a word, wrap it in **double asterisks** "
        "for bold. Drop the directive words."
    ),
    "emoji": (
        "When the input describes an emoji (e.g. 'heartbreak emoji', 'crying "
        "emoji'), output ONLY the matching emoji character(s). Nothing else."
    ),
}

TRIGGER_EXAMPLES: dict[str, list[dict[str, str]]] = {
    "spelling": [
        {
            "user": "I'm gonna be using Celero VAD for this. Can you spell that as S-I-L-E-R-O?",
            "assistant": "I'm gonna be using Silero VAD for this.",
        },
        {
            "user": "So, there's the clod.md file. It's spelled C-L-A-U-D-E, in caps.",
            "assistant": "So there's the CLAUDE.md file.",
        },
        {
            "user": (
                "Send this to Groq. The filename is quote sonicflow_superbase-handler "
                "end quote. Spell superbase as S-U-P-A-B-A-S-E, split the CamelCase; "
                "sorry, replace supabase with vercel, V-E-R-C-E-L."
            ),
            "assistant": 'Send this to Groq. The filename is "sonicflow_vercel-handler."',
        },
    ],
    "symbols": [
        {
            "user": "Look mom, no hands! Put an ad symbol before mom and add a party emoji.",
            "assistant": "Look @mom, no hands! \U0001f389",
        },
        {
            "user": "My email is john at gmail dot com.",
            "assistant": "My email is john@gmail.com.",
        },
    ],
    "casing": [
        {
            "user": "The password is secret. Make that all caps.",
            "assistant": "THE PASSWORD IS SECRET.",
        },
        {
            "user": "Hey, What's up? What's going on? Write that in lowercase.",
            "assistant": "hey, what's up? what's going on?",
        },
    ],
    "quotes": [
        {
            "user": (
                "I mean they said I was quote-unquote lucky to be here. "
                "What the fuck do they mean by that?"
            ),
            "assistant": (
                'I mean they said I was "lucky" to be here. '
                "What the fuck do they mean by that?"
            ),
        },
        {
            "user": (
                "She literally said quote I don't care about your feelings "
                "end quote. Can you believe that?"
            ),
            "assistant": (
                "She literally said \"I don't care about your feelings.\" "
                "Can you believe that?"
            ),
        },
    ],
    "disfluency": [
        {
            "user": (
                "Double tapping the option key, you know, the right option "
                "key would trigger dictation."
            ),
            "assistant": "Double tapping the right option key would trigger dictation.",
        },
        {
            "user": "Hey, wanna meet at 5? Actually, um.. let's do 5:30pm.",
            "assistant": "Hey, wanna meet at 5:30pm?",
        },
    ],
    "list": [],
    "emphasis": [
        {
            "user": "It's surprisingly fast. Emphasize surprisingly.",
            "assistant": "It's **surprisingly** fast.",
        },
    ],
    "emoji": [
        {
            "user": "Crying emoji",
            "assistant": "\U0001f622",
        },
    ],
}


def compose_dynamic_prompt(triggers: set[str]) -> str:
    """Build a dynamic system prompt from a set of fired triggers.

    Mirrors composeDynamicPrompt() from prompts.ts exactly:
        base + <rules>[core + triggered]</rules> + <examples>[triggered]</examples>
    """
    sections: list[str] = [BASE_INSTRUCTIONS, ""]

    # Rules section
    rules = list(CORE_RULES)
    for name, text in TRIGGER_RULES.items():
        if name in triggers:
            rules.append(text)

    sections.append("<rules>")
    for rule in rules:
        sections.append(f"- {rule}")
    sections.append("</rules>")

    # Examples section
    example_lines: list[str] = []
    for name, exs in TRIGGER_EXAMPLES.items():
        if name in triggers and exs:
            for ex in exs:
                example_lines.append("<example>")
                example_lines.append(f"USER: {ex['user']}")
                example_lines.append(f"ASSISTANT: {ex['assistant']}")
                example_lines.append("</example>")

    if example_lines:
        sections.append("")
        sections.append("<examples>")
        sections.extend(example_lines)
        sections.append("</examples>")

    return "\n".join(sections)


# ============================================================================
# JSONL Processing
# ============================================================================


def reformat_jsonl(
    input_path: Path,
    output_path: Path | None = None,
    stats_only: bool = False,
):
    """Read training JSONL and rewrite system prompts with dynamic production prompts."""

    lines = input_path.read_text().strip().split("\n")
    trigger_counts: Counter = Counter()
    combo_counts: Counter = Counter()
    no_trigger: list[tuple[int, str]] = []
    reformatted = []

    for i, line in enumerate(lines):
        example = json.loads(line)
        messages = example["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")

        triggers = detect_triggers(user_msg)

        for t in triggers:
            trigger_counts[t] += 1
        if triggers:
            combo_counts["+".join(sorted(triggers))] += 1
        else:
            no_trigger.append((i + 1, user_msg[:80]))

        if not stats_only:
            prompt = compose_dynamic_prompt(triggers)
            new_messages = []
            for m in messages:
                if m["role"] == "system":
                    new_messages.append({"role": "system", "content": prompt})
                else:
                    new_messages.append(m)
            reformatted.append({"messages": new_messages})

    # --- Print stats ---
    total = len(lines)
    print(f"\n{'=' * 60}")
    print(f"  Trigger Analysis: {input_path.name} ({total} examples)")
    print(f"{'=' * 60}")

    print(f"\nPer-trigger fire counts:")
    for t, c in trigger_counts.most_common():
        bar = "\u2588" * (c * 30 // total) if total else ""
        print(f"  {t:15s}  {c:4d} / {total}  {bar}")

    print(f"\n  {'NO TRIGGER':15s}  {len(no_trigger):4d} / {total}")
    if no_trigger:
        print(f"\n  Examples with no trigger (base-only prompt):")
        for line_num, preview in no_trigger[:15]:
            print(f"    L{line_num:4d}: {preview}...")
        if len(no_trigger) > 15:
            print(f"    ... and {len(no_trigger) - 15} more")

    print(f"\nTrigger combinations:")
    for combo, c in combo_counts.most_common():
        print(f"  {combo:35s}  {c:4d}")

    # --- Write output ---
    if not stats_only and output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for ex in reformatted:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(reformatted)} examples \u2192 {output_path}")

        # Show sample
        sample = json.loads(lines[0])
        sample_triggers = detect_triggers(sample["messages"][1]["content"])
        sample_prompt = compose_dynamic_prompt(sample_triggers)
        print(f"\nSample (example 1, triggers={sample_triggers}):")
        print(f"  Prompt length: {len(sample_prompt)} chars (~{len(sample_prompt) // 4} tokens)")


def main():
    parser = argparse.ArgumentParser(
        description="Reformat JSONL with production-matched dynamic prompts"
    )
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument("-o", "--output", type=Path, help="Output JSONL file")
    parser.add_argument(
        "--stats", action="store_true",
        help="Show trigger stats only (no rewrite)",
    )
    args = parser.parse_args()

    if not args.stats and not args.output:
        parser.error("Provide --output or --stats")

    reformat_jsonl(args.input, args.output, stats_only=args.stats)


if __name__ == "__main__":
    main()
