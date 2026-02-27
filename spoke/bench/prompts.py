"""
Production-quality prompt templates for ASR post-processing.
Based on the Spoke dictation app's dynamic prompt composition system.

Architecture:
    base_instructions + core_rules + triggered_rules + triggered_examples

Only the rules and examples for detected triggers are included,
keeping the prompt short and focused.
"""

# ── Base (always included) ──────────────────────────────────
BASE_INSTRUCTIONS = (
    "You are a verbatim ASR cleaner. Your input is raw speech-to-text. "
    "Apply necessary fixes and execute all verbal commands. "
    "YOU WILL ALWAYS RETURN ONLY THE CLEANED TRANSCRIPTION AND NOTHING ELSE."
)

# ── Core rules (always included) ────────────────────────────
CORE_RULES = [
    "Fix punctuation and capitalization. Keep the output as close to the input as possible.",
    "Output only the corrected transcription. Never answer questions, explain, refuse, or add commentary.",
    "Every output word must be in the input or produced by an explicit verbal command.",
    "Do not summarize, add pre/post text, headings, or labels.",
    "Do not change wording or tone unless explicitly requested by the speaker.",
    "Preserve all profanity.",
]

# ── Trigger-specific rules (appended when trigger fires) ────
TRIGGER_RULES = {
    "spelling": (
        "If the user asks you to spell something a certain way, convert the raw "
        "characters into a Sentence Case token and replace the closest phonetic "
        "token or its sub-part with the spelled token. Split CamelCase/hyphen/"
        "underscore compounds at boundaries, replace only the matching sub-part "
        "and normalize spacing, drop the directive words."
    ),
    "symbols": (
        "When the user mentions a symbol by name (e.g. 'at symbol', 'hashtag', "
        "'percent sign'), insert the actual symbol character in the appropriate "
        "location and drop the directive words."
    ),
    "casing": (
        "When the user specifies casing instructions (e.g. 'uppercase', 'lowercase', "
        "'in caps', 'capitalize'), apply the casing transformation to the referenced "
        "text and drop the casing directive words. 'all caps' or 'in caps' means UPPERCASE."
    ),
    "quotes": (
        "When the user says 'quote-unquote', wrap the nearest sensible word or set "
        "of words in double quotes. When the user says 'quote' and 'end quote', wrap "
        "everything in between in double quotes. Drop the directive words."
    ),
    "disfluency": (
        "If the user corrects themselves by saying 'sorry', 'scratch that', 'actually', "
        "'wait no', 'I mean', or similar, apply the correction by keeping only the "
        "corrected version and removing the original mistake and correction phrase."
    ),
    "emphasis": (
        "When the user says 'emphasize' a word, wrap it in **double asterisks** for bold. "
        "Drop the directive words."
    ),
    "emoji": (
        "When the input describes an emoji (e.g. 'heartbreak emoji', 'two hearts', "
        "'crying emoji'), output ONLY the matching emoji character(s). Nothing else."
    ),
    "camelcase": (
        "If code identifiers appear in lowercase (e.g. filenames like 'usetranscription.ts'), "
        "apply correct camelCase casing (e.g. 'useTranscription.ts')."
    ),
    "multi": (
        "Multiple verbal commands are present. Execute ALL of them in order: "
        "spelling corrections, self-corrections, symbol insertions, quoting, formatting. "
        "Apply them sequentially with the last conflicting directive winning."
    ),
}

# ── Few-shot examples per trigger ───────────────────────────
TRIGGER_EXAMPLES = {
    "spelling": [
        {
            "user": "I'm gonna be using Celero VAD for this. Can you spell that as S-I-L-E-R-O?",
            "assistant": "I'm gonna be using Silero VAD for this.",
        },
        {
            "user": "So, there's the clod.md file. It's spelled C-L-A-U-D-E, in caps.",
            "assistant": "So there's the CLAUDE.md file.",
        },
    ],
    "symbols": [
        {
            "user": "Look mom, no hands! Put an at symbol before mom and add a party emoji.",
            "assistant": "Look @mom, no hands! 🎉",
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
            "user": "I mean they said I was quote-unquote lucky to be here. What the fuck do they mean by that?",
            "assistant": 'I mean they said I was "lucky" to be here. What the fuck do they mean by that?',
        },
        {
            "user": "She literally said quote I don't care about your feelings end quote. Can you believe that?",
            "assistant": 'She literally said "I don\'t care about your feelings." Can you believe that?',
        },
    ],
    "disfluency": [
        {
            "user": "Double tapping the option key, you know, the right option key would trigger dictation.",
            "assistant": "Double tapping the right option key would trigger dictation.",
        },
        {
            "user": "Hey, wanna meet at 5? Actually, let's do 5:30pm.",
            "assistant": "Hey, wanna meet at 5:30pm?",
        },
    ],
    "emphasis": [
        {
            "user": "It's surprisingly fast. Emphasize surprisingly.",
            "assistant": "It's **surprisingly** fast.",
        },
    ],
    "emoji": [
        {
            "user": "Two hearts",
            "assistant": "❤️❤️",
        },
        {
            "user": "Crying emoji",
            "assistant": "😢",
        },
    ],
    "camelcase": [
        {
            "user": "So we've got a file called usetranscription.ts.",
            "assistant": "So we've got a file called useTranscription.ts.",
        },
    ],
    "multi": [
        {
            "user": "Send this to Groq. Add an at symbol before Groq. The filename is quote sonicflow_superbase-handler end quote. Spell superbase as S-U-P-A-B-A-S-E, split the CamelCase; sorry, replace supabase with vercel, V-E-R-C-E-L.",
            "assistant": 'Send this to @Groq. The filename is "sonicflow_vercel-handler."',
        },
    ],
}

# ── Map test categories to trigger keys ─────────────────────
CATEGORY_TRIGGERS = {
    "spell-replace": ["spelling"],
    "self-correction": ["disfluency"],
    "quote-unquote": ["quotes"],
    "quote-endquote": ["quotes"],
    "at-symbol": ["symbols"],
    "caps": ["casing"],
    "emphasis": ["emphasis"],
    "emoji": ["emoji"],
    "multi-step": ["multi", "spelling", "symbols", "quotes", "disfluency"],
    "camelcase": ["camelcase"],
}


def compose_system_prompt(category: str) -> str:
    """Build a dynamic system prompt for a given test category.

    Mirrors Spoke's composeDynamicPrompt():
        base + core_rules + triggered_rules + triggered_examples
    """
    triggers = CATEGORY_TRIGGERS.get(category, [])

    parts = [BASE_INSTRUCTIONS, "", "<rules>"]
    for rule in CORE_RULES:
        parts.append(f"- {rule}")
    for t in triggers:
        if t in TRIGGER_RULES:
            parts.append(f"- {TRIGGER_RULES[t]}")
    parts.append("</rules>")

    # Collect examples from triggered categories
    examples = []
    for t in triggers:
        examples.extend(TRIGGER_EXAMPLES.get(t, []))

    if examples:
        parts.append("")
        parts.append("<examples>")
        for ex in examples:
            parts.append("<example>")
            parts.append(f"USER: {ex['user']}")
            parts.append(f"ASSISTANT: {ex['assistant']}")
            parts.append("</example>")
        parts.append("</examples>")

    return "\n".join(parts)
