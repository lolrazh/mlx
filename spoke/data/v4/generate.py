#!/usr/bin/env python3
"""V4 data generation: ~400 regular + ~300 hard negatives via Kimi K2.5.

Generates raw JSON per sub-category into spoke/data/v4/raw/.
Resumable — re-run safely to continue from where you left off.

Usage:
    python spoke/data/v4/generate.py              # generate all
    python spoke/data/v4/generate.py spell-simple  # generate one sub-category
    python spoke/data/v4/generate.py --hard-neg    # generate hard negatives only
    python spoke/data/v4/generate.py --regular     # generate regular only
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).parent  # spoke/data/v4/
SPOKE = ROOT.parent.parent    # spoke/
RAW_DIR = ROOT / "raw"
RAW_DIR.mkdir(exist_ok=True)

load_dotenv(SPOKE / ".env")

# ──────────────────────────────────────────────────────────────────────────────
# Kimi K2.5 API
# ──────────────────────────────────────────────────────────────────────────────

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
        max_tokens=6000,
        temperature=0.9,
        top_p=0.95,
    )
    return response.choices[0].message.content


def parse_json(text: str) -> list[dict]:
    """Flexibly parse JSON array from LLM output."""
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


def is_duplicate(new: dict, existing: list[dict], threshold: float = 0.75) -> bool:
    new_words = set(new["input"].lower().split())
    for ex in existing:
        ex_words = set(ex["input"].lower().split())
        if not new_words or not ex_words:
            continue
        overlap = len(new_words & ex_words) / min(len(new_words), len(ex_words))
        if overlap > threshold:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Test set holdout (never generate similar examples)
# ──────────────────────────────────────────────────────────────────────────────

def load_holdout() -> set[str]:
    """Load test set inputs to avoid contamination."""
    holdout = set()
    test_path = SPOKE / "bench" / "test_set.json"
    if test_path.exists():
        with open(test_path) as f:
            for ex in json.load(f):
                holdout.add(ex["input"].lower().strip())
    # Also load v3 valid
    valid_path = ROOT.parent / "v3" / "valid.jsonl"
    if valid_path.exists():
        with open(valid_path) as f:
            for line in f:
                msg = json.loads(line.strip())
                holdout.add(msg["messages"][1]["content"].lower().strip())
    return holdout

HOLDOUT: set[str] = set()  # populated at runtime


# ──────────────────────────────────────────────────────────────────────────────
# PROMPTS: Regular examples
# ──────────────────────────────────────────────────────────────────────────────

HEADER = """You are generating training data for a small LLM that post-processes dictation transcripts. The model receives raw ASR output and must execute embedded verbal commands — NOT explain them, NOT chat about them — just output the cleaned text.

IMPORTANT RULES:
- Every output word must either appear in the input or be produced by an explicit directive
- Inputs should sound like real spoken language (contractions, run-ons, natural phrasing)
- Vary sentence length (short 5-word to long 25-word)
- Include tech topics AND everyday topics
- The assistant response is ONLY the cleaned text — no explanations, no commentary
- Do NOT capitalize the user input unless ASR would (ASR typically produces sentence case)

Output as JSON array: [{"input": "...", "ideal": "..."}, ...]
"""

PROMPTS = {

# ── SPELL-REPLACE ─────────────────────────────────────────────────────────

"spell-simple": HEADER + """
Task: Generate 15 SIMPLE SPELL-REPLACE examples.

The speaker says a word, then spells it out letter-by-letter. The spelled version MATCHES the original word (confirming spelling). The model must assemble the letters and remove the spelling instruction.

Examples:
- Input: "I'm using Supabase. Can you spell that S-U-P-A-B-A-S-E?"
  Output: "I'm using Supabase."

- Input: "The library is called Langchain, spell that L-A-N-G-C-H-A-I-N."
  Output: "The library is called Langchain."

Rules:
- The spelled word MATCHES what's already in the text (no replacement needed)
- Vary trigger phrases: "spell that", "can you spell that", "that's spelled", "spell it as", "spelled"
- Vary position: beginning, middle, end of sentence
- Use real words ASR would transcribe correctly: tech terms, common nouns, names
- Word length: 4-10 letters
- Remove the spelling instruction from output, keep everything else
""",

"spell-corrective": HEADER + """
Task: Generate 15 CORRECTIVE SPELL-REPLACE examples.

The speaker says a word that ASR misheard, then spells the CORRECT version. The model must assemble the letters, find the misheard word, and replace it.

Examples:
- Input: "We deployed to Versell. Spell that V-E-R-C-E-L."
  Output: "We deployed to Vercel."

- Input: "I'm gonna be using Celero VAD for this. Can you spell that as S-I-L-E-R-O?"
  Output: "I'm gonna be using Silero VAD for this."

- Input: "The new framework is called Nux. Spell that N-U-X-T."
  Output: "The new framework is called Nuxt."

Rules:
- The ASR word should be a PLAUSIBLE mishearing (phonetically similar to the correct word)
- GOOD ASR errors: "Versell" → Vercel, "Celero" → Silero, "Wispor" → Wispr
- BAD ASR errors: totally unrelated words, or words that are already correct
- Focus on: startup names, niche libraries, non-English proper nouns, brand names
- The spelled version is DIFFERENT from what's in the text (actual correction)
- Vary trigger phrases and positions
""",

"spell-compound": HEADER + """
Task: Generate 15 COMPOUND SPELL-REPLACE examples.

The speaker spells a word AND does another operation in the same sentence. This tests the model's ability to chain operations.

Combinations to use:
1. Spell + self-correction: "I'm using Wispor, spell that W-I-S-P-R. Wait no, I said the wrong library, it's Deepgram."
2. Spell + emphasis: "Check out Superbace, spell that S-U-P-A-B-A-S-E. Emphasize Supabase."
3. Spell + at-symbol: "Message Devops about Surealdb, spell that S-U-R-R-E-A-L-D-B. Tag devops with an at symbol."
4. Spell + caps: "The status is urgant, spell that U-R-G-E-N-T. All caps."
5. Spell + emoji: "We shipped Wisper Flow, spell Wisper as W-I-S-P-R. Fire emoji."

Examples:
- Input: "Hey this is a transcription test for Whisper Flow. Can you spell that W-I-S-P-R? And emphasize transcription."
  Output: "Hey this is a **transcription** test for Wispr Flow."

- Input: "Deploy Koobernetees to production. Spell that K-U-B-E-R-N-E-T-E-S. Tag devops with an at symbol."
  Output: "Deploy Kubernetes to production. @devops"

Rules:
- Each example must have spell-replace PLUS at least one other operation
- Both operations must be correctly executed in the output
- The spelling should be a CORRECTIVE replacement (ASR misheard the word)
- Make ASR errors phonetically plausible
- Vary the second operation across the set
""",

# ── COMPOUND OPERATIONS ───────────────────────────────────────────────────

"compound-selfcorr": HEADER + """
Task: Generate 15 COMPOUND examples combining SELF-CORRECTION with another operation.

The speaker corrects themselves AND does something else (emoji, emphasis, caps, quote, at-symbol).

Examples:
- Input: "The meeting is at 3pm. Wait no, 4pm. And emphasize the time."
  Output: "The meeting is at **4pm**."

- Input: "Send it to the dev channel. Actually, the staging channel. Tag it with an at symbol."
  Output: "Send it to the @staging channel."

- Input: "We need to ship today. Sorry, tomorrow. Fire emoji."
  Output: "We need to ship tomorrow. 🔥"

- Input: "The status is pending. Wait no, blocked. Make it all caps."
  Output: "The status is BLOCKED."

Rules:
- Correction trigger + one other operation per example
- Vary the second operation: emphasis, at-symbol, caps, emoji, quote-unquote
- Correction must be clearly scoped (what exactly is being corrected)
- Both operations must be correctly applied in output
- Sound like natural speech
""",

"compound-quote": HEADER + """
Task: Generate 15 COMPOUND examples combining QUOTES with another operation.

The speaker uses quote-unquote or quote...end quote AND does something else.

Examples:
- Input: "He called it quote-unquote revolutionary. And that word should be in all caps."
  Output: "He called it \"REVOLUTIONARY\"."

- Input: "She said quote I'm done end quote. Use the fire emoji after that."
  Output: "She said \"I'm done.\" 🔥"

- Input: "Their quote-unquote solution broke the build. Emphasize solution."
  Output: "Their \"**solution**\" broke the build."

- Input: "The response was quote we'll look into it end quote. Tag support with an at symbol."
  Output: "The response was \"we'll look into it.\" @support"

Rules:
- Every example has a quote operation PLUS one other operation
- Use both "quote-unquote X" (sarcastic, wraps 1-2 words) and "quote ... end quote" (direct speech, wraps phrases)
- Second operations: caps, emphasis, emoji, at-symbol, self-correction
- Quote scope must be unambiguous
- The "end quote" marker must clearly close the quoted text
""",

"compound-3plus": HEADER + """
Task: Generate 15 COMPOUND examples requiring 3 OR MORE operations in a single input.

These are the hardest examples. The speaker chains multiple verbal commands naturally.

Examples:
- Input: "The function is called getserverprops. It's quote-unquote stable now. Emphasize stable. And add the fire emoji."
  Output: "The function is called getServerProps. It's \"**stable**\" now. 🔥"

- Input: "Tell the dev team we're shipping Kubernetees, spell that K-U-B-E-R-N-E-T-E-S. Tag dev with an at symbol. Use all caps on shipping."
  Output: "Tell the @dev team we're SHIPPING Kubernetes."

- Input: "The status is pending. Wait no, approved. Make it bold. And add the party emoji."
  Output: "The status is **approved**. 🎉"

- Input: "She said quote we need to pivot end quote. Actually, quote we need to adapt end quote. Emphasize adapt."
  Output: "She said \"we need to **adapt**.\""

Rules:
- MINIMUM 3 distinct operations per example
- Operations: spell-replace, self-correction, quote, emphasis, caps, emoji, at-symbol, camelCase
- All operations must be correctly applied in output
- Sound natural — people do chain commands in real speech
- Make some genuinely hard (4 operations)
- Don't make every example about tech — include everyday topics
""",

# ── SELF-CORRECTION ───────────────────────────────────────────────────────

"selfcorr-partial": HEADER + """
Task: Generate 15 PARTIAL LIST CORRECTION examples.

The speaker lists items, then corrects ONLY THE LAST ITEM. The rest of the list stays intact.

Examples:
- Input: "We support Python, Java, and Rust. Actually wait, not Rust, Go."
  Output: "We support Python, Java, and Go."

- Input: "The stack is React, Express, and MongoDB. Sorry, PostgreSQL not MongoDB."
  Output: "The stack is React, Express, and PostgreSQL."

- Input: "Invite Alice, Bob, and Charlie. Wait, not Charlie. David."
  Output: "Invite Alice, Bob, and David."

- Input: "We need milk, eggs, and butter. Actually, cheese instead of butter."
  Output: "We need milk, eggs, and cheese."

CRITICAL RULES:
- The correction ONLY replaces the LAST item in the list
- All other items remain EXACTLY as they were
- This is the #1 failure mode — models either keep everything or delete too much
- Vary list length: 2-item pairs AND 3+ item lists
- Vary triggers: "actually", "wait", "sorry", "scratch that", "no wait", "I mean"
- Vary domains: tech stacks, groceries, people names, cities, tools, foods
""",

"selfcorr-mid": HEADER + """
Task: Generate 15 MID-SENTENCE CORRECTION examples.

The speaker corrects a specific detail in the MIDDLE of a sentence, not at the end.

Examples:
- Input: "The endpoint is slash API slash users. Wait no, slash API slash accounts."
  Output: "The endpoint is /API/accounts."

- Input: "It costs twenty dollars. No wait, twenty-five dollars."
  Output: "It costs twenty-five dollars."

- Input: "The release is on March 3rd. Actually, March 10th."
  Output: "The release is on March 10th."

- Input: "She started in 2019. Wait, 2020, and has been here for six years."
  Output: "She started in 2020 and has been here for six years."

Rules:
- The correction targets a specific value (date, number, name, URL path)
- Surrounding context is preserved on BOTH sides of the correction
- Vary what gets corrected: numbers, dates, names, paths, versions, prices
- The correction should be clearly scoped to ONE detail
""",

"selfcorr-ambiguous": HEADER + """
Task: Generate 15 AMBIGUOUS self-correction examples — a mix of REAL corrections and NON-corrections.

Some use trigger words ("actually", "I mean") as corrections. Others use them in normal speech (NOT corrections). The model must learn the difference.

REAL CORRECTIONS (output changes):
- Input: "We're moving to AWS. Sorry, I mean we're EVALUATING AWS."
  Output: "We're evaluating AWS."

- Input: "The budget is fifty thousand. Actually, that was last quarter. This quarter it's forty."
  Output: "The budget is forty thousand."

NON-CORRECTIONS (output = input, minor punctuation only):
- Input: "I love TypeScript and Python equally. Well, actually, maybe TypeScript a bit more."
  Output: "I love TypeScript and Python equally. Well, actually, maybe TypeScript a bit more."

- Input: "The project uses React. Actually, it's built on Next.js, which uses React."
  Output: "The project uses React. Actually, it's built on Next.js, which uses React."

- Input: "I mean, it's not the worst thing in the world but it's definitely not ideal."
  Output: "I mean, it's not the worst thing in the world but it's definitely not ideal."

Rules:
- Generate roughly HALF real corrections and HALF non-corrections
- Non-corrections use trigger words as: qualifiers, clarifications, filler, opinions
- The model must learn: "actually" as new information ≠ "actually" as replacement
- This is CRITICAL training data — these examples teach boundary discrimination
- Label each with "correction": true/false in the JSON so reviewers can verify
""",

# ── CAPS ──────────────────────────────────────────────────────────────────

"caps": HEADER + """
Task: Generate 15 CAPS/UPPERCASE examples.

The speaker wants text in ALL CAPS. Commands vary from whole-sentence to single-word.

Examples:
- Input: "This is urgent. All caps."
  Output: "THIS IS URGENT."

- Input: "Warning, system overload. All caps on warning."
  Output: "WARNING, system overload."

- Input: "Do not merge that branch. Make it all uppercase."
  Output: "DO NOT MERGE THAT BRANCH."

- Input: "The label should say fragile. Capitalize that."
  Output: "The label should say FRAGILE."

Rules:
- Mix WHOLE SENTENCE caps and SINGLE WORD/PHRASE caps
- Vary the command: "all caps", "make it uppercase", "capitalize", "in caps", "use all caps"
- When command targets a specific word, ONLY that word gets capitalized
- When command targets the whole sentence, everything gets capitalized
- Include tech topics, warnings, labels, headers, and everyday usage
""",

# ── EMPHASIS ──────────────────────────────────────────────────────────────

"emphasis": HEADER + """
Task: Generate 15 EMPHASIS examples.

The speaker wants a word or phrase emphasized (bold). Emphasis in our system uses **bold** markdown.

Examples:
- Input: "We need this done today. Bold today."
  Output: "We need this done **today**."

- Input: "The key insight is that latency matters more than throughput. Emphasize latency and throughput."
  Output: "The key insight is that **latency** matters more than **throughput**."

- Input: "Do not push to main. Emphasize not."
  Output: "Do **not** push to main."

- Input: "It's not just fast, it's blazingly fast. Emphasize blazingly fast."
  Output: "It's not just fast, it's **blazingly fast**."

Rules:
- Emphasis = **bold** (double asterisks)
- Commands: "emphasize X", "bold X", "stress X", "make X bold"
- Include single-word AND multi-word emphasis targets
- Include examples emphasizing 2+ different words in one sentence
- The emphasized word must appear verbatim in the input
""",

# ── EMOJI ─────────────────────────────────────────────────────────────────

"emoji": HEADER + """
Task: Generate 15 EMOJI examples.

The speaker verbally names an emoji. The model inserts the emoji character and REMOVES the word "emoji" if present.

Examples:
- Input: "That was incredible rocket emoji"
  Output: "That was incredible 🚀"

- Input: "Great news about the launch fire emoji absolutely crushing it"
  Output: "Great news about the launch 🔥 absolutely crushing it"

- Input: "Saluting face emoji"
  Output: "🫡"

- Input: "The bug is fixed party popper emoji and sparkles emoji"
  Output: "The bug is fixed 🎉 and ✨"

CRITICAL RULES:
- The word "emoji" must be STRIPPED from the output — only the emoji character remains
- Cover diverse emoji: 🔥 💀 🚀 👀 💪 🙌 😍 ✨ 🎂 👍 🎉 🫡 🤌 😭 🤔 💯
- Mix standalone ("skull emoji" → 💀) and inline ("great job fire emoji love it" → "great job 🔥 love it")
- Include multi-emoji: "fire and rocket emoji" → "🔥 and 🚀"
- Include less common emoji: chef's kiss 🤌, saluting face 🫡, melting face 🫠
- Vary where the emoji appears: start, middle, end of sentence
""",

# ── DISFLUENCY ────────────────────────────────────────────────────────────

"disfluency": HEADER + """
Task: Generate 15 DISFLUENCY REMOVAL examples.

The system prompt says: "Remove um, uh, ah but keep other filler words."

The speaker uses verbal fillers (um, uh, ah) that should be removed. Other filler words like "you know", "like", "so" are KEPT.

Examples:
- Input: "So um I was thinking we could uh maybe refactor the auth module."
  Output: "So I was thinking we could maybe refactor the auth module."

- Input: "The um the deployment failed because ah the config was wrong."
  Output: "The deployment failed because the config was wrong."

- Input: "I uh I think we should um probably just revert the commit."
  Output: "I think we should probably just revert the commit."

- Input: "It's uh actually a really good framework, um, once you get used to it."
  Output: "It's actually a really good framework, once you get used to it."

Rules:
- ONLY remove: "um", "uh", "ah" (and their variants: "umm", "uhh", "ahh")
- KEEP: "like", "you know", "so", "well", "I mean", "basically", "actually", "right"
- Fix any awkward spacing/punctuation left after removal
- Sometimes "um" creates duplicate words ("the um the" → "the"), remove the duplicate
- Vary placement: beginning, middle, multiple occurrences
- Include examples with 2-3 fillers in one sentence
- Mix tech and everyday topics
""",

}

# ──────────────────────────────────────────────────────────────────────────────
# PROMPTS: Hard negatives
# ──────────────────────────────────────────────────────────────────────────────

HN_HEADER = """You are generating HARD NEGATIVE training data for a small LLM that post-processes dictation transcripts.

Hard negatives are sentences where a TRIGGER KEYWORD appears in NORMAL speech — NOT as a verbal command. The correct output is the input with ONLY punctuation/capitalization fixes. NO semantic changes.

The model must learn: sometimes trigger words appear naturally and the right answer is to LEAVE THE TEXT ALONE.

CRITICAL: output must be IDENTICAL to input (or nearly — only minor punctuation/capitalization fixes allowed). Never delete, rearrange, or add words.

Output as JSON array: [{"input": "...", "ideal": "..."}, ...]
"""

PROMPTS_HN = {

"hn-disfluency": HN_HEADER + """
Task: Generate 20 hard negatives where DISFLUENCY trigger words ("actually", "sorry", "wait", "I mean", "scratch that", "no no", "my bad", "oops") appear in NORMAL speech, NOT as corrections.

Examples:
- Input: "Actually, I think the architecture is really clean."
  Output: "Actually, I think the architecture is really clean."

- Input: "Sorry about the delay, I was in a meeting."
  Output: "Sorry about the delay, I was in a meeting."

- Input: "Wait for the tests to pass before deploying."
  Output: "Wait for the tests to pass before deploying."

- Input: "I mean, it's not perfect but it ships."
  Output: "I mean, it's not perfect but it ships."

- Input: "Actually it turns out the bug was in the config all along."
  Output: "Actually it turns out the bug was in the config all along."

- Input: "I'm sorry but I don't think we should merge this yet."
  Output: "I'm sorry but I don't think we should merge this yet."

Key variations:
- "actually" as sentence starter ("Actually, I agree"), mid-sentence ("I actually think..."), emphasis ("It's actually quite good")
- "sorry" as apology ("Sorry to bother you"), politeness ("Sorry about that"), empathy ("I'm sorry to hear that")
- "wait" as instruction ("Wait for the CI"), anticipation ("Wait until you see this"), exclamation ("Wait, that's amazing!")
- "I mean" as filler ("I mean, it's fine"), clarification ("I mean the backend, not the frontend")
- "no" as disagreement ("No, I don't think so"), negative ("No worries at all")
- "my bad" as mild acknowledgment ("My bad, I should have caught that")
- "scratch that" used literally ("Can you scratch that itch on my back?") — this is rare but important
""",

"hn-quote": HN_HEADER + """
Task: Generate 20 hard negatives where "QUOTE" trigger words appear in NORMAL speech, NOT as verbal quotation commands.

The word "quote" has many non-command meanings: price quotes, literary quotes, quoting someone, quotation marks discussion, etc.

Examples:
- Input: "Can you send me a quote for the enterprise plan?"
  Output: "Can you send me a quote for the enterprise plan?"

- Input: "That's a direct quote from the documentation."
  Output: "That's a direct quote from the documentation."

- Input: "The quote on the landing page needs updating."
  Output: "The quote on the landing page needs updating."

- Input: "He quotes that paper in every single meeting."
  Output: "He quotes that paper in every single meeting."

- Input: "I'll get you a price quote by end of day."
  Output: "I'll get you a price quote by end of day."

- Input: "The insurance quote came back higher than expected."
  Output: "The insurance quote came back higher than expected."

Variations:
- Price/business quotes ("Get me a quote", "the quote was too high")
- Literary/attribution quotes ("a famous quote", "to quote Einstein")
- Quotation marks discussion ("use curly quotes", "the quotes are mismatched")
- "Unquote" in normal speech is rare — but "quoted" and "quoting" are common
""",

"hn-symbols": HN_HEADER + """
Task: Generate 20 hard negatives where SYMBOL trigger words ("tag", "symbol", "hashtag", "at sign", "percent") appear in NORMAL speech.

Examples:
- Input: "The HTML tag needs a class attribute."
  Output: "The HTML tag needs a class attribute."

- Input: "Add a price tag to each item in the store."
  Output: "Add a price tag to each item in the store."

- Input: "The dollar symbol is used in jQuery selectors."
  Output: "The dollar symbol is used in jQuery selectors."

- Input: "We need to tag this release before deploying."
  Output: "We need to tag this release before deploying."

- Input: "The hashtag trend is dying down."
  Output: "The hashtag trend is dying down."

- Input: "About 15 percent of users reported the issue."
  Output: "About 15 percent of users reported the issue."

Variations:
- "tag" as HTML tag, git tag, price tag, name tag, dog tag, luggage tag
- "symbol" as currency symbol, stock ticker symbol, math symbol, debug symbol
- "hashtag" as social media term, discussion about hashtags
- "at" in normal preposition use ("at the office", "at 3pm")
- "percent" as a number ("50 percent off", "the test passed 99 percent of the time")
""",

"hn-casing": HN_HEADER + """
Task: Generate 15 hard negatives where CASING trigger words ("caps", "capitalize", "uppercase", "lowercase") appear in NORMAL speech, NOT as formatting commands.

Examples:
- Input: "Check if caps lock is on, that might be the issue."
  Output: "Check if caps lock is on, that might be the issue."

- Input: "Don't capitalize every word in the title."
  Output: "Don't capitalize every word in the title."

- Input: "The uppercase version of the string is cached."
  Output: "The uppercase version of the string is cached."

- Input: "We need to convert the input to lowercase before comparing."
  Output: "We need to convert the input to lowercase before comparing."

- Input: "CSS text-transform can handle capitalize and uppercase."
  Output: "CSS text-transform can handle capitalize and uppercase."

Variations:
- "caps" as caps lock key, baseball caps, bottle caps
- "capitalize" as discussing text processing, not commanding it
- "uppercase/lowercase" in programming context (string methods, comparisons)
- Discussing formatting rules rather than executing them
""",

"hn-spelling": HN_HEADER + """
Task: Generate 15 hard negatives where SPELLING trigger words ("spell", "spelled", "spelling") appear in NORMAL speech, NOT as spell-out commands.

Examples:
- Input: "I can never spell bureaucracy correctly."
  Output: "I can never spell bureaucracy correctly."

- Input: "Can you spell that out for me? I didn't catch it."
  Output: "Can you spell that out for me? I didn't catch it."

- Input: "The spelling of that word is unusual."
  Output: "The spelling of that word is unusual."

- Input: "How do you spell your last name?"
  Output: "How do you spell your last name?"

- Input: "Check your spelling before submitting the PR."
  Output: "Check your spelling before submitting the PR."

- Input: "Spell check caught three errors in the README."
  Output: "Spell check caught three errors in the README."

Variations:
- "spell" as question ("How do you spell...?"), inability ("I can't spell...")
- "spelled" past tense ("It's spelled differently in British English")
- "spelling" as noun ("The spelling is wrong", "a spelling bee")
- "spell check" as tool name
- General literacy discussion, not commanding a spell-out
""",

}

# ──────────────────────────────────────────────────────────────────────────────
# Generation engine
# ──────────────────────────────────────────────────────────────────────────────

# Target counts per sub-category
TARGETS_REGULAR = {
    "spell-simple": 35,
    "spell-corrective": 45,
    "spell-compound": 35,
    "compound-selfcorr": 25,
    "compound-quote": 25,
    "compound-3plus": 30,
    "selfcorr-partial": 25,
    "selfcorr-mid": 25,
    "selfcorr-ambiguous": 25,
    "caps": 30,
    "emphasis": 30,
    "emoji": 40,
    "disfluency": 30,
}

TARGETS_HN = {
    "hn-disfluency": 110,
    "hn-quote": 55,
    "hn-symbols": 45,
    "hn-casing": 35,
    "hn-spelling": 35,
}


def load_existing(subcat: str) -> list[dict]:
    """Load previously generated examples for this sub-category (resume support)."""
    path = RAW_DIR / f"{subcat}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_progress(subcat: str, examples: list[dict]):
    """Save current progress."""
    path = RAW_DIR / f"{subcat}.json"
    with open(path, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)


def generate_subcategory(client: OpenAI, subcat: str, target: int, prompt_template: str):
    """Generate examples for one sub-category."""
    existing = load_existing(subcat)
    if len(existing) >= target:
        print(f"  {subcat}: already have {len(existing)}/{target} — skipping")
        return existing

    print(f"\n{'='*60}")
    print(f"  {subcat}: {len(existing)}/{target} (need {target - len(existing)} more)")
    print(f"{'='*60}")

    # Also load v3 source data for dedup (avoid generating duplicates of existing training data)
    v3_pool = []
    v3_source = ROOT.parent / "v3" / "source"
    if v3_source.exists():
        for f in v3_source.glob("*.json"):
            with open(f) as fh:
                v3_pool.extend(json.load(fh))

    all_examples = list(existing)
    batch = 0
    max_batches = 15  # safety limit
    consecutive_failures = 0

    while len(all_examples) < target and batch < max_batches:
        batch += 1
        remaining = target - len(all_examples)
        print(f"  Batch {batch} (have {len(all_examples)}, need {remaining} more)...")

        # Build prompt with diversity hints
        prompt = prompt_template
        if batch > 1 or all_examples:
            prompt += f"\n\nIMPORTANT: This is batch {batch}. "
            prompt += f"I already have {len(all_examples)} examples. "
            prompt += "Generate examples that are VERY DIFFERENT from what you've seen. "
            prompt += "Use different topics, sentence structures, and command phrasings.\n"

            # Show recent to avoid dupes
            recent = all_examples[-5:]
            prompt += "\nExamples I ALREADY HAVE (don't repeat similar ones):\n"
            for s in recent:
                inp = s.get("input", "")[:100]
                prompt += f'- Input: "{inp}"\n'

        # Call API
        try:
            raw = call_kimi(client, prompt)
        except Exception as e:
            print(f"    API error: {e}")
            consecutive_failures += 1
            if consecutive_failures >= 3:
                print("    Too many consecutive failures, moving on")
                break
            time.sleep(2)
            continue

        consecutive_failures = 0
        examples = parse_json(raw)
        if not examples:
            print(f"    Could not parse response")
            continue

        print(f"    Got {len(examples)} raw examples")

        # Validate + dedup
        passed = 0
        for ex in examples:
            if not isinstance(ex, dict) or "input" not in ex or "ideal" not in ex:
                continue

            # Check holdout contamination
            if ex["input"].lower().strip() in HOLDOUT:
                continue

            # Dedup against all sources
            if is_duplicate(ex, all_examples + v3_pool):
                continue

            all_examples.append(ex)
            passed += 1

            if len(all_examples) >= target:
                break

        print(f"    Passed: {passed}")

        # Save progress after each batch
        save_progress(subcat, all_examples)
        time.sleep(0.5)  # Rate limit courtesy

    print(f"  {subcat}: DONE — {len(all_examples)} examples")
    save_progress(subcat, all_examples)
    return all_examples


def main():
    global HOLDOUT
    HOLDOUT = load_holdout()
    print(f"Loaded {len(HOLDOUT)} holdout examples (test + valid)")

    client = get_client()

    # Parse args
    args = sys.argv[1:]
    run_regular = True
    run_hn = True
    specific = None

    if "--regular" in args:
        run_hn = False
        args.remove("--regular")
    elif "--hard-neg" in args:
        run_regular = False
        args.remove("--hard-neg")

    if args:
        specific = args[0]

    # Generate regular examples
    if run_regular:
        print(f"\n{'#'*60}")
        print(f"# REGULAR EXAMPLES (target: {sum(TARGETS_REGULAR.values())})")
        print(f"{'#'*60}")

        for subcat, target in TARGETS_REGULAR.items():
            if specific and subcat != specific:
                continue
            prompt = PROMPTS.get(subcat)
            if not prompt:
                print(f"  WARNING: No prompt for {subcat}")
                continue
            generate_subcategory(client, subcat, target, prompt)

    # Generate hard negatives
    if run_hn:
        print(f"\n{'#'*60}")
        print(f"# HARD NEGATIVES (target: {sum(TARGETS_HN.values())})")
        print(f"{'#'*60}")

        for subcat, target in TARGETS_HN.items():
            if specific and subcat != specific:
                continue
            prompt = PROMPTS_HN.get(subcat)
            if not prompt:
                print(f"  WARNING: No prompt for {subcat}")
                continue
            generate_subcategory(client, subcat, target, prompt)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    total_reg = 0
    for subcat in TARGETS_REGULAR:
        data = load_existing(subcat)
        target = TARGETS_REGULAR[subcat]
        status = "✓" if len(data) >= target else "…"
        print(f"  {status} {subcat}: {len(data)}/{target}")
        total_reg += len(data)

    total_hn = 0
    for subcat in TARGETS_HN:
        data = load_existing(subcat)
        target = TARGETS_HN[subcat]
        status = "✓" if len(data) >= target else "…"
        print(f"  {status} {subcat}: {len(data)}/{target}")
        total_hn += len(data)

    print(f"\n  Regular: {total_reg}/{sum(TARGETS_REGULAR.values())}")
    print(f"  Hard neg: {total_hn}/{sum(TARGETS_HN.values())}")
    print(f"  Total: {total_reg + total_hn}")
    print(f"\nNext: python spoke/data/v4/build.py")


if __name__ == "__main__":
    main()
