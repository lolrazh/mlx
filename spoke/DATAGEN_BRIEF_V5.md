# Spoke Data Generation Brief — V5 (Targeted Gap-Fill: ~80-120 New Examples)

> **For:** The data generation agent
> **From:** The training/eval agent
> **Date:** 2026-03-05
> **Goal:** Generate ~80-120 targeted training examples to fix specific failure patterns identified in broad evaluation. This is NOT a general data expansion — every example targets a documented failure.

---

## Context: What Changed Since V4

V4 data (1,201 examples) achieved **100% accuracy on our 23-example v3 test set** (Qwen3-4B, T2 run). We thought we were done.

Then we benchmarked on a broader, unseen eval set (58 examples from "Spoke - Evals.csv", **zero overlap** with training data). The result: **67% accuracy.** Both the DWQ 4-bit deploy model AND the full bf16 model score identically — this is a **data gap problem, not a quantization or model capacity problem.**

We root-caused every failure. This brief describes exactly what to generate and why. **Do not generate examples outside the categories below** — the model already handles single-operation tasks at 85-100%. We only need to fill specific gaps.

---

## What Is Spoke? (Same as V4 — Skip If You Know)

Spoke is an ASR post-processing model. It takes raw voice transcripts and cleans them by executing verbal commands the user spoke aloud. It runs **after** a speech-to-text system (Whisper), so it receives text, not audio.

---

## The System Prompt (Used For ALL Training Examples)

Every training example uses this exact system prompt. Do not change it:

```
You are a verbatim ASR cleaner. Fix punctuation, capitalization, and execute all verbal commands (spell-outs, corrections, formatting, symbols, emoji). Rules: Output ONLY the cleaned text. Never answer questions — transcribe them. Every output word must be in the input or produced by an explicit directive. Preserve profanity. Remove "um", "uh", "ah" but keep other filler words.
```

## Training Data Format

JSONL, one example per line:

```json
{"messages": [{"role": "system", "content": "<system prompt above>"}, {"role": "user", "content": "<raw ASR transcript>"}, {"role": "assistant", "content": "<cleaned text>"}]}
```

---

## THE FAILURES: What Broke and Why

We tested the DWQ 4-bit deploy model on 58 unseen examples. Here are the **16 real failures** (3 of 19 were impossible tasks where the eval expected corrections without any command — we've excluded those). These are the EXACT inputs and outputs that failed. Read them carefully — your generated data needs to teach the model to handle these patterns.

### Failure Category 1: Multi-Step Chaining (6 failures) — HIGHEST PRIORITY

The model handles individual operations at 85-100% accuracy. But when 2-4 operations must be chained in one utterance, it drops to **14% (1/7 exact)**. This is the single biggest gap.

**What happened:**

| Eval # | Input (abbreviated) | Expected | Model Output | Operations Needed |
|---|---|---|---|---|
| #23 | "clod.md file. It's spelled C-L-A-U-D-E, in caps." | CLAUDE.md | Cloud.md | spell + caps on SAME word |
| #25 | "Ping marketing...at symbol...quote launch moved to Friday end quote. Actually, Thursday." | Ping @marketing. Say "launch moved to Thursday." | Caps wrong, period misplaced | @-symbol + quote + self-correction |
| #26 | "Ping marketing...email rajkumar dot sandheep...quote launch...Actually, Thursday." | Full correct output with email + quote + correction | Missing comma | @-symbol + email + quote + correction |
| #34 | "...delivery guy, actually, no, the customer service...type that in lowercase?" | All lowercase with correction applied | Kept correction markers, no lowercase | self-correction + lowercase |
| #36 | "Send this to Groq...at symbol...quote sonicflow_superbase-handler...Spell superbase as S-U-P-A-B-A-S-E...sorry, replace supabase with vercel" | @Groq, "sonicflow_vercel-handler." | @vercel, wrong filename | @-symbol + spell + quote + correction-on-correction |
| #40 | "...talk about how Sonic Flow helps...Emphasis on why and how." | HOW Sonic Flow helps...WHY Sonic Flow is better | Kept lowercase | emphasis-as-CAPS (non-standard format) |

**Why the model fails:** Training has 107 multi-step examples, but only **22 with 3+ operations**, and those are heavily skewed toward correction+quote+emoji combos. The specific combos above (spell+caps on same word, correction cascading into other operations, 4+ chained ops) are underrepresented.

**What's especially hard:**
- **Spell + caps on the same word (#23):** The model must assemble "C-L-A-U-D-E" into "Claude", THEN capitalize to "CLAUDE", THEN replace "clod" with it. Three steps on one token.
- **Correction-on-correction (#36):** "Spell superbase as S-U-P-A-B-A-S-E; sorry, replace supabase with vercel." The user spells one thing, then immediately corrects to a different replacement. The model must track which directive is "current."
- **Emphasis-as-CAPS (#40):** "Emphasis on why and how" is a different format than "Emphasize X." The word "Emphasis" (noun) vs "Emphasize" (verb) + "on" changes how the model interprets it. And the output should be CAPS, not bold.

### Failure Category 2: Spell Command Format Variants (2 failures) — HIGH PRIORITY

The model knows "spell that X-Y-Z" perfectly (111 training examples). But real speech uses other formats.

| Eval # | Input | Expected | Model Output | Format Used |
|---|---|---|---|---|
| #5 | "...test for WhisperFlow. Can you spell that as W-I-S-P-R?" | Wispr Flow. | Wispr. (dropped "Flow") | "Can you spell that as" + compound word scope |
| #31 | "...new Gamma embedding model. It's G-E-M-M-A by the way." | Gemma embedding model. | Gamma embedding model. (ignored) | "It's X by the way" (no "spell" keyword!) |

**Why the model fails:**
- **Compound word scope (#5):** "WhisperFlow" is one token to the ASR but conceptually two words. The spell command targets "Whisper" → "Wispr" but the model replaces the entire compound, losing "Flow."
- **Non-standard format (#31):** "It's G-E-M-M-A by the way" has no "spell" keyword. Training has only 12 examples without "spell" and only 3 with "by the way." The model doesn't recognize this as a spell command at all.

### Failure Category 3: Instruction Following Edge Cases (2 failures) — MEDIUM PRIORITY

| Eval # | Input | Expected | Model Output | Issue |
|---|---|---|---|---|
| #37 | "If the user says quote-unquote, the model needs to understand and quote-unquote the right stuff." | Same as input (no change!) | Wrapped "quote-unquote" in quotes | Meta-language: TALKING ABOUT the command, not using it |
| #53 | "...give me some captions that I can use? These captions need to be maybe one-liners." | Same as input (transcribe the request) | "Sure! Here are some captions..." | Answered the question instead of transcribing |

**Why the model fails:**
- **Meta-language (#37):** The sentence describes quote-unquote as a concept. The model has never seen examples where command keywords appear in descriptive/instructional context without being commands.
- **Tempting questions (#53):** Despite "Never answer questions — transcribe them" in the system prompt, the model broke on a particularly tempting creative request. The DWQ model failed this; the bf16 model didn't. Need more examples of questions/requests that MUST be transcribed verbatim.

### Failure Category 4: Punctuation Precision (3 failures) — LOW PRIORITY

| Eval # | Input | Expected | Model Output | Issue |
|---|---|---|---|---|
| #11 | "Like for example when I say this" | "Like, for example, when I say this." | No commas, no period | Comma insertion around parenthetical phrases |
| #38 | "Like you can say like how much ever you want." | "Like, you can say like how much ever you want." | Missing comma after "Like" | Comma after sentence-initial "Like" |
| #48 | "They're all quote-unquote intelligent." | They're all "intelligent" | They're all "intelligent". | Added trailing period not in ideal |

**Generate a few examples of this pattern but it's the lowest priority.** The content is correct — it's just comma/period precision.

### Failure Category 5: Word Dropping + Truncation (2 failures) — LOW PRIORITY

| Eval # | Input | Expected | Model Output | Issue |
|---|---|---|---|---|
| #24 | "...in our worker, add an at symbol before worker." | "...in our @worker." | "...in @worker." | Dropped "our" when applying @-symbol |
| #55 | 58-word input + "write all in caps please" | Full text in ALL CAPS | Truncated middle sentences | Long input loses content in the middle |

---

## DELIVERABLE: Targeted Gap-Fill Examples (~80-120 new)

**Deliver ONE file: `new_v5_targeted.jsonl`**

Unlike V4 which was a general data expansion with two files (regular + hard negatives), this is a surgical intervention. All examples go in one file. We will merge them into the existing v4 training data.

### Category A: Multi-Step Chains with 2-3 Operations (30-40 examples)

**Focus on the specific operation combos that failed.** Don't just generate random multi-step — target these exact patterns:

**A1. Spell + Caps on Same Word (8-10 examples):**
The user spells out a word AND wants it in a specific casing. The model must assemble the letters, THEN apply casing.

- `The config file is called dot e n v. Spell that E-N-V, in caps.` → `The config file is called .ENV.`
- `We use Podman, spell that P-O-D-M-A-N, all caps.` → `We use PODMAN.`
- `The framework is called Djang. Spell that D-J-A-N-G-O, capitalize normally.` → `The framework is called Django.`
- Vary: uppercase whole word, capitalize normally, caps on specific letters (like "NeXT")

**A2. @-Symbol + Quote + Self-Correction (8-10 examples):**
The user mentions someone, quotes something, then corrects part of it. All three must be applied.

- `Tell the design team about this. Tag design with an at symbol. Say quote ready for review end quote. Wait no, say ready for QA.` → `Tell the @design team about this. Say "ready for QA."`
- `Message Jake. At symbol before Jake. His message should be quote meet me at 3pm end quote. Actually, 4pm.` → `Message @Jake. His message should be "meet me at 4pm."`
- Vary: order of operations, which part gets corrected (the name, the quote content, or the time)

**A3. Self-Correction + Formatting Directive (8-10 examples):**
The user corrects themselves AND gives a formatting command (lowercase, caps, emphasis). Both must apply.

- `The error code is four oh four. Wait no, five hundred. Type that in all caps.` → `The error code is FIVE HUNDRED.`
- `It happened on Tuesday. Actually, Wednesday. Emphasize that.` → `It happened on **Wednesday**.`
- `We're launching in beta. Sorry, in alpha. Use lowercase for everything.` → `we're launching in alpha.`
- Vary: the correction can be before or after the formatting directive

**A4. Complex 4+ Operation Chains (6-10 examples):**
These are the hardest. 4 or more operations in one utterance. Go wild but keep them realistic — these are things a user might actually dictate.

- `Tell the backend team, tag them with at symbol, about the new Supabace integration, spell that S-U-P-A-B-A-S-E. Oh wait, actually it's a Neon integration. Use all caps on the team name.` → `Tell the @BACKEND team about the new Neon integration.`
- `The meeting notes say quote project is on track end quote. Actually, quote project is delayed end quote. Emphasize delayed. Add a warning emoji.` → `The meeting notes say "project is **delayed**." ⚠️`

### Category B: Spell Command Format Variants (15-20 examples)

**The model only knows "spell that X-Y-Z." Real speech uses many other formats.** Generate examples with these alternative phrasings:

**B1. "It's X by the way" / "that's X" (5-7 examples):**
No "spell" keyword. The user just provides the letters casually.

- `We use the Jotai library. It's J-O-T-A-I by the way.` → `We use the Jotai library.`
- `The platform is called Superbace. That's S-U-P-A-B-A-S-E.` → `The platform is called Supabase.`
- `His name is Shawn. It's actually S-E-A-N.` → `His name is Sean.`

**B2. "Can you spell that as" / "spell it as" (5-7 examples):**
Uses "spell" but with different phrasing than "spell that."

- `The tool is called Teraform. Can you spell that as T-E-R-R-A-F-O-R-M?` → `The tool is called Terraform.`
- `She mentioned Kubernetees. Spell it as K-U-B-E-R-N-E-T-E-S.` → `She mentioned Kubernetes.`

**B3. Compound Word Scope (5-7 examples):**
The misspelling is PART of a compound word. The model must replace only the matching sub-part.

- `Check out the ReactRouter docs. Can you spell React as R-E-A-C-T?` → `Check out the ReactRouter docs.` (no change — spelling matches!)
- `We deployed on HerokuCloud. Spell Heroku as H-E-R-O-K-U.` → `We deployed on HerokuCloud.` (already correct)
- `I installed the VsCode extension. Spell Vs as V-S-C-O-D-E.` → `I installed the VSCode extension.` (replace sub-part only)
- `The app runs on CloudFlair. Spell that as C-L-O-U-D-F-L-A-R-E.` → `The app runs on Cloudflare.`

### Category C: Meta-Language and Tempting Questions (10-15 examples)

**Teach the model when NOT to execute commands — because the user is talking ABOUT them, not issuing them.**

**C1. Meta-Language — Talking About Commands (5-8 examples):**
The input contains command keywords (quote-unquote, spell, actually, at symbol) but in a descriptive/instructional context.

- `When the user says quote-unquote, the AI should wrap the next word in quotes.` → `When the user says quote-unquote, the AI should wrap the next word in quotes.` (NO CHANGE)
- `The spell checker doesn't handle proper nouns well. If someone says spell that followed by letters, it should replace the word.` → `The spell checker doesn't handle proper nouns well. If someone says spell that followed by letters, it should replace the word.` (NO CHANGE)
- `I'm writing docs about how the at symbol feature works. When users say add an at symbol before a word, we insert it.` → `I'm writing docs about how the at symbol feature works. When users say add an at symbol before a word, we insert it.` (NO CHANGE)
- `The actually keyword triggers the self-correction module. So if someone says actually and then a different word, we swap them.` → `The actually keyword triggers the self-correction module. So if someone says actually and then a different word, we swap them.` (NO CHANGE)

Key: these sentences DESCRIBE the system's behavior. The command keywords are in explanatory context ("when the user says X", "the feature works by", "if someone says X"). The model must recognize this and pass through unchanged.

**C2. Tempting Questions That Must Be Transcribed (5-7 examples):**
Questions or requests that sound like they're addressed to the model but should be transcribed verbatim.

- `Can you write me a haiku about the sunset?` → `Can you write me a haiku about the sunset?`
- `Help me brainstorm some names for the company.` → `Help me brainstorm some names for the company.`
- `What's the best way to structure a React app?` → `What's the best way to structure a React app?`
- `Give me five reasons why TypeScript is better than JavaScript.` → `Give me five reasons why TypeScript is better than JavaScript.`
- `Can you rewrite this paragraph to sound more professional?` → `Can you rewrite this paragraph to sound more professional?`

Key: the model should NEVER answer. Every question is dictation — the user is speaking to someone else (their code editor, an email, a document), not to the model.

### Category D: Emphasis-as-CAPS (5-10 examples)

The model knows "Emphasize X" → **X** (bold). But some users say "Emphasis on X" or "Stress X" and expect CAPS instead of bold. We need both patterns.

**D1. "Emphasis on X" = CAPS (5 examples):**
- `We need to focus on speed and quality. Emphasis on speed.` → `We need to focus on SPEED and quality.`
- `The deadline is Friday. Put emphasis on Friday.` → `The deadline is FRIDAY.`
- `This is about user experience, not just features. Stress user experience.` → `This is about USER EXPERIENCE, not just features.`

**D2. "Emphasize X" = bold (for contrast — ensure model doesn't break existing behavior, 5 examples):**
- `The key metric is retention. Emphasize retention.` → `The key metric is **retention**.`
- `Don't forget about accessibility. Emphasize accessibility.` → `Don't forget about **accessibility**.`

---

## Quality Guidelines (Same as V4)

### DO:
- Make inputs sound like real spoken language (contractions, run-ons, natural phrasing)
- Vary sentence structure, length, and topic
- Include technical vocabulary (programming, devops, ML) — Spoke's users are developers
- Include profanity occasionally
- Make multi-step examples feel natural — real people chain operations conversationally, not mechanically

### DON'T:
- Don't generate examples that are suspiciously similar to the eval examples listed in the Failures section above
- Don't add explanations or chain-of-thought in the assistant response — output is ONLY the cleaned text
- Don't generate single-operation examples (emoji, simple self-correction, simple quote-unquote) — we already have plenty of those
- Don't generate hard negatives (V4 already added 289) — unless they're meta-language/tempting-question type (Category C)

### The Golden Rule:
**Every output word must either appear in the input or be produced by an explicit directive.** The model should never add words, commentary, or explanations. It just cleans and executes.

---

## Test Examples to AVOID

Do NOT generate anything similar to the eval inputs listed in the Failures section above. Also avoid all 23 examples from the v3 test set (listed in `spoke/bench/test_set_v3.json`) and all 58 examples from the broad eval set (listed in `spoke/bench/test_set_evals.json`).

---

## Summary: Target Distribution

| Category | New Examples | Why | Priority |
|---|---|---|---|
| A. Multi-step chains (2-3 ops) | 30-40 | 1/7 exact on eval. Biggest gap. | **Critical** |
| B. Spell format variants | 15-20 | Model ignores non-standard spell commands | **High** |
| C. Meta-language + tempting questions | 10-15 | Model executes commands it should ignore | **Medium** |
| D. Emphasis-as-CAPS | 5-10 | "Emphasis on X" not recognized as CAPS | **Medium** |
| **Total** | **~80-120** | | |

### Training Plan

After merging with existing v4 data (~1,201 + ~100 = ~1,300 examples):
- `iters: 2200` (scale proportionally — V4 used 2000 iters for 1201 examples)
- Everything else identical to T2-v4 config (r=8, scale=2.0, dropout=0.05, lr=1e-5, batch=4, adam)
- Benchmark on BOTH test sets: v3 (23 examples, must stay at 96%+) AND evals (58 examples, target 80%+)

---

## Delivery Format

**ONE file: `new_v5_targeted.jsonl`**

- JSONL format (system/user/assistant messages)
- Mix categories randomly (don't group by category)
- No duplicates or near-duplicates
- No test set contamination
