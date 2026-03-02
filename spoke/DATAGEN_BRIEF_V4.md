# Spoke Data Generation Brief — V4 (Target: 1,200-1,300 Training Examples)

> **For:** The data generation agent
> **From:** The training/eval agent
> **Date:** 2026-03-02
> **Goal:** Generate ~700-800 new training examples (two deliverables: ~400-500 regular + ~300 hard negatives)

---

## What Is Spoke?

Spoke is an ASR (speech-to-text) post-processing model. It takes a raw voice transcript and cleans it up by executing verbal commands the user spoke aloud. The model runs **after** a speech-to-text system (like Whisper), so it receives text, not audio.

**Example:**
- Input: `I'm gonna be using Celero VAD for this. Can you spell that as S-I-L-E-R-O?`
- Output: `I'm gonna be using Silero VAD for this.`

The user spoke the spelling command out loud. The ASR transcribed it literally. Spoke's job is to **execute** that command — assemble the letters, replace the original word, and remove the instruction.

### How Spoke Gets Triggered (IMPORTANT — Read This)

In production, Spoke does NOT run on every transcript. A **trigger system** scans the raw ASR output for keyword patterns. If none fire, the transcript passes through untouched (no LLM call).

The triggers are **keyword-based regex**, NOT structural-only:

| Trigger | Regex Fires On | Example False Positive |
|---|---|---|
| `spelling` | "spell", "spelled", "spelling" + space-separated letters | "I can never spell bureaucracy" |
| `symbols` | "symbol", "tag", "hashtag", "at sign", "percent" | "The HTML tag needs a class" |
| `casing` | "uppercase", "lowercase", "caps", "capitalize" | "Check if caps lock is on" |
| `quotes` | "quote", "quotes", "quote-unquote" | "Send me a quote for the plan" |
| `disfluency` | "sorry", "wait no", "actually", "I mean", "scratch that" | "Actually, I think it looks great" |
| `list` | Sequential numbers/ordinals (≥3 items) | Less false-positive prone |

**This means the model WILL see sentences where trigger words appear in normal speech.** "Actually, I think the design is solid" fires the disfluency trigger, the LLM runs, and the model must learn to output the text unchanged. This is why **hard negatives are critical** — they teach the model when NOT to edit.

---

## The System Prompt (Used For ALL Training Examples)

Every training example uses this exact system prompt. Do not change it:

```
You are a verbatim ASR cleaner. Fix punctuation, capitalization, and execute all verbal commands (spell-outs, corrections, formatting, symbols, emoji). Rules: Output ONLY the cleaned text. Never answer questions — transcribe them. Every output word must be in the input or produced by an explicit directive. Preserve profanity. Remove "um", "uh", "ah" but keep other filler words.
```

---

## Training Data Format

JSONL, one example per line:

```json
{"messages": [{"role": "system", "content": "<system prompt above>"}, {"role": "user", "content": "<raw ASR transcript>"}, {"role": "assistant", "content": "<cleaned text>"}]}
```

---

## Current Dataset (V3): What We Have

**535 training examples**, distributed across these categories:

| Category | Count | % | Notes |
|---|---|---|---|
| camelCase (implicit) | ~150 | 28% | Code identifiers lowered by ASR → restore casing |
| quote (all types) | ~117 | 22% | quote-unquote, quote...end quote |
| self-correction | ~70 | 13% | "Wait no, sorry, X" → apply correction |
| emoji | ~48 | 9% | "heart emoji" → emoji character |
| at-symbol | ~47 | 9% | "add an at symbol before X" → @X |
| other/multi | ~59 | 11% | Mixed categories, multi-step |
| caps | ~17 | 3% | "Use all caps" → ALL CAPS |
| spell-replace | ~15 | 3% | "spell that X-Y-Z" → assemble + replace |
| emphasis | ~12 | 2% | "Emphasize X" → **X** |

### Critical Imbalance

**spell-replace has only 15 examples (3%) but is one of the hardest categories.** The model struggles most with spelling operations, especially compound ones. This needs 4-5x more examples.

**caps and emphasis are also thin** (17 and 12 respectively). These are simpler operations but still need more diversity.

**camelCase dominates** (~28% of all training data). This is the easiest category (just fix casing on code identifiers). It's overrepresented.

---

## Current Results: Where Models Struggle

We've trained 4 different models (Qwen3 4B, Llama 3.2 3B, Gemma 3 4B, LFM2 2.6B). They all converge to **87-91% accuracy** on the 23-example test set. Here's where they fail:

### Failures on V3 Test Set (trained categories only)

All 4 models score 87-91%. The remaining 9-13% errors are:

**1. Self-correction #6 — partial replacement ambiguity**
- Input: `We're using React and Vue. Wait no, sorry, Svelte.`
- Expected: `We're using React and Svelte.`
- Most models output: `We're using React and Vue. Wait no, sorry, Svelte.` (no edit) or `We're using Svelte.` (over-delete)
- **Why it's hard:** "Wait no, sorry" corrects only "Vue" → "Svelte", but keeps "React." The model must understand that only the LAST item in a list is being corrected, not the whole sentence. Only Llama gets this right.

**2. Quote scope — where to place the closing quote**
- Input: `I mean they said I was quote-unquote lucky to be here.`
- Expected: `I mean they said I was "lucky" to be here.` (quotes around just "lucky")
- Most models: `I mean they said I was "lucky to be here".` (quotes around the whole phrase)
- **Why it's hard:** "quote-unquote" typically wraps the immediately following word/phrase, but the boundary is ambiguous. Even humans might disagree on this one.

**3. Quote truncation — dropping content inside quotes**
- Input: `The debug log shows quote connection timed out after 30 seconds end quote`
- Expected: `..."connection timed out after 30 seconds"...`
- Some models: `..."connection timed out"...` (drops "after 30 seconds")
- **Why it's hard:** Long quoted content gets truncated. The model "forgets" the middle of the quote.

### Failures on V2 Cross-Test (includes untrained categories)

When tested on the v2 test set (which includes categories NOT in training), all models drop to **65-74%**. The extra failures are:

**4. formatting-xml (#15, #16) — UNIVERSAL FAILURE**
- Every model fails these. "Wrap in a result tag" → should produce `<result>...</result>`. No model generates XML tags.
- This category was intentionally removed from v3 training. NOT a priority for new data.

**5. email (#17) — UNIVERSAL FAILURE**
- Input: `plus one, literally the word plus and the number one, at sign temp hyphen mail dot o r g`
- Expected: `plusone@temp-mail.org`
- This requires assembling an email from spoken components. Extremely hard, compound operation. NOT in v3 training.

**6. code-aware (#21) — UNIVERSAL FAILURE**
- Input: `the jsx compiler option should be set to react jsx`
- Expected: `the JSX compiler option should be set to react-jsx`
- Requires domain knowledge (JSX is an acronym, react-jsx is hyphenated). Not a verbal command — just code literacy. NOT in v3 training.

**7. hard-negative (#22, #23) — most models struggle**
- Input: `Never mind the formatting, just get the content right first.`
- Expected: Same (no change!)
- Llama and LFM2 delete "Never mind" because it sounds like a correction command.
- **This IS relevant** — Spoke gets triggered by keywords, but sometimes those keywords are used normally.

### The "Wispr" Compound Operation Failure

This is our biggest frustration. We tested all models on:
- Input: `Hey this is a transcription test for Whisper Flow. Can you spell that W-I-S-P-R?`
- Expected: `Hey this is a transcription test for Wispr Flow.`

**No model gets this fully right.** They can assemble the letters (W-I-S-P-R → Wispr) but fail to correctly replace "Whisper" with "Wispr" while keeping "Flow." This is a compound operation: (1) assemble letters, (2) identify which word to replace, (3) preserve surrounding context, (4) remove the instruction.

The models learned the PATTERN of spell-replace from 15 examples, but they didn't learn the REASONING. With only 15 spell-replace examples, there isn't enough diversity for the model to generalize to novel cases.

---

## Our Thesis: Why We're Stuck at 87-91%

1. **Data ceiling, not model ceiling.** Four different architectures (1.2B to 4.5B params) all converge to the same range. The bottleneck is training data, not model capacity.

2. **Pattern matching, not understanding.** The model learned surface patterns ("when you see 'spell that X-Y-Z', replace the previous word"). It fails when the pattern is novel or when multiple operations must be chained.

3. **Category imbalance.** Spell-replace (hardest, 3% of data), caps (3%), and emphasis (2%) are severely underrepresented. camelCase (easiest, 28%) is overrepresented.

4. **No compound operation examples.** The training data has almost no examples that require chaining 2+ operations. Real speech often combines operations naturally.

---

## DELIVERABLE 1: Regular Correction Examples (~400-500 new)

### Priority 1: Spell-Replace Diversity (80-100 new examples)

This is the category with the highest failure rate and lowest training count. We need:

**Simple spell-replace (30 examples):**
- `The library is called Langchain, spell that L-A-N-G-C-H-A-I-N.` → `The library is called Langchain.` (no change needed — spelling matches!)
- `I'm using the Supabase platform. Can you spell that S-U-P-A-B-A-S-E?` → `I'm using the Supabase platform.`
- `She ordered the Gochujang, spell that G-O-C-H-U-J-A-N-G.` → `She ordered the Gochujang.`

Key: vary the position (beginning, middle, end of sentence), vary the trigger phrase ("spell that", "spell it as", "can you spell that", "that's spelled"), vary the word length (4-10 letters).

**Corrective spell-replace (40 examples):**
Where the spelling DIFFERS from the original word (the main use case):
- `The app uses Faunadb for storage. Spell that F-A-U-N-A-D-B.` → `The app uses Faunadb for storage.`
- `Check out the Langchain docs. Wait, spell that L-A-N-G-C-H-E-I-N.` → `Check out the Langchein docs.`
- `We deployed to Versell. Spell that V-E-R-C-E-L.` → `We deployed to Vercel.`

Key: the ASR often mishears proper nouns. The user then spells the correct version. The model must (1) assemble the letters, (2) find the closest-matching word in the sentence, (3) replace it.

**Compound spell-replace (30 examples):**
Where spelling happens alongside other operations:
- `I'm using Wisper for transcription. Spell that W-I-S-P-R. And emphasize transcription.` → `I'm using Wispr for **transcription**.`
- `The API is called Anyscale. Spell that A-N-Y-S-C-A-L-E. Put an at symbol before API.` → `The @API is called Anyscale.`
- `We're migrating to Surealdb, spell that S-U-R-R-E-A-L-D-B. Actually wait, we're migrating from Mongo, not to it.` → `We're migrating from SurrealDB, not to it.` (spell + self-correction — very hard!)

### Priority 2: Compound / Multi-Step Operations (80-100 new examples)

Examples that require chaining 2 or more operations in a single input. This is the category most responsible for our ceiling.

**Spell + other (20 examples):**
- See compound spell-replace above

**Self-correction + other (20 examples):**
- `The meeting is at 3pm. Wait no, 4pm. And emphasize the time.` → `The meeting is at **4pm**.`
- `Send it to the dev channel. Actually, the staging channel. Tag it with an at symbol.` → `Send it to the @staging channel.`

**Quote + other (20 examples):**
- `He called it quote-unquote revolutionary. And that word should be in all caps.` → `He called it "REVOLUTIONARY".`
- `She said quote I'm done end quote. Use the fire emoji after that.` → `She said "I'm done" 🔥`

**3+ operations (20-30 examples):**
- `The function is called getserverprops. It's quote-unquote stable now. Emphasize stable. And add the fire emoji.` → `The function is called getServerProps. It's "**stable**" now. 🔥`
- `Message the devops team about Kubernetes. Spell that K-U-B-E-R-N-E-T-E-S. Tag devops with an at symbol. Show excitement.` → `Message the @devops team about Kubernetes!`

### Priority 3: Self-Correction Diversity (60-80 new examples)

Focus on the AMBIGUOUS cases — partial corrections in lists, corrections that only affect one word, corrections with natural-sounding phrasing:

**Partial list correction (20 examples):**
- `We support Python, Java, and Rust. Actually wait, not Rust, Go.` → `We support Python, Java, and Go.`
- `The stack is React, Express, and MongoDB. Sorry, PostgreSQL not MongoDB.` → `The stack is React, Express, and PostgreSQL.`
- `Invite Alice, Bob, and Charlie. Wait, not Charlie. David.` → `Invite Alice, Bob, and David.`

Key: only the LAST item changes, the rest stays. This is what model fails on — it either keeps everything or deletes too much.

**Mid-sentence correction (20 examples):**
- `The endpoint is slash API slash users. Wait no, slash API slash accounts.` → `The endpoint is /API/accounts.`
- `It costs twenty dollars. No wait, twenty-five.` → `It costs twenty-five dollars.`
- `The release is on March 3rd. Actually, March 10th.` → `The release is on March 10th.`

**Ambiguous correction (20 examples):**
Where it's less obvious what's being corrected:
- `I love TypeScript and Python equally. Well, actually, maybe TypeScript a bit more.` → `I love TypeScript and Python equally. Well, actually, maybe TypeScript a bit more.` (this is NOT a correction — it's a qualifier! Keep as-is.)
- `The project uses React. Actually, it's built on Next.js, which uses React.` → `The project uses React. Actually, it's built on Next.js, which uses React.` (clarification, not correction — keep as-is!)
- `We're moving to AWS. Sorry, I mean we're EVALUATING AWS.` → `We're evaluating AWS.` (correction + nuance)

### Priority 4: Caps and Emphasis Expansion (40-50 new examples)

These are thin categories. We need diversity in how the command is phrased:

**Caps (20-25 examples):**
- `This is urgent. All caps.` → `THIS IS URGENT.`
- `Do not merge that branch. Make it all uppercase.` → `DO NOT MERGE THAT BRANCH.`
- `The headline should be exciting. Capitalize the whole thing.` → `THE HEADLINE SHOULD BE EXCITING.`
- `Warning, system overload. All caps on warning.` → `WARNING, system overload.` (caps on specific word only!)

**Emphasis (20-25 examples):**
- `We need this done today. Bold today.` → `We need this done **today**.`
- `The key insight is that latency matters more than throughput. Emphasize latency and throughput.` → `The key insight is that **latency** matters more than **throughput**.`
- `Do not push to main. Emphasize not.` → `Do **not** push to main.`
- `It's not just fast, it's blazingly fast. Emphasize blazingly fast.` → `It's not just fast, it's **blazingly fast**.`

### Priority 5: Emoji Diversity (30-40 new examples)

We have 48 emoji examples but they may lack diversity. Ensure coverage of:

- Multi-emoji: `Celebration emoji and sparkles emoji` → `🎉✨`
- Emoji + text: `Great news about the launch rocket emoji` → `Great news about the launch 🚀`
- Less common emoji: `Saluting face emoji` → `🫡`, `Chef's kiss emoji` → `🤌`
- Emoji in context: `The bug is finally fixed. Party popper emoji.` → `The bug is finally fixed. 🎉`

### Priority 6: Disfluency Handling (20-30 new examples)

We only have ~3 disfluency examples. Need more:
- `So um I was thinking we could uh maybe refactor the auth module.` → `So I was thinking we could maybe refactor the auth module.`
- `The um the deployment failed because ah the config was wrong.` → `The deployment failed because the config was wrong.`
- `I uh I think we should um probably just revert the commit.` → `I think we should probably just revert the commit.`

---

## DELIVERABLE 2: Hard Negatives (~300 examples, SEPARATE FILE)

**These go in a separate JSONL file**, not mixed with the regular examples. We will use them in a second training phase at lower learning rate.

Hard negatives are sentences where a trigger keyword appears in normal English — NOT as a command. The correct output is the input with only punctuation/capitalization fixes (no semantic changes).

### Why This Matters

The trigger system fires on bare keywords. "Actually, I think the design is solid" fires the disfluency trigger. The model runs. If every training example the model has ever seen says "when you see 'actually', apply a correction," it WILL incorrectly edit this sentence. Hard negatives teach it: **sometimes the right answer is to leave it alone.**

### Distribution by Trigger Type

**Disfluency hard negatives (100-120 examples) — MOST CRITICAL:**

"actually", "sorry", "wait", and "I mean" appear in ~30% of normal English. These need the most coverage.

- `Actually, I think the architecture is really clean.` → `Actually, I think the architecture is really clean.`
- `Sorry about the delay, I was in a meeting.` → `Sorry about the delay, I was in a meeting.`
- `Wait for the tests to pass before deploying.` → `Wait for the tests to pass before deploying.`
- `I mean, it's not perfect but it ships.` → `I mean, it's not perfect but it ships.`
- `I'm sorry but I don't think we should merge this yet.` → `I'm sorry but I don't think we should merge this yet.`
- `Actually it turns out the bug was in the config all along.` → `Actually it turns out the bug was in the config all along.`
- `My bad, I should have caught that in code review.` → `My bad, I should have caught that in code review.`

Key variations needed:
- "actually" as sentence starter vs mid-sentence ("I actually think...", "It's actually quite good")
- "sorry" as apology vs politeness ("sorry to bother you", "sorry about that")
- "wait" as instruction ("wait for", "wait until") vs hesitation
- "I mean" as filler vs clarification ("I mean, it's fine", "I mean the backend, not the frontend")
- "oops" as mild reaction, not correction ("oops, wrong channel")

**Quote hard negatives (50-60 examples):**

- `Can you send me a quote for the enterprise plan?` → `Can you send me a quote for the enterprise plan?`
- `That's a direct quote from the documentation.` → `That's a direct quote from the documentation.`
- `The quote on the landing page needs updating.` → `The quote on the landing page needs updating.`
- `I'll get you a price quote by end of day.` → `I'll get you a price quote by end of day.`
- `He quotes that paper in every meeting.` → `He quotes that paper in every meeting.`

**Symbols hard negatives (40-50 examples):**

- `The HTML tag needs a class attribute.` → `The HTML tag needs a class attribute.`
- `Add a price tag to each item in the store.` → `Add a price tag to each item in the store.`
- `The dollar symbol is used in jQuery selectors.` → `The dollar symbol is used in jQuery selectors.`
- `We need to tag this release before deploying.` → `We need to tag this release before deploying.`
- `The hashtag trend is dying down.` → `The hashtag trend is dying down.`

**Casing hard negatives (30-40 examples):**

- `Check if caps lock is on, that might be the issue.` → `Check if caps lock is on, that might be the issue.`
- `Don't capitalize every word in the title.` → `Don't capitalize every word in the title.`
- `The uppercase version of the string is cached.` → `The uppercase version of the string is cached.`
- `Make sure the caps lock key isn't stuck.` → `Make sure the caps lock key isn't stuck.`

**Spelling hard negatives (30-40 examples):**

- `I can never spell bureaucracy correctly.` → `I can never spell bureaucracy correctly.`
- `Can you spell that out for me? I didn't catch it.` → `Can you spell that out for me? I didn't catch it.`
- `The spelling of that word is unusual.` → `The spelling of that word is unusual.`
- `How do you spell your last name?` → `How do you spell your last name?`
- `Check your spelling before submitting the PR.` → `Check your spelling before submitting the PR.`

### Hard Negative Quality Rules

- Output = input with ONLY punctuation/capitalization fixes
- Never delete or rearrange words
- The trigger word must appear naturally, not as a command
- Vary sentence length (short 5-word to long 20-word)
- Include tech topics AND everyday topics
- Make them genuinely tricky — the best hard negatives are ones where a naive model would incorrectly edit

---

## Quality Guidelines

### DO:
- Make inputs sound like real spoken language (contractions, run-ons, natural phrasing)
- Vary the trigger phrases (don't always use "spell that" — use "spell it as", "can you spell", "that's spelled", etc.)
- Include technical vocabulary (programming, devops, ML terms) since Spoke's user base is developers
- Make some examples long (2-3 sentences) and some short (1 sentence)
- Include profanity occasionally (the system prompt says to preserve it)
- Vary sentence structure and topic (don't make everything about code)

### DON'T:
- Don't make every example perfectly grammatical — these are transcripts of spoken language
- Don't include categories we intentionally removed: formatting-xml, email addresses, code-aware (JSX/hyphenation)
- Don't add explanations or chain-of-thought in the assistant response — output is ONLY the cleaned text
- Don't capitalize the user input unless the ASR would (ASR typically produces lowercase or sentence case)
- Don't make hard negatives too obvious — see Deliverable 2 section for details

### The Golden Rule:
**Every output word must either appear in the input or be produced by an explicit directive.** The model should never add words, commentary, or explanations. It just cleans and executes.

---

## Delivery Format

**IMPORTANT: Deliver TWO separate files. Do NOT mix them.**

### File 1: `new_regular.jsonl` (~400-500 examples)
- Regular correction examples (spell-replace, compound ops, self-correction, caps, emphasis, emoji, disfluency)
- Every example has a meaningful edit (input ≠ output semantically)
- Mix categories randomly (don't group by category)

### File 2: `new_hard_negatives.jsonl` (~300 examples)
- Hard negative examples ONLY (trigger keyword present but no edit needed)
- Output = input with only punctuation/capitalization fixes
- Mix trigger types randomly

**Why separate?** We train in two phases. Phase 1 uses regular examples to teach correction skills. Phase 2 uses hard negatives at 10x lower learning rate to teach when NOT to correct. Mixing them would confuse the model.

### Format (same for both files)
- Each line follows the JSONL format above (system/user/assistant messages)
- No duplicate or near-duplicate examples
- No test set contamination — do NOT reproduce any of the test examples listed below

### Test Examples to AVOID (do not generate anything similar to these):

**V3 test set (23 examples):**
1. `I'm gonna be using Celero VAD for this. Can you spell that as S-I-L-E-R-O?`
2. `He went to the Khadai, spell that K-A-D-A-I.`
3. `The new restaurant on 4th serves amazing Kibbeh Nayeh, spell that K-I-B-B-I-N-A-Y...`
4. `The backend is actually powered by Cloudflare Workers and Groq. Wait no, sorry, Fireworks.`
5. `Let's meet at 11am, Saturday. Actually scratch that, let's meet at 12pm, Thursday.`
6. `We're using React and Vue. Wait no, sorry, Svelte.`
7. `I mean they said I was quote-unquote lucky to be here...`
8. `Quote-unquote gluten-free bread...`
9. `I mean they said I was quote lucky to be here. end quote...`
10. `The debug log shows quote connection timed out after 30 seconds end quote...`
11. `He described the interview as quote the most stressful hour of my life end quote.`
12. `I wanna fix app.py and test.py. Add an at symbol before the file names.`
13. `We're so excited to make this. Use all caps.`
14. `It's surprisingly fast. Emphasize surprisingly.`
15. `Heartbreak emoji`
16. `Praying hands emoji for the victims`
17. `So we've got a file called usetranscription.ts.`
18. `Their quote-unquote innovation is just copying what everyone else already does.`
19. `Notify the ops team about the outage. Add an at symbol before ops.`
20. `Stop ignoring my emails. Make that all caps.`
21. `This is absolutely critical for the launch. Emphasize critical.`
22. `We just shipped the new feature fire emoji`
23. `The main page uses a component called navigationbar that wraps everything.`

**V2 test extras (also avoid):**
- `The result was unexpected. Emphasize unexpected. And wrap the whole sentence in a result tag.`
- `The function returns null on error. XML open error, XML close error, around null.`
- `The invite list includes plus one, literally the word plus and the number one, at sign temp hyphen mail dot o r g...`
- `Check your tsconfig, the jsx compiler option should be set to react jsx...`
- `Wait until you see the new office, it's completely different from the old one.`
- `Never mind the formatting, just get the content right first.`

---

## Summary: Target Distribution

### File 1: Regular Examples (~400-500 new)

| Category | New Examples | Current | Total After | Why |
|---|---|---|---|---|
| Spell-replace | 100-120 | 15 | ~130 | Hardest category, biggest failure mode |
| Compound/multi-step | 100-120 | ~10 | ~120 | The ceiling-breaker — chained operations |
| Self-correction | 60-80 | 70 | ~140 | Need ambiguous + partial corrections |
| Caps | 25-30 | 17 | ~45 | Thin category, needs diversity |
| Emphasis | 25-30 | 12 | ~40 | Thin category, needs diversity |
| Emoji | 30-40 | 48 | ~85 | Need more diversity |
| Disfluency | 20-30 | 3 | ~28 | Severely underrepresented |
| **Subtotal** | **~450** | **535** | **~985** | |

### File 2: Hard Negatives (~300 new, SEPARATE FILE)

| Trigger Type | Count | Why |
|---|---|---|
| Disfluency ("actually", "sorry", "wait", "I mean") | 100-120 | Most common false triggers in English |
| Quotes ("quote") | 50-60 | "quote" has many non-command meanings |
| Symbols ("tag", "symbol") | 40-50 | Common in tech vocabulary |
| Casing ("caps", "capitalize", "uppercase") | 30-40 | Appears in tech discussions |
| Spelling ("spell") | 30-40 | "How do you spell..." is common |
| **Subtotal** | **~300** | |

### Grand Total: ~1,285 training examples (985 regular + 300 hard negatives)

Training plan:
- **Phase 1:** 985 regular examples, lr=1e-5, ~1500 iters
- **Phase 2:** 300 hard negatives, lr=1e-6, ~300 iters (resume from Phase 1 checkpoint)
