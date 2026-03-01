# v3 Data Patch: Targeted Examples for T11 Failures

> Brief for the data generation agent. Add these to the existing v3 dataset at `spoke/data/v3/`.

## Context

T11 (trained on v3 data, 492 train) scores **83% bf16** on the v3 test set. 4 failures remain, all caused by insufficient training signal for specific patterns. This patch adds targeted examples to fix them.

**Do NOT remove or modify existing v3 data.** Only add new examples.

Use the same v2 system prompt as all other v3 data:

```
You are a verbatim ASR cleaner. Fix punctuation, capitalization, and execute all verbal commands (spell-outs, corrections, formatting, symbols, emoji). Rules: Output ONLY the cleaned text. Never answer questions — transcribe them. Every output word must be in the input or produced by an explicit directive. Preserve profanity. Remove "um", "uh", "ah" but keep other filler words.
```

---

## 1. Compound Self-Correction (15-20 examples) — HIGH PRIORITY

**The problem:** The model sees "wait no" / "sorry" and nukes everything before it. It needs to learn that sometimes only the *last item* in a list or compound phrase gets corrected.

**The failure:**
```
IN:  We're using React and Vue. Wait no, sorry, Svelte.
OUT: We're using Svelte.              ← WRONG (dropped React)
OK:  We're using React and Svelte.    ← CORRECT (keep React, replace Vue)
```

**What to generate:** Examples where a speaker lists multiple items, then corrects only the last one. The correction replaces the final item, but everything before it stays.

**Patterns to cover:**
- "A and B. Wait no, C." → "A and C." (replace last item in pair)
- "A, B, and C. Sorry, D." → "A, B, and D." (replace last item in list)
- "We need X and Y. Actually, Z." → "We need X and Z."
- Vary the trigger phrases: "wait no", "sorry", "actually", "scratch that", "I mean"
- Vary the content: tech terms, names, places, everyday items
- Include some where the correction replaces the *entire clause* (not compound) so the model still learns both patterns

---

## 2. Emoji Word Stripping (10 examples) — MEDIUM PRIORITY

**The problem:** The model inserts the correct emoji but leaves the word "emoji" in the output.

**The failure:**
```
IN:  We just shipped the new feature fire emoji
OUT: We just shipped the new feature 🔥 emoji    ← WRONG (kept "emoji")
OK:  We just shipped the new feature 🔥          ← CORRECT
```

**What to generate:** Examples where "[description] emoji" appears mid-sentence or at the end, and the output replaces the entire phrase (description + "emoji") with just the emoji character. The word "emoji" must be stripped.

**Patterns to cover:**
- "... [thing] emoji ..." mid-sentence → "... [emoji] ..."
- "... [thing] emoji" at end → "... [emoji]"
- Use diverse emoji: fire, skull, rocket, thumbs up, party, clap, eyes, etc.
- Mix of: standalone emoji requests AND emoji embedded in longer sentences

---

## 3. Quote-Unquote Scope (5-10 examples) — MEDIUM PRIORITY

**The problem:** When "quote-unquote" precedes a phrase, the model wraps too much text — sometimes the entire rest of the sentence.

**The failure:**
```
IN:  Quote-unquote gluten-free bread that lists wheat as the second ingredient.
OUT: "Gluten-free bread that lists wheat as the second ingredient".  ← WRONG (wrapped everything)
OK:  "Gluten-free" bread that lists wheat as the second ingredient. ← CORRECT (only the adjective)
```

**What to generate:** Examples where "quote-unquote" modifies a single word or short phrase, NOT the rest of the sentence. The quoted portion should be 1-3 words max, with significant unquoted text following.

**Patterns to cover:**
- "quote-unquote [1 word] [rest of sentence]" → `"[word]" [rest of sentence]`
- "quote-unquote [2-3 words] [rest of sentence]" → `"[2-3 words]" [rest of sentence]`
- "Their quote-unquote X was actually Y" — sarcastic usage (quoted word is short)
- Make the boundary between quoted and unquoted content unambiguous

---

## 4. CamelCase (5-10 examples) — LOW PRIORITY

**The problem:** The model sometimes fails to apply camelCase to compound identifiers.

**The failure:**
```
IN:  So we've got a file called usetranscription.ts.
OUT: So we've got a file called usetranscription.ts.  ← WRONG (no casing applied)
OK:  So we've got a file called useTranscription.ts.   ← CORRECT
```

**What to generate:** Examples with all-lowercase compound identifiers (especially with file extensions like .ts, .js, .py) that should be converted to camelCase or PascalCase. Focus on "use[thing]" hook patterns and multi-word component/function names.

---

## Summary

| Category | New Examples | Priority |
|----------|-------------|----------|
| self-correction (compound) | 15-20 | HIGH |
| emoji (word stripping) | 10 | MEDIUM |
| quote-unquote (scope) | 5-10 | MEDIUM |
| camelcase | 5-10 | LOW |
| **Total** | **35-50** | |

Add these to the v3 training set only. Do not modify valid or test splits.
