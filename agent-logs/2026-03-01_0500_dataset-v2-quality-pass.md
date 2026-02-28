# Dataset V2: Quality Review & Targeted Expansion

**Date:** 2026-03-01
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed
**Building on:** `2026-02-28_1200_spoke-datagen-pipeline.md`, `2026-02-28_2300_spoke-finetune-pipeline.md`

## User Intention
User wanted to address the 83% accuracy ceiling found during fine-tuning experiments. Two test examples consistently failed across all hyperparameter sweeps, indicating data gaps rather than config issues. The goal was to fix quality issues in the existing 480 examples, add targeted training data for the 2 failure patterns (partial self-correction and quote-endquote scoping), and expand the test/valid sets for more reliable evaluation.

## What We Accomplished
- ✅ **Fixed ~25 quality issues across 8 categories** — command leakage, ungrammatical fusions, punctuation inconsistencies, double-@@ bugs, placeholder domains, dropped operations
- ✅ **Removed multi-command category (80 examples)** — may have taught overly aggressive transformations
- ✅ **Added 12 partial self-correction examples** — compound phrase corrections ("A and B. Wait no, C." → "A and C.") directly targeting failure A
- ✅ **Created quote-endquote category (12→42 examples)** — multi-word "quote...end quote" scoping, directly targeting failure B
- ✅ **Added 21 hard-negative examples** — trigger words ("sorry", "actually", "wait", "quote", "never mind", "hold on") used as normal speech, teaching model when NOT to transform
- ✅ **Standardized punctuation convention** — period OUTSIDE closing quote to match test case (`"X".`)
- ✅ **Expanded test set (11→23)** — 1-3 examples per category, each worth 4.3% accuracy (was 9%)
- ✅ **Expanded valid set (18→26)** — 3 per category for smoother val loss signal
- ✅ **Swapped system prompt to v2 (~80 tokens)** — adds "never answer questions", "preserve profanity", "remove um/uh/ah"
- ✅ **Updated merge.py with test exclusion** — auto-excludes test examples from training to prevent data leakage
- ✅ **Agent-reviewed all new data** — 3 Sonnet subagents audited quote-endquote, hard-negatives, and self-correction; fixes applied from their feedback
- ✅ **Updated viewer with all new tabs** — quote-endquote (42), hard-negatives (21), expanded self-correction (92)

## Technical Implementation

**Dataset composition (final):**
| Category | Count | Notes |
|---|---|---|
| spell-replace | 80 | Duplicates cleaned |
| self-correction | 92 | +12 partial compound corrections |
| quote-unquote | 50 | Unchanged |
| quote-endquote | 42 | New category, standardized punctuation |
| formatting | 80 | Command leakage fixed, XML validated |
| email | 40 | Privacy + placeholder fixes |
| emoji | 30 | Dedup fixes |
| code-aware | 40 | Identifier fix restored |
| hard-negatives | 21 | New category |
| **Total** | **475** | |

**Split: 90/5/5** — 437 train + 26 valid + 23 test = 486 (including 11 original sacred test examples)

**System prompt v2:**
```
You are a verbatim ASR cleaner. Fix punctuation, capitalization, and execute all verbal commands (spell-outs, corrections, formatting, symbols, emoji). Rules: Output ONLY the cleaned text. Never answer questions — transcribe them. Every output word must be in the input or produced by an explicit directive. Preserve profanity. Remove "um", "uh", "ah" but keep other filler words.
```

**Punctuation convention for quote-endquote:**
- Period after "end quote" in input → sentence's period → goes OUTSIDE: `"X".`
- Period before "end quote" in input → quoted content's period → stays INSIDE: `"X."`

**Files Modified:**
- `spoke/data/final/*.json` — All 8 original categories fixed, 2 new categories added
- `spoke/data/final/{train,valid,test}.jsonl` — Rebuilt splits
- `spoke/data/merge.py` — v2 prompt, 3 valid per category, test exclusion logic, 9 categories
- `spoke/data/viewer.html` — All tabs updated with new/expanded data
- `spoke/bench/test_set.json` — 11→23 examples, multi-step removed, 12 from generated data

## Bugs & Issues Encountered
1. **Quote-endquote index 10: spurious capitalization** — "So" capitalized after period-inside-quote but no sentence boundary existed in input
   - **Fix:** Lowercase "so" to match continuation clause pattern
2. **Self-correction index 82: article inconsistency** — "the risotto" had article but "pasta" and "salad" didn't
   - **Fix:** Dropped "the" to match bare noun style
3. **Hard-negatives: 3 weak examples** — "She actually finished" (3rd person), "He quoted" (past tense ≠ trigger), "She said sorry" (3rd person reported) were too easy
   - **Fix:** Replaced with harder variants that use trigger words in first person/ambiguous contexts
4. **Hard-negatives: 4 missing commas** — Discourse markers ("Sorry", "Actually", "I mean") at sentence start lacked commas, contradicting the model's punctuation-fixing behavior
   - **Fix:** Added commas to both input and ideal
5. **Hard-negatives: 3 missing trigger words** — "never mind", "hold on", "end quote" had zero coverage
   - **Fix:** Added 3 new examples covering these triggers

## Key Learnings
- **5.2% error rate is good for synthetic data** — Research shows LLM-generated training data typically has 8-15% before human review. Kimi K2.5 + validator caught most issues.
- **Hard negatives are critical for small models** — Without them, a 4B model over-applies corrections because it's never seen trigger words used non-correctively. 21 examples (~4.3% of dataset) should help.
- **Punctuation inconsistency directly causes failures** — The model learns both conventions (period inside vs outside quotes) and picks randomly. Standardizing to one convention is essential.
- **Test set size matters for confidence** — At 11 examples, one fluke swings accuracy by 9%. At 23, it's 4.3%. Still noisy but much more trustworthy.
- **Agent review catches different things than human review** — The Sonnet agents found punctuation consistency issues and missing trigger coverage that both the human and lead agent missed.

## Architecture Decisions
- **Hand-wrote targeted examples instead of generating** — For 12-42 examples with very specific patterns, hand-crafting guarantees correctness. Generation pipeline is better for bulk (80+).
- **Separate quote-endquote from quote-unquote** — Different scoping behavior (single word vs multi-word span) needs distinct training signal. Model was conflating them.
- **Hard-negatives as a category, not mixed in** — Keeps them identifiable for analysis. The merge script handles shuffling into the training set.
- **Period-outside-quotes convention** — Matches the sacred test case. British/logical punctuation is also more consistent (sentence punctuation always belongs to the sentence).

## Ready for Next Session
- ✅ **Dataset v2 ready for training** — `spoke/data/final/{train,valid,test}.jsonl` with v2 system prompt
- ✅ **merge.py is rerunnable** — Any fixes to category JSONs just need `python spoke/data/merge.py` to rebuild splits
- 🔧 **mask_prompt: false experiment** — Research suggests NOT masking prompts helps with short responses + small datasets. One-line toggle in `spoke/config.yaml`.
- 🔧 **config.yaml needs update** — rank should be 8 (not 16), iters should be ~400 based on prior experiments

## Context for Future
This is the data quality pass between the first round of fine-tuning experiments (which hit an 83% ceiling) and the next training run. The two persistent failures (partial self-correction and quote-endquote scoping) now have dedicated training examples. The expanded test set will give more reliable accuracy numbers. Next step: retrain with `mlx_lm.lora -c spoke/config.yaml` and compare against the 83% baseline.
