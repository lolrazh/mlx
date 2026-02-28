# Spoke Dataset Generation Pipeline

**Date:** 2026-02-28
**Agent:** Claude Sonnet 4.6
**Status:** 🔄 Ongoing

## User Intention
Build a high-quality synthetic training dataset for fine-tuning Qwen3-4B to post-process ASR (dictation) transcripts. The model must execute embedded verbal meta-commands (spell-outs, self-corrections, quote-unquote, @-mentions, XML tags, formatting, emoji, email dictation) and output clean text. The user wants a fully automated generate→validate→review loop, with themselves reviewing each category before moving to the next.

## What We Accomplished
- ✅ **spell-replace (80 examples)** - Generated, validated, manually reviewed. Realistic ASR error patterns (niche brand/name misspellings), not well-known words like Kubernetes or Figma.
- ✅ **self-correction (80 examples)** - Generated and validated (near 100% pass rate). Parked pending review.
- ✅ **quote-unquote (50 examples)** - Generated and validated. Parked pending review.
- ✅ **formatting (80 examples)** - Generated and validated. Covers 5 sub-types: ALL CAPS, lowercase, bold/emphasis, @-symbol insertion, XML tag wrapping.
- ✅ **email (40 examples)** - Generated, validated, manually fixed 6 bad examples. Covers email dictation, URL dictation, self-corrections mid-address, multi-email.
- ✅ **Viewer** (`spoke/data/viewer.html`) - Browser-based dataset reviewer with tabs per category. Filterable seeds tab + per-category generated tabs.
- ✅ **Generation pipeline** (`spoke/data/generate.py`) - Automated generate→validate→fix→accumulate loop via Kimi K2.5 (Baseten API).
- ✅ **Validator** (`spoke/data/validate.py`) - Category-specific programmatic validators for all 8 categories.
- ⚠️ **emoji (30 target)** - Not yet generated.
- ⚠️ **code-aware (40 target)** - Not yet generated.
- ⚠️ **multi-command (80 target)** - Not yet generated.

## Technical Implementation

**Pipeline flow:**
```
generate batch (Kimi K2.5 via Baseten OpenAI-compatible API)
  → dedup (word overlap threshold 0.8)
  → validate (category-specific checks)
  → fix flagged (up to 2 API calls with error descriptions)
  → accumulate → incremental save → repeat until target
```

**API:** Baseten, model `moonshotai/Kimi-K2.5`, OpenAI-compatible at `https://inference.baseten.co/v1`. Key in `spoke/.env` as `BASETEN_API_KEY`.

**Dataset format (final):** Chat JSONL for `mlx-lm --mask-prompt`:
```jsonl
{"messages": [{"role": "system", "content": "Clean the transcript..."}, {"role": "user", "content": "<raw>"}, {"role": "assistant", "content": "<cleaned>"}]}
```

**Category targets:** spell-replace(80), self-correction(80), quote-unquote(50), formatting(80), email(40), emoji(30), code-aware(40), multi-command(80) = **~480 total**

**Files Modified:**
- `spoke/data/generate.py` - Full pipeline script
- `spoke/data/validate.py` - Category-specific validators
- `spoke/data/viewer.html` - Browser viewer (tabs for seeds + each generated category)
- `spoke/data/prompts/formatting.md` - Rewritten to cover 5 sub-types incl. @-symbol and XML
- `spoke/DATAGEN.md` - Merged at-symbol into formatting; updated targets and progress tracker
- `spoke/data/final/spell-replace.json` - 80 examples
- `spoke/data/final/self-correction.json` - 80 examples
- `spoke/data/final/quote-unquote.json` - 50 examples
- `spoke/data/final/formatting.json` - 80 examples
- `spoke/data/final/email.json` - 40 examples

## Bugs & Issues Encountered

1. **`validate_formatting` didn't cover @-symbol or XML triggers** — After merging at-symbol into formatting, the validator still only checked for caps/lowercase/bold. Every @-symbol and XML example failed with "No formatting trigger found", giving 0% fix rate.
   - **Fix:** Rewrote `validate_formatting` to detect 5 sub-types and run sub-type-specific checks.

2. **Pipeline progress lost on crash** — `generate.py` only saved to `final/` at the very end. A null API response crash on batch 21 lost 68 accumulated formatting examples.
   - **Fix:** Added incremental save after every batch (`final_file.write_text(...)` inside the loop).

3. **`parse_json_response` crash on null API response** — `response.choices[0].message.content` can be `None`; calling `.strip()` on it raised `AttributeError`.
   - **Fix:** Added `if text is None: return []` guard at the top of `parse_json_response`.

4. **Double-`@` bug in email examples** — When an example combined a Slack @-mention with a dictated email address, the model output `@team@company.com` (two `@` signs). Affected examples #3, #5, #10, #19.
   - **Fix:** Manually corrected ideals in `email.json` directly.

5. **Unrealistic ASR errors in spell-replace** — Early batches had "Deloitte" or "Kubernetes" as the misheard word — ASR would never make these errors.
   - **Fix:** Updated `spell-replace.md` prompt with explicit GOOD/BAD examples. Added guidance: the correct word (spelled out) should be unusual/niche; the ASR error should be a phonetically plausible common-word substitution.

6. **SAME-WORD false positive in spell-replace validator** — "Can you spell Massaman M-A-S-S-A-M-A-N" — "Massaman" appears in the instruction phrase, not as the ASR content word. Validator flagged it incorrectly.
   - **Fix:** Added instruction-stripping regex patterns before the SAME-WORD check.

## Key Learnings

- **Kimi K2.5 generates ~9 examples per "Generate 10" request** — consistently returns one fewer. Plan batch sizes accordingly.
- **Fix attempts on structurally wrong examples always fail** — If the validator flags something for a structural reason (wrong trigger, missing `@`, etc.) and the issue is in the prompt's category definition (not the example itself), fix attempts get 0% success. Fix the validator or prompt, not the examples.
- **Category expansion requires validator updates** — Expanding a category's scope (e.g. adding XML to formatting) must be paired with validator updates, otherwise the pass rate tanks completely.
- **Incremental saves are essential** — Baseten API occasionally returns null responses; without incremental saving, a single null response on batch N can erase N*batch_size accumulated examples.
- **Double-`@` is a predictable failure mode** — Any example that combines @-mention (Slack) + email address in the same sentence risks the model conflating them. Future multi-command examples should clearly separate these operations in the ideal.
- **`at-symbol` belongs in `formatting`** — Both are annotation/decoration commands. The validator and prompt naturally absorbed it with no loss of category clarity.
- **Manual review outperforms fix prompts for semantic issues** — Programmatic validators catch structural errors reliably, but "is this ASR error realistic?" requires human/agent judgment. Cheaper to review and drop than to loop fix prompts.

## Architecture Decisions

- **`at-symbol` merged into `formatting`** — Both categories are "annotate/decorate text" commands with no content change. Combined target: 80. Keeps category taxonomy clean.
- **Disfluency pass deferred** — User wants at least 25% of examples to have natural speech disfluencies (ums, ahs, "like", false starts). Plan: run Sonnet sub-agents as a post-processing sweep after all categories are generated. Keeps generation prompts simple now.
- **No train/valid/test split yet** — Split happens after ALL data is generated. Currently all examples are in per-category `final/*.json` files.
- **Seed examples kept separate** — Original 60 gold examples in `evals.csv` / `viewer.html` are not merged into the generated files. They may be folded into training after final review.

## Ready for Next Session
- ✅ **spell-replace, self-correction, quote-unquote, formatting, email** — Complete and in `spoke/data/final/`
- ✅ **Viewer** — All 5 completed categories browsable at `spoke/data/viewer.html`
- ✅ **Pipeline** — Stable, incremental-saving, null-safe
- 🔧 **emoji** — Prompt exists at `spoke/data/prompts/emoji.md`, not yet generated (target: 30)
- 🔧 **code-aware** — Prompt exists, not yet generated (target: 40)
- 🔧 **multi-command** — Prompt exists, not yet generated (target: 80) — hardest category, review carefully
- 🔧 **Disfluency pass** — Post-processing with sub-agents (≥25% of examples get ums/ahs/false starts)
- 🔧 **Final merge** — Combine all categories into `train.jsonl` / `valid.jsonl` / `test.jsonl`

## Context for Future
Five of eight categories are done (330/480 examples). The remaining three are emoji (simple), code-aware (lightweight validator, mostly semantic), and multi-command (hardest — combines 2+ operations). After generation, the dataset needs a disfluency pass (~25% of inputs), then a train/valid/test split, then conversion to chat JSONL format for `mlx-lm` fine-tuning with `--mask-prompt`. The fine-tuning pipeline is being developed in a parallel agent session.
