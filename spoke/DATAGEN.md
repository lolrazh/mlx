# Spoke: Data Generation Pipeline

Companion doc to `PLAN.md`. This covers the dataset curation pipeline in detail.
The fine-tuning pipeline is handled separately.

## Seed Data

60 gold examples in `spoke/data/evals.csv`.

After filtering out passthrough (9), grammar-only (2), and prompt injection (2):
**49 usable seeds** across 9 categories.

Viewer: `spoke/data/viewer.html` — open in browser, filter by category/split, search by `#id`.

## Categories

| # | Category | Seeds | Target | Notes |
|---|----------|-------|--------|-------|
| 1 | spell-replace | ~13 | 80 | ASR mishears word → user spells correct version |
| 2 | self-correction | ~11 | 80 | "wait no", "sorry", "scratch that", "actually" |
| 3 | quote-unquote | ~4 | 50 | "quote-unquote X" or "quote X end quote" → "X" |
| 4 | at-symbol | ~5 | 50 | "at symbol before X" → @X |
| 5 | email | ~1 | 40 | "dot com", "at gmail" → proper email format |
| 6 | formatting | ~7 | 60 | Caps, lowercase, bold/emphasis, excitement |
| 7 | emoji | ~4 | 30 | Verbal emoji name → actual emoji |
| 8 | code-aware | ~3 | 40 | CamelCase, tech terms, filenames |
| 9 | multi-command | ~3 | 80 | 2+ operations combined in one utterance |
| | **Total** | **~49** | **~510** | |

Hard categories (spell-replace, self-correction, multi-command) get more examples
because they have more failure modes and variations.

No passthrough examples — the regex router handles clean inputs.

## Pipeline Per Category

```
1. GENERATE    Copy prompt from spoke/data/prompts/<category>.md
               Paste into Kimi K2.5, get JSON back
               Save to spoke/data/raw/<category>.json

2. VALIDATE    python spoke/data/validate.py <category>
               Runs category-specific programmatic checks
               Splits into passed + flagged

3. FIX         Take flagged examples, send back to Kimi with error descriptions
               Re-validate fixed examples
               Repeat until all pass

4. REVIEW      Updated HTML viewer shows validated data per category
               Human reviews, leaves comments on any issues
               Bulk-fix commented examples

5. MERGE       Combine all passed examples into final dataset
```

### Directory Structure

```
spoke/data/
├── evals.csv                    # Original 60 gold examples
├── viewer.html                  # Browser-based data reviewer
├── validate.py                  # Validator script (all categories)
├── prompts/
│   ├── spell-replace.md         # Generation prompt
│   ├── self-correction.md
│   ├── quote-unquote.md
│   ├── at-symbol.md
│   ├── email.md
│   ├── formatting.md
│   ├── emoji.md
│   ├── code-aware.md
│   └── multi-command.md
├── raw/                         # Raw Kimi output (unvalidated)
│   ├── spell-replace.json
│   └── ...
├── validated/                   # Validator output
│   ├── spell-replace_passed.json
│   ├── spell-replace_flagged.json
│   └── ...
└── final/                       # Merged, reviewed, ready for training
    ├── train.jsonl
    ├── valid.jsonl
    └── test.jsonl
```

## Validators

Each category has a programmatic validator in `validate.py`.

| Category | Key Checks | Confidence |
|----------|-----------|------------|
| spell-replace | Spelled word ≠ input word; letters assemble correctly; spelling instruction removed from ideal | High |
| self-correction | Trigger word present; output shorter than input; output differs from input | Medium |
| quote-unquote | Quote trigger in input; `"` in ideal; trigger removed from ideal | High |
| at-symbol | @-trigger in input; `@` in ideal; trigger removed from ideal | High |
| email | Valid email pattern in ideal; dictated email components in input | High |
| formatting | Format trigger in input; formatting applied in ideal; trigger removed | Medium |
| emoji | Emoji reference in input; actual emoji in ideal; no emoji in input | High |
| code-aware | Output differs from input (lightweight — mostly needs LLM judgment) | Low |
| multi-command | 2+ operation types detected in input; output differs from input | Medium |

"High confidence" validators catch most issues programmatically.
"Low/Medium" validators catch obvious failures but may miss semantic errors — human review matters more for these.

## Generation Prompt Template

Each category prompt follows this structure:

```
System context: You are generating training data for a small LLM that
post-processes dictation transcripts. The model receives raw ASR output
and must execute embedded verbal commands — NOT explain them, NOT chat
about them — just output the cleaned text.

Task: Generate diverse (input, ideal_output) pairs for: [CATEGORY]

Category description: [What the category does, what the model must do]

Rules: [Category-specific constraints and variation requirements]

Seed examples: [3-5 real examples from evals.csv]

Output format: JSON array of {"input": "...", "ideal": "..."}
```

Prompts saved in `spoke/data/prompts/<category>.md`.

## Fix Prompt Template

When examples fail validation, send them back to Kimi with specific error info:

```
These ASR post-processing training examples have specific errors.
Fix ONLY the described error in each example. Keep everything else intact.

[For each flagged example:]
Example N:
  Input: "..."
  Ideal: "..."
  Error: [exact error from validator, e.g. "SAME-WORD: 'Massaman' already
         appears correctly in input — no ASR error to correct"]
  Fix needed: [category-specific instruction, e.g. "Change the word in the
              input to a plausible ASR misheard version of the spelled word"]

Output the fixed examples as a JSON array.
```

## Quality Bar

An example passes if:
1. **Programmatic checks pass** — validator returns no errors
2. **Semantically correct** — the ideal output faithfully executes all commands
3. **Realistic** — the input sounds like something a person would actually dictate
4. **No data leakage** — ideal doesn't contain leftover instructions or meta-text

## Dataset Format (Final)

Chat JSONL for `mlx-lm` with `--mask-prompt`:

```jsonl
{"messages": [{"role": "system", "content": "Clean the transcript by executing all verbal commands (spell-outs, corrections, formatting, symbols, emoji). Output ONLY the cleaned text."}, {"role": "user", "content": "<raw transcript>"}, {"role": "assistant", "content": "<cleaned text>"}]}
```

System prompt is identical across all examples — short and consistent.

## Sizing Rationale

Research findings for LoRA fine-tuning on Qwen3-4B:

- **500 is the floor** for narrow, structured tasks like ours
- **500–1,000 is the sweet spot** — LIMA showed 1K curated > 50K synthetic
- **Beyond 2,000, diminishing returns** for focused tasks
- **Quality >>> quantity** — spend 60-70% of time on curation, not generation
- **LoRA inherently prevents overfitting** on small datasets
- **Our task is ideal for small data**: constrained output space, pattern-based, base model already knows English

Strategy: Start with ~510, evaluate after fine-tuning. If specific categories are weak, add targeted examples for those categories only.

## Progress Tracker

| Category | Prompt | Generated | Validated | Reviewed | Done |
|----------|--------|-----------|-----------|----------|------|
| spell-replace | ✅ | 9 raw | 4 passed, 5 flagged | - | - |
| self-correction | - | - | - | - | - |
| quote-unquote | - | - | - | - | - |
| at-symbol | - | - | - | - | - |
| email | - | - | - | - | - |
| formatting | - | - | - | - | - |
| emoji | - | - | - | - | - |
| code-aware | - | - | - | - | - |
| multi-command | - | - | - | - | - |
