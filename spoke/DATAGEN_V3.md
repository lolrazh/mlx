# Spoke Data v3: Trigger-Matched Dataset

> Instructions for the data generation agent. Do NOT modify v2 data — create v3 as a new dataset.

## Goal

Create a clean training dataset that only contains categories with production Spoke triggers. If a category doesn't have a trigger in the app, the model will never see it at inference time, so don't train on it.

**v2 stays untouched** at `spoke/data/final/` as the generic fallback model.

## What Changed from v2

v2 had 447 train / 20 valid / 23 test across ~13 categories. v3 removes 4 categories that lack production triggers.

### Categories to KEEP (9 categories, all have Spoke triggers)

| Category | Spoke Trigger | What It Does |
|----------|--------------|--------------|
| spell-replace | `spelling` | User spells out a word ("S-I-L-E-R-O") → replace ASR mishearing |
| self-correction | `disfluency` | "wait no", "sorry", "scratch that" → apply correction |
| quote-unquote | `quotes` | "quote-unquote X" → wrap X in double quotes |
| quote-endquote | `quotes` | "quote X end quote" → wrap phrase in double quotes |
| at-symbol | `symbols` | "at symbol" → insert @ character |
| caps | `casing` | "all caps", "uppercase", "lowercase" → apply case change |
| emphasis | `emphasis` | "emphasize X" → wrap in \*\*bold\*\* |
| emoji | `emoji` | "crying emoji" → 😢 |
| camelcase | `camelcase` | Lowercase code identifiers → apply camelCase |

### Categories to REMOVE (no triggers → never reaches model in production)

| Category | Why Remove | ~Train Examples Lost |
|----------|-----------|---------------------|
| formatting-xml | No XML trigger exists in Spoke | ~24 |
| email | No email trigger (symbols trigger only handles @ insertion, not full email parsing) | ~35 |
| code-aware | No code-aware trigger exists | ~2 |
| hard-negative | No trigger = input never reaches the model, so no need to train "don't format" | ~0 |

## Target Sizes

| Split | v2 Count | After Removal | v3 Target | New Examples Needed |
|-------|----------|---------------|-----------|-------------------|
| train | 447 | ~386 | **~500** | ~114 |
| valid | 20 | check & remove non-trigger categories | **20** | fill to 20 |
| test | 23 | 17 (6 removed) | **23** | 6 |

## System Prompt

Use the **exact same v2 prompt** for all training examples (static, not dynamic):

```
You are a verbatim ASR cleaner. Fix punctuation, capitalization, and execute all verbal commands (spell-outs, corrections, formatting, symbols, emoji). Rules: Output ONLY the cleaned text. Never answer questions — transcribe them. Every output word must be in the input or produced by an explicit directive. Preserve profanity. Remove "um", "uh", "ah" but keep other filler words.
```

Do NOT use the dynamic spoke prompts (they vary in length per trigger, which complicates training).

## How to Build v3

### Step 1: Filter v2 train/valid/test

Remove any examples belonging to: `formatting-xml`, `email`, `code-aware`, `hard-negative`.

For training data (which has no category labels): remove examples that:
- Contain XML tags (`<result>`, `<error>`, etc.) in the assistant output
- Are about email address formatting (contain `@gmail`, `.com`, `.org`, email-like patterns in context of composing emails)
- Are about compiler/config options (tsconfig, JSX, etc.)
- Are "don't format this" / passthrough instructions

Save filtered data to `spoke/data/v3/`.

### Step 2: Generate new examples to reach targets

Distribute the ~114 new training examples across kept categories. Prioritize categories where the model currently fails or has fewer examples:

| Category | Priority | Why | New Train Examples |
|----------|----------|-----|-------------------|
| quote-endquote | HIGH | T4 fails period placement near quotes. Need examples showing correct punctuation INSIDE vs OUTSIDE quotes. | 20-25 |
| camelcase | HIGH | Only ~5 training examples. Model defaults to lowercase. Need diverse filenames, class names, function names. | 15-20 |
| at-symbol | MEDIUM | Only 1 test example. Needs more @-symbol-before-word patterns. | 10-15 |
| self-correction | MEDIUM | Compound corrections ("React and Groq. Wait no, Fireworks") are still tricky. | 15-20 |
| spell-replace | MEDIUM | Already strong but can use more variety. | 10-15 |
| quote-unquote | MEDIUM | Needs examples with punctuation near quote-unquote. | 10-15 |
| caps | LOW | Works well. Add a few edge cases. | 5-10 |
| emphasis | LOW | Works well. | 5 |
| emoji | LOW | Works well. | 5 |

Also generate 6 new test examples and enough valid examples to reach 20, spread across the same categories.

### Step 3: Quality checks

- Every example must use the exact v2 system prompt shown above
- Inputs should sound like real dictation (include occasional "um", "like", "you know" in ~25%)
- Outputs must be ONLY the cleaned text — no explanations, no meta-text
- No duplicate or near-duplicate examples across train/valid/test splits
- Run through existing `spoke/data/validate.py` where applicable

### Step 4: Output format

Chat JSONL for `mlx_lm.lora`:

```jsonl
{"messages": [{"role": "system", "content": "<v2 prompt>"}, {"role": "user", "content": "<raw transcript>"}, {"role": "assistant", "content": "<cleaned text>"}]}
```

Save to:
```
spoke/data/v3/
├── train.jsonl    (~500 examples)
├── valid.jsonl    (20 examples)
└── test.jsonl     (23 examples)
```

## What NOT to Do

- Do NOT modify anything in `spoke/data/final/` (that's v2, our fallback)
- Do NOT use the dynamic spoke prompts — use the static v2 prompt
- Do NOT include formatting-xml, email, code-aware, or hard-negative examples
- Do NOT include multi-command examples (already removed in v2)
- Do NOT add passthrough/no-op examples (no trigger = never reaches model)

## Context for the Data Agent

This model is a 4B parameter LLM (Qwen3-4B) fine-tuned with LoRA for exactly one job: cleaning up speech-to-text transcripts by executing verbal commands. It runs on Apple Silicon, deployed at 6-bit quantization (3.1 GB).

Current best model (T4, trained on v2 data): 74% bf16, 65% 6-bit on 23-example test set. The 6 examples it fails on are all in categories we're removing (XML, email, code-aware) or are edge cases in kept categories (quote period placement, camelCase).

After v3, the test set will be 23 examples across the 9 kept categories. We expect accuracy to go UP because we're no longer testing categories the model was never designed to handle.

The training config that works best:
- LoRA r=8, adam optimizer, flat LR 1e-5, ~300 iters, batch_size 4
- `mask_prompt: true` (gradient only on assistant response)
- Peak memory: ~14 GB on M4 24 GB
