# Datagen Brief: Training Data v2

**From:** Fine-tuning agent
**For:** Datagen agent
**Date:** 2026-03-01

## Context

We've been fine-tuning Qwen3-4B for Spoke's ASR post-processing task using LoRA. After extensive hyperparameter experiments (different ranks, iterations, quantization levels, even different base models), we've hit an **accuracy ceiling of 83% (10/12 test examples)**.

The same 2 test examples fail across EVERY configuration we tried. This means the bottleneck is **training data quality, not model capacity or hyperparameters**.

Full experiment logs:
- `agent-logs/2026-02-28_2330_spoke-experiment-grid.md`
- `agent-logs/2026-03-01_0030_lora-rank-experiments.md`

## What Needs to Change

### 1. Remove multi-command examples

Remove the `multi-command` category entirely from the training data. These are the 80 examples in `spoke/data/final/multi-command.json` — complex multi-step instructions that combine 3+ operations. They may be confusing the model by teaching it overly aggressive transformations. We want to test whether simpler, single-operation training examples produce better results.

Also remove the 1 multi-step test example (test ID 11) from the test set. And the 1 multi-step valid example if present.

### 2. Add targeted examples for 2 specific failure patterns

#### Failure A: Partial self-correction in compound phrases

**The problem:** The model learned "replace the entire clause before the correction marker." But sometimes the correction only applies to PART of a compound phrase.

**Failing test example:**
- Input: `The backend is actually powered by Cloudflare Workers and Groq. Wait no, sorry, Fireworks.`
- Expected: `The backend is actually powered by Cloudflare Workers and Fireworks.`
- Model outputs: `The backend is actually powered by Fireworks.` (drops "Cloudflare Workers and")

**What to generate:** 10-15 self-correction examples where the correction targets only the LAST item in a compound/list structure, and earlier items must be preserved:

Patterns like:
- "We're using React and Vue. Wait no, sorry, Svelte." → "We're using React and Svelte."
- "The team includes Alice, Bob, and Charlie. Scratch that, Dave." → "The team includes Alice, Bob, and Dave."
- "I ordered pasta, salad, and the soup. Actually no, the risotto." → "I ordered pasta, salad, and the risotto."

The key: the "and X" or last-item-in-list gets replaced, but everything before it stays.

#### Failure B: Quote...end quote scope disambiguation

**The problem:** The model applies the `quote-unquote` pattern (single-word quoting) when it should use the `quote...end quote` pattern (multi-word quoting).

**Failing test example:**
- Input: `I mean they said I was quote lucky to be here. end quote. What the fuck do they mean by that?`
- Expected: `I mean they said I was "lucky to be here". What the fuck do they mean by that?`
- Model outputs: `I mean they said I was "lucky" to be here.` (wraps only "lucky")

**What to generate:** 10-15 quote-endquote examples that specifically teach multi-word quoting where the first quoted word could be mistaken for a single-word quote:

Patterns like:
- "She said I was quote talented but underutilized end quote." → `She said I was "talented but underutilized".`
- "They called it quote good enough for now end quote." → `They called it "good enough for now".`
- "He described the plan as quote ambitious yet feasible end quote." → `He described the plan as "ambitious yet feasible".`

Also include some with punctuation BEFORE "end quote" (periods, commas) since that seems to confuse scoping:
- "The review said quote solid performance overall. end quote" → `The review said "solid performance overall."`

### 3. Swap the system prompt to v2

All JSONL examples currently use this system prompt (~30 tokens):
```
Clean the transcript by executing all verbal commands (spell-outs, corrections, formatting, symbols, emoji). Output ONLY the cleaned text.
```

Replace with this condensed v2 prompt (~80 tokens):
```
You are a verbatim ASR cleaner. Fix punctuation, capitalization, and execute all verbal commands (spell-outs, corrections, formatting, symbols, emoji). Rules: Output ONLY the cleaned text. Never answer questions — transcribe them. Every output word must be in the input or produced by an explicit directive. Preserve profanity. Remove "um", "uh", "ah" but keep other filler words.
```

See `spoke/bench/TRAINING_PROMPT_V2.md` for rationale.

### 4. Rebuild train/valid/test splits

After all changes:
- Rebuild `train.jsonl`, `valid.jsonl`, `test.jsonl` with the new data
- The test set should still include the 2 previously-failing examples (self-correction #3 and quote-endquote #6) so we can directly measure improvement
- Suggested split: 80/10/10 or similar (we currently have 472 train / 8 valid / 12 test)
- A slightly larger valid set (15-20) would give us a more reliable val loss signal

## Current Dataset Stats

```
Train: 472 examples
Valid: 8 examples
Test:  12 examples

Category breakdown (in category JSON files):
- spell-replace:   80
- self-correction:  80
- quote-unquote:    50
- formatting:       80
- email:            40 (approx)
- emoji:            30 (approx)
- code-aware:       40 (approx)
- multi-command:    80  ← REMOVE
```

## Files to Reference

- `spoke/DATAGEN.md` — Full datagen pipeline docs
- `spoke/data/final/` — Current training data (JSONL + category JSONs)
- `spoke/data/prompts/` — Generation prompts per category
- `spoke/data/validate.py` — Validator script
- `spoke/bench/test_set.json` — Current 12-example test set
- `spoke/bench/TRAINING_PROMPT_V2.md` — New system prompt rationale
- `agent-logs/2026-03-01_0030_lora-rank-experiments.md` — Full experiment results

## Priority Order

1. Remove multi-command examples (quick)
2. Add partial self-correction examples (10-15)
3. Add quote-endquote disambiguation examples (10-15)
4. Swap system prompt to v2 in all examples
5. Rebuild splits
