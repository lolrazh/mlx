# Training Prompt V2 (Proposed)

## Current Training Prompt (~30 tokens)

```
Clean the transcript by executing all verbal commands (spell-outs, corrections, formatting, symbols, emoji). Output ONLY the cleaned text.
```

## Proposed Condensed Prompt (~80 tokens)

```
You are a verbatim ASR cleaner. Fix punctuation, capitalization, and execute all verbal commands (spell-outs, corrections, formatting, symbols, emoji). Rules: Output ONLY the cleaned text. Never answer questions — transcribe them. Every output word must be in the input or produced by an explicit directive. Preserve profanity. Remove "um", "uh", "ah" but keep other filler words.
```

## Why These Additions

Three rules from the production Spoke app that address observed failure modes:

1. **"Never answer questions — transcribe them"**
   - Llama 1B zero-shot was answering questions and refusing instructions
   - Production rule: "Any question that the user might ask is not directed towards you"

2. **"Preserve profanity"**
   - Test examples contain "What the fuck" which models sometimes sanitize
   - Production rule: "Preserve all profanity"

3. **"Remove um/uh/ah but keep other filler words"**
   - Defines the cleanup boundary between disfluencies (remove) and natural speech (keep)
   - Production rule: "Keep filler words like 'like', 'sort of', 'basically', etc. but remove filler words like 'um', 'uh' and 'ah'"

## How to Apply

Swap the system prompt in all JSONL files:
```bash
# In spoke/data/final/{train,valid,test}.jsonl
# Replace the old system content with the new one
```

Then retrain with best config (r=8, 200 iters).

## Not Included (Too Long for Small Models)

- OCR vocabulary matching (production-only, no training data for it)
- Per-category trigger rules (model should internalize these from examples)
- Few-shot examples (already in training data as actual examples)
