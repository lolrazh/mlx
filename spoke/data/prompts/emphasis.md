You are generating training data for a small LLM that post-processes dictation transcripts. The model receives raw ASR output and must execute embedded verbal commands — NOT explain them, NOT chat about them — just output the cleaned text.

Task: Generate diverse (input, ideal_output) pairs for the category: EMPHASIS / BOLD

The speaker wants certain words emphasized. The model must apply the emphasis and remove the verbal instruction.

## Sub-types

### 1. Bold (markdown **word**)
- "emphasize X", "bold X", "make X bold"
- The target word(s) get wrapped in **double asterisks**

Input: "It's surprisingly fast. Emphasize surprisingly."
Output: "It's **surprisingly** fast."

Input: "We need to refactor the auth module and the cache layer. Bold auth and cache."
Output: "We need to refactor the **auth** module and the **cache** layer."

### 2. Stress via CAPS
- "stress X", "emphasis on X", "emphasize X and Y"
- The target word(s) become UPPERCASE (selective, not whole sentence)

Input: "She said the deadline moved to Friday. Stress Friday and deadline."
Output: "She said the DEADLINE moved to FRIDAY."

Input: "The test results show regression in performance. Emphasis on regression and performance."
Output: "The test results show REGRESSION in PERFORMANCE."

### 3. Bold + excitement
- Combining emphasis with "show excitement"

Input: "The performance improvement is significant. Make significant bold and show excitement."
Output: "The performance improvement is **significant**!"

## Important notes
- The emphasis instruction must be REMOVED from the output
- When "bold" or "emphasize" is used → **word** (markdown bold)
- When "stress" or "emphasis on" is used → WORD (uppercase)
- Multiple words can be emphasized in one instruction
- The word being emphasized appears UNCHANGED in the sentence — only its formatting changes

Generate 10 new pairs. Rules:
- Mix sub-types: ~4 bold (**), ~4 stress (CAPS), ~2 combined or edge cases
- Vary number of emphasized words: 1, 2, or 3 words per example
- Vary sentence length and topic
- Make emphasis targets varied: nouns, verbs, adjectives, adverbs
- Make instructions sound natural — "put emphasis on", "stress the word", "bold that"
- Include 2-3 with light disfluencies

Output as JSON array:
[{"input": "...", "ideal": "..."}, ...]
