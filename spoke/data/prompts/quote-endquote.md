You are generating training data for a small LLM that post-processes dictation transcripts. The model receives raw ASR output and must execute embedded verbal commands — NOT explain them, NOT chat about them — just output the cleaned text.

Task: Generate diverse (input, ideal_output) pairs for the category: QUOTE...END QUOTE (multi-word quoting)

In this category, the speaker says "quote" to open a quoted span and "end quote" to close it. The model must wrap the span in double quotes and remove the verbal triggers.

## Key distinction: quote-unquote vs quote...end quote

- "quote-unquote X" → wraps a SINGLE word: "X"
- "quote X Y Z end quote" → wraps a MULTI-WORD span: "X Y Z"

This category is ONLY about the multi-word "quote...end quote" pattern.

## Period placement convention (CRITICAL)

This is the hardest part. There are exactly 3 patterns:

### Pattern 1: Period BEFORE "end quote" in input → period INSIDE closing quote
The speaker dictated a period as part of the quoted content.

Input: "The error says quote connection timed out. end quote"
Output: "The error says \"connection timed out.\""

Input: "The sign reads quote no parking after 6pm. end quote Can you believe that?"
Output: "The sign reads \"no parking after 6pm.\" Can you believe that?"

### Pattern 2: No period before "end quote", quote ends sentence → period OUTSIDE
The period belongs to the outer sentence, not the quoted content.

Input: "She described it as quote completely unacceptable end quote."
Output: "She described it as \"completely unacceptable\"."

Input: "His motto is quote move fast end quote."
Output: "His motto is \"move fast\"."

### Pattern 3: Quote in mid-sentence → no period near the quote
The sentence continues after the closing quote.

Input: "He called it quote revolutionary end quote but I disagree."
Output: "He called it \"revolutionary\" but I disagree."

## Real examples

Input: "The review said quote solid performance overall. end quote"
Output: "The review said \"solid performance overall.\""

Input: "Their policy states quote all returns must be processed within 30 days end quote."
Output: "Their policy states \"all returns must be processed within 30 days\"."

Input: "The teacher wrote quote needs improvement in communication skills end quote on the report card."
Output: "The teacher wrote \"needs improvement in communication skills\" on the report card."

Input: "The warning label says quote do not operate heavy machinery. end quote Seems serious."
Output: "The warning label says \"do not operate heavy machinery.\" Seems serious."

Input: "She prefaced it with quote I say this with love end quote and then tore the design apart."
Output: "She prefaced it with \"I say this with love\" and then tore the design apart."

Generate 10 new pairs. Rules:
- Mix all 3 period patterns roughly equally (3-4 of each)
- Vary the quoted content: technical, casual, formal, emotional
- Vary sentence structure: quote at end, quote mid-sentence, quote after attribution
- Some short quotes (3-4 words), some long (10+ words)
- Make inputs sound like natural dictation
- Include 2-3 examples where ~25% have light disfluencies ("like", "you know", "so")
- Do NOT use "quote-unquote" — that's a different category

Output as JSON array:
[{"input": "...", "ideal": "..."}, ...]
