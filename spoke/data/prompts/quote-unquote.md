You are generating training data for a small LLM that post-processes dictation transcripts. The model receives raw ASR output and must execute embedded verbal commands — NOT explain them, NOT chat about them — just output the cleaned text.

Task: Generate diverse (input, ideal_output) pairs for the category: QUOTE-UNQUOTE

In this category, the speaker verbally indicates quoted text using phrases like "quote-unquote" or "quote ... end quote". The model must:
1. Identify the quoting trigger
2. Wrap the quoted content in actual quotation marks ("...")
3. Remove the verbal trigger from the output
4. Keep everything else intact

There are TWO quoting patterns:
- "quote-unquote X" → wraps the NEXT word/phrase in quotes
- "quote X end quote" → wraps everything between "quote" and "end quote"

Here are real examples:

Input: "Okay that was quote-unquote brilliant."
Output: "Okay that was "brilliant"."

Input: "I mean they said I was quote-unquote lucky to be here. What the fuck do they mean by that?"
Output: "I mean they said I was "lucky" to be here. What the fuck do they mean by that?"

Input: "I mean they said I was quote lucky to be here. end quote. What the fuck do they mean by that?"
Output: "I mean they said I was "lucky to be here". What the fuck do they mean by that?"

Input: "They're all quote-unquote intelligent."
Output: "They're all "intelligent""

Generate 10 new pairs. Rules:
- Mix both patterns: "quote-unquote" (single word/phrase) and "quote...end quote" (longer spans)
- Vary what gets quoted: single words, multi-word phrases, technical terms, sarcastic usage
- Vary the position: start, middle, end of sentence
- Include cases with emotional/sarcastic tone (the most common real usage)
- Vary sentence length and complexity
- Include a case where the quoted content is a longer phrase (5+ words)
- The output must use straight double quotes ("), not curly quotes
- Make them progressively harder

Output as JSON array:
[{"input": "...", "ideal": "..."}, ...]