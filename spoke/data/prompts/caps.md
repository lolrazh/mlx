You are generating training data for a small LLM that post-processes dictation transcripts. The model receives raw ASR output and must execute embedded verbal commands — NOT explain them, NOT chat about them — just output the cleaned text.

Task: Generate diverse (input, ideal_output) pairs for the category: CASE TRANSFORMATION

The speaker requests a case change on their text. The model must apply the case change and remove the verbal instruction.

## Sub-types

### 1. All caps (whole sentence)
- "use all caps", "in caps", "write that in caps", "capitalize everything", "make it uppercase"
- Entire output is UPPERCASE

Input: "The meeting is cancelled. Put that in all caps."
Output: "THE MEETING IS CANCELLED."

Input: "I was thinking about the proposal and honestly I think we should reject it. Use all caps."
Output: "I WAS THINKING ABOUT THE PROPOSAL AND HONESTLY I THINK WE SHOULD REJECT IT."

### 2. All lowercase (whole sentence)
- "type that in lowercase", "make it all lowercase", "lowercase please"
- Entire output is lowercase (including proper nouns, start of sentence)

Input: "The Quick Brown Fox Jumps Over The Lazy Dog. Type this in lowercase."
Output: "the quick brown fox jumps over the lazy dog."

Input: "The meeting is at 3pm. Make it all lowercase."
Output: "the meeting is at 3pm."

### 3. Caps + excitement
- Combining caps with "show excitement" → caps + "!" instead of "."

Input: "The deadline is tomorrow. Show excitement and also put it in all caps."
Output: "THE DEADLINE IS TOMORROW!"

## Important notes
- The instruction to change case must be REMOVED from the output
- Numbers and special characters stay as-is (3pm stays 3pm, not 3PM)
- If the instruction says "make it all lowercase", even proper nouns and sentence starts become lowercase
- Excitement ("!") replaces the final period when requested

Generate 10 new pairs. Rules:
- Mix sub-types: ~4 all caps, ~4 lowercase, ~2 caps + excitement or other combo
- Vary sentence length: short (5 words) to long (25+ words)
- Vary content: work, casual, technical, emotional
- Make instructions sound natural and varied — don't always say "use all caps"
- Include some where the instruction comes BEFORE the content ("Put this in caps. I love this team.")
- Include 2-3 with light disfluencies ("like", "you know", "um")

Output as JSON array:
[{"input": "...", "ideal": "..."}, ...]
