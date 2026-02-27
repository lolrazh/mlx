You are generating training data for a small LLM that post-processes dictation transcripts. The model receives raw ASR output and must execute embedded verbal commands — NOT explain them, NOT chat about them — just output the cleaned text.

Task: Generate diverse (input, ideal_output) pairs for the category: EMOJI

In this category, the speaker verbally names an emoji. The model must:
1. Identify the emoji reference
2. Replace it with the actual emoji character
3. If the input is ONLY an emoji name, output ONLY the emoji

Here are real examples:

Input: "Two hearts"
Output: "❤️❤️"

Input: "Crying emoji"
Output: "😢"

Input: "Heartbreak emoji"
Output: "💔"

Input: "Broken heart emoji"
Output: "💔"

Generate 10 new pairs. Rules:
- Cover common emoji categories: faces (😊😂🤔😭), hands (👍👏🙏✌️), objects (🔥⭐💯🎉), hearts (❤️💔💜), animals (🐶🐱), food (🍕🍔)
- Vary the phrasing: "X emoji", just "X" (like "Two hearts"), "give me a X emoji"
- Include cases with MULTIPLE emojis: "Three fire emojis" → "🔥🔥🔥"
- Include cases where the emoji is embedded in a sentence: "That's amazing fire emoji" → "That's amazing 🔥"
- Include commonly confused emoji: "crying" (😢) vs "crying laughing" (😂)
- The word "emoji" may or may not appear in the input
- Vary between standalone emoji (input is just the emoji name) and inline emoji (within a sentence)
- Make them progressively harder

Output as JSON array:
[{"input": "...", "ideal": "..."}, ...]