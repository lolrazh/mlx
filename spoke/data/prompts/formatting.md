You are generating training data for a small LLM that post-processes dictation transcripts. The model receives raw ASR output and must execute embedded verbal commands — NOT explain them, NOT chat about them — just output the cleaned text.

Task: Generate diverse (input, ideal_output) pairs for the category: FORMATTING COMMANDS

In this category, the speaker requests text formatting: ALL CAPS, lowercase, **bold/emphasis**, or excitement (!). The model must:
1. Identify the formatting command
2. Apply the requested formatting to the correct text
3. Remove the formatting instruction from the output
4. Keep everything else intact

Formatting types:
- ALL CAPS: "use all caps", "in caps", "write that in uppercase", "in capital letters"
- lowercase: "type that in lowercase", "can you type that in lowercase"
- Bold/emphasis: "emphasize X", "bold X", "emphasis on X"
- Selective caps: "emphasis on why and how" → "WHY" and "HOW" get caps
- Excitement: "show excitement" → adds "!" or removes period

Here are real examples:

Input: "We're so excited to make this. Use all caps."
Output: "WE'RE SO EXCITED TO MAKE THIS."

Input: "It's surprisingly fast. Emphasize surprisingly."
Output: "It's **surprisingly** fast."

Input: "And I think instead of talking about the features, we talk about how Sonic Flow actually helps, how Sonic Flow actually behaves, why Sonic Flow is better than the other apps and stuff. Emphasis on why and how."
Output: "And I think instead of talking about the features, we talk about HOW Sonic Flow actually helps, HOW Sonic Flow actually behaves, WHY Sonic Flow is better than the other apps and stuff."

Input: "So, yeah, my food's about to come. And yeah, the delivery guy, actually, no, the customer service said 10 more minutes. But yeah, I hope he's not lying. Can you type that in lowercase?"
Output: "so, yeah, my food's about to come. and yeah, the customer service said 10 more minutes. but yeah, i hope he's not lying."

Input: "Look mom, no hands. Tag mom with an at symbol. And show excitement."
Output: "Look @mom, no hands!"

Generate 10 new pairs. Rules:
- Cover ALL formatting types: caps, lowercase, bold/emphasis, selective caps, excitement
- For caps: the ENTIRE preceding text gets uppercased
- For lowercase: the ENTIRE preceding text gets lowercased
- For emphasis: only the specified word(s) get **bold** or CAPS
- For selective caps: specific words get uppercased (like "emphasis on X and Y")
- Vary sentence length from short (5 words) to long (30+ words)
- The formatting instruction should sound natural ("write that in all caps", "make it bold", etc.)
- Include 1-2 examples where formatting is combined with another operation (like self-correction + formatting in the food example above)
- Make them progressively harder

Output as JSON array:
[{"input": "...", "ideal": "..."}, ...]