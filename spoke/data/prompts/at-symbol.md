You are generating training data for a small LLM that post-processes dictation transcripts. The model receives raw ASR output and must execute embedded verbal commands — NOT explain them, NOT chat about them — just output the cleaned text.

Task: Generate diverse (input, ideal_output) pairs for the category: AT-SYMBOL INSERTION

In this category, the speaker verbally asks to insert an @ symbol before a word. The model must:
1. Identify the @ insertion trigger ("at symbol before X", "tag X with an at symbol", "add an at symbol before X")
2. Insert @ before the target word
3. Remove the verbal instruction from the output
4. Keep everything else intact

Here are real examples:

Input: "You can see that in our worker, add an at symbol before worker."
Output: "You can see that in our @worker."

Input: "I wanna fix app.py and test.py. Add an at symbol before the file names."
Output: "I wanna fix @app.py and @test.py."

Input: "Hey mom, this app is really cool. Tag mom with an at symbol."
Output: "Hey @mom, this app is really cool."

Input: "Look mom, no hands. Tag mom with an at symbol. And show excitement."
Output: "Look @mom, no hands!"

Generate 10 new pairs. Rules:
- Vary the trigger phrasing: "add an at symbol before", "tag X with an at symbol", "at symbol before", "put an at before"
- Vary the context: code mentions (@decorator, @username), social media (@handle), Slack/Discord (@channel)
- Include cases with MULTIPLE @ insertions in one sentence
- Include cases where @ is applied to a word that already appeared earlier in the sentence
- Vary sentence length and complexity
- The instruction should feel natural, like someone dictating
- Make them progressively harder

Output as JSON array:
[{"input": "...", "ideal": "..."}, ...]