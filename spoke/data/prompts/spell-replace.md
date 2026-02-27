You are generating training data for a small LLM that post-processes dictation transcripts. The model receives raw ASR output and must execute embedded verbal commands — NOT explain them, NOT chat about them — just output the cleaned text.

Task: Generate diverse (input, ideal_output) pairs for the category: SPELL-AND-REPLACE

In this category, the speaker dictates a word (often misheard by ASR), then spells out the correct version letter by letter. The model must:
1. Find the misspelled/misheard word
2. Replace it with the spelled-out version
3. Remove the spelling instruction from the output
4. Keep everything else intact

Here are real examples:

Input: "I'm gonna be using Celero VAD for this. Can you spell that as S-I-L-E-R-O?"
Output: "I'm gonna be using Silero VAD for this."

Input: "He went to the Khadai, spell that K-A-D-A-I."
Output: "He went to the Kadai."

Input: "It's basically a competitor to Aqua Voice, Willow Voice, and Whisper Flow. Can you spell Whisper as W-I-S-P-R?"
Output: "It's basically a competitor to Aqua Voice, Willow Voice, and Wispr Flow."

Input: "Jor-bill, spell that J-O-R-B-L-E"
Output: "Jorble"

Input: "She runs a studio called Monazo. Spell that M-O-N-O-U-Z-O."
Output: "She runs a studio called Monouzo."

Generate 10 new pairs. Rules:
- Vary the topic (tech, food, places, people, brands, made-up words)
- Vary sentence length (short and long)
- Vary where the spell command appears (end, middle, after the word)
- Vary the phrasing ("spell that as", "it's spelled", "spell X as", etc.)
- The misspelled word should sound plausible as an ASR error
- Include some where only PART of a compound name gets respelled
- Make them progressively harder

Output as JSON array:
[{"input": "...", "ideal": "..."}, ...]
