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
- CRITICAL — Getting the ASR error right:
  The CORRECT word (what the letters spell out) should be the UNUSUAL one — a niche brand, an uncommon name, a non-standard spelling. The ASR ERROR (what appears in the input text) should be a more common-sounding phonetic approximation that ASR would plausibly output instead.

  GOOD: Input has "Celerobad" (ASR's guess), spelled S-I-L-E-R-O → "Silero" (correct, niche ML library)
  GOOD: Input has "Whisper" (common word ASR defaults to), spelled W-I-S-P-R → "Wispr" (correct brand name)
  BAD: Input has "Viterbi" (already correct), spelled W-I-T-E-R-B-Y → "Witerby" (nonsense)
  BAD: Input has "Kubernetes" (common, ASR knows this), any spelling → forced error

  Focus on words ASR would genuinely struggle with:
  - Niche startups or brand names with unusual spellings
  - Personal names from other languages
  - Made-up words, codenames, product names
  - Non-English food, place names, or cultural terms with non-obvious transliterations
  - Portmanteau brand names (like Wispr, Supabase)
- Include some where only PART of a compound name gets respelled
- Make them progressively harder

Output as JSON array:
[{"input": "...", "ideal": "..."}, ...]
