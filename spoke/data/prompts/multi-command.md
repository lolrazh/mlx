You are generating training data for a small LLM that post-processes dictation transcripts. The model receives raw ASR output and must execute embedded verbal commands — NOT explain them, NOT chat about them — just output the cleaned text.

Task: Generate diverse (input, ideal_output) pairs for the category: MULTI-COMMAND

In this category, the speaker uses TWO OR MORE different operations in a single utterance. The model must execute ALL of them correctly. Operations include:
- Spell-and-replace: "spell that X-X-X" (ASR misheard word → spelled correction)
- Self-correction: "wait no, sorry, actually" (remove mistake, keep correction)
- Quote-unquote: "quote-unquote X" → "X"
- At-symbol: "add an at symbol before X" → @X
- Email dictation: "at gmail dot com" → @gmail.com
- Formatting: "in caps", "emphasize X", "lowercase"
- Emoji: verbal emoji name → actual emoji

Here are real examples:

Input: "Ping marketing on this. Add an at symbol before marketing. Say quote launch moved to Friday end quote. Actually, Thursday."
Output: "Ping @marketing on this. Say "launch moved to Thursday.""
(Operations: @-symbol + quote-unquote + self-correction)

Input: "Ping marketing on this—add an at symbol before marketing—and email rajkumar dot sandheep at gmail dot com, sorry, rajkumar.sandheep@gmail.com, saying quote launch moved to Friday end quote. Actually, Thursday."
Output: "Ping @marketing on this and email rajkumar.sandheep@gmail.com, saying "launch moved to Thursday.""
(Operations: @-symbol + email + quote-unquote + self-correction)

Input: "Send this to Groq. Add an at symbol before Groq. The filename is quote sonicflow_superbase-handler end quote. Spell superbase as S-U-P-A-B-A-S-E, split the CamelCase; sorry, replace supabase with vercel, V-E-R-C-E-L."
Output: "Send this to @Groq. The filename is "sonicflow_vercel-handler.""
(Operations: @-symbol + quote-unquote + spell-replace + self-correction + code-aware)

Input: "So, there's the clod.md file. It's spelled C-L-A-U-D-E, in caps."
Output: "So, there's the CLAUDE.md file."
(Operations: spell-replace + formatting/caps)

Generate 10 new pairs. Rules:
- EVERY example must combine 2+ DIFFERENT operation types
- Start with 2-operation combos, build up to 3-4 operations
- Common combos to include:
  - spell + self-correction (user misspells the spelling, then corrects)
  - @-symbol + quote-unquote (messaging context)
  - formatting + self-correction (change format AND fix a mistake)
  - emoji + formatting (emoji in formatted text)
  - spell + @-symbol (spell a handle, then @ it)
- ALL operations must be correctly executed in the ideal output
- The input should sound like natural continuous speech, not artificial
- Label which operations are present in a comment (but still output valid JSON)
- Make them progressively harder (2 ops → 3 ops → 4+ ops)

Output as JSON array:
[{"input": "...", "ideal": "...", "ops": ["op1", "op2", ...]}, ...]