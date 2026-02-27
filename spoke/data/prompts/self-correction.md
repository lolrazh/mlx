You are generating training data for a small LLM that post-processes dictation transcripts. The model receives raw ASR output and must execute embedded verbal commands — NOT explain them, NOT chat about them — just output the cleaned text.

Task: Generate diverse (input, ideal_output) pairs for the category: SELF-CORRECTION

In this category, the speaker corrects themselves mid-sentence. They say something wrong, then use a trigger phrase to indicate the correction. The model must:
1. Identify the self-correction trigger ("wait no", "sorry", "scratch that", "actually", etc.)
2. Remove the incorrect part AND the trigger phrase
3. Keep the corrected version and all other content intact
4. Handle cases where the correction replaces a word, a phrase, or an entire clause

Here are real examples:

Input: "The backend is actually powered by Cloudflare Workers and Groq. Wait no, sorry, Fireworks."
Output: "The backend is actually powered by Cloudflare Workers and Fireworks."

Input: "This is powered by AMD. Wait no, sorry, Nvidia."
Output: "This is powered by Nvidia."

Input: "Let's meet at 11am, Saturday. Actually scratch that, let's meet at 12pm, Thursday."
Output: "Let's meet at 12pm, Thursday."

Input: "It works if the endpoint for STT and LLM, sorry, not endpoint, the provider is the same."
Output: "It works if the provider for STT and LLM is the same."

Input: "The app is pretty lightweight. Actually, say really lightweight."
Output: "The app is really lightweight."

Input: "Double tapping the option key, you know, the right option key would trigger dictation."
Output: "Double tapping the right option key would trigger dictation."

Generate 10 new pairs. Rules:
- Vary the correction trigger: "wait no", "sorry", "scratch that", "actually", "I meant", "no no", "you know", "let me rephrase"
- Vary what gets corrected: a single word, a phrase, a number, a name, an entire sentence
- Vary the topic (work, food, scheduling, tech, casual conversation)
- Include cases where the correction is at the END of the sentence (replacing the last thing said)
- Include cases where the correction is in the MIDDLE (replacing something earlier)
- Include a case where someone restates something more precisely ("you know, the X one")
- The input should sound like natural speech with realistic disfluencies
- Make them progressively harder

Output as JSON array:
[{"input": "...", "ideal": "..."}, ...]