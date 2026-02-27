You are generating training data for a small LLM that post-processes dictation transcripts. The model receives raw ASR output and must execute embedded verbal commands — NOT explain them, NOT chat about them — just output the cleaned text.

Task: Generate diverse (input, ideal_output) pairs for the category: CODE-AWARE CORRECTIONS

In this category, ASR mis-transcribes technical terms, filenames, or code identifiers. The model must recognize the technical context and fix the formatting. Common patterns:
1. CamelCase restoration: "usetranscription" → "useTranscription"
2. Technical term casing: "mchips" → "M chips", "onnx" → "ONNX"
3. Filename formatting: "dev, vars" → "dev.vars"
4. Code identifier patterns: recognizing when a flat word should be camelCase/PascalCase

Here are real examples:

Input: "So we've got a file called usetranscription.ts."
Output: "So we've got a file called useTranscription.ts."

Input: "And the thing is, this is only a macOS app and only for mchips as well."
Output: "And the thing is, this is only a macOS app and only for M chips as well."

Input: "And I have my secret variables also set up in the Cloudflare dashboard. So I'll just change dev, vars and the secret in the Cloudflare dashboard as well."
Output: "And I have my secret variables also set up in the Cloudflare dashboard. So I'll just change dev.vars and the secret in the Cloudflare dashboard as well."

Generate 10 new pairs. Rules:
- Cover different code-aware patterns:
  - camelCase hooks: "usestate" → "useState", "useeffect" → "useEffect"
  - PascalCase components: "appcontainer" → "AppContainer"
  - Tech acronyms: "llm" → "LLM", "api" → "API", "css" → "CSS", "gpu" → "GPU"
  - File references: "package dot json" → "package.json", "tsconfig" → "tsconfig"
  - Dotted paths: "next, config" → "next.config", "env, local" → ".env.local"
- The input should sound like natural dictation where ASR flattened the casing or punctuation
- Vary the surrounding sentence context (casual, technical, instructional)
- Include some where the tech term is embedded in a longer explanation
- Do NOT include any spelling commands (X-X-X) — that's a different category
- Make them progressively harder

Output as JSON array:
[{"input": "...", "ideal": "..."}, ...]