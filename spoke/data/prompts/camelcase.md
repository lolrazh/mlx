You are generating training data for a small LLM that post-processes dictation transcripts. The model receives raw ASR output and must execute embedded verbal commands — NOT explain them, NOT chat about them — just output the cleaned text.

Task: Generate diverse (input, ideal_output) pairs for the category: CAMELCASE RESTORATION

ASR (speech-to-text) transcribes everything in lowercase. When the speaker mentions code identifiers — React hooks, component names, class names, function names — the model must restore the correct camelCase or PascalCase.

## Patterns

### 1. React hooks (use + CamelCase)
- "usestate" → "useState"
- "useeffect" → "useEffect"
- "usecallback" → "useCallback"
- Custom hooks too: "usetranscription" → "useTranscription", "useauthcontext" → "useAuthContext"

### 2. PascalCase components / classes
- "appcontainer" → "AppContainer"
- "errorboundary" → "ErrorBoundary"
- "defaultlayout" → "DefaultLayout"

### 3. camelCase functions / variables
- "handlesubmit" → "handleSubmit"
- "getusername" → "getUserName"
- "calculatetokenoverlap" → "calculateTokenOverlap"

### 4. Framework-specific
- "swiftui" → "SwiftUI"
- "viewcontroller" → "ViewController"
- "nextjs" → "Next.js" (only if speaker clearly means the framework)

## Real examples

Input: "So in my React component, I'm calling usestate to manage the form data, but I keep forgetting to import it properly."
Output: "So in my React component, I'm calling useState to manage the form data, but I keep forgetting to import it properly."

Input: "The main wrapper is called appcontainer and it handles all the routing logic between the different views."
Output: "The main wrapper is called AppContainer and it handles all the routing logic between the different views."

Input: "I think we should rename handlesubmit to something clearer since we now have both handlesubmit and handleformsubmit in the same file."
Output: "I think we should rename handleSubmit to something clearer since we now have both handleSubmit and handleFormSubmit in the same file."

Input: "So I'm looking at this usememo hook and it's not memoizing the llm inference results even though the gpu is available."
Output: "So I'm looking at this useMemo hook and it's not memoizing the LLM inference results even though the GPU is available."

Generate 10 new pairs. Rules:
- Mix patterns: ~3 React hooks (include custom hooks, not just standard ones), ~3 PascalCase, ~3 camelCase functions, ~1 framework-specific
- Custom hooks are CRITICAL: "useaudiorecorder" → "useAudioRecorder", "usedarkmode" → "useDarkMode" — the model must generalize the use+PascalCase pattern to NOVEL hook names
- Include identifiers with 2, 3, and 4+ word compounds
- Vary surrounding context: casual dev chat, code review, debugging, teaching
- Make inputs sound like natural speech — ASR flattens everything to lowercase
- Include 2-3 examples where the same identifier appears multiple times in the sentence
- Do NOT include tech acronyms (LLM, GPU, API) or file paths (package.json) — that's a different category

Output as JSON array:
[{"input": "...", "ideal": "..."}, ...]
