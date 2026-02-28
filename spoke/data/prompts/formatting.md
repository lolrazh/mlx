You are generating training data for a small LLM that post-processes dictation transcripts. The model receives raw ASR output and must execute embedded verbal commands — NOT explain them, NOT chat about them — just output the cleaned text.

Task: Generate diverse (input, ideal_output) pairs for the category: FORMATTING & ANNOTATION

This category covers all commands that annotate or transform how text appears — without changing its meaning. The model must identify the command, apply it, and remove the verbal instruction from the output.

## Sub-types

### 1. Case transformation
- ALL CAPS: "use all caps", "write that in caps", "in uppercase", "capitalize everything"
- Lowercase: "type that in lowercase", "make it all lowercase"
- Selective caps: "emphasis on X and Y" → just those words get CAPS

### 2. Inline emphasis
- Bold (markdown): "emphasize X", "bold X", "make X bold" → **X**
- Selective uppercase: "stress X" or "caps on X" → X becomes CAPS

### 3. Excitement / punctuation
- "show excitement", "make it excited", "add some energy" → adds "!" or changes "." to "!"

### 4. @ symbol insertion
- "add an at symbol before X", "tag X with an at symbol", "put an at before X", "at symbol before X"
- Target can be a name, a username, a filename, a Slack channel, a decorator
- Multiple @ in one sentence is valid

### 5. XML / custom tag wrapping
- "wrap that in XML tags", "put that in XML tags", "wrap X in a <Y> tag"
- "XML open Y ... XML close Y", "open tag Y ... close tag Y"
- User may say: "wrap that in a strong tag", "put it in an em tag", "surround with a div"
- Tag name comes from the user — could be any word
- The model should understand intent even with loose phrasing

## Real examples

Input: "We're so excited to make this. Use all caps."
Output: "WE'RE SO EXCITED TO MAKE THIS."

Input: "It's surprisingly fast. Emphasize surprisingly."
Output: "It's **surprisingly** fast."

Input: "And I think instead of talking about the features, we talk about how Sonic Flow actually helps, how Sonic Flow actually behaves, why Sonic Flow is better than the other apps and stuff. Emphasis on why and how."
Output: "And I think instead of talking about the features, we talk about HOW Sonic Flow actually helps, HOW Sonic Flow actually behaves, WHY Sonic Flow is better than the other apps and stuff."

Input: "So, yeah, my food's about to come. And yeah, the customer service said 10 more minutes. But yeah, I hope he's not lying. Can you type that in lowercase?"
Output: "so, yeah, my food's about to come. and yeah, the customer service said 10 more minutes. but yeah, i hope he's not lying."

Input: "You can see that in our worker, add an at symbol before worker."
Output: "You can see that in our @worker."

Input: "I wanna fix app.py and test.py. Add an at symbol before the file names."
Output: "I wanna fix @app.py and @test.py."

Input: "Hey mom, this app is really cool. Tag mom with an at symbol."
Output: "Hey @mom, this app is really cool."

Input: "Look mom, no hands. Tag mom with an at symbol. And show excitement."
Output: "Look @mom, no hands!"

Input: "The error message says unauthorized. Wrap unauthorized in XML tags."
Output: "The error message says <unauthorized>."

Input: "Use the word important here. Wrap it in a strong tag."
Output: "Use the word <strong>important</strong> here."

Input: "The status is pending. XML open status, XML close status."
Output: "The status is <status>pending</status>."

Input: "Wrap the whole sentence in a div tag. The user clicked submit."
Output: "<div>The user clicked submit.</div>"

Generate 10 new pairs. Rules:
- Cover ALL sub-types: case, emphasis, @-symbol, XML tags — roughly 2-3 examples per type
- For XML: vary the phrasing widely. Users will say "wrap in", "put in", "xml open/close", "surround with a tag", "in a X tag". Be loose and varied.
- For @: vary context — code (@decorator, @property), social (@handle), Slack (@channel, @here), filenames
- For caps/lowercase: vary whether the scope is the whole sentence or just specific words
- Vary sentence length: some short (5 words), some long (30+ words)
- Make instructions sound natural, like someone talking — not robotic
- Include 1-2 examples where formatting combines with another operation (e.g. @-symbol + excitement, caps + self-correction)

Output as JSON array:
[{"input": "...", "ideal": "..."}, ...]
