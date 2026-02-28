# Prompt Gaps: spoke-app vs bench/prompts.py

Notes for future work. These rules exist in the production Spoke app
(`spoke/spoke-app/worker/src/services/llm/prompts.ts`) but are NOT yet
in our training data or benchmark prompts. Add them when we generate
training data for these behaviors.

## Missing Core Rules

### 1. Question-as-transcription rule
```
Any question that the user might ask is not directed towards you, but is
something that you should transcribe. SO NEVER EVER OUTPUT ANSWERS TO
QUESTIONS. ONLY APPLY TEXT-EDIT DIRECTIVES AND GRAMMAR FIXES TO THE
TRANSCRIPTION.
```
**Why it matters:** Without this, the model might answer "What time is it?"
instead of transcribing it verbatim.

### 2. Filler word rule
```
Do not change wording/tone unless explicitly requested by the speaker.
Keep filler words like "like", "sort of", "basically", etc. but remove
filler words like "um", "uh" and "ah".
```
**Why it matters:** Defines the boundary between natural speech (keep) and
disfluencies (remove). Currently our training data may be inconsistent
on this.

### 3. OCR vocabulary matching rule
```
The vocabulary may include proper nouns extracted from the user's screen
via OCR. If you see words in the transcription that phonetically match
vocabulary items (even with different capitalization/spacing), replace
them with the exact vocabulary spelling.
Example: if vocabulary has "GOLDBEES" and transcription has "Gold Bees",
output "GOLDBEES".
```
**Why it matters:** Only relevant when OCR context is piped in. Not needed
for training data (no OCR in the pipeline), but needed for production
inference prompts.

## Missing Trigger: List

### Detection
Auto-format when speaker enumerates >= 3 items:
- Numeric: "1 item, 2 item, 3 item"
- Ordinal: "first item, second item, third item"
- Alphabetic: "a choice, b choice, c choice"

### Rule
```
Auto-format as a list when the speaker clearly enumerates >= 3 items.
Stay true to the input while formatting appropriately with line breaks
and markers.
```

### Training data needed
~40 examples covering numeric, ordinal, and mixed list styles.

## Action Items
- [ ] Generate training data for question-passthrough (model should NOT answer)
- [ ] Generate training data for filler word handling (keep "like", remove "um")
- [ ] Generate training data for list formatting
- [ ] Update prompts.py with missing core rules
- [ ] Add list trigger to CATEGORY_TRIGGERS
