These are training examples for an ASR post-processing model. Each has a specific error that needs fixing.

IMPORTANT CONTEXT: In these examples, a user is dictating and the ASR (speech-to-text) MISHEARS a word. The user then spells out the CORRECT version letter by letter. So the word in the input text must be DIFFERENT from what the letters spell out — it's an ASR error being corrected.

For example:
- CORRECT: Input has "Celerobad" (ASR error), user spells S-I-L-E-R-O → ideal replaces with "Silero"
- WRONG: Input has "Massaman", user spells M-A-S-S-A-M-A-N → same word, no correction happening

Fix each example below. For SAME-WORD errors, change the word in the input to a plausible ASR misheard version (something that sounds similar but is spelled wrong). For other errors, follow the specific fix instruction.

---

Example 1:
Input: "The restaurant serves authentic Pad Thai, Tom Yum, and Massaman curry. Can you spell Massaman M-A-S-S-A-M-A-N?"
Ideal: "The restaurant serves authentic Pad Thai, Tom Yum, and Massaman curry."
Error: SAME-WORD — "Massaman" already appears correctly in the input. The input should have an ASR-misheard version.
Fix: Change "Massaman" (the first one, not the spelling) in the input to something like "Masaman" or "Mossaman" — a plausible ASR error. Keep the spelling and ideal the same.

Example 2:
Input: "He's reading a book by Chiamaka Adichie. It's spelled C-H-I-A-M-A-K-A Adichie, not the way I said it."
Ideal: "He's reading a book by Chiamaka Adichie."
Error: SAME-WORD — "Chiamaka" already appears correctly in the input.
Fix: Change "Chiamaka" in the input to a plausible ASR misheard version (e.g., "Chimaka" or "Chiamacha").

Example 3:
Input: "My favorite coffee shop is called Stumptown, but the one in Portland spells it S-T-U-M-P-T-O-W-N, while the Seattle location has a different vibe entirely and I prefer their roast."
Ideal: "My favorite coffee shop is called Stumptown, but the one in Portland spells it Stumptown, while the Seattle location has a different vibe entirely and I prefer their roast."
Error: SAME-WORD — "Stumptown" already correct. Also, the ideal output still contains "spells it Stumptown" which is leftover instruction text.
Fix: Change "Stumptown" in the input to an ASR error. Also fix the ideal to remove the spelling instruction — it should just read naturally.

Example 4:
Input: "We're integrating with the Twillio API for SMS, spell Twillio T-W-I-L-I-O, and also using SendGrid for email which I think is spelled normally."
Ideal: "We're integrating with the Twilio API for SMS and also using SendGrid for email which I think is spelled normally."
Error: Ideal still contains "which I think is spelled normally" — this is leftover meta-commentary that should be removed.
Fix: Remove "which I think is spelled normally" from the ideal. The input word "Twillio" vs spelled "Twilio" is actually correct (good ASR error!), so keep that.

Example 5:
Input: "She founded a startup called Brex, spell that B-R-E-X, not B-R-I-X like the building blocks, and they focus on corporate credit cards for tech companies, which is different from Stripe though people confuse them."
Ideal: "She founded a startup called Brex, not Brix like the building blocks, and they focus on corporate credit cards for tech companies, which is different from Stripe though people confuse them."
Error: SAME-WORD — "Brex" is already correct in the input. Also the ideal keeps "not Brix like the building blocks" which is conversational aside, not cleaned output.
Fix: Change the input word to an ASR error (e.g., "Brecks"). Also clean up the ideal to remove the aside.

---

Output the 5 fixed examples as a JSON array: [{"input": "...", "ideal": "..."}, ...]
