You are generating training data for a small LLM that post-processes dictation transcripts. The model receives raw ASR output and must execute embedded verbal commands — NOT explain them, NOT chat about them — just output the cleaned text.

Task: Generate diverse (input, ideal_output) pairs for the category: EMAIL/URL DICTATION

In this category, the speaker dictates an email address or URL verbally, using phrases like "at gmail dot com", "dot org", etc. The model must:
1. Identify the verbally dictated email/URL
2. Convert it to proper format (user@domain.com, website.com)
3. Remove any meta-commentary about the spelling
4. Keep the surrounding sentence intact

Here is a real example:

Input: "Ping marketing on this—add an at symbol before marketing—and email rajkumar dot sandheep at gmail dot com, sorry, rajkumar.sandheep@gmail.com, saying quote launch moved to Friday end quote. Actually, Thursday."
Output: "Ping @marketing on this and email rajkumar.sandheep@gmail.com, saying "launch moved to Thursday.""

Generate 10 new pairs. Rules:
- Vary the email format: firstname.lastname@, firstname@, initials@, username@
- Vary the domain: gmail.com, outlook.com, company domains, .org, .io, .dev
- Include some URL dictation too: "check out example dot com slash docs"
- Vary how the speaker dictates: "at gmail dot com", "at sign gmail period com"
- Include cases where the speaker corrects part of the email ("sorry, not sandheep, sandeep")
- Include cases where the email is embedded in a longer sentence
- Some should be short and simple, others long and complex
- The input should sound like natural dictated speech
- Make them progressively harder

Output as JSON array:
[{"input": "...", "ideal": "..."}, ...]