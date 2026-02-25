"""Step 1: Load Shakespeare and build a character-level tokenizer."""

# In MNIST, input was already numbers (pixel values).
# Text isn't numbers. We need to convert characters → numbers.
# That's what a tokenizer does.
#
# Real GPT uses ~50,000 "tokens" (words and word-pieces).
# We'll use individual characters (~65 unique chars). Simpler, same idea.

with open("shakespeare.txt", "r") as f:
    text = f.read()

print(f"Dataset: {len(text):,} characters")
print(f"First 200 chars:\n")
print(text[:200])

# Build vocabulary: every unique character in the text
chars = sorted(set(text))
vocab_size = len(chars)
print(f"\n--- Vocabulary ({vocab_size} characters) ---")
print(repr("".join(chars)))

# Tokenizer: two lookup tables, that's all it is
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Encode: text → numbers
sample = "hello"
encoded = [char_to_idx[ch] for ch in sample]
print(f"\nEncode '{sample}' → {encoded}")

# Decode: numbers → text
decoded = "".join(idx_to_char[i] for i in encoded)
print(f"Decode {encoded} → '{decoded}'")

print(f"\nThis is the entire tokenizer. Character in, number out.")
print(f"The model will never see letters — only these numbers.")
