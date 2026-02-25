"""Step 2: Embeddings — turning character IDs into something the model can learn from."""

import mlx.core as mx
import mlx.nn as nn

# --- Setup (same tokenizer from step 1) ---

with open("shakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}

# --- The two embedding tables ---

n_embd = 16       # each character becomes 16 numbers
block_size = 32   # how many characters the model looks at at once (its "context window")

# Character embedding: "what character is this?"
# Shape: (65, 16) — one row of 16 numbers per character in the vocabulary
tok_emb = nn.Embedding(vocab_size, n_embd)

# Position embedding: "where in the sequence is this character?"
# Shape: (32, 16) — one row of 16 numbers per position (0 through 31)
pos_emb = nn.Embedding(block_size, n_embd)

# --- Feed in a sentence ---

sentence = "to be or not to be"
token_ids = mx.array([[char_to_idx[ch] for ch in sentence]])  # shape: (1, 18)

positions = mx.arange(len(sentence))  # [0, 1, 2, ..., 17]

char_vectors = tok_emb(token_ids)     # shape: (1, 18, 16) — each char is now 16 numbers
pos_vectors = pos_emb(positions)      # shape: (18, 16) — each position is 16 numbers

x = char_vectors + pos_vectors        # combine: "what" + "where"

print(f"Sentence: '{sentence}'")
print(f"Token IDs: {token_ids}")
print(f"\nAfter embedding:")
print(f"  char_vectors shape: {char_vectors.shape}  (1 batch, 18 chars, 16 dims each)")
print(f"  pos_vectors shape:  {pos_vectors.shape}   (18 positions, 16 dims each)")
print(f"  combined shape:     {x.shape}          (1 batch, 18 chars, 16 dims each)")
print(f"\nThe letter 't' at position 0:")
print(f"  {x[0, 0, :]}")
print(f"\nThe letter 't' at position 14 (second 'to'):")
print(f"  {x[0, 14, :]}")
print(f"\n  Same letter, different vectors — because the position is different.")
print(f"  The model can tell WHERE in the sentence each character appears.")
