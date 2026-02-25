"""Step 3: Self-attention — how characters look at each other."""

import mlx.core as mx
import mlx.nn as nn

# Let's work with a tiny example: 4 characters, embedding dim of 8
# Imagine these are the embeddings for "t", "o", " ", "b" after step 2

seq_len = 4
n_embd = 8
x = mx.random.normal((1, seq_len, n_embd))  # (1 batch, 4 chars, 8 dims)

# === Q, K, V — three projections of the same input ===
#
# Each is a Linear layer (matmul). Same input, three different weight matrices,
# three different outputs. Each "asks a different question" about the data.

head_size = 8
q_proj = nn.Linear(n_embd, head_size)  # "what am I looking for?"
k_proj = nn.Linear(n_embd, head_size)  # "what do I contain?"
v_proj = nn.Linear(n_embd, head_size)  # "what info do I carry?"

Q = q_proj(x)  # (1, 4, 8)
K = k_proj(x)  # (1, 4, 8)
V = v_proj(x)  # (1, 4, 8)

print("=== Q, K, V ===")
print(f"Input shape:  {x.shape}   (1 batch, 4 chars, 8 dims)")
print(f"Q shape:      {Q.shape}   (1 batch, 4 chars, 8 dims)")
print(f"K shape:      {K.shape}")
print(f"V shape:      {V.shape}")
print("Each character now has 3 separate vectors, produced by 3 separate matmuls.")

# === Attention scores: Q dot K ===
#
# Each character's Q gets compared against every character's K.
# High score = "that character is relevant to me."

scores = Q @ K.transpose(0, 2, 1)  # (1, 4, 8) @ (1, 8, 4) → (1, 4, 4)
scores = scores / (head_size ** 0.5)  # scale down to prevent extreme values

print(f"\n=== Attention scores ===")
print(f"Shape: {scores.shape}  (4x4 — every char scored against every other char)")
print(f"Raw scores:\n{scores[0]}")

# === Causal mask: can't look at the future ===
#
# Row 0 (first char) can only see column 0 (itself)
# Row 1 (second char) can see columns 0-1
# Row 2 can see 0-2, Row 3 can see 0-3
# Everything else gets set to -infinity so softmax turns it into 0

mask = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)
scores = scores + mask

print(f"\nCausal mask (upper triangle = -inf):\n{mask}")
print(f"Masked scores:\n{scores[0]}")
print("Characters can only attend to themselves and what came before.")

# === Softmax: turn scores into weights that sum to 1 ===

weights = mx.softmax(scores, axis=-1)

print(f"\n=== Attention weights (after softmax) ===")
print(f"{weights[0]}")
print("Each row sums to 1. These are 'how much to pay attention to each character.'")
print(f"Row sums: {weights[0].sum(axis=-1)}")

# === Final output: weighted sum of V ===
#
# Each character collects information from other characters,
# weighted by how relevant they are.

output = weights @ V  # (1, 4, 4) @ (1, 4, 8) → (1, 4, 8)

print(f"\n=== Output ===")
print(f"Shape: {output.shape}  (same as input — but now each char knows about the others)")
print(f"\nBefore attention: each character only knew about itself.")
print(f"After attention: each character has gathered info from relevant past characters.")
print(f"That's the whole mechanism.")
