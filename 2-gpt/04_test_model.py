"""Step 4: Run the model before training — verify it works, see the shapes."""

import mlx.core as mx
import mlx.utils
from model import GPT

with open("shakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Build the model
model = GPT(vocab_size, n_embd=64, n_heads=4, n_layers=4, block_size=128)
mx.eval(model.parameters())

# Count parameters
n_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
print(f"Model parameters: {n_params:,}")
print(f"(GPT-2 Small has 124M. We have {n_params/1000:.0f}K. Same architecture, tiny scale.)")

# Feed it a sentence
sentence = "to be or "
token_ids = mx.array([[char_to_idx[ch] for ch in sentence]])
print(f"\nInput: '{sentence}'")
print(f"Token IDs shape: {token_ids.shape}")

logits = model(token_ids)
print(f"Output shape: {logits.shape}  (1 batch, {len(sentence)} positions, {vocab_size} scores each)")

# What does the model predict for the LAST character?
last_scores = logits[0, -1, :]  # scores for next char after "to be or "
predicted_id = mx.argmax(last_scores).item()
print(f"\nAfter 'to be or ', model predicts: '{idx_to_char[predicted_id]}'")
print(f"(Random garbage — hasn't trained yet)")

# Generate a few characters greedily
generated = list(sentence)
input_ids = [char_to_idx[ch] for ch in sentence]

for _ in range(40):
    x = mx.array([input_ids[-128:]])  # only last 128 chars (block_size)
    logits = model(x)
    next_id = mx.argmax(logits[0, -1, :]).item()
    generated.append(idx_to_char[next_id])
    input_ids.append(next_id)

print(f"\nGenerated (untrained): '{''.join(generated)}'")
print("Pure noise. After training, this will look like Shakespeare.")
