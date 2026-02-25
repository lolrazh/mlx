"""Step 5: Train the GPT. Same loop as MNIST, bigger model."""

import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np
from model import GPT

# --- Data setup ---

with open("shakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

data = mx.array([char_to_idx[ch] for ch in text])

# Split: 90% train, 10% val
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

# --- Hyperparameters ---

block_size = 128    # context window (how many characters the model sees at once)
batch_size = 32     # how many sequences per training step
n_embd = 64         # embedding dimension
n_heads = 4         # attention heads
n_layers = 4        # transformer blocks
learning_rate = 3e-4
max_steps = 1000

# --- Model ---

model = GPT(vocab_size, n_embd=n_embd, n_heads=n_heads, n_layers=n_layers, block_size=block_size)
mx.eval(model.parameters())

n_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
print(f"Model: {n_params:,} parameters")

optimizer = optim.Adam(learning_rate=learning_rate)


def get_batch(split_data):
    """Grab a random batch of sequences from the data."""
    starts = np.random.randint(0, len(split_data) - block_size, size=batch_size)
    x = mx.stack([split_data[int(s):int(s) + block_size] for s in starts])       # input
    y = mx.stack([split_data[int(s) + 1:int(s) + block_size + 1] for s in starts])  # target (shifted by 1)
    return x, y


def loss_fn(model, x, y):
    """Cross-entropy loss, same as MNIST but across a sequence."""
    logits = model(x)                       # (batch, seq_len, vocab_size)
    logits = logits.reshape(-1, vocab_size)  # flatten to (batch*seq_len, vocab_size)
    y = y.reshape(-1)                        # flatten to (batch*seq_len,)
    return nn.losses.cross_entropy(logits, y, reduction="mean")


loss_and_grad = nn.value_and_grad(model, loss_fn)


def generate(prompt, length=100):
    """Generate text from a prompt."""
    ids = [char_to_idx[ch] for ch in prompt]
    for _ in range(length):
        x = mx.array([ids[-block_size:]])
        logits = model(x)
        # Sample from the distribution instead of always picking the highest
        probs = mx.softmax(logits[0, -1, :], axis=-1)
        next_id = mx.random.categorical(mx.log(probs)).item()
        ids.append(next_id)
    return "".join(idx_to_char[i] for i in ids)


# --- Training loop ---

print(f"\nTraining for {max_steps} steps...\n")

for step in range(max_steps):
    x, y = get_batch(train_data)
    loss, grads = loss_and_grad(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.state)

    if step % 100 == 0:
        # Check validation loss
        vx, vy = get_batch(val_data)
        val_loss = loss_fn(model, vx, vy).item()
        print(f"step {step:4d}  train_loss: {loss.item():.3f}  val_loss: {val_loss:.3f}")

    if step % 500 == 0:
        print(f"\n--- Sample at step {step} ---")
        print(generate("The ", 200))
        print("---\n")

# Final generation
print(f"\nstep {max_steps}  train_loss: {loss.item():.3f}")
print(f"\n=== Final generation ===\n")
print(generate("ROMEO:\n", 300))
