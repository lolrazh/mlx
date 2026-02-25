"""Step 6: Bigger model, longer training. Let's see what it can do."""

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
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

# --- Scaled up ---

block_size = 256    # 2x context window (sees more history)
batch_size = 32
n_embd = 128        # 2x embedding dim (richer representations)
n_heads = 8         # 2x heads (more ways to attend)
n_layers = 6        # 1.5x depth (more transformations)
learning_rate = 3e-4
max_steps = 3000

# --- Model ---

model = GPT(vocab_size, n_embd=n_embd, n_heads=n_heads, n_layers=n_layers, block_size=block_size)
mx.eval(model.parameters())

n_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
print(f"Model: {n_params:,} parameters (was 216K, now {n_params/1000:.0f}K)")

optimizer = optim.Adam(learning_rate=learning_rate)


def get_batch(split_data):
    starts = np.random.randint(0, len(split_data) - block_size, size=batch_size)
    x = mx.stack([split_data[int(s):int(s) + block_size] for s in starts])
    y = mx.stack([split_data[int(s) + 1:int(s) + block_size + 1] for s in starts])
    return x, y


def loss_fn(model, x, y):
    logits = model(x)
    logits = logits.reshape(-1, vocab_size)
    y = y.reshape(-1)
    return nn.losses.cross_entropy(logits, y, reduction="mean")


loss_and_grad = nn.value_and_grad(model, loss_fn)


def generate(prompt, length=200):
    ids = [char_to_idx[ch] for ch in prompt]
    for _ in range(length):
        x = mx.array([ids[-block_size:]])
        logits = model(x)
        probs = mx.softmax(logits[0, -1, :], axis=-1)
        next_id = mx.random.categorical(mx.log(probs)).item()
        ids.append(next_id)
    return "".join(idx_to_char[i] for i in ids)


# --- Training ---

print(f"Training for {max_steps} steps...\n")
start_time = time.perf_counter()

for step in range(max_steps + 1):
    x, y = get_batch(train_data)
    loss, grads = loss_and_grad(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.state)

    if step % 500 == 0:
        vx, vy = get_batch(val_data)
        val_loss = loss_fn(model, vx, vy).item()
        elapsed = time.perf_counter() - start_time
        print(f"step {step:4d}  train: {loss.item():.3f}  val: {val_loss:.3f}  ({elapsed:.1f}s)")

    if step in (0, 500, 1500, 3000):
        print(f"\n--- Sample at step {step} ---")
        print(generate("ROMEO:\n", 300))
        print("---\n")

total = time.perf_counter() - start_time
print(f"\nDone in {total:.1f}s")
