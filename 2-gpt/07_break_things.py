"""Step 7: Break things. Remove pieces and see what happens.

Run each experiment one at a time by uncommenting the one you want.
Each trains for 500 steps (fast) so you can compare.
"""

import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np
from model import GPT, TransformerBlock, Attention, MultiHeadAttention, MLP

# --- Data (same as always) ---

with open("shakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

data = mx.array([char_to_idx[ch] for ch in text])
split = int(0.9 * len(data))
train_data = data[:split]

block_size = 128
batch_size = 32


def get_batch():
    starts = np.random.randint(0, len(train_data) - block_size, size=batch_size)
    x = mx.stack([train_data[int(s):int(s) + block_size] for s in starts])
    y = mx.stack([train_data[int(s) + 1:int(s) + block_size + 1] for s in starts])
    return x, y


def train_and_test(model, label, steps=500):
    """Train any model for `steps` steps and show results."""
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    optimizer = optim.Adam(learning_rate=3e-4)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {label}")
    print(f"Parameters: {n_params:,}")
    print(f"{'='*60}")

    tic = time.perf_counter()
    for step in range(steps):
        x, y = get_batch()
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.state)

        if step % 100 == 0:
            print(f"  step {step:3d}  loss: {loss.item():.3f}")

    print(f"  final loss: {loss.item():.3f}  ({time.perf_counter()-tic:.1f}s)")

    # Generate
    ids = [char_to_idx[ch] for ch in "The "]
    for _ in range(150):
        inp = mx.array([ids[-block_size:]])
        logits = model(inp)
        probs = mx.softmax(logits[0, -1, :], axis=-1)
        next_id = mx.random.categorical(mx.log(probs)).item()
        ids.append(next_id)
    print(f"  Sample: {''.join(idx_to_char[i] for i in ids)}")


def loss_fn(model, x, y):
    logits = model(x)
    logits = logits.reshape(-1, vocab_size)
    y = y.reshape(-1)
    return nn.losses.cross_entropy(logits, y, reduction="mean")


# =============================================================
# EXPERIMENT 1: Normal model (baseline)
# =============================================================
print("\n" + "#"*60)
print("# BASELINE vs BROKEN MODELS")
print("#"*60)

train_and_test(
    GPT(vocab_size, n_embd=64, n_heads=4, n_layers=4, block_size=block_size),
    "Normal GPT (baseline)"
)


# =============================================================
# EXPERIMENT 2: No positional embeddings
# What happens when the model can't tell WHERE characters are?
# =============================================================

class GPT_NoPosition(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, 64)
        # NO pos_emb!
        self.blocks = [TransformerBlock(64, 4) for _ in range(4)]
        self.ln_f = nn.LayerNorm(64)
        self.head = nn.Linear(64, vocab_size)

    def __call__(self, x):
        x = self.tok_emb(x)  # no position info added
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

train_and_test(GPT_NoPosition(), "NO positional embeddings")


# =============================================================
# EXPERIMENT 3: No attention (just MLP)
# What happens when characters can't see each other?
# =============================================================

class Block_NoAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(64)
        self.mlp = MLP(64)

    def __call__(self, x):
        return x + self.mlp(self.ln(x))

class GPT_NoAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, 64)
        self.pos_emb = nn.Embedding(block_size, 64)
        self.blocks = [Block_NoAttention() for _ in range(4)]
        self.ln_f = nn.LayerNorm(64)
        self.head = nn.Linear(64, vocab_size)

    def __call__(self, x):
        seq_len = x.shape[1]
        x = self.tok_emb(x) + self.pos_emb(mx.arange(seq_len))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

train_and_test(GPT_NoAttention(), "NO attention (MLP only)")


# =============================================================
# EXPERIMENT 4: Only 1 layer (very shallow)
# =============================================================

train_and_test(
    GPT(vocab_size, n_embd=64, n_heads=4, n_layers=1, block_size=block_size),
    "Only 1 transformer block (was 4)"
)


# =============================================================
# Summary
# =============================================================
print("\n" + "="*60)
print("COMPARE: which loss was lowest? That tells you what matters most.")
print("="*60)
