"""The full GPT model. Every piece, assembled."""

import mlx.core as mx
import mlx.nn as nn


class Attention(nn.Module):
    """One attention head — what we built in step 3."""

    def __init__(self, n_embd, head_size):
        super().__init__()
        self.q = nn.Linear(n_embd, head_size, bias=False)
        self.k = nn.Linear(n_embd, head_size, bias=False)
        self.v = nn.Linear(n_embd, head_size, bias=False)
        self.scale = head_size ** -0.5

    def __call__(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        # Attention scores + causal mask
        scores = (Q @ K.transpose(0, 2, 1)) * self.scale
        seq_len = x.shape[1]
        mask = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)
        scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        return weights @ V


class MultiHeadAttention(nn.Module):
    """Run multiple attention heads in parallel, concatenate results.

    Like looking at the sequence through several different lenses at once.
    4 heads of size 16 each = same as 1 head of size 64, but more expressive.
    """

    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.heads = [Attention(n_embd, n_embd // n_heads) for _ in range(n_heads)]
        self.proj = nn.Linear(n_embd, n_embd)  # combine heads back together

    def __call__(self, x):
        # Run all heads, concatenate along the last dimension
        out = mx.concatenate([h(x) for h in self.heads], axis=-1)
        return self.proj(out)


class MLP(nn.Module):
    """Feed-forward network. Each character processes its own info.

    Same idea as our MNIST model — Linear → activation → Linear.
    The inner layer is 4x wider (standard transformer trick).
    """

    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)

    def __call__(self, x):
        x = nn.gelu(self.fc1(x))  # gelu is a smoother version of relu
        return self.fc2(x)


class TransformerBlock(nn.Module):
    """One transformer block = attention + MLP + layer norms + residual connections."""

    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_heads)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def __call__(self, x):
        # Residual connections: add the input back to the output
        # This lets information flow through even if attention/mlp mess up
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """The full model. Embeddings → transformer blocks → predict next character."""

    def __init__(self, vocab_size, n_embd=64, n_heads=4, n_layers=4, block_size=128):
        super().__init__()
        self.block_size = block_size

        # Embeddings (from step 2)
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        # Stack of transformer blocks
        self.blocks = [TransformerBlock(n_embd, n_heads) for _ in range(n_layers)]
        self.ln_f = nn.LayerNorm(n_embd)

        # Final projection: embedding dim → vocab size (one score per character)
        self.head = nn.Linear(n_embd, vocab_size)

    def __call__(self, x):
        seq_len = x.shape[1]
        positions = mx.arange(seq_len)

        # Embed: token IDs → vectors, add position info
        x = self.tok_emb(x) + self.pos_emb(positions)

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Project to vocabulary: each position gets a score per character
        logits = self.head(x)  # shape: (batch, seq_len, vocab_size)
        return logits
