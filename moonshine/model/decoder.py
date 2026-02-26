"""Moonshine decoder: 14-layer transformer with RoPE + cross-attention.

Key features:
  - Partial RoPE (50% of head dims = 32 out of 64) on self-attention only
  - Cross-attention to encoder output (K/V cached after first computation)
  - Gated SiLU FFN: fc1 outputs 2x width, splits into (value, gate),
    computes silu(gate) * value, then fc2
  - Standard LayerNorm (no unit_offset, no bias)
  - Three LayerNorms per layer: input, post-attention, final
"""

import mlx.core as mx
import mlx.nn as nn

from .config import ModelArgs


class KVCache:
    """Simple KV cache for autoregressive decoding.

    Stores key and value tensors, concatenating new entries each step.
    """

    def __init__(self):
        self.keys: mx.array | None = None
        self.values: mx.array | None = None

    @property
    def offset(self) -> int:
        return self.keys.shape[2] if self.keys is not None else 0

    def update(self, keys: mx.array, values: mx.array):
        if self.keys is not None:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        else:
            self.keys = keys
            self.values = values
        return self.keys, self.values


class DecoderSelfAttention(nn.Module):
    """Self-attention with partial RoPE and causal masking.

    RoPE applies to the first 32 of 64 head dims (partial_rotary_factor=0.5).
    Uses interleaved RoPE (traditional=False in MLX).
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim ** -0.5
        dim = args.hidden_size  # 640

        self.q_proj = nn.Linear(dim, dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Partial RoPE: only first rope_dims dimensions get rotation
        # traditional=True = interleaved pairs (0,1),(2,3)... matching HF's rotate_half
        self.rope = nn.RoPE(
            dims=args.rope_dims,       # 32
            traditional=True,           # interleaved pairs
            base=args.rope_theta,       # 10000
        )

    def __call__(self, x: mx.array, mask: mx.array = None,
                 cache: KVCache = None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE with offset for cached positions
        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # Update KV cache
        if cache is not None:
            k, v = cache.update(k, v)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(out)


class DecoderCrossAttention(nn.Module):
    """Cross-attention from decoder to encoder output.

    No RoPE, no causal mask. K/V from encoder are computed once and cached.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim ** -0.5
        dim = args.hidden_size  # 640

        self.q_proj = nn.Linear(dim, dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array, encoder_out: mx.array,
                 cache: KVCache = None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Use cached K/V if available, otherwise compute from encoder output
        if cache is not None and cache.keys is not None:
            k, v = cache.keys, cache.values
        else:
            S = encoder_out.shape[1]
            k = self.k_proj(encoder_out).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = self.v_proj(encoder_out).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            if cache is not None:
                cache.update(k, v)

        # Scaled dot-product attention (no causal mask)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        weights = mx.softmax(scores, axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(out)


class GatedFFN(nn.Module):
    """Gated feed-forward: fc1 outputs 2x width, splits into (value, gate).

    fc1: 640 → 5120 (intermediate_size * 2)
    Split: first half = value, second half = gate
    Output: silu(gate) * value
    fc2: 2560 → 640
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.intermediate_size * 2)
        self.fc2 = nn.Linear(args.intermediate_size, args.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        # Split: first half = value, second half = gate
        value, gate = mx.split(x, 2, axis=-1)
        x = nn.silu(gate) * value
        return self.fc2(x)


class DecoderLayer(nn.Module):
    """Decoder transformer block with three sub-layers:
      1. LN → SelfAttn (causal + RoPE) → residual
      2. LN → CrossAttn (to encoder) → residual
      3. LN → GatedFFN → residual
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = DecoderSelfAttention(args)
        self.encoder_attn = DecoderCrossAttention(args)
        self.mlp = GatedFFN(args)
        self.input_layernorm = nn.LayerNorm(args.hidden_size, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(args.hidden_size, bias=False)
        self.final_layernorm = nn.LayerNorm(args.hidden_size, bias=False)

    def __call__(self, x: mx.array, encoder_out: mx.array,
                 mask: mx.array = None,
                 self_attn_cache: KVCache = None,
                 cross_attn_cache: KVCache = None) -> mx.array:
        # Self-attention
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache=self_attn_cache)
        h = x + r

        # Cross-attention
        r = self.encoder_attn(
            self.post_attention_layernorm(h),
            encoder_out,
            cache=cross_attn_cache,
        )
        h = h + r

        # Feed-forward
        r = self.mlp(self.final_layernorm(h))
        return h + r


class Decoder(nn.Module):
    """Full Moonshine decoder.

    Includes:
    - Token embedding (vocab → 640)
    - Positional embedding for encoder output (added before cross-projection)
    - Linear projection encoder dim → decoder dim (768 → 640)
    - 14 decoder layers
    - Final LayerNorm
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # Token embedding
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        # Positional embedding for encoder output (768-dim, added before projection)
        self.pos_emb = nn.Embedding(args.max_position_embeddings, args.encoder.hidden_size)

        # Project encoder dim to decoder dim
        self.proj = nn.Linear(args.encoder.hidden_size, args.hidden_size, bias=False)

        # Decoder layers
        self.layers = [DecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.LayerNorm(args.hidden_size, bias=False)

    def __call__(self, tokens: mx.array, encoder_out: mx.array,
                 mask: mx.array = None,
                 cache: list = None) -> mx.array:
        """
        Args:
            tokens: (B, T) token ids
            encoder_out: (B, S, 768) encoder hidden states
            mask: causal mask for self-attention
            cache: list of (self_attn_cache, cross_attn_cache) per layer
        Returns:
            (B, T, 640) decoder hidden states
        """
        # Add positional embeddings to encoder output, then project
        S = encoder_out.shape[1]
        pos_ids = mx.arange(S)
        encoder_out = encoder_out + self.pos_emb(pos_ids)
        encoder_out = self.proj(encoder_out)

        # Token embeddings
        x = self.embed_tokens(tokens)

        # Run through decoder layers
        for i, layer in enumerate(self.layers):
            sa_cache = cache[i][0] if cache is not None else None
            xa_cache = cache[i][1] if cache is not None else None
            x = layer(x, encoder_out, mask=mask,
                      self_attn_cache=sa_cache, cross_attn_cache=xa_cache)

        return self.norm(x)
