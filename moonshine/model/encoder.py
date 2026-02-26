"""Moonshine encoder: 14-layer transformer with sliding window attention.

Key differences from a standard transformer:
  - Custom LayerNorm with unit_offset (gamma + 1.0 instead of gamma)
  - No positional embeddings — uses sliding window attention masks instead
  - Attention projects 768→640 (10 heads x 64), output 640→768
  - GELU activation in FFN
  - Per-layer sliding window sizes: first/last 2 layers have [16,4], middle 10 have [16,0]
"""

import mlx.core as mx
import mlx.nn as nn

from .config import EncoderArgs
from .embedder import Embedder


class MoonshineLayerNorm(nn.Module):
    """LayerNorm with unit_offset: output = (1 + gamma) * normalize(x).

    The gamma parameter is initialized to 0 so the initial effective scale is 1.0.
    This differs from standard LayerNorm where gamma IS the scale directly.
    """

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # gamma starts at 0.0; effective weight = gamma + 1.0 = 1.0
        self.gamma = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        normed = (x - mean) * mx.rsqrt(var + self.eps)
        return normed * (self.gamma + 1.0)


class EncoderAttention(nn.Module):
    """Multi-head self-attention for the encoder.

    No RoPE, no causal masking — uses a sliding window mask instead.
    Projects: Q/K/V 768→640, O 640→768 (since 10 heads x 64 = 640 ≠ 768).
    """

    def __init__(self, args: EncoderArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim ** -0.5
        attn_dim = args.num_attention_heads * args.head_dim  # 640

        self.q_proj = nn.Linear(args.hidden_size, attn_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, attn_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(args.hidden_size, attn_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(attn_dim, args.hidden_size, bias=args.attention_bias)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(out)


class EncoderFFN(nn.Module):
    """Feed-forward network: fc1(768→3072) → GELU → fc2(3072→768)."""

    def __init__(self, args: EncoderArgs):
        super().__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.intermediate_size)
        self.fc2 = nn.Linear(args.intermediate_size, args.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """Pre-norm transformer block: LN → Attn → residual → LN → FFN → residual."""

    def __init__(self, args: EncoderArgs):
        super().__init__()
        self.self_attn = EncoderAttention(args)
        self.mlp = EncoderFFN(args)
        self.input_layernorm = MoonshineLayerNorm(args.hidden_size)
        self.post_attention_layernorm = MoonshineLayerNorm(args.hidden_size)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask=mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


def make_sliding_window_mask(seq_len: int, left: int, right: int) -> mx.array:
    """Create a bidirectional sliding window attention mask.

    Returns an additive mask: 0 where attention is allowed, -inf where blocked.
    Shape: (1, 1, seq_len, seq_len) for broadcasting over (B, heads, Q, K).

    Args:
        seq_len: sequence length
        left: how many positions to the left (including self) to attend to
        right: how many positions to the right to attend to (0 = no lookahead)
    """
    # dist[i,j] = i - j (positive means j is to the left of i)
    rows = mx.arange(seq_len)[:, None]
    cols = mx.arange(seq_len)[None, :]
    dist = rows - cols

    # Left: dist >= 0 and dist < left_window
    # Right: dist < 0 and -dist < right_window  (i.e. dist > -right_window)
    left_ok = (dist >= 0) & (dist < left)
    if right > 0:
        right_ok = (dist < 0) & (-dist < right)
        allowed = left_ok | right_ok
    else:
        allowed = left_ok

    # Convert to additive mask: 0 where allowed, -inf where blocked
    mask = mx.where(allowed, mx.array(0.0), mx.array(float("-inf")))
    return mask[None, None, :, :]  # (1, 1, L, L)


class Encoder(nn.Module):
    """Full Moonshine encoder: embedder → 14 transformer layers → final norm."""

    def __init__(self, args: EncoderArgs):
        super().__init__()
        self.args = args
        self.embedder = Embedder(args)
        self.layers = [EncoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.final_norm = MoonshineLayerNorm(args.hidden_size)

    def __call__(self, audio: mx.array) -> mx.array:
        """
        Args:
            audio: (B, audio_len) raw waveform
        Returns:
            (B, seq_len, 768) encoder hidden states
        """
        # Embed audio into token-like representations
        x = self.embedder(audio)
        seq_len = x.shape[1]

        # Pre-compute per-layer sliding window masks
        masks = []
        for left, right in self.args.sliding_windows:
            masks.append(make_sliding_window_mask(seq_len, left, right))

        # Run through transformer layers
        for layer, mask in zip(self.layers, masks):
            x = layer(x, mask=mask)

        return self.final_norm(x)
