"""Audio embedder: raw waveform → encoder input representations.

Pipeline:
  1. Reshape waveform into 5ms frames (80 samples each at 16kHz)
  2. Frame-level CMVN (per-frame mean-center + RMS normalize)
  3. Learned asinh compression
  4. Linear projection (80 → 768) + SiLU
  5. CausalConv1d (768 → 1536, k=5, s=2) + SiLU
  6. CausalConv1d (1536 → 768, k=5, s=2)

Total effect: 4x temporal downsampling via the two strided convolutions.
"""

import mlx.core as mx
import mlx.nn as nn

from .config import EncoderArgs


class CausalConv1d(nn.Module):
    """1D convolution with left-only padding (causal).

    PyTorch Conv1d stores weights as (out, in, kernel).
    MLX nn.Conv1d stores weights as (out, kernel, in).
    The sanitize function in weights.py handles this transpose.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, bias: bool = True):
        super().__init__()
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, seq_len, channels) — MLX uses channels-last by default
        # Left-pad along the sequence dimension
        pad_widths = [(0, 0), (self.left_pad, 0), (0, 0)]
        x = mx.pad(x, pad_widths)
        return self.conv(x)


class AsinhCompression(nn.Module):
    """Learned asinh compression: asinh(exp(log_k) * x)."""

    def __init__(self, k_init: float = 0.75):
        super().__init__()
        self.log_k = mx.array(mx.log(mx.array(k_init)))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.arcsinh(mx.exp(self.log_k) * x)


class Embedder(nn.Module):
    """Audio waveform → encoder input embeddings."""

    def __init__(self, args: EncoderArgs):
        super().__init__()
        self.frame_len = args.frame_len  # 80

        # Learnable asinh compression
        self.comp = AsinhCompression()

        # Linear projection from frame to hidden dim (no bias)
        self.linear = nn.Linear(args.frame_len, args.hidden_size, bias=False)

        # Two causal convolutions for 4x downsampling
        self.conv1 = CausalConv1d(
            args.hidden_size, args.hidden_size * 2,
            kernel_size=5, stride=2,
        )
        self.conv2 = CausalConv1d(
            args.hidden_size * 2, args.hidden_size,
            kernel_size=5, stride=2,
        )

    @staticmethod
    def _cmvn(x: mx.array) -> mx.array:
        """Frame-level cepstral mean and variance normalization."""
        mean = mx.mean(x, axis=-1, keepdims=True)
        centered = x - mean
        rms = mx.sqrt(mx.mean(mx.square(centered), axis=-1, keepdims=True) + 1e-6)
        return centered / rms

    def __call__(self, audio: mx.array) -> mx.array:
        """
        Args:
            audio: (B, audio_len) raw waveform at 16kHz
        Returns:
            (B, seq_len, 768) encoder input embeddings
        """
        B = audio.shape[0]
        # 1. Reshape into frames: (B, num_frames, 80)
        num_frames = audio.shape[1] // self.frame_len
        x = audio[:, :num_frames * self.frame_len].reshape(B, num_frames, self.frame_len)

        # 2-3. CMVN + asinh compression
        x = self._cmvn(x)
        x = self.comp(x)

        # 4. Linear + SiLU: (B, num_frames, 768)
        x = nn.silu(self.linear(x))

        # 5. Conv1 + SiLU: (B, ~num_frames/2, 1536)
        x = nn.silu(self.conv1(x))

        # 6. Conv2: (B, ~num_frames/4, 768)
        x = self.conv2(x)

        return x
