"""Top-level Moonshine model: encoder + decoder + output projection."""

import mlx.core as mx
import mlx.nn as nn

from .config import ModelArgs
from .encoder import Encoder
from .decoder import Decoder


class Moonshine(nn.Module):
    """Moonshine ASR: encode audio → decode text tokens.

    Architecture:
      Audio → Encoder (embedder + 14 layers) → 768-dim
        → Project to 640-dim + positional embeddings (handled inside Decoder)
        → Decoder (14 layers with RoPE + cross-attention) → 640-dim
        → proj_out → vocab logits (32768)
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.encoder = Encoder(args.encoder)
        self.decoder = Decoder(args)
        self.proj_out = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def encode(self, audio: mx.array) -> mx.array:
        """Encode audio waveform to hidden states."""
        return self.encoder(audio)

    def decode(self, tokens: mx.array, encoder_out: mx.array,
               mask: mx.array = None, cache: list = None) -> mx.array:
        """Decode tokens with cross-attention to encoder output.

        Returns logits over vocabulary.
        """
        h = self.decoder(tokens, encoder_out, mask=mask, cache=cache)
        return self.proj_out(h)
