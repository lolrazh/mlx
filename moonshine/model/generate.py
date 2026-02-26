"""Greedy decoding for Moonshine ASR.

Simple autoregressive loop:
  1. Encode audio → encoder_out
  2. Start with BOS token (id=1)
  3. Loop: decode one token, argmax, append
  4. Stop at EOS (id=2) or max_tokens
"""

import mlx.core as mx
import mlx.nn as nn

from .decoder import KVCache
from .moonshine import Moonshine


def generate(
    model: Moonshine,
    audio: mx.array,
    max_tokens: int = 512,
) -> list[int]:
    """Greedy decoding from audio input.

    Args:
        model: Loaded Moonshine model
        audio: (1, audio_len) raw waveform at 16kHz
        max_tokens: Maximum number of tokens to generate

    Returns:
        List of token ids (excluding BOS, including EOS)
    """
    bos = model.args.bos_token_id
    eos = model.args.eos_token_id
    n_layers = model.args.num_hidden_layers

    # 1. Encode audio
    encoder_out = model.encode(audio)
    mx.eval(encoder_out)

    # 2. Initialize KV caches (self-attention + cross-attention per layer)
    cache = [(KVCache(), KVCache()) for _ in range(n_layers)]

    # 3. Start with BOS token
    tokens = [bos]
    token_input = mx.array([[bos]])

    for _ in range(max_tokens):
        # Decode one step — no causal mask needed for single-token input with cache
        logits = model.decode(token_input, encoder_out, cache=cache)
        mx.eval(logits)

        # Greedy: take argmax of last position
        next_token = logits[0, -1].argmax().item()
        tokens.append(next_token)

        if next_token == eos:
            break

        # Next input is just the new token
        token_input = mx.array([[next_token]])

    return tokens


def pad_audio(audio, frame_len: int = 80) -> mx.array:
    """Pad audio waveform length to be a multiple of frame_len.

    This replaces HF's Wav2Vec2FeatureExtractor which just does padding.

    Args:
        audio: numpy array or mx.array of shape (audio_len,)
        frame_len: Frame size in samples (default 80 for 5ms @ 16kHz)

    Returns:
        mx.array of shape (1, padded_len) — batched and padded
    """
    if not isinstance(audio, mx.array):
        audio = mx.array(audio)

    if audio.ndim == 1:
        audio = audio[None, :]  # Add batch dim

    length = audio.shape[1]
    remainder = length % frame_len
    if remainder != 0:
        pad_len = frame_len - remainder
        audio = mx.pad(audio, [(0, 0), (0, pad_len)])

    return audio
