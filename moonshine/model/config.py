"""Moonshine model configuration dataclasses.

Maps HuggingFace config.json fields to clean, typed dataclasses
for the MLX implementation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class EncoderArgs:
    hidden_size: int = 768
    num_attention_heads: int = 10
    num_key_value_heads: int = 10
    head_dim: int = 64
    num_hidden_layers: int = 14
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    attention_bias: bool = False
    sample_rate: int = 16000
    frame_ms: float = 5.0
    max_position_embeddings: int = 4096
    sliding_windows: List[Tuple[int, int]] = field(default_factory=lambda: [
        [16, 4], [16, 4],
        [16, 0], [16, 0], [16, 0], [16, 0], [16, 0],
        [16, 0], [16, 0], [16, 0], [16, 0], [16, 0],
        [16, 4], [16, 4],
    ])

    @property
    def frame_len(self) -> int:
        return int(round(self.sample_rate * self.frame_ms / 1000.0))


@dataclass
class ModelArgs:
    # Decoder config
    hidden_size: int = 640
    num_attention_heads: int = 10
    num_key_value_heads: int = 10
    head_dim: int = 64
    num_hidden_layers: int = 14
    intermediate_size: int = 2560
    hidden_act: str = "silu"
    attention_bias: bool = False
    vocab_size: int = 32768
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = False

    # RoPE
    partial_rotary_factor: float = 0.5
    rope_theta: float = 10000.0

    # Special tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    # Encoder config (nested)
    encoder: EncoderArgs = field(default_factory=EncoderArgs)

    @property
    def rope_dims(self) -> int:
        """Number of dimensions that get rotary embeddings."""
        return int(self.head_dim * self.partial_rotary_factor)

    @classmethod
    def from_dict(cls, config: dict) -> "ModelArgs":
        """Parse a HuggingFace config.json dict into ModelArgs."""
        # Parse encoder config
        enc_cfg = config.get("encoder_config", {})
        encoder = EncoderArgs(
            hidden_size=enc_cfg.get("hidden_size", 768),
            num_attention_heads=enc_cfg.get("num_attention_heads", 10),
            num_key_value_heads=enc_cfg.get("num_key_value_heads", 10),
            head_dim=enc_cfg.get("head_dim", 64),
            num_hidden_layers=enc_cfg.get("num_hidden_layers", 14),
            intermediate_size=enc_cfg.get("intermediate_size", 3072),
            hidden_act=enc_cfg.get("hidden_act", "gelu"),
            attention_bias=enc_cfg.get("attention_bias", False),
            sample_rate=enc_cfg.get("sample_rate", 16000),
            frame_ms=enc_cfg.get("frame_ms", 5.0),
            max_position_embeddings=enc_cfg.get("max_position_embeddings", 4096),
            sliding_windows=[
                tuple(w) for w in enc_cfg.get("sliding_windows", [])
            ],
        )

        # Parse RoPE params
        rope_params = config.get("rope_parameters", {})

        return cls(
            hidden_size=config.get("hidden_size", 640),
            num_attention_heads=config.get("num_attention_heads", 10),
            num_key_value_heads=config.get("num_key_value_heads", 10),
            head_dim=config.get("head_dim", 64),
            num_hidden_layers=config.get("num_hidden_layers", 14),
            intermediate_size=config.get("intermediate_size", 2560),
            hidden_act=config.get("hidden_act", "silu"),
            attention_bias=config.get("attention_bias", False),
            vocab_size=config.get("vocab_size", 32768),
            max_position_embeddings=config.get("max_position_embeddings", 4096),
            tie_word_embeddings=config.get("tie_word_embeddings", False),
            partial_rotary_factor=rope_params.get("partial_rotary_factor", 0.5),
            rope_theta=rope_params.get("rope_theta", 10000.0),
            bos_token_id=config.get("bos_token_id", 1),
            eos_token_id=config.get("eos_token_id", 2),
            pad_token_id=config.get("pad_token_id", 0),
            encoder=encoder,
        )
