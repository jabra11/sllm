"""Model size presets for Initial Setup V1.

Decoder sizes count the tied embedding of the 49k SmolLM2 vocab; MoE presets roughly match the dense
ones in *active* parameters. Run `python -m scripts.count_params` to check.
"""

from dataclasses import dataclass, field


@dataclass
class EncoderConfig:
    num_mels: int = 80
    dim: int = 384
    num_layers: int = 12
    num_heads: int = 6
    ff_mult: int = 4
    conv_kernel: int = 31
    dropout: float = 0.1
    # Chunked attention, counted in frames after 4x subsampling (40ms each).
    chunk_size: int = 10
    right_context: int = 5


@dataclass
class DecoderConfig:
    kind: str = "dense"  # "dense" | "moe"
    dim: int = 640
    num_layers: int = 16
    num_heads: int = 10
    num_kv_heads: int = 2
    head_dim: int = 64
    ffn_dim: int = 1920  # dense only
    num_experts: int = 16  # moe only
    experts_per_tok: int = 2
    expert_ffn_dim: int = 512
    shared_expert_ffn_dim: int = 512


@dataclass
class ModelConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    # Frames stacked by the adapter on top of the encoder's 40ms output: 2 -> 80ms per speech token.
    adapter_stack: int = 2


# Target sizes (encoder / decoder incl. embedding): 20/80M, 40/160M, 100/400M, 200/800M.
SIZES = {
    1: dict(
        encoder=dict(dim=256, num_layers=12, num_heads=4),
        decoder=dict(dim=512, num_layers=16, num_heads=8, ffn_dim=1792,
                     expert_ffn_dim=448, shared_expert_ffn_dim=448),
    ),
    2: dict(
        encoder=dict(dim=384, num_layers=11, num_heads=6),
        decoder=dict(dim=768, num_layers=16, num_heads=12, ffn_dim=2304,
                     expert_ffn_dim=576, shared_expert_ffn_dim=576),
    ),
    3: dict(
        encoder=dict(dim=512, num_layers=16, num_heads=8),
        decoder=dict(dim=1024, num_layers=24, num_heads=16, ffn_dim=3584, num_experts=32, experts_per_tok=4,
                     expert_ffn_dim=640, shared_expert_ffn_dim=640),
    ),
    4: dict(
        encoder=dict(dim=768, num_layers=14, num_heads=12),
        decoder=dict(dim=1408, num_layers=28, num_heads=22, ffn_dim=4096, num_experts=32, experts_per_tok=4,
                     expert_ffn_dim=768, shared_expert_ffn_dim=768),
    ),
}


def make_config(size: int, decoder_kind: str = "dense") -> ModelConfig:
    preset = SIZES[size]
    return ModelConfig(
        encoder=EncoderConfig(**preset["encoder"]),
        decoder=DecoderConfig(kind=decoder_kind, **preset["decoder"]),
    )
