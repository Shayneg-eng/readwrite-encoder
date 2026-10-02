from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 30522
    dim: int = 384
    chunk_size: int = 16
    n_read_layers: int = 2
    n_heads: int = 6
    ffn_mult: int = 4
    dropout: float = 0.0
    pad_id: int = 0

    def __post_init__(self):
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
