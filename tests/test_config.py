from rwe.config import ModelConfig


def test_defaults_match_spec():
    cfg = ModelConfig()
    assert cfg.dim == 384
    assert cfg.chunk_size == 16
    assert cfg.n_read_layers == 2
    assert cfg.vocab_size == 30522
    assert cfg.dropout == 0.0


def test_dim_must_divide_heads():
    import pytest
    with pytest.raises(ValueError):
        ModelConfig(dim=30, n_heads=4)
