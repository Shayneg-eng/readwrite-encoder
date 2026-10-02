import torch
from rwe.config import ModelConfig
from rwe.model import ReadWriteEncoder


def tiny(chunk_size=4, **kw):
    return ModelConfig(vocab_size=50, dim=32, n_heads=4, chunk_size=chunk_size,
                       n_read_layers=1, **kw)


def batch(B=3, L=10):
    torch.manual_seed(0)
    ids = torch.randint(1, 50, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    return ids, mask


def test_output_shape():
    model = ReadWriteEncoder(tiny())
    ids, mask = batch()
    assert model(ids, mask).shape == (3, 32)


def test_states_shape_and_chunk_count():
    model = ReadWriteEncoder(tiny(chunk_size=4))
    ids, mask = batch(L=10)
    emb, states, valid = model.forward_with_states(ids, mask)
    assert states.shape == (3, 3, 32)  # ceil(10/4) = 3 chunks
    assert valid.shape == (3, 3)
    assert torch.allclose(states[:, -1], emb)


def test_chunk_size_one_is_per_token():
    model = ReadWriteEncoder(tiny(chunk_size=1))
    ids, mask = batch(L=7)
    _, states, _ = model.forward_with_states(ids, mask)
    assert states.shape[1] == 7


def test_vector_changes_as_it_reads():
    model = ReadWriteEncoder(tiny(chunk_size=4))
    ids, mask = batch(L=12)
    _, states, _ = model.forward_with_states(ids, mask)
    assert not torch.allclose(states[:, 0], states[:, 1])
    assert not torch.allclose(states[:, 1], states[:, 2])
