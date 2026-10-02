import torch
from rwe.config import ModelConfig
from rwe.model import ReadWriteEncoder


def make(chunk_size=4):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=50, dim=32, n_heads=4, chunk_size=chunk_size, n_read_layers=2)
    return ReadWriteEncoder(cfg).eval()


def pad_to(ids, L):
    B, n = ids.shape
    out = torch.zeros(B, L, dtype=torch.long)
    out[:, :n] = ids
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, :n] = True
    return out, mask


def test_padding_invariance():
    model = make()
    ids = torch.randint(1, 50, (2, 10))
    base = model(ids, torch.ones_like(ids, dtype=torch.bool))
    for L in (11, 12, 19, 32):  # partial-chunk pad and fully padded chunks
        padded, mask = pad_to(ids, L)
        assert torch.allclose(model(padded, mask), base, atol=1e-5), L


def test_batched_matches_individual():
    model = make()
    torch.manual_seed(1)
    a = torch.randint(1, 50, (1, 6))
    b = torch.randint(1, 50, (1, 13))
    ids = torch.zeros(2, 13, dtype=torch.long)
    mask = torch.zeros(2, 13, dtype=torch.bool)
    ids[0, :6], mask[0, :6] = a[0], True
    ids[1, :], mask[1, :] = b[0], True
    out = model(ids, mask)
    ea = model(a, torch.ones_like(a, dtype=torch.bool))
    eb = model(b, torch.ones_like(b, dtype=torch.bool))
    assert torch.allclose(out[0], ea[0], atol=1e-5)
    assert torch.allclose(out[1], eb[0], atol=1e-5)


def test_streaming_matches_batch_pass():
    model = make(chunk_size=4)
    ids = torch.randint(1, 50, (3, 11))
    mask = torch.ones_like(ids, dtype=torch.bool)
    expected = model(ids, mask)
    v = model.init_state(3)
    for s in range(0, 11, 4):
        v = model.step(v, ids[:, s:s + 4], mask[:, s:s + 4])
    assert torch.allclose(model.readout(v), expected, atol=1e-6)


def test_padded_chunk_leaves_state_unchanged():
    model = make(chunk_size=4)
    v = model.init_state(2)
    ids = torch.zeros(2, 4, dtype=torch.long)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    assert torch.equal(model.step(v, ids, mask), v)


def test_all_parameters_receive_gradient():
    model = make()
    model.train()
    ids = torch.randint(1, 50, (4, 9))
    mask = torch.ones_like(ids, dtype=torch.bool)
    model(ids, mask).pow(2).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_tokens_are_read_conditioned_on_state():
    from rwe.model import ReadLayer
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=50, dim=32, n_heads=4, chunk_size=4, n_read_layers=1)
    layer = ReadLayer(cfg).eval()
    x = torch.randn(2, 4, 32)
    key_pad = torch.zeros(2, 4, dtype=torch.bool)
    out1 = layer(x, torch.randn(2, 1, 32), key_pad)
    out2 = layer(x, torch.randn(2, 1, 32), key_pad)
    assert not torch.allclose(out1, out2, atol=1e-4)


def test_no_nan_with_fully_padded_sample():
    model = make()
    ids = torch.zeros(2, 8, dtype=torch.long)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    ids[0], mask[0] = torch.randint(1, 50, (8,)), True
    assert torch.isfinite(model(ids, mask)).all()
