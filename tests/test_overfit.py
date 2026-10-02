import torch
from rwe.config import ModelConfig
from rwe.model import ReadWriteEncoder
from rwe.losses import distill_loss, cosine_loss


def test_model_can_overfit_64_random_texts():
    torch.manual_seed(0)
    N, L, D = 64, 24, 32
    lengths = torch.randint(6, L + 1, (N,))
    ids = torch.randint(1, 100, (N, L))
    mask = torch.arange(L)[None, :] < lengths[:, None]
    ids = ids * mask
    teacher = torch.randn(N, D)

    cfg = ModelConfig(vocab_size=100, dim=D, n_heads=4, chunk_size=4, n_read_layers=1)
    model = ReadWriteEncoder(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    with torch.no_grad():
        start = cosine_loss(model(ids, mask), teacher).item()
    for _ in range(300):
        final, states, valid = model.forward_with_states(ids, mask)
        loss, _ = distill_loss(final, states, valid, teacher)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        end = cosine_loss(model(ids, mask), teacher).item()

    assert start > 0.8
    assert end < 0.3, (start, end)
