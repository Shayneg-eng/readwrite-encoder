import torch
from rwe.losses import cosine_loss, sim_matrix_loss, intermediate_loss, distill_loss


def test_cosine_loss_zero_for_identical_and_scale_invariant():
    t = torch.randn(5, 16)
    assert cosine_loss(t, t).abs() < 1e-6
    assert cosine_loss(3 * t, t).abs() < 1e-6


def test_cosine_loss_two_for_opposite():
    t = torch.randn(5, 16)
    assert torch.allclose(cosine_loss(-t, t), torch.tensor(2.0), atol=1e-6)


def test_sim_matrix_loss_zero_for_rotation_and_positive_otherwise():
    t = torch.randn(6, 16)
    q, _ = torch.linalg.qr(torch.randn(16, 16))
    assert sim_matrix_loss(t @ q, t) < 1e-6  # rotation keeps pairwise cosines
    assert sim_matrix_loss(torch.randn(6, 16), t) > 1e-3


def test_intermediate_loss_zero_when_states_equal_teacher():
    t = torch.randn(3, 16)
    states = t[:, None, :].expand(3, 4, 16).contiguous()
    valid = torch.ones(3, 4, dtype=torch.bool)
    assert intermediate_loss(states, valid, t).abs() < 1e-6


def test_intermediate_loss_ignores_invalid_chunks():
    t = torch.randn(2, 16)
    states = torch.randn(2, 4, 16)
    valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    base = intermediate_loss(states, valid, t)
    changed = states.clone()
    changed[0, 2:] = torch.randn(2, 16)
    assert torch.allclose(intermediate_loss(changed, valid, t), base)


def test_intermediate_loss_weights_later_chunks_more():
    t = torch.randn(1, 16)
    good = t.clone()
    bad = -t
    valid = torch.ones(1, 2, dtype=torch.bool)
    early_bad = torch.stack([bad, good], dim=1)
    late_bad = torch.stack([good, bad], dim=1)
    assert intermediate_loss(late_bad, valid, t) > intermediate_loss(early_bad, valid, t)


def test_distill_loss_combines_parts():
    final = torch.randn(4, 16)
    states = torch.randn(4, 3, 16)
    valid = torch.ones(4, 3, dtype=torch.bool)
    teacher = torch.randn(4, 16)
    total, parts = distill_loss(final, states, valid, teacher)
    expected = parts["cos"] + parts["sim"] + 0.2 * parts["inter"]
    assert abs(total.item() - expected) < 1e-5
    assert set(parts) == {"cos", "sim", "inter"}
