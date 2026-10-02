import torch
import torch.nn.functional as F


def cosine_loss(student, teacher):
    return (1 - F.cosine_similarity(student, teacher, dim=-1)).mean()


def sim_matrix_loss(student, teacher):
    s = F.normalize(student, dim=-1)
    t = F.normalize(teacher, dim=-1)
    return F.mse_loss(s @ s.T, t @ t.T)


def intermediate_loss(states, valid, teacher):
    """Pull each chunk's readout toward the teacher's full-text vector.

    Chunk t gets weight (t + 1) / T, so later chunks count more. Invalid
    (fully padded) chunks get weight 0.
    """
    T = states.size(1)
    cos = F.cosine_similarity(states, teacher[:, None, :], dim=-1)
    ramp = torch.arange(1, T + 1, device=states.device, dtype=states.dtype) / T
    w = ramp[None, :] * valid.to(states.dtype)
    per_sample = ((1 - cos) * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-8)
    return per_sample.mean()


def distill_loss(final, states, valid, teacher, w_cos=1.0, w_sim=1.0, w_inter=0.2):
    cos = cosine_loss(final, teacher)
    sim = sim_matrix_loss(final, teacher)
    inter = intermediate_loss(states, valid, teacher)
    total = w_cos * cos + w_sim * sim + w_inter * inter
    return total, {"cos": cos.item(), "sim": sim.item(), "inter": inter.item()}
