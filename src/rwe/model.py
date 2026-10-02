import torch
import torch.nn as nn

from .config import ModelConfig


def _with_state_slot(key_pad):
    """Prepend an always-visible key slot (the state vector) to a padding mask."""
    return torch.cat([key_pad.new_zeros(key_pad.size(0), 1), key_pad], dim=1)


class ReadLayer(nn.Module):
    """Transformer layer whose tokens attend to their chunk and to the state slot."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.dim
        self.norm_attn = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.ffn_mult * d), nn.GELU(),
                                 nn.Linear(cfg.ffn_mult * d, d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, v, key_pad):
        kv = self.norm_attn(torch.cat([v, x], dim=1))
        a, _ = self.attn(self.norm_attn(x), kv, kv,
                         key_padding_mask=_with_state_slot(key_pad), need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ffn(self.norm_ffn(x)))


class WriteCell(nn.Module):
    """Summarize the chunk with the state as query, then apply a gated residual edit."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.dim
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.delta = nn.Sequential(nn.Linear(2 * d, cfg.ffn_mult * d), nn.GELU(),
                                   nn.Linear(cfg.ffn_mult * d, d))
        self.gate = nn.Linear(2 * d, d)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, v, x, key_pad):
        kv = self.norm_kv(torch.cat([v, x], dim=1))
        s, _ = self.attn(self.norm_q(v), kv, kv,
                         key_padding_mask=_with_state_slot(key_pad), need_weights=False)
        z = torch.cat([v, s], dim=-1)
        g = torch.sigmoid(self.gate(z))
        return self.out_norm(v + g * self.delta(z))


class ReadWriteEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx=cfg.pad_id)
        self.pos = nn.Embedding(cfg.chunk_size, cfg.dim)
        self.v0 = nn.Parameter(torch.randn(cfg.dim) * 0.02)
        self.read = nn.ModuleList([ReadLayer(cfg) for _ in range(cfg.n_read_layers)])
        self.write = WriteCell(cfg)
        self.head = nn.Linear(cfg.dim, cfg.dim)

    def init_state(self, batch: int) -> torch.Tensor:
        return self.v0.view(1, 1, -1).expand(batch, 1, -1)

    def step(self, v, ids, mask):
        k = ids.size(1)
        x = self.tok(ids) + self.pos(torch.arange(k, device=ids.device))
        key_pad = ~mask
        for layer in self.read:
            x = layer(x, v, key_pad)
        new_v = self.write(v, x, key_pad)
        valid = mask.any(dim=1).view(-1, 1, 1)
        return torch.where(valid, new_v, v)

    def readout(self, v):
        return self.head(v.squeeze(1))

    def forward_with_states(self, ids, mask):
        B, L = ids.shape
        k = self.cfg.chunk_size
        v = self.init_state(B)
        states, valid = [], []
        for s in range(0, L, k):
            m = mask[:, s:s + k]
            v = self.step(v, ids[:, s:s + k], m)
            states.append(self.readout(v))
            valid.append(m.any(dim=1))
        return self.readout(v), torch.stack(states, dim=1), torch.stack(valid, dim=1)

    def forward(self, ids, mask):
        return self.forward_with_states(ids, mask)[0]
