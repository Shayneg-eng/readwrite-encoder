# M0: Read-Write Encoder Model and CPU Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the chunked read-write encoder, its distillation losses, and a CPU test suite that proves the model's key properties.

**Architecture:** One 384-d state vector is carried across chunks of k tokens. Per chunk, tokens are encoded while attending to the current vector (read), then a gated residual edit updates the vector (write). Weights are shared across chunks. Pure PyTorch, no custom kernels.

**Tech Stack:** Python 3.10+, PyTorch, pytest.

**Spec:** `docs/specs/2026-09-29-readwrite-encoder-design.md`

## Global Constraints

- Vector size 384, chunk size k default 16, 2 read layers default, max training length 256 tokens.
- Position is encoded within a chunk only (0 to k-1); no global position embedding.
- Pad tokens are masked; a fully padded chunk leaves the vector unchanged.
- Pure PyTorch: no custom CUDA kernels, works on native Windows.
- Loss weights: cosine 1.0, similarity-matrix 1.0, intermediate 0.2.
- Tests run on CPU only.
- Scope: this plan covers M0 only. Data prep, teacher cache, training loop, evals and the training smoke run belong to later plans (M1-M3).

## File Structure

| File | Responsibility |
|---|---|
| `requirements.txt` | Python dependencies for M0 |
| `pytest.ini` | Puts `src/` on the import path |
| `src/rwe/__init__.py` | Package marker |
| `src/rwe/config.py` | `ModelConfig` dataclass |
| `src/rwe/model.py` | `ReadLayer`, `WriteCell`, `ReadWriteEncoder` |
| `src/rwe/losses.py` | Distillation losses |
| `tests/test_config.py`, `tests/test_model.py`, `tests/test_model_properties.py`, `tests/test_losses.py`, `tests/test_overfit.py` | Tests |

---

### Task 1: Scaffold and config

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `src/rwe/__init__.py`, `src/rwe/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `ModelConfig(vocab_size=30522, dim=384, chunk_size=16, n_read_layers=2, n_heads=6, ffn_mult=4, dropout=0.0, pad_id=0)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (ModuleNotFoundError: rwe)

- [ ] **Step 3: Write the scaffold and implementation**

```text
# requirements.txt
torch
pytest
```

```ini
# pytest.ini
[pytest]
pythonpath = src
testpaths = tests
```

```python
# src/rwe/__init__.py
"""Read-Write Encoder: O(n) text embedding model."""
```

```python
# src/rwe/config.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pytest.ini src tests
git commit -m "feat: scaffold package and ModelConfig"
```

---

### Task 2: Encoder forward pass

**Files:**
- Create: `src/rwe/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `ModelConfig` from Task 1.
- Produces, on `ReadWriteEncoder(cfg)`:
  - `init_state(batch: int) -> Tensor[B,1,D]`
  - `step(v: Tensor[B,1,D], ids: LongTensor[B,k'], mask: BoolTensor[B,k']) -> Tensor[B,1,D]` (mask True = real token; k' <= chunk_size)
  - `readout(v: Tensor[B,1,D]) -> Tensor[B,D]`
  - `forward(ids: LongTensor[B,L], mask: BoolTensor[B,L]) -> Tensor[B,D]`
  - `forward_with_states(ids, mask) -> (emb Tensor[B,D], states Tensor[B,T,D], valid BoolTensor[B,T])` where T = ceil(L / chunk_size), `states` are the readout of the vector after each chunk, and `valid[b,t]` is True if chunk t of sample b has any real token.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_model.py -v`
Expected: FAIL (ModuleNotFoundError: rwe.model)

- [ ] **Step 3: Write minimal implementation**

```python
# src/rwe/model.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_model.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/rwe/model.py tests/test_model.py
git commit -m "feat: chunked read-write encoder forward pass"
```

---

### Task 3: Model properties (padding, streaming, gradients)

**Files:**
- Test: `tests/test_model_properties.py`
- Modify: `src/rwe/model.py` only if a test fails

**Interfaces:**
- Consumes: `ReadWriteEncoder.init_state`, `.step`, `.readout`, `.forward` from Task 2.

- [ ] **Step 1: Write the tests**

```python
# tests/test_model_properties.py
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
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/test_model_properties.py -v`
Expected: PASS (7 passed). If a test fails, fix `src/rwe/model.py` (not the test) unless the test itself is wrong, and re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_model_properties.py src/rwe/model.py
git commit -m "test: padding, streaming, state-conditioning, gradient and NaN properties of the encoder"
```

---

### Task 4: Distillation losses

**Files:**
- Create: `src/rwe/losses.py`
- Test: `tests/test_losses.py`

**Interfaces:**
- Produces:
  - `cosine_loss(student: Tensor[B,D], teacher: Tensor[B,D]) -> Tensor[]`
  - `sim_matrix_loss(student, teacher) -> Tensor[]`
  - `intermediate_loss(states: Tensor[B,T,D], valid: BoolTensor[B,T], teacher: Tensor[B,D]) -> Tensor[]`
  - `distill_loss(final, states, valid, teacher, w_cos=1.0, w_sim=1.0, w_inter=0.2) -> (total Tensor[], parts dict[str, float])`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_losses.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_losses.py -v`
Expected: FAIL (ModuleNotFoundError: rwe.losses)

- [ ] **Step 3: Write minimal implementation**

```python
# src/rwe/losses.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_losses.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/rwe/losses.py tests/test_losses.py
git commit -m "feat: distillation losses (cosine, similarity-matrix, intermediate)"
```

---

### Task 5: Tiny overfit test and docs

**Files:**
- Create: `tests/test_overfit.py`, `README.md`
- Modify: `ORGANIZATION.md`

**Interfaces:**
- Consumes: `ReadWriteEncoder.forward_with_states` (Task 2) and `distill_loss` (Task 4).

- [ ] **Step 1: Write the test**

```python
# tests/test_overfit.py
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
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_overfit.py -v`
Expected: PASS. If `end` stays above 0.3, raise the step count or learning rate in the test; do not loosen the assertion below the point where it still shows real learning.

- [ ] **Step 3: Write README.md and update ORGANIZATION.md**

`README.md`: project purpose (O(n) read-write text embedder), how to set up (`python -m venv .venv`, install CUDA PyTorch from pytorch.org, `pip install -r requirements.txt`), how to run tests (`pytest`), pointer to the spec and plans.

`ORGANIZATION.md`: replace the "planned layout" wording with the actual current layout and add `docs/plans/` and `tests/` rows.

- [ ] **Step 4: Run the full suite**

Run: `pytest -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_overfit.py README.md ORGANIZATION.md docs/plans
git commit -m "test: tiny overfit run; add README and update organization"
```

---

## Self-Review

- **Spec coverage:** model section (state, chunking, read, summarize, write, readout, padding, chunk-local positions) is Tasks 1-3; losses with the stated weights are Task 4; the CPU validation list from spec section 9 is covered by Tasks 2-5 except the full training-loop smoke run, which needs `train.py` and moves to the M2 plan. Data, teacher cache, evals, benchmarks and the contrastive stage are M1-M4 plans.
- **Placeholders:** none; all code steps contain code.
- **Type consistency:** `init_state/step/readout/forward/forward_with_states` signatures in Task 2 are the ones used in Tasks 3 and 5; loss names and signatures in Task 4 are the ones used in Task 5.
