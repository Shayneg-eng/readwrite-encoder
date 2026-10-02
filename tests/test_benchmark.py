import torch

from rwe.benchmark_scaling import TransformerBaseline, loglog_slope, time_model
from rwe.config import ModelConfig
from rwe.model import ReadWriteEncoder


def test_loglog_slope_recovers_known_exponents():
    L = [100, 200, 400, 800]
    assert abs(loglog_slope(L, [l * 3.0 for l in L]) - 1.0) < 1e-9
    assert abs(loglog_slope(L, [l ** 2 * 1e-6 for l in L]) - 2.0) < 1e-9


def test_student_latency_scales_roughly_linearly_on_cpu():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=100, dim=64, n_heads=4, chunk_size=16, n_read_layers=1)
    model = ReadWriteEncoder(cfg).eval()
    lengths = [256, 512, 1024, 2048]
    res = time_model(lambda i, m: model(i, m), lengths, batch=2, device="cpu", repeats=3, vocab=100)
    slope = loglog_slope(lengths, [r["seconds"] for r in res])
    assert 0.6 < slope < 1.5, slope


def test_baseline_runs():
    m = TransformerBaseline(vocab=100, dim=32, layers=1, heads=4, max_len=64).eval()
    ids = torch.randint(1, 100, (2, 20))
    assert m(ids, torch.ones(2, 20, dtype=torch.bool)).shape == (2, 32)
