import argparse
import csv
import os
import statistics
import time

import numpy as np
import torch
import torch.nn as nn

from .config import ModelConfig
from .model import ReadWriteEncoder


class TransformerBaseline(nn.Module):
    """MiniLM-L6-shaped full-attention encoder with random weights, mean pooled.

    Only for latency/memory comparison: quadratic attention vs the student's linear scan.
    """

    def __init__(self, vocab=30522, dim=384, layers=6, heads=12, max_len=8192):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, batch_first=True, norm_first=True,
                                           dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, ids, mask):
        x = self.tok(ids) + self.pos(torch.arange(ids.size(1), device=ids.device))
        h = self.enc(x, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp_min(1)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_model(fn, lengths, batch, device, repeats=3, vocab=30522, warmup=1):
    """Median forward latency per length. Returns dicts with seconds/peak_mb (None on OOM)."""
    device = torch.device(device)
    results = []
    for L in lengths:
        ids = torch.randint(1, vocab, (batch, L), device=device)
        mask = torch.ones(batch, L, dtype=torch.bool, device=device)
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for _ in range(warmup):
                    fn(ids, mask)
                ts = []
                for _ in range(repeats):
                    _sync(device)
                    t0 = time.perf_counter()
                    fn(ids, mask)
                    _sync(device)
                    ts.append(time.perf_counter() - t0)
            peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None
            results.append({"length": L, "seconds": statistics.median(ts), "peak_mb": peak})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append({"length": L, "seconds": None, "peak_mb": None})
    return results


def loglog_slope(lengths, times):
    """Exponent p in time ~ length^p (1.0 = linear, 2.0 = quadratic)."""
    return float(np.polyfit(np.log(lengths), np.log(times), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--read-layers", type=int, default=2)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="runs/benchmark.csv")
    args = ap.parse_args()

    dev = torch.device(args.device)
    if args.checkpoint:
        from .checkpoint import load_model
        student = load_model(args.checkpoint, dev)
    else:
        student = ReadWriteEncoder(ModelConfig(chunk_size=args.chunk_size,
                                               n_read_layers=args.read_layers)).to(dev).eval()
    base = TransformerBaseline(max_len=max(args.lengths)).to(dev).eval()

    rows = {}
    for name, m in (("student", student), ("transformer", base)):
        rows[name] = time_model(lambda i, k, m=m: m(i, k), args.lengths, args.batch, dev)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "length", "seconds", "peak_mb"])
        for name, rs in rows.items():
            for r in rs:
                w.writerow([name, r["length"], r["seconds"], r["peak_mb"]])

    print(f"device {dev}, batch {args.batch}, chunk_size {student.cfg.chunk_size}")
    print("| length | student s | student MB | transformer s | transformer MB |")
    print("|---|---|---|---|---|")
    fmt = lambda v, p: "OOM" if v is None else f"{v:.{p}f}"
    for s, t in zip(rows["student"], rows["transformer"]):
        print(f"| {s['length']} | {fmt(s['seconds'], 4)} | {fmt(s['peak_mb'], 0) if s['peak_mb'] is not None else '-'} "
              f"| {fmt(t['seconds'], 4)} | {fmt(t['peak_mb'], 0) if t['peak_mb'] is not None else '-'} |")
    for name, rs in rows.items():
        ok = [r for r in rs if r["seconds"]]
        if len(ok) >= 3:
            print(f"{name}: latency ~ length^{loglog_slope([r['length'] for r in ok], [r['seconds'] for r in ok]):.2f}")


if __name__ == "__main__":
    main()
