"""Offline synthetic distillation task: can the vector carry a whole text across many chunks?

Each 'text' is a sequence of words w0..w{V-1}. The 'teacher' embedding is the normalized mean
of a fixed random vector per word, i.e. an order-insensitive bag of words. To match it the
student must remember every chunk's contribution in one vector. Cosine to the teacher as a
function of text length shows where the fixed-size state starts to lose information.
"""
import argparse
import time

import numpy as np
import torch

from .config import ModelConfig
from .data import TokenStore
from .model import ReadWriteEncoder
from .train import TrainConfig, holdout_cos, train


class SyntheticTokenizer:
    def __init__(self, vocab_words):
        self.vocab_size = vocab_words + 1
        self.pad_id = 0

    def __call__(self, texts, max_len):
        return [[int(w[1:]) + 1 for w in t.split()][:max_len] for t in texts]


def make_synthetic(n, vocab_words=100, min_len=4, max_len=64, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    word_vecs = np.random.default_rng(12345).normal(size=(vocab_words, dim)).astype(np.float32)
    texts, vecs = [], np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        L = int(rng.integers(min_len, max_len + 1))
        words = rng.integers(0, vocab_words, size=L)
        texts.append(" ".join(f"w{w}" for w in words))
        v = word_vecs[words].mean(0)
        vecs[i] = v / np.linalg.norm(v)
    return texts, vecs


def run(chunk_size, train_max_len, eval_lens, dim=64, layers=1, n=6000, steps=1500,
        batch=64, lr=2e-3, vocab_words=100, seed=0, out_dir="runs/synthetic"):
    texts, teacher = make_synthetic(n, vocab_words, 4, train_max_len, dim, seed)
    tok = SyntheticTokenizer(vocab_words)
    store = TokenStore.from_seqs(tok(texts, 10 ** 6))
    cfg = ModelConfig(vocab_size=tok.vocab_size, dim=dim, n_heads=4, chunk_size=chunk_size,
                      n_read_layers=layers)
    n_train = n - 500
    tc = TrainConfig(steps=steps, batch_size=batch, lr=lr, warmup=100, eval_every=10 ** 9,
                     ckpt_every=10 ** 9, log_every=250, seed=seed, w_cos=1.0, w_sim=1.0,
                     w_inter=0.2, out_dir=f"{out_dir}/k{chunk_size}_len{train_max_len}")
    torch.manual_seed(seed)
    model = ReadWriteEncoder(cfg)
    t0 = time.time()
    train(model, store, teacher, tc, torch.device("cpu"), n_train=n_train, log=lambda *_: None)
    train_s = time.time() - t0
    res = {"train_seconds": train_s,
           "in_dist_cos": holdout_cos(model, store, teacher, n_train, n, torch.device("cpu"))}
    for L in eval_lens:  # fixed-length probes, including longer than anything seen in training
        et, ev = make_synthetic(300, vocab_words, L, L, dim, seed + 99)
        es = TokenStore.from_seqs(tok(et, 10 ** 6))
        res[f"cos@{L}"] = holdout_cos(model, es, ev, 0, 300, torch.device("cpu"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--train-max-len", type=int, default=64)
    ap.add_argument("--eval-lens", type=int, nargs="+", default=[16, 64, 128, 256, 512])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--layers", type=int, default=1)
    args = ap.parse_args()
    print(f"train lengths 4..{args.train_max_len}; probes at {args.eval_lens}")
    for k in args.chunk_sizes:
        r = run(k, args.train_max_len, args.eval_lens, steps=args.steps, layers=args.layers)
        line = " ".join(f"{key}={val:.3f}" for key, val in r.items())
        print(f"k={k:>2}: {line}", flush=True)


if __name__ == "__main__":
    main()
