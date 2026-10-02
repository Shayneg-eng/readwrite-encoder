import csv
import os

import numpy as np
import torch

from rwe.checkpoint import load_model
from rwe.config import ModelConfig
from rwe.data import TokenStore
from rwe.model import ReadWriteEncoder
from rwe.synthetic import SyntheticTokenizer, make_synthetic
from rwe.train import TrainConfig, holdout_cos, lr_at, train


def test_lr_schedule_warms_up_then_decays_to_floor():
    tc = TrainConfig(steps=1000, warmup=100, lr=1e-3, min_lr_frac=0.1)
    assert lr_at(0, tc) < lr_at(50, tc) < lr_at(99, tc)
    assert abs(lr_at(100, tc) - 1e-3) < 1e-9
    assert lr_at(500, tc) < lr_at(100, tc)
    assert abs(lr_at(1000, tc) - 1e-4) < 1e-9


def setup(n=1500, dim=32, chunk=4):
    texts, teacher = make_synthetic(n, vocab_words=50, min_len=4, max_len=24, dim=dim, seed=0)
    tok = SyntheticTokenizer(50)
    store = TokenStore.from_seqs(tok(texts, 64))
    cfg = ModelConfig(vocab_size=tok.vocab_size, dim=dim, n_heads=4, chunk_size=chunk,
                      n_read_layers=1)
    return store, teacher, cfg


def test_train_learns_bag_of_words_distillation(tmp_path):
    store, teacher, cfg = setup()
    n_train = 1300
    torch.manual_seed(0)
    model = ReadWriteEncoder(cfg)
    before = holdout_cos(model, store, teacher, n_train, len(store), torch.device("cpu"))
    tc = TrainConfig(steps=500, batch_size=64, lr=3e-3, warmup=20, eval_every=500,
                     ckpt_every=250, log_every=50, out_dir=str(tmp_path))
    seen = {}
    def ev(m):
        seen["c"] = holdout_cos(m, store, teacher, n_train, len(store), torch.device("cpu"))
        return {"holdout_cos": seen["c"]}
    train(model, store, teacher, tc, torch.device("cpu"), n_train=n_train, eval_fn=ev,
          log=lambda *_: None)
    assert before < 0.5
    assert seen["c"] > 0.75, (before, seen["c"])
    assert os.path.exists(tmp_path / "eval.csv") and os.path.exists(tmp_path / "last.pt")
    reloaded = load_model(str(tmp_path / "last.pt"))
    assert reloaded.cfg == cfg


def test_resume_continues_from_saved_step(tmp_path):
    store, teacher, cfg = setup(n=400)
    kw = dict(batch_size=32, lr=1e-3, warmup=5, eval_every=10 ** 9, ckpt_every=10, log_every=1,
              out_dir=str(tmp_path))
    torch.manual_seed(0)
    train(ReadWriteEncoder(cfg), store, teacher, TrainConfig(steps=10, **kw),
          torch.device("cpu"), log=lambda *_: None)
    torch.manual_seed(1)
    train(ReadWriteEncoder(cfg), store, teacher, TrainConfig(steps=20, **kw),
          torch.device("cpu"), log=lambda *_: None)
    rows = list(csv.DictReader(open(tmp_path / "log.csv")))
    steps = [int(r["step"]) for r in rows]
    assert steps == list(range(1, 21))  # 10 + resumed 10, no restart from 0
    assert torch.load(tmp_path / "last.pt")["step"] == 20
