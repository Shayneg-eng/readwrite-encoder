import argparse
import contextlib
import csv
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import yaml

from .checkpoint import save_ckpt
from .config import ModelConfig
from .data import LengthBucketSampler, TokenStore, collate
from .losses import distill_loss
from .model import ReadWriteEncoder


@dataclass
class TrainConfig:
    steps: int = 30000
    batch_size: int = 128
    accum: int = 1
    lr: float = 5e-4
    warmup: int = 1000
    min_lr_frac: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 1000
    ckpt_every: int = 1000
    log_every: int = 50
    seed: int = 0
    w_cos: float = 1.0
    w_sim: float = 1.0
    w_inter: float = 0.2
    amp: str = "auto"  # auto | bf16 | fp16 | off
    out_dir: str = "runs/base"


def lr_at(step, tc):
    if step < tc.warmup:
        return tc.lr * (step + 1) / tc.warmup
    p = min(1.0, (step - tc.warmup) / max(1, tc.steps - tc.warmup))
    return tc.lr * (tc.min_lr_frac + (1 - tc.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def amp_dtype(device, amp):
    if device.type != "cuda" or amp == "off":
        return None
    if amp == "bf16" or (amp == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def _batch_stream(store, teacher, sampler, pad_id, start_epoch=0):
    epoch = start_epoch
    while True:
        for idx in sampler.epoch(epoch):
            ids, mask = collate([store[i] for i in idx], pad_id)
            tv = torch.from_numpy(np.stack([np.asarray(teacher[i], dtype=np.float32) for i in idx]))
            yield ids, mask, tv
        epoch += 1


def holdout_cos(model, store, teacher, start, end, device, batch_size=128):
    """Mean cosine between student and teacher vectors on rows [start, end)."""
    was_training = model.training
    model.eval()
    idx = np.arange(start, end)
    idx = idx[np.argsort(store.lengths[idx], kind="stable")]
    total, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            ids, mask = collate([store[i] for i in b], model.cfg.pad_id)
            tv = torch.from_numpy(np.stack([np.asarray(teacher[i], dtype=np.float32) for i in b]))
            out = model(ids.to(device), mask.to(device)).float().cpu()
            total += torch.nn.functional.cosine_similarity(out, tv, dim=-1).sum().item()
            n += len(b)
    model.train(was_training)
    return total / max(1, n)


def train(model, store, teacher, tc, device, n_train=None, eval_fn=None, log=print):
    os.makedirs(tc.out_dir, exist_ok=True)
    n_train = len(store) if n_train is None else n_train
    model.to(device).train()
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=tc.lr, betas=(0.9, 0.98))
    dtype = amp_dtype(device, tc.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    micro = tc.batch_size // tc.accum
    sampler = LengthBucketSampler(store.lengths[:n_train], micro, seed=tc.seed)
    n_batches = max(1, n_train // micro)

    step = 0
    ck_path = os.path.join(tc.out_dir, "last.pt")
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        log(f"resumed from step {step}")
    stream = _batch_stream(store, teacher, sampler, model.cfg.pad_id,
                           start_epoch=(step * tc.accum) // n_batches)

    log_path = os.path.join(tc.out_dir, "log.csv")
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    writer = csv.writer(log_f)
    if new_log:
        writer.writerow(["step", "lr", "loss", "cos", "sim", "inter", "grad_norm", "elapsed_s"])
    eval_path = os.path.join(tc.out_dir, "eval.csv")
    eval_header_written = os.path.exists(eval_path)

    t0 = time.time()
    while step < tc.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc)
        opt.zero_grad(set_to_none=True)
        agg = {"loss": 0.0, "cos": 0.0, "sim": 0.0, "inter": 0.0}
        for _ in range(tc.accum):
            ids, mask, tv = next(stream)
            ids, mask, tv = ids.to(device), mask.to(device), tv.to(device)
            ctx = (torch.autocast(device.type, dtype=dtype) if dtype is not None
                   else contextlib.nullcontext())
            with ctx:
                final, states, valid = model.forward_with_states(ids, mask)
            loss, parts = distill_loss(final.float(), states.float(), valid, tv,
                                       tc.w_cos, tc.w_sim, tc.w_inter)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            scaler.scale(loss / tc.accum).backward()
            agg["loss"] += loss.item() / tc.accum
            for k, v in parts.items():
                agg[k] += v / tc.accum
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item()
        scaler.step(opt)
        scaler.update()
        step += 1

        if step % tc.log_every == 0 or step == tc.steps:
            writer.writerow([step, f"{lr_at(step - 1, tc):.3e}", f"{agg['loss']:.5f}",
                             f"{agg['cos']:.5f}", f"{agg['sim']:.5f}", f"{agg['inter']:.5f}",
                             f"{gn:.3f}", f"{time.time() - t0:.1f}"])
            log_f.flush()
            log(f"step {step}/{tc.steps} loss {agg['loss']:.4f} cos {agg['cos']:.4f} "
                f"sim {agg['sim']:.5f} gn {gn:.2f} ({time.time() - t0:.0f}s)")
        if eval_fn is not None and (step % tc.eval_every == 0 or step == tc.steps):
            model.eval()
            with torch.no_grad():
                res = eval_fn(model)
            model.train()
            with open(eval_path, "a", newline="") as ef:
                ew = csv.writer(ef)
                if not eval_header_written:
                    ew.writerow(["step"] + list(res))
                    eval_header_written = True
                ew.writerow([step] + [f"{v:.5f}" for v in res.values()])
            log(f"eval @ {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in res.items()))
        if step % tc.ckpt_every == 0 or step == tc.steps:
            save_ckpt(ck_path, model, opt, step, tc)
    log_f.close()
    return model


def _apply_overrides(cfg, overrides):
    for item in overrides:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides like model.chunk_size=1")
    ap.add_argument("--sts", action="store_true", help="also evaluate STS-B dev (needs network)")
    ap.add_argument("--init-tok-emb", default=None, help="teacher_tok_emb.pt to warm-start embeddings")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    _apply_overrides(cfg, args.set)
    model_cfg = ModelConfig(**cfg["model"])
    tc = TrainConfig(**cfg["train"])
    data_dir = cfg["data"]["dir"]
    holdout = cfg["data"]["holdout"]

    store = TokenStore.load(os.path.join(data_dir, "tokens.npz"))
    teacher = np.load(os.path.join(data_dir, "teacher.npy"), mmap_mode="r")
    n_train = len(store) - holdout
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(tc.seed)
    model = ReadWriteEncoder(model_cfg)
    if args.init_tok_emb and not os.path.exists(os.path.join(tc.out_dir, "last.pt")):
        w = torch.load(args.init_tok_emb)
        if tuple(w.shape) == tuple(model.tok.weight.shape):
            model.tok.weight.data.copy_(w)
            print("initialized token embeddings from teacher")
        else:
            print(f"skipping tok-emb init: shape {tuple(w.shape)} != {tuple(model.tok.weight.shape)}")

    sts_eval = None
    if args.sts:
        from .encode import encode
        from .eval_sts import eval_pairs, load_stsb
        from .tokenization import HFTokenizer

        tok = HFTokenizer()
        a, b, gold = load_stsb("validation")
        sts_eval = lambda m: eval_pairs(
            lambda texts: encode(m, tok, texts, device=device), a, b, gold)

    def eval_fn(m):
        res = {"holdout_cos": holdout_cos(m, store, teacher, n_train, len(store), device)}
        if sts_eval is not None:
            res["stsb_dev_spearman"] = sts_eval(m)
        return res

    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
          f"{n_train} train / {holdout} holdout texts, device {device}")
    train(model, store, teacher, tc, device, n_train=n_train, eval_fn=eval_fn)


if __name__ == "__main__":
    main()
