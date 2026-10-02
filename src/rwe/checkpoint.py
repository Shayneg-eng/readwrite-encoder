import os
from dataclasses import asdict

import torch

from .config import ModelConfig
from .model import ReadWriteEncoder


def save_ckpt(path, model, opt, step, tc):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "cfg": asdict(model.cfg), "tc": asdict(tc)}, tmp)
    os.replace(tmp, path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = ReadWriteEncoder(ModelConfig(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval()
