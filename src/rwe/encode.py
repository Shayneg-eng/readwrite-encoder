import numpy as np
import torch

from .data import collate


def encode(model, tokenizer, texts, max_len=256, batch_size=128, device="cpu"):
    """Embed texts; returns float32 array [N, dim] in the original order (not normalized)."""
    was_training = model.training
    model.eval()
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), model.cfg.dim), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            seqs = tokenizer([texts[i] for i in idx], max_len)
            ids, mask = collate(seqs, tokenizer.pad_id)
            out[idx] = model(ids.to(device), mask.to(device)).float().cpu().numpy()
    model.train(was_training)
    return out
