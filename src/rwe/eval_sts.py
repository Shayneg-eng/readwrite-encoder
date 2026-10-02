import argparse

import numpy as np


def rankdata(a):
    """Average ranks (1-based), ties share the mean rank."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def cosine_pairs(ea, eb):
    ea = ea / np.clip(np.linalg.norm(ea, axis=1, keepdims=True), 1e-12, None)
    eb = eb / np.clip(np.linalg.norm(eb, axis=1, keepdims=True), 1e-12, None)
    return (ea * eb).sum(axis=1)


def eval_pairs(encode_fn, a, b, gold):
    """Spearman between cosine similarity of encoded pairs and gold scores."""
    return spearman(cosine_pairs(encode_fn(a), encode_fn(b)), gold)


def load_stsb(split):
    from .hf_data import iter_rows

    rows = list(iter_rows("sentence-transformers/stsb", split, ["sentence1", "sentence2", "score"]))
    return ([r[0] for r in rows], [r[1] for r in rows],
            np.asarray([r[2] for r in rows], dtype=np.float64))


def main():
    import torch

    from .checkpoint import load_model
    from .encode import encode
    from .tokenization import HFTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--teacher", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    tok = HFTokenizer(args.teacher)
    a, b, gold = load_stsb(args.split)
    student = eval_pairs(lambda t: encode(model, tok, t, device=device), a, b, gold)

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(args.teacher)
    st.max_seq_length = 256
    teacher = eval_pairs(lambda t: st.encode(t, batch_size=128), a, b, gold)
    print(f"STS-B {args.split} Spearman: student {student:.4f}  teacher {teacher:.4f}  "
          f"ratio {student / teacher:.3f}")


if __name__ == "__main__":
    main()
