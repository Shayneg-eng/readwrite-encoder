import argparse
import json
import os

import numpy as np

from .data import TokenStore


def main():
    import torch
    from sentence_transformers import SentenceTransformer

    from .tokenization import HFTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default="data/texts.jsonl")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--teacher", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--shard", type=int, default=20000)
    args = ap.parse_args()

    texts = [json.loads(line)["text"] for line in open(args.texts, encoding="utf-8")]
    n = len(texts)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{n} texts")

    # 1) Tokenize with the teacher's tokenizer, shard by shard to bound memory.
    tok_path = os.path.join(args.out_dir, "tokens.npz")
    if not os.path.exists(tok_path):
        tok = HFTokenizer(args.teacher)
        parts = []
        for s in range(0, n, 100_000):
            parts.append(TokenStore.from_seqs(tok(texts[s:s + 100_000], args.max_len)))
            print(f"tokenized {min(n, s + 100_000)}/{n}")
        store = TokenStore.concat(parts)
        store.save(tok_path)
        print(f"saved {tok_path}; mean length {store.lengths.mean():.1f}, max {store.lengths.max()}")

    # 2) Teacher vectors (fp16 memmap), resumable.
    st = SentenceTransformer(args.teacher)
    st.max_seq_length = args.max_len
    dim = st.get_sentence_embedding_dimension()
    vec_path = os.path.join(args.out_dir, "teacher.npy")
    prog_path = vec_path + ".progress"
    done = int(open(prog_path).read()) if os.path.exists(prog_path) and os.path.exists(vec_path) else 0
    mm = np.lib.format.open_memmap(vec_path, mode="r+" if done else "w+", dtype=np.float16,
                                   shape=(n, dim))
    for s in range(done, n, args.shard):
        e = min(n, s + args.shard)
        mm[s:e] = st.encode(texts[s:e], batch_size=args.batch_size, convert_to_numpy=True)
        mm.flush()
        with open(prog_path, "w") as f:
            f.write(str(e))
        print(f"teacher encoded {e}/{n}")

    # 3) Teacher token embeddings for warm-starting the student.
    emb = st[0].auto_model.embeddings.word_embeddings.weight.detach().cpu()
    torch.save(emb, os.path.join(args.out_dir, "teacher_tok_emb.pt"))
    print(f"saved teacher token embeddings {tuple(emb.shape)}")


if __name__ == "__main__":
    main()
