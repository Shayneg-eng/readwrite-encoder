import argparse
import json
import os
import random
import re

from .hf_data import iter_rows

_WS = re.compile(r"\s+")

DEFAULT_LIMITS = {"allnli": 350_000, "wikitext": 300_000, "nq": 150_000, "stsb": 20_000}


def clean(text):
    """Collapse whitespace; drop empties, tiny strings and wikitext headings like '= Title ='."""
    t = _WS.sub(" ", text or "").strip()
    if len(t) < 3 or (t.startswith("=") and t.endswith("=")):
        return ""
    return t


def build(sources, limits, exclude=frozenset(), seed=0):
    """Merge text iterables into deduplicated, shuffled rows.

    sources: {name: iterable of raw strings}; limits: {name: max kept texts};
    exclude: strings (e.g. eval sentences) that must never appear in the output.
    """
    seen = {clean(x).lower() for x in exclude}
    rows = []
    for name, it in sources.items():
        cap = limits.get(name, 10 ** 9)
        kept = 0
        for raw in it:
            if kept >= cap:
                break
            t = clean(raw)
            key = t.lower()
            if not t or key in seen:
                continue
            seen.add(key)
            rows.append({"source": name, "text": t})
            kept += 1
    random.Random(seed).shuffle(rows)
    return rows


def src_allnli():
    for a, b in iter_rows("sentence-transformers/all-nli", "train", ["anchor", "positive"], config="pair"):
        yield a
        yield b


def src_wikitext():
    for (t,) in iter_rows("Salesforce/wikitext", "train", ["text"], config="wikitext-103-raw-v1"):
        if len(t.split()) >= 20:
            yield t


def src_nq():
    for q, a in iter_rows("sentence-transformers/natural-questions", "train", ["query", "answer"]):
        yield q
        yield a


def src_stsb_train():
    for a, b in iter_rows("sentence-transformers/stsb", "train", ["sentence1", "sentence2"]):
        yield a
        yield b


def stsb_eval_sentences():
    out = []
    for split in ("validation", "test"):
        for a, b in iter_rows("sentence-transformers/stsb", split, ["sentence1", "sentence2"]):
            out.append(a)
            out.append(b)
    return out


def _safe(name, gen_fn):
    try:
        yield from gen_fn()
    except Exception as e:  # a broken source should not kill the whole prep
        print(f"[warn] source {name} stopped early: {e!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/texts.jsonl")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all source limits")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    limits = {k: int(v * args.scale) for k, v in DEFAULT_LIMITS.items()}
    exclude = stsb_eval_sentences()  # leakage guard; if this fails we must not proceed
    print(f"excluding {len(exclude)} STS-B dev/test sentences")
    sources = {
        "stsb": _safe("stsb", src_stsb_train),
        "allnli": _safe("allnli", src_allnli),
        "wikitext": _safe("wikitext", src_wikitext),
        "nq": _safe("nq", src_nq),
    }
    rows = build(sources, limits, exclude, args.seed)
    if not rows:
        raise SystemExit("no texts collected; check network access to huggingface.co")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_source = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print(f"wrote {len(rows)} texts to {args.out}: {by_source}")


if __name__ == "__main__":
    main()
