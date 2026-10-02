import zlib


class HashTokenizer:
    """Offline whitespace tokenizer for tests. Ids are 1..vocab_size-1; 0 is padding."""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_id = 0

    def __call__(self, texts, max_len):
        out = []
        for t in texts:
            ids = [1 + zlib.crc32(w.encode()) % (self.vocab_size - 1) for w in t.lower().split()]
            out.append(ids[:max_len])
        return out


class HFTokenizer:
    """The teacher's WordPiece tokenizer (needs `transformers` and network on first use)."""

    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2"):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        self.vocab_size = self.tok.vocab_size
        self.pad_id = self.tok.pad_token_id

    def __call__(self, texts, max_len):
        enc = self.tok(list(texts), truncation=True, max_length=max_len, add_special_tokens=True)
        return enc["input_ids"]
