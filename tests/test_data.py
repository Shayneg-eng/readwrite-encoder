import numpy as np
import torch

from rwe.data import LengthBucketSampler, TokenStore, collate
from rwe.tokenization import HashTokenizer


def test_hash_tokenizer_is_deterministic_and_truncates():
    tok = HashTokenizer(vocab_size=500)
    a = tok(["Hello world foo bar"], 3)[0]
    b = tok(["hello WORLD foo bar"], 3)[0]
    assert a == b and len(a) == 3
    assert all(1 <= i < 500 for i in a)


def test_token_store_roundtrip_and_concat(tmp_path):
    seqs = [[1, 2, 3], [4], [], [5, 6]]
    s = TokenStore.from_seqs(seqs)
    assert len(s) == 4
    assert [list(s[i]) for i in range(4)] == seqs
    assert list(s.lengths) == [3, 1, 0, 2]
    p = str(tmp_path / "t.npz")
    s.save(p)
    s2 = TokenStore.load(p)
    assert [list(s2[i]) for i in range(4)] == seqs
    c = TokenStore.concat([s, s2])
    assert len(c) == 8 and list(c[7]) == [5, 6]


def test_collate_pads_right_and_masks():
    ids, mask = collate([np.array([5, 6, 7]), np.array([8])], pad_id=0)
    assert ids.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert mask.tolist() == [[True, True, True], [True, False, False]]


def test_collate_handles_empty_sequence():
    ids, mask = collate([np.array([], dtype=np.int32)], pad_id=0)
    assert ids.shape == (1, 1) and not mask.any()


def test_sampler_valid_unique_and_length_homogeneous():
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 200, size=2000)
    sam = LengthBucketSampler(lengths, batch_size=32, seed=1, bucket_mult=10)
    batches = sam.epoch(0)
    flat = np.concatenate(batches)
    assert len(flat) == len(set(flat.tolist()))
    assert flat.min() >= 0 and flat.max() < 2000
    assert all(len(b) == 32 for b in batches)
    spread = np.mean([lengths[b].max() - lengths[b].min() for b in batches])
    random_spread = np.mean([lengths[r].max() - lengths[r].min()
                             for r in np.array_split(rng.permutation(2000)[:1984], 62)])
    assert spread < 0.5 * random_spread


def test_sampler_differs_across_epochs_but_is_reproducible():
    lengths = np.arange(1, 641)
    sam = LengthBucketSampler(lengths, batch_size=16, seed=3, bucket_mult=5)
    e0, e0b, e1 = sam.epoch(0), sam.epoch(0), sam.epoch(1)
    assert all((a == b).all() for a, b in zip(e0, e0b))
    assert any((a != b).any() for a, b in zip(e0, e1))
