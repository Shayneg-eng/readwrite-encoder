import numpy as np
import torch


class TokenStore:
    """Variable-length token id sequences stored as one flat int32 array plus offsets."""

    def __init__(self, flat, offsets):
        self.flat = flat
        self.offsets = offsets

    @classmethod
    def from_seqs(cls, seqs):
        lengths = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
        offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        flat = np.empty(offsets[-1], dtype=np.int32)
        for i, s in enumerate(seqs):
            flat[offsets[i]:offsets[i + 1]] = s
        return cls(flat, offsets)

    @classmethod
    def concat(cls, stores):
        flat = np.concatenate([s.flat for s in stores])
        lengths = np.concatenate([s.lengths for s in stores])
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        return cls(flat, offsets)

    def save(self, path):
        np.savez(path, flat=self.flat, offsets=self.offsets)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["flat"], d["offsets"])

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i):
        return self.flat[self.offsets[i]:self.offsets[i + 1]]

    @property
    def lengths(self):
        return np.diff(self.offsets)


def collate(seqs, pad_id):
    """Right-pad sequences into (ids[B,L], mask[B,L]); mask is True for real tokens."""
    L = max(1, max(len(s) for s in seqs))
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        n = len(s)
        if n:
            ids[i, :n] = torch.as_tensor(np.asarray(s), dtype=torch.long)
            mask[i, :n] = True
    return ids, mask


class LengthBucketSampler:
    """Shuffle, sort within large buckets by length, cut into batches, shuffle batches.

    Keeps batches length-homogeneous (less padding) while staying random across epochs.
    Incomplete final batches are dropped.
    """

    def __init__(self, lengths, batch_size, seed=0, bucket_mult=50):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.bucket = batch_size * bucket_mult

    def epoch(self, epoch):
        rng = np.random.default_rng(self.seed + epoch)
        perm = rng.permutation(len(self.lengths))
        batches = []
        for s in range(0, len(perm), self.bucket):
            chunk = perm[s:s + self.bucket]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for b in range(0, len(chunk) - self.batch_size + 1, self.batch_size):
                batches.append(chunk[b:b + self.batch_size])
        order = rng.permutation(len(batches))
        return [batches[i] for i in order]
