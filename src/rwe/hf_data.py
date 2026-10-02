"""Read Hugging Face dataset parquet files directly (huggingface_hub + pyarrow).

Avoids the `datasets` library, whose dill-based hashing breaks on newer Python versions.
"""

_SPLIT_ALIASES = {"validation": ("validation", "valid", "dev"), "train": ("train",),
                  "test": ("test",)}


def select_parquet(files, split, config=None):
    """Pick parquet files in a repo listing that belong to `split` (and `config` if given)."""
    names = _SPLIT_ALIASES.get(split, (split,))
    out = []
    for f in files:
        if not f.endswith(".parquet"):
            continue
        parts = f.split("/")
        base, dirs = parts[-1], parts[:-1]
        if config is not None and config not in dirs:
            continue
        if any(base.startswith(n) or n in dirs for n in names):
            out.append(f)
    return sorted(out)


def iter_rows(repo, split, columns, config=None):
    """Yield tuples of `columns` for every row of a dataset split, shard by shard."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = select_parquet(HfApi().list_repo_files(repo, repo_type="dataset"), split, config)
    if not files:
        raise FileNotFoundError(f"no parquet files for {repo} split={split} config={config}")
    for f in files:
        path = hf_hub_download(repo, f, repo_type="dataset")
        pf = pq.ParquetFile(path)
        missing = [c for c in columns if c not in pf.schema_arrow.names]
        if missing:
            raise KeyError(f"{repo}/{f} has columns {pf.schema_arrow.names}, missing {missing}")
        for batch in pf.iter_batches(columns=list(columns), batch_size=10_000):
            yield from zip(*[batch.column(c).to_pylist() for c in columns])
