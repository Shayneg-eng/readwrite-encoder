import pyarrow as pa
import pyarrow.parquet as pq

from rwe.hf_data import select_parquet


def test_select_by_basename_prefix_and_config():
    files = ["README.md", "pair/train-00000-of-00001.parquet", "pair/dev-00000-of-00001.parquet",
             "triplet/train-00000-of-00001.parquet", "pair/test-00000-of-00001.parquet"]
    assert select_parquet(files, "train", "pair") == ["pair/train-00000-of-00001.parquet"]
    assert select_parquet(files, "validation", "pair") == ["pair/dev-00000-of-00001.parquet"]
    assert select_parquet(files, "test", "triplet") == []


def test_select_by_split_directory_and_multiple_shards():
    files = ["data/train/0000.parquet", "data/train/0001.parquet", "data/validation/0000.parquet",
             "wikitext-103-raw-v1/train-00001-of-00002.parquet",
             "wikitext-103-raw-v1/train-00000-of-00002.parquet"]
    assert select_parquet(files, "train") == [
        "data/train/0000.parquet", "data/train/0001.parquet",
        "wikitext-103-raw-v1/train-00000-of-00002.parquet",
        "wikitext-103-raw-v1/train-00001-of-00002.parquet"]
    assert select_parquet(files, "train", "wikitext-103-raw-v1") == [
        "wikitext-103-raw-v1/train-00000-of-00002.parquet",
        "wikitext-103-raw-v1/train-00001-of-00002.parquet"]


def test_iter_rows_reads_local_parquet(tmp_path, monkeypatch):
    import huggingface_hub

    path = tmp_path / "train-00000-of-00001.parquet"
    pq.write_table(pa.table({"sentence1": ["a", "b"], "sentence2": ["c", "d"], "score": [0.5, 1.0]}),
                   path)

    class FakeApi:
        def list_repo_files(self, repo, repo_type=None):
            return ["train-00000-of-00001.parquet", "README.md"]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, f, repo_type=None: str(path))
    from rwe.hf_data import iter_rows

    assert list(iter_rows("x/y", "train", ["sentence1", "score"])) == [("a", 0.5), ("b", 1.0)]
