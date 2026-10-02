# Read-Write Encoder

An O(n) text embedding model. It reads text in chunks while carrying one 384-d vector. The current vector conditions how the next chunk is read, then the model edits the vector. The final vector is the embedding.

Training plan: distill from `all-MiniLM-L6-v2`, then fine-tune contrastively. See `docs/specs/2026-09-29-readwrite-encoder-design.md`.

## Status

- M0-M3 code (model, losses, data prep, teacher cache, training, STS eval, scaling benchmark): written and tested offline (41 CPU tests).
- Offline results (speed scaling, synthetic memory test): `docs/results/2026-09-29-offline-experiments.md`.
- Not yet run: the real distillation and STS numbers (needs your GPU machine and Hugging Face access).
- M4 (contrastive stage): not started.

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
# Install PyTorch with CUDA from https://pytorch.org/get-started/locally/ first, then:
pip install -r requirements.txt
pytest
```

All tests run on CPU and take about 10 seconds.

## Run the real experiment (Windows, PowerShell)

```powershell
.\scripts\run_all.ps1 -Quick   # first look: ~14k texts, 2000 steps
.\scripts\run_all.ps1          # full run: ~800k texts, 30000 steps
```

It prepares text, caches teacher vectors, trains, prints STS-B test Spearman for the student and teacher, and runs the scaling benchmark. Logs are in `runs/<name>/log.csv` and `eval.csv`. Chunk-size ablation: add `--set model.chunk_size=1 train.out_dir=runs/k1` to the train command.
