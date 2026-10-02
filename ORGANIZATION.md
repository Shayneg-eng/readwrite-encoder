# Organization

Update this file whenever the structure changes.

| Path | Purpose |
|---|---|
| `README.md` | Project overview, status, setup and how to run |
| `requirements.txt`, `pytest.ini` | Dependencies and test configuration |
| `configs/base.yaml` | Default experiment config |
| `scripts/run_all.ps1` | One-command real experiment (`-Quick` for a first look) |
| `docs/specs/` | Design specs |
| `docs/plans/` | Implementation plans, one per milestone group |
| `docs/results/` | Written-up experiment results |
| `src/rwe/` | Code: `config`, `model`, `losses`, `tokenization`, `data`, `hf_data`, `checkpoint`, `train`, `encode`, `eval_sts`, `prepare_data`, `teacher_cache`, `benchmark_scaling`, `synthetic` |
| `tests/` | CPU tests (offline) |

Created at run time, not in git:

| Path | Purpose |
|---|---|
| `data/`, `data_quick/` | Prepared text, tokens, teacher vectors |
| `runs/` | Checkpoints, logs, benchmark CSVs |
