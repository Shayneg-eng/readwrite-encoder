# Full v1 experiment: data -> teacher cache -> distill -> STS eval -> scaling benchmark.
# Run from the project root in PowerShell with the venv active and CUDA PyTorch installed.
#   .\scripts\run_all.ps1 -Quick   # ~14k texts, 2000 steps: a first look in well under an hour
#   .\scripts\run_all.ps1          # ~800k texts, 30000 steps: the real run
param([switch]$Quick)
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

function Step($label, [scriptblock]$cmd) {
    Write-Host "== $label"
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "$label failed" }
}

if ($Quick) { $dir = "data_quick"; $scale = "0.02"; $out = "runs/quick"; $extra = @("train.steps=2000", "train.eval_every=500", "train.ckpt_every=500") }
else        { $dir = "data";       $scale = "1.0";  $out = "runs/base";  $extra = @() }

Step "install dependencies" { pip install -r requirements.txt }
Step "tests" { python -m pytest -q }
Step "prepare text" { python -m rwe.prepare_data --out "$dir/texts.jsonl" --scale $scale }
Step "teacher cache" { python -m rwe.teacher_cache --texts "$dir/texts.jsonl" --out-dir $dir }
Step "train" { python -m rwe.train --config configs/base.yaml --sts --init-tok-emb "$dir/teacher_tok_emb.pt" --set "data.dir=$dir" "train.out_dir=$out" @extra }
Step "STS-B test vs teacher" { python -m rwe.eval_sts --checkpoint "$out/last.pt" --split test }
Step "scaling benchmark" { python -m rwe.benchmark_scaling --checkpoint "$out/last.pt" --out "$out/benchmark.csv" }
Write-Host "Done. Logs: $out/log.csv and $out/eval.csv"
