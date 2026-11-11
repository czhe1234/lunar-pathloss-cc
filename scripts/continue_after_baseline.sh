#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
parent_pid=$(cat logs/b0_raw_s20261031.pid)
while kill -0 "$parent_pid" 2>/dev/null; do sleep 5; done
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('runs/b0_raw_s20261031')
s=json.loads((p/'progress.json').read_text())
assert s['status']=='completed',s
assert (p/'result.json').exists()
PY
if [ ! -f runs/b0_raw_s20261031/predictions/train_dev_bf16/inference.json ]; then
./run_gpu3.sh scripts/predict.py --run b0_raw_s20261031 --split train --part dev > logs/b0_independent_inference.log 2>&1
fi
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/score_predictions.py --predictions runs/b0_raw_s20261031/predictions/train_dev_bf16 > logs/b0_independent_score.log 2>&1
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/inspect_predictions.py --predictions runs/b0_raw_s20261031/predictions/train_dev_bf16 > logs/b0_error_maps.log 2>&1 &
./run_gpu3.sh scripts/check_geometry_features.py > logs/geometry_sanity.log 2>&1
printf '%s\n' "$$" > logs/b1_geometry_s20261031.pid
exec ./run_gpu3.sh scripts/train.py --run b1_geometry_s20261031 --mode geometry --batch 32 --epochs 60 --patience 10 --seed 20261031 --compile > logs/b1_geometry_s20261031.log 2>&1
