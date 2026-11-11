#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
.venv/bin/python - <<'CHECK'
import json
from pathlib import Path
for s in ['a','b']:
 r=json.loads(Path(f'runs/c21_small_{s}_refine_s20261129/complete.json').read_text())
 assert r['status']=='small_fit_gate_passed',r
CHECK
while [ ! -f features/c21_phase/complete.json ]; do sleep 10; done
./run_gpu3.sh src/c21_pair_benchmark.py
./run_gpu3.sh src/c21_pair.py
