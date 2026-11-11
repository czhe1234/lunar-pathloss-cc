#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
test -f runs/c21_pair_s20261123/complete.json
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python src/c21_compare.py
./run_gpu3.sh src/c21_fit_panel.py
