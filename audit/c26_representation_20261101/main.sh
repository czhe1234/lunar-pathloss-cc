#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
./run_gpu.sh 1 src/c26_features.py --kind main
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 .venv/bin/python src/c26_ridge.py --kind main
