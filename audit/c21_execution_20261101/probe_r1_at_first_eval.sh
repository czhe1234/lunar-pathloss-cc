#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
while [ ! -f runs/c21r1_pair_s20261123/best_B.json ]; do sleep 10; done
env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python src/c21_branch_probe_r1.py
