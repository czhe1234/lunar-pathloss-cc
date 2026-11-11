#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
./run_gpu3.sh src/c21_precision.py
./run_gpu3.sh src/c21_train.py --name c21_small_a_refine_s20261129 --geometry 0 --seed 20261129 --resume runs/c21_small_a_s20261119/best.pt --lr .0003 --queries 4096
./run_gpu3.sh src/c21_train.py --name c21_small_b_refine_s20261129 --geometry 1 --seed 20261129 --resume runs/c21_small_b_s20261119/best.pt --lr .0003 --queries 4096
