#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
./run_gpu3.sh src/c21_train_r1.py --name c21r1_small_a_s20261119 --geometry 0 --steps 6500
./run_gpu3.sh src/c21_train_r1.py --name c21r1_small_b_s20261119 --geometry 1 --steps 6500
./run_gpu3.sh src/c21_pair_r1.py
