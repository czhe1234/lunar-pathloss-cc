#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
./run_gpu.sh 1 src/c22_train.py --variant c22_full_low --steps 2 --eval-every 2 --suffix _smoke
./run_gpu.sh 1 src/c22_train.py --variant c22_full_low --steps 6000 --eval-every 1000
