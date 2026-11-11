#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
./run_gpu.sh 4 src/c22_train.py --variant c22_fit128_rewarm --steps 2 --eval-every 2 --suffix _smoke
./run_gpu.sh 4 src/c22_train.py --variant c22_fit128_rewarm --steps 6000 --eval-every 1000
