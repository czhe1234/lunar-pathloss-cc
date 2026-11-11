#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
./run_gpu.sh 3 src/c25_train.py --variant c25_batch64_scaled --steps 16 --eval-every 16 --suffix _smoke
./run_gpu.sh 3 src/c25_train.py --variant c25_batch64_scaled --steps 6000 --eval-every 1000
./run_gpu.sh 3 src/c22_evaluate.py --run c25_batch64_scaled --checkpoint best.pt --part dev --dest dev_complete
