#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
./run_gpu3.sh scripts/c3_check.py
./run_gpu3.sh scripts/c3_cache.py --split train --batch 16
./run_gpu3.sh scripts/c3_train.py --run c3_surface_profile_s20261031 --epochs 60 --batch 32 --compile
