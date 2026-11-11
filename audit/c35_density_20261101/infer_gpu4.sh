#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
while [[ ! -f runs/c34_periodic8/predictions/train_dev_c34_fresh_shard0of3/inference.json ]]; do sleep 2; done
sleep 2
exec ./run_gpu.sh 4 scripts/c34_predict.py --run c35_periodic8 --fresh --shard 0 --shards 3
