set -eu
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
while [ ! -f runs/c32_cold/complete.json ]; do sleep 5; done
sleep 2
./run_gpu.sh 1 src/c33_train.py --band 415 --smoke
./run_gpu.sh 1 src/c33_train.py --band 415
