#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0
export TORCHINDUCTOR_COMPILE_THREADS=8
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
mkdir -p "$PWD/.cache/tmp"
export TMPDIR="$PWD/.cache/tmp"
export XDG_CACHE_HOME="$PWD/.cache"
export TORCH_HOME="$PWD/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/torchinductor"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
exec .venv/bin/python "$@"
