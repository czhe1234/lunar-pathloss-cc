#!/usr/bin/env bash
set -euo pipefail
cd /home/chenzhe/work/lunar-pathloss-rm-challenge/CC
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 {1|3|4} PYTHON_SCRIPT [ARGS...]" >&2
    exit 2
fi
cc_gpu_card="$1"
shift
case "$cc_gpu_card" in
    1) export CC_GPU_UUID=GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009 ;;
    3) export CC_GPU_UUID=GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0 ;;
    4) export CC_GPU_UUID=GPU-70b6fceb-c7c4-9467-3c81-097c9284c160 ;;
    *) echo "Only physical cards 1, 3, and 4 are authorized." >&2; exit 2 ;;
esac
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$CC_GPU_UUID"
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
# Nonblocking per-card lock coordinates this project's jobs, not other users.
exec flock --nonblock --conflict-exit-code 75 --no-fork "$PWD/.gpu$cc_gpu_card.lock" .venv/bin/python "$@"
