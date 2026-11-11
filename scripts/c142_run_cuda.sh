#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if (( $# < 1 )); then
    echo "Usage: scripts/c142_run_cuda.sh {1|3|4} [--maps MAP ...]" >&2
    exit 2
fi
card=$1
shift
case "$card" in
    1) device=GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009 ;;
    3) device=GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0 ;;
    4) device=GPU-70b6fceb-c7c4-9467-3c81-097c9284c160 ;;
    *) echo "Only physical cards 1, 3 and 4 are allowed." >&2; exit 2 ;;
esac

check_idle() {
    local jobs stats used utilization
    jobs=$(nvidia-smi -i "$device" --query-compute-apps=pid --format=csv,noheader,nounits) || return 1
    stats=$(nvidia-smi -i "$device" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) || return 1
    read -r used utilization <<< "${stats/,/ }"
    [[ -z "$jobs" && "$used" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || return 1
    (( used <= 256 && utilization == 0 ))
}

for check in 1 2 3; do
    if ! check_idle; then
        echo "Physical card $card ($device) is occupied or its idle state could not be verified." >&2
        exit 75
    fi
    if (( check < 3 )); then sleep 1; fi
done

driver=$(nvidia-smi -i "$device" --query-gpu=driver_version --format=csv,noheader)
optix="$PWD/.cache/c142-optix-$driver"
test -r "$optix/lib/libnvoptix.so.$driver"
test -r "$optix/lib/libnvidia-rtcore.so.$driver"
test -r "$optix/data/nvoptix.bin"
mkdir -p .cache/{c142-drjit,c142-mpl,c142-cuda,c142-optix,tmp}
exec bwrap --ro-bind / / --bind "$PWD" "$PWD" --dev-bind /dev /dev --proc /proc \
    --bind "$PWD/.cache/c142-drjit" "$HOME/.drjit" \
    --tmpfs /usr/share/nvidia --ro-bind "$optix/data/nvoptix.bin" /usr/share/nvidia/nvoptix.bin \
    env PYTHONPATH="$PWD/.rt_packages" PYTHONDONTWRITEBYTECODE=1 \
    DRJIT_LIBOPTIX_PATH="$optix/lib/libnvoptix.so.$driver" \
    LD_LIBRARY_PATH="$optix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    MPLCONFIGDIR="$PWD/.cache/c142-mpl" CUDA_CACHE_PATH="$PWD/.cache/c142-cuda" \
    OPTIX_CACHE_PATH="$PWD/.cache/c142-optix" \
    ./run_gpu.sh "$card" src/c142_backend.py replay --backend cuda "$@"
