#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cc147_card=$1
shift
case "$cc147_card" in
1) export CC_GPU_UUID=GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009 ;;
3) export CC_GPU_UUID=GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0 ;;
4) export CC_GPU_UUID=GPU-70b6fceb-c7c4-9467-3c81-097c9284c160 ;;
*) exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="$CC_GPU_UUID"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
driver=$(nvidia-smi -i "$CC_GPU_UUID" --query-gpu=driver_version --format=csv,noheader)
optix="$PWD/.cache/c142-optix-$driver"
test -r "$optix/lib/libnvoptix.so.$driver"
test -r "$optix/data/nvoptix.bin"
mkdir -p .cache/{c147-drjit,c147-mpl,c147-cuda,c147-optix,tmp}
exec bwrap --ro-bind / / --bind "$PWD" "$PWD" --dev-bind /dev /dev --proc /proc \
 --bind "$PWD/.cache/c147-drjit" "$HOME/.drjit" \
 --tmpfs /usr/share/nvidia --ro-bind "$optix/data/nvoptix.bin" /usr/share/nvidia/nvoptix.bin \
 env PYTHONPATH="$PWD/.rt_packages" PYTHONDONTWRITEBYTECODE=1 \
 DRJIT_LIBOPTIX_PATH="$optix/lib/libnvoptix.so.$driver" \
 LD_LIBRARY_PATH="$optix/lib" \
 MPLCONFIGDIR="$PWD/.cache/c147-mpl" CUDA_CACHE_PATH="$PWD/.cache/c147-cuda" \
 OPTIX_CACHE_PATH="$PWD/.cache/c147-optix" TMPDIR="$PWD/.cache/tmp" \
 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
 .venv/bin/python -u "${CC147_SCRIPT:-src/c147_sionna.py}" "$@"
