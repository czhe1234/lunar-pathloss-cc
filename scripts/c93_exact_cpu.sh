#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p "$PWD/.cache/c79-drjit" "$PWD/.cache/c79-mpl" "$PWD/.cache/tmp"
exec bwrap --ro-bind / / --bind "$PWD" "$PWD" --bind "$PWD/.cache/c79-drjit" "$HOME/.drjit" \
 env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH="$PWD/.rt_packages" \
 XDG_CACHE_HOME="$PWD/.cache" MPLCONFIGDIR="$PWD/.cache/c79-mpl" TMPDIR="$PWD/.cache/tmp" .venv/bin/python -u src/c93_exact_patch.py "$@"
