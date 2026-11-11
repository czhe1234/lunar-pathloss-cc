#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/src"
compile_cuda() {
    /usr/local/cuda-11.8/bin/nvcc -O3 -Xcompiler -fPIC -shared -gencode arch=compute_80,code=compute_80 -gencode arch=compute_80,code=sm_80 "$@"
}
g++ -O3 -fPIC -fopenmp -shared c66_lattice.cpp -o libc66_lattice.so
compile_cuda c67_transport.cu -o libc67_transport.so
compile_cuda c70_geometry.cu -o libc70_geometry.so
compile_cuda c73_visibility.cu -o libc73_visibility.so
compile_cuda --fmad=false c74_surface.cu -o libc74_surface.so
