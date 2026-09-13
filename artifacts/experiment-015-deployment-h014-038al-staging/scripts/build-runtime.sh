#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWARM_HOME="${SWARM_HOME:-/opt/swarm}"
PYTHON="$SWARM_HOME/venv/bin/python"

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf -- "$BUILD_DIR"' EXIT
cp "$ROOT/native/src/backend_cuda.cu" "$BUILD_DIR/"
cp "$ROOT/native/src/backend_cuda.h" "$BUILD_DIR/"
cp "$ROOT/native/src/backend_gpu_compat.h" "$BUILD_DIR/"
cd "$BUILD_DIR"
nvcc -O3 -std=c++17 -shared -Xcompiler=-fPIC,-Wall,-Wextra \
  -gencode=arch=compute_86,code=sm_86 \
  -gencode=arch=compute_86,code=compute_86 \
  -DCOLI_CUDA_BUILDING_DLL -DCOLI_CUDA_MIN_CC=86 \
  -DCOLI_CUDA_HAS_FORWARD_PTX=1 backend_cuda.cu -lcudart \
  -o libcoli_cuda-sm86.so
install -d "$SWARM_HOME/native"
install -m 0555 libcoli_cuda-sm86.so "$SWARM_HOME/native/libcoli_cuda-sm86.so"
sha256sum "$SWARM_HOME/native/libcoli_cuda-sm86.so" | tee "$SWARM_HOME/native/runtime.sha256"
