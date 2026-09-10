#!/usr/bin/env bash
set -euo pipefail
cd /workspace/e026
printf 'BOOTSTRAP_START %s\n' "$(date --iso-8601=ns)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq cmake ninja-build build-essential git python3
mkdir -p source
tar xzf remote-source-001.tar.gz -C source
cd source
git apply --check e026-instrumentation.diff
git apply e026-instrumentation.diff
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_RPC=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86-real -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF
cmake --build build --parallel 6 --target ggml-rpc-server
printf 'BOOTSTRAP_DONE %s\n' "$(date --iso-8601=ns)"
sha256sum build/bin/ggml-rpc-server build/bin/*.so*
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,compute_cap --format=csv
nvcc --version
g++ --version
ls -l build/bin/ggml-rpc-server
