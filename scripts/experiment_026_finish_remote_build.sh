#!/usr/bin/env bash
set -euo pipefail
cd /workspace/e026/source
git apply --ignore-space-change --check e026-instrumentation.diff
git apply --ignore-space-change e026-instrumentation.diff
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_RPC=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86-real -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF
cmake --build build --parallel 6 --target ggml-rpc-server
printf 'BUILD_READY %s\n' "$(date --iso-8601=ns)"
sha256sum build/bin/ggml-rpc-server build/bin/*.so*
