#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/e027
MODEL="$ROOT/models/Qwen3.8-27B-Q4_K_M.gguf"
MODEL_SHA=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
MODEL_URL="https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/0669b98607d47046c7c2b3f801011d54a08cfccf/Qwen3.8-27B-Q4_K_M.gguf"

cd "$ROOT"
printf 'E027_BOOTSTRAP_START %s\n' "$(date --iso-8601=ns)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq aria2 ca-certificates cmake ninja-build build-essential git python3

mkdir -p source models
tar xzf remote-source.tar.gz -C source
cd source
git apply --ignore-space-change --check e027-stage-range.patch
git apply --ignore-space-change e027-stage-range.patch

(
    cd "$ROOT/models"
    aria2c --continue=true --max-connection-per-server=8 --split=8 \
        --min-split-size=16M --file-allocation=none --auto-file-renaming=false \
        --allow-overwrite=false --summary-interval=20 \
        --out="$(basename "$MODEL")" "$MODEL_URL"
    printf '%s  %s\n' "$MODEL_SHA" "$MODEL" | sha256sum --check --strict
) > "$ROOT/model-download.log" 2>&1 &
download_pid=$!

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DGGML_RPC=OFF -DCMAKE_CUDA_ARCHITECTURES=native -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_OPENSSL=OFF
cmake --build build --parallel 8 --target llama-e027-stage
wait "$download_pid"

printf 'E027_BOOTSTRAP_DONE %s\n' "$(date --iso-8601=ns)"
sha256sum build/bin/llama-e027-stage build/bin/lib*.so* "$MODEL"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,compute_cap --format=csv
nvcc --version | tail -n 4
touch "$ROOT/READY"
