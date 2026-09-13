#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWARM_HOME="${SWARM_HOME:-/opt/swarm}"
PYTHON="$SWARM_HOME/venv/bin/python"

: "${EXPECTED_PACKAGE_LOCK_SHA256:?set out-of-band EXPECTED_PACKAGE_LOCK_SHA256}"
command -v python3 >/dev/null
command -v nvidia-smi >/dev/null
GPU_ROW="$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits --id=0)"
python3 -c 'import sys; p=[x.strip() for x in sys.argv[1].split(",")]; assert len(p)==3 and "RTX 3090" in p[0] and p[1]=="8.6" and float(p[2])>=24576' "$GPU_ROW"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 11)'
printf '%s  %s
' "$EXPECTED_PACKAGE_LOCK_SHA256" "$ROOT/package-lock.json" | sha256sum -c -
python3 "$ROOT/scripts/verify-package.py" "$ROOT"
python3 -m venv "$SWARM_HOME/venv"
"$PYTHON" -m pip install --require-hashes \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --requirement "$ROOT/locks/linux-cu130-requirements.txt"
"$PYTHON" -m pip install --no-deps "$ROOT/wheels/swarm_inference_lab-0.1.0rc11-py3-none-any.whl"
"$PYTHON" -m swarm_inference.experiments.experiment_014 --help >/dev/null
