#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWARM_HOME="${SWARM_HOME:-/opt/swarm}"
PYTHON="$SWARM_HOME/venv/bin/python"

echo "NOT REQUIRED: Experiment 014 selected the 93-worker whole-layer topology." >&2
echo "The logical fine-worker result is characterization evidence only; do not gate this fleet on it." >&2
exit 78
