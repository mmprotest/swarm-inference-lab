#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWARM_HOME="${SWARM_HOME:-/opt/swarm}"
PYTHON="$SWARM_HOME/venv/bin/python"

: "${COORDINATOR_ADVERTISE:?set COORDINATOR_ADVERTISE host:port}"
exec "$SWARM_HOME/venv/bin/swarm" coordinator --config "$ROOT/config/coordinator.yaml" \
  --listen 0.0.0.0:50051 --advertise "$COORDINATOR_ADVERTISE" \
  --state "$SWARM_HOME/coordinator" --model-path "$ROOT/model-metadata" --dtype float32
