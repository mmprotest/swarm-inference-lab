#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWARM_HOME="${SWARM_HOME:-/opt/swarm}"
PYTHON="$SWARM_HOME/venv/bin/python"

: "${COORDINATOR_ENDPOINT:?set COORDINATOR_ENDPOINT}"
: "${TOPOLOGY_ID:?set exact TOPOLOGY_ID returned by deployment}"
"$SWARM_HOME/venv/bin/swarm" model unload --coordinator "$COORDINATOR_ENDPOINT" \
  --topology-id "$TOPOLOGY_ID" --force
echo "Model unloaded. Provider instances must be terminated through the approved cost-guard workflow."
