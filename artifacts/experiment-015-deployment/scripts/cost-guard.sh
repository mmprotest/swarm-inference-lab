#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWARM_HOME="${SWARM_HOME:-/opt/swarm}"
PYTHON="$SWARM_HOME/venv/bin/python"

: "${FLEET_STATUS:?set FLEET_STATUS to the measured fleet JSON}"
"$PYTHON" -m swarm_inference.experiments.experiment_014 fleet-cost-guard \
  --fleet-status "$FLEET_STATUS" --policy "$ROOT/policies/cost-guard.json" \
  --output "$SWARM_HOME/coordinator/fleet-cost-guard.json"
"$PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "ALLOW"' \
  "$SWARM_HOME/coordinator/fleet-cost-guard.json"
