#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWARM_HOME="${SWARM_HOME:-/opt/swarm}"
PYTHON="$SWARM_HOME/venv/bin/python"

: "${COORDINATOR_ENDPOINT:?set COORDINATOR_ENDPOINT}"
CERT="$SWARM_HOME/certificates/linux-runtime-certificate.json"
test -f "$CERT"
"$PYTHON" -c 'from pathlib import Path; from swarm_inference.experiments.experiment_014.runtime_qualification import validate_linux_runtime_certificate; import sys; validate_linux_runtime_certificate(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))' \
  "$CERT" "$ROOT/native/native-source-manifest.json" "$ROOT/manifests/k3-3090-placement-manifest.json"
"$SWARM_HOME/venv/bin/swarm" workers --coordinator "$COORDINATOR_ENDPOINT" --json > "$SWARM_HOME/coordinator/workers.json"
bash "$ROOT/scripts/cost-guard.sh"
"$PYTHON" -m swarm_inference.experiments.experiment_014 bind-deployment-plan \
  --placement "$ROOT/manifests/k3-3090-placement-manifest.json" \
  --workers-status "$SWARM_HOME/coordinator/workers.json" \
  --runtime-certificate "$CERT" --native-source-manifest "$ROOT/native/native-source-manifest.json" \
  --output "$SWARM_HOME/coordinator/bound-plan.json"
"$SWARM_HOME/venv/bin/swarm" model deploy --coordinator "$COORDINATOR_ENDPOINT" \
  --plan "$SWARM_HOME/coordinator/bound-plan.json"
