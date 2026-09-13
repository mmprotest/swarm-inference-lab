#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWARM_HOME="${SWARM_HOME:-/opt/swarm}"
PYTHON="$SWARM_HOME/venv/bin/python"

: "${WORKER_PUBLIC_IDENTITIES:?directory containing exactly 93 public identity JSON files}"
COUNT="$(find "$WORKER_PUBLIC_IDENTITIES" -maxdepth 1 -type f -name 'k3-worker-*.json' | wc -l)"
test "$COUNT" -eq 93
for IDENTITY in "$WORKER_PUBLIC_IDENTITIES"/k3-worker-*.json; do
  FINGERPRINT="$($PYTHON -c 'import json,sys; value=json.load(open(sys.argv[1])); assert "private_key" not in value; print(value["fingerprint"])' "$IDENTITY")"
  "$SWARM_HOME/venv/bin/swarm" identity trust --coordinator-state "$SWARM_HOME/coordinator" \
    --fingerprint "$FINGERPRINT" --label "$(basename "$IDENTITY" .json)" \
    --notes experiment-015-hash-locked-fleet
done
