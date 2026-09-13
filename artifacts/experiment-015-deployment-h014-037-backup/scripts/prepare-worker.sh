#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWARM_HOME="${SWARM_HOME:-/opt/swarm}"
PYTHON="$SWARM_HOME/venv/bin/python"

: "${WORKER_ID:?set WORKER_ID, for example k3-worker-089}"
: "${NETWORK_RTT_MS:?set measured NETWORK_RTT_MS}"
: "${NETWORK_BANDWIDTH_GBPS:?set measured NETWORK_BANDWIDTH_GBPS}"
REQ="$ROOT/requirements/pre-canary/$WORKER_ID.json"
STATE="$SWARM_HOME/workers/$WORKER_ID"
mkdir -p "$STATE/receipts" "$STATE/cache" "$STATE/packages"
ASSIGNMENT="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["assignment_sha256"])' "$REQ")"
REVISION="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_revision"])' "$REQ")"
"$PYTHON" -m swarm_inference.experiments.experiment_014 inspect-deployment-node \
  --output "$STATE/receipts/preflight-observation.json" \
  --network-rtt-ms "$NETWORK_RTT_MS" \
  --network-bandwidth-gbps "$NETWORK_BANDWIDTH_GBPS" \
  --assignment-sha256 "$ASSIGNMENT" --checkpoint-revision "$REVISION" \
  --package-version 0.1.0rc11 --disk-path "$STATE"
"$PYTHON" -m swarm_inference.experiments.experiment_014 node-admission \
  --observation "$STATE/receipts/preflight-observation.json" --requirements "$REQ" \
  --mode PRE_CANARY --output "$STATE/receipts/preflight-admission.json"
"$PYTHON" -m swarm_inference.experiments.experiment_014 acquire-worker \
  --distribution-manifest "$ROOT/manifests/k3-weight-distribution-manifest.json" \
  --worker-id "$WORKER_ID" --cache-directory "$STATE/cache" \
  --output "$STATE/packages/$WORKER_ID.safetensors"
"$PYTHON" -m swarm_inference.experiments.experiment_014 activate-worker-snapshot \
  --placement "$ROOT/manifests/k3-3090-placement-manifest.json" \
  --worker-id "$WORKER_ID" --package "$STATE/packages/$WORKER_ID.safetensors" \
  --config "$ROOT/model-metadata/config.json" --output-directory "$STATE/snapshot"
test -f "$STATE/snapshot/activation.json"
