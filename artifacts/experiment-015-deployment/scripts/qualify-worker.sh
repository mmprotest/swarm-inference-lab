#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWARM_HOME="${SWARM_HOME:-/opt/swarm}"
PYTHON="$SWARM_HOME/venv/bin/python"

: "${WORKER_ID:?set WORKER_ID}"
: "${NETWORK_RTT_MS:?set measured NETWORK_RTT_MS}"
: "${NETWORK_BANDWIDTH_GBPS:?set measured NETWORK_BANDWIDTH_GBPS}"
STATE="$SWARM_HOME/workers/$WORKER_ID"
CERT="$SWARM_HOME/certificates/linux-runtime-certificate.json"
RUNTIME="$SWARM_HOME/native/libcoli_cuda-sm86.so"
test -f "$CERT"
test -f "$RUNTIME"
RUNTIME_SHA="$($PYTHON -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$RUNTIME")"
"$PYTHON" -m swarm_inference.experiments.experiment_014 worker-requirements \
  --placement "$ROOT/manifests/k3-3090-placement-manifest.json" \
  --distribution "$ROOT/manifests/k3-weight-distribution-manifest.json" \
  --worker-id "$WORKER_ID" --package-version 0.1.0rc11 \
  --runtime-sha256 "$RUNTIME_SHA" --output "$STATE/requirements.json"
"$PYTHON" -m swarm_inference.experiments.experiment_014 assigned-stage-canary \
  --placement "$ROOT/manifests/k3-3090-placement-manifest.json" \
  --native-source-manifest "$ROOT/native/native-source-manifest.json" \
  --runtime-certificate "$CERT" --runtime "$RUNTIME" --snapshot "$STATE/snapshot" \
  --worker-id "$WORKER_ID" --output "$STATE/receipts/assigned-stage-canary.json"
ASSIGNMENT="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["assignment_sha256"])' "$STATE/requirements.json")"
REVISION="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_revision"])' "$STATE/requirements.json")"
"$PYTHON" -m swarm_inference.experiments.experiment_014 inspect-deployment-node \
  --runtime "$RUNTIME" --assigned-stage-canary "$STATE/receipts/assigned-stage-canary.json" \
  --output "$STATE/receipts/fleet-observation.json" --network-rtt-ms "$NETWORK_RTT_MS" \
  --network-bandwidth-gbps "$NETWORK_BANDWIDTH_GBPS" --assignment-sha256 "$ASSIGNMENT" \
  --checkpoint-revision "$REVISION" --package-version 0.1.0rc11 --disk-path "$STATE"
"$PYTHON" -m swarm_inference.experiments.experiment_014 node-admission \
  --observation "$STATE/receipts/fleet-observation.json" \
  --requirements "$STATE/requirements.json" --mode FLEET \
  --output "$STATE/receipts/fleet-admission.json"
"$PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "PASS"' \
  "$STATE/receipts/fleet-admission.json"
if ! test -f "$STATE/identity.json"; then
  "$SWARM_HOME/venv/bin/swarm" identity create --path "$STATE/identity.json" \
    --kind worker --json >/dev/null
fi
chmod 0600 "$STATE/identity.json"
"$SWARM_HOME/venv/bin/swarm" identity show --path "$STATE/identity.json" --json \
  > "$STATE/receipts/worker-public-identity.json"
