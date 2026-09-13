#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWARM_HOME="${SWARM_HOME:-/opt/swarm}"
PYTHON="$SWARM_HOME/venv/bin/python"

: "${WORKER_ID:?set WORKER_ID}"
: "${COORDINATOR_ENDPOINT:?set COORDINATOR_ENDPOINT}"
: "${COORDINATOR_FINGERPRINT:?set pinned COORDINATOR_FINGERPRINT}"
: "${WORKER_ADVERTISE:?set WORKER_ADVERTISE host:50052}"
: "${WORKER_DATA_ADVERTISE:?set WORKER_DATA_ADVERTISE host:50053}"
: "${UPLOAD_MBPS:?set measured UPLOAD_MBPS}"
: "${DOWNLOAD_MBPS:?set measured DOWNLOAD_MBPS}"
: "${NETWORK_RTT_MS:?set measured NETWORK_RTT_MS}"
: "${NETWORK_BANDWIDTH_GBPS:?set measured NETWORK_BANDWIDTH_GBPS}"
STATE="$SWARM_HOME/workers/$WORKER_ID"
CERT="$SWARM_HOME/certificates/linux-runtime-certificate.json"
RUNTIME="$SWARM_HOME/native/libcoli_cuda-sm86.so"
"$PYTHON" -c 'from pathlib import Path; from swarm_inference.experiments.experiment_014.runtime_qualification import validate_linux_runtime_certificate; import sys; validate_linux_runtime_certificate(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))' \
  "$CERT" "$ROOT/native/native-source-manifest.json" "$ROOT/manifests/k3-3090-placement-manifest.json"
ASSIGNMENT="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["assignment_sha256"])' "$STATE/requirements.json")"
REVISION="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_revision"])' "$STATE/requirements.json")"
"$PYTHON" -m swarm_inference.experiments.experiment_014 inspect-deployment-node \
  --runtime "$RUNTIME" --assigned-stage-canary "$STATE/receipts/assigned-stage-canary.json" \
  --output "$STATE/receipts/launch-observation.json" --network-rtt-ms "$NETWORK_RTT_MS" \
  --network-bandwidth-gbps "$NETWORK_BANDWIDTH_GBPS" --assignment-sha256 "$ASSIGNMENT" \
  --checkpoint-revision "$REVISION" --package-version 0.1.0rc11 --disk-path "$STATE"
"$PYTHON" -m swarm_inference.experiments.experiment_014 node-admission \
  --observation "$STATE/receipts/launch-observation.json" \
  --requirements "$STATE/requirements.json" --mode FLEET \
  --output "$STATE/receipts/launch-admission.json"
"$PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "PASS"' \
  "$STATE/receipts/launch-admission.json"
exec "$SWARM_HOME/venv/bin/swarm" worker --coordinator "$COORDINATOR_ENDPOINT" \
  --backend torch-cuda --memory-limit-gb 24 --worker-id "$WORKER_ID" \
  --listen 0.0.0.0:50052 --advertise "$WORKER_ADVERTISE" \
  --identity "$STATE/identity.json" --trusted-coordinator-fingerprint "$COORDINATOR_FINGERPRINT" \
  --model-shard-root "$STATE" --model-snapshot "$STATE/snapshot" \
  --model-identity "$STATE/snapshot/model-identity.json" --no-allow-model-download \
  --stage-runtime --device native-cuda:0 --dtype float32 \
  --data-listen 0.0.0.0:50053 --data-advertise "$WORKER_DATA_ADVERTISE" \
  --max-stage-sessions 8 --upload-bandwidth-mbps "$UPLOAD_MBPS" \
  --download-bandwidth-mbps "$DOWNLOAD_MBPS"
