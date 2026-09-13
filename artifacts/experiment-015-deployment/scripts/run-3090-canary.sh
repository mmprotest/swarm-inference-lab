#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWARM_HOME="${SWARM_HOME:-/opt/swarm}"
PYTHON="$SWARM_HOME/venv/bin/python"

RUNTIME="$SWARM_HOME/native/libcoli_cuda-sm86.so"
CERT_DIR="$SWARM_HOME/certificates"
mkdir -p "$CERT_DIR" "$SWARM_HOME/canary"
for ID in k3-worker-000 k3-worker-089 k3-worker-091 k3-worker-092; do
  test -f "$SWARM_HOME/workers/$ID/snapshot/activation.json"
done
"$PYTHON" -m swarm_inference.experiments.experiment_014 physical-3090-canary \
  --placement "$ROOT/manifests/k3-3090-placement-manifest.json" \
  --native-source-manifest "$ROOT/native/native-source-manifest.json" \
  --runtime "$RUNTIME" --fixture-npz "$ROOT/canary/fixtures.npz" \
  --fixture-manifest "$ROOT/canary/fixtures.json" --evidence-root "$ROOT/evidence" \
  --stage-zero-snapshot "$SWARM_HOME/workers/k3-worker-000/snapshot" \
  --kda-snapshot "$SWARM_HOME/workers/k3-worker-089/snapshot" \
  --mla-snapshot "$SWARM_HOME/workers/k3-worker-091/snapshot" \
  --final-snapshot "$SWARM_HOME/workers/k3-worker-092/snapshot" \
  --receipt "$SWARM_HOME/canary/physical-3090-canary.json" \
  --certificate "$CERT_DIR/linux-runtime-certificate.json"
test -f "$CERT_DIR/linux-runtime-certificate.json"
