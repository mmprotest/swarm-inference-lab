#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWARM_HOME="${SWARM_HOME:-/opt/swarm}"
PYTHON="$SWARM_HOME/venv/bin/python"

: "${RUNTIME_CERTIFICATE_URL:?set immutable RUNTIME_CERTIFICATE_URL}"
: "${PHYSICAL_CERTIFICATE_SHA256:?set canary-reported PHYSICAL_CERTIFICATE_SHA256}"
: "${QUALIFIED_RUNTIME_URL:?set immutable QUALIFIED_RUNTIME_URL}"
mkdir -p "$SWARM_HOME/certificates" "$SWARM_HOME/native"
CERT="$SWARM_HOME/certificates/linux-runtime-certificate.json"
RUNTIME="$SWARM_HOME/native/libcoli_cuda-sm86.so"
curl --fail --location --continue-at - --output "$CERT.partial" "$RUNTIME_CERTIFICATE_URL"
printf '%s  %s
' "$PHYSICAL_CERTIFICATE_SHA256" "$CERT.partial" | sha256sum -c -
mv "$CERT.partial" "$CERT"
RUNTIME_SHA="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["binary"]["sha256"])' "$CERT")"
curl --fail --location --continue-at - --output "$RUNTIME.partial" "$QUALIFIED_RUNTIME_URL"
printf '%s  %s
' "$RUNTIME_SHA" "$RUNTIME.partial" | sha256sum -c -
install -m 0555 "$RUNTIME.partial" "$RUNTIME.new"
mv "$RUNTIME.new" "$RUNTIME"
rm -f -- "$RUNTIME.partial"
"$PYTHON" -c 'from pathlib import Path; from swarm_inference.experiments.experiment_014.runtime_qualification import validate_linux_runtime_certificate; import sys; validate_linux_runtime_certificate(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))' \
  "$CERT" "$ROOT/native/native-source-manifest.json" "$ROOT/manifests/k3-3090-placement-manifest.json"
