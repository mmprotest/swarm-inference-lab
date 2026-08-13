#!/usr/bin/env bash
set -euo pipefail

: "${SWARM_RUN_ID:?SWARM_RUN_ID is required}"
: "${SWARM_CONTROLLER:?SWARM_CONTROLLER is required}"
: "${SWARM_RUN_CREDENTIAL_FILE:?SWARM_RUN_CREDENTIAL_FILE is required}"
: "${SWARM_WORKER_MANIFEST_ROOT:?SWARM_WORKER_MANIFEST_ROOT is required}"

test -f "${SWARM_RUN_CREDENTIAL_FILE}"
test -d "${SWARM_WORKER_MANIFEST_ROOT}"

# Vast already runs this repository's image as the instance container.  Docker
# nesting is unsupported and intentionally absent.  The provisioning state
# machine installs the runtime-only credential and manifests before invoking
# this unattended in-container bootstrap.
exec python -m swarm_inference.experiments.experiment_020.worker_main \
  host-agent --workers-per-pod 8
