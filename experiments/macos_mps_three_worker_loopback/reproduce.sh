#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$SCRIPT_DIR/macos_mps_loopback_lib.sh"
CONFIG_PATH="$REPOSITORY_ROOT/configs/experiments/macos_mps_three_worker_loopback.yaml"
STATE_ROOT="$REPOSITORY_ROOT/.local-state"
OUTPUT_DIRECTORY=""

usage() {
  printf '%s\n' \
    "Usage: $0 [--config PATH] [--state-root PATH] [--output-directory PATH]" \
    "" \
    "Runs the real OLMoE model through three process-isolated MPS stages on one Mac."
}

while (($#)); do
  case "$1" in
    --config)
      CONFIG_PATH=$2
      shift 2
      ;;
    --state-root)
      STATE_ROOT=$2
      shift 2
      ;;
    --output-directory)
      OUTPUT_DIRECTORY=$2
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$REPOSITORY_ROOT"
CONFIG_PATH=$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")
STATE_ROOT=$(cd "$STATE_ROOT" && pwd)

if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
  printf '%s\n' "This reproduction requires Apple Silicon macOS." >&2
  exit 2
fi

command -v uv >/dev/null
[[ -f "$CONFIG_PATH" ]]
[[ -f "$STATE_ROOT/security/cluster.json" ]]
[[ -f "$STATE_ROOT/security/node-configuration.json" ]]
[[ -f "$STATE_ROOT/security/node-identity.json" ]]

CONFIG_SCHEMA_VERSION=$(config_value schema_version)
if [[ "$CONFIG_SCHEMA_VERSION" != macos-mps-three-worker-loopback-v1 ]]; then
  printf 'Unsupported configuration schema: %s\n' "$CONFIG_SCHEMA_VERSION" >&2
  exit 2
fi
EXPERIMENT_NAME=$(config_value name)
OUTPUT_ROOT=$(config_value output_root)
MODEL_ID=$(config_value model.model_id)
MODEL_REVISION=$(config_value model.revision)
TOKENIZER_REVISION=$(config_value model.tokenizer_revision)
DTYPE=$(config_value model.dtype)
DEVICE=$(config_value model.device)
STAGE_COUNT=$(config_value planning.stage_count)
PARTITION=$(config_value planning.partition)
MODE=$(config_value planning.mode)
MEMORY_LIMIT_GIB=$(config_value additional_workers.memory_limit_gib)
PROMPT=$(config_value generation.prompt)
MAX_NEW_TOKENS=$(config_value generation.max_new_tokens)
TEMPERATURE=$(config_value generation.temperature)
SEED=$(config_value generation.seed)
WORKER_START_TIMEOUT=$(config_value timeouts.worker_start_seconds)
DEPLOYMENT_TIMEOUT=$(config_value timeouts.deployment_seconds)
REQUEST_TIMEOUT=$(config_value timeouts.request_seconds)
SHUTDOWN_TIMEOUT=$(config_value timeouts.shutdown_seconds)
WORKER_IDS=($(read_config_list additional_workers.worker_ids))
ALLOCATED_PORTS=($(allocate_ports))
CONTROL_PORTS=(${ALLOCATED_PORTS[0]} ${ALLOCATED_PORTS[1]})
DATA_PORTS=(${ALLOCATED_PORTS[2]} ${ALLOCATED_PORTS[3]})

if [[ ${#WORKER_IDS[@]} -ne 2 ]]; then
  printf '%s\n' "The three-worker loopback requires exactly two additional workers." >&2
  exit 2
fi

COORDINATOR_ENDPOINT=$(json_value "$STATE_ROOT/security/cluster.json" coordinator_endpoint)
COORDINATOR_FINGERPRINT=$(json_value "$STATE_ROOT/security/cluster.json" coordinator_fingerprint)
SOURCE_ADDRESS=$(json_value "$STATE_ROOT/security/node-configuration.json" endpoints.source_address)
MODEL_DIRECTORY=${MODEL_ID//\//--}
MODEL_SNAPSHOT="$STATE_ROOT/artifacts/source-cache/materialized/$MODEL_DIRECTORY/snapshots/$MODEL_REVISION"
[[ -d "$MODEL_SNAPSHOT" ]]

if [[ -z "$OUTPUT_DIRECTORY" ]]; then
  RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  OUTPUT_DIRECTORY="$REPOSITORY_ROOT/$OUTPUT_ROOT/$EXPERIMENT_NAME-$RUN_STAMP"
elif [[ "$OUTPUT_DIRECTORY" != /* ]]; then
  OUTPUT_DIRECTORY="$REPOSITORY_ROOT/$OUTPUT_DIRECTORY"
fi
if [[ -e "$OUTPUT_DIRECTORY" ]]; then
  printf 'Output directory already exists: %s\n' "$OUTPUT_DIRECTORY" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIRECTORY"
cp "$CONFIG_PATH" "$OUTPUT_DIRECTORY/resolved-config.yaml"
uv run python -c '
import json
import sys
control = [int(sys.argv[2]), int(sys.argv[3])]
data = [int(sys.argv[4]), int(sys.argv[5])]
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump({"control_ports": control, "data_ports": data}, stream, indent=2, sort_keys=True)
    stream.write("\n")
' "$OUTPUT_DIRECTORY/runtime-endpoints.json" "${CONTROL_PORTS[@]}" "${DATA_PORTS[@]}"

uv run swarm node status --state-root "$STATE_ROOT" --json >"$OUTPUT_DIRECTORY/node.before.json"
uv run swarm workers --coordinator "$COORDINATOR_ENDPOINT" --no-include-unhealthy --json >"$OUTPUT_DIRECTORY/workers.before.json"
validate_initial_workers "$OUTPUT_DIRECTORY/workers.before.json"
write_provenance "$OUTPUT_DIRECTORY/provenance.json"

WORKER_PID_1=""
WORKER_PID_2=""
TOPOLOGY_ID=""

emergency_cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$TOPOLOGY_ID" ]]; then
    run_timed "$SHUTDOWN_TIMEOUT" uv run swarm model unload \
      --coordinator "$COORDINATOR_ENDPOINT" \
      --topology-id "$TOPOLOGY_ID" \
      >"$OUTPUT_DIRECTORY/emergency-unload.json" 2>"$OUTPUT_DIRECTORY/emergency-unload.stderr"
  fi
  [[ -z "$WORKER_PID_1" ]] || kill "$WORKER_PID_1" 2>/dev/null
  [[ -z "$WORKER_PID_2" ]] || kill "$WORKER_PID_2" 2>/dev/null
  [[ -z "$WORKER_PID_1" ]] || wait "$WORKER_PID_1" 2>/dev/null
  [[ -z "$WORKER_PID_2" ]] || wait "$WORKER_PID_2" 2>/dev/null
  exit "$exit_status"
}
trap emergency_cleanup EXIT INT TERM

start_worker() {
  local worker_index=$1
  local worker_label=$((worker_index + 2))
  uv run swarm worker \
    --coordinator "$COORDINATOR_ENDPOINT" \
    --backend torch-mps \
    --memory-limit-gb "$MEMORY_LIMIT_GIB" \
    --listen "0.0.0.0:${CONTROL_PORTS[$worker_index]}" \
    --advertise "$SOURCE_ADDRESS:${CONTROL_PORTS[$worker_index]}" \
    --identity "$STATE_ROOT/security/node-identity.json" \
    --trusted-coordinator-fingerprint "$COORDINATOR_FINGERPRINT" \
    --worker-id "${WORKER_IDS[$worker_index]}" \
    --data-listen "0.0.0.0:${DATA_PORTS[$worker_index]}" \
    --data-advertise "$SOURCE_ADDRESS:${DATA_PORTS[$worker_index]}" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --model-cache-dir "$STATE_ROOT/artifacts/source-cache" \
    --model-snapshot "$MODEL_SNAPSHOT" \
    --stage-runtime \
    >"$OUTPUT_DIRECTORY/worker-$worker_label.log" 2>&1 &
  STARTED_WORKER_PID=$!
}

start_worker 0
WORKER_PID_1=$STARTED_WORKER_PID
start_worker 1
WORKER_PID_2=$STARTED_WORKER_PID

WORKER_DEADLINE=$((SECONDS + WORKER_START_TIMEOUT))
while true; do
  uv run swarm workers --coordinator "$COORDINATOR_ENDPOINT" --no-include-unhealthy --json >"$OUTPUT_DIRECTORY/workers.ready.json"
  HEALTHY_COUNT=$(healthy_worker_count "$OUTPUT_DIRECTORY/workers.ready.json")
  [[ "$HEALTHY_COUNT" -eq 3 ]] && break
  if ((SECONDS >= WORKER_DEADLINE)); then
    printf 'Timed out waiting for three healthy workers; observed %s.\n' "$HEALTHY_COUNT" >&2
    exit 1
  fi
  sleep 1
done

uv run swarm model plan \
  --coordinator "$COORDINATOR_ENDPOINT" \
  --model-id "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --dtype "$DTYPE" \
  --stage-count "$STAGE_COUNT" \
  --partition "$PARTITION" \
  --mode "$MODE" \
  --require-distributed \
  --output "$OUTPUT_DIRECTORY/plan.json" \
  >"$OUTPUT_DIRECTORY/plan.stdout"

run_timed "$DEPLOYMENT_TIMEOUT" uv run swarm model deploy \
  --coordinator "$COORDINATOR_ENDPOINT" \
  --plan "$OUTPUT_DIRECTORY/plan.json" \
  >"$OUTPUT_DIRECTORY/deployment.json"
TOPOLOGY_ID=$(json_value "$OUTPUT_DIRECTORY/deployment.json" topology_id)

uv run swarm topology --coordinator "$COORDINATOR_ENDPOINT" --json >"$OUTPUT_DIRECTORY/topology.ready.json"
run_timed "$REQUEST_TIMEOUT" uv run swarm submit \
  --coordinator "$COORDINATOR_ENDPOINT" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  --model-id "$MODEL_ID" \
  --model-revision "$MODEL_REVISION" \
  --stream \
  --ndjson \
  >"$OUTPUT_DIRECTORY/inference.ndjson"
uv run swarm sessions --coordinator "$COORDINATOR_ENDPOINT" --json >"$OUTPUT_DIRECTORY/sessions.after-request.json"

run_timed "$SHUTDOWN_TIMEOUT" uv run swarm model unload \
  --coordinator "$COORDINATOR_ENDPOINT" \
  --topology-id "$TOPOLOGY_ID" \
  >"$OUTPUT_DIRECTORY/unload.json"
TOPOLOGY_ID=""
kill "$WORKER_PID_1" "$WORKER_PID_2"
wait "$WORKER_PID_1" 2>/dev/null || true
wait "$WORKER_PID_2" 2>/dev/null || true
WORKER_PID_1=""
WORKER_PID_2=""

SHUTDOWN_DEADLINE=$((SECONDS + SHUTDOWN_TIMEOUT))
while true; do
  uv run swarm workers --coordinator "$COORDINATOR_ENDPOINT" --no-include-unhealthy --json >"$OUTPUT_DIRECTORY/workers.after.json"
  HEALTHY_COUNT=$(healthy_worker_count "$OUTPUT_DIRECTORY/workers.after.json")
  [[ "$HEALTHY_COUNT" -eq 1 ]] && break
  if ((SECONDS >= SHUTDOWN_DEADLINE)); then
    printf 'Timed out waiting for the temporary workers to expire; observed %s healthy workers.\n' "$HEALTHY_COUNT" >&2
    exit 1
  fi
  sleep 1
done

summarize_evidence

trap - EXIT INT TERM
printf 'evidence=%s\n' "$OUTPUT_DIRECTORY"
