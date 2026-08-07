#!/usr/bin/env bash

config_value() {
  uv run python -c '
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    value = yaml.safe_load(stream)
for component in sys.argv[2].split("."):
    value = value[component]
print(str(value).lower() if isinstance(value, bool) else value)
' "$CONFIG_PATH" "$1"
}

read_config_list() {
  uv run python -c '
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    value = yaml.safe_load(stream)
for component in sys.argv[2].split("."):
    value = value[component]
print("\n".join(str(item) for item in value))
' "$CONFIG_PATH" "$1"
}

json_value() {
  uv run python -c '
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for component in sys.argv[2].split("."):
    value = value[component]
print(value)
' "$1" "$2"
}

healthy_worker_count() {
  uv run python -c '
import json
import sys
workers = json.load(open(sys.argv[1], encoding="utf-8"))["workers"]
ready = [worker for worker in workers if worker["detail"] == "healthy" and worker["last_error"] is None]
print(len(ready))
' "$1"
}

allocate_ports() {
  uv run python -c '
import socket
sockets = []
try:
    for _ in range(4):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("0.0.0.0", 0))
        sockets.append(listener)
    print("\n".join(str(listener.getsockname()[1]) for listener in sockets))
finally:
    for listener in sockets:
        listener.close()
'
}

run_timed() {
  local timeout_seconds=$1
  shift
  uv run python -c '
import subprocess
import sys
completed = subprocess.run(sys.argv[2:], timeout=float(sys.argv[1]), check=False)
raise SystemExit(completed.returncode)
' "$timeout_seconds" "$@"
}

validate_initial_workers() {
  uv run python -c '
import json
import sys
workers = json.load(open(sys.argv[1], encoding="utf-8"))["workers"]
if len(workers) != 1:
    raise SystemExit(f"expected exactly one healthy worker before the run, found {len(workers)}")
worker = workers[0]
if worker["capability"]["backend"] != "torch-mps" or worker["loaded_stages"]:
    raise SystemExit("the persistent worker must be an unloaded torch-mps worker")
' "$1"
}

write_provenance() {
  uv run python -c '
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
output, repository, config = sys.argv[1:]
metadata = {
    "captured_at": datetime.now(timezone.utc).isoformat(),
    "repository_commit": subprocess.check_output(["git", "-C", repository, "rev-parse", "HEAD"], text=True).strip(),
    "working_tree_dirty": bool(subprocess.check_output(["git", "-C", repository, "status", "--porcelain"], text=True).strip()),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "configuration": config,
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(metadata, stream, indent=2, sort_keys=True)
    stream.write("\n")
' "$1" "$REPOSITORY_ROOT" "$CONFIG_PATH"
}

summarize_evidence() {
  uv run python - "$CONFIG_PATH" "$OUTPUT_DIRECTORY" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
run_directory = Path(sys.argv[2])
with config_path.open(encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

def read_json(name):
    with (run_directory / name).open(encoding="utf-8") as stream:
        return json.load(stream)

plan = read_json("plan.json")
assignments = plan["assignments"]
expected_stage_count = config["planning"]["stage_count"]
if len(assignments) != expected_stage_count:
    raise SystemExit("plan did not assign the configured stage count")
ordered = sorted(assignments, key=lambda item: item["stage_id"])
cursor = 0
for expected_stage_id, assignment in enumerate(ordered):
    stage = assignment["assignment"]
    if assignment["stage_id"] != expected_stage_id or stage["layer_start"] != cursor:
        raise SystemExit("plan stage ranges are not ordered and contiguous")
    cursor = stage["layer_end"]
if cursor != config["model"]["layer_count"]:
    raise SystemExit("plan does not cover every configured model layer")

deployment = read_json("deployment.json")
if deployment["phase"] != "ready" or not deployment["ready"]:
    raise SystemExit("deployment did not reach ready")
required = ("loaded", "ownership_verified", "route_installed", "peer_verified")
for worker in deployment["workers"]:
    if not all(worker[field] for field in required):
        raise SystemExit(f"worker {worker['worker_id']} did not pass deployment verification")

events = []
with (run_directory / "inference.ndjson").open(encoding="utf-8") as stream:
    for line in stream:
        if line.strip():
            events.append(json.loads(line))
generated = [event for event in events if event["event_type"] == "TOKEN_GENERATED"]
positions = [event["token_position"] for event in generated]
token_ids = [event["token_id"] for event in generated]
expected_tokens = config["generation"]["expected_token_ids"]
if positions != list(range(len(expected_tokens))) or token_ids != expected_tokens:
    raise SystemExit("generated token sequence does not match the configured exact oracle")
opened = [event for event in events if event["event_type"] == "SESSION_OPENED"]
completed = [event for event in events if event["event_type"] == "REQUEST_COMPLETED"]
if (
    len(opened) != 1
    or f"{expected_stage_count} persistent stages" not in opened[0]["status_detail"]
):
    raise SystemExit("request did not open on the configured persistent stages")
if len(completed) != 1 or completed[0]["final_token_ids"] != expected_tokens:
    raise SystemExit("request did not complete with the configured exact token oracle")
if read_json("sessions.after-request.json")["sessions"]:
    raise SystemExit("request left session state behind")
unload = read_json("unload.json")["deployment"]
if unload["phase"] != "unloaded" or unload["ready"]:
    raise SystemExit("topology did not unload cleanly")
after_workers = read_json("workers.after.json")["workers"]
ready_after = [
    worker
    for worker in after_workers
    if worker["detail"] == "healthy" and worker["last_error"] is None
]
before_worker = read_json("workers.before.json")["workers"][0]
if (
    len(ready_after) != 1
    or ready_after[0]["capability"]["worker_id"]
    != before_worker["capability"]["worker_id"]
    or ready_after[0]["loaded_stages"]
):
    raise SystemExit("cluster did not return to one healthy unloaded worker")

summary = {
    "schema_version": 1,
    "experiment": config["name"],
    "verdict": "PASS",
    "evidence_classification": config["evidence_classification"],
    "execution_mode": config["execution_mode"],
    "physical_distributed_evidence": False,
    "worker_count": len(deployment["workers"]),
    "stage_ranges": [[item["assignment"]["layer_start"], item["assignment"]["layer_end"]] for item in ordered],
    "generated_token_ids": token_ids,
    "generated_text": "".join(event["decoded_text_fragment"] for event in generated),
    "timing_metrics": completed[0]["timing_metrics"],
    "topology_id": deployment["topology_id"],
    "model_revision": deployment["model"]["model_revision"],
}
with (run_directory / "summary.json").open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, sort_keys=True)
    stream.write("\n")
manifest_lines = []
for path in sorted(run_directory.iterdir()):
    if path.is_file() and path.name != "manifest.sha256":
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {path.name}")
(run_directory / "manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}
