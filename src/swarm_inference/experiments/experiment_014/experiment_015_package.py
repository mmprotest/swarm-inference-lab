"""Build and statically certify the fail-closed Experiment 015 handoff."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from swarm_inference.experiments.experiment_014.remote_acquisition import (
    AcquisitionError,
    _resolve_distribution_placement,
)
from swarm_inference.experiments.experiment_014.runtime_qualification import (
    BUILD_ARGUMENTS,
    NATIVE_SOURCE_FILES,
    RuntimeQualificationError,
    build_native_source_manifest,
    validate_native_source_manifest,
)
from swarm_inference.model.kimi_tokenizer import KIMI_TOKENIZER_ASSETS

PACKAGE_SCHEMA = "experiment-015-k3-deployment-package-v1"
VALIDATION_SCHEMA = "experiment-014-k3-deployment-package-validation-v1"
PACKAGE_VERSION = "0.1.0rc11"
PLACEMENT_NAME = "k3-3090-placement-manifest.json"
EXECUTION_NAME = "k3-execution-plan.json"
DISTRIBUTION_NAME = "k3-weight-distribution-manifest.json"
CAPACITY_EVIDENCE_NAME = "h014-038ak-final-capacity-topology-economics.json"
COARSE_NETWORK_EVIDENCE_NAME = "h014-038ah-network-analysis.json"
REQUIRED_SCRIPTS = (
    "build-runtime.sh",
    "install-worker.sh",
    "install-qualified-runtime.sh",
    "prepare-worker.sh",
    "qualify-worker.sh",
    "run-3090-canary.sh",
    "run-worker.sh",
    "run-coordinator.sh",
    "trust-workers.sh",
    "bind-and-deploy.sh",
    "cost-guard.sh",
    "collect-artifacts.sh",
    "cleanup.sh",
    "run-two-gpu-fine-canary.sh",
)


class Experiment015PackageError(RuntimeError):
    """The deployment handoff is incomplete, mutable, or unsafe."""


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Experiment015PackageError(f"invalid JSON document: {path}") from exc
    if not isinstance(value, dict):
        raise Experiment015PackageError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise Experiment015PackageError(f"package source is absent or linked: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _portable_manifest(source: Path, destination: Path, placement: Path) -> None:
    value = _read(source)
    value["placement_manifest"] = placement.name
    value["placement_manifest_sha256"] = _sha256(placement)
    _atomic_json(destination, value)


def _worker_requirements(
    placement: dict[str, Any], distribution: dict[str, Any], worker: dict[str, Any]
) -> dict[str, Any]:
    downloads = {row["worker_id"]: row for row in distribution["workers"]}
    worker_id = str(worker["worker_id"])
    download = downloads[worker_id]
    return {
        "schema_version": "experiment-015-k3-admission-v1",
        "kind": "worker_requirements",
        "status": "PASS",
        "worker_id": worker_id,
        "minimum_vram_bytes": int(worker["memory"]["physical_vram_bytes"]),
        "minimum_disk_free_bytes": int(download["temporary_disk_bytes"]) + 32 * 1024**3,
        "network": placement["network_admission"],
        "assignment_sha256": worker["assignment_sha256"],
        "checkpoint_revision": placement["checkpoint"]["revision"],
        "checkpoint_fingerprint": placement["checkpoint"]["checkpoint_fingerprint"],
        "package_version": PACKAGE_VERSION,
        "runtime_sha256": "",
        "runtime_identity": "PENDING_PHYSICAL_3090_CANARY",
        "owned_layers": worker["owned_layers"],
        "worker_role": worker["worker_role"],
        "source_weight_bytes": worker["source_weight_bytes"],
        "cold_download_bytes": download["download_bytes_cold_cache"],
        "placement_manifest_sha256": _sha256(Path(placement["_path"])),
        "distribution_manifest_sha256": _sha256(Path(distribution["_path"])),
    }


def _standalone_verifier() -> str:
    return '''#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
lock_path = root / "package-lock.json"
lock = json.loads(lock_path.read_text(encoding="utf-8"))
entries = lock.get("files")
if lock.get("status") != "PASS" or not isinstance(entries, dict):
    raise SystemExit("package lock is not passing")
actual = sorted(
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path.name != "package-lock.json"
)
if actual != sorted(entries):
    raise SystemExit("package file set differs from package lock")
for relative, identity in entries.items():
    path = root / relative
    if path.is_symlink() or path.stat().st_size != int(identity["bytes"]):
        raise SystemExit(f"package size/link identity differs: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != identity["sha256"]:
        raise SystemExit(f"package SHA-256 differs: {relative}")
print(json.dumps({"status": "PASS", "file_count": len(entries) + 1}, sort_keys=True))
'''


def _scripts(wheel_name: str) -> dict[str, str]:
    common = '''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWARM_HOME="${SWARM_HOME:-/opt/swarm}"
PYTHON="$SWARM_HOME/venv/bin/python"
'''
    build = common + '''
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf -- "$BUILD_DIR"' EXIT
cp "$ROOT/native/src/backend_cuda.cu" "$BUILD_DIR/"
cp "$ROOT/native/src/backend_cuda.h" "$BUILD_DIR/"
cp "$ROOT/native/src/backend_gpu_compat.h" "$BUILD_DIR/"
cd "$BUILD_DIR"
nvcc -O3 -std=c++17 -shared -Xcompiler=-fPIC,-Wall,-Wextra \\
  -gencode=arch=compute_86,code=sm_86 \\
  -gencode=arch=compute_86,code=compute_86 \\
  -DCOLI_CUDA_BUILDING_DLL -DCOLI_CUDA_MIN_CC=86 \\
  -DCOLI_CUDA_HAS_FORWARD_PTX=1 backend_cuda.cu -lcudart \\
  -o libcoli_cuda-sm86.so
install -d "$SWARM_HOME/native"
install -m 0555 libcoli_cuda-sm86.so "$SWARM_HOME/native/libcoli_cuda-sm86.so"
sha256sum "$SWARM_HOME/native/libcoli_cuda-sm86.so" | tee "$SWARM_HOME/native/runtime.sha256"
'''
    install = common + f'''
: "${{EXPECTED_PACKAGE_LOCK_SHA256:?set out-of-band EXPECTED_PACKAGE_LOCK_SHA256}}"
command -v python3 >/dev/null
command -v nvidia-smi >/dev/null
GPU_ROW="$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits --id=0)"
python3 -c 'import sys; p=[x.strip() for x in sys.argv[1].split(",")]; assert len(p)==3 and "RTX 3090" in p[0] and p[1]=="8.6" and float(p[2])>=24576' "$GPU_ROW"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 11)'
printf '%s  %s\n' "$EXPECTED_PACKAGE_LOCK_SHA256" "$ROOT/package-lock.json" | sha256sum -c -
python3 "$ROOT/scripts/verify-package.py" "$ROOT"
python3 -m venv "$SWARM_HOME/venv"
"$PYTHON" -m pip install --require-hashes \\
  --extra-index-url https://download.pytorch.org/whl/cu130 \\
  --requirement "$ROOT/locks/linux-cu130-requirements.txt"
"$PYTHON" -m pip install --no-deps "$ROOT/wheels/{wheel_name}"
"$PYTHON" -m swarm_inference.experiments.experiment_014 --help >/dev/null
'''
    install_qualified = common + '''
: "${RUNTIME_CERTIFICATE_URL:?set immutable RUNTIME_CERTIFICATE_URL}"
: "${PHYSICAL_CERTIFICATE_SHA256:?set canary-reported PHYSICAL_CERTIFICATE_SHA256}"
: "${QUALIFIED_RUNTIME_URL:?set immutable QUALIFIED_RUNTIME_URL}"
mkdir -p "$SWARM_HOME/certificates" "$SWARM_HOME/native"
CERT="$SWARM_HOME/certificates/linux-runtime-certificate.json"
RUNTIME="$SWARM_HOME/native/libcoli_cuda-sm86.so"
curl --fail --location --continue-at - --output "$CERT.partial" "$RUNTIME_CERTIFICATE_URL"
printf '%s  %s\n' "$PHYSICAL_CERTIFICATE_SHA256" "$CERT.partial" | sha256sum -c -
mv "$CERT.partial" "$CERT"
RUNTIME_SHA="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["binary"]["sha256"])' "$CERT")"
curl --fail --location --continue-at - --output "$RUNTIME.partial" "$QUALIFIED_RUNTIME_URL"
printf '%s  %s\n' "$RUNTIME_SHA" "$RUNTIME.partial" | sha256sum -c -
install -m 0555 "$RUNTIME.partial" "$RUNTIME.new"
mv "$RUNTIME.new" "$RUNTIME"
rm -f -- "$RUNTIME.partial"
"$PYTHON" -c 'from pathlib import Path; from swarm_inference.experiments.experiment_014.runtime_qualification import validate_linux_runtime_certificate; import sys; validate_linux_runtime_certificate(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))' \\
  "$CERT" "$ROOT/native/native-source-manifest.json" "$ROOT/manifests/k3-3090-placement-manifest.json"
'''
    prepare = common + '''
: "${WORKER_ID:?set WORKER_ID, for example k3-worker-089}"
: "${NETWORK_RTT_MS:?set measured NETWORK_RTT_MS}"
: "${NETWORK_BANDWIDTH_GBPS:?set measured NETWORK_BANDWIDTH_GBPS}"
REQ="$ROOT/requirements/pre-canary/$WORKER_ID.json"
STATE="$SWARM_HOME/workers/$WORKER_ID"
mkdir -p "$STATE/receipts" "$STATE/cache" "$STATE/packages"
ASSIGNMENT="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["assignment_sha256"])' "$REQ")"
REVISION="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_revision"])' "$REQ")"
"$PYTHON" -m swarm_inference.experiments.experiment_014 inspect-deployment-node \\
  --output "$STATE/receipts/preflight-observation.json" \\
  --network-rtt-ms "$NETWORK_RTT_MS" \\
  --network-bandwidth-gbps "$NETWORK_BANDWIDTH_GBPS" \\
  --assignment-sha256 "$ASSIGNMENT" --checkpoint-revision "$REVISION" \\
  --package-version 0.1.0rc11 --disk-path "$STATE"
"$PYTHON" -m swarm_inference.experiments.experiment_014 node-admission \\
  --observation "$STATE/receipts/preflight-observation.json" --requirements "$REQ" \\
  --mode PRE_CANARY --output "$STATE/receipts/preflight-admission.json"
"$PYTHON" -m swarm_inference.experiments.experiment_014 acquire-worker \\
  --distribution-manifest "$ROOT/manifests/k3-weight-distribution-manifest.json" \\
  --worker-id "$WORKER_ID" --cache-directory "$STATE/cache" \\
  --output "$STATE/packages/$WORKER_ID.safetensors"
"$PYTHON" -m swarm_inference.experiments.experiment_014 activate-worker-snapshot \\
  --placement "$ROOT/manifests/k3-3090-placement-manifest.json" \\
  --worker-id "$WORKER_ID" --package "$STATE/packages/$WORKER_ID.safetensors" \\
  --config "$ROOT/model-metadata/config.json" --output-directory "$STATE/snapshot"
test -f "$STATE/snapshot/activation.json"
'''
    qualify = common + '''
: "${WORKER_ID:?set WORKER_ID}"
: "${NETWORK_RTT_MS:?set measured NETWORK_RTT_MS}"
: "${NETWORK_BANDWIDTH_GBPS:?set measured NETWORK_BANDWIDTH_GBPS}"
STATE="$SWARM_HOME/workers/$WORKER_ID"
CERT="$SWARM_HOME/certificates/linux-runtime-certificate.json"
RUNTIME="$SWARM_HOME/native/libcoli_cuda-sm86.so"
test -f "$CERT"
test -f "$RUNTIME"
RUNTIME_SHA="$($PYTHON -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$RUNTIME")"
"$PYTHON" -m swarm_inference.experiments.experiment_014 worker-requirements \\
  --placement "$ROOT/manifests/k3-3090-placement-manifest.json" \\
  --distribution "$ROOT/manifests/k3-weight-distribution-manifest.json" \\
  --worker-id "$WORKER_ID" --package-version 0.1.0rc11 \\
  --runtime-sha256 "$RUNTIME_SHA" --output "$STATE/requirements.json"
"$PYTHON" -m swarm_inference.experiments.experiment_014 assigned-stage-canary \\
  --placement "$ROOT/manifests/k3-3090-placement-manifest.json" \\
  --native-source-manifest "$ROOT/native/native-source-manifest.json" \\
  --runtime-certificate "$CERT" --runtime "$RUNTIME" --snapshot "$STATE/snapshot" \\
  --worker-id "$WORKER_ID" --output "$STATE/receipts/assigned-stage-canary.json"
ASSIGNMENT="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["assignment_sha256"])' "$STATE/requirements.json")"
REVISION="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_revision"])' "$STATE/requirements.json")"
"$PYTHON" -m swarm_inference.experiments.experiment_014 inspect-deployment-node \\
  --runtime "$RUNTIME" --assigned-stage-canary "$STATE/receipts/assigned-stage-canary.json" \\
  --output "$STATE/receipts/fleet-observation.json" --network-rtt-ms "$NETWORK_RTT_MS" \\
  --network-bandwidth-gbps "$NETWORK_BANDWIDTH_GBPS" --assignment-sha256 "$ASSIGNMENT" \\
  --checkpoint-revision "$REVISION" --package-version 0.1.0rc11 --disk-path "$STATE"
"$PYTHON" -m swarm_inference.experiments.experiment_014 node-admission \\
  --observation "$STATE/receipts/fleet-observation.json" \\
  --requirements "$STATE/requirements.json" --mode FLEET \\
  --output "$STATE/receipts/fleet-admission.json"
"$PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "PASS"' \\
  "$STATE/receipts/fleet-admission.json"
if ! test -f "$STATE/identity.json"; then
  "$SWARM_HOME/venv/bin/swarm" identity create --path "$STATE/identity.json" \\
    --kind worker --json >/dev/null
fi
chmod 0600 "$STATE/identity.json"
"$SWARM_HOME/venv/bin/swarm" identity show --path "$STATE/identity.json" --json \\
  > "$STATE/receipts/worker-public-identity.json"
'''
    canary = common + '''
RUNTIME="$SWARM_HOME/native/libcoli_cuda-sm86.so"
CERT_DIR="$SWARM_HOME/certificates"
mkdir -p "$CERT_DIR" "$SWARM_HOME/canary"
for ID in k3-worker-000 k3-worker-089 k3-worker-091 k3-worker-092; do
  test -f "$SWARM_HOME/workers/$ID/snapshot/activation.json"
done
"$PYTHON" -m swarm_inference.experiments.experiment_014 physical-3090-canary \\
  --placement "$ROOT/manifests/k3-3090-placement-manifest.json" \\
  --native-source-manifest "$ROOT/native/native-source-manifest.json" \\
  --runtime "$RUNTIME" --fixture-npz "$ROOT/canary/fixtures.npz" \\
  --fixture-manifest "$ROOT/canary/fixtures.json" --evidence-root "$ROOT/evidence" \\
  --stage-zero-snapshot "$SWARM_HOME/workers/k3-worker-000/snapshot" \\
  --kda-snapshot "$SWARM_HOME/workers/k3-worker-089/snapshot" \\
  --mla-snapshot "$SWARM_HOME/workers/k3-worker-091/snapshot" \\
  --final-snapshot "$SWARM_HOME/workers/k3-worker-092/snapshot" \\
  --receipt "$SWARM_HOME/canary/physical-3090-canary.json" \\
  --certificate "$CERT_DIR/linux-runtime-certificate.json"
test -f "$CERT_DIR/linux-runtime-certificate.json"
'''
    coordinator = common + '''
: "${COORDINATOR_ADVERTISE:?set COORDINATOR_ADVERTISE host:port}"
exec "$SWARM_HOME/venv/bin/swarm" coordinator --config "$ROOT/config/coordinator.yaml" \\
  --listen 0.0.0.0:50051 --advertise "$COORDINATOR_ADVERTISE" \\
  --state "$SWARM_HOME/coordinator" --model-path "$ROOT/model-metadata" --dtype float32
'''
    worker = common + '''
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
"$PYTHON" -c 'from pathlib import Path; from swarm_inference.experiments.experiment_014.runtime_qualification import validate_linux_runtime_certificate; import sys; validate_linux_runtime_certificate(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))' \\
  "$CERT" "$ROOT/native/native-source-manifest.json" "$ROOT/manifests/k3-3090-placement-manifest.json"
ASSIGNMENT="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["assignment_sha256"])' "$STATE/requirements.json")"
REVISION="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_revision"])' "$STATE/requirements.json")"
"$PYTHON" -m swarm_inference.experiments.experiment_014 inspect-deployment-node \\
  --runtime "$RUNTIME" --assigned-stage-canary "$STATE/receipts/assigned-stage-canary.json" \\
  --output "$STATE/receipts/launch-observation.json" --network-rtt-ms "$NETWORK_RTT_MS" \\
  --network-bandwidth-gbps "$NETWORK_BANDWIDTH_GBPS" --assignment-sha256 "$ASSIGNMENT" \\
  --checkpoint-revision "$REVISION" --package-version 0.1.0rc11 --disk-path "$STATE"
"$PYTHON" -m swarm_inference.experiments.experiment_014 node-admission \\
  --observation "$STATE/receipts/launch-observation.json" \\
  --requirements "$STATE/requirements.json" --mode FLEET \\
  --output "$STATE/receipts/launch-admission.json"
"$PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "PASS"' \\
  "$STATE/receipts/launch-admission.json"
exec "$SWARM_HOME/venv/bin/swarm" worker --coordinator "$COORDINATOR_ENDPOINT" \\
  --backend torch-cuda --memory-limit-gb 24 --worker-id "$WORKER_ID" \\
  --listen 0.0.0.0:50052 --advertise "$WORKER_ADVERTISE" \\
  --identity "$STATE/identity.json" --trusted-coordinator-fingerprint "$COORDINATOR_FINGERPRINT" \\
  --model-shard-root "$STATE" --model-snapshot "$STATE/snapshot" \\
  --model-identity "$STATE/snapshot/model-identity.json" --no-allow-model-download \\
  --stage-runtime --device native-cuda:0 --dtype float32 \\
  --data-listen 0.0.0.0:50053 --data-advertise "$WORKER_DATA_ADVERTISE" \\
  --max-stage-sessions 8 --upload-bandwidth-mbps "$UPLOAD_MBPS" \\
  --download-bandwidth-mbps "$DOWNLOAD_MBPS"
'''
    trust = common + '''
: "${WORKER_PUBLIC_IDENTITIES:?directory containing exactly 93 public identity JSON files}"
COUNT="$(find "$WORKER_PUBLIC_IDENTITIES" -maxdepth 1 -type f -name 'k3-worker-*.json' | wc -l)"
test "$COUNT" -eq 93
for IDENTITY in "$WORKER_PUBLIC_IDENTITIES"/k3-worker-*.json; do
  FINGERPRINT="$($PYTHON -c 'import json,sys; value=json.load(open(sys.argv[1])); assert "private_key" not in value; print(value["fingerprint"])' "$IDENTITY")"
  "$SWARM_HOME/venv/bin/swarm" identity trust --coordinator-state "$SWARM_HOME/coordinator" \\
    --fingerprint "$FINGERPRINT" --label "$(basename "$IDENTITY" .json)" \\
    --notes experiment-015-hash-locked-fleet
done
'''
    cost = common + '''
: "${FLEET_STATUS:?set FLEET_STATUS to the measured fleet JSON}"
"$PYTHON" -m swarm_inference.experiments.experiment_014 fleet-cost-guard \\
  --fleet-status "$FLEET_STATUS" --policy "$ROOT/policies/cost-guard.json" \\
  --output "$SWARM_HOME/coordinator/fleet-cost-guard.json"
"$PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "ALLOW"' \\
  "$SWARM_HOME/coordinator/fleet-cost-guard.json"
'''
    bind = common + '''
: "${COORDINATOR_ENDPOINT:?set COORDINATOR_ENDPOINT}"
CERT="$SWARM_HOME/certificates/linux-runtime-certificate.json"
test -f "$CERT"
"$PYTHON" -c 'from pathlib import Path; from swarm_inference.experiments.experiment_014.runtime_qualification import validate_linux_runtime_certificate; import sys; validate_linux_runtime_certificate(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))' \\
  "$CERT" "$ROOT/native/native-source-manifest.json" "$ROOT/manifests/k3-3090-placement-manifest.json"
"$SWARM_HOME/venv/bin/swarm" workers --coordinator "$COORDINATOR_ENDPOINT" --json > "$SWARM_HOME/coordinator/workers.json"
bash "$ROOT/scripts/cost-guard.sh"
"$PYTHON" -m swarm_inference.experiments.experiment_014 bind-deployment-plan \\
  --placement "$ROOT/manifests/k3-3090-placement-manifest.json" \\
  --workers-status "$SWARM_HOME/coordinator/workers.json" \\
  --runtime-certificate "$CERT" --native-source-manifest "$ROOT/native/native-source-manifest.json" \\
  --output "$SWARM_HOME/coordinator/bound-plan.json"
"$SWARM_HOME/venv/bin/swarm" model deploy --coordinator "$COORDINATOR_ENDPOINT" \\
  --plan "$SWARM_HOME/coordinator/bound-plan.json"
'''
    collect = common + r'''
DEST="${1:-$SWARM_HOME/collected/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$DEST"
for SOURCE in "$SWARM_HOME/canary" "$SWARM_HOME/certificates" "$SWARM_HOME/coordinator"; do
  if test -d "$SOURCE"; then cp -a "$SOURCE" "$DEST/"; fi
done
find "$SWARM_HOME/workers" -path '*/receipts/*.json' -type f -exec cp --parents '{}' "$DEST/" \; 2>/dev/null || true
find "$DEST" -type f -exec sha256sum '{}' \; | sort > "$DEST/SHA256SUMS"
echo "$DEST"
'''
    cleanup = common + '''
: "${COORDINATOR_ENDPOINT:?set COORDINATOR_ENDPOINT}"
: "${TOPOLOGY_ID:?set exact TOPOLOGY_ID returned by deployment}"
"$SWARM_HOME/venv/bin/swarm" model unload --coordinator "$COORDINATOR_ENDPOINT" \\
  --topology-id "$TOPOLOGY_ID" --force
echo "Model unloaded. Provider instances must be terminated through the approved cost-guard workflow."
'''
    fine = common + '''
echo "NOT REQUIRED: Experiment 014 selected the 93-worker whole-layer topology." >&2
echo "The logical fine-worker result is characterization evidence only; do not gate this fleet on it." >&2
exit 78
'''
    return {
        "build-runtime.sh": build,
        "install-worker.sh": install,
        "install-qualified-runtime.sh": install_qualified,
        "prepare-worker.sh": prepare,
        "qualify-worker.sh": qualify,
        "run-3090-canary.sh": canary,
        "run-worker.sh": worker,
        "run-coordinator.sh": coordinator,
        "trust-workers.sh": trust,
        "bind-and-deploy.sh": bind,
        "cost-guard.sh": cost,
        "collect-artifacts.sh": collect,
        "cleanup.sh": cleanup,
        "run-two-gpu-fine-canary.sh": fine,
    }


def _build_release_archive(package_directory: Path) -> Path:
    package = package_directory.resolve()
    archive = package.with_suffix(".zip")
    if archive.exists():
        raise Experiment015PackageError(f"refusing to replace release archive: {archive}")
    temporary = archive.with_suffix(archive.suffix + ".partial")
    prefix = package.name
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as bundle:
        for path in sorted(item for item in package.rglob("*") if item.is_file()):
            if path.is_symlink():
                raise Experiment015PackageError(f"release archive source is linked: {path}")
            relative = f"{prefix}/{path.relative_to(package).as_posix()}"
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if path.parent.name == "scripts" else 0o100644) << 16
            with path.open("rb") as source, bundle.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    os.replace(temporary, archive)
    return archive


def _validate_release_archive(package_directory: Path, archive_path: Path) -> dict[str, Any]:
    package = package_directory.resolve()
    archive = archive_path.resolve()
    expected_paths = sorted(
        f"{package.name}/{path.relative_to(package).as_posix()}"
        for path in package.rglob("*")
        if path.is_file()
    )
    with zipfile.ZipFile(archive) as bundle:
        if sorted(bundle.namelist()) != expected_paths:
            raise Experiment015PackageError("release archive file set differs")
        for member in expected_paths:
            source = package / Path(member).relative_to(package.name)
            digest = hashlib.sha256()
            with bundle.open(member) as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != _sha256(source):
                raise Experiment015PackageError(f"release archive member differs: {member}")
    return {
        "path": str(archive),
        "sha256": _sha256(archive),
        "bytes": archive.stat().st_size,
        "member_count": len(expected_paths),
        "status": "PASS",
    }


def build_experiment_015_package(
    repository_root: Path,
    output_directory: Path,
    wheel_path: Path,
    checkpoint_metadata_directory: Path,
) -> dict[str, Any]:
    """Assemble a no-checkout release that remains locked until a physical canary."""

    root = repository_root.resolve()
    output = output_directory.resolve()
    if output.exists():
        raise Experiment015PackageError(f"refusing to replace package directory: {output}")
    artifact_root = root / "artifacts" / "experiment-014"
    deployment = artifact_root / "deployment"
    inputs = {
        "placement": deployment / "h014-037a-final-physical-placement.json",
        "execution": deployment / "h014-037a-final-execution-plan.json",
        "distribution": deployment / "h014-037a-final-distribution-manifest.json",
        "fixture_npz": deployment / "h014-037b4-physical-canary-fixtures.npz",
        "fixture_manifest": deployment / "h014-037b4-physical-canary-fixtures.json",
        "capacity": artifact_root / "performance" / CAPACITY_EVIDENCE_NAME,
        "coarse_network": artifact_root / "coarse" / COARSE_NETWORK_EVIDENCE_NAME,
        "dependency_lock": deployment
        / "h014-037b4d-linux-cu130-requirements.lock.txt",
        "uv_lock": root / "uv.lock",
        "wheel": wheel_path.resolve(),
    }
    for name, path in inputs.items():
        if not path.is_file() or path.is_symlink():
            raise Experiment015PackageError(f"missing immutable {name}: {path}")
    output.mkdir(parents=True)
    manifests = output / "manifests"
    placement_path = manifests / PLACEMENT_NAME
    _copy(inputs["placement"], placement_path)
    placement = _read(placement_path)
    if placement.get("status") != "PASS" or placement.get("node_count") != 93:
        raise Experiment015PackageError("source placement is not the passing 93-worker plan")
    if (
        placement.get("runtime", {}).get("status")
        != "PROMOTED_FOR_PRE_CANARY_USE"
        or any(
            worker.get("model_and_runtime", {}).get("promoted_for_pre_canary_use")
            is not True
            or worker.get("model_and_runtime", {}).get(
                "candidate_pending_final_same_binary_regression"
            )
            is not False
            for worker in placement.get("workers", [])
        )
    ):
        raise Experiment015PackageError("source placement is not bound to the promoted runtime")
    _portable_manifest(inputs["distribution"], manifests / DISTRIBUTION_NAME, placement_path)
    _portable_manifest(inputs["execution"], manifests / EXECUTION_NAME, placement_path)
    distribution = _read(manifests / DISTRIBUTION_NAME)
    if distribution.get("status") != "PASS" or len(distribution.get("workers", [])) != 93:
        raise Experiment015PackageError("portable distribution is not passing for 93 workers")

    native_source = output / "native" / "src"
    for name in NATIVE_SOURCE_FILES:
        _copy(root / "third_party" / "colibri" / "c" / name, native_source / name)
    del placement
    build_native_source_manifest(
        native_source, placement_path, output / "native" / "native-source-manifest.json"
    )
    placement = _read(placement_path)

    wheel_destination = output / "wheels" / inputs["wheel"].name
    _copy(inputs["wheel"], wheel_destination)
    dependency_lock = output / "locks" / "linux-cu130-requirements.txt"
    _copy(inputs["dependency_lock"], dependency_lock)
    _copy(inputs["uv_lock"], output / "locks" / "uv.lock")
    metadata_source = checkpoint_metadata_directory.resolve()
    metadata_hashes: dict[str, str] = {}
    for name in ("config.json", *KIMI_TOKENIZER_ASSETS):
        source = metadata_source / name
        destination = output / "model-metadata" / name
        _copy(source, destination)
        metadata_hashes[name] = _sha256(destination)
    if metadata_hashes["config.json"] != placement["checkpoint"]["config_sha256"]:
        raise Experiment015PackageError("packaged Kimi config differs from placement")
    _atomic_json(
        output / "model-metadata" / "metadata-lock.json",
        {
            "model_id": "moonshotai/Kimi-K3",
            "revision": placement["checkpoint"]["revision"],
            "checkpoint_fingerprint": placement["checkpoint"]["checkpoint_fingerprint"],
            "files": metadata_hashes,
        },
    )

    _copy(inputs["fixture_npz"], output / "canary" / "fixtures.npz")
    fixture = _read(inputs["fixture_manifest"])
    fixture["fixture_npz"] = "fixtures.npz"
    fixture["fixture_npz_sha256"] = _sha256(output / "canary" / "fixtures.npz")
    _atomic_json(output / "canary" / "fixtures.json", fixture)

    capacity = _read(inputs["capacity"])
    coarse_network = _read(inputs["coarse_network"])
    coarse_recommendation = coarse_network.get("recommendation", {})
    if (
        capacity.get("status") != "PASS"
        or coarse_network.get("status") != "PASS"
        or coarse_recommendation.get("coupled_operating_point") is not True
        or float(coarse_recommendation.get("capacity_retention_percent", 0.0)) < 90.0
        or capacity.get("sources", {}).get("coarse_network", {}).get("sha256")
        != _sha256(inputs["coarse_network"])
    ):
        raise Experiment015PackageError("coarse network evidence is not coupled/passing")
    coarse_policy = {
        "class": "kimi_coarse_stage_fp32_v1",
        "maximum_rtt_ms": float(coarse_recommendation["maximum_tested_rtt_ms"]),
        "minimum_bandwidth_gbps": float(
            coarse_recommendation["minimum_tested_bandwidth_gbps"]
        ),
        "modeled_capacity_retention_percent": float(
            coarse_recommendation["capacity_retention_percent"]
        ),
        "payload_bytes": int(coarse_network["measured_inputs"]["activation_payload_bytes"]),
        "wire_bytes": int(coarse_network["measured_inputs"]["production_direction_wire_bytes"]),
        "source_receipt_sha256": _sha256(inputs["coarse_network"]),
        "required_for_selected_topology": True,
    }
    placement_network = placement["network_admission"]
    if (
        float(placement_network["maximum_rtt_ms"])
        != coarse_policy["maximum_rtt_ms"]
        or float(placement_network["minimum_bandwidth_gbps"])
        != coarse_policy["minimum_bandwidth_gbps"]
        or int(placement_network["activation_payload_bytes"])
        != coarse_policy["payload_bytes"]
    ):
        raise Experiment015PackageError("placement network admission differs from evidence")

    evidence_sources = (
        artifact_root / "cuda" / "h014-038-regression-full-93-layer.json",
        artifact_root / "k3-cuda-operation-matrix.json",
        artifact_root / "rtx3090-sm86-certification.json",
        artifact_root / "cuda" / "h014-038-promotion.json",
        artifact_root / "cuda" / "h014-038-source-manifest.json",
        artifact_root / "persistent" / "h014-038-regression-final-stage.json",
        artifact_root / "persistent" / "h014-038-regression-nonfinal-stages.json",
        artifact_root / "persistent" / "h014-038-regression-stage-zero.json",
        artifact_root / "persistent" / "h014-038aj2-stage-zero-steady.json",
        deployment / "h014-037b1-deployment-identity-fixture.json",
        deployment / "h014-037b2a-tokenizer-product-seam.json",
        deployment / "h014-037b3-cross-platform-runtime-identity.json",
        deployment / "h014-037b4b-portable-acquisition-fixture.json",
        deployment / "h014-037b-admission-cost-guard.json",
        deployment / "h014-037b-coarse-stage-recovery.json",
        deployment / "h014-037b-final-logical-rehearsal.json",
        inputs["capacity"],
        artifact_root / "sub-layer" / "h014-sub-011-promoted-four-worker-real-expert.json",
        artifact_root / "sub-layer" / "h014-sub-012a-006f-promoted-single-interval-batch.json",
        artifact_root / "sub-layer" / "h014-sub-005-scaling-network.json",
        artifact_root / "sub-layer" / "h014-sub-009b-recovery.json",
        artifact_root / "coarse" / "h014-038ae-promoted-stage0-stage1-tcp.json",
        inputs["coarse_network"],
    )
    for source in evidence_sources:
        _copy(source, output / "evidence" / source.name)

    placement["_path"] = str(placement_path)
    distribution["_path"] = str(manifests / DISTRIBUTION_NAME)
    worker_ids = [f"k3-worker-{index:03d}" for index in range(93)]
    workers = sorted(placement["workers"], key=lambda row: int(row["worker_index"]))
    if [row["worker_id"] for row in workers] != worker_ids:
        raise Experiment015PackageError("placement worker identities are not canonical")
    for worker in workers:
        _atomic_json(
            output / "requirements" / "pre-canary" / f"{worker['worker_id']}.json",
            _worker_requirements(placement, distribution, worker),
        )
    del placement["_path"], distribution["_path"]

    _atomic_json(
        output / "policies" / "cost-guard.json",
        {
            "expected_nodes": 93,
            "maximum_gpu_price_per_hour_usd": 0.05,
            "maximum_fleet_cost_per_hour_usd": 4.65,
            "sub_layer_canary_required": False,
            "physical_single_3090_canary_required": True,
        },
    )
    _atomic_json(
        output / "policies" / "network-classes.json",
        {
            "coarse_stage": coarse_policy,
            "fine_microwork": {
                "class": "kimi_microwork_low_latency_v1",
                "maximum_tested_viable_rtt_ms": 0.5,
                "minimum_tested_viable_bandwidth_gbps": 2.5,
                "exact_threshold_rtt_ms_at_100_gbps": 0.646,
                "exact_threshold_bandwidth_gbps_at_0_25_ms": 1.986,
                "required_for_selected_topology": False,
            },
        },
    )
    _write(
        output / "config" / "coordinator.yaml",
        '''kind: product-stage-ring
schema_version: "1"
default_dtype: float32
local_only_by_default: true
worker_heartbeat_timeout_s: 15
deployment_lease_seconds: 2592000
control_timeout_s: 120
request_timeout_s: 300
event_queue_capacity: 256
token_ingress_capacity: 256
planning_max_sequence_tokens: 8192
maximum_candidate_workers: 93
maximum_stage_count: 93
planning_beam_width: 512
network_measurement_ttl_seconds: 900
network_probe_max_bytes: 16777216
network_probe_timeout_seconds: 10
allow_unmeasured_links_for_explicit_plans: false
balanced_throughput_weight: 0.45
balanced_memory_headroom_weight: 0.25
balanced_reliability_weight: 0.20
balanced_participation_weight: 0.10
maximum_active_sessions_per_worker: 8
coordinator_id: k3-experiment-015
route_future_tolerance_s: 30
route_nonce_cache_capacity: 4096
cleanup_timeout_s: 10
recovery_timeout_s: 120
maximum_recovery_attempts: 2
trusted_worker_fingerprints: []
require_trusted_workers: true
trust_store_path: /opt/swarm/coordinator/trusted-workers.json''',
    )
    for name, body in _scripts(wheel_destination.name).items():
        _write(output / "scripts" / name, body)
    _write(output / "scripts" / "verify-package.py", _standalone_verifier())
    for path in (output / "scripts").iterdir():
        path.chmod(0o755)

    release = {
        "schema_version": PACKAGE_SCHEMA,
        "status": "READY_FOR_SINGLE_3090_CANARY",
        "experiment_014_local_certification": "PASS",
        "experiment_014_promotion_receipt_sha256": placement["source_artifacts"][
            "promotion_receipt"
        ]["sha256"],
        "windows_precanary_reference_sha256": placement["runtime"][
            "cuda_library_sha256"
        ],
        "physical_3090_canary": "NOT_RUN",
        "fleet_activation_allowed": False,
        "selected_topology": "WHOLE-LAYER",
        "worker_count": 93,
        "sub_layer_canary_required": False,
        "package_version": PACKAGE_VERSION,
        "wheel": {"file": f"wheels/{wheel_destination.name}", "sha256": _sha256(wheel_destination)},
        "dependency_lock": {
            "file": "locks/linux-cu130-requirements.txt",
            "sha256": _sha256(dependency_lock),
            "uv_lock_sha256": _sha256(output / "locks" / "uv.lock"),
            "python": "3.11",
            "torch": "2.13.0+cu130",
            "cuda_toolkit": "13.0.3.0",
            "triton": "3.7.1",
        },
        "placement_sha256": _sha256(placement_path),
        "distribution_sha256": _sha256(manifests / DISTRIBUTION_NAME),
        "native_source_manifest_sha256": _sha256(output / "native" / "native-source-manifest.json"),
        "physical_certificate_path": "/opt/swarm/certificates/linux-runtime-certificate.json",
        "full_fleet_unlock": (
            "physical certificate plus exact canary ELF, per-node assigned-stage FLEET "
            "admission, and exact 93-worker cost guard"
        ),
    }
    _atomic_json(output / "RELEASE.json", release)
    _write(
        output / "README.md",
        '''# Experiment 015: physical Kimi K3 cluster handoff

This is a no-checkout, hash-locked release for the selected 93-worker whole-layer topology.
Experiment 014 local pre-canary certification is complete. This package is ready to run the
preregistered single RTX 3090 canary; it is not proof that the physical canary passed.

1. Verify `package-lock.json` with the installed package validator.
2. Run `scripts/install-worker.sh` and `scripts/build-runtime.sh` on the canary RTX 3090.
3. Prepare workers 000, 089, 091 and 092 with `scripts/prepare-worker.sh`.
4. Run `scripts/run-3090-canary.sh`. Only this may emit the Linux physical certificate.
5. Publish the canary-built ELF and certificate immutably; record the certificate SHA out of band.
6. Rent the remaining nodes only if the canary passes and price is at most USD 0.05/GPU-hour.
7. On every fleet node run `install-qualified-runtime.sh`, `prepare-worker.sh`, then
   `qualify-worker.sh`; never rebuild the fleet ELF independently.
8. Exchange public identity fingerprints, start all admitted workers, then run
   `bind-and-deploy.sh`.

The selected initial fleet does not use sub-layer microworkers. Their measured logical result is
functional but uneconomic; the optional two-GPU script is deliberately non-activating.''',
    )
    lock_files: dict[str, dict[str, Any]] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "package-lock.json":
            relative = path.relative_to(output).as_posix()
            lock_files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    _atomic_json(
        output / "package-lock.json",
        {"schema_version": PACKAGE_SCHEMA, "status": "PASS", "files": lock_files},
    )
    validation = validate_experiment_015_package(output)
    archive = _build_release_archive(output)
    validation["release_archive"] = _validate_release_archive(output, archive)
    validation["out_of_band_archive_sha256_required"] = True
    return validation


def _fail(message: str) -> None:
    raise Experiment015PackageError(message)


def validate_experiment_015_package(package_directory: Path) -> dict[str, Any]:
    """Verify package closure and all local fail-closed deployment seams."""

    root = package_directory.resolve()
    lock = _read(root / "package-lock.json")
    entries = lock.get("files")
    if lock.get("status") != "PASS" or not isinstance(entries, dict):
        _fail("package lock is not passing")
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "package-lock.json"
    )
    if actual != sorted(entries):
        _fail("package file set differs from package lock")
    for relative, identity in entries.items():
        path = root / relative
        if path.is_symlink() or path.stat().st_size != int(identity["bytes"]):
            _fail(f"package size/link identity differs: {relative}")
        if _sha256(path) != identity["sha256"]:
            _fail(f"package SHA-256 differs: {relative}")

    release = _read(root / "RELEASE.json")
    if (
        release.get("status") != "READY_FOR_SINGLE_3090_CANARY"
        or release.get("experiment_014_local_certification") != "PASS"
        or release.get("physical_3090_canary") != "NOT_RUN"
        or release.get("fleet_activation_allowed") is not False
        or release.get("worker_count") != 93
        or release.get("selected_topology") != "WHOLE-LAYER"
    ):
        _fail("release state is not the fail-closed pre-canary state")
    if (root / "DEPLOYMENT-BLOCKED.json").exists():
        _fail("stale blocked-package marker remains")
    certificates = root / "certificates"
    if certificates.exists() and any(certificates.iterdir()):
        _fail("a physical certificate may not be manufactured in the static package")

    placement_path = root / "manifests" / PLACEMENT_NAME
    placement = _read(placement_path)
    if (
        placement.get("status") != "PASS"
        or placement.get("node_count") != 93
        or placement.get("topology", {}).get("class") != "WHOLE-LAYER"
        or placement.get("coverage", {}).get("required_tensor_count") != 497052
        or placement.get("coverage", {}).get("unassigned_required_tensor_count") != 0
        or placement.get("runtime", {}).get("status")
        != "PROMOTED_FOR_PRE_CANARY_USE"
    ):
        _fail("placement is not exact 93-worker whole-layer ownership")
    expected_ids = [f"k3-worker-{index:03d}" for index in range(93)]
    if [row["worker_id"] for row in placement["workers"]] != expected_ids:
        _fail("placement worker IDs differ")
    if any(
        worker.get("model_and_runtime", {}).get("promoted_for_pre_canary_use")
        is not True
        or worker.get("model_and_runtime", {}).get(
            "candidate_pending_final_same_binary_regression"
        )
        is not False
        for worker in placement["workers"]
    ):
        _fail("placement workers are not bound to the promoted runtime")
    if (
        release.get("experiment_014_promotion_receipt_sha256")
        != placement.get("source_artifacts", {})
        .get("promotion_receipt", {})
        .get("sha256")
        or release.get("windows_precanary_reference_sha256")
        != placement.get("runtime", {}).get("cuda_library_sha256")
    ):
        _fail("release identity differs from the promoted placement")

    distribution_path = root / "manifests" / DISTRIBUTION_NAME
    distribution = _read(distribution_path)
    try:
        resolved = _resolve_distribution_placement(distribution_path, distribution)
    except AcquisitionError as exc:
        raise Experiment015PackageError(str(exc)) from exc
    if resolved != placement_path or [row["worker_id"] for row in distribution["workers"]] != expected_ids:
        _fail("portable distribution ownership differs")
    execution = _read(root / "manifests" / EXECUTION_NAME)
    if (
        execution.get("status") != "PASS"
        or execution.get("node_count") != 93
        or execution.get("placement_manifest") != PLACEMENT_NAME
        or execution.get("placement_manifest_sha256") != _sha256(placement_path)
    ):
        _fail("logical execution plan is not portable/exact")

    source_directory = root / "native" / "src"
    if sorted(path.name for path in source_directory.iterdir()) != sorted(NATIVE_SOURCE_FILES):
        _fail("native source allowlist is incomplete or expanded")
    try:
        validate_native_source_manifest(
            root / "native" / "native-source-manifest.json", placement_path
        )
    except RuntimeQualificationError as exc:
        raise Experiment015PackageError(str(exc)) from exc
    manifest = _read(root / "native" / "native-source-manifest.json")
    if manifest.get("build_arguments") != list(BUILD_ARGUMENTS):
        _fail("native build contract differs")

    requirements = sorted((root / "requirements" / "pre-canary").glob("*.json"))
    if len(requirements) != 93 or [path.stem for path in requirements] != expected_ids:
        _fail("pre-canary requirements are not exact for 93 workers")
    for path, worker in zip(requirements, placement["workers"], strict=True):
        requirement = _read(path)
        if (
            requirement.get("status") != "PASS"
            or requirement.get("runtime_sha256") != ""
            or requirement.get("runtime_identity") != "PENDING_PHYSICAL_3090_CANARY"
            or requirement.get("assignment_sha256") != worker["assignment_sha256"]
        ):
            _fail(f"pre-canary worker requirement differs: {path.name}")

    cost = _read(root / "policies" / "cost-guard.json")
    if cost != {
        "expected_nodes": 93,
        "maximum_fleet_cost_per_hour_usd": 4.65,
        "maximum_gpu_price_per_hour_usd": 0.05,
        "physical_single_3090_canary_required": True,
        "sub_layer_canary_required": False,
    }:
        _fail("cost guard is missing or relaxed")
    network = _read(root / "policies" / "network-classes.json")
    coarse_evidence_path = root / "evidence" / COARSE_NETWORK_EVIDENCE_NAME
    coarse_evidence = _read(coarse_evidence_path)
    coarse_recommendation = coarse_evidence.get("recommendation", {})
    expected_coarse_policy = {
        "class": "kimi_coarse_stage_fp32_v1",
        "maximum_rtt_ms": float(coarse_recommendation["maximum_tested_rtt_ms"]),
        "minimum_bandwidth_gbps": float(
            coarse_recommendation["minimum_tested_bandwidth_gbps"]
        ),
        "modeled_capacity_retention_percent": float(
            coarse_recommendation["capacity_retention_percent"]
        ),
        "payload_bytes": int(coarse_evidence["measured_inputs"]["activation_payload_bytes"]),
        "wire_bytes": int(coarse_evidence["measured_inputs"]["production_direction_wire_bytes"]),
        "source_receipt_sha256": _sha256(coarse_evidence_path),
        "required_for_selected_topology": True,
    }
    if (
        coarse_evidence.get("status") != "PASS"
        or coarse_recommendation.get("coupled_operating_point") is not True
        or float(coarse_recommendation.get("capacity_retention_percent", 0.0)) < 90.0
        or network.get("coarse_stage") != expected_coarse_policy
        or float(placement["network_admission"]["maximum_rtt_ms"])
        != expected_coarse_policy["maximum_rtt_ms"]
        or float(placement["network_admission"]["minimum_bandwidth_gbps"])
        != expected_coarse_policy["minimum_bandwidth_gbps"]
        or network.get("fine_microwork", {}).get("required_for_selected_topology") is not False
    ):
        _fail("coarse/fine network classes differ from measured selection")

    fixture = _read(root / "canary" / "fixtures.json")
    if (
        fixture.get("physical_execution_status") != "NOT_RUN"
        or fixture.get("placement_sha256") != _sha256(placement_path)
        or fixture.get("fixture_npz_sha256") != _sha256(root / "canary" / "fixtures.npz")
        or sorted(fixture.get("roles", {}).values()) != [0, 89, 91, 92]
    ):
        _fail("physical canary fixture identity differs")
    for reference in fixture["reference_evidence"].values():
        path = root / "evidence" / reference["file"]
        if not path.is_file() or _sha256(path) != reference["sha256"]:
            _fail("canary reference evidence differs")

    wheel = root / release["wheel"]["file"]
    if _sha256(wheel) != release["wheel"]["sha256"]:
        _fail("release wheel identity differs")
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
    for suffix in (
        "deployment_canary.py",
        "runtime_qualification.py",
        "remote_acquisition.py",
        "kimi_tokenizer.py",
    ):
        if not any(name.endswith(suffix) for name in members):
            _fail(f"release wheel lacks {suffix}")
    if any(".uv-cache" in name or ".git/" in name for name in members):
        _fail("release wheel contains development state")
    dependency_identity = release.get("dependency_lock", {})
    dependency_lock = root / str(dependency_identity.get("file", ""))
    if (
        not dependency_lock.is_file()
        or _sha256(dependency_lock) != dependency_identity.get("sha256")
        or _sha256(root / "locks" / "uv.lock")
        != dependency_identity.get("uv_lock_sha256")
        or dependency_identity.get("python") != "3.11"
        or dependency_identity.get("torch") != "2.13.0+cu130"
        or dependency_identity.get("cuda_toolkit") != "13.0.3.0"
        or dependency_identity.get("triton") != "3.7.1"
    ):
        _fail("CUDA dependency lock identity differs")
    dependency_text = dependency_lock.read_text(encoding="utf-8")
    for requirement in (
        "torch==2.13.0+cu130",
        "cuda-toolkit==13.0.3.0",
        "triton==3.7.1",
    ):
        if requirement not in dependency_text:
            _fail("CUDA dependency lock content differs")
    if "--hash=sha256:" not in dependency_text:
        _fail("CUDA dependency lock has no package hashes")

    scripts = root / "scripts"
    if sorted(path.name for path in scripts.glob("*.sh")) != sorted(REQUIRED_SCRIPTS):
        _fail("deployment script set differs")
    combined = "\n".join((scripts / name).read_text(encoding="utf-8") for name in REQUIRED_SCRIPTS)
    if "git clone" in combined or "git checkout" in combined:
        _fail("deployment depends on a repository checkout")
    for name in REQUIRED_SCRIPTS:
        body = (scripts / name).read_text(encoding="utf-8")
        if not body.startswith("#!/usr/bin/env bash\nset -euo pipefail"):
            _fail(f"shell fail-closed preamble differs: {name}")
    verifier = scripts / "verify-package.py"
    if not verifier.is_file() or verifier.read_text(encoding="utf-8") != _standalone_verifier().rstrip() + "\n":
        _fail("standalone pre-install package verifier differs")
    install_worker = (scripts / "install-worker.sh").read_text(encoding="utf-8")
    if (
        "EXPECTED_PACKAGE_LOCK_SHA256" not in install_worker
        or "verify-package.py" not in install_worker
        or "RTX 3090" not in install_worker
        or "compute_cap" not in install_worker
        or "sys.version_info[:2] == (3, 11)" not in install_worker
        or "--require-hashes" not in install_worker
        or "linux-cu130-requirements.txt" not in install_worker
        or "--no-deps" not in install_worker
        or "pip install --upgrade" in install_worker
        or install_worker.index("nvidia-smi") > install_worker.index("pip install")
    ):
        _fail("pre-install package/hardware verification has a bypass")
    bind = (scripts / "bind-and-deploy.sh").read_text(encoding="utf-8")
    if (
        "validate_linux_runtime_certificate" not in bind
        or "--runtime-certificate" not in bind
        or "cost-guard.sh" not in bind
        or "model deploy" not in bind
    ):
        _fail("fleet activation has an unqualified bypass")
    canary = (scripts / "run-3090-canary.sh").read_text(encoding="utf-8")
    if not all(f"k3-worker-{index:03d}" in canary for index in (0, 89, 91, 92)):
        _fail("physical canary does not bind all four representative roles")
    if "physical-3090-canary" not in canary:
        _fail("physical canary does not invoke the production runner")
    qualified_runtime = (scripts / "install-qualified-runtime.sh").read_text(
        encoding="utf-8"
    )
    if (
        "PHYSICAL_CERTIFICATE_SHA256" not in qualified_runtime
        or qualified_runtime.count("sha256sum -c") < 2
        or "validate_linux_runtime_certificate" not in qualified_runtime
        or "QUALIFIED_RUNTIME_URL" not in qualified_runtime
        or 'mv "$RUNTIME.new" "$RUNTIME"' not in qualified_runtime
    ):
        _fail("qualified runtime acquisition has a hash/certificate bypass")
    qualify = (scripts / "qualify-worker.sh").read_text(encoding="utf-8")
    if (
        "assigned-stage-canary" not in qualify
        or "--assigned-stage-canary" not in qualify
        or "--mode FLEET" not in qualify
        or "worker-requirements" not in qualify
    ):
        _fail("per-node assigned-stage FLEET admission has a bypass")
    worker_launch = (scripts / "run-worker.sh").read_text(encoding="utf-8")
    if (
        "launch-admission.json" not in worker_launch
        or "validate_linux_runtime_certificate" not in worker_launch
        or "--mode FLEET" not in worker_launch
        or "status\"] == \"PASS\"" not in worker_launch
    ):
        _fail("worker registration does not require passing FLEET admission")
    trust = (scripts / "trust-workers.sh").read_text(encoding="utf-8")
    if (
        "--fingerprint" not in trust
        or "--identity" in trust
        or '"private_key" not in value' not in trust
    ):
        _fail("worker trust transports private identity material")

    gates = {
        "package_lock_exact": True,
        "no_repository_checkout": True,
        "release_pre_canary_fail_closed": True,
        "exact_93_worker_placement": True,
        "portable_worker_scoped_distribution": True,
        "logical_execution_plan_portable": True,
        "native_source_allowlist_exact": True,
        "native_build_contract_exact": True,
        "wheel_contains_production_canary": True,
        "all_93_pre_canary_requirements_exact": True,
        "cost_guard_exact": True,
        "coarse_and_fine_network_classes_separate": True,
        "physical_canary_fixture_exact": True,
        "physical_certificate_not_manufactured": True,
        "certificate_only_fleet_binding": True,
        "four_role_canary_required": True,
        "qualified_runtime_distributed_by_exact_hash": True,
        "per_node_assigned_stage_canary_required": True,
        "fleet_admission_required_before_registration": True,
        "standalone_preinstall_package_verification": True,
        "hardware_preflight_before_install": True,
        "qualified_runtime_activation_atomic": True,
        "public_fingerprint_only_trust": True,
        "certificate_revalidated_at_worker_launch": True,
        "python_and_cuda_dependencies_hash_locked": True,
    }
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "PASS",
        "package_directory": str(root),
        "package_lock_sha256": _sha256(root / "package-lock.json"),
        "file_count": len(entries) + 1,
        "package_bytes": sum(int(row["bytes"]) for row in entries.values())
        + (root / "package-lock.json").stat().st_size,
        "wheel_sha256": _sha256(wheel),
        "placement_sha256": _sha256(placement_path),
        "distribution_sha256": _sha256(distribution_path),
        "native_source_manifest_sha256": _sha256(
            root / "native" / "native-source-manifest.json"
        ),
        "physical_canary_status": "NOT_RUN",
        "fleet_activation_allowed": False,
        "acceptance_gates": gates,
    }


def benchmark_package_negative_controls(
    package_directory: Path, output_path: Path, *, cycle_id: str = "H014-037b4"
) -> dict[str, Any]:
    """Exercise bounded semantic and byte-integrity rejection controls."""

    root = package_directory.resolve()
    positive = validate_experiment_015_package(root)
    archive_path = root.with_suffix(".zip")
    archive = (
        _validate_release_archive(root, archive_path)
        if archive_path.is_file()
        else {"status": "ABSENT"}
    )
    controls: dict[str, dict[str, Any]] = {}

    def record(name: str, action: Any, expected: str) -> None:
        try:
            action()
        except (Experiment015PackageError, AcquisitionError, RuntimeQualificationError) as exc:
            message = str(exc)
            controls[name] = {
                "rejected": expected in message,
                "validator_rejected": True,
                "error": message,
                "expected": expected,
            }
        else:
            controls[name] = {
                "rejected": False,
                "validator_rejected": False,
                "error": None,
                "expected": expected,
            }

    def detach_write(path: Path, payload: bytes) -> None:
        path.unlink()
        path.write_bytes(payload)

    def update_lock(clone: Path, relative: str) -> None:
        lock_path = clone / "package-lock.json"
        lock = _read(lock_path)
        target = clone / relative
        lock["files"][relative] = {
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
        }
        _atomic_json(lock_path, lock)

    def mutate_and_validate(
        controls_root: Path,
        name: str,
        mutation: Any,
        expected: str,
    ) -> None:
        clone = controls_root / name
        shutil.copytree(root, clone, copy_function=os.link)
        mutation(clone)
        record(name, lambda: validate_experiment_015_package(clone), expected)
        shutil.rmtree(clone)

    with TemporaryDirectory(prefix="h014-037b4-controls-") as temporary_name:
        temp = Path(temporary_name)
        mutate_and_validate(
            temp,
            "changed_byte",
            lambda clone: detach_write(
                clone / "README.md", (clone / "README.md").read_bytes() + b"tamper\n"
            ),
            "package size/link identity differs",
        )
        mutate_and_validate(
            temp,
            "removed_file",
            lambda clone: (clone / "README.md").unlink(),
            "package file set differs",
        )

        def broaden_source(clone: Path) -> None:
            relative = "native/src/unapproved.cu"
            (clone / relative).write_text("// unapproved\n", encoding="utf-8")
            update_lock(clone, relative)

        mutate_and_validate(
            temp,
            "broadened_native_source",
            broaden_source,
            "native source allowlist",
        )

        def relax_cost(clone: Path) -> None:
            relative = "policies/cost-guard.json"
            policy = _read(clone / relative)
            policy["maximum_gpu_price_per_hour_usd"] = 0.30
            policy["maximum_fleet_cost_per_hour_usd"] = 27.90
            (clone / relative).unlink()
            _atomic_json(clone / relative, policy)
            update_lock(clone, relative)

        mutate_and_validate(temp, "relaxed_cost_guard", relax_cost, "cost guard")

        def inject_checkout(clone: Path) -> None:
            relative = "scripts/install-worker.sh"
            body = (clone / relative).read_bytes() + b"git checkout main\n"
            detach_write(clone / relative, body)
            update_lock(clone, relative)

        mutate_and_validate(
            temp,
            "repository_checkout_dependency",
            inject_checkout,
            "repository checkout",
        )

        def inject_logical_certificate(clone: Path) -> None:
            relative = "certificates/linux-runtime-certificate.json"
            _atomic_json(
                clone / relative,
                {"status": "PASS", "evidence_kind": "logical_fixture"},
            )
            update_lock(clone, relative)

        mutate_and_validate(
            temp,
            "logical_certificate",
            inject_logical_certificate,
            "physical certificate",
        )

        def remove_activation_gates(clone: Path) -> None:
            relative = "scripts/bind-and-deploy.sh"
            body = (clone / relative).read_text(encoding="utf-8")
            body = body.replace("validate_linux_runtime_certificate", "certificate_bypass")
            body = body.replace("--runtime-certificate", "--bypass-certificate")
            body = body.replace("cost-guard.sh", "bypass-cost.sh")
            detach_write(clone / relative, body.encode())
            update_lock(clone, relative)

        mutate_and_validate(
            temp,
            "unqualified_fleet_launch",
            remove_activation_gates,
            "unqualified bypass",
        )

        def bypass_runtime_hash(clone: Path) -> None:
            relative = "scripts/install-qualified-runtime.sh"
            body = (clone / relative).read_text(encoding="utf-8")
            body = body.replace("sha256sum -c", "sha256sum --version", 1)
            detach_write(clone / relative, body.encode())
            update_lock(clone, relative)

        mutate_and_validate(
            temp,
            "wrong_qualified_runtime_sha",
            bypass_runtime_hash,
            "hash/certificate bypass",
        )

        def bypass_assigned_canary(clone: Path) -> None:
            relative = "scripts/qualify-worker.sh"
            body = (clone / relative).read_text(encoding="utf-8")
            body = body.replace("assigned-stage-canary", "bypassed-stage-check")
            detach_write(clone / relative, body.encode())
            update_lock(clone, relative)

        mutate_and_validate(
            temp,
            "missing_assigned_stage_receipt",
            bypass_assigned_canary,
            "assigned-stage FLEET admission",
        )

        def bypass_preinstall_verifier(clone: Path) -> None:
            relative = "scripts/install-worker.sh"
            body = (clone / relative).read_text(encoding="utf-8")
            body = body.replace("verify-package.py", "bypass-package.py")
            detach_write(clone / relative, body.encode())
            update_lock(clone, relative)

        mutate_and_validate(
            temp,
            "missing_preinstall_package_verification",
            bypass_preinstall_verifier,
            "pre-install package/hardware verification",
        )

        def inject_private_identity_transport(clone: Path) -> None:
            relative = "scripts/trust-workers.sh"
            body = (clone / relative).read_text(encoding="utf-8")
            body = body.replace('--fingerprint "$FINGERPRINT"', '--identity "$IDENTITY"')
            detach_write(clone / relative, body.encode())
            update_lock(clone, relative)

        mutate_and_validate(
            temp,
            "private_identity_transport",
            inject_private_identity_transport,
            "private identity material",
        )

        def relax_dependency_lock(clone: Path) -> None:
            relative = "scripts/install-worker.sh"
            body = (clone / relative).read_text(encoding="utf-8")
            body = body.replace("--require-hashes", "--disable-hash-check")
            detach_write(clone / relative, body.encode())
            update_lock(clone, relative)

        mutate_and_validate(
            temp,
            "unhashed_dependency_install",
            relax_dependency_lock,
            "pre-install package/hardware verification",
        )
        portable = _read(root / "manifests" / DISTRIBUTION_NAME)
        distribution_path = root / "manifests" / DISTRIBUTION_NAME
        for name, reference, expected in (
            ("absolute_placement", str(root / "manifests" / PLACEMENT_NAME), "one sibling"),
            ("missing_placement", "missing.json", "absent or linked"),
            ("traversal_placement", "../placement.json", "one sibling"),
        ):
            candidate = dict(portable)
            candidate["placement_manifest"] = reference
            record(
                name,
                lambda candidate=candidate: _resolve_distribution_placement(
                    distribution_path, candidate
                ),
                expected,
            )
        candidate = dict(portable)
        candidate["placement_manifest_sha256"] = "0" * 64
        record(
            "placement_hash_mismatch",
            lambda: _resolve_distribution_placement(distribution_path, candidate),
            "SHA-256 differs",
        )

    gates = {name: bool(row["rejected"]) for name, row in controls.items()}
    receipt = {
        "schema_version": VALIDATION_SCHEMA,
        "cycle_id": cycle_id,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "hypothesis": (
            "The no-checkout Experiment 015 package is closed under its SHA-256 lock and "
            "rejects path, source, cost, certificate, and activation bypass tampering."
        ),
        "positive_validation": positive,
        "release_archive": archive,
        "out_of_band_archive_sha256_required": True,
        "negative_controls": controls,
        "acceptance_gates": gates,
        "physical_3090_canary": "NOT_RUN",
        "fleet_activation_allowed": False,
    }
    _atomic_json(output_path.resolve(), receipt)
    return receipt


__all__ = [
    "Experiment015PackageError",
    "benchmark_package_negative_controls",
    "build_experiment_015_package",
    "validate_experiment_015_package",
]
