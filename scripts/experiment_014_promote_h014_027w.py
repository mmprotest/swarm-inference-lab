"""Fail-closed promotion of the exact H014-027w CUDA runtime DLL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_014.compatibility import (
    certify_sm86_complete_kimi,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "experiment-014"
DEPLOYMENT = ROOT / "artifacts" / "experiment-015-deployment"
CANDIDATE_HASH = "c666e37c36667bb102088aa7a5bd678c9f20f297f98a27f2b75f4c1ee0fe1968"
OLD_DEPLOYMENT_HASH = "c93b8f9d428032cec45527bd2ff1ef9cb237114976dd9e43c00b268819441044"
COMPONENT_NAMES = (
    "expert",
    "router",
    "dense-small",
    "dense-large",
    "final-norm",
    "embedding",
    "lm-head",
    "shared-expert",
    "moe-reduction",
    "attnres",
    "kda-stage",
    "mla-stage",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    _require(_sha256(temporary) == _sha256(source), f"copy hash mismatch: {source}")
    os.replace(temporary, destination)


def _archive_once(source: Path, archive: Path) -> None:
    if archive.exists():
        _require(
            _sha256(archive) == _sha256(source),
            f"existing archive differs from source: {archive}",
        )
    else:
        _atomic_copy(source, archive)


def _health() -> dict[str, Any]:
    fields = "name,uuid,pci.bus_id,compute_cap,memory.total,memory.free,memory.used"
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if completed.returncode:
        return {"status": "UNAVAILABLE", "stderr": completed.stderr.strip()}
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    names = fields.split(",")
    if len(values) != len(names):
        return {"status": "UNPARSEABLE", "stdout": completed.stdout.strip()}
    return {"status": "MEASURED", **dict(zip(names, values, strict=True))}


def _all_zero(mapping: dict[str, Any]) -> bool:
    return bool(mapping) and all(int(value) == 0 for value in mapping.values())


def _ascii_audit(path: Path) -> dict[str, Any]:
    text = path.read_bytes().decode("ascii", errors="ignore")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "contains_batch_capability_export": (
            "coli_cuda_kimi_expert_max_certified_batch" in text
        ),
        "contains_candidate_dll_name": (
            "coli_cuda-sm86-h014-027w-candidate.dll" in text
        ),
        "contains_generic_dll_name": "coli_cuda-sm86.dll" in text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate every promotion prerequisite without changing deployment files",
    )
    args = parser.parse_args()
    candidate = ARTIFACT / "cuda/native/coli_cuda-sm86-h014-027w-candidate.dll"
    deployment_dll = DEPLOYMENT / "native/windows/coli_cuda-sm86.dll"
    deployment_lib = DEPLOYMENT / "native/windows/coli_cuda-sm86.lib"
    deployment_exp = DEPLOYMENT / "native/windows/coli_cuda-sm86.exp"
    safety_path = ARTIFACT / "performance/h014-027w-failclosed-batch-validation.json"
    qualification_path = (
        ARTIFACT / "cuda/h014-027w-same-binary-qualification-candidate.json"
    )
    candidate_certificate_path = (
        ARTIFACT / "cuda/h014-027w-sm86-certification-candidate.json"
    )
    graph_path = ARTIFACT / "cuda/h014-027w-regression-full-93-layer.json"
    final_path = ARTIFACT / "persistent/h014-027w-regression-final-stage.json"
    nonfinal_path = ARTIFACT / "persistent/h014-027w-regression-nonfinal-stages.json"
    stage_zero_path = ARTIFACT / "persistent/h014-027w-regression-stage-zero.json"
    canonical_matrix = ARTIFACT / "k3-cuda-operation-matrix.json"
    canonical_certificate = ARTIFACT / "rtx3090-sm86-certification.json"
    final_matrix = ARTIFACT / "cuda/h014-027w-final-operation-matrix.json"
    final_certificate = ARTIFACT / "cuda/h014-027w-final-sm86-certification.json"
    final_qualification = ARTIFACT / "cuda/h014-027w-same-binary-qualification.json"
    promotion_path = ARTIFACT / "cuda/h014-027w-promotion.json"
    package_manifest = DEPLOYMENT / "native/windows/manifest.json"
    cuobjdump = Path(
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/cuobjdump.exe"
    )
    component_paths = [
        ARTIFACT / f"cuda/h014-027w-regression-{name}.json"
        for name in COMPONENT_NAMES
    ]
    required_paths = [
        candidate,
        deployment_dll,
        deployment_lib,
        deployment_exp,
        safety_path,
        qualification_path,
        candidate_certificate_path,
        graph_path,
        final_path,
        nonfinal_path,
        stage_zero_path,
        canonical_matrix,
        canonical_certificate,
        cuobjdump,
        *component_paths,
    ]
    for path in required_paths:
        _require(path.is_file(), f"missing promotion input: {path}")

    safety = _load(safety_path)
    qualification = _load(qualification_path)
    candidate_certificate = _load(candidate_certificate_path)
    graph = _load(graph_path)
    final = _load(final_path)
    nonfinal = _load(nonfinal_path)
    stage_zero = _load(stage_zero_path)
    health_before = _health()
    _require(health_before["status"] == "MEASURED", "GPU is not healthy before promotion")
    _require(_sha256(candidate) == CANDIDATE_HASH, "candidate DLL hash mismatch")
    _require(
        _sha256(deployment_dll) == OLD_DEPLOYMENT_HASH,
        "deployment DLL is not the expected retained pre-promotion binary",
    )
    _require(safety.get("status") == "PASS", "H014-027w safety receipt did not pass")
    _require(
        all(bool(value) for value in safety.get("gates", {}).values()),
        "H014-027w safety gates are incomplete",
    )
    _require(
        safety["candidate"]["batch4_rejection"]["native_call_attempts"] == 0,
        "batch-4 request crossed the native boundary",
    )
    _require(
        qualification.get("status") == "PASS"
        and qualification["binary"]["sha256"] == CANDIDATE_HASH
        and qualification["summary"]["benchmark_artifacts"] == 12
        and qualification["summary"]["critical_operation_classes"] == 11
        and qualification["summary"]["same_binary_hashes"] == 1,
        "same-binary component qualification is incomplete",
    )
    _require(
        candidate_certificate.get("status") == "PASS"
        and candidate_certificate["summary"]["certified"] == 11
        and candidate_certificate["summary"]["blockers"] == 0,
        "candidate sm_86 certificate is incomplete",
    )
    for path in component_paths:
        document = _load(path)
        _require(document.get("status") == "PASS", f"component failed: {path.name}")
        _require(
            document["backend"]["cuda_library_sha256"] == CANDIDATE_HASH,
            f"component binary mismatch: {path.name}",
        )
    counts = graph["coverage"]["operation_counts"]
    _require(
        graph.get("status") == "PASS"
        and graph["backend"]["cuda_library_sha256"] == CANDIDATE_HASH
        and graph["coverage"]["layers_executed"] == 93
        and len(graph["coverage"]["covered_operations"]) == 11
        and counts["router"] == 276
        and counts["MXFP4_routed_expert"] == 4416
        and graph["correctness"]["routing_equality"] is True
        and graph["correctness"]["stateful_decode_executed"] is True,
        "full graph regression is incomplete",
    )
    _require(
        final.get("status") == "PASS"
        and final["backend"]["cuda_library_sha256"] == CANDIDATE_HASH
        and _all_zero(final["lifecycle"]["warm_delta"]),
        "final-stage P1 regression is incomplete",
    )
    _require(
        nonfinal.get("status") == "PASS"
        and nonfinal["backend"]["cuda_library_sha256"] == CANDIDATE_HASH
        and {stage["layer"] for stage in nonfinal["stages"]} == {1, 3}
        and all(stage["status"] == "PASS" for stage in nonfinal["stages"])
        and all(stage["lifecycle"]["pass"] for stage in nonfinal["stages"]),
        "non-final P1 regressions are incomplete",
    )
    _require(
        stage_zero.get("status") == "PASS"
        and stage_zero["backend"]["cuda_library_sha256"] == CANDIDATE_HASH
        and stage_zero["stage"]["status"] == "PASS"
        and stage_zero["stage"]["lifecycle"]["pass"] is True,
        "stage-zero P1 regression is incomplete",
    )
    if args.check:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "decision": "READY_TO_PROMOTE",
                    "candidate_sha256": CANDIDATE_HASH,
                    "current_deployment_sha256": _sha256(deployment_dll),
                    "component_receipts": len(component_paths),
                    "full_graph_layers": graph["coverage"]["layers_executed"],
                    "router_calls": counts["router"],
                    "routed_expert_calls": counts["MXFP4_routed_expert"],
                    "p1_roles": ["stage_zero", "KDA", "Gated_MLA", "final_head"],
                    "gpu": health_before,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    backup_dll = ARTIFACT / "cuda/native/h014-027w-prepromotion-coli_cuda-sm86.dll"
    old_matrix = ARTIFACT / "cuda/h014-025al-operation-matrix-retained.json"
    old_certificate = ARTIFACT / "cuda/h014-025al-sm86-certification-retained.json"
    _archive_once(deployment_dll, backup_dll)
    _archive_once(canonical_matrix, old_matrix)
    _archive_once(canonical_certificate, old_certificate)
    _atomic_copy(canonical_matrix, final_matrix)

    preflight = {
        "schema_version": "experiment-014-h014-027w-promotion-v1",
        "cycle_id": "H014-027w",
        "status": "PREPARED",
        "candidate_sha256": CANDIDATE_HASH,
        "old_deployment_sha256": OLD_DEPLOYMENT_HASH,
        "health_before": health_before,
        "validated_inputs": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                safety_path,
                qualification_path,
                candidate_certificate_path,
                graph_path,
                final_path,
                nonfinal_path,
                stage_zero_path,
                *component_paths,
            )
        },
    }
    _atomic_json(promotion_path, preflight)

    promoted = False
    try:
        _atomic_copy(candidate, deployment_dll)
        promoted = True
        _require(_sha256(deployment_dll) == CANDIDATE_HASH, "promotion copy mismatch")
        certification = certify_sm86_complete_kimi(
            final_matrix,
            component_paths,
            deployment_dll,
            cuobjdump,
            final_qualification,
            final_certificate,
            cycle_id="H014-027w",
        )
        _require(certification["status"] == "PASS", "final path certification failed")
        _atomic_copy(final_matrix, canonical_matrix)
        _atomic_copy(final_certificate, canonical_certificate)
        health_after = _health()
        _require(health_after["status"] == "MEASURED", "GPU unhealthy after promotion")
        _require(
            health_after["uuid"] == health_before["uuid"],
            "GPU identity changed during promotion",
        )
        link_audit = {
            "lib": _ascii_audit(deployment_lib),
            "exp": _ascii_audit(deployment_exp),
            "classification": "LEGACY_DEVELOPMENT_LINK_ARTIFACTS_EXCLUDED_FROM_RUNTIME",
            "reason": (
                "The generic link artifacts predate the capability export; the candidate "
                "link artifacts embed a candidate-specific DLL name. Runtime certification "
                "loads the exact DLL directly. Regenerate the triplet after H014-028."
            ),
        }
        manifest = {
            "schema_version": "experiment-015-native-runtime-manifest-v1",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "status": "PASS",
            "runtime": {
                "path": "native/windows/coli_cuda-sm86.dll",
                "sha256": CANDIDATE_HASH,
                "bytes": deployment_dll.stat().st_size,
                "max_certified_batch": 2,
                "sm86_cubin": True,
                "compute86_ptx": True,
                "cpu_mathematical_fallbacks": 0,
            },
            "qualification": {
                "path": str(final_qualification.relative_to(ROOT)),
                "sha256": _sha256(final_qualification),
            },
            "sm86_certificate": {
                "path": str(canonical_certificate.relative_to(ROOT)),
                "sha256": _sha256(canonical_certificate),
            },
            "link_artifacts": link_audit,
        }
        _atomic_json(package_manifest, manifest)
        receipt = {
            **preflight,
            "status": "PASS",
            "decision": "PROMOTED",
            "deployment": {
                "path": str(deployment_dll.relative_to(ROOT)),
                "sha256": _sha256(deployment_dll),
                "bytes": deployment_dll.stat().st_size,
                "retained_prior_binary": str(backup_dll.relative_to(ROOT)),
                "retained_prior_binary_sha256": _sha256(backup_dll),
            },
            "regenerated": {
                "operation_matrix": {
                    "path": str(canonical_matrix.relative_to(ROOT)),
                    "sha256": _sha256(canonical_matrix),
                },
                "qualification": {
                    "path": str(final_qualification.relative_to(ROOT)),
                    "sha256": _sha256(final_qualification),
                },
                "sm86_certificate": {
                    "path": str(canonical_certificate.relative_to(ROOT)),
                    "sha256": _sha256(canonical_certificate),
                },
                "package_manifest": {
                    "path": str(package_manifest.relative_to(ROOT)),
                    "sha256": _sha256(package_manifest),
                },
            },
            "health_after": health_after,
            "link_artifact_decision": link_audit,
        }
        _atomic_json(promotion_path, receipt)
    except Exception as exc:
        if promoted:
            _atomic_copy(backup_dll, deployment_dll)
        failure = {
            **preflight,
            "status": "FAIL_ROLLED_BACK" if promoted else "FAIL",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "deployment_sha256_after_rollback": _sha256(deployment_dll),
            "health_after_failure": _health(),
        }
        _atomic_json(promotion_path, failure)
        raise

    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
