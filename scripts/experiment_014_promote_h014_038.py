"""Fail-closed promotion of the exact H014-038 Kimi CUDA/runtime evidence chain."""

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
CUDA = ARTIFACT / "cuda"
PERSISTENT = ARTIFACT / "persistent"
CANDIDATE_SHA256 = "c3bdb40d49a1b0e512ddb1e84485e0bdcf9d2e6ac6fa4049d81ff2ecd12ae326"
ORACLE_INPUT_FINGERPRINTS = (
    "sha256:4c234e5cc2baa412f4f64f470c5af83a6ca84a958ae2da56b3612cd0de4dab27",
    "sha256:05bf26eda750a1f025c1368d03901e759b49017bb0199b49bdfca80ee0f37375",
    "sha256:efea01868cd7d154a44531f84fa0e211f91a67936e927ec031d8be6a5233c798",
)
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
SOURCE_PATHS = (
    "third_party/colibri/c/backend_cuda.cu",
    "third_party/colibri/c/backend_cuda.h",
    "third_party/colibri/c/backend_gpu_compat.h",
    "src/swarm_inference/execution/kimi_cuda_runtime.py",
    "src/swarm_inference/execution/kimi_k3_graph_runtime.py",
    "src/swarm_inference/execution/kimi_k3_stage.py",
    "src/swarm_inference/experiments/experiment_014/__main__.py",
    "src/swarm_inference/experiments/experiment_014/cuda.py",
    "src/swarm_inference/experiments/experiment_014/deployment_admission.py",
    "src/swarm_inference/experiments/experiment_014/full_cuda.py",
    "src/swarm_inference/experiments/experiment_014/performance.py",
    "src/swarm_inference/experiments/experiment_014/persistent_stages.py",
    "src/swarm_inference/experiments/experiment_014/coarse_stage_transport.py",
    "src/swarm_inference/experiments/experiment_014/coarse_network_analysis.py",
    "src/swarm_inference/experiments/experiment_014/complete_stage_batch.py",
    "src/swarm_inference/experiments/experiment_014/deployment_canary.py",
    "src/swarm_inference/experiments/experiment_014/experiment_015_package.py",
    "src/swarm_inference/experiments/experiment_014/final_deployment.py",
    "src/swarm_inference/experiments/experiment_014/remote_acquisition.py",
    "src/swarm_inference/experiments/experiment_014/deployment_recovery.py",
    "src/swarm_inference/experiments/experiment_014/runtime_qualification.py",
    "src/swarm_inference/experiments/experiment_014/sub_layer_microwork.py",
    "src/swarm_inference/experiments/experiment_014/sub_layer_batch.py",
    "src/swarm_inference/experiments/experiment_014/performance_model.py",
    "src/swarm_inference/experiments/experiment_014/capacity_topology.py",
    "src/swarm_inference/model/kimi_k3.py",
    "src/swarm_inference/model/kimi_tokenizer.py",
    "src/swarm_inference/model/mxfp4.py",
    "src/swarm_inference/worker/stage_runtime.py",
    "src/swarm_inference/transport/stage_tensor.py",
    "pyproject.toml",
    "uv.lock",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    _require(_sha256(temporary) == _sha256(source), f"copy hash mismatch: {source}")
    os.replace(temporary, destination)


def _health() -> dict[str, Any]:
    fields = "name,uuid,pci.bus_id,compute_cap,memory.total,memory.free,memory.used,pstate"
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


def _source_manifest() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    bundle = hashlib.sha256()
    for relative in SOURCE_PATHS:
        source = ROOT / relative
        _require(source.is_file(), f"missing final source: {relative}")
        digest = _sha256(source)
        files[relative] = {"bytes": source.stat().st_size, "sha256": digest}
        bundle.update(relative.encode("utf-8"))
        bundle.update(b"\0")
        bundle.update(digest.encode("ascii"))
        bundle.update(b"\n")
    return {
        "schema_version": "experiment-014-h014-038-source-manifest-v1",
        "cycle_id": "H014-038",
        "status": "PASS",
        "source_bundle_sha256": bundle.hexdigest(),
        "files": files,
    }


def _oracle_manifest() -> dict[str, Any]:
    directory = ARTIFACT / "oracle-full-93-idot0"
    names = (
        "serial-oracle-receipt.json",
        "hidden-trace.f32",
        "routes.txt",
        "prefill-logits.f32",
        "states.txt",
    )
    files: dict[str, dict[str, Any]] = {}
    bundle = hashlib.sha256()
    for name in names:
        source = directory / name
        _require(source.is_file(), f"missing canonical oracle member: {source}")
        digest = _sha256(source)
        files[name] = {"bytes": source.stat().st_size, "sha256": digest}
        bundle.update(name.encode("utf-8"))
        bundle.update(b"\0")
        bundle.update(digest.encode("ascii"))
        bundle.update(b"\n")
    _require(
        files["hidden-trace.f32"]["sha256"]
        == "0a432e25af5897e1d0ba62eb5560f5da8c8fbadb3eb93db63571b6db562b1a8d",
        "canonical oracle trace hash changed",
    )
    _require(
        files["routes.txt"]["sha256"]
        == "c734d864e8ac40ada5fd1fbd6c64da5959b9610561313ec63e7b79a499a91288",
        "canonical oracle routes hash changed",
    )
    _require(
        files["prefill-logits.f32"]["sha256"]
        == "8cd947eb8f2fddda751b0527d81f5f68c306a57c2526fefbcea937d81f590248",
        "canonical oracle logits hash changed",
    )
    return {
        "schema_version": "experiment-014-h014-038-oracle-family-v1",
        "cycle_id": "H014-038j",
        "status": "PASS",
        "family": "checkpoint-faithful deterministic router K3_IDOT=0",
        "directory": str(directory.resolve()),
        "bundle_sha256": bundle.hexdigest(),
        "expected_final_stage_input_fingerprints": list(ORACLE_INPUT_FINGERPRINTS),
        "files": files,
    }


def _prepare_pass(stage: dict[str, Any]) -> bool:
    prepare = stage.get("prepare", {})
    fixture = prepare.get("stage_fixture", {})
    transport = fixture.get("canonical_transport_round_trip", {})
    quiescence = fixture.get("thread_quiescence", {})
    cpu_threads = fixture.get("cpu_transport_threads", {})
    return bool(
        prepare.get("pass") is True
        and fixture.get("iterations") == 7
        and fixture.get("temporary_memory_recovered") is True
        and fixture.get("active_sessions_after") == 0
        and fixture.get("research_records_removed") is True
        and fixture.get("serving_execute_count_restored") is True
        and transport.get("pass") is True
        and transport.get("source_iteration") == 6
        and transport.get("final_warm_iteration") == 7
        and transport.get("shape") == [1, 9, 7168]
        and transport.get("dtype") == "float32"
        and transport.get("raw_bytes") == 258_048
        and transport.get("encoded_bytes") == 258_048
        and quiescence.get("pass") is True
        and quiescence.get("minimum_observation_ms") == 3_500.0
        and quiescence.get("required_stable_samples") == 20
        and int(quiescence.get("stable_samples_observed", 0)) >= 20
        and cpu_threads.get("intraop_threads") == 1
        and cpu_threads.get("interop_threads") == 1
        and fixture.get("single_host_thread") is True
        and len(set(fixture.get("host_thread_native_ids", []))) == 1
        and len(set(fixture.get("host_thread_idents", []))) == 1
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    candidate = CUDA / "native/coli_cuda-sm86-h014-033c-mla16k-candidate.dll"
    final_binary = CUDA / "native/coli_cuda-sm86-h014-038-final.dll"
    source_matrix = CUDA / "h014-038-final-operation-matrix.json"
    promoted_matrix = CUDA / "h014-038-promoted-operation-matrix.json"
    promoted_qualification = CUDA / "h014-038-promoted-same-binary-qualification.json"
    promoted_certificate = CUDA / "h014-038-promoted-sm86-certification.json"
    canonical_matrix = ARTIFACT / "k3-cuda-operation-matrix.json"
    canonical_certificate = ARTIFACT / "rtx3090-sm86-certification.json"
    graph_path = CUDA / "h014-038-regression-full-93-layer.json"
    final_path = PERSISTENT / "h014-038-regression-final-stage.json"
    final_repeat_paths = (
        PERSISTENT / "h014-038u-final-repeat-1.json",
        PERSISTENT / "h014-038u-final-repeat-2.json",
    )
    nonfinal_path = PERSISTENT / "h014-038-regression-nonfinal-stages.json"
    stage_zero_path = PERSISTENT / "h014-038-regression-stage-zero.json"
    post_freeze_paths = {
        "complete_kda_batch": ARTIFACT
        / "performance/h014-038ah2-complete-stage-batch-layer89.json",
        "complete_mla_batch": ARTIFACT
        / "performance/h014-038ah2-complete-stage-batch-layer91.json",
        "steady_stage_zero": PERSISTENT / "h014-038aj2-stage-zero-steady.json",
        "sub_layer_correctness": ARTIFACT
        / "sub-layer/h014-sub-011-promoted-four-worker-real-expert.json",
        "sub_layer_batch": ARTIFACT
        / "sub-layer/h014-sub-012a-006f-promoted-single-interval-batch.json",
        "coarse_transport": ARTIFACT
        / "coarse/h014-038ae-promoted-stage0-stage1-tcp.json",
        "coarse_network": ARTIFACT / "coarse/h014-038ah-network-analysis.json",
        "capacity": ARTIFACT
        / "performance/h014-038ak-final-capacity-topology-economics.json",
    }
    source_manifest_path = CUDA / "h014-038-source-manifest.json"
    oracle_manifest_path = CUDA / "h014-038-canonical-oracle-family.json"
    promotion_path = CUDA / "h014-038-promotion.json"
    cuobjdump = Path(
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/cuobjdump.exe"
    )
    component_paths = [
        CUDA / f"h014-038-regression-{name}.json" for name in COMPONENT_NAMES
    ]
    required = (
        candidate,
        source_matrix,
        graph_path,
        final_path,
        *final_repeat_paths,
        nonfinal_path,
        stage_zero_path,
        *post_freeze_paths.values(),
        cuobjdump,
        *component_paths,
    )
    for path in required:
        _require(path.is_file(), f"missing promotion prerequisite: {path}")
    _require(_sha256(candidate) == CANDIDATE_SHA256, "candidate binary hash mismatch")

    source_manifest = _source_manifest()
    oracle_manifest = _oracle_manifest()
    graph = _load(graph_path)
    final = _load(final_path)
    final_repeats = [_load(path) for path in final_repeat_paths]
    nonfinal = _load(nonfinal_path)
    stage_zero = _load(stage_zero_path)
    post_freeze = {
        name: _load(path) for name, path in post_freeze_paths.items()
    }
    health_before = _health()
    _require(health_before.get("status") == "MEASURED", "GPU unhealthy before promotion")

    for path in component_paths:
        value = _load(path)
        _require(value.get("status") == "PASS", f"component failed: {path.name}")
        _require(
            value.get("backend", {}).get("cuda_library_sha256") == CANDIDATE_SHA256,
            f"component binary mismatch: {path.name}",
        )

    counts = graph.get("coverage", {}).get("operation_counts", {})
    _require(
        graph.get("status") == "PASS"
        and graph.get("backend", {}).get("cuda_library_sha256") == CANDIDATE_SHA256
        and graph.get("coverage", {}).get("layers_executed") == 93
        and graph.get("coverage", {}).get("no_cpu_mathematical_fallback") is True
        and counts.get("router") == 276
        and counts.get("MXFP4_routed_expert") == 4_416
        and graph.get("correctness", {}).get("routing_equality") is True
        and graph.get("correctness", {}).get("stateful_decode_executed") is True
        and float(
            graph.get("correctness", {}).get("maximum_layer_relative_l2_error", 1.0)
        )
        <= 2e-6,
        "full 93-layer graph regression is incomplete",
    )

    final_inputs = tuple(
        row.get("input_fingerprint")
        for row in final.get("correctness", {}).get("positions", [])
    )
    final_stage = {
        "prepare": {
            "pass": True,
            "stage_fixture": final["lifecycle"]["before"]["executor"][
                "prepare_warmup"
            ]["stage_fixture"],
        }
    }
    _require(
        final.get("status") == "PASS"
        and final.get("backend", {}).get("cuda_library_sha256") == CANDIDATE_SHA256
        and final.get("correctness", {}).get("pass") is True
        and final.get("lifecycle", {}).get("compute_thread_consistent") is True
        and final_inputs == ORACLE_INPUT_FINGERPRINTS
        and _all_zero(final.get("lifecycle", {}).get("warm_delta", {}))
        and all(
            _all_zero(row.get("lifecycle_delta", {}))
            for row in final.get("lifecycle", {}).get("per_generation_deltas", [])
        )
        and _prepare_pass(final_stage),
        "final-stage regression/READY chain is incomplete",
    )
    for path, repeat in zip(final_repeat_paths, final_repeats, strict=True):
        repeat_inputs = tuple(
            row.get("input_fingerprint")
            for row in repeat.get("correctness", {}).get("positions", [])
        )
        _require(
            repeat.get("status") == "PASS"
            and repeat.get("backend", {}).get("cuda_library_sha256")
            == CANDIDATE_SHA256
            and repeat.get("correctness", {}).get("pass") is True
            and repeat.get("lifecycle", {}).get("pass") is True
            and repeat.get("lifecycle", {}).get("compute_thread_consistent") is True
            and repeat_inputs == ORACLE_INPUT_FINGERPRINTS
            and _all_zero(repeat.get("lifecycle", {}).get("warm_delta", {})),
            f"independent final-stage repeat failed: {path.name}",
        )

    stages = nonfinal.get("stages", [])
    _require(
        nonfinal.get("status") == "PASS"
        and nonfinal.get("backend", {}).get("cuda_library_sha256") == CANDIDATE_SHA256
        and nonfinal.get("final_stage_regression", {}).get("sha256")
        == _sha256(final_path)
        and {row.get("layer") for row in stages} == {1, 3},
        "non-final dependency chain is incomplete",
    )
    for stage in stages:
        batch = stage.get("production_batch", {})
        _require(
            stage.get("status") == "PASS"
            and stage.get("correctness", {}).get("pass") is True
            and stage.get("lifecycle", {}).get("pass") is True
            and stage.get("lifecycle", {}).get("compute_thread_consistent") is True
            and _prepare_pass(stage)
            and batch.get("pass") is True
            and batch.get("certified_batch") == 8
            and batch.get("native_supported_batches") == [1, 2, 4, 8, 16]
            and batch.get("batch_8", {}).get("pass") is True
            and batch.get("batch_9_guard", {}).get("pass") is True
            and batch.get("post_guard_batch_1", {}).get("pass") is True
            and batch.get("memory", {}).get("stable_allocator_plateau") is True,
            f"non-final role failed final gates: layer {stage.get('layer')}",
        )

    zero = stage_zero.get("stage", {})
    _require(
        stage_zero.get("status") == "PASS"
        and stage_zero.get("backend", {}).get("cuda_library_sha256")
        == CANDIDATE_SHA256
        and stage_zero.get("prior_stage_regression", {}).get("sha256")
        == _sha256(nonfinal_path)
        and zero.get("status") == "PASS"
        and zero.get("correctness", {}).get("pass") is True
        and zero.get("lifecycle", {}).get("pass") is True
        and zero.get("lifecycle", {}).get("compute_thread_consistent") is True
        and _prepare_pass(zero),
        "stage-zero dependency chain is incomplete",
    )

    for name in (
        "complete_kda_batch",
        "complete_mla_batch",
        "sub_layer_correctness",
        "sub_layer_batch",
        "coarse_transport",
    ):
        value = post_freeze[name]
        _require(
            value.get("status") == "PASS"
            and value.get("sources", {}).get("cuda_library", {}).get("sha256")
            == CANDIDATE_SHA256,
            f"post-freeze final-binary receipt failed: {name}",
        )
    steady_zero = post_freeze["steady_stage_zero"]
    steady_window = steady_zero.get("stage", {}).get("steady_performance", {})
    _require(
        steady_zero.get("status") == "PASS"
        and steady_zero.get("backend", {}).get("cuda_library_sha256")
        == CANDIDATE_SHA256
        and steady_window.get("status") == "PASS"
        and steady_window.get("hypothesis_supported") is True
        and all(steady_window.get("acceptance_gates", {}).values()),
        "post-freeze steady stage-zero receipt failed",
    )
    coarse_network = post_freeze["coarse_network"]
    coarse_recommendation = coarse_network.get("recommendation", {})
    _require(
        coarse_network.get("status") == "PASS"
        and all(coarse_network.get("acceptance_gates", {}).values())
        and coarse_recommendation.get("coupled_operating_point") is True
        and coarse_recommendation.get("maximum_tested_rtt_ms") == 5.0
        and coarse_recommendation.get("minimum_tested_bandwidth_gbps") == 10.0
        and float(coarse_recommendation.get("capacity_retention_percent", 0.0))
        >= 90.0,
        "post-freeze coarse-network admission failed",
    )
    capacity = post_freeze["capacity"]
    capacity_sources = capacity.get("sources", {})
    capacity_dependencies = {
        "complete_kda_batch": "complete_kda_batch",
        "complete_mla_batch": "complete_mla_batch",
        "stage_zero": "steady_stage_zero",
        "sub_layer_batch": "sub_layer_batch",
        "coarse_transport": "coarse_transport",
        "coarse_network": "coarse_network",
    }
    _require(
        capacity.get("status") == "PASS"
        and all(capacity.get("acceptance_gates", {}).values())
        and capacity.get("recommended_experiment_015_topology", {}).get(
            "candidate_id"
        )
        == "B"
        and capacity.get("recommended_experiment_015_topology", {}).get(
            "worker_count"
        )
        == 93
        and all(
            capacity_sources.get(source_name, {}).get("sha256")
            == _sha256(post_freeze_paths[path_name])
            for source_name, path_name in capacity_dependencies.items()
        ),
        "post-freeze capacity/topology evidence chain failed",
    )

    preflight = {
        "schema_version": "experiment-014-h014-038-promotion-v1",
        "cycle_id": "H014-038",
        "status": "VALIDATED",
        "decision": "READY_TO_PROMOTE" if args.check else "PROMOTION_PREPARED",
        "candidate": {
            "path": str(candidate.relative_to(ROOT)),
            "sha256": _sha256(candidate),
            "bytes": candidate.stat().st_size,
        },
        "health_before": health_before,
        "source_manifest": source_manifest,
        "oracle_manifest": oracle_manifest,
        "validated_inputs": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                source_matrix,
                graph_path,
                final_path,
                *final_repeat_paths,
                nonfinal_path,
                stage_zero_path,
                *post_freeze_paths.values(),
                *component_paths,
            )
        },
        "summary": {
            "component_receipts": len(component_paths),
            "critical_operation_classes": 11,
            "full_graph_layers": 93,
            "router_calls": 276,
            "routed_expert_calls": 4_416,
            "p1_roles": ["stage_zero", "KDA", "Gated_MLA", "final_head"],
            "production_batch": 8,
            "native_max_batch": 16,
            "first_rejected_batch": 9,
            "physical_sm86_execution": "PENDING_EXPERIMENT_015_CANARY",
            "post_freeze_binary_receipts": 6,
            "coarse_admission": "5 ms / 10 Gbps coupled",
            "recommended_topology": "B / WHOLE-LAYER / 93 workers",
        },
    }
    if args.check:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    _atomic_copy(candidate, final_binary)
    _require(_sha256(final_binary) == CANDIDATE_SHA256, "final binary copy mismatch")
    _atomic_copy(source_matrix, promoted_matrix)
    certification = certify_sm86_complete_kimi(
        promoted_matrix,
        component_paths,
        final_binary,
        cuobjdump,
        promoted_qualification,
        promoted_certificate,
        cycle_id="H014-038",
    )
    _require(certification.get("status") == "PASS", "promoted sm_86 certification failed")
    _atomic_copy(promoted_matrix, canonical_matrix)
    _atomic_copy(promoted_certificate, canonical_certificate)
    _atomic_json(source_manifest_path, source_manifest)
    _atomic_json(oracle_manifest_path, oracle_manifest)
    health_after = _health()
    _require(
        health_after.get("status") == "MEASURED"
        and health_after.get("uuid") == health_before.get("uuid"),
        "GPU health/identity changed during promotion",
    )
    receipt = {
        **preflight,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "decision": "PROMOTED_FOR_PRE_CANARY_USE",
        "final_binary": {
            "path": str(final_binary.relative_to(ROOT)),
            "sha256": _sha256(final_binary),
            "bytes": final_binary.stat().st_size,
            "sm86_sass": True,
            "compute86_ptx": True,
            "physical_sm86_execution": False,
        },
        "regenerated": {
            "operation_matrix": {
                "path": str(canonical_matrix.relative_to(ROOT)),
                "sha256": _sha256(canonical_matrix),
            },
            "same_binary_qualification": {
                "path": str(promoted_qualification.relative_to(ROOT)),
                "sha256": _sha256(promoted_qualification),
            },
            "sm86_certificate": {
                "path": str(canonical_certificate.relative_to(ROOT)),
                "sha256": _sha256(canonical_certificate),
            },
            "source_manifest": {
                "path": str(source_manifest_path.relative_to(ROOT)),
                "sha256": _sha256(source_manifest_path),
            },
            "oracle_manifest": {
                "path": str(oracle_manifest_path.relative_to(ROOT)),
                "sha256": _sha256(oracle_manifest_path),
            },
        },
        "health_after": health_after,
        "invalid_or_nonpromotable_evidence": {
            "h014_038_missing_bos": "invalid fixture input",
            "h014_038a_timeout": "incomplete external timeout",
            "h014_038h": "mixed oracle family; lifecycle diagnostic only",
            "h014_038i": "mixed oracle family; lifecycle diagnostic only",
        },
    }
    _atomic_json(promotion_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
