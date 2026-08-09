"""Validate the persistent collective on the immutable Qwen3 output-head path."""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_012.baseline_harness import (
    WorkerPool,
    _sha256_file,
    _source_identity,
    _write_json,
)
from swarm_inference.experiments.experiment_012.delegation_harness import (
    build_delegated_topology,
    topology_record,
)
from swarm_inference.experiments.experiment_012.real_model_harness import (
    AGGREGATION,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SAFETENSORS_SHA256,
    WORKER_COUNT,
    _run_reference_process,
    _validate_microshards,
    _validate_worker_tensors,
)
from swarm_inference.experiments.experiment_013.persistent_harness import (
    PersistentCollectiveRunner,
    _install_collective,
    _worker_statuses,
)
from swarm_inference.experiments.experiment_013.persistent_protocol import (
    LEAN_ARCHITECTURE,
)
from swarm_inference.microworker_protocol import NETWORK_PROFILES
from swarm_inference.model.shard_builder import (
    inspect_native_model,
    model_inspection_payload,
    resolve_model,
)

HYPOTHESIS_ID = "H013-010"
BRANCH_FACTOR = 2


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def run_real_model_validation(
    *,
    output_directory: Path,
    microshard_directory: Path,
    model_path: Path | None,
    reference_directory: Path | None,
    max_new_tokens: int,
    operation_deadline_s: float,
    startup_deadline_s: float,
    atol: float,
    rtol: float,
    minimum_cosine: float,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    hypothesis_path = output_directory / "hypothesis.json"
    hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
    if hypothesis.get("hypothesis_id") != HYPOTHESIS_ID:
        raise ValueError("real-model hypothesis must be preregistered")
    _write_json(
        output_directory / "hypothesis-identity.json",
        {
            "path": str(hypothesis_path.resolve()),
            "sha256": _sha256_file(hypothesis_path),
            "bytes": hypothesis_path.stat().st_size,
        },
    )
    repo_root = Path(__file__).resolve().parents[4]
    persistent_script = Path(__file__).with_name("persistent_protocol.py").resolve()
    base_protocol_script = repo_root / "src" / "swarm_inference" / "microworker_protocol.py"
    source_files = [
        Path(__file__).resolve(),
        persistent_script,
        Path(__file__).with_name("persistent_harness.py").resolve(),
        base_protocol_script,
    ]
    _write_json(
        output_directory / "source-identity.json", _source_identity(repo_root, source_files)
    )
    _write_json(
        output_directory / "environment.json",
        {
            "captured_unix_ns": time.time_ns(),
            "platform": platform.platform(),
            "root_process_id": os.getpid(),
            "network_evidence": "single-host physical TCP loopback",
        },
    )

    pool: WorkerPool | None = None
    runner: PersistentCollectiveRunner | None = None
    rows: list[dict[str, Any]] = []
    workers: list[Any] = []
    reference: dict[str, Any] = {}
    tensor_validation: dict[str, Any] = {}
    resolver_evidence: dict[str, Any] = {}
    shard_validation: dict[str, Any] = {}
    installation: dict[str, Any] = {}
    root_prepare: dict[str, Any] = {}
    statuses: dict[str, Any] = {}
    shutdown: dict[str, Any] = {}
    fatal_error: dict[str, Any] | None = None
    topology_payload: dict[str, Any] = {}
    try:
        resolution_started_ns = time.perf_counter_ns()
        if model_path is None:
            resolved = resolve_model(MODEL_ID, revision=MODEL_REVISION, allow_download=True)
            acquisition_mode = "canonical_registry_resolver"
        else:
            resolved = resolve_model(str(model_path), revision=MODEL_REVISION, allow_download=False)
            acquisition_mode = "verified_local_immutable_snapshot"
        inspection = model_inspection_payload(inspect_native_model(resolved))
        source_weight = resolved.path / "model.safetensors"
        source_hash = _sha256_file(source_weight)
        resolver_evidence = {
            "requested_model_id": MODEL_ID,
            "requested_revision": MODEL_REVISION,
            "resolved_model_id": MODEL_ID,
            "resolver_returned_model_id": resolved.model_id,
            "resolved_revision": resolved.revision,
            "resolved_path": str(resolved.path),
            "downloaded": resolved.downloaded,
            "acquisition_mode": acquisition_mode,
            "source_safetensors_sha256": source_hash,
            "resolution_elapsed_ns": time.perf_counter_ns() - resolution_started_ns,
            "inspection": inspection,
        }
        if (
            resolved.revision != MODEL_REVISION
            or source_hash != MODEL_SAFETENSORS_SHA256
            or inspection["architecture"] != "Qwen3ForCausalLM"
            or inspection["hidden_size"] != 1024
            or inspection["vocabulary_size"] != 151936
        ):
            raise ValueError("immutable model identity differs from Experiment 012")
        _write_json(output_directory / "model-resolution.json", resolver_evidence)
        shard_validation = _validate_microshards(microshard_directory)
        _write_json(output_directory / "microshard-validation.json", shard_validation)
        if reference_directory is None:
            reference = _run_reference_process(
                output_directory=output_directory,
                model_path=resolved.path,
                max_new_tokens=max_new_tokens,
            )
            reference_mode = "fresh_independent_full_model"
        else:
            reference_path = reference_directory / "reference.json"
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            if (
                reference.get("model_id") != MODEL_ID
                or reference.get("model_revision") != MODEL_REVISION
                or reference.get("manual_matches_generate") is not True
                or int(reference.get("max_new_tokens", 0)) != max_new_tokens
            ):
                raise ValueError("reused reference identity differs from the locked path")
            reference_mode = "hash_verified_experiment_012_reference_reuse"
            reference_files = [reference_path]
            for step in reference["steps"]:
                reference_files.extend((Path(step["hidden_path"]), Path(step["logits_path"])))
            _write_json(
                output_directory / "reference-reuse.json",
                {
                    "mode": reference_mode,
                    "source_directory": str(reference_directory.resolve()),
                    "source_experiment": "012/H012-012",
                    "cuda_unavailable_for_fresh_reference": True,
                    "files": [
                        {
                            "path": str(path.resolve()),
                            "bytes": path.stat().st_size,
                            "sha256": _sha256_file(path),
                        }
                        for path in reference_files
                    ],
                },
            )

        runtime_profiles = tuple(
            {
                "name": "real_output_head_rank",
                "compute_delay_ms": 0.0,
                "capacity_score": 1.0,
                "maximum_payload_bytes": 1 << 20,
                "real_model_shard": {
                    "manifest_path": record["manifest_path"],
                    "weight_file": record["weight_file"],
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "tensor_name": "lm_head.weight",
                    "torch_threads": 1,
                    "evidence_directory": str(
                        (
                            output_directory
                            / "raw"
                            / "worker-tensors"
                            / f"worker-{int(record['rank']):06d}"
                        ).resolve()
                    ),
                },
            }
            for record in shard_validation["records"]
        )
        pool = WorkerPool(
            count=WORKER_COUNT,
            directory=output_directory / "workers",
            startup_deadline_s=startup_deadline_s,
            protocol_script=base_protocol_script,
            runtime_profiles=runtime_profiles,
            include_site_packages=True,
            protocol_module=("swarm_inference.experiments.experiment_013.persistent_protocol"),
        )
        workers = pool.start()
        topology = build_delegated_topology(
            workers,
            branch_factor=BRANCH_FACTOR,
            topology_id="h013-010-persistent-real-output-head",
            route_lease_id="h013-010-persistent-real-lease",
            route_generation=1,
        )
        topology_payload = topology_record(topology)
        _write_json(output_directory / "topology.json", topology_payload)
        setup_started_ns = time.perf_counter_ns()
        installation = _install_collective(
            topology,
            collective_id="h013-010-real-model-collective",
            architecture=LEAN_ARCHITECTURE,
            profile=NETWORK_PROFILES["same_host_shaped"],
            mailbox_depth=2,
        )
        runner = PersistentCollectiveRunner(
            topology=topology,
            collective_id="h013-010-real-model-collective",
            output_directory=output_directory,
            payload_bytes=0,
            operation_deadline_s=operation_deadline_s,
            cycle_id=HYPOTHESIS_ID,
            architecture=LEAN_ARCHITECTURE,
        )
        root_prepare = runner.prepare(NETWORK_PROFILES["same_host_shaped"])
        setup_ms = (time.perf_counter_ns() - setup_started_ns) / 1_000_000
        workload = {
            "kind": "real_model_lm_head",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tensor_name": "lm_head.weight",
            "hidden_size": 1024,
            "vocabulary_size": 151936,
            "hidden_dtype": "float32-le",
        }
        operation_plan: list[tuple[dict[str, Any], str, int]] = []
        first_step = dict(reference["steps"][0])
        operation_plan.append((first_step, "warmup", 0))
        operation_plan.append((first_step, "repeat", 0))
        operation_plan.append((first_step, "repeat", 1))
        operation_plan.extend((dict(step), "measured", 0) for step in reference["steps"])
        for generation, (step, phase, trial_index) in enumerate(operation_plan, start=1):
            hidden_bytes = Path(step["hidden_path"]).read_bytes()
            row = runner.run_trial(
                trial_index=trial_index,
                warmup=phase == "warmup",
                execution_generation=generation,
                profile=NETWORK_PROFILES["same_host_shaped"],
                operation_id_override=(
                    f"h013-010-step{int(step['step_index']):02d}-{phase}-{trial_index}-{generation}"
                ),
                aggregation=AGGREGATION,
                workload=workload,
                payload_b64=base64.b64encode(hidden_bytes).decode("ascii"),
            )
            row.update(
                {
                    "evidence_class": "real_immutable_model_output_head",
                    "phase": phase,
                    "step_index": int(step["step_index"]),
                    "reference_token_id": int(step["token_id"]),
                    "selected_token_id": (
                        int(row["actual"]["token_id"]) if row["actual"] else None
                    ),
                }
            )
            row["token_equal"] = row["selected_token_id"] == row["reference_token_id"]
            row["correctness"] = bool(row["correctness"] and row["token_equal"])
            if not row["correctness"]:
                row["status"] = "incorrect"
            rows.append(row)
        statuses = _worker_statuses(topology)
        _write_json(output_directory / "worker-statuses.json", statuses)
        _write_json(
            output_directory / "lifecycle.json",
            {
                "collective_setup_ms": setup_ms,
                "installation": installation,
                "root_prepare": root_prepare,
            },
        )
    except BaseException as error:
        fatal_error = {
            "error_type": type(error).__name__,
            "error": str(error),
            "timestamp_unix_ns": time.time_ns(),
        }
        _write_json(output_directory / "fatal-error.json", fatal_error)
    finally:
        if runner is not None:
            _write_json(output_directory / "root-teardown.json", runner.close())
        if pool is not None:
            shutdown = pool.stop()
            _write_json(output_directory / "worker-shutdown.json", shutdown)

    operation_steps = {str(row["operation_id"]): int(row["step_index"]) for row in rows}
    if reference and operation_steps:
        tensor_validation = _validate_worker_tensors(
            evidence_directory=output_directory / "raw" / "worker-tensors",
            reference=reference,
            operation_steps=operation_steps,
            atol=atol,
            rtol=rtol,
            minimum_cosine=minimum_cosine,
        )
        _write_json(
            output_directory / "correctness" / "tensor-comparisons.json",
            tensor_validation,
        )

    worker_events = [
        event
        for worker in workers
        for event in _read_events(output_directory / "workers" / worker.worker_id / "trace.jsonl")
    ]
    reduction_events = [
        event for event in worker_events if event.get("event") == "persistent_subtree_reduced"
    ]
    repeats = [row for row in rows if row.get("phase") == "repeat"]
    ready_proofs = [worker.ready.get("model_shard_proof") for worker in workers]
    measured = [row for row in rows if not row.get("warmup")]
    checks = {
        "resolver_identity": bool(resolver_evidence)
        and resolver_evidence.get("resolved_model_id") == MODEL_ID
        and resolver_evidence.get("resolved_revision") == MODEL_REVISION
        and resolver_evidence.get("source_safetensors_sha256") == MODEL_SAFETENSORS_SHA256,
        "reference_generation": bool(reference)
        and reference.get("manual_matches_generate") is True,
        "worker_process_isolation": len(workers) == WORKER_COUNT
        and len({worker.process_id for worker in workers}) == WORKER_COUNT,
        "verified_partial_loads": len(ready_proofs) == WORKER_COUNT
        and all(proof and not proof["complete_model_loaded"] for proof in ready_proofs)
        and all(
            proof and proof["local_tensor_hash"] == proof["expected_local_tensor_hash"]
            for proof in ready_proofs
        ),
        "token_identity": bool(rows) and all(bool(row["token_equal"]) for row in rows),
        "tensor_correctness": tensor_validation.get("all_passed") is True,
        "deterministic_repetition": len(repeats) == 2
        and repeats[0].get("actual") == repeats[1].get("actual"),
        "persistent_hierarchy": len(reduction_events) == len(rows) * WORKER_COUNT
        and any(event.get("child_worker_ids") for event in reduction_events),
        "bounded_root": bool(rows)
        and all(
            row["root_direct_degree"] == BRANCH_FACTOR
            and row["root_messages_total"] == 2 * BRANCH_FACTOR
            and row["root_leaf_rpc_count"] == 0
            for row in rows
        ),
        "warm_reuse": bool(measured)
        and all(
            row["total_connection_count"] == 0
            and row["new_task_creation"] == 0
            and row["topology_rebuilds"] == 0
            for row in measured
        ),
    }
    status = "PASS" if fatal_error is None and all(checks.values()) else "FAIL"
    summary = {
        "experiment_id": "013",
        "hypothesis_id": HYPOTHESIS_ID,
        "status": status,
        "architecture": LEAN_ARCHITECTURE,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checks": checks,
        "reference_token_ids": reference.get("manual_token_ids"),
        "reference_mode": reference_mode if reference else None,
        "persistent_token_ids": [row.get("selected_token_id") for row in rows],
        "tensor_validation": {
            key: tensor_validation.get(key)
            for key in (
                "expected_comparison_count",
                "actual_comparison_count",
                "all_passed",
                "maximum_absolute_error",
                "minimum_cosine_observed",
            )
        },
        "root_metrics": {
            "degree": BRANCH_FACTOR,
            "messages": 2 * BRANCH_FACTOR,
            "leaf_rpcs": 0,
            "bytes_median": (
                statistics.median(int(row["root_bytes_total"]) for row in measured)
                if measured
                else None
            ),
        },
        "warm_latency_ms": {
            "p50": (
                statistics.median(float(row["end_to_end_latency_ms"]) for row in measured)
                if measured
                else None
            ),
            "values": [float(row["end_to_end_latency_ms"]) for row in measured],
        },
        "total_traffic_bytes_median": (
            statistics.median(int(row["total_bytes"]) for row in measured) if measured else None
        ),
        "hierarchy_depth": topology_payload.get("hierarchy_depth"),
        "rows": rows,
        "worker_process_ids": [worker.process_id for worker in workers],
        "worker_status_count": len(statuses),
        "shutdown": shutdown,
        "fatal_error": fatal_error,
    }
    _write_json(output_directory / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microshards", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--reference-directory", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--operation-deadline-s", type=float, default=60.0)
    parser.add_argument("--startup-deadline-s", type=float, default=180.0)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    args = parser.parse_args(argv)
    result = run_real_model_validation(
        output_directory=args.output.resolve(),
        microshard_directory=args.microshards.resolve(),
        model_path=args.model_path.resolve() if args.model_path else None,
        reference_directory=(
            args.reference_directory.resolve() if args.reference_directory else None
        ),
        max_new_tokens=args.max_new_tokens,
        operation_deadline_s=args.operation_deadline_s,
        startup_deadline_s=args.startup_deadline_s,
        atol=args.atol,
        rtol=args.rtol,
        minimum_cosine=args.minimum_cosine,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
