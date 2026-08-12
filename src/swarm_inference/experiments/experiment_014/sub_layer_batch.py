"""Incremental batching through the real four-worker Kimi expert collective."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swarm_inference.execution.kimi_k3_stage import PersistentKimiStageExecutor, _timing
from swarm_inference.experiments.experiment_014.batch_safety import _health_snapshot
from swarm_inference.experiments.experiment_014.complete_stage_batch import _batch_boundaries
from swarm_inference.experiments.experiment_014.cuda import (
    _array_fingerprint,
    _device_identity,
    _numerical_metrics,
    _sha256_file,
)
from swarm_inference.experiments.experiment_014.persistent_stages import _stage_fixtures
from swarm_inference.experiments.experiment_014.sub_layer_microwork import (
    _atomic_json,
    _microwork_lifecycle_delta,
    _microwork_process_snapshot,
    _PersistentExpertCollective,
    _request,
    _start_worker,
    _WorkerProxy,
)

SCHEMA_VERSION = "experiment-014-k3-real-expert-microwork-batch-v1"
BUFFERED_SCHEMA_VERSION = "experiment-014-k3-real-expert-microwork-batch-v2"
OVERLAP_SCHEMA_VERSION = "experiment-014-k3-real-expert-microwork-overlap-v1"
INCREMENTAL_BATCHES = (1, 2, 4, 8)
H014_SUB_006C_BATCH8_WALL_P50_MS = 21.03625
H014_SUB_006C_BATCH8_WALL_P99_MS = 22.290927
H014_SUB_006C_MESSAGES_PER_ROW = 1.0
H014_SUB_006C_TENSOR_BYTES_PER_ROW = 286720.0
H014_SUB_006F_BATCH8_WALL_P50_MS = 17.4095
H014_SUB_006F_BATCH8_WALL_P99_MS = 19.02525
H014_SUB_030E_SHARED_BATCH8_WALL_P50_MS = 1.3241


def _close_sessions(executor: PersistentKimiStageExecutor, session_ids: list[str]) -> None:
    for session_id in session_ids:
        with suppress(KeyError):
            executor.close_session(session_id)


def _reference_batches(
    request: Any,
    checkpoint: Path,
    cuda_library: Path,
    fixtures: list[np.ndarray],
    *,
    device: int,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], dict[int, list[np.ndarray]], dict[int, list[dict[str, Any]]]]:
    executor = PersistentKimiStageExecutor(
        request=request,
        checkpoint=checkpoint,
        cuda_library=cuda_library,
        device=device,
    )
    outputs: dict[int, list[np.ndarray]] = {}
    states: dict[int, list[dict[str, Any]]] = {}
    receipt: dict[str, Any] = {
        "resident_device_bytes": executor.resident_device_bytes,
        "tracked_device_bytes": executor.tracked_device_bytes,
        "weight_fingerprint": executor.weight_fingerprint,
        "batches": {},
    }
    try:
        executor.prepare_for_ready()
        executor.set_research_telemetry_mode("minimal")
        for batch in INCREMENTAL_BATCHES:
            correctness_ids = [f"h014-sub-006-ref-b{batch}-c-{row}" for row in range(batch)]
            for session_id in correctness_ids:
                executor.open_session(session_id, maximum_context_override=3)
            batch_outputs: list[np.ndarray] = []
            route_rows: list[list[list[int]]] = []
            try:
                for position in range(3):
                    record = executor.execute_decode_batch(
                        session_ids=tuple(correctness_ids),
                        hidden_states=_batch_boundaries(
                            fixtures, batch=batch, position=position
                        ),
                        cache_position_start=position,
                    )
                    batch_outputs.append(np.asarray(record["boundary_output"]).copy())
                    route_rows.append(record["selected_expert_ids"])
                batch_states = [
                    executor.session_state_evidence(session_id)
                    for session_id in correctness_ids
                ]
            finally:
                _close_sessions(executor, correctness_ids)
            outputs[batch] = batch_outputs
            states[batch] = batch_states

            performance_ids = [
                f"h014-sub-006-ref-b{batch}-p-{row}" for row in range(batch)
            ]
            for session_id in performance_ids:
                executor.open_session(
                    session_id,
                    maximum_context_override=warmup + iterations + 3,
                )
            wall_ms: list[float] = []
            device_ms: list[float] = []
            routing: list[dict[str, Any]] = []
            phase_rows: list[dict[str, Any]] = []
            try:
                for position in range(warmup + iterations):
                    started = time.perf_counter_ns()
                    record = executor.execute_decode_batch(
                        session_ids=tuple(performance_ids),
                        hidden_states=_batch_boundaries(
                            fixtures, batch=batch, position=position
                        ),
                        cache_position_start=position,
                    )
                    if position >= warmup:
                        wall_ms.append((time.perf_counter_ns() - started) / 1e6)
                        device_ms.append(float(record["device_ms"]))
                        routing.append(record["routing"])
                for phase_index in range(3):
                    position = warmup + iterations + phase_index
                    phase_rows.append(
                        executor.execute_decode_batch(
                            session_ids=tuple(performance_ids),
                            hidden_states=_batch_boundaries(
                                fixtures, batch=batch, position=position
                            ),
                            cache_position_start=position,
                            profile_phases=True,
                        )
                    )
            finally:
                _close_sessions(executor, performance_ids)
            receipt["batches"][str(batch)] = {
                "wall": _timing(wall_ms),
                "device": _timing(device_ms),
                "aggregate_wall_rows_per_second": batch * 1000.0 / _timing(wall_ms)["p50_ms"],
                "aggregate_device_rows_per_second": batch
                * 1000.0
                / _timing(device_ms)["p50_ms"],
                "routing": {
                    "mean_unique_experts": float(
                        np.mean([row["unique_experts"] for row in routing])
                    ),
                    "effective_weight_reuse_rows_per_native_call": sum(
                        row["total_selections"] for row in routing
                    )
                    / sum(row["native_routed_expert_calls"] for row in routing),
                },
                "routed_expert_phase_device": _timing(
                    [
                        float(row["phase_device_ms"]["routed_expert_compute"])
                        for row in phase_rows
                    ]
                ),
                "routes": route_rows,
            }
        receipt["prepare"] = executor.prepare_warmup
        receipt["status"] = "PASS"
        return receipt, outputs, states
    finally:
        executor.close()


def _batch_correctness(
    executor: PersistentKimiStageExecutor,
    collective: _PersistentExpertCollective,
    fixtures: list[np.ndarray],
    reference_outputs: list[np.ndarray],
    reference_states: list[dict[str, Any]],
    *,
    batch: int,
    overlap_parent_shared: bool = False,
) -> dict[str, Any]:
    session_ids = [f"h014-sub-006-ext-b{batch}-c-{row}" for row in range(batch)]
    for session_id in session_ids:
        executor.open_session(session_id, maximum_context_override=3)
    comparisons: list[dict[str, Any]] = []
    external_records: list[dict[str, Any]] = []
    try:
        for position in range(3):
            record = executor.execute_decode_batch(
                session_ids=tuple(session_ids),
                hidden_states=_batch_boundaries(
                    fixtures, batch=batch, position=position
                ),
                cache_position_start=position,
                external_expert_dispatch=(
                    collective if overlap_parent_shared else collective.dispatch_batch
                ),
                overlap_parent_shared=overlap_parent_shared,
            )
            external = record["external_expert_collective"]
            external_records.append(external)
            outputs = np.asarray(record["boundary_output"])
            for row in range(batch):
                comparisons.append(
                    {
                        "position": position,
                        "row": row,
                        "metrics": _numerical_metrics(
                            outputs[row], reference_outputs[position][row]
                        ),
                        "output_fingerprint": _array_fingerprint(outputs[row]),
                        "reference_fingerprint": _array_fingerprint(
                            reference_outputs[position][row]
                        ),
                        "all_16_executed_exactly_once": external[
                            "all_selected_experts_executed_once"
                        ],
                    }
                )
        observed_states = [
            executor.session_state_evidence(session_id) for session_id in session_ids
        ]
    finally:
        _close_sessions(executor, session_ids)
    maximum_error = max(
        float(row["metrics"]["relative_l2_error"]) for row in comparisons
    )
    state_equal = all(
        observed["fingerprint"] == expected["fingerprint"]
        for observed, expected in zip(observed_states, reference_states, strict=True)
    )
    passed = (
        maximum_error <= 1e-7
        and state_equal
        and all(bool(row["all_16_executed_exactly_once"]) for row in comparisons)
    )
    return {
        "comparisons": comparisons,
        "maximum_relative_l2_error": maximum_error,
        "state_fingerprint_equality": state_equal,
        "observed_states": observed_states,
        "reference_states": reference_states,
        "external_records": external_records,
        "pass": passed,
    }


def _measure_batch(
    executor: PersistentKimiStageExecutor,
    collective: _PersistentExpertCollective,
    fixtures: list[np.ndarray],
    *,
    batch: int,
    warmup: int,
    iterations: int,
    overlap_parent_shared: bool = False,
) -> dict[str, Any]:
    session_ids = [f"h014-sub-006-ext-b{batch}-p-{row}" for row in range(batch)]
    for session_id in session_ids:
        executor.open_session(
            session_id, maximum_context_override=warmup + iterations + 3
        )
    wall_ms: list[float] = []
    device_ms: list[float] = []
    external_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    before: dict[str, Any] | None = None
    try:
        for position in range(warmup + iterations):
            if position == warmup:
                before = _microwork_process_snapshot()
                before_executor = executor.lifecycle_snapshot()
            started = time.perf_counter_ns()
            record = executor.execute_decode_batch(
                session_ids=tuple(session_ids),
                hidden_states=_batch_boundaries(
                    fixtures, batch=batch, position=position
                ),
                cache_position_start=position,
                external_expert_dispatch=(
                    collective if overlap_parent_shared else collective.dispatch_batch
                ),
                overlap_parent_shared=overlap_parent_shared,
            )
            if position >= warmup:
                wall_ms.append((time.perf_counter_ns() - started) / 1e6)
                device_ms.append(float(record["device_ms"]))
                external_rows.append(record["external_expert_collective"])
        after = _microwork_process_snapshot()
        after_executor = executor.lifecycle_snapshot()
        for phase_index in range(3):
            position = warmup + iterations + phase_index
            phase_rows.append(
                executor.execute_decode_batch(
                    session_ids=tuple(session_ids),
                    hidden_states=_batch_boundaries(
                        fixtures, batch=batch, position=position
                    ),
                    cache_position_start=position,
                    profile_phases=True,
                    external_expert_dispatch=(
                        collective
                        if overlap_parent_shared
                        else collective.dispatch_batch
                    ),
                    overlap_parent_shared=overlap_parent_shared,
                )
            )
        state_bytes_each = executor.kv_cache_bytes(session_ids[0])
    finally:
        _close_sessions(executor, session_ids)
    if before is None:
        raise RuntimeError("sub-layer batch did not enter retained execution")
    lifecycle = _microwork_lifecycle_delta(before, after)
    lifecycle.update(
        {
            "weight_loading": int(after_executor["weight_load_count"])
            - int(before_executor["weight_load_count"]),
            "model_materialization": int(after_executor["model_materialization_count"])
            - int(before_executor["model_materialization_count"]),
            "persistent_buffer_allocation": int(
                after_executor["persistent_buffer_allocation_count"]
            )
            - int(before_executor["persistent_buffer_allocation_count"]),
        }
    )
    wall = _timing(wall_ms)
    device = _timing(device_ms)
    total_selections = sum(int(row["total_selections"]) for row in external_rows)
    native_calls = sum(int(row["native_expert_calls"]) for row in external_rows)
    worker_task_counts = [
        int(worker["tasks"])
        for row in external_rows
        for worker in row["worker_records"]
    ]
    worker_activation_rows = [
        int(worker["activation_rows"])
        for row in external_rows
        for worker in row["worker_records"]
    ]

    def critical_worker_phase(name: str) -> dict[str, Any]:
        return _timing(
            [
                max(float(worker[name]) for worker in row["worker_records"])
                for row in external_rows
            ]
        )

    if overlap_parent_shared:
        external_phase_values = [
            sum(
                float(row["phase_wall_ms"][name])
                for name in (
                    "external_expert_start",
                    "shared_expert_overlap",
                    "external_expert_collect",
                )
            )
            for row in phase_rows
        ]
        overlap_evidence: dict[str, Any] = {
            "parent_overlap_window": _timing(
                [float(row["parent_overlap_window_ms"]) for row in external_rows]
            ),
            "collection_wait": _timing(
                [float(row["collection_ms"]) for row in external_rows]
            ),
            "shared_expert_overlap_wall": _timing(
                [
                    float(row["phase_wall_ms"]["shared_expert_overlap"])
                    for row in phase_rows
                ]
            ),
        }
    else:
        external_phase_values = [
            float(row["phase_wall_ms"]["external_expert_collective"])
            for row in phase_rows
        ]
        overlap_evidence = {"enabled": False}

    return {
        "batch": batch,
        "wall": wall,
        "device": device,
        "aggregate_wall_rows_per_second": batch * 1000.0 / wall["p50_ms"],
        "aggregate_device_rows_per_second": batch * 1000.0 / device["p50_ms"],
        "per_row_wall_service_ms": wall["p50_ms"] / batch,
        "expert_collective_roundtrip": _timing(
            [float(row["roundtrip_ms"]) for row in external_rows]
        ),
        "critical_worker_device": _timing(
            [float(row["worker_compute_critical_device_ms"]) for row in external_rows]
        ),
        "worker_phase_decomposition": {
            "critical_activation_h2d_enqueue_wall": critical_worker_phase(
                "activation_h2d_enqueue_wall_ms"
            ),
            "critical_input_gather_and_expert_device": critical_worker_phase(
                "input_gather_and_expert_device_ms"
            ),
            "critical_output_d2h_wall": critical_worker_phase(
                "output_d2h_wall_ms"
            ),
        },
        "exposed_coordination": _timing(
            [float(row["exposed_loopback_and_coordination_ms"]) for row in external_rows]
        ),
        "external_phase_wall": _timing(external_phase_values),
        "parent_shared_overlap": overlap_evidence,
        "routing_and_reuse": {
            "total_selections": total_selections,
            "mean_unique_experts": float(
                np.mean([int(row["unique_experts"]) for row in external_rows])
            ),
            "mean_repeated_expert_hits": float(
                np.mean([int(row["repeated_expert_hits"]) for row in external_rows])
            ),
            "native_expert_calls": native_calls,
            "effective_weight_reuse_rows_per_native_call": total_selections
            / native_calls,
            "avoided_expert_weight_launches": total_selections - native_calls,
        },
        "communication": {
            "mean_total_transport_bytes": float(
                np.mean([int(row["total_transport_bytes"]) for row in external_rows])
            ),
            "mean_transport_bytes_per_row": float(
                np.mean([int(row["total_transport_bytes"]) for row in external_rows])
            )
            / batch,
            "mean_messages": float(
                np.mean([int(row["messages"]) for row in external_rows])
            ),
            "mean_messages_per_row": float(
                np.mean([int(row["messages"]) for row in external_rows])
            )
            / batch,
            "mean_critical_path_payload_bytes": float(
                np.mean([int(row["critical_path_payload_bytes"]) for row in external_rows])
            ),
            "returned_expert_output_bytes": batch * 16 * executor.config.latent * 4,
            "mean_input_activation_bytes": float(
                np.mean([int(row["input_activation_bytes"]) for row in external_rows])
            ),
            "synchronization_points": 2,
        },
        "load_balance": {
            "worker_tasks_min": min(worker_task_counts),
            "worker_tasks_max": max(worker_task_counts),
            "worker_tasks_mean": float(np.mean(worker_task_counts)),
            "worker_activation_rows_min": min(worker_activation_rows),
            "worker_activation_rows_max": max(worker_activation_rows),
            "worker_activation_rows_mean": float(np.mean(worker_activation_rows)),
        },
        "state_bytes_each": state_bytes_each,
        "state_bytes_total": state_bytes_each * batch,
        "lifecycle_delta": lifecycle,
        "lifecycle_deltas_zero": all(int(value) == 0 for value in lifecycle.values()),
    }


def _safe_fixture(
    executor: PersistentKimiStageExecutor,
    collective: _PersistentExpertCollective,
    fixture: np.ndarray,
    reference: np.ndarray,
    *,
    batch: int,
    overlap_parent_shared: bool = False,
) -> dict[str, Any]:
    session_id = f"h014-sub-006-post-b{batch}-safe"
    executor.open_session(session_id, maximum_context_override=3)
    try:
        record = executor.execute_decode_batch(
            session_ids=(session_id,),
            hidden_states=torch.from_numpy(fixture.copy()),
            cache_position_start=0,
            external_expert_dispatch=(
                collective if overlap_parent_shared else collective.dispatch_batch
            ),
            overlap_parent_shared=overlap_parent_shared,
        )
        output = np.asarray(record["boundary_output"])[0]
        metrics = _numerical_metrics(output, reference[0])
    finally:
        executor.close_session(session_id)
    executor.runtime.synchronize()
    worker_health = collective.health()
    health = _health_snapshot(executor.runtime.device)
    passed = (
        float(metrics["relative_l2_error"]) <= 1e-7
        and executor.runtime.error_state_ok()
        and all(bool(row["cuda_error_state_ok"]) for row in worker_health)
        and health["status"] == "MEASURED"
    )
    return {
        "metrics": metrics,
        "worker_health": worker_health,
        "coordinator_cuda_error_state_ok": executor.runtime.error_state_ok(),
        "free_vram_bytes": executor.runtime.mem_info()["free_bytes"],
        "nvidia_smi": health,
        "pass": passed,
    }


def benchmark_sub_layer_batch(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    graph_certification: Path,
    output_path: Path,
    *,
    layer: int = 89,
    workers: int = 4,
    device: int = 0,
    warmup: int = 5,
    iterations: int = 20,
    cycle_id: str = "H014-SUB-006",
    overlap_parent_shared: bool = False,
) -> dict[str, Any]:
    """Run real four-worker expert batching incrementally through batch eight."""
    if layer != 89 or workers != 4:
        raise ValueError("H014-SUB-006 is fixed to real layer 89 and four workers")
    if warmup < 5 or iterations < 20:
        raise ValueError("sub-layer batch requires >=5/20 warm/retained calls")
    paths = {
        "checkpoint": checkpoint.resolve(),
        "cuda_library": cuda_library.resolve(),
        "oracle_trace": oracle_trace.resolve(),
        "graph_certification": graph_certification.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    graph = json.loads(paths["graph_certification"].read_text(encoding="utf-8"))
    graph_fixture = graph.get("fixture", {})
    provenance = {
        "graph_status": graph.get("status"),
        "trace_matches": graph_fixture.get("oracle_trace_sha256")
        == _sha256_file(paths["oracle_trace"]),
    }
    provenance["pass"] = provenance["graph_status"] == "PASS" and provenance[
        "trace_matches"
    ]
    if not provenance["pass"]:
        raise ValueError("sub-layer batch trace is not joined to the passing graph")

    buffered_worker = cycle_id.upper().endswith(("006D", "006E", "006F"))
    single_interval_worker = cycle_id.upper().endswith("006F")
    if overlap_parent_shared:
        hypothesis_payload: dict[str, Any] = {
            "prediction": (
                "Persistent start/collect overlaps the parent-resident shared expert "
                "with routed workers while preserving exact batches 1/2/4/8; batch-8 "
                "wall p50 is <=16.74745 ms and p99 is <=19.02525 ms."
            ),
            "maximum_batch8_wall_p50_ms": 16.74745,
            "maximum_batch8_wall_p99_ms": H014_SUB_006F_BATCH8_WALL_P99_MS,
            "minimum_shared_wall_hidden_percent": 50.0,
            "fixed_h014_sub_006f_batch8_wall_p50_ms": H014_SUB_006F_BATCH8_WALL_P50_MS,
            "fixed_shared_batch8_wall_p50_ms": H014_SUB_030E_SHARED_BATCH8_WALL_P50_MS,
        }
    elif buffered_worker:
        hypothesis_payload = {
            "prediction": (
                "One worker-level activation upload, device gather, grouped expert "
                "execution and one output download preserve exact batches 1/2/4/8, "
                "improve batch-8 complete-layer capacity >=1.20x versus H014-SUB-006c, "
                "retain >=80% resident throughput and do not increase tensor payload or messages."
            ),
            "minimum_capacity_gain_vs_h014_sub_006c": 1.20,
            "minimum_resident_batch_throughput_retention_percent": 80.0,
            "maximum_messages_per_row": H014_SUB_006C_MESSAGES_PER_ROW,
            "maximum_tensor_bytes_per_row": H014_SUB_006C_TENSOR_BYTES_PER_ROW,
            "fixed_h014_sub_006c_batch8_wall_p50_ms": H014_SUB_006C_BATCH8_WALL_P50_MS,
            "maximum_batch8_wall_p99_ms": (
                H014_SUB_006C_BATCH8_WALL_P99_MS
                if single_interval_worker
                else None
            ),
        }
    else:
        hypothesis_payload = {
            "prediction": (
                "Four persistent expert workers execute batches 1/2/4/8 exactly; "
                "batch 8 obtains >=2.0 rows/native call, >=75% messages/row reduction, "
                ">=50% resident-batch throughput and >=1.5x capacity vs serial distributed rows."
            ),
            "minimum_effective_weight_reuse": 2.0,
            "minimum_message_reduction_percent": 75.0,
            "minimum_resident_batch_throughput_retention_percent": 50.0,
            "minimum_capacity_gain_vs_distributed_batch1": 1.5,
        }
    receipt: dict[str, Any] = {
        "schema_version": (
            OVERLAP_SCHEMA_VERSION
            if overlap_parent_shared
            else BUFFERED_SCHEMA_VERSION
            if buffered_worker
            else SCHEMA_VERSION
        ),
        "cycle_id": cycle_id,
        "status": "RUNNING",
        "hypothesis": hypothesis_payload,
        "configuration": {
            "layer": layer,
            "workers": workers,
            "batches": list(INCREMENTAL_BATCHES),
            "warmup_calls": warmup,
            "retained_calls": iterations,
            "ownership": "expert_id modulo 4; 224 experts/worker",
            "shared_expert_placement": "parent coordinator",
            "overlap_parent_shared": overlap_parent_shared,
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "provenance": provenance,
        "progress": [],
        "batches": {},
    }

    def retain(phase: str, **evidence: Any) -> None:
        receipt["progress"].append({"phase": phase, **evidence})
        _atomic_json(output_path, receipt)

    proxies: list[_WorkerProxy] = []
    coordinator: PersistentKimiStageExecutor | None = None
    retain("preregistered")
    receipt["gpu_health_before"] = _health_snapshot(device)
    receipt["device_identity"] = _device_identity(device)
    retain("gpu_health_before", status=receipt["gpu_health_before"]["status"])
    if receipt["gpu_health_before"]["status"] != "MEASURED":
        raise RuntimeError("nvidia-smi unavailable before sub-layer batching")
    request = _request(
        paths["checkpoint"],
        paths["cuda_library"],
        layer=layer,
        device=device,
        cycle_id=cycle_id,
        maximum_context=warmup + iterations + 8,
    )
    fixtures, _ = _stage_fixtures(paths["checkpoint"], paths["oracle_trace"], layer=layer)
    try:
        reference, reference_outputs, reference_states = _reference_batches(
            request,
            paths["checkpoint"],
            paths["cuda_library"],
            fixtures,
            device=device,
            warmup=warmup,
            iterations=iterations,
        )
        receipt["resident_reference"] = reference
        retain(
            "resident_reference_released",
            resident_device_bytes=reference["resident_device_bytes"],
            gpu_health=_health_snapshot(device)["status"],
        )

        context = mp.get_context("spawn")
        ownership_sets = [
            frozenset(expert for expert in range(896) if expert % workers == worker_id)
            for worker_id in range(workers)
        ]
        for worker_id, ownership in enumerate(ownership_sets):
            proxy = _start_worker(
                context,
                checkpoint=paths["checkpoint"],
                cuda_library=paths["cuda_library"],
                layer=layer,
                device=device,
                worker_id=worker_id,
                ownership=ownership,
                batch_capacity=8,
            )
            proxies.append(proxy)
            retain(
                f"worker_{worker_id}_ready",
                tracked_worker_bytes=proxy.ready["tracked_worker_bytes"],
            )
        collective = _PersistentExpertCollective(proxies, latent=3584)
        coordinator = PersistentKimiStageExecutor(
            request=request,
            checkpoint=paths["checkpoint"],
            cuda_library=paths["cuda_library"],
            device=device,
            owned_expert_ids=frozenset(),
        )
        receipt["distributed_residency"] = {
            "coordinator_resident_device_bytes": coordinator.resident_device_bytes,
            "workers": [
                {
                    "worker_id": proxy.worker_id,
                    "owned_experts": len(proxy.ownership),
                    "tracked_worker_bytes": proxy.ready["tracked_worker_bytes"],
                    "fraction_of_complete_layer_percent": 100.0
                    * int(proxy.ready["tracked_worker_bytes"])
                    / int(reference["resident_device_bytes"]),
                    "batch_capacity": proxy.ready["batch_capacity"],
                }
                for proxy in proxies
            ],
            "all_workers_less_than_complete_layer": all(
                int(proxy.ready["tracked_worker_bytes"])
                < int(reference["resident_device_bytes"])
                for proxy in proxies
            ),
        }
        retain("coordinator_and_collective_ready")

        for batch in INCREMENTAL_BATCHES:
            retain(
                f"armed_batch_{batch}",
                prior_batches=[
                    size
                    for size in INCREMENTAL_BATCHES
                    if size < batch
                    and receipt["batches"].get(str(size), {}).get("status") == "PASS"
                ],
            )
            correctness = _batch_correctness(
                coordinator,
                collective,
                fixtures,
                reference_outputs[batch],
                reference_states[batch],
                batch=batch,
                overlap_parent_shared=overlap_parent_shared,
            )
            if not correctness["pass"]:
                receipt["batches"][str(batch)] = {
                    "correctness": correctness,
                    "status": "FAIL",
                }
                retain(f"batch_{batch}_correctness_failed")
                break
            receipt["batches"][str(batch)] = {
                "correctness": correctness,
                "status": "CORRECTNESS_PASS_PERFORMANCE_PENDING",
            }
            retain(
                f"batch_{batch}_correctness_persisted",
                maximum_relative_l2_error=correctness["maximum_relative_l2_error"],
                state_fingerprint_equality=correctness[
                    "state_fingerprint_equality"
                ],
            )
            performance = _measure_batch(
                coordinator,
                collective,
                fixtures,
                batch=batch,
                warmup=warmup,
                iterations=iterations,
                overlap_parent_shared=overlap_parent_shared,
            )
            safe = _safe_fixture(
                coordinator,
                collective,
                fixtures[0],
                reference_outputs[1][0],
                batch=batch,
                overlap_parent_shared=overlap_parent_shared,
            )
            row = {
                "correctness": correctness,
                "performance": performance,
                "resident_reference": reference["batches"][str(batch)],
                "post_batch_safe_fixture": safe,
            }
            row["status"] = (
                "PASS"
                if correctness["pass"]
                and performance["lifecycle_deltas_zero"]
                and safe["pass"]
                else "FAIL"
            )
            receipt["batches"][str(batch)] = row
            retain(
                f"batch_{batch}_persisted_and_checked",
                status=row["status"],
                wall_p50_ms=performance["wall"]["p50_ms"],
                nvidia_smi=safe["nvidia_smi"]["status"],
            )
            if row["status"] != "PASS":
                break

        execution_pass = len(receipt["batches"]) == len(INCREMENTAL_BATCHES) and all(
            row["status"] == "PASS" for row in receipt["batches"].values()
        )
        if execution_pass:
            batch1 = receipt["batches"]["1"]["performance"]
            batch8 = receipt["batches"]["8"]["performance"]
            reference8 = receipt["batches"]["8"]["resident_reference"]
            throughput_retention = 100.0 * (
                batch8["aggregate_wall_rows_per_second"]
                / reference8["aggregate_wall_rows_per_second"]
            )
            if overlap_parent_shared:
                saved_ms = H014_SUB_006F_BATCH8_WALL_P50_MS - batch8["wall"][
                    "p50_ms"
                ]
                shared_hidden_percent = 100.0 * saved_ms / (
                    H014_SUB_030E_SHARED_BATCH8_WALL_P50_MS
                )
                gates = {
                    "batch8_wall_p50_at_most_16_74745_ms": batch8["wall"][
                        "p50_ms"
                    ]
                    <= 16.74745,
                    "batch8_wall_p99_not_above_h014_sub_006f": batch8["wall"][
                        "p99_ms"
                    ]
                    <= H014_SUB_006F_BATCH8_WALL_P99_MS,
                    "shared_wall_hidden_at_least_50_percent": shared_hidden_percent
                    >= 50.0,
                    "messages_per_row_unchanged": batch8["communication"][
                        "mean_messages_per_row"
                    ]
                    <= H014_SUB_006C_MESSAGES_PER_ROW,
                }
                receipt["batch8_evaluation"] = {
                    "wall_p50_ms": batch8["wall"]["p50_ms"],
                    "wall_p99_ms": batch8["wall"]["p99_ms"],
                    "saved_vs_h014_sub_006f_p50_ms": saved_ms,
                    "shared_wall_hidden_percent": shared_hidden_percent,
                    "resident_batch_throughput_retention_percent": throughput_retention,
                    "parent_overlap_window_p50_ms": batch8[
                        "parent_shared_overlap"
                    ]["parent_overlap_window"]["p50_ms"],
                    "collection_wait_p50_ms": batch8["parent_shared_overlap"][
                        "collection_wait"
                    ]["p50_ms"],
                    "gates": gates,
                }
            elif buffered_worker:
                capacity_gain = H014_SUB_006C_BATCH8_WALL_P50_MS / batch8["wall"][
                    "p50_ms"
                ]
                tensor_bytes_per_row = (
                    batch8["communication"]["mean_input_activation_bytes"]
                    + batch8["communication"]["returned_expert_output_bytes"]
                ) / 8.0
                gates = {
                    "capacity_gain_vs_h014_sub_006c_at_least_1_20": capacity_gain
                    >= 1.20,
                    "resident_batch_throughput_retention_at_least_80_percent": throughput_retention
                    >= 80.0,
                    "messages_per_row_not_increased": batch8["communication"][
                        "mean_messages_per_row"
                    ]
                    <= H014_SUB_006C_MESSAGES_PER_ROW,
                    "tensor_payload_bytes_per_row_not_increased": tensor_bytes_per_row
                    <= H014_SUB_006C_TENSOR_BYTES_PER_ROW,
                }
                if single_interval_worker:
                    gates["batch8_wall_p99_not_above_h014_sub_006c"] = batch8[
                        "wall"
                    ]["p99_ms"] <= H014_SUB_006C_BATCH8_WALL_P99_MS
                receipt["batch8_evaluation"] = {
                    "capacity_gain_vs_h014_sub_006c": capacity_gain,
                    "resident_batch_throughput_retention_percent": throughput_retention,
                    "messages_per_row": batch8["communication"][
                        "mean_messages_per_row"
                    ],
                    "tensor_payload_bytes_per_row": tensor_bytes_per_row,
                    "batch8_wall_p99_ms": batch8["wall"]["p99_ms"],
                    "gates": gates,
                }
            else:
                reuse = batch8["routing_and_reuse"][
                    "effective_weight_reuse_rows_per_native_call"
                ]
                message_reduction = 100.0 * (
                    1.0
                    - batch8["communication"]["mean_messages_per_row"]
                    / batch1["communication"]["mean_messages_per_row"]
                )
                capacity_gain = (
                    8.0 * batch1["wall"]["p50_ms"] / batch8["wall"]["p50_ms"]
                )
                gates = {
                    "effective_weight_reuse_at_least_2": reuse >= 2.0,
                    "message_reduction_at_least_75_percent": message_reduction >= 75.0,
                    "resident_batch_throughput_retention_at_least_50_percent": throughput_retention
                    >= 50.0,
                    "capacity_gain_vs_distributed_batch1_at_least_1_5": capacity_gain
                    >= 1.5,
                }
                receipt["batch8_evaluation"] = {
                    "effective_weight_reuse_rows_per_native_call": reuse,
                    "message_reduction_percent_vs_batch1_per_row": message_reduction,
                    "resident_batch_throughput_retention_percent": throughput_retention,
                    "capacity_gain_vs_distributed_batch1": capacity_gain,
                    "gates": gates,
                }
            hypothesis_supported = all(gates.values())
        else:
            hypothesis_supported = False
        receipt["execution_pass"] = execution_pass
        receipt["hypothesis_supported"] = hypothesis_supported
        receipt["inspection"] = {
            "actual_bottleneck": "pending numerical inspection",
            "fixture_limitation": (
                "Three retained real boundaries are row-rotated; route overlap is real "
                "for this trace but not a production prompt distribution."
            ),
        }
        receipt["decision"] = {
            "sub_layer_batch": "RETAIN" if hypothesis_supported else "CHARACTERIZED_NOT_RETAINED",
            "next_hypothesis": "pending numerical inspection",
        }
        receipt["gpu_health_after"] = _health_snapshot(device)
        receipt["status"] = (
            "PASS"
            if execution_pass and receipt["gpu_health_after"]["status"] == "MEASURED"
            else "FAIL"
        )
        retain("complete", status=receipt["status"])
        return receipt
    except BaseException as exc:
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        with suppress(Exception):
            receipt["gpu_health_after_failure"] = _health_snapshot(device)
        _atomic_json(output_path, receipt)
        return receipt
    finally:
        if coordinator is not None:
            coordinator.close()
        if proxies:
            receipt["worker_shutdown"] = [
                proxy.close() for proxy in reversed(proxies)
            ]
            _atomic_json(output_path, receipt)
