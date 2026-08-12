"""Incremental complete-stage Kimi batching for H014-029."""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from collections import Counter
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swarm_inference.execution.kimi_k3_stage import (
    PersistentKimiStageExecutor,
    _process_snapshot,
    _timing,
    _warm_lifecycle_delta,
)
from swarm_inference.experiments.experiment_014.batch_safety import _health_snapshot
from swarm_inference.experiments.experiment_014.cuda import (
    _array_fingerprint,
    _device_identity,
    _numerical_metrics,
    _sha256_file,
)
from swarm_inference.experiments.experiment_014.full_cuda import _parse_oracle_routes
from swarm_inference.experiments.experiment_014.persistent_stages import (
    MODEL_CONTENT_FINGERPRINT,
    MODEL_REVISION,
    TOKENIZER_REVISION,
    _CaptureConnectionPool,
    _resolve_worker_identity_manifest,
    _source_assignment,
    _stage_fixtures,
)
from swarm_inference.protocol.stage_worker import LoadStageRequest
from swarm_inference.worker.stage_runtime import PersistentStageRuntime

SCHEMA_VERSION = "experiment-014-k3-complete-stage-static-batch-v1"
INCREMENTAL_BATCHES = (1, 2, 4, 8, 16)
CORRECTNESS_RELATIVE_L2_GATE = 1e-7
MINIMUM_MATERIAL_THROUGHPUT_GAIN = 1.5


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _batch_boundaries(
    fixtures: list[np.ndarray], *, batch: int, position: int
) -> torch.Tensor:
    values = np.concatenate(
        [fixtures[(row + position) % len(fixtures)] for row in range(batch)], axis=0
    )
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


def _close_sessions(executor: PersistentKimiStageExecutor, session_ids: list[str]) -> None:
    for session_id in session_ids:
        with suppress(KeyError):
            executor.close_session(session_id)


def _serial_oracle_and_baseline(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    expected_boundaries: list[np.ndarray],
    expected_routes: dict[int, list[int]],
    *,
    layer: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    correctness_session = f"h014-029-layer-{layer}-serial-oracle"
    executor.open_session(correctness_session, maximum_context_override=3)
    correctness: list[dict[str, Any]] = []
    try:
        for position in range(3):
            result = executor.execute_decode(
                session_id=correctness_session,
                hidden_states=torch.from_numpy(fixtures[position].copy()),
                cache_position_start=position,
            )
            record = executor.execution_records[-1]
            output = result.stage_boundary_hidden_states.detach().cpu().numpy().copy()
            metrics = _numerical_metrics(output, expected_boundaries[position])
            routes = list(record["selected_expert_ids"])
            correctness.append(
                {
                    "position": position,
                    "metrics": metrics,
                    "routing_equality": routes == expected_routes[position],
                    "selected_expert_ids": routes,
                    "expected_expert_ids": expected_routes[position],
                    "output_fingerprint": _array_fingerprint(output),
                    "expected_output_fingerprint": _array_fingerprint(
                        expected_boundaries[position]
                    ),
                }
            )
    finally:
        executor.close_session(correctness_session)

    baseline_session = f"h014-029-layer-{layer}-serial-baseline"
    executor.open_session(
        baseline_session, maximum_context_override=warmup + iterations
    )
    wall_ms: list[float] = []
    device_ms: list[float] = []
    before_warm: dict[str, Any] | None = None
    try:
        for position in range(warmup + iterations):
            if position == warmup:
                before_warm = executor.lifecycle_snapshot()
            started = time.perf_counter_ns()
            executor.execute_decode(
                session_id=baseline_session,
                hidden_states=torch.from_numpy(
                    fixtures[position % len(fixtures)].copy()
                ),
                cache_position_start=position,
            )
            call_wall_ms = (time.perf_counter_ns() - started) / 1e6
            if position >= warmup:
                record = executor.execution_records[-1]
                wall_ms.append(call_wall_ms)
                device_ms.append(float(record["device_ms"]))
        after_warm = executor.lifecycle_snapshot()
        state = executor.session_state_evidence(baseline_session)
    finally:
        executor.close_session(baseline_session)
    if before_warm is None:
        raise RuntimeError("serial baseline did not reach retained execution")
    maximum_error = max(
        float(row["metrics"]["relative_l2_error"]) for row in correctness
    )
    oracle_pass = maximum_error <= 3e-5 and all(
        bool(row["routing_equality"]) for row in correctness
    )
    lifecycle_delta = {
        key: int(after_warm[key]) - int(before_warm[key])
        for key in (
            "weight_load_count",
            "model_materialization_count",
            "persistent_buffer_allocation_count",
        )
    }
    return {
        "oracle_correctness": {
            "positions": correctness,
            "maximum_relative_l2_error": maximum_error,
            "pass": oracle_pass,
        },
        "wall": _timing(wall_ms),
        "device": _timing(device_ms),
        "rows_per_second_at_device_p50": 1000.0 / _timing(device_ms)["p50_ms"],
        "retained_iterations": iterations,
        "warmup_iterations": warmup,
        "state": state,
        "lifecycle_delta": lifecycle_delta,
        "lifecycle_deltas_zero": all(value == 0 for value in lifecycle_delta.values()),
        "status": "PASS"
        if oracle_pass and all(value == 0 for value in lifecycle_delta.values())
        else "FAIL",
    }


def _batch_serial_equivalence(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    *,
    layer: int,
    batch: int,
) -> dict[str, Any]:
    serial_ids = [f"h014-029-l{layer}-b{batch}-serial-{row}" for row in range(batch)]
    batch_ids = [f"h014-029-l{layer}-b{batch}-batch-{row}" for row in range(batch)]
    serial_outputs: list[list[np.ndarray]] = []
    serial_routes: list[list[list[int]]] = []
    serial_states: list[dict[str, Any]] = []
    for session_id in serial_ids:
        executor.open_session(session_id, maximum_context_override=3)
    try:
        for position in range(3):
            outputs_at_position: list[np.ndarray] = []
            routes_at_position: list[list[int]] = []
            for row, session_id in enumerate(serial_ids):
                result = executor.execute_decode(
                    session_id=session_id,
                    hidden_states=torch.from_numpy(
                        fixtures[(row + position) % len(fixtures)].copy()
                    ),
                    cache_position_start=position,
                )
                outputs_at_position.append(
                    result.stage_boundary_hidden_states.detach().cpu().numpy()[0].copy()
                )
                routes_at_position.append(
                    list(executor.execution_records[-1]["selected_expert_ids"])
                )
            serial_outputs.append(outputs_at_position)
            serial_routes.append(routes_at_position)
        serial_states = [executor.session_state_evidence(item) for item in serial_ids]
    finally:
        _close_sessions(executor, serial_ids)

    for session_id in batch_ids:
        executor.open_session(session_id, maximum_context_override=3)
    comparisons: list[dict[str, Any]] = []
    every_selected_expert_executed_once = True
    try:
        for position in range(3):
            record = executor.execute_decode_batch(
                session_ids=tuple(batch_ids),
                hidden_states=_batch_boundaries(
                    fixtures, batch=batch, position=position
                ),
                cache_position_start=position,
            )
            every_selected_expert_executed_once = (
                every_selected_expert_executed_once
                and bool(record["routing"]["all_selected_experts_executed_once"])
                and all(
                    len(route) == executor.config.topk
                    and len(set(route)) == executor.config.topk
                    for route in record["selected_expert_ids"]
                )
            )
            for row in range(batch):
                observed = np.ascontiguousarray(record["boundary_output"][row])
                expected = serial_outputs[position][row]
                metrics = _numerical_metrics(observed, expected)
                comparisons.append(
                    {
                        "position": position,
                        "row": row,
                        "metrics": metrics,
                        "routing_equality": record["selected_expert_ids"][row]
                        == serial_routes[position][row],
                        "observed_fingerprint": _array_fingerprint(observed),
                        "serial_fingerprint": _array_fingerprint(expected),
                    }
                )
        batch_states = [executor.session_state_evidence(item) for item in batch_ids]
    finally:
        _close_sessions(executor, batch_ids)

    maximum_error = max(
        float(row["metrics"]["relative_l2_error"]) for row in comparisons
    )
    routes_equal = all(bool(row["routing_equality"]) for row in comparisons)
    states_equal = all(
        observed["fingerprint"] == expected["fingerprint"]
        and observed["bytes"] == expected["bytes"]
        and observed["cache_sequence_length"] == expected["cache_sequence_length"]
        for observed, expected in zip(batch_states, serial_states, strict=True)
    )
    return {
        "positions": 3,
        "rows": batch,
        "comparisons": comparisons,
        "maximum_relative_l2_error": maximum_error,
        "routing_equality": routes_equal,
        "state_fingerprint_equality": states_equal,
        "all_16_experts_executed_once": every_selected_expert_executed_once,
        "pass": maximum_error <= CORRECTNESS_RELATIVE_L2_GATE
        and routes_equal
        and states_equal
        and every_selected_expert_executed_once,
    }


def _phase_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({name for record in records for name in record["phase_device_ms"]})
    return {
        name: {
            "device": _timing(
                [float(record["phase_device_ms"][name]) for record in records]
            ),
            "wall": _timing(
                [float(record["phase_wall_ms"][name]) for record in records]
            ),
        }
        for name in names
    }


def _measure_batch(
    executor: PersistentKimiStageExecutor,
    runtime: PersistentStageRuntime,
    fixtures: list[np.ndarray],
    serial_baseline: dict[str, Any],
    *,
    layer: int,
    batch: int,
    warmup: int,
    iterations: int,
    phase_iterations: int,
) -> dict[str, Any]:
    session_ids = [f"h014-029-l{layer}-b{batch}-perf-{row}" for row in range(batch)]
    memory_before_open = executor.runtime.mem_info()
    for session_id in session_ids:
        executor.open_session(
            session_id,
            maximum_context_override=warmup + iterations + phase_iterations,
        )
    memory_after_open = executor.runtime.mem_info()
    wall_ms: list[float] = []
    device_ms: list[float] = []
    retained_routing: list[dict[str, Any]] = []
    phase_records: list[dict[str, Any]] = []
    before_warm: dict[str, Any] | None = None
    try:
        for position in range(warmup + iterations):
            if position == warmup:
                before_warm = _process_snapshot(runtime, executor)
            started = time.perf_counter_ns()
            record = executor.execute_decode_batch(
                session_ids=tuple(session_ids),
                hidden_states=_batch_boundaries(
                    fixtures, batch=batch, position=position
                ),
                cache_position_start=position,
            )
            call_wall_ms = (time.perf_counter_ns() - started) / 1e6
            if position >= warmup:
                wall_ms.append(call_wall_ms)
                device_ms.append(float(record["device_ms"]))
                retained_routing.append(record["routing"])
        after_retained = _process_snapshot(runtime, executor)
        for phase_index in range(phase_iterations):
            position = warmup + iterations + phase_index
            phase_records.append(
                executor.execute_decode_batch(
                    session_ids=tuple(session_ids),
                    hidden_states=_batch_boundaries(
                        fixtures, batch=batch, position=position
                    ),
                    cache_position_start=position,
                    profile_phases=True,
                )
            )
        after_phase_profile = _process_snapshot(runtime, executor)
        states = [executor.session_state_evidence(item) for item in session_ids]
        after_state_inspection = _process_snapshot(runtime, executor)
        after_warm = after_state_inspection
        state_bytes_each = executor.kv_cache_bytes(session_ids[0])
        memory_after_run = executor.runtime.mem_info()
    finally:
        _close_sessions(executor, session_ids)
    memory_after_close = executor.runtime.mem_info()
    if before_warm is None:
        raise RuntimeError("complete-stage batch did not reach its retained window")

    device = _timing(device_ms)
    wall = _timing(wall_ms)
    serial_device_p50 = float(serial_baseline["device"]["p50_ms"])
    aggregate_rows_per_second = batch * 1000.0 / float(device["p50_ms"])
    throughput_gain = float(device["p50_ms"])
    throughput_gain = batch * serial_device_p50 / throughput_gain
    total_selections = sum(int(row["total_selections"]) for row in retained_routing)
    native_calls = sum(
        int(row["native_routed_expert_calls"]) for row in retained_routing
    )
    unique_experts = [int(row["unique_experts"]) for row in retained_routing]
    repeated_hits = sum(int(row["repeated_expert_hits"]) for row in retained_routing)
    maximum_group = max(
        int(row["maximum_rows_for_one_expert"]) for row in retained_routing
    )
    group_histogram: Counter[str] = Counter()
    for row in retained_routing:
        group_histogram.update(row["expert_group_size_histogram"])
    lifecycle_delta = _warm_lifecycle_delta(before_warm, after_warm)
    lifecycle_zero = all(value == 0 for value in lifecycle_delta.values())
    lifecycle_phases = {
        "retained_execution": _warm_lifecycle_delta(before_warm, after_retained),
        "phase_profile": _warm_lifecycle_delta(
            after_retained, after_phase_profile
        ),
        "state_inspection": _warm_lifecycle_delta(
            after_phase_profile, after_state_inspection
        ),
    }
    return {
        "batch": batch,
        "warmup_calls": warmup,
        "retained_calls": iterations,
        "wall": wall,
        "device": device,
        "aggregate_rows_per_second": aggregate_rows_per_second,
        "per_row_device_service_ms": float(device["p50_ms"]) / batch,
        "throughput_gain_vs_serial_batch1": throughput_gain,
        "ideal_parallel_capacity_retention_percent": 100.0 * throughput_gain / batch,
        "routing": {
            "real_route_source": "three retained stateful Kimi decode trace positions",
            "total_selections": total_selections,
            "mean_unique_experts_per_batch": float(np.mean(unique_experts)),
            "mean_pairwise_overlap_fraction": repeated_hits / total_selections,
            "maximum_rows_for_one_expert": maximum_group,
            "native_routed_expert_calls": native_calls,
            "effective_weight_reuse_rows_per_native_call": total_selections
            / native_calls,
            "avoided_routed_expert_weight_launches": total_selections - native_calls,
            "expert_group_size_histogram": dict(group_histogram),
        },
        "phase_decomposition": _phase_summary(phase_records),
        "phase_profile_calls": phase_iterations,
        "memory": {
            "resident_stage_bytes": executor.resident_device_bytes,
            "persistent_batch_workspace_bytes": executor.lifecycle_snapshot()[
                "batch_workspace_bytes"
            ],
            "session_state_bytes_each": state_bytes_each,
            "session_allocation_free_delta_bytes": max(
                0,
                int(memory_before_open["free_bytes"])
                - int(memory_after_open["free_bytes"]),
            ),
            "free_bytes_before_open": memory_before_open["free_bytes"],
            "free_bytes_after_open": memory_after_open["free_bytes"],
            "free_bytes_after_run": memory_after_run["free_bytes"],
            "free_bytes_after_close": memory_after_close["free_bytes"],
            "state_evidence": states,
        },
        "lifecycle_delta": lifecycle_delta,
        "lifecycle_phase_deltas": lifecycle_phases,
        "lifecycle_thread_ids": {
            "before_retained": before_warm["os_thread_ids"],
            "after_retained": after_retained["os_thread_ids"],
            "after_phase_profile": after_phase_profile["os_thread_ids"],
            "after_state_inspection": after_state_inspection["os_thread_ids"],
        },
        "lifecycle_deltas_zero": lifecycle_zero,
        "status": "PASS" if lifecycle_zero else "FAIL",
    }


def _known_safe_fixture(
    executor: PersistentKimiStageExecutor,
    fixture: np.ndarray,
    *,
    layer: int,
    batch: int,
) -> dict[str, Any]:
    session_id = f"h014-029-l{layer}-post-b{batch}-safe"
    executor.open_session(session_id, maximum_context_override=3)
    try:
        result = executor.execute_decode(
            session_id=session_id,
            hidden_states=torch.from_numpy(fixture.copy()),
            cache_position_start=0,
        )
        output = result.stage_boundary_hidden_states.detach().cpu().numpy().copy()
        executor.runtime.synchronize()
        error_state_ok = executor.runtime.error_state_ok()
        memory = executor.runtime.mem_info()
    finally:
        executor.close_session(session_id)
    health = _health_snapshot(executor.runtime.device)
    return {
        "standard_batch1_path": True,
        "output_finite": bool(np.isfinite(output).all()),
        "output_fingerprint": _array_fingerprint(output),
        "cuda_synchronize": "PASS",
        "cuda_error_state_ok": error_state_ok,
        "free_vram_bytes": memory["free_bytes"],
        "nvidia_smi": health,
        "pass": bool(np.isfinite(output).all())
        and error_state_ok
        and health["status"] == "MEASURED",
    }


async def _benchmark_complete_stage_batch(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    graph_certification: Path,
    output_path: Path,
    *,
    layer: int,
    device: int,
    warmup: int,
    iterations: int,
    phase_iterations: int,
    target_batch: int,
    cycle_id: str,
) -> dict[str, Any]:
    if target_batch not in INCREMENTAL_BATCHES:
        raise ValueError(f"target batch must be one of {INCREMENTAL_BATCHES}")
    if warmup < 3 or iterations < 20 or phase_iterations < 3:
        raise ValueError("complete-stage batch requires >=3/20/3 warm/retained/phase calls")
    resolved_identity, identity_evidence = _resolve_worker_identity_manifest(
        identity_manifest, layer=layer
    )
    paths = {
        "checkpoint": checkpoint.resolve(),
        "cuda_library": cuda_library.resolve(),
        "oracle_trace": oracle_trace.resolve(),
        "oracle_routes": oracle_routes.resolve(),
        "identity_manifest": resolved_identity,
        "graph_certification": graph_certification.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    graph_receipt = json.loads(paths["graph_certification"].read_text(encoding="utf-8"))
    certified_fixture = graph_receipt.get("fixture", {})
    oracle_provenance = {
        "graph_status": graph_receipt.get("status"),
        "certified_trace_sha256": certified_fixture.get("oracle_trace_sha256"),
        "configured_trace_sha256": _sha256_file(paths["oracle_trace"]),
        "certified_routes_sha256": certified_fixture.get("oracle_routes_sha256"),
        "configured_routes_sha256": _sha256_file(paths["oracle_routes"]),
    }
    oracle_provenance["pass"] = (
        oracle_provenance["graph_status"] == "PASS"
        and oracle_provenance["certified_trace_sha256"]
        == oracle_provenance["configured_trace_sha256"]
        and oracle_provenance["certified_routes_sha256"]
        == oracle_provenance["configured_routes_sha256"]
    )
    if not oracle_provenance["pass"]:
        raise ValueError(
            "complete-stage oracle does not match the supplied passing graph certification"
        )
    attempted_batches = tuple(
        batch for batch in INCREMENTAL_BATCHES if batch <= target_batch
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "RUNNING",
        "hypothesis": {
            "prediction": (
                "Grouping rows only when their real top-16 routes share an expert, "
                "while batching latent projections and the shared expert, will retain "
                "serial semantics and yield at least 1.5x complete-stage aggregate "
                "throughput at one certified batch without GPU or lifecycle degradation."
            ),
            "minimum_material_throughput_gain": MINIMUM_MATERIAL_THROUGHPUT_GAIN,
        },
        "implementation": {
            "attention_state": "independent persistent state per decode stream",
            "router": "one exact real Kimi router call per row",
            "routed_experts": "expert-ID grouped gather/row-cooperative execute/scatter",
            "reduction": "unchanged deterministic per-row top-16 reduction",
            "shared_expert": "row-cooperative across all active rows",
            "workspace": "fixed persistent allocation before READY",
            "larger_than_target_executed": False,
        },
        "configuration": {
            "layer": layer,
            "attention_type": "KDA" if layer % 4 != 3 else "Gated_MLA",
            "device": device,
            "attempted_batches": list(attempted_batches),
            "warmup_calls": warmup,
            "retained_calls": iterations,
            "phase_profile_calls": phase_iterations,
            "fixture": "three exact retained stateful decode boundaries, row-rotated",
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "progress": [],
        "batches": {},
        "oracle_provenance": oracle_provenance,
        "identity_evidence": identity_evidence,
    }

    def retain(phase: str, **evidence: Any) -> None:
        receipt["progress"].append({"phase": phase, **evidence})
        _atomic_json(output_path, receipt)

    retain("preregistered")
    health_before = _health_snapshot(device)
    receipt["gpu_health_before"] = health_before
    receipt["device_identity"] = _device_identity(device)
    retain("gpu_health_before", status=health_before["status"])
    if health_before["status"] != "MEASURED":
        raise RuntimeError("nvidia-smi unavailable before complete-stage batching")

    assignment = _source_assignment(checkpoint, layer=layer, device=f"native-cuda:{device}")
    fixtures, expected_boundaries = _stage_fixtures(
        checkpoint, oracle_trace, layer=layer
    )
    expected_routes = _parse_oracle_routes(oracle_routes)[layer]
    connection_pool = _CaptureConnectionPool()
    runtime = PersistentStageRuntime(
        worker_id=f"{cycle_id.lower()}-worker-{layer:03d}",
        device=f"native-cuda:{device}",
        dtype="float32",
        memory_limit_bytes=31 * 1024**3,
        maximum_sessions=2 * target_batch + 4,
        configured_model_path=checkpoint,
        configured_model_identity_path=paths["identity_manifest"],
        connection_pool=connection_pool,  # type: ignore[arg-type]
    )
    request = LoadStageRequest(
        worker_id=runtime.worker_id,
        request_id=f"{cycle_id.lower()}-layer-{layer}-load",
        model_id="moonshotai/Kimi-K3",
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        topology_id=f"{cycle_id.lower()}-layer-{layer}",
        route_generation=1,
        stage_count=93,
        assignment=assignment,
        adapter_id="kimi_k3_cuda",
        fast_path_id="colibri-kimi-k3-cuda",
        fast_path_mode="resident",
        fast_path_context_bucket=warmup + iterations + phase_iterations + 3,
        model_content_fingerprint=MODEL_CONTENT_FINGERPRINT,
        native_runtime_library=str(cuda_library),
        native_runtime_library_sha256=_sha256_file(cuda_library),
        device=f"native-cuda:{device}",
        dtype="float32",
        model_path=str(checkpoint),
    )
    try:
        load_started = time.perf_counter_ns()
        load_response = await runtime.load_stage(request)
        executor = runtime.loaded_executor
        if not isinstance(executor, PersistentKimiStageExecutor):
            raise TypeError("registered adapter returned a non-Kimi stage executor")
        receipt["load"] = {
            "accepted": load_response.accepted,
            "elapsed_ms": (time.perf_counter_ns() - load_started) / 1e6,
            "assignment_source_weight_bytes": assignment.weight_bytes,
            "resident_device_bytes": executor.resident_device_bytes,
            "weight_fingerprint": executor.weight_fingerprint,
            "lifecycle": executor.lifecycle_snapshot(),
        }
        retain(
            "production_stage_loaded_and_prepared",
            accepted=load_response.accepted,
            resident_device_bytes=executor.resident_device_bytes,
        )
        supported_batches = tuple(executor.runtime.expert_supported_batches)
        if tuple(
            size for size in supported_batches if size <= target_batch
        ) != attempted_batches:
            raise RuntimeError(
                "candidate exact batch contract omits the incremental target prefix: "
                f"supported={supported_batches}, target_prefix={attempted_batches}"
            )
        receipt["implementation"]["binary_supported_batches"] = list(
            supported_batches
        )
        serial_baseline = _serial_oracle_and_baseline(
            executor,
            fixtures,
            expected_boundaries,
            expected_routes,
            layer=layer,
            warmup=warmup,
            iterations=iterations,
        )
        receipt["serial_baseline"] = serial_baseline
        retain("serial_baseline", status=serial_baseline["status"])
        if serial_baseline["status"] != "PASS":
            raise RuntimeError("serial complete-stage baseline regressed")

        for batch in attempted_batches:
            retain(
                f"armed_batch_{batch}",
                prior_batches_passed=[
                    size
                    for size in attempted_batches
                    if size < batch and receipt["batches"].get(str(size), {}).get("status") == "PASS"
                ],
                nvidia_smi=_health_snapshot(device)["status"],
            )
            correctness = _batch_serial_equivalence(
                executor, fixtures, layer=layer, batch=batch
            )
            if not correctness["pass"]:
                receipt["batches"][str(batch)] = {
                    "correctness": correctness,
                    "status": "FAIL",
                }
                retain(f"batch_{batch}_correctness_failed")
                raise RuntimeError(f"batch {batch} differs from serial execution")
            measured = _measure_batch(
                executor,
                runtime,
                fixtures,
                serial_baseline,
                layer=layer,
                batch=batch,
                warmup=warmup,
                iterations=iterations,
                phase_iterations=phase_iterations,
            )
            safe = _known_safe_fixture(
                executor, fixtures[0], layer=layer, batch=batch
            )
            executor.runtime.synchronize()
            error_state_ok = executor.runtime.error_state_ok()
            health = _health_snapshot(device)
            row = {
                "correctness": correctness,
                "performance": measured,
                "post_batch_safe_fixture": safe,
                "post_batch_checks": {
                    "cuda_synchronize": "PASS",
                    "cuda_error_state_ok": error_state_ok,
                    "free_vram_bytes": executor.runtime.mem_info()["free_bytes"],
                    "nvidia_smi": health,
                },
            }
            row["status"] = (
                "PASS"
                if correctness["pass"]
                and measured["status"] == "PASS"
                and safe["pass"]
                and error_state_ok
                and health["status"] == "MEASURED"
                else "FAIL"
            )
            receipt["batches"][str(batch)] = row
            retain(
                f"batch_{batch}_persisted_and_checked",
                status=row["status"],
                device_p50_ms=measured["device"]["p50_ms"],
                throughput_gain=measured["throughput_gain_vs_serial_batch1"],
                effective_weight_reuse=measured["routing"][
                    "effective_weight_reuse_rows_per_native_call"
                ],
                cuda_error_state_ok=error_state_ok,
                nvidia_smi=health["status"],
            )
            if row["status"] != "PASS":
                raise RuntimeError(f"batch {batch} failed a retained safety gate")

        passing = [
            row["performance"]
            for row in receipt["batches"].values()
            if row["status"] == "PASS"
        ]
        best = max(passing, key=lambda row: float(row["aggregate_rows_per_second"]))
        material = [
            row
            for row in passing
            if float(row["throughput_gain_vs_serial_batch1"])
            >= MINIMUM_MATERIAL_THROUGHPUT_GAIN
        ]
        receipt["inspection"] = {
            "best_measured_batch": best["batch"],
            "best_aggregate_rows_per_second": best["aggregate_rows_per_second"],
            "best_throughput_gain_vs_serial_batch1": best[
                "throughput_gain_vs_serial_batch1"
            ],
            "material_batches": [row["batch"] for row in material],
            "fixture_limitation": (
                "Only three exact retained stateful decode positions are row-rotated; "
                "route overlap is measured but may overstate a diverse production fleet."
            ),
            "actual_bottleneck": "pending numerical inspection",
        }
        receipt["hypothesis_supported"] = bool(material)
        receipt["decision"] = {
            "safety_and_correctness": "RETAIN",
            "production_batch": "PENDING_DIVERSE_ROUTE_CORPUS",
            "next_hypothesis": (
                "Repeat the winning and limiting sizes on a diverse real-activation "
                "route corpus before selecting the production batch."
            ),
        }
        receipt["gpu_health_after"] = _health_snapshot(device)
        receipt["status"] = (
            "PASS"
            if all(row["status"] == "PASS" for row in receipt["batches"].values())
            and receipt["gpu_health_after"]["status"] == "MEASURED"
            else "FAIL"
        )
        retain("complete", status=receipt["status"])
        return receipt
    finally:
        await runtime.close()


def benchmark_complete_stage_batch(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    graph_certification: Path,
    output_path: Path,
    *,
    layer: int,
    device: int = 0,
    warmup: int = 10,
    iterations: int = 50,
    phase_iterations: int = 5,
    target_batch: int = 16,
    cycle_id: str = "H014-029",
) -> dict[str, Any]:
    """Run one layer's batch sizes incrementally with an atomic receipt per size."""
    try:
        return asyncio.run(
            _benchmark_complete_stage_batch(
                checkpoint,
                cuda_library,
                oracle_trace,
                oracle_routes,
                identity_manifest,
                graph_certification,
                output_path,
                layer=layer,
                device=device,
                warmup=warmup,
                iterations=iterations,
                phase_iterations=phase_iterations,
                target_batch=target_batch,
                cycle_id=cycle_id,
            )
        )
    except Exception as exc:
        if output_path.exists():
            receipt = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            receipt = {
                "schema_version": SCHEMA_VERSION,
                "cycle_id": cycle_id,
                "status": "RUNNING",
            }
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "gpu_health": _health_snapshot(device),
        }
        _atomic_json(output_path, receipt)
        raise
