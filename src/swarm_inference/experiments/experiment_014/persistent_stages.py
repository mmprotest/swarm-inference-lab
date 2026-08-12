"""H014-026c persistent non-final Kimi CUDA stage experiment."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swarm_inference.execution.kimi_k3_stage import (
    PersistentKimiStageExecutor,
    _added_thread_metadata,
    _process_snapshot,
    _timing,
    _warm_lifecycle_delta,
)
from swarm_inference.experiments.experiment_014.batch_safety import _health_snapshot
from swarm_inference.experiments.experiment_014.cuda import (
    KimiCudaError,
    _array_fingerprint,
    _numerical_metrics,
    _sha256_file,
)
from swarm_inference.experiments.experiment_014.full_cuda import (
    _CheckpointReader,
    _parse_oracle_routes,
)
from swarm_inference.model.kimi_k3 import KimiK3CudaAdapter, _SafetensorCatalog
from swarm_inference.model.partition import StageAssignment
from swarm_inference.protocol.stage_ring import Operation, StageMessage, encode_message
from swarm_inference.protocol.stage_worker import (
    CloseStageSessionRequest,
    InstallStageRouteRequest,
    LoadStageRequest,
    OpenStageSessionRequest,
    StageRouteEndpoint,
)
from swarm_inference.transport.stage_tensor import pack_tensor, unpack_tensor
from swarm_inference.worker.stage_runtime import PersistentStageRuntime

SCHEMA_VERSION = "experiment-014-k3-persistent-nonfinal-stages-v1"
MODEL_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
TOKENIZER_REVISION = (
    "sha256:49f733745c76dbd69bd90fa109a66929af087e881bcffadbd4068b68635fd526"
)
MODEL_CONTENT_FINGERPRINT = (
    "25162130a11904bac1220a7d654a3f7dfd616ea5f4035d488e40ac74ddea8f94"
)


def _resolve_worker_identity_manifest(
    identity_manifest: Path, *, layer: int
) -> tuple[Path, dict[str, str]]:
    """Resolve and fail closed on the worker identity used by a P1 fixture."""

    source = identity_manifest.expanduser().resolve()
    expected_worker_id = f"k3-worker-{layer:03d}"
    candidate = (
        source / f"{expected_worker_id}-model-identity.json"
        if source.is_dir()
        else source
    )
    if not candidate.is_file():
        raise ValueError(f"missing scoped identity for {expected_worker_id}: {candidate}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid scoped identity for {expected_worker_id}") from exc
    if not isinstance(value, dict) or value.get("worker_id") != expected_worker_id:
        raise ValueError(f"scoped identity belongs to the wrong worker: {expected_worker_id}")
    assignment_sha256 = value.get("assignment_sha256")
    if (
        not isinstance(assignment_sha256, str)
        or len(assignment_sha256) != 64
        or any(character not in "0123456789abcdef" for character in assignment_sha256)
    ):
        raise ValueError(
            f"scoped identity has no valid assignment SHA-256: {expected_worker_id}"
        )
    return candidate, {
        "worker_id": expected_worker_id,
        "assignment_sha256": assignment_sha256,
        "manifest_sha256": _sha256_file(candidate),
    }


async def _run_production_batch_fixture(
    runtime: PersistentStageRuntime,
    executor: PersistentKimiStageExecutor,
    *,
    stage_input: np.ndarray,
    expected_boundary: np.ndarray,
    expected_expert_ids: list[int],
) -> dict[str, Any]:
    """Prove real B8, pre-CUDA B9 rejection, and a post-guard safe B1."""

    batch_ids = tuple(f"__h014_038_b8_{row}" for row in range(8))
    memory_before = await runtime._run_compute(executor.runtime.mem_info)
    for session_id in batch_ids:
        await runtime._run_compute(
            executor.open_session,
            session_id,
            maximum_context_override=3,
        )
    batch_before = _process_snapshot(runtime, executor)
    try:
        batch_record = await runtime._run_compute(
            executor.execute_decode_batch,
            session_ids=batch_ids,
            hidden_states=torch.from_numpy(np.repeat(stage_input, 8, axis=0)),
            cache_position_start=0,
        )
        batch_after = _process_snapshot(runtime, executor)
        batch_states = [
            await runtime._run_compute(executor.session_state_evidence, item)
            for item in batch_ids
        ]
    finally:
        for session_id in batch_ids:
            await runtime._run_compute(executor.close_session, session_id)
    memory_after_batch_close = await runtime._run_compute(executor.runtime.mem_info)
    comparisons = [
        _numerical_metrics(expected_boundary[0], batch_record["boundary_output"][row])
        for row in range(8)
    ]
    batch_delta = _warm_lifecycle_delta(batch_before, batch_after)
    compute_thread_native_id = runtime.compute_executor_snapshot()["thread_native_id"]
    batch_pass = (
        executor.lifecycle_snapshot()["batch_capacity"] == 8
        and all(float(row["relative_l2_error"]) <= 3e-5 for row in comparisons)
        and all(
            route == expected_expert_ids
            for route in batch_record["selected_expert_ids"]
        )
        and bool(batch_record["routing"]["all_selected_experts_executed_once"])
        and all(row["cache_sequence_length"] == 1 for row in batch_states)
        and len({row["fingerprint"] for row in batch_states}) == 1
        and all(value == 0 for value in batch_delta.values())
        and batch_record["host_thread_native_id"] == compute_thread_native_id
    )

    guard_before = _process_snapshot(runtime, executor)
    records_before = len(executor.execution_records)
    rejection = ""
    try:
        await runtime._run_compute(
            executor.execute_decode_batch,
            session_ids=tuple(f"__h014_038_b9_{row}" for row in range(9)),
            hidden_states=torch.from_numpy(np.repeat(stage_input, 9, axis=0)),
            cache_position_start=0,
        )
    except KimiCudaError as exc:
        rejection = str(exc)
    await runtime._run_compute(executor.runtime.synchronize)
    guard_error_state_ok = await runtime._run_compute(executor.runtime.error_state_ok)
    guard_after = _process_snapshot(runtime, executor)
    guard_pass = (
        rejection
        == "Kimi complete-stage batch rejected before CUDA work: requested=9, certified_max=8"
        and guard_before["executor"]["execute_count"]
        == guard_after["executor"]["execute_count"]
        and guard_before["executor"]["batch_execute_count"]
        == guard_after["executor"]["batch_execute_count"]
        and records_before == len(executor.execution_records)
        and guard_error_state_ok
    )

    safe_id = "__h014_038_post_guard_safe"
    await runtime._run_compute(
        executor.open_session,
        safe_id,
        maximum_context_override=3,
    )
    try:
        safe_result = await runtime._run_compute(
            executor.execute_decode,
            session_id=safe_id,
            hidden_states=torch.from_numpy(stage_input.copy()),
            cache_position_start=0,
        )
        safe_record = executor.execution_records[-1]
        safe_metrics = _numerical_metrics(
            expected_boundary,
            safe_result.stage_boundary_hidden_states.detach().cpu().numpy(),
        )
        safe_error_state_ok = await runtime._run_compute(
            executor.runtime.error_state_ok
        )
        safe_pass = (
            float(safe_metrics["relative_l2_error"]) <= 3e-5
            and list(safe_record["selected_expert_ids"]) == expected_expert_ids
            and int(safe_record["routed_expert_execution_count"]) == 16
            and bool(safe_record["all_selected_experts_executed_once"])
            and safe_record["host_thread_native_id"] == compute_thread_native_id
            and safe_error_state_ok
        )
    finally:
        await runtime._run_compute(executor.close_session, safe_id)
    memory_after = await runtime._run_compute(executor.runtime.mem_info)
    initial_retained_bytes = max(
        0,
        int(memory_before["free_bytes"])
        - int(memory_after_batch_close["free_bytes"]),
    )
    post_safe_additional_bytes = max(
        0,
        int(memory_after_batch_close["free_bytes"])
        - int(memory_after["free_bytes"]),
    )
    memory_recovered = (
        initial_retained_bytes <= 4 * 1024**2
        and post_safe_additional_bytes <= 1024**2
    )
    return {
        "pass": batch_pass and guard_pass and safe_pass and memory_recovered,
        "certified_batch": 8,
        "native_supported_batches": list(executor.runtime.expert_supported_batches),
        "batch_8": {
            "pass": batch_pass,
            "device_ms": float(batch_record["device_ms"]),
            "rows_per_second": 8000.0 / float(batch_record["device_ms"]),
            "maximum_relative_l2_error": max(
                float(row["relative_l2_error"]) for row in comparisons
            ),
            "routing_equality": all(
                route == expected_expert_ids
                for route in batch_record["selected_expert_ids"]
            ),
            "all_selected_experts_executed_once": bool(
                batch_record["routing"]["all_selected_experts_executed_once"]
            ),
            "lifecycle_delta": batch_delta,
            "host_thread_native_id": batch_record["host_thread_native_id"],
            "expected_compute_thread_native_id": compute_thread_native_id,
            "state_fingerprints_equal_for_equal_inputs": (
                len({row["fingerprint"] for row in batch_states}) == 1
            ),
        },
        "batch_9_guard": {
            "pass": guard_pass,
            "requested_batch": 9,
            "rejection": rejection,
            "execution_records_delta": len(executor.execution_records) - records_before - 1,
            "cuda_error_state_ok": guard_error_state_ok,
        },
        "post_guard_batch_1": {
            "pass": safe_pass,
            "relative_l2_error": float(safe_metrics["relative_l2_error"]),
            "routing_equality": list(safe_record["selected_expert_ids"])
            == expected_expert_ids,
            "routed_expert_execution_count": int(
                safe_record["routed_expert_execution_count"]
            ),
            "all_selected_experts_executed_once": bool(
                safe_record["all_selected_experts_executed_once"]
            ),
            "host_thread_native_id": safe_record["host_thread_native_id"],
            "expected_compute_thread_native_id": compute_thread_native_id,
        },
        "memory": {
            "before": memory_before,
            "after_batch_close": memory_after_batch_close,
            "after_safe_close": memory_after,
            "initial_retained_bytes": initial_retained_bytes,
            "initial_retained_limit_bytes": 4 * 1024**2,
            "post_safe_additional_bytes": post_safe_additional_bytes,
            "post_safe_additional_limit_bytes": 1024**2,
            "stable_allocator_plateau": memory_recovered,
        },
    }


class _CaptureConnectionPool:
    """Already-connected successor that captures the canonical forwarded frame."""

    def __init__(self) -> None:
        self.forwarded: list[StageMessage] = []
        self._response_sequences: dict[tuple[str, int, int], int] = {}
        self._closed = False
        self._metrics = {
            "connections_created": 1,
            "connection_reuses": 0,
            "reconnects": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "wire_bytes_sent": 0,
            "payload_bytes_sent": 0,
            "backpressure_events": 0,
            "failures": 0,
            "connection_evictions": 0,
            "response_timeouts": 0,
            "fault_injections": 0,
        }

    async def send(self, endpoint: str, message: StageMessage) -> StageMessage:
        if self._closed:
            raise RuntimeError("capture connection is closed")
        del endpoint
        self.forwarded.append(message)
        frame = encode_message(message)
        self._metrics["connection_reuses"] += 1
        self._metrics["messages_sent"] += 1
        self._metrics["messages_received"] += 1
        self._metrics["wire_bytes_sent"] += frame.wire_bytes
        self._metrics["payload_bytes_sent"] += frame.payload_bytes
        packed = pack_tensor(torch.tensor([0], dtype=torch.int64), requested_mode="none")
        key = (message.session_id, message.stage_id, message.source_stage)
        sequence = self._response_sequences.get(key, 0)
        self._response_sequences[key] = sequence + 1
        return StageMessage(
            operation=Operation.TOKEN_RESULT,
            model_revision=message.model_revision,
            tokenizer_revision=message.tokenizer_revision,
            topology_id=message.topology_id,
            stage_id=message.stage_id,
            layer_start=message.layer_start,
            layer_end=message.layer_end,
            session_id=message.session_id,
            request_id=message.request_id,
            sequence_number=sequence,
            token_position=message.token_position,
            source_stage=message.stage_id,
            destination_stage=message.source_stage,
            tensor_shape=packed.shape,
            tensor_dtype=packed.dtype,
            compression_mode=packed.compression_mode,
            payload=packed.payload,
            attributes={
                "model_id": message.attributes["model_id"],
                "route_generation": message.attributes["route_generation"],
                "request_generation": message.attributes["request_generation"],
                "replay_only": message.attributes["replay_only"],
                "source_worker_id": message.attributes["destination_worker_id"],
                "destination_worker_id": message.attributes["source_worker_id"],
                "cache_sequence_length": message.attributes["cache_sequence_length"],
                "tensor": packed.attributes(),
            },
        )

    async def remove(self, endpoint: str) -> None:
        del endpoint

    def snapshot(self) -> dict[str, object]:
        return {
            **self._metrics,
            "active_connections": 0 if self._closed else 1,
            "endpoints": [] if self._closed else ["capture://successor"],
            "queue_capacity": 1,
        }

    async def close(self) -> None:
        self._closed = True


def _source_assignment(
    checkpoint: Path,
    *,
    layer: int,
    device: str = "native-cuda:0",
    catalog: _SafetensorCatalog | None = None,
) -> StageAssignment:
    adapter = KimiK3CudaAdapter()
    catalog = catalog or _SafetensorCatalog(checkpoint)
    provisional = StageAssignment(
        stage_id=layer,
        layer_start=layer,
        layer_end=layer + 1,
        layer_ids=(layer,),
        weight_bytes=1,
        estimated_compute_ns=1,
        measured_compute_ns=None,
        kv_cache_bytes_per_token=6400,
        peak_temporary_bytes=512 * 1024**2,
        activation_bytes=9 * 7168 * 4,
        device=device,
        owns_embeddings=layer == 0,
        owns_final_norm=layer == 92,
        owns_output_projection=layer == 92,
    )
    names = adapter._owned_tensor_names(catalog, provisional)
    source_bytes = sum(catalog.tensor_info(name)[3] for name in names)
    return StageAssignment(
        stage_id=provisional.stage_id,
        layer_start=provisional.layer_start,
        layer_end=provisional.layer_end,
        layer_ids=provisional.layer_ids,
        weight_bytes=source_bytes,
        estimated_compute_ns=provisional.estimated_compute_ns,
        measured_compute_ns=provisional.measured_compute_ns,
        kv_cache_bytes_per_token=provisional.kv_cache_bytes_per_token,
        peak_temporary_bytes=provisional.peak_temporary_bytes,
        activation_bytes=provisional.activation_bytes,
        device=provisional.device,
        owns_embeddings=provisional.owns_embeddings,
        owns_final_norm=provisional.owns_final_norm,
        owns_output_projection=provisional.owns_output_projection,
    )


def _stage_fixtures(
    checkpoint: Path,
    oracle_trace: Path,
    *,
    layer: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    reader = _CheckpointReader(checkpoint)
    config = reader.config
    trace = np.memmap(
        oracle_trace,
        mode="r",
        dtype="<f4",
        shape=(3 * (config.layers + 1), config.hidden),
    )
    embedding = reader.array("language_model.model.embed_tokens.weight")
    token_ids = (163584, 18699, 11)
    inputs: list[np.ndarray] = []
    expected: list[np.ndarray] = []
    for position, token_id in enumerate(token_ids):
        base = position * (config.layers + 1)
        boundary = np.zeros((1, 9, config.hidden), dtype=np.float32)
        if layer == 0:
            stage_input = np.ascontiguousarray([[token_id]], dtype=np.int64)
            token_bits = np.asarray(embedding[token_id], dtype=np.uint16)
            boundary[0, 0] = (
                token_bits.astype(np.uint32) << np.uint32(16)
            ).view(np.float32)
        else:
            boundary[0, 0] = trace[base + layer - 1]
            stage_input = boundary
        for snapshot_layer in range(0, layer, config.residual_block):
            slot = 1 + snapshot_layer // config.residual_block
            if snapshot_layer == 0:
                token_bits = np.asarray(embedding[token_id], dtype=np.uint16)
                boundary[0, slot] = (
                    token_bits.astype(np.uint32) << np.uint32(16)
                ).view(np.float32)
            else:
                boundary[0, slot] = trace[base + snapshot_layer - 1]
        output = boundary.copy()
        output[0, 0] = trace[base + layer]
        if layer % config.residual_block == 0:
            block_count = (layer + config.residual_block - 1) // config.residual_block
            output[0, 1 + block_count] = boundary[0, 0]
        inputs.append(stage_input)
        expected.append(output)
    return inputs, expected


def _incoming_message(
    boundary: np.ndarray,
    *,
    layer: int,
    position: int,
    session_id: str,
    topology_id: str,
    cycle_id: str,
) -> StageMessage:
    packed = pack_tensor(torch.from_numpy(boundary.copy()), requested_mode="none")
    return StageMessage(
        operation=Operation.PREFILL if position == 0 else Operation.DECODE,
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        topology_id=topology_id,
        stage_id=layer,
        layer_start=layer,
        layer_end=layer + 1,
        session_id=session_id,
        request_id=f"{cycle_id.lower()}-layer-{layer}-position-{position}",
        sequence_number=position,
        token_position=position,
        source_stage=layer - 1,
        destination_stage=layer,
        tensor_shape=packed.shape,
        tensor_dtype=packed.dtype,
        compression_mode=packed.compression_mode,
        payload=packed.payload,
        attributes={
            "model_id": "moonshotai/Kimi-K3",
            "route_generation": 1,
            "request_generation": 1,
            "replay_only": False,
            "source_worker_id": (
                "coordinator" if layer == 0 else f"k3-worker-{layer - 1:03d}"
            ),
            "destination_worker_id": f"k3-worker-{layer:03d}",
            "cache_position_start": position,
            "deadline_ns": time.time_ns() + 600_000_000_000,
            "tensor": packed.attributes(),
        },
    )


async def _measure_stage_zero_steady_performance(
    runtime: PersistentStageRuntime,
    executor: PersistentKimiStageExecutor,
    connection_pool: _CaptureConnectionPool,
    inputs: list[np.ndarray],
    *,
    open_session: Any,
    close_session: Any,
    topology_id: str,
    cycle_id: str,
    post_quiescence_device_p50_ms: float,
) -> dict[str, Any]:
    """Separate sustained stage-zero service from the P1 idle lifecycle probe."""

    warmup_calls = 21
    retained_calls = 100

    async def run_window(prefix: str, calls: int) -> tuple[list[float], list[float], str]:
        wall_ms: list[float] = []
        device_ms: list[float] = []
        session_id = prefix
        await open_session(session_id)
        for index in range(calls):
            incoming = _incoming_message(
                inputs[index % len(inputs)],
                layer=0,
                position=index,
                session_id=session_id,
                topology_id=topology_id,
                cycle_id=f"{cycle_id}-{prefix}-{index:03d}",
            )
            started = time.perf_counter_ns()
            await runtime.handle_message(incoming)
            wall_ms.append((time.perf_counter_ns() - started) / 1e6)
            device_ms.append(float(executor.execution_records[-1]["device_ms"]))
        return wall_ms, device_ms, session_id

    health_before = _health_snapshot(executor.runtime.device)
    memory_before = await runtime._run_compute(executor.runtime.mem_info)
    executor.execution_records.clear()
    connection_pool.forwarded.clear()
    warmup_wall_ms, warmup_device_ms, warmup_session = await run_window(
        f"{cycle_id.lower()}-steady-warm", warmup_calls
    )
    await close_session(warmup_session)
    executor.execution_records.clear()
    connection_pool.forwarded.clear()
    retained_session = f"{cycle_id.lower()}-steady-retained"
    await open_session(retained_session)
    before = _process_snapshot(runtime, executor)
    retained_wall_ms: list[float] = []
    retained_device_ms: list[float] = []
    for index in range(retained_calls):
        incoming = _incoming_message(
            inputs[index % len(inputs)],
            layer=0,
            position=index,
            session_id=retained_session,
            topology_id=topology_id,
            cycle_id=f"{cycle_id}-steady-retained-{index:03d}",
        )
        started = time.perf_counter_ns()
        await runtime.handle_message(incoming)
        retained_wall_ms.append((time.perf_counter_ns() - started) / 1e6)
        retained_device_ms.append(float(executor.execution_records[-1]["device_ms"]))
    await runtime._run_compute(executor.runtime.synchronize)
    error_state_ok = await runtime._run_compute(executor.runtime.error_state_ok)
    after = _process_snapshot(runtime, executor)
    await close_session(retained_session)
    after_close = _process_snapshot(runtime, executor)
    memory_after = await runtime._run_compute(executor.runtime.mem_info)
    health_after = _health_snapshot(executor.runtime.device)
    wall = _timing(retained_wall_ms)
    device = _timing(retained_device_ms)
    lifecycle_delta = _warm_lifecycle_delta(before, after)
    memory_recovered = (
        int(memory_after["free_bytes"]) + 4 * 1024**2
        >= int(memory_before["free_bytes"])
    )
    gates = {
        "retained_call_count_exact": len(retained_device_ms) == retained_calls,
        "device_p50_at_most_3_5ms": float(device["p50_ms"]) <= 3.5,
        "at_least_two_times_faster_than_post_quiescence_probe": (
            float(device["p50_ms"]) * 2.0 <= post_quiescence_device_p50_ms
        ),
        "warm_lifecycle_deltas_zero": all(
            value == 0 for value in lifecycle_delta.values()
        ),
        "all_sessions_closed": int(after_close["executor"]["active_sessions"]) == 0,
        "temporary_state_memory_recovered": memory_recovered,
        "cuda_error_state_ok": bool(error_state_ok),
        "gpu_health_before_measured": health_before["status"] == "MEASURED",
        "gpu_health_after_measured": health_after["status"] == "MEASURED",
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "hypothesis_supported": all(gates.values()),
        "warmup_calls": warmup_calls,
        "retained_calls": retained_calls,
        "warm_session_calls": warmup_calls,
        "retained_session_calls": retained_calls,
        "wall": wall,
        "device": device,
        "warmup": {
            "wall": _timing(warmup_wall_ms),
            "device": _timing(warmup_device_ms),
            "first_device_ms": warmup_device_ms[0],
        },
        "post_quiescence_probe_device_p50_ms": post_quiescence_device_p50_ms,
        "post_quiescence_to_steady_p50_ratio": (
            post_quiescence_device_p50_ms / float(device["p50_ms"])
        ),
        "lifecycle_delta": lifecycle_delta,
        "active_sessions_during_retained_snapshot": after["executor"][
            "active_sessions"
        ],
        "active_sessions_after_close": after_close["executor"]["active_sessions"],
        "memory": {
            "free_bytes_before": memory_before["free_bytes"],
            "free_bytes_after": memory_after["free_bytes"],
            "temporary_state_memory_recovered": memory_recovered,
        },
        "cuda_error_state_ok": bool(error_state_ok),
        "gpu_health_before": health_before,
        "gpu_health_after": health_after,
        "acceptance_gates": gates,
        "measurement_semantics": (
            "continuous persistent production runtime after PREPARE; one warm session "
            "and one retained session, with session creation outside lifecycle timing"
        ),
    }


async def _run_stage(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    *,
    layer: int,
    cycle_id: str = "H014-026c",
) -> dict[str, Any]:
    identity_manifest, identity_evidence = _resolve_worker_identity_manifest(
        identity_manifest, layer=layer
    )
    catalog = _SafetensorCatalog(checkpoint)
    assignment = _source_assignment(checkpoint, layer=layer, catalog=catalog)
    previous_assignment = (
        _source_assignment(checkpoint, layer=layer - 1, catalog=catalog)
        if layer > 0
        else None
    )
    next_assignment = _source_assignment(
        checkpoint, layer=layer + 1, catalog=catalog
    )
    inputs, expected_boundaries = _stage_fixtures(
        checkpoint, oracle_trace, layer=layer
    )
    expected_routes = _parse_oracle_routes(oracle_routes).get(layer, {})
    request_prefix = cycle_id.lower()
    topology_id = f"{request_prefix}-layer-{layer}"
    connection_pool = _CaptureConnectionPool()
    runtime = PersistentStageRuntime(
        worker_id=f"k3-worker-{layer:03d}",
        device="native-cuda:0",
        dtype="float32",
        memory_limit_bytes=31 * 1024**3,
        maximum_sessions=2,
        configured_model_path=checkpoint,
        configured_model_identity_path=identity_manifest,
        connection_pool=connection_pool,  # type: ignore[arg-type]
    )
    request = LoadStageRequest(
        worker_id=runtime.worker_id,
        request_id=f"{request_prefix}-layer-{layer}-load",
        model_id="moonshotai/Kimi-K3",
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        topology_id=topology_id,
        route_generation=1,
        stage_count=93,
        assignment=assignment,
        adapter_id="kimi_k3_cuda",
        fast_path_id="colibri-kimi-k3-cuda",
        fast_path_mode="resident",
        fast_path_context_bucket=128 if layer == 0 else 3,
        model_content_fingerprint=MODEL_CONTENT_FINGERPRINT,
        native_runtime_library=str(cuda_library),
        native_runtime_library_sha256=_sha256_file(cuda_library),
        device="native-cuda:0",
        dtype="float32",
        model_path=str(checkpoint),
    )
    lease_expiry = time.time_ns() + 3_600_000_000_000
    try:
        load_response = await runtime.load_stage(request)
        executor = runtime.loaded_executor
        if not isinstance(executor, PersistentKimiStageExecutor):
            raise TypeError("registered adapter returned a non-Kimi stage executor")
        prepare = executor.prepare_warmup or {}
        prepare_stage_fixture = prepare.get("stage_fixture", {})
        prepare_transport = prepare_stage_fixture.get(
            "canonical_transport_round_trip", {}
        )
        prepare_thread_quiescence = prepare_stage_fixture.get(
            "thread_quiescence", {}
        )
        prepare_cpu_transport_threads = prepare_stage_fixture.get(
            "cpu_transport_threads", {}
        )
        prepare_compute_executor = runtime.compute_executor_snapshot()
        prepare_compute_thread_native_id = prepare_compute_executor["thread_native_id"]
        prepare_compute_thread_ident = prepare_compute_executor["thread_ident"]
        prepare_pass = (
            prepare.get("count") == 1
            and prepare.get("primitive") == "isolated assigned-stage Kimi CUDA fixture"
            and prepare_stage_fixture.get("iterations") == 7
            and prepare_stage_fixture.get("output_finite") is True
            and prepare_stage_fixture.get("research_records_removed") is True
            and prepare_stage_fixture.get("serving_execute_count_restored") is True
            and prepare_stage_fixture.get("temporary_memory_recovered") is True
            and prepare_transport.get("pass") is True
            and prepare_transport.get("source_iteration") == 6
            and prepare_transport.get("final_warm_iteration") == 7
            and prepare_transport.get("shape") == [1, 9, 7168]
            and prepare_transport.get("dtype") == "float32"
            and prepare_transport.get("compression_mode") == "none"
            and prepare_transport.get("raw_bytes") == 258_048
            and prepare_transport.get("encoded_bytes") == 258_048
            and prepare_thread_quiescence.get("pass") is True
            and prepare_thread_quiescence.get("minimum_observation_ms") == 3_500.0
            and prepare_thread_quiescence.get("required_stable_samples") == 20
            and prepare_thread_quiescence.get("stable_samples_observed", 0) >= 20
            and prepare_cpu_transport_threads.get("intraop_threads") == 1
            and prepare_cpu_transport_threads.get("interop_threads") == 1
            and prepare_compute_executor.get("max_workers") == 1
            and prepare_compute_executor.get("thread_start_count") == 1
            and prepare_stage_fixture.get("single_host_thread") is True
            and set(prepare_stage_fixture.get("host_thread_native_ids", []))
            == {prepare_compute_thread_native_id}
            and set(prepare_stage_fixture.get("host_thread_idents", []))
            == {prepare_compute_thread_ident}
        )
        route_response = await runtime.install_route(
            InstallStageRouteRequest(
                worker_id=runtime.worker_id,
                request_id=f"{request_prefix}-layer-{layer}-route",
                model_id=request.model_id,
                model_revision=MODEL_REVISION,
                tokenizer_revision=TOKENIZER_REVISION,
                topology_id=topology_id,
                route_generation=1,
                assignment=assignment,
                device="native-cuda:0",
                dtype="float32",
                previous_stage=(
                    StageRouteEndpoint(
                        worker_id=f"k3-worker-{layer - 1:03d}",
                        stage_id=layer - 1,
                        data_endpoint=f"127.0.0.1:{19000 + layer - 1}",
                        assignment=previous_assignment,
                    )
                    if previous_assignment is not None
                    else None
                ),
                next_stage=StageRouteEndpoint(
                    worker_id=f"k3-worker-{layer + 1:03d}",
                    stage_id=layer + 1,
                    data_endpoint=f"127.0.0.1:{19000 + layer + 1}",
                    assignment=next_assignment,
                ),
                stage_count=93,
                lease_expiry_unix_ns=lease_expiry,
            )
        )

        async def open_session(session_id: str) -> None:
            await runtime.open_session(
                OpenStageSessionRequest(
                    worker_id=runtime.worker_id,
                    request_id=f"{request_prefix}-{session_id}-open",
                    model_id=request.model_id,
                    model_revision=MODEL_REVISION,
                    tokenizer_revision=TOKENIZER_REVISION,
                    topology_id=topology_id,
                    route_generation=1,
                    stage_id=layer,
                    device="native-cuda:0",
                    dtype="float32",
                    session_id=session_id,
                    request_generation=1,
                    lease_expiry_unix_ns=lease_expiry,
                )
            )

        async def close_session(session_id: str) -> None:
            await runtime.close_session(
                CloseStageSessionRequest(
                    worker_id=runtime.worker_id,
                    request_id=f"{request_prefix}-{session_id}-close",
                    model_id=request.model_id,
                    model_revision=MODEL_REVISION,
                    tokenizer_revision=TOKENIZER_REVISION,
                    topology_id=topology_id,
                    route_generation=1,
                    stage_id=layer,
                    device="native-cuda:0",
                    dtype="float32",
                    session_id=session_id,
                    request_generation=1,
                    lease_expiry_unix_ns=lease_expiry,
                )
            )

        await open_session("warmup")
        await runtime.handle_message(
            _incoming_message(
                inputs[0],
                layer=layer,
                position=0,
                session_id="warmup",
                topology_id=topology_id,
                cycle_id=cycle_id,
            )
        )
        await close_session("warmup")
        executor.execution_records.clear()
        connection_pool.forwarded.clear()

        await open_session("retained")
        before = _process_snapshot(runtime, executor)
        per_call: list[dict[str, Any]] = []
        input_frames: list[dict[str, int]] = []
        for position, boundary in enumerate(inputs):
            incoming = _incoming_message(
                boundary,
                layer=layer,
                position=position,
                session_id="retained",
                topology_id=topology_id,
                cycle_id=cycle_id,
            )
            call_before = _process_snapshot(runtime, executor)
            await runtime.handle_message(incoming)
            call_after = _process_snapshot(runtime, executor)
            forwarded = connection_pool.forwarded[-1]
            incoming_frame = encode_message(incoming)
            forwarded_frame = encode_message(forwarded)
            input_frames.append(
                {
                    "incoming_payload_bytes": incoming_frame.payload_bytes,
                    "incoming_wire_bytes": incoming_frame.wire_bytes,
                    "forwarded_payload_bytes": forwarded_frame.payload_bytes,
                    "forwarded_wire_bytes": forwarded_frame.wire_bytes,
                }
            )
            per_call.append(
                {
                    "position": position,
                    "lifecycle_delta": _warm_lifecycle_delta(call_before, call_after),
                }
            )
        immediate_after = _process_snapshot(runtime, executor)
        thread_observation: list[dict[str, Any]] = []
        observation_before = immediate_after
        for sample in range(1, 21):
            await asyncio.sleep(0.05)
            observation_after = _process_snapshot(runtime, executor)
            if observation_before["os_thread_ids"] != observation_after["os_thread_ids"]:
                thread_observation.append(
                    {
                        "sample": sample,
                        "elapsed_ms": sample * 50,
                        "lifecycle_delta": _warm_lifecycle_delta(
                            observation_before,
                            observation_after,
                        ),
                        "added_threads": _added_thread_metadata(
                            observation_before,
                            observation_after,
                        ),
                    }
                )
            observation_before = observation_after
        after = observation_before
        lifecycle_delta = _warm_lifecycle_delta(before, after)
        records = list(executor.execution_records)
        correctness: list[dict[str, Any]] = []
        for position, (record, forwarded) in enumerate(
            zip(records, connection_pool.forwarded, strict=True)
        ):
            actual_boundary_tensor, _ = unpack_tensor(
                forwarded.payload, dict(forwarded.attributes["tensor"])
            )
            actual_boundary = np.ascontiguousarray(actual_boundary_tensor.numpy())
            expected_boundary = expected_boundaries[position]
            correctness.append(
                {
                    "position": position,
                    "host_thread_native_id": int(record["host_thread_native_id"]),
                    "host_thread_ident": int(record["host_thread_ident"]),
                    "host_thread_name": str(record["host_thread_name"]),
                    "input_fingerprint": _array_fingerprint(inputs[position]),
                    "weight_fingerprint": executor.weight_fingerprint,
                    "output_fingerprint": _array_fingerprint(actual_boundary),
                    "expected_output_fingerprint": _array_fingerprint(
                        expected_boundary
                    ),
                    "boundary_metrics": _numerical_metrics(
                        expected_boundary, actual_boundary
                    ),
                    "residual_rows_bit_exact": bool(
                        np.array_equal(expected_boundary[:, 1:], actual_boundary[:, 1:])
                    ),
                    "boundary_shape": list(actual_boundary.shape),
                    "boundary_dtype": str(actual_boundary.dtype),
                    "expected_expert_ids": expected_routes.get(position, []),
                    "selected_expert_ids": record["selected_expert_ids"],
                    "routing_equality": record["selected_expert_ids"]
                    == expected_routes.get(position, []),
                }
            )
        state = executor.session_state_evidence("retained")
        maximum_relative_error = max(
            float(row["boundary_metrics"]["relative_l2_error"])
            for row in correctness
        )
        compute_executor = before["compute_executor"]
        compute_thread_native_id = compute_executor["thread_native_id"]
        compute_thread_ident = compute_executor["thread_ident"]
        compute_thread_consistent = (
            compute_thread_native_id is not None
            and compute_thread_ident is not None
            and compute_executor["max_workers"] == 1
            and compute_executor["thread_start_count"] == 1
            and prepare_compute_thread_native_id == compute_thread_native_id
            and prepare_compute_thread_ident == compute_thread_ident
            and {row["host_thread_native_id"] for row in correctness}
            == {compute_thread_native_id}
            and {row["host_thread_ident"] for row in correctness}
            == {compute_thread_ident}
        )
        lifecycle_pass = (
            compute_thread_consistent
            and all(value == 0 for value in lifecycle_delta.values())
            and all(
                all(value == 0 for value in item["lifecycle_delta"].values())
                for item in per_call
            )
        )
        correctness_pass = (
            maximum_relative_error <= 3e-5
            and all(row["routing_equality"] for row in correctness)
            and all(row["residual_rows_bit_exact"] for row in correctness)
            and all(row["boundary_shape"] == [1, 9, 7168] for row in correctness)
            and all(row["boundary_dtype"] == "float32" for row in correctness)
            and state["cache_sequence_length"] == 3
            and state["finite"]
            and state["nonzero_prefix"]
            and state["zero_suffix"]
        )
        production_batch = (
            await _run_production_batch_fixture(
                runtime,
                executor,
                stage_input=inputs[0],
                expected_boundary=expected_boundaries[0],
                expected_expert_ids=expected_routes.get(0, []),
            )
            if layer > 0
            else None
        )
        production_batch_pass = production_batch is None or bool(
            production_batch["pass"]
        )
        result = {
            "layer": layer,
            "attention_type": state["attention_type"],
            "status": (
                "PASS"
                if correctness_pass
                and lifecycle_pass
                and prepare_pass
                and production_batch_pass
                else "FAIL"
            ),
            "backend": {
                "identity": "nvidia_cuda_persistent_kimi_stage",
                "implementation_class": (
                    f"{executor.__class__.__module__}.{executor.__class__.__name__}"
                ),
                "cpu_mathematical_fallbacks": 0,
                "cuda_library_sha256": executor.cuda_library_sha256,
            },
            "assignment": {
                "source_weight_bytes": assignment.weight_bytes,
                "owned_tensor_count": executor.ownership.parameter_count,
                "ownership_hash": executor.ownership.ownership_hash,
                "weight_fingerprint": executor.weight_fingerprint,
                "identity_manifest": str(identity_manifest),
                **identity_evidence,
            },
            "load": {
                "accepted": load_response.accepted,
                "route_accepted": route_response.accepted,
                "elapsed_ms": executor.load_ns / 1e6,
                "tracked_device_bytes": executor.tracked_device_bytes,
                "resident_device_bytes": executor.resident_device_bytes,
                "memory_before": executor.memory_before,
                "memory_after": executor.memory_after,
                "resident_expert_count": len(executor._experts),
            },
            "prepare": {
                "pass": prepare_pass,
                **prepare,
            },
            "correctness": {
                "relative_error_gate": 3e-5,
                "maximum_boundary_relative_l2_error": maximum_relative_error,
                "routing_equality": all(
                    row["routing_equality"] for row in correctness
                ),
                "residual_rows_bit_exact": all(
                    row["residual_rows_bit_exact"] for row in correctness
                ),
                "state": state,
                "positions": correctness,
                "pass": correctness_pass,
            },
            "lifecycle": {
                "warm_delta": lifecycle_delta,
                "per_generation_deltas": per_call,
                "before": before,
                "immediate_after": immediate_after,
                "after": after,
                "post_generation_thread_observation": {
                    "duration_ms": 1_000,
                    "sample_interval_ms": 50,
                    "samples": 20,
                    "transitions": thread_observation,
                },
                "compute_thread_consistent": compute_thread_consistent,
                "compute_thread_native_id": compute_thread_native_id,
                "compute_thread_ident": compute_thread_ident,
                "pass": lifecycle_pass,
            },
            "benchmark": {
                "warmup_generations": 1,
                "retained_generations": len(records),
                "wall": _timing([float(record["wall_ms"]) for record in records]),
                "device": _timing(
                    [float(record["device_ms"]) for record in records]
                ),
                "wire": input_frames,
                "instrumentation_mode": "production-minimal plus lifecycle counters",
            },
            "production_batch": production_batch,
        }
        await close_session("retained")
        if layer == 0:
            steady_performance = await _measure_stage_zero_steady_performance(
                runtime,
                executor,
                connection_pool,
                inputs,
                open_session=open_session,
                close_session=close_session,
                topology_id=topology_id,
                cycle_id=cycle_id,
                post_quiescence_device_p50_ms=float(
                    result["benchmark"]["device"]["p50_ms"]
                ),
            )
            result["steady_performance"] = steady_performance
            result["status"] = (
                "PASS"
                if result["status"] == "PASS"
                and steady_performance["status"] == "PASS"
                else "FAIL"
            )
        return result
    finally:
        await runtime.close()


async def _benchmark_nonfinal_stages(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    final_regression: Path,
    cycle_id: str,
) -> dict[str, Any]:
    paths = [
        checkpoint,
        cuda_library,
        oracle_trace,
        oracle_routes,
        identity_manifest,
        final_regression,
    ]
    resolved = [path.expanduser().resolve() for path in paths]
    (
        checkpoint,
        cuda_library,
        oracle_trace,
        oracle_routes,
        identity_manifest,
        final_regression,
    ) = resolved
    final_receipt = json.loads(final_regression.read_text(encoding="utf-8"))
    final_pass = (
        final_receipt.get("status") == "PASS"
        and final_receipt.get("correctness", {}).get("pass") is True
        and final_receipt.get("lifecycle", {}).get("pass") is True
        and final_receipt.get("backend", {}).get("cuda_library_sha256")
        == _sha256_file(cuda_library)
    )
    stages = [
        await _run_stage(
            checkpoint,
            cuda_library,
            oracle_trace,
            oracle_routes,
            identity_manifest,
            layer=layer,
            cycle_id=cycle_id,
        )
        for layer in (1, 3)
    ]
    status = "PASS" if final_pass and all(row["status"] == "PASS" for row in stages) else "FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": status,
        "hypothesis": (
            "One production registered Kimi executor can retain representative KDA and "
            "Gated-MLA stages, emit the complete canonical boundary, preserve state, and "
            "execute without warm reconstruction."
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "model_revision": MODEL_REVISION,
            "content_fingerprint": MODEL_CONTENT_FINGERPRINT,
        },
        "backend": {
            "cuda_library": str(cuda_library),
            "cuda_library_sha256": _sha256_file(cuda_library),
            "cpu_mathematical_fallbacks": 0,
        },
        "final_stage_regression": {
            "path": str(final_regression),
            "sha256": _sha256_file(final_regression),
            "pass": final_pass,
        },
        "stages": stages,
        "inspection": {
            "canonical_boundary": "float32 [1,9,7168]",
            "materialized_stage_classes": ["KDA+MoE", "Gated_MLA+MoE"],
            "actual_bottleneck": (
                "pending retained measurements"
                if status != "PASS"
                else "one-layer-only production ownership and experiment-module base dependency"
            ),
        },
    }


def benchmark_persistent_nonfinal_stages(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    final_regression: Path,
    output_path: Path,
    *,
    cycle_id: str = "H014-026c",
) -> dict[str, Any]:
    result = asyncio.run(
        _benchmark_nonfinal_stages(
            checkpoint,
            cuda_library,
            oracle_trace,
            oracle_routes,
            identity_manifest,
            final_regression,
            cycle_id,
        )
    )
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return {
        "status": result["status"],
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "summary": {
            "layers": [row["layer"] for row in result["stages"]],
            "maximum_relative_l2_error": max(
                row["correctness"]["maximum_boundary_relative_l2_error"]
                for row in result["stages"]
            ),
            "all_routes_exact": all(
                row["correctness"]["routing_equality"] for row in result["stages"]
            ),
            "all_lifecycle_deltas_zero": all(
                row["lifecycle"]["pass"] for row in result["stages"]
            ),
        },
    }


async def _benchmark_stage_zero(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    prior_regression: Path,
    cycle_id: str,
) -> dict[str, Any]:
    paths = [
        checkpoint,
        cuda_library,
        oracle_trace,
        oracle_routes,
        identity_manifest,
        prior_regression,
    ]
    (
        checkpoint,
        cuda_library,
        oracle_trace,
        oracle_routes,
        identity_manifest,
        prior_regression,
    ) = [path.expanduser().resolve() for path in paths]
    regression = json.loads(prior_regression.read_text(encoding="utf-8"))
    regression_pass = (
        regression.get("status") == "PASS"
        and regression.get("backend", {}).get("cuda_library_sha256")
        == _sha256_file(cuda_library)
        and all(row.get("status") == "PASS" for row in regression.get("stages", []))
        and regression.get("final_stage_regression", {}).get("pass") is True
    )
    stage = await _run_stage(
        checkpoint,
        cuda_library,
        oracle_trace,
        oracle_routes,
        identity_manifest,
        layer=0,
        cycle_id=cycle_id,
    )
    status = "PASS" if regression_pass and stage["status"] == "PASS" else "FAIL"
    return {
        "schema_version": "experiment-014-k3-persistent-stage-zero-v2",
        "cycle_id": cycle_id,
        "status": status,
        "hypothesis": (
            "A registered stage-zero worker keeps embedding and dense layer-0 weights "
            "resident, executes successive token IDs on CUDA, emits the complete canonical "
            "boundary, advances state, and performs no warm reconstruction."
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "model_revision": MODEL_REVISION,
            "content_fingerprint": MODEL_CONTENT_FINGERPRINT,
        },
        "backend": {
            "cuda_library": str(cuda_library),
            "cuda_library_sha256": _sha256_file(cuda_library),
            "cpu_mathematical_fallbacks": 0,
        },
        "prior_stage_regression": {
            "path": str(prior_regression),
            "sha256": _sha256_file(prior_regression),
            "pass": regression_pass,
        },
        "stage": stage,
        "inspection": {
            "warm_path_weight_loading": 0,
            "warm_path_model_materialization": 0,
            "canonical_boundary": "float32 [1,9,7168]",
            "actual_bottleneck": (
                "pending retained measurement"
                if status != "PASS"
                else "production implementation still inherits experiment-module base code"
            ),
        },
    }


def benchmark_persistent_stage_zero(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    prior_regression: Path,
    output_path: Path,
    *,
    cycle_id: str = "H014-026e",
) -> dict[str, Any]:
    result = asyncio.run(
        _benchmark_stage_zero(
            checkpoint,
            cuda_library,
            oracle_trace,
            oracle_routes,
            identity_manifest,
            prior_regression,
            cycle_id,
        )
    )
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return {
        "status": result["status"],
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "summary": {
            "maximum_relative_l2_error": result["stage"]["correctness"][
                "maximum_boundary_relative_l2_error"
            ],
            "residual_rows_bit_exact": result["stage"]["correctness"][
                "residual_rows_bit_exact"
            ],
            "lifecycle_deltas_zero": result["stage"]["lifecycle"]["pass"],
            "resident_device_bytes": result["stage"]["load"][
                "resident_device_bytes"
            ],
        },
    }


__all__ = [
    "benchmark_persistent_nonfinal_stages",
    "benchmark_persistent_stage_zero",
]
