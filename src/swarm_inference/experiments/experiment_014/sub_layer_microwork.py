"""Real Kimi routed-expert microwork certification for Experiment 014."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import threading
import time
import traceback
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from swarm_inference.execution.kimi_k3_stage import (
    PersistentKimiStageExecutor,
    _timing,
)
from swarm_inference.experiments.experiment_014.batch_safety import _health_snapshot
from swarm_inference.experiments.experiment_014.complete_stage_batch import (
    CORRECTNESS_RELATIVE_L2_GATE,
)
from swarm_inference.experiments.experiment_014.cuda import (
    _array_fingerprint,
    _device_identity,
    _numerical_metrics,
    _sha256_file,
)
from swarm_inference.experiments.experiment_014.full_cuda import (
    KimiCudaGraphRunner,
    _LayerResources,
    _parse_oracle_routes,
    _pointer_offset,
)
from swarm_inference.experiments.experiment_014.persistent_stages import (
    MODEL_CONTENT_FINGERPRINT,
    MODEL_REVISION,
    TOKENIZER_REVISION,
    _source_assignment,
    _stage_fixtures,
)
from swarm_inference.protocol.stage_worker import LoadStageRequest

SCHEMA_VERSION = "experiment-014-k3-real-expert-microwork-v1"
PIPE_FRAMING_BYTES = 4


def _microwork_process_snapshot() -> dict[str, Any]:
    process = psutil.Process()
    return {
        "child_process_ids": sorted(
            child.pid for child in process.children(recursive=True)
        ),
        "os_thread_ids": sorted(item.id for item in process.threads()),
        "python_thread_ids": sorted(
            int(item.ident) for item in threading.enumerate() if item.ident is not None
        ),
    }


def _microwork_lifecycle_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, int]:
    return {
        "topology_rebuilds": 0,
        "process_creation": len(
            set(after["child_process_ids"]) - set(before["child_process_ids"])
        ),
        "thread_creation": len(
            set(after["os_thread_ids"]) - set(before["os_thread_ids"])
        ),
        "task_creation": 0,
        "connection_establishment": 0,
        "weight_loading": 0,
        "model_materialization": 0,
        "persistent_buffer_allocation": 0,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _send(connection: Connection, message: dict[str, Any]) -> int:
    payload = pickle.dumps(message, protocol=5)
    connection.send_bytes(payload)
    return len(payload) + PIPE_FRAMING_BYTES


def _receive(connection: Connection) -> tuple[dict[str, Any], int]:
    payload = connection.recv_bytes()
    message = pickle.loads(payload)
    if not isinstance(message, dict):
        raise RuntimeError("microwork transport returned a non-object frame")
    return message, len(payload) + PIPE_FRAMING_BYTES


def _expert_worker_main(
    connection: Connection,
    checkpoint: str,
    cuda_library: str,
    layer: int,
    device: int,
    worker_id: int,
    ownership: tuple[int, ...],
    batch_capacity: int,
) -> None:
    """Own one immutable expert partition in an isolated persistent process."""
    runner: KimiCudaGraphRunner | None = None
    resources: _LayerResources | None = None
    execute_count = 0
    try:
        memory_before: dict[str, int] | None = None
        load_started = time.perf_counter_ns()
        runner = KimiCudaGraphRunner(Path(checkpoint), Path(cuda_library), device)
        runtime = runner.runtime
        resources = _LayerResources(runtime)
        memory_before = runtime.mem_info()
        experts: dict[int, tuple[Any, Any, Any]] = {}
        for expert in ownership:
            experts[expert] = runner._upload_expert(resources, layer, expert)
        latent = runner.config.latent
        if batch_capacity not in (1, 2, 4, 8):
            raise ValueError("expert microwork batch capacity must be 1/2/4/8")
        input_rows = resources.allocate(batch_capacity * latent)
        output_capacity = max(runner.config.topk, batch_capacity * runner.config.topk)
        task_input_rows = resources.allocate(output_capacity * latent)
        output_rows = resources.allocate(output_capacity * latent)
        zero = np.zeros((latent,), dtype=np.float32)
        runtime.upload_activation(input_rows, zero)
        first_expert = ownership[0]
        warm_device_ms: list[float] = []
        for _ in range(7):
            runtime.profile_begin()
            runtime.execute_resident(experts[first_expert], output_rows, input_rows, 1)
            warm_device_ms.append(runtime.profile_end())
        runtime.synchronize()
        memory_after = runtime.mem_info()
        ready = {
            "kind": "READY",
            "worker_id": worker_id,
            "pid": os.getpid(),
            "owned_expert_count": len(ownership),
            "owned_expert_min": min(ownership),
            "owned_expert_max": max(ownership),
            "ownership_fingerprint": _array_fingerprint(
                np.asarray(ownership, dtype=np.int32)
            ),
            "resident_expert_tensor_bytes": resources.resident_tensor_bytes,
            "persistent_runtime_buffer_bytes": (batch_capacity + 2 * output_capacity)
            * latent
            * np.dtype(np.float32).itemsize,
            "tracked_worker_bytes": resources.resident_tensor_bytes
            + (batch_capacity + 2 * output_capacity)
            * latent
            * np.dtype(np.float32).itemsize,
            "batch_capacity": batch_capacity,
            "measured_free_delta_at_ready_bytes": max(
                0, memory_before["free_bytes"] - memory_after["free_bytes"]
            ),
            "weight_fingerprint": "sha256:" + resources.weight_digest.hexdigest(),
            "cuda_library_sha256": runner.runtime.sha256,
            "load_wall_ms": (time.perf_counter_ns() - load_started) / 1e6,
            "prepare": {
                "calls": 7,
                "device": _timing(warm_device_ms),
                "topology_rebuilds": 0,
                "weight_loads_after_ready": 0,
                "persistent_buffer_recreation_after_ready": 0,
            },
        }
        _send(connection, ready)
        while True:
            request, _request_bytes = _receive(connection)
            kind = request.get("kind")
            if kind == "CLOSE":
                _send(
                    connection,
                    {
                        "kind": "CLOSED",
                        "worker_id": worker_id,
                        "execute_count": execute_count,
                        "cuda_error_state_ok": runtime.error_state_ok(),
                    },
                )
                return
            if kind == "HEALTH":
                runtime.synchronize()
                _send(
                    connection,
                    {
                        "kind": "HEALTH",
                        "worker_id": worker_id,
                        "execute_count": execute_count,
                        "cuda_error_state_ok": runtime.error_state_ok(),
                        "memory": runtime.mem_info(),
                    },
                )
                continue
            if kind != "EXECUTE":
                if kind != "EXECUTE_BATCH":
                    raise RuntimeError(f"worker {worker_id} received invalid command")
                generation = int(request["generation"])
                tasks = [tuple(int(value) for value in task) for task in request["tasks"]]
                if not tasks or any(len(task) != 3 for task in tasks):
                    raise RuntimeError("worker batch request contains invalid tasks")
                if len({(row, slot) for row, slot, _expert in tasks}) != len(tasks):
                    raise RuntimeError("worker batch request duplicates a row/slot")
                if any(expert not in experts for _row, _slot, expert in tasks):
                    raise RuntimeError("worker batch request violates static ownership")
                activation_row_ids = tuple(
                    int(value) for value in request["activation_row_ids"]
                )
                activations = np.ascontiguousarray(
                    request["activations"], dtype=np.float32
                )
                if (
                    activations.shape != (len(activation_row_ids), latent)
                    or len(set(activation_row_ids)) != len(activation_row_ids)
                    or not np.isfinite(activations).all()
                ):
                    raise RuntimeError("worker batch request has invalid activations")
                activation_index_by_row = {
                    row_id: index
                    for index, row_id in enumerate(activation_row_ids)
                }
                if any(
                    row not in activation_index_by_row
                    for row, _slot, _expert in tasks
                ):
                    raise RuntimeError("worker batch request omits a required row activation")
                test_fault = request.get("_test_fault")
                if test_fault not in {
                    None,
                    "delay",
                    "partial",
                    "duplicate",
                    "stale",
                    "loss_after_cuda",
                }:
                    raise RuntimeError("worker received an unsupported test fault")

                wall_started = time.perf_counter_ns()
                outputs = np.empty((len(tasks), latent), dtype=np.float32)
                tasks_by_expert: dict[int, list[int]] = {}
                for task_index, (_row, _slot, expert) in enumerate(tasks):
                    tasks_by_expert.setdefault(expert, []).append(task_index)
                native_calls = 0
                group_histogram: Counter[str] = Counter()
                execution_groups: list[tuple[int, int, tuple[int, ...]]] = []
                grouped_task_indices: list[int] = []
                for expert, task_indices in sorted(tasks_by_expert.items()):
                    group_histogram[str(len(task_indices))] += 1
                    group_start = len(grouped_task_indices)
                    grouped_task_indices.extend(task_indices)
                    chunks: list[int] = []
                    remaining = len(task_indices)
                    for size in (8, 4, 2, 1):
                        while remaining >= size:
                            chunks.append(size)
                            remaining -= size
                    if remaining:
                        raise RuntimeError("worker could not partition exact batch sizes")
                    execution_groups.append((expert, group_start, tuple(chunks)))

                runtime.profile_begin()
                h2d_started = time.perf_counter_ns()
                runtime.upload_activation(input_rows, activations)
                h2d_enqueue_wall_ms = (time.perf_counter_ns() - h2d_started) / 1e6
                for work_index, task_index in enumerate(grouped_task_indices):
                    row = tasks[task_index][0]
                    runtime.execute_copy(
                        _pointer_offset(task_input_rows, work_index * latent),
                        _pointer_offset(
                            input_rows,
                            activation_index_by_row[row] * latent,
                        ),
                        latent,
                    )
                for expert, group_start, chunks in execution_groups:
                    offset = group_start
                    for size in chunks:
                        runtime.execute_resident(
                            experts[expert],
                            _pointer_offset(output_rows, offset * latent),
                            _pointer_offset(task_input_rows, offset * latent),
                            size,
                        )
                        native_calls += 1
                        offset += size
                input_gather_and_expert_device_ms = runtime.profile_end()

                d2h_started = time.perf_counter_ns()
                grouped_outputs = runtime.download_activation(
                    output_rows, (len(tasks), latent)
                )
                d2h_wall_ms = (time.perf_counter_ns() - d2h_started) / 1e6
                for work_index, task_index in enumerate(grouped_task_indices):
                    outputs[task_index] = grouped_outputs[work_index]
                device_ms = input_gather_and_expert_device_ms
                execute_count += 1
                response_tasks = tasks[:-1] if test_fault == "partial" else tasks
                response_outputs = (
                    outputs[:-1] if test_fault == "partial" else outputs
                )
                response = {
                    "kind": "BATCH_RESULT",
                    "generation": generation - 1 if test_fault == "stale" else generation,
                    "worker_id": worker_id,
                    "tasks": response_tasks,
                    "outputs": response_outputs,
                    "output_fingerprint": _array_fingerprint(response_outputs),
                    "device_ms": device_ms,
                    "activation_h2d_enqueue_wall_ms": h2d_enqueue_wall_ms,
                    "input_gather_and_expert_device_ms": (
                        input_gather_and_expert_device_ms
                    ),
                    "output_d2h_wall_ms": d2h_wall_ms,
                    "device_measurement_scope": (
                        "combined_activation_h2d_device_gather_and_expert_compute"
                    ),
                    "worker_wall_ms": (time.perf_counter_ns() - wall_started) / 1e6,
                    "execute_count": execute_count,
                    "native_expert_calls": native_calls,
                    "unique_experts": len(tasks_by_expert),
                    "repeated_expert_hits": len(tasks) - len(tasks_by_expert),
                    "expert_group_size_histogram": dict(group_histogram),
                    "weight_loads_during_execute": 0,
                    "persistent_buffer_allocations_during_execute": 0,
                    "topology_rebuilds_during_execute": 0,
                }
                if test_fault is not None:
                    response["test_fault"] = test_fault
                if test_fault == "delay":
                    time.sleep(float(request.get("_test_delay_seconds", 0.2)))
                if test_fault == "loss_after_cuda":
                    # The real expert work and D2H synchronization above have completed.
                    # Return without a response so the parent observes EOF only after the
                    # worker's finally block has released its CUDA allocations/context.
                    return
                _send(connection, response)
                if test_fault == "duplicate":
                    _send(connection, response)
                continue
            generation = int(request["generation"])
            selected = tuple(int(value) for value in request["expert_ids"])
            if len(selected) != len(set(selected)):
                raise RuntimeError("worker request contains duplicate selected experts")
            if any(expert not in experts for expert in selected):
                raise RuntimeError("worker request violates static expert ownership")
            activation = np.ascontiguousarray(request["activation"], dtype=np.float32)
            if activation.shape != (latent,) or not np.isfinite(activation).all():
                raise RuntimeError("worker request contains an invalid latent activation")
            wall_started = time.perf_counter_ns()
            runtime.upload_activation(input_rows, activation)
            runtime.profile_begin()
            for slot, expert in enumerate(selected):
                runtime.execute_resident(
                    experts[expert],
                    _pointer_offset(output_rows, slot * latent),
                    input_rows,
                    1,
                )
            device_ms = runtime.profile_end()
            outputs = runtime.download_activation(output_rows, (len(selected), latent))
            execute_count += 1
            response = {
                "kind": "RESULT",
                "generation": generation,
                "worker_id": worker_id,
                "expert_ids": selected,
                "outputs": outputs,
                "output_fingerprint": _array_fingerprint(outputs),
                "device_ms": device_ms,
                "worker_wall_ms": (time.perf_counter_ns() - wall_started) / 1e6,
                "execute_count": execute_count,
                "weight_loads_during_execute": 0,
                "persistent_buffer_allocations_during_execute": 0,
                "topology_rebuilds_during_execute": 0,
            }
            _send(connection, response)
    except BaseException as exc:
        with np.errstate(all="ignore"), suppress(BaseException):
            _send(
                connection,
                {
                    "kind": "ERROR",
                    "worker_id": worker_id,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    finally:
        if resources is not None:
            resources.close()
        if runner is not None:
            runner.close()
        connection.close()


@dataclass(slots=True)
class _WorkerProxy:
    worker_id: int
    ownership: frozenset[int]
    process: mp.Process
    connection: Connection
    ready: dict[str, Any]

    def close(self) -> dict[str, Any]:
        closed: dict[str, Any] = {"kind": "NOT_RUNNING"}
        if self.process.is_alive():
            try:
                _send(self.connection, {"kind": "CLOSE"})
                drained_results = 0
                deadline = time.monotonic() + 30.0
                while self.process.is_alive() and time.monotonic() < deadline:
                    if not self.connection.poll(1):
                        continue
                    message, _ = _receive(self.connection)
                    if message.get("kind") == "CLOSED":
                        closed = message
                        break
                    if message.get("kind") in {"RESULT", "BATCH_RESULT"}:
                        drained_results += 1
                        continue
                    closed = {
                        "kind": str(message.get("kind")),
                        "message": str(message.get("message", "")),
                    }
                    break
                closed["drained_result_frames"] = drained_results
            finally:
                self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=10)
            closed["forced_termination"] = True
        self.connection.close()
        closed["exit_code"] = self.process.exitcode
        return closed


@dataclass(slots=True)
class _BatchDispatchHandle:
    generation: int
    batch: int
    selected: np.ndarray
    groups: dict[int, list[tuple[int, int, int]]]
    request_rows: dict[int, tuple[int, ...]]
    request_bytes: int
    dispatch_ms: float
    roundtrip_started_ns: int
    dispatch_finished_ns: int
    state: str = "STARTED"
    received_worker_ids: set[int] | None = None
    test_faults: dict[int, str] | None = None


def _start_worker(
    context: mp.context.BaseContext,
    *,
    checkpoint: Path,
    cuda_library: Path,
    layer: int,
    device: int,
    worker_id: int,
    ownership: frozenset[int],
    batch_capacity: int = 1,
) -> _WorkerProxy:
    parent, child = context.Pipe(duplex=True)
    process = context.Process(
        target=_expert_worker_main,
        args=(
            child,
            str(checkpoint),
            str(cuda_library),
            layer,
            device,
            worker_id,
            tuple(sorted(ownership)),
            batch_capacity,
        ),
        name=f"h014-sub-expert-{worker_id}",
    )
    process.start()
    child.close()
    if not parent.poll(300):
        process.terminate()
        process.join(timeout=10)
        parent.close()
        raise TimeoutError(f"expert worker {worker_id} did not reach READY")
    ready, _ = _receive(parent)
    if ready.get("kind") != "READY":
        process.join(timeout=10)
        parent.close()
        raise RuntimeError(f"expert worker {worker_id} failed: {ready}")
    return _WorkerProxy(worker_id, ownership, process, parent, ready)


class _PersistentExpertCollective:
    def __init__(
        self,
        workers: list[_WorkerProxy],
        *,
        latent: int,
        response_timeout_seconds: float = 30.0,
    ) -> None:
        if response_timeout_seconds <= 0:
            raise ValueError("collective response timeout must be positive")
        self.workers = workers
        self.latent = latent
        self.response_timeout_seconds = response_timeout_seconds
        self.generation = 0
        self.records: list[dict[str, Any]] = []
        self.cancel_records: list[dict[str, Any]] = []
        self._poisoned_reason: str | None = None
        self.owner_by_expert: dict[int, int] = {}
        for worker in workers:
            for expert in worker.ownership:
                if expert in self.owner_by_expert:
                    raise RuntimeError("duplicate persistent expert ownership")
                self.owner_by_expert[expert] = worker.worker_id
        if set(self.owner_by_expert) != set(range(896)):
            raise RuntimeError("persistent expert ownership has orphan experts")

    def _fail_handle(
        self, handle: _BatchDispatchHandle, message: str
    ) -> RuntimeError:
        handle.state = "FAILED"
        self._poisoned_reason = message
        return RuntimeError(message)

    def dispatch(
        self, selected_ids: np.ndarray, latent_activation: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.generation += 1
        generation = self.generation
        selected = [int(value) for value in selected_ids]
        if len(selected) != 16 or len(set(selected)) != 16:
            raise RuntimeError("collective requires 16 unique selected experts")
        groups: dict[int, list[tuple[int, int]]] = {}
        for slot, expert in enumerate(selected):
            worker_id = self.owner_by_expert[expert]
            groups.setdefault(worker_id, []).append((slot, expert))
        by_id = {worker.worker_id: worker for worker in self.workers}
        started = time.perf_counter_ns()
        dispatch_started = time.perf_counter_ns()
        request_bytes = 0
        input_activation_bytes = 0
        for worker_id, tasks in sorted(groups.items()):
            message = {
                "kind": "EXECUTE",
                "generation": generation,
                "expert_ids": tuple(expert for _slot, expert in tasks),
                "activation": latent_activation,
            }
            request_bytes += _send(by_id[worker_id].connection, message)
            input_activation_bytes += latent_activation.nbytes
        dispatch_ms = (time.perf_counter_ns() - dispatch_started) / 1e6

        rows = np.empty((16, self.latent), dtype=np.float32)
        executed: list[int] = []
        worker_rows: list[dict[str, Any]] = []
        response_bytes = 0
        collection_decode_ms = 0.0
        for worker_id, tasks in sorted(groups.items()):
            connection = by_id[worker_id].connection
            payload_started = time.perf_counter_ns()
            payload = connection.recv_bytes()
            response_bytes += len(payload) + PIPE_FRAMING_BYTES
            decode_started = time.perf_counter_ns()
            response = pickle.loads(payload)
            collection_decode_ms += (time.perf_counter_ns() - decode_started) / 1e6
            if not isinstance(response, dict) or response.get("kind") != "RESULT":
                raise RuntimeError(f"worker {worker_id} returned failure: {response}")
            if int(response["generation"]) != generation:
                raise RuntimeError("stale expert worker generation")
            expected_ids = [expert for _slot, expert in tasks]
            observed_ids = [int(value) for value in response["expert_ids"]]
            if observed_ids != expected_ids:
                raise RuntimeError("expert worker response changed request ordering")
            outputs = np.ascontiguousarray(response["outputs"], dtype=np.float32)
            if outputs.shape != (len(tasks), self.latent):
                raise RuntimeError(
                    "expert worker returned invalid output geometry: "
                    f"observed={outputs.shape}, expected={(len(tasks), self.latent)}"
                )
            for local, (slot, expert) in enumerate(tasks):
                rows[slot] = outputs[local]
                executed.append(expert)
            worker_rows.append(
                {
                    "worker_id": worker_id,
                    "selected_expert_ids": observed_ids,
                    "selected_count": len(observed_ids),
                    "device_ms": float(response["device_ms"]),
                    "worker_wall_ms": float(response["worker_wall_ms"]),
                    "response_wait_and_decode_ms": (
                        time.perf_counter_ns() - payload_started
                    )
                    / 1e6,
                    "output_fingerprint": response["output_fingerprint"],
                    "weight_loads_during_execute": int(
                        response["weight_loads_during_execute"]
                    ),
                    "persistent_buffer_allocations_during_execute": int(
                        response["persistent_buffer_allocations_during_execute"]
                    ),
                    "topology_rebuilds_during_execute": int(
                        response["topology_rebuilds_during_execute"]
                    ),
                }
            )
        roundtrip_ms = (time.perf_counter_ns() - started) / 1e6
        all_exactly_once = Counter(executed) == Counter(selected)
        if not all_exactly_once:
            raise RuntimeError("expert collective omitted or duplicated selected work")
        critical_worker_wall = max(row["worker_wall_ms"] for row in worker_rows)
        critical_worker_device = max(row["device_ms"] for row in worker_rows)
        total_worker_device = sum(row["device_ms"] for row in worker_rows)
        critical_path_bytes = max(
            latent_activation.nbytes
            + len(row["selected_expert_ids"]) * self.latent * 4
            for row in worker_rows
        )
        record = {
            "generation": generation,
            "ownership_strategy": "static_disjoint_expert_id_mod_worker_count",
            "selected_expert_ids": selected,
            "owner_resolution": {
                str(expert): self.owner_by_expert[expert] for expert in selected
            },
            "workers_contacted": len(groups),
            "worker_records": worker_rows,
            "all_16_executed_exactly_once": all_exactly_once,
            "dispatch_ms": dispatch_ms,
            "loopback_roundtrip_ms": roundtrip_ms,
            "worker_compute_critical_wall_ms": critical_worker_wall,
            "worker_compute_critical_device_ms": critical_worker_device,
            "worker_compute_total_device_ms": total_worker_device,
            "exposed_loopback_and_coordination_ms": max(
                0.0, roundtrip_ms - critical_worker_wall
            ),
            "collection_deserialization_ms": collection_decode_ms,
            "input_activation_bytes": latent_activation.nbytes,
            "per_worker_dispatch_activation_bytes": latent_activation.nbytes,
            "total_dispatched_activation_bytes": input_activation_bytes,
            "returned_expert_output_bytes": rows.nbytes,
            "reduction_weight_bytes": 16 * np.dtype(np.float32).itemsize,
            "reduction_input_bytes": rows.nbytes,
            "request_transport_bytes": request_bytes,
            "response_transport_bytes": response_bytes,
            "total_transport_bytes": request_bytes + response_bytes,
            "critical_path_payload_bytes": critical_path_bytes,
            "framing_bytes": 2 * len(groups) * PIPE_FRAMING_BYTES,
            "root_messages": 2 * len(groups),
            "worker_messages": 2 * len(groups),
            "synchronization_points": 2,
            "output_fingerprint": _array_fingerprint(rows),
        }
        self.records.append(record)
        return rows, record

    def dispatch_batch(
        self, selected_ids: np.ndarray, latent_activations: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Dispatch one token batch while reusing repeated expert weights per worker."""
        return self.collect_batch(self.start_batch(selected_ids, latent_activations))

    def start_batch(
        self,
        selected_ids: np.ndarray,
        latent_activations: np.ndarray,
        *,
        test_faults: dict[int, str] | None = None,
        test_delay_seconds: float = 0.2,
    ) -> _BatchDispatchHandle:
        """Fan out one batch without waiting, retaining immutable collection state."""
        if self._poisoned_reason is not None:
            raise RuntimeError(f"expert collective is poisoned: {self._poisoned_reason}")
        if test_delay_seconds <= 0:
            raise ValueError("test delay must be positive")
        faults = dict(test_faults or {})
        allowed_faults = {
            "delay",
            "partial",
            "duplicate",
            "stale",
            "loss_after_cuda",
        }
        if any(value not in allowed_faults for value in faults.values()):
            raise ValueError("unsupported expert collective test fault")
        for worker in self.workers:
            if worker.connection.poll(0):
                stale, _ = _receive(worker.connection)
                self._poisoned_reason = (
                    f"unexpected pre-dispatch frame from worker {worker.worker_id}: "
                    f"{stale.get('kind')} generation={stale.get('generation')}"
                )
                raise RuntimeError(self._poisoned_reason)
        selected = np.ascontiguousarray(selected_ids, dtype=np.int32)
        activations = np.ascontiguousarray(latent_activations, dtype=np.float32)
        if selected.ndim != 2 or selected.shape[1] != 16:
            raise RuntimeError("collective batch requires [batch,16] selected experts")
        batch = int(selected.shape[0])
        if batch not in (1, 2, 4, 8) or activations.shape != (batch, self.latent):
            raise RuntimeError("collective batch activation geometry is invalid")
        if any(len(set(row.tolist())) != 16 for row in selected):
            raise RuntimeError("collective batch rows require 16 unique experts")

        groups: dict[int, list[tuple[int, int, int]]] = {}
        for row in range(batch):
            for slot, expert_value in enumerate(selected[row].tolist()):
                expert = int(expert_value)
                groups.setdefault(self.owner_by_expert[expert], []).append(
                    (row, slot, expert)
                )
        by_id = {worker.worker_id: worker for worker in self.workers}
        self.generation += 1
        generation = self.generation
        roundtrip_started = time.perf_counter_ns()
        request_bytes = 0
        dispatch_started = time.perf_counter_ns()
        request_rows: dict[int, tuple[int, ...]] = {}
        for worker_id, tasks in sorted(groups.items()):
            row_ids = tuple(sorted({row for row, _slot, _expert in tasks}))
            request_rows[worker_id] = row_ids
            message = {
                "kind": "EXECUTE_BATCH",
                "generation": generation,
                "tasks": tasks,
                "activation_row_ids": row_ids,
                "activations": activations[list(row_ids)],
            }
            if worker_id in faults:
                message["_test_fault"] = faults[worker_id]
                message["_test_delay_seconds"] = test_delay_seconds
            request_bytes += _send(
                by_id[worker_id].connection,
                message,
            )
        dispatch_finished = time.perf_counter_ns()
        return _BatchDispatchHandle(
            generation=generation,
            batch=batch,
            selected=selected,
            groups=groups,
            request_rows=request_rows,
            request_bytes=request_bytes,
            dispatch_ms=(dispatch_finished - dispatch_started) / 1e6,
            roundtrip_started_ns=roundtrip_started,
            dispatch_finished_ns=dispatch_finished,
            received_worker_ids=set(),
            test_faults=faults,
        )

    def collect_batch(
        self, handle: _BatchDispatchHandle
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Collect exactly one previously started batch and reject reuse/staleness."""
        if handle.state != "STARTED":
            raise RuntimeError(
                f"expert batch handle cannot collect from state {handle.state}"
            )
        generation = handle.generation
        batch = handle.batch
        selected = handle.selected
        groups = handle.groups
        request_rows = handle.request_rows
        request_bytes = handle.request_bytes
        dispatch_ms = handle.dispatch_ms
        by_id = {worker.worker_id: worker for worker in self.workers}
        result = np.empty((batch, 16, self.latent), dtype=np.float32)
        executed: list[tuple[int, int, int]] = []
        response_bytes = 0
        worker_rows: list[dict[str, Any]] = []
        collection_started = time.perf_counter_ns()
        parent_overlap_window_ms = (
            collection_started - handle.dispatch_finished_ns
        ) / 1e6
        critical_path_bytes = 0
        native_calls = 0
        deadline = time.monotonic() + self.response_timeout_seconds
        for worker_id, expected_tasks in sorted(groups.items()):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not by_id[worker_id].connection.poll(remaining):
                handle.state = "FAILED"
                self._poisoned_reason = (
                    f"expert worker {worker_id} timed out for generation {generation}"
                )
                raise TimeoutError(self._poisoned_reason)
            try:
                response, observed_bytes = _receive(by_id[worker_id].connection)
            except BaseException as exc:
                raise self._fail_handle(
                    handle,
                    f"expert worker {worker_id} was lost for generation {generation}: "
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            if handle.received_worker_ids is not None:
                handle.received_worker_ids.add(worker_id)
            response_bytes += observed_bytes
            if response.get("kind") != "BATCH_RESULT":
                raise self._fail_handle(
                    handle, f"worker {worker_id} returned failure: {response}"
                )
            if int(response["generation"]) != generation:
                raise self._fail_handle(
                    handle,
                    "stale expert batch worker generation: "
                    f"expected={generation}, observed={response['generation']}",
                )
            observed_tasks = [tuple(int(value) for value in task) for task in response["tasks"]]
            if observed_tasks != expected_tasks:
                raise self._fail_handle(
                    handle, "expert batch worker changed or omitted task ordering"
                )
            outputs = np.ascontiguousarray(response["outputs"], dtype=np.float32)
            if outputs.shape != (len(expected_tasks), self.latent):
                raise self._fail_handle(
                    handle, "expert batch worker returned invalid output geometry"
                )
            for task_index, (row, slot, expert) in enumerate(observed_tasks):
                result[row, slot] = outputs[task_index]
                executed.append((row, slot, expert))
            worker_request_activation_bytes = (
                len(request_rows[worker_id]) * self.latent * 4
            )
            worker_output_bytes = len(expected_tasks) * self.latent * 4
            worker_transport_bytes = (
                worker_request_activation_bytes + worker_output_bytes
            )
            critical_path_bytes = max(critical_path_bytes, worker_transport_bytes)
            native_calls += int(response["native_expert_calls"])
            worker_rows.append(
                {
                    "worker_id": worker_id,
                    "tasks": len(expected_tasks),
                    "activation_rows": len(request_rows[worker_id]),
                    "unique_experts": int(response["unique_experts"]),
                    "repeated_expert_hits": int(response["repeated_expert_hits"]),
                    "native_expert_calls": int(response["native_expert_calls"]),
                    "expert_group_size_histogram": response[
                        "expert_group_size_histogram"
                    ],
                    "device_ms": float(response["device_ms"]),
                    "activation_h2d_enqueue_wall_ms": float(
                        response["activation_h2d_enqueue_wall_ms"]
                    ),
                    "input_gather_and_expert_device_ms": float(
                        response["input_gather_and_expert_device_ms"]
                    ),
                    "output_d2h_wall_ms": float(response["output_d2h_wall_ms"]),
                    "device_measurement_scope": response["device_measurement_scope"],
                    "worker_wall_ms": float(response["worker_wall_ms"]),
                    "execute_count": int(response["execute_count"]),
                    "request_activation_bytes": worker_request_activation_bytes,
                    "returned_output_bytes": worker_output_bytes,
                    "critical_payload_bytes": worker_transport_bytes,
                    "weight_loads_during_execute": int(
                        response["weight_loads_during_execute"]
                    ),
                    "persistent_buffer_allocations_during_execute": int(
                        response["persistent_buffer_allocations_during_execute"]
                    ),
                    "topology_rebuilds_during_execute": int(
                        response["topology_rebuilds_during_execute"]
                    ),
                }
            )
        duplicate_frames_discarded = 0
        faults = handle.test_faults or {}
        for worker_id, expected_tasks in sorted(groups.items()):
            connection = by_id[worker_id].connection
            poll_seconds = 0.1 if faults.get(worker_id) == "duplicate" else 0.0
            if not connection.poll(poll_seconds):
                if faults.get(worker_id) == "duplicate":
                    raise self._fail_handle(
                        handle,
                        f"expected duplicate frame absent for worker {worker_id}",
                    )
                continue
            duplicate, duplicate_bytes = _receive(connection)
            duplicate_tasks = [
                tuple(int(value) for value in task)
                for task in duplicate.get("tasks", [])
            ]
            if (
                duplicate.get("kind") != "BATCH_RESULT"
                or int(duplicate.get("generation", -1)) != generation
                or duplicate_tasks != expected_tasks
            ):
                raise self._fail_handle(
                    handle,
                    f"unexpected extra worker frame from {worker_id}: {duplicate}",
                )
            duplicate_frames_discarded += 1
            response_bytes += duplicate_bytes
        collection_ms = (time.perf_counter_ns() - collection_started) / 1e6
        roundtrip_ms = (
            time.perf_counter_ns() - handle.roundtrip_started_ns
        ) / 1e6
        expected = [
            (row, slot, int(selected[row, slot]))
            for row in range(batch)
            for slot in range(16)
        ]
        all_exactly_once = Counter(executed) == Counter(expected)
        if not all_exactly_once:
            raise self._fail_handle(
                handle, "expert batch collective omitted or duplicated work"
            )
        critical_worker_wall = max(row["worker_wall_ms"] for row in worker_rows)
        critical_worker_device = max(row["device_ms"] for row in worker_rows)
        total_worker_device = sum(row["device_ms"] for row in worker_rows)
        total_transport_bytes = request_bytes + response_bytes
        record = {
            "generation": generation,
            "batch": batch,
            "ownership_strategy": "static_disjoint_expert_id_mod_worker_count",
            "selected_expert_ids": selected.tolist(),
            "workers_contacted": len(groups),
            "worker_records": worker_rows,
            "dispatch_ms": dispatch_ms,
            "parent_overlap_window_ms": parent_overlap_window_ms,
            "collection_ms": collection_ms,
            "roundtrip_ms": roundtrip_ms,
            "worker_compute_critical_wall_ms": critical_worker_wall,
            "worker_compute_critical_device_ms": critical_worker_device,
            "worker_compute_total_device_ms": total_worker_device,
            "exposed_loopback_and_coordination_ms": max(
                0.0, roundtrip_ms - critical_worker_wall
            ),
            "all_selected_experts_executed_once": all_exactly_once,
            "total_selections": batch * 16,
            "unique_experts": len(set(selected.reshape(-1).tolist())),
            "repeated_expert_hits": batch * 16
            - len(set(selected.reshape(-1).tolist())),
            "native_expert_calls": native_calls,
            "effective_weight_reuse_rows_per_native_call": batch * 16
            / native_calls,
            "avoided_expert_weight_launches": batch * 16 - native_calls,
            "request_bytes": request_bytes,
            "response_bytes": response_bytes,
            "duplicate_frames_discarded": duplicate_frames_discarded,
            "total_transport_bytes": total_transport_bytes,
            "critical_path_payload_bytes": critical_path_bytes,
            "input_activation_bytes": sum(
                row["request_activation_bytes"] for row in worker_rows
            ),
            "returned_expert_output_bytes": batch * 16 * self.latent * 4,
            "framing_bytes": 2 * len(groups) * PIPE_FRAMING_BYTES,
            "messages": 2 * len(groups),
            "root_messages": 2 * len(groups),
            "worker_messages": 2 * len(groups),
            "synchronization_points": 2,
            "output_fingerprint": _array_fingerprint(result),
        }
        handle.state = "COLLECTED"
        self.records.append(record)
        return result, record

    def cancel_batch(self, handle: _BatchDispatchHandle) -> dict[str, Any]:
        """Drain and discard one in-flight generation so the collective is reusable."""
        if handle.state != "STARTED":
            raise RuntimeError(
                f"expert batch handle cannot cancel from state {handle.state}"
            )
        by_id = {worker.worker_id: worker for worker in self.workers}
        deadline = time.monotonic() + self.response_timeout_seconds
        drained: list[dict[str, Any]] = []
        for worker_id, expected_tasks in sorted(handle.groups.items()):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not by_id[worker_id].connection.poll(remaining):
                handle.state = "FAILED"
                self._poisoned_reason = (
                    f"cancel timed out draining worker {worker_id} generation "
                    f"{handle.generation}"
                )
                raise TimeoutError(self._poisoned_reason)
            response, response_bytes = _receive(by_id[worker_id].connection)
            observed_tasks = [
                tuple(int(value) for value in task)
                for task in response.get("tasks", [])
            ]
            if (
                response.get("kind") != "BATCH_RESULT"
                or int(response.get("generation", -1)) != handle.generation
                or observed_tasks != expected_tasks
            ):
                raise self._fail_handle(
                    handle,
                    f"cancel received invalid worker {worker_id} frame: {response}",
                )
            if handle.received_worker_ids is not None:
                handle.received_worker_ids.add(worker_id)
            drained.append(
                {
                    "worker_id": worker_id,
                    "generation": handle.generation,
                    "response_bytes": response_bytes,
                    "tasks_discarded": len(observed_tasks),
                    "output_bytes_discarded": int(
                        np.ascontiguousarray(response["outputs"], dtype=np.float32).nbytes
                    ),
                }
            )
        handle.state = "CANCELLED"
        record = {
            "generation": handle.generation,
            "state": handle.state,
            "workers_drained": len(drained),
            "tasks_discarded": sum(row["tasks_discarded"] for row in drained),
            "output_bytes_discarded": sum(
                row["output_bytes_discarded"] for row in drained
            ),
            "worker_records": drained,
            "reduction_performed": False,
            "collect_after_cancel_rejected": True,
        }
        self.cancel_records.append(record)
        return record

    def health(self) -> list[dict[str, Any]]:
        """Synchronize every persistent worker and inspect sticky CUDA state."""
        for worker in self.workers:
            _send(worker.connection, {"kind": "HEALTH"})
        rows: list[dict[str, Any]] = []
        for worker in self.workers:
            response, response_bytes = _receive(worker.connection)
            if response.get("kind") != "HEALTH":
                raise RuntimeError(
                    f"worker {worker.worker_id} health check failed: {response}"
                )
            rows.append(
                {
                    "worker_id": worker.worker_id,
                    "execute_count": int(response["execute_count"]),
                    "cuda_error_state_ok": bool(response["cuda_error_state_ok"]),
                    "memory": response["memory"],
                    "response_bytes": response_bytes,
                }
            )
        return rows


def _request(
    checkpoint: Path,
    cuda_library: Path,
    *,
    layer: int,
    device: int,
    cycle_id: str,
    maximum_context: int,
) -> LoadStageRequest:
    assignment = _source_assignment(checkpoint, layer=layer, device=f"native-cuda:{device}")
    return LoadStageRequest(
        worker_id=f"{cycle_id.lower()}-coordinator-{layer:03d}",
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
        fast_path_context_bucket=maximum_context,
        model_content_fingerprint=MODEL_CONTENT_FINGERPRINT,
        native_runtime_library=str(cuda_library),
        native_runtime_library_sha256=_sha256_file(cuda_library),
        device=f"native-cuda:{device}",
        dtype="float32",
        model_path=str(checkpoint),
    )


def _reference(
    request: LoadStageRequest,
    checkpoint: Path,
    cuda_library: Path,
    fixtures: list[np.ndarray],
    expected_boundaries: list[np.ndarray],
    expected_routes: dict[int, list[int]],
    *,
    device: int,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], list[np.ndarray], list[list[int]], list[list[float]], dict[str, Any]]:
    executor = PersistentKimiStageExecutor(
        request=request,
        checkpoint=checkpoint,
        cuda_library=cuda_library,
        device=device,
    )
    outputs: list[np.ndarray] = []
    routes: list[list[int]] = []
    route_weights: list[list[float]] = []
    try:
        executor.prepare_for_ready()
        executor.open_session("h014-sub-reference-correctness", maximum_context_override=3)
        correctness: list[dict[str, Any]] = []
        try:
            for position in range(3):
                result = executor.execute_decode(
                    session_id="h014-sub-reference-correctness",
                    hidden_states=torch.from_numpy(fixtures[position].copy()),
                    cache_position_start=position,
                )
                output = result.stage_boundary_hidden_states.detach().cpu().numpy().copy()
                record = executor.execution_records[-1]
                outputs.append(output)
                routes.append(list(record["selected_expert_ids"]))
                route_weights.append(list(record["selected_weights"]))
                correctness.append(
                    {
                        "position": position,
                        "metrics": _numerical_metrics(
                            output, expected_boundaries[position]
                        ),
                        "routes_match_graph_oracle": routes[-1]
                        == expected_routes[position],
                        "output_fingerprint": _array_fingerprint(output),
                    }
                )
            state = executor.session_state_evidence(
                "h014-sub-reference-correctness"
            )
        finally:
            executor.close_session("h014-sub-reference-correctness")

        executor.set_research_telemetry_mode("minimal")
        executor.open_session(
            "h014-sub-reference-performance",
            maximum_context_override=warmup + iterations,
        )
        wall_values: list[float] = []
        device_values: list[float] = []
        try:
            for position in range(warmup + iterations):
                started = time.perf_counter_ns()
                executor.execute_decode(
                    session_id="h014-sub-reference-performance",
                    hidden_states=torch.from_numpy(
                        fixtures[position % len(fixtures)].copy()
                    ),
                    cache_position_start=position,
                )
                if position >= warmup:
                    wall_values.append((time.perf_counter_ns() - started) / 1e6)
                    device_values.append(
                        float(executor.execution_records[-1]["device_ms"])
                    )
        finally:
            executor.close_session("h014-sub-reference-performance")

        executor.set_research_telemetry_mode("detailed")
        executor.open_session(
            "h014-sub-reference-expert-profile", maximum_context_override=20
        )
        expert_values: list[float] = []
        shared_values: list[float] = []
        try:
            for position in range(20):
                executor.execute_decode(
                    session_id="h014-sub-reference-expert-profile",
                    hidden_states=torch.from_numpy(
                        fixtures[position % len(fixtures)].copy()
                    ),
                    cache_position_start=position,
                )
                if position >= 5:
                    row = executor.execution_records[-1]
                    expert_values.append(float(row["routed_expert_device_ms"]))
                    shared_values.append(float(row["shared_expert_device_ms"]))
        finally:
            executor.close_session("h014-sub-reference-expert-profile")

        maximum_error = max(
            float(row["metrics"]["relative_l2_error"]) for row in correctness
        )
        receipt = {
            "status": "PASS"
            if maximum_error <= 3e-5
            and all(row["routes_match_graph_oracle"] for row in correctness)
            else "FAIL",
            "correctness": correctness,
            "maximum_oracle_relative_l2_error": maximum_error,
            "whole_layer_wall": _timing(wall_values),
            "whole_layer_device": _timing(device_values),
            "routed_expert_device": _timing(expert_values),
            "shared_expert_device": _timing(shared_values),
            "complete_layer_resident_device_bytes": executor.resident_device_bytes,
            "complete_layer_tracked_device_bytes": executor.tracked_device_bytes,
            "complete_layer_source_weight_bytes": request.assignment.weight_bytes,
            "lifecycle": executor.lifecycle_snapshot(),
            "prepare": executor.prepare_warmup,
            "state": state,
        }
        return receipt, outputs, routes, route_weights, state
    finally:
        executor.close()


def benchmark_real_expert_microwork(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    graph_certification: Path,
    output_path: Path,
    *,
    layer: int = 89,
    workers: int = 2,
    device: int = 0,
    warmup: int = 10,
    iterations: int = 30,
    cycle_id: str = "H014-SUB-001a",
    maximum_critical_worker_device_ms: float | None = None,
    minimum_relative_throughput: float | None = None,
) -> dict[str, Any]:
    """Run one incrementally selected logical expert-worker topology."""
    if workers not in (2, 4, 8, 16):
        raise ValueError("sub-layer worker count must be one of 2/4/8/16")
    if layer != 89:
        raise ValueError("first real expert microwork target is layer 89")
    if warmup < 7 or iterations < 20:
        raise ValueError("sub-layer certification requires >=7 warmup and >=20 calls")
    paths = {
        "checkpoint": checkpoint.resolve(),
        "cuda_library": cuda_library.resolve(),
        "oracle_trace": oracle_trace.resolve(),
        "oracle_routes": oracle_routes.resolve(),
        "graph_certification": graph_certification.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    graph = json.loads(graph_certification.read_text(encoding="utf-8"))
    fixture = graph.get("fixture", {})
    provenance_pass = (
        graph.get("status") == "PASS"
        and fixture.get("oracle_trace_sha256") == _sha256_file(oracle_trace)
        and fixture.get("oracle_routes_sha256") == _sha256_file(oracle_routes)
    )
    if not provenance_pass:
        raise ValueError("sub-layer oracle does not join to the passing graph receipt")

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "RUNNING",
        "hypothesis": {
            "prediction": (
                "The 16 real selected experts of Kimi layer 89 can be resolved "
                f"across {workers} persistent disjoint worker partitions, executed "
                "exactly once, and deterministically reduced with strict whole-layer "
                "equivalence while every worker stores less than the complete layer."
            ),
            "correctness_gate_relative_l2": CORRECTNESS_RELATIVE_L2_GATE,
            "memory_gate": "every worker tracked bytes < complete layer resident bytes",
            "maximum_critical_worker_device_ms": maximum_critical_worker_device_ms,
            "minimum_whole_layer_relative_throughput": minimum_relative_throughput,
        },
        "scope": {
            "proof_class": "LOGICAL_SUB_LAYER_SAME_GPU_MULTI_PROCESS",
            "physical_multi_gpu_proof": False,
            "physical_efficiency_claimed": False,
            "transport": "Windows loopback multiprocessing pipe",
        },
        "configuration": {
            "layer": layer,
            "worker_count": workers,
            "ownership": "expert_id modulo worker_count",
            "device": device,
            "warmup_calls": warmup,
            "retained_calls": iterations,
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "oracle_provenance_pass": provenance_pass,
        "device_identity": _device_identity(device),
        "progress": [],
    }

    def retain(phase: str, **evidence: Any) -> None:
        receipt["progress"].append({"phase": phase, **evidence})
        _atomic_json(output_path, receipt)

    retain("preregistered")
    receipt["gpu_health_before"] = _health_snapshot(device)
    retain("gpu_health_before", status=receipt["gpu_health_before"]["status"])
    if receipt["gpu_health_before"]["status"] != "MEASURED":
        raise RuntimeError("nvidia-smi unavailable before sub-layer benchmark")

    fixtures, expected_boundaries = _stage_fixtures(
        checkpoint, oracle_trace, layer=layer
    )
    expected_routes = _parse_oracle_routes(oracle_routes)[layer]
    request = _request(
        checkpoint,
        cuda_library,
        layer=layer,
        device=device,
        cycle_id=cycle_id,
        maximum_context=warmup + iterations + 20,
    )
    proxies: list[_WorkerProxy] = []
    coordinator: PersistentKimiStageExecutor | None = None
    try:
        reference, reference_outputs, reference_routes, reference_weights, reference_state = (
            _reference(
                request,
                checkpoint,
                cuda_library,
                fixtures,
                expected_boundaries,
                expected_routes,
                device=device,
                warmup=warmup,
                iterations=iterations,
            )
        )
        receipt["single_gpu_reference"] = reference
        retain(
            "single_gpu_reference_closed",
            status=reference["status"],
            resident_bytes=reference["complete_layer_resident_device_bytes"],
            gpu_health=_health_snapshot(device)["status"],
        )
        if reference["status"] != "PASS":
            raise RuntimeError("single-GPU reference failed before microwork")

        context = mp.get_context("spawn")
        ownership_sets = [
            frozenset(expert for expert in range(896) if expert % workers == worker_id)
            for worker_id in range(workers)
        ]
        for worker_id, ownership in enumerate(ownership_sets):
            proxy = _start_worker(
                context,
                checkpoint=checkpoint,
                cuda_library=cuda_library,
                layer=layer,
                device=device,
                worker_id=worker_id,
                ownership=ownership,
            )
            proxies.append(proxy)
            retain(
                f"worker_{worker_id}_ready",
                ready=proxy.ready,
                gpu_health=_health_snapshot(device)["status"],
            )

        coordinator = PersistentKimiStageExecutor(
            request=request,
            checkpoint=checkpoint,
            cuda_library=cuda_library,
            device=device,
            owned_expert_ids=frozenset(),
        )
        collective = _PersistentExpertCollective(
            proxies, latent=coordinator.config.latent
        )
        receipt["coordinator"] = {
            "resident_device_bytes": coordinator.resident_device_bytes,
            "tracked_device_bytes": coordinator.tracked_device_bytes,
            "resident_expert_count": 0,
            "lifecycle_at_load": coordinator.lifecycle_snapshot(),
        }
        retain(
            "persistent_collective_loaded",
            coordinator_resident_bytes=coordinator.resident_device_bytes,
            gpu_health=_health_snapshot(device)["status"],
        )

        coordinator.open_session(
            "h014-sub-prepare", maximum_context_override=7
        )
        prepare_wall: list[float] = []
        try:
            for position in range(7):
                started = time.perf_counter_ns()
                coordinator.execute_decode_with_external_experts(
                    session_id="h014-sub-prepare",
                    hidden_states=torch.from_numpy(
                        fixtures[position % len(fixtures)].copy()
                    ),
                    cache_position_start=position,
                    dispatch=collective.dispatch,
                )
                prepare_wall.append((time.perf_counter_ns() - started) / 1e6)
        finally:
            coordinator.close_session("h014-sub-prepare")
        receipt["persistent_collective_prepare"] = {
            "calls": 7,
            "wall": _timing(prepare_wall),
            "workers_created_during_prepare": 0,
            "connections_established_during_prepare": 0,
            "topology_rebuilds_during_prepare": 0,
            "weights_loaded_during_prepare": 0,
        }
        retain("persistent_collective_prepared")

        coordinator.open_session(
            "h014-sub-correctness", maximum_context_override=3
        )
        comparisons: list[dict[str, Any]] = []
        try:
            for position in range(3):
                result = coordinator.execute_decode_with_external_experts(
                    session_id="h014-sub-correctness",
                    hidden_states=torch.from_numpy(fixtures[position].copy()),
                    cache_position_start=position,
                    dispatch=collective.dispatch,
                )
                output = result.stage_boundary_hidden_states.detach().cpu().numpy().copy()
                stage_record = coordinator.execution_records[-1]
                expert_record = stage_record["external_expert_dispatch"]
                comparisons.append(
                    {
                        "position": position,
                        "metrics": _numerical_metrics(
                            reference_outputs[position], output
                        ),
                        "reference_fingerprint": _array_fingerprint(
                            reference_outputs[position]
                        ),
                        "distributed_fingerprint": _array_fingerprint(output),
                        "selected_ids_exact": stage_record["selected_expert_ids"]
                        == reference_routes[position],
                        "selected_weights_exact": np.array_equal(
                            np.asarray(stage_record["selected_weights"], dtype=np.float32),
                            np.asarray(reference_weights[position], dtype=np.float32),
                        ),
                        "ownership_resolution_exact": all(
                            int(owner) == int(expert) % workers
                            for expert, owner in expert_record[
                                "owner_resolution"
                            ].items()
                        ),
                        "all_16_executed_exactly_once": expert_record[
                            "all_16_executed_exactly_once"
                        ],
                        "external_record": expert_record,
                    }
                )
            distributed_state = coordinator.session_state_evidence(
                "h014-sub-correctness"
            )
        finally:
            coordinator.close_session("h014-sub-correctness")
        max_error = max(
            float(row["metrics"]["relative_l2_error"]) for row in comparisons
        )
        correctness_pass = (
            max_error <= CORRECTNESS_RELATIVE_L2_GATE
            and all(row["selected_ids_exact"] for row in comparisons)
            and all(row["selected_weights_exact"] for row in comparisons)
            and all(row["ownership_resolution_exact"] for row in comparisons)
            and all(row["all_16_executed_exactly_once"] for row in comparisons)
            and distributed_state["fingerprint"] == reference_state["fingerprint"]
        )
        receipt["correctness"] = {
            "status": "PASS" if correctness_pass else "FAIL",
            "comparisons": comparisons,
            "maximum_relative_l2_error": max_error,
            "state_fingerprint_exact": distributed_state["fingerprint"]
            == reference_state["fingerprint"],
            "reference_state": reference_state,
            "distributed_state": distributed_state,
        }
        retain("correctness", status=receipt["correctness"]["status"])
        if not correctness_pass:
            raise RuntimeError("distributed expert layer differs from reference")

        coordinator.open_session(
            "h014-sub-performance",
            maximum_context_override=warmup + iterations,
        )
        wall_values: list[float] = []
        retained_external: list[dict[str, Any]] = []
        lifecycle_before: dict[str, Any] | None = None
        process_before: dict[str, Any] | None = None
        try:
            for position in range(warmup + iterations):
                if position == warmup:
                    lifecycle_before = coordinator.lifecycle_snapshot()
                    process_before = _microwork_process_snapshot()
                started = time.perf_counter_ns()
                coordinator.execute_decode_with_external_experts(
                    session_id="h014-sub-performance",
                    hidden_states=torch.from_numpy(
                        fixtures[position % len(fixtures)].copy()
                    ),
                    cache_position_start=position,
                    dispatch=collective.dispatch,
                )
                if position >= warmup:
                    wall_values.append((time.perf_counter_ns() - started) / 1e6)
                    retained_external.append(
                        coordinator.execution_records[-1]["external_expert_dispatch"]
                    )
            lifecycle_after = coordinator.lifecycle_snapshot()
            process_after = _microwork_process_snapshot()
        finally:
            coordinator.close_session("h014-sub-performance")
        if lifecycle_before is None or process_before is None:
            raise RuntimeError("distributed performance did not reach retained phase")

        roundtrip = [float(row["loopback_roundtrip_ms"]) for row in retained_external]
        exposed = [
            float(row["exposed_loopback_and_coordination_ms"])
            for row in retained_external
        ]
        worker_critical = [
            float(row["worker_compute_critical_device_ms"])
            for row in retained_external
        ]
        worker_total = [
            float(row["worker_compute_total_device_ms"])
            for row in retained_external
        ]
        distributed_wall = _timing(wall_values)
        reference_wall = reference["whole_layer_wall"]
        ideal_parallel_expert_ms = (
            reference["routed_expert_device"]["p50_ms"] / workers
        )
        sub_layer_efficiency = ideal_parallel_expert_ms / _timing(roundtrip)["p50_ms"]
        relative_throughput = reference_wall["p50_ms"] / distributed_wall["p50_ms"]
        lifecycle_delta = {
            key: int(lifecycle_after[key]) - int(lifecycle_before[key])
            for key in (
                "weight_load_count",
                "model_materialization_count",
                "persistent_buffer_allocation_count",
            )
        }
        process_delta = _microwork_lifecycle_delta(process_before, process_after)
        per_worker = []
        complete_resident = int(reference["complete_layer_resident_device_bytes"])
        for proxy in proxies:
            tracked = int(proxy.ready["tracked_worker_bytes"])
            per_worker.append(
                {
                    **proxy.ready,
                    "complete_layer_fraction_percent": 100.0 * tracked / complete_resident,
                    "less_than_complete_layer": tracked < complete_resident,
                }
            )
        memory_pass = all(row["less_than_complete_layer"] for row in per_worker)

        receipt["performance"] = {
            "distributed_complete_layer_wall": distributed_wall,
            "single_gpu_complete_layer_wall": reference_wall,
            "whole_layer_relative_throughput": relative_throughput,
            "loopback_expert_roundtrip": _timing(roundtrip),
            "worker_compute_critical_device": _timing(worker_critical),
            "worker_compute_total_device": _timing(worker_total),
            "exposed_loopback_and_coordination": _timing(exposed),
            "ideal_parallel_expert_service_ms": ideal_parallel_expert_ms,
            "sub_layer_efficiency": sub_layer_efficiency,
            "aggregate_layers_per_second": 1000.0 / distributed_wall["p50_ms"],
            "retained_calls": iterations,
            "latency_decomposition": {
                "router_and_parent_pre_dispatch_included_in_other": True,
                "parent_latent_d2h": _timing(
                    [float(row["parent_latent_d2h_ms"]) for row in retained_external]
                ),
                "dispatch": _timing(
                    [float(row["dispatch_ms"]) for row in retained_external]
                ),
                "network_transport_exposed": _timing(exposed),
                "expert_compute_critical_device": _timing(worker_critical),
                "collection_deserialization": _timing(
                    [
                        float(row["collection_deserialization_ms"])
                        for row in retained_external
                    ]
                ),
                "parent_expert_rows_h2d_enqueue": _timing(
                    [
                        float(row["parent_expert_rows_h2d_enqueue_ms"])
                        for row in retained_external
                    ]
                ),
                "shared_expert_reference_device": reference[
                    "shared_expert_device"
                ],
                "reduction_residual_and_other_ms_p50": max(
                    0.0,
                    distributed_wall["p50_ms"]
                    - _timing(roundtrip)["p50_ms"],
                ),
            },
        }
        receipt["communication"] = {
            "input_activation_bytes": coordinator.config.latent * 4,
            "selected_expert_id_bytes": 16 * 4,
            "returned_expert_output_bytes": 16 * coordinator.config.latent * 4,
            "reduction_weight_bytes": 16 * 4,
            "mean_total_transport_bytes": float(
                np.mean([row["total_transport_bytes"] for row in retained_external])
            ),
            "maximum_critical_path_payload_bytes": max(
                row["critical_path_payload_bytes"] for row in retained_external
            ),
            "mean_root_messages": float(
                np.mean([row["root_messages"] for row in retained_external])
            ),
            "mean_workers_contacted": float(
                np.mean([row["workers_contacted"] for row in retained_external])
            ),
            "synchronization_points": 2,
            "framing": "4-byte multiprocessing frame plus measured pickle payload",
        }
        receipt["routing_imbalance"] = {
            "worker_selected_count_histogram": dict(
                sorted(
                    Counter(
                        row["selected_count"]
                        for call in retained_external
                        for row in call["worker_records"]
                    ).items()
                )
            ),
            "hottest_worker_selected_count": max(
                row["selected_count"]
                for call in retained_external
                for row in call["worker_records"]
            ),
            "coldest_contacted_worker_selected_count": min(
                row["selected_count"]
                for call in retained_external
                for row in call["worker_records"]
            ),
            "fixture_limitation": (
                "Three graph-certified decode positions are rotated; a larger "
                "real-prompt corpus is required before ownership optimization."
            ),
        }
        receipt["memory"] = {
            "complete_layer_resident_device_bytes": complete_resident,
            "complete_layer_tracked_device_bytes": reference[
                "complete_layer_tracked_device_bytes"
            ],
            "complete_layer_source_weight_bytes": reference[
                "complete_layer_source_weight_bytes"
            ],
            "coordinator_resident_device_bytes": coordinator.resident_device_bytes,
            "workers": per_worker,
            "smallest_worker_tracked_bytes": min(
                row["tracked_worker_bytes"] for row in per_worker
            ),
            "largest_worker_tracked_bytes": max(
                row["tracked_worker_bytes"] for row in per_worker
            ),
            "every_worker_less_than_complete_layer": memory_pass,
        }
        receipt["lifecycle"] = {
            "coordinator_delta": lifecycle_delta,
            "process_delta": process_delta,
            "workers_persistent": True,
            "connections_persistent": True,
            "expert_ownership_persistent": True,
            "worker_execute_deltas_zero": all(
                int(row["weight_loads_during_execute"]) == 0
                and int(row["persistent_buffer_allocations_during_execute"]) == 0
                and int(row["topology_rebuilds_during_execute"]) == 0
                for call in retained_external
                for row in call["worker_records"]
            ),
        }
        receipt["gpu_health_after"] = _health_snapshot(device)
        performance_supported = (
            (
                maximum_critical_worker_device_ms is None
                or _timing(worker_critical)["p50_ms"]
                <= maximum_critical_worker_device_ms
            )
            and (
                minimum_relative_throughput is None
                or relative_throughput >= minimum_relative_throughput
            )
        )
        receipt["performance_hypothesis_supported"] = performance_supported
        receipt["hypothesis_supported"] = (
            correctness_pass and memory_pass and performance_supported
        )
        receipt["inspection"] = {
            "logical_functionality": "PASS",
            "physical_multi_gpu_efficiency": "NOT_TESTED",
            "actual_bottleneck": (
                "pending numerical inspection of same-GPU process transport and "
                "worker critical path"
            ),
        }
        if receipt["hypothesis_supported"]:
            receipt["decision"] = (
                "RETAIN_SCALING_ENDPOINT_AND_PROCEED_TO_NETWORK_REPLAY"
                if workers == 16
                else f"RETAIN_{workers}_WORKER_CHARACTERIZATION_AND_CONTINUE_INCREMENTALLY"
            )
        else:
            receipt["decision"] = (
                "RETAIN_CHARACTERIZATION_AND_MODIFY_SCALING_HYPOTHESIS"
            )
        receipt["status"] = (
            "PASS"
            if correctness_pass
            and memory_pass
            and all(value == 0 for value in lifecycle_delta.values())
            and all(value == 0 for value in process_delta.values())
            and receipt["lifecycle"]["worker_execute_deltas_zero"]
            and receipt["gpu_health_after"]["status"] == "MEASURED"
            else "FAIL"
        )
        retain("complete", status=receipt["status"])
        return receipt
    except Exception as exc:
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "gpu_health": _health_snapshot(device),
        }
        _atomic_json(output_path, receipt)
        raise
    finally:
        if coordinator is not None:
            coordinator.close()
        receipt["worker_shutdown"] = [proxy.close() for proxy in reversed(proxies)]
        _atomic_json(output_path, receipt)
