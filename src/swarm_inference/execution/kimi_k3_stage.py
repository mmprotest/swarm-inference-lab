"""Canonical production implementation for persistent Kimi K3 CUDA stages."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import hashlib
import json
import os
import statistics
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from swarm_inference.execution.interfaces import StageExecutionResult, WeightOwnership
from swarm_inference.execution.kimi_cuda_runtime import (
    KimiCudaError,
    _array_fingerprint,
    _numerical_metrics,
    _quantize_bf16_rows_int8,
    _sha256_file,
)
from swarm_inference.execution.kimi_k3_graph_runtime import (
    KimiCudaGraphRunner,
    _digest_array,
    _LayerResources,
    _parse_oracle_routes,
    _pointer_offset,
)
from swarm_inference.execution.verification import (
    VerificationBlock,
    build_grouped_expert_plan,
    expert_reuse_statistics,
)
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

SCHEMA_VERSION = "experiment-014-k3-persistent-final-stage-v1"
PRODUCTION_MAX_CERTIFIED_BATCH = 8
VERIFICATION_MAX_ROWS = 17
PREPARE_MEMORY_RECOVERY_TOLERANCE_BYTES = 4 * 1024**2
PREPARE_THREAD_QUIESCENCE_TIMEOUT_S = 8.0
PREPARE_THREAD_QUIESCENCE_MINIMUM_OBSERVATION_S = 3.5
PREPARE_THREAD_QUIESCENCE_SAMPLE_INTERVAL_S = 0.05
PREPARE_THREAD_QUIESCENCE_STABLE_SAMPLES = 20
_KIMI_CPU_TRANSPORT_LOCK = threading.Lock()
_KIMI_CPU_TRANSPORT_CONFIGURED = False


def _production_batch_capacity(supported_batches: tuple[int, ...]) -> int:
    certified = tuple(
        size
        for size in supported_batches
        if 0 < size <= PRODUCTION_MAX_CERTIFIED_BATCH
    )
    if not certified:
        raise KimiCudaError("native runtime exposes no production-certified batch")
    return max(certified)


def _prepare_retained_device_bytes(before_free: int, after_free: int) -> int:
    return max(0, int(before_free) - int(after_free))


def _configure_kimi_cpu_transport_threads() -> dict[str, int]:
    """Prevent lazy native CPU-pool growth after a Kimi worker reports READY."""

    global _KIMI_CPU_TRANSPORT_CONFIGURED
    with _KIMI_CPU_TRANSPORT_LOCK:
        if not _KIMI_CPU_TRANSPORT_CONFIGURED:
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError as exc:
                if torch.get_num_interop_threads() != 1:
                    raise KimiCudaError(
                        "Kimi CPU transport inter-op pool was initialized before "
                        "the single-thread READY contract"
                    ) from exc
            _KIMI_CPU_TRANSPORT_CONFIGURED = True
        intraop_threads = int(torch.get_num_threads())
        interop_threads = int(torch.get_num_interop_threads())
        if intraop_threads != 1 or interop_threads != 1:
            raise KimiCudaError(
                "Kimi CPU transport requires one intra-op and one inter-op thread"
            )
        return {
            "intraop_threads": intraop_threads,
            "interop_threads": interop_threads,
        }


def _wait_for_process_thread_quiescence(
    *,
    timeout_s: float = PREPARE_THREAD_QUIESCENCE_TIMEOUT_S,
    minimum_observation_s: float = PREPARE_THREAD_QUIESCENCE_MINIMUM_OBSERVATION_S,
    sample_interval_s: float = PREPARE_THREAD_QUIESCENCE_SAMPLE_INTERVAL_S,
    stable_samples: int = PREPARE_THREAD_QUIESCENCE_STABLE_SAMPLES,
) -> dict[str, Any]:
    """Require a bounded stable OS-thread set before a worker reports READY."""

    if (
        timeout_s <= 0
        or minimum_observation_s <= 0
        or minimum_observation_s >= timeout_s
        or sample_interval_s <= 0
        or stable_samples < 2
    ):
        raise ValueError("invalid Kimi PREPARE thread-quiescence parameters")
    process = psutil.Process()
    started = time.perf_counter()
    previous: tuple[int, ...] | None = None
    stable_count = 0
    sample_count = 0
    transitions: list[dict[str, Any]] = []
    while True:
        thread_ids = tuple(sorted(item.id for item in process.threads()))
        sample_count += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if previous is None:
            transitions.append(
                {
                    "sample": sample_count,
                    "elapsed_ms": elapsed_ms,
                    "thread_count": len(thread_ids),
                    "added_thread_ids": [],
                    "removed_thread_ids": [],
                }
            )
            stable_count = 1
            previous = thread_ids
        elif thread_ids == previous:
            stable_count += 1
        else:
            before = set(previous)
            after = set(thread_ids)
            transitions.append(
                {
                    "sample": sample_count,
                    "elapsed_ms": elapsed_ms,
                    "thread_count": len(thread_ids),
                    "added_thread_ids": sorted(after - before),
                    "removed_thread_ids": sorted(before - after),
                }
            )
            stable_count = 1
            previous = thread_ids
        if (
            stable_count >= stable_samples
            and elapsed_ms >= minimum_observation_s * 1000.0
        ):
            return {
                "pass": True,
                "timeout_ms": timeout_s * 1000.0,
                "minimum_observation_ms": minimum_observation_s * 1000.0,
                "sample_interval_ms": sample_interval_s * 1000.0,
                "required_stable_samples": stable_samples,
                "sample_count": sample_count,
                "stable_samples_observed": stable_count,
                "elapsed_ms": elapsed_ms,
                "final_thread_count": len(thread_ids),
                "final_thread_ids": list(thread_ids),
                "transitions": transitions,
            }
        if elapsed_ms >= timeout_s * 1000.0:
            raise KimiCudaError(
                "Kimi PREPARE process threads did not quiesce before READY"
            )
        time.sleep(sample_interval_s)


@dataclass(slots=True)
class _FinalStageSession:
    resources: _LayerResources
    attention_state: dict[str, ctypes.c_void_p]
    attention_scratch: dict[str, ctypes.c_void_p]
    input_row: ctypes.c_void_p
    prefix_row: ctypes.c_void_p
    hidden_scratch: ctypes.c_void_p
    normalized: ctypes.c_void_p
    attention_output: ctypes.c_void_p
    mixed: ctypes.c_void_p
    mlp_input: ctypes.c_void_p
    expert_rows: ctypes.c_void_p
    route_weights: ctypes.c_void_p
    latent_input: ctypes.c_void_p
    reduced: ctypes.c_void_p
    routed_output: ctypes.c_void_p
    shared_output: ctypes.c_void_p
    residual_scratch: ctypes.c_void_p
    final_mixed: ctypes.c_void_p
    final_normalized: ctypes.c_void_p
    logits: ctypes.c_void_p
    token_id: ctypes.c_void_p
    maximum_context: int
    cache_length: int = 0


def _timing(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(percent: float) -> float:
        if not ordered:
            return 0.0
        position = (len(ordered) - 1) * percent
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "minimum_ms": min(ordered),
        "maximum_ms": max(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "standard_deviation_ms": statistics.pstdev(ordered),
    }


class PersistentKimiStageExecutor:
    """Own one complete Kimi layer on one persistent CUDA context."""

    def __init__(
        self,
        *,
        request: LoadStageRequest,
        checkpoint: Path,
        cuda_library: Path,
        device: int = 0,
        owned_expert_ids: frozenset[int] | None = None,
    ) -> None:
        assignment = request.assignment
        if (
            request.stage_count != 93
            or assignment.stage_id != assignment.layer_start
            or assignment.layer_end != assignment.layer_start + 1
            # Endpoint ownership is independent of transformer-layer ownership.
            # In particular, Experiment 024 executes the admitted transformer-only
            # layer-0 whole candidate after the endpoint has produced its embedding.
            or (assignment.owns_embeddings and assignment.layer_start != 0)
            or assignment.owns_final_norm != (assignment.layer_start == 92)
            or assignment.owns_output_projection != (assignment.layer_start == 92)
        ):
            raise ValueError(
                "persistent Kimi execution requires one checkpoint-aligned transformer stage"
            )
        self.cpu_transport_thread_contract = _configure_kimi_cpu_transport_threads()
        self.request = request
        self.checkpoint = checkpoint.expanduser().resolve()
        self.cuda_library = cuda_library.expanduser().resolve()
        self.cuda_library_sha256 = _sha256_file(self.cuda_library)
        if request.verifier_precision_mode != "exact-fp32":
            raise KimiCudaError(
                "the Colibri Kimi runtime does not implement requested verifier "
                f"precision mode {request.verifier_precision_mode!r}; refusing "
                "silent fallback to exact-fp32"
            )
        self.runner = KimiCudaGraphRunner(self.checkpoint, self.cuda_library, device)
        self.runtime = self.runner.runtime
        self._telemetry_mode = "minimal"
        self.config = self.runner.config
        self.layer = assignment.layer_start
        self._owns_embeddings = assignment.owns_embeddings
        self._owns_final_endpoint = assignment.owns_final_norm
        self._closed = False
        self._sessions: dict[str, _FinalStageSession] = {}
        self.execution_records: list[dict[str, Any]] = []
        self._weight_load_count = 0
        self._model_materialization_count = 0
        self._persistent_buffer_allocation_count = 0
        self._execute_count = 0
        self._session_open_count = 0
        self._session_close_count = 0
        self._batch_execute_count = 0
        self._prepare_warmup_count = 0
        self.prepare_warmup: dict[str, Any] | None = None
        self._layer_resources = _LayerResources(self.runtime)
        self._endpoint_resources = _LayerResources(self.runtime)
        self._batch_resources = _LayerResources(self.runtime)
        production_capacity = _production_batch_capacity(
            self.runtime.expert_supported_batches
        )
        self._verification_major_enabled = request.fast_path_mode in {
            "verification-major",
            "verification-major-kda-window",
        }
        self._kda_short_window_enabled = (
            request.fast_path_mode == "verification-major-kda-window"
        )
        if (
            self._kda_short_window_enabled
            and self.runtime.kda_short_window_function is None
        ):
            raise KimiCudaError(
                "fast_path_mode='verification-major-kda-window' was explicitly "
                "requested but the native runtime has no short-window KDA export"
            )
        requested_capacity = int(request.fast_path_batch_bucket)
        if self._verification_major_enabled and requested_capacity > VERIFICATION_MAX_ROWS:
            raise ValueError(
                "verification-major Kimi execution supports at most "
                f"{VERIFICATION_MAX_ROWS} target rows"
            )
        self._batch_capacity = (
            max(production_capacity, requested_capacity)
            if self._verification_major_enabled
            else production_capacity
        )
        self._batch_workspace_bytes = 0
        memory_before = self.runtime.mem_info()
        load_started = time.perf_counter_ns()
        self._weights = self.runner._load_layer_weights(self._layer_resources, self.layer)
        self._embedding: ctypes.c_void_p | None = None
        if self._owns_embeddings:
            embedding_name = "language_model.model.embed_tokens.weight"
            embedding = self.runner.reader.array(embedding_name)
            if tuple(embedding.shape) != (self.config.vocab, self.config.hidden):
                raise KimiCudaError("persistent Kimi embedding table has invalid geometry")
            self._embedding = self._layer_resources.track_tensor(
                embedding_name,
                self.runtime.upload_bf16_embedding(embedding),
                (embedding,),
            )
        experts: dict[int, tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]] = {}
        if self._weights["mlp_type"] == "moe":
            expert_ids = (
                tuple(range(self.config.experts))
                if owned_expert_ids is None
                else tuple(sorted(owned_expert_ids))
            )
            if any(expert < 0 or expert >= self.config.experts for expert in expert_ids):
                raise ValueError("Kimi expert ownership contains an invalid expert ID")
            for loaded, expert in enumerate(expert_ids, start=1):
                experts[expert] = self.runner._upload_expert(
                    self._layer_resources, self.layer, expert
                )
                if loaded % 128 == 0 or loaded == len(expert_ids):
                    print(
                        f"[k3-persistent:load] layer {self.layer} expert "
                        f"{loaded}/{len(expert_ids)}",
                        flush=True,
                    )
        self._experts = experts
        self._expert_ownership = frozenset(experts)
        if self._owns_final_endpoint:
            self._load_final_endpoint()
        self._allocate_batch_workspace()
        self.runtime.synchronize()
        combined_weight_digest = hashlib.sha256()
        combined_weight_digest.update(self._layer_resources.weight_digest.digest())
        combined_weight_digest.update(self._endpoint_resources.weight_digest.digest())
        self.weight_fingerprint = "sha256:" + combined_weight_digest.hexdigest()
        self._weight_load_count = 1
        self._model_materialization_count = 1
        self.load_ns = time.perf_counter_ns() - load_started
        self.memory_before = memory_before
        self.memory_after = self.runtime.mem_info()
        self.tracked_device_bytes = (
            self._layer_resources.resident_tensor_bytes
            + self._layer_resources.resident_vector_bytes
            + self._endpoint_resources.resident_tensor_bytes
            + self._endpoint_resources.resident_vector_bytes
            + self._batch_workspace_bytes
        )
        measured_free_delta = max(
            0,
            int(self.memory_before["free_bytes"]) - int(self.memory_after["free_bytes"]),
        )
        self.resident_device_bytes = max(self.tracked_device_bytes, measured_free_delta)
        endpoint_names = (
            {
                "language_model.model.output_attn_res_norm.weight",
                "language_model.model.output_attn_res_proj.weight",
                "language_model.model.norm.weight",
                "language_model.lm_head.weight",
            }
            if self._owns_final_endpoint
            else set()
        )
        source_names = tuple(
            sorted(
                name
                for name in self.runner.reader.weight_map
                if name.startswith(f"language_model.model.layers.{self.layer}.")
                or (self._owns_embeddings and name == "language_model.model.embed_tokens.weight")
                or name in endpoint_names
            )
        )
        source_bytes = sum(self.runner.reader.array(name).nbytes for name in source_names)
        if source_bytes != assignment.weight_bytes:
            raise ValueError(
                f"final-stage source bytes {source_bytes} differ from assignment "
                f"{assignment.weight_bytes}"
            )
        ownership_payload = json.dumps(
            {
                "names": source_names,
                "cuda_library_sha256": self.cuda_library_sha256,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        self._ownership = WeightOwnership(
            stage_id=assignment.stage_id,
            layer_start=assignment.layer_start,
            layer_end=assignment.layer_end,
            parameter_names=source_names,
            parameter_bytes=assignment.weight_bytes,
            parameter_count=len(source_names),
            owns_embeddings=assignment.owns_embeddings,
            owns_final_norm=assignment.owns_final_norm,
            owns_output_projection=assignment.owns_output_projection,
            ownership_hash=hashlib.sha256(ownership_payload).hexdigest(),
        )

    @property
    def ownership(self) -> WeightOwnership:
        return self._ownership

    def _load_final_endpoint(self) -> None:
        resources = self._endpoint_resources
        norm_name = "language_model.model.output_attn_res_norm.weight"
        projection_name = "language_model.model.output_attn_res_proj.weight"
        norm = self.runner.reader.f32(norm_name)
        projection = self.runner.reader.f32(projection_name)
        resources.tensor_names.extend((norm_name, projection_name))
        _digest_array(resources.weight_digest, norm_name, norm)
        _digest_array(resources.weight_digest, projection_name, projection)
        self._final_score = resources.upload_vector(
            "language_model.model.output_attn_res_score_weight",
            np.ascontiguousarray(norm * projection, dtype=np.float32),
        )
        final_norm_name = "language_model.model.norm.weight"
        self._final_norm = resources.upload_vector(
            final_norm_name, self.runner.reader.f32(final_norm_name)
        )
        head_name = "language_model.lm_head.weight"
        quantized = _quantize_bf16_rows_int8(self.runner.reader.array(head_name))
        self._head = resources.track_tensor(
            head_name,
            self.runtime.upload_int8(quantized),
            (quantized.weights, quantized.scales),
        )

    def _allocate_zero(
        self,
        resources: _LayerResources,
        shape: tuple[int, ...],
    ) -> ctypes.c_void_p:
        values = np.zeros(shape, dtype=np.float32)
        pointer = resources.allocate(values.size)
        self._persistent_buffer_allocation_count += 1
        self.runtime.upload_activation(pointer, values)
        return pointer

    def _run_prepare_warmup(self, *, minimum_device_ms: float = 100.0) -> dict[str, Any]:
        """Make a resident CUDA context compute-ready before advertising READY."""
        if minimum_device_ms <= 0.0 or self._prepare_warmup_count:
            raise ValueError("Kimi PREPARE warmup must run exactly once with a positive gate")
        byte_count = self.config.hidden * np.dtype(np.float32).itemsize
        memory_before = self.runtime.mem_info()
        source = self.runtime.allocate(byte_count)
        destination = self.runtime.allocate(byte_count)
        zeros = np.zeros(self.config.hidden, dtype=np.float32)
        wall_started = time.perf_counter_ns()
        device_ms = 0.0
        launches = 0
        batch_launches = 4096
        batches: list[float] = []
        try:
            self.runtime.upload_activation(source, zeros)
            self.runtime.upload_activation(destination, zeros)
            while device_ms < minimum_device_ms:
                self.runtime.profile_begin()
                for _ in range(batch_launches):
                    self.runtime.execute_add(destination, source, self.config.hidden)
                batch_ms = self.runtime.profile_end()
                batches.append(batch_ms)
                device_ms += batch_ms
                launches += batch_launches
                if launches >= 1_000_000:
                    raise KimiCudaError("Kimi PREPARE warmup did not reach its device-time gate")
        finally:
            self.runtime.free(destination)
            self.runtime.free(source)
        memory_after = self.runtime.mem_info()
        wall_ms = (time.perf_counter_ns() - wall_started) / 1e6
        self._prepare_warmup_count += 1
        return {
            "count": self._prepare_warmup_count,
            "primitive": "generic resident float32 add",
            "minimum_device_ms": minimum_device_ms,
            "measured_device_ms": device_ms,
            "measured_wall_ms": wall_ms,
            "kernel_launches": launches,
            "batch_device_ms": batches,
            "temporary_device_bytes": 2 * byte_count,
            "h2d_bytes": 2 * byte_count,
            "expert_weight_bytes_read": 0,
            "temporary_buffers_released_before_ready": True,
            "free_bytes_before": memory_before["free_bytes"],
            "free_bytes_after": memory_after["free_bytes"],
            "temporary_memory_recovered": (
                memory_after["free_bytes"] >= memory_before["free_bytes"]
            ),
        }

    def prepare_for_ready(self) -> None:
        """Finish one-time CUDA readiness work at the worker PREPARE boundary."""
        if self.prepare_warmup is not None or self._prepare_warmup_count:
            raise RuntimeError("Kimi executor PREPARE warmup was invoked more than once")
        stage_fixture = self._run_prepare_stage_fixture()
        self._prepare_warmup_count += 1
        total_wall_ms = float(stage_fixture["measured_wall_ms"])
        maximum_temporary_bytes = int(stage_fixture["temporary_device_bytes"])
        self.prepare_warmup = {
            "count": self._prepare_warmup_count,
            "primitive": "isolated assigned-stage Kimi CUDA fixture",
            "measured_device_ms": float(stage_fixture["measured_device_ms"]),
            "measured_wall_ms": total_wall_ms,
            "temporary_device_bytes": maximum_temporary_bytes,
            "temporary_memory_recovered": bool(stage_fixture["temporary_memory_recovered"]),
            "stage_fixture": stage_fixture,
            "total_measured_wall_ms": total_wall_ms,
            "maximum_temporary_device_bytes": maximum_temporary_bytes,
            "cpu_transport_threads": self.cpu_transport_thread_contract,
        }

    def _run_prepare_stage_fixture(self, *, iterations: int = 7) -> dict[str, Any]:
        if iterations < 3:
            raise ValueError("Kimi stage-specific PREPARE fixture requires at least 3 calls")
        session_id = "__kimi_prepare_ready__"
        memory_before = self.runtime.mem_info()
        records_before = len(self.execution_records)
        execute_count_before = self._execute_count
        wall_started = time.perf_counter_ns()
        self.open_session(session_id, maximum_context_override=iterations)
        memory_after_open = self.runtime.mem_info()
        device_ms: list[float] = []
        final_boundary: np.ndarray | None = None
        selected_expert_ids: list[int] = []
        host_thread_native_ids: list[int] = []
        host_thread_idents: list[int] = []
        transport_evidence: dict[str, Any] | None = None
        thread_quiescence: dict[str, Any] | None = None
        try:
            zero_boundary = torch.zeros((1, 9, self.config.hidden), dtype=torch.float32)
            token_ids = torch.zeros((1, 1), dtype=torch.int64)
            for position in range(iterations):
                result = (
                    self.execute_prefill(
                        session_id=session_id,
                        token_ids=token_ids,
                        cache_position_start=position,
                    )
                    if self._owns_embeddings
                    else self.execute_decode(
                        session_id=session_id,
                        hidden_states=zero_boundary,
                        cache_position_start=position,
                    )
                )
                record = self.execution_records[-1]
                if record["device_ms"] is None:
                    raise KimiCudaError("Kimi PREPARE fixture omitted whole-stage timing")
                device_ms.append(float(record["device_ms"]))
                host_thread_native_ids.append(int(record["host_thread_native_id"]))
                host_thread_idents.append(int(record["host_thread_ident"]))
                selected_expert_ids = list(record["selected_expert_ids"])
                final_boundary = result.stage_boundary_hidden_states.detach().cpu().numpy().copy()
                if position == iterations - 2:
                    transport_os_threads_before = sorted(
                        item.id for item in psutil.Process().threads()
                    )
                    transport_python_threads_before = sorted(
                        int(item.ident)
                        for item in threading.enumerate()
                        if item.ident is not None
                    )
                    transport_source = torch.from_numpy(final_boundary.copy())
                    packed_boundary = pack_tensor(
                        transport_source,
                        requested_mode="none",
                    )
                    restored_boundary, transport_decode_ns = unpack_tensor(
                        packed_boundary.payload,
                        packed_boundary.attributes(),
                    )
                    transport_bit_exact = bool(
                        torch.equal(transport_source, restored_boundary)
                    )
                    if not transport_bit_exact:
                        raise KimiCudaError(
                            "Kimi PREPARE boundary transport was not bit exact"
                        )
                    transport_os_threads_after_round_trip = sorted(
                        item.id for item in psutil.Process().threads()
                    )
                    transport_python_threads_after_round_trip = sorted(
                        int(item.ident)
                        for item in threading.enumerate()
                        if item.ident is not None
                    )
                    transport_evidence = {
                        "pass": transport_bit_exact,
                        "source_iteration": position + 1,
                        "shape": list(packed_boundary.shape),
                        "dtype": packed_boundary.dtype,
                        "compression_mode": packed_boundary.compression_mode,
                        "raw_bytes": packed_boundary.raw_bytes,
                        "encoded_bytes": packed_boundary.encoded_bytes,
                        "raw_checksum": packed_boundary.raw_checksum,
                        "encode_ns": packed_boundary.encode_ns,
                        "decode_ns": transport_decode_ns,
                        "os_thread_count_before": len(transport_os_threads_before),
                        "os_thread_count_after_round_trip": len(
                            transport_os_threads_after_round_trip
                        ),
                        "os_thread_ids_added_by_round_trip": sorted(
                            set(transport_os_threads_after_round_trip)
                            - set(transport_os_threads_before)
                        ),
                        "python_thread_count_before": len(
                            transport_python_threads_before
                        ),
                        "python_thread_count_after_round_trip": len(
                            transport_python_threads_after_round_trip
                        ),
                        "python_thread_ids_added_by_round_trip": sorted(
                            set(transport_python_threads_after_round_trip)
                            - set(transport_python_threads_before)
                        ),
                    }
            thread_quiescence = _wait_for_process_thread_quiescence()
        finally:
            self.close_session(session_id)
        memory_after_close = self.runtime.mem_info()
        retained_after_close_bytes = _prepare_retained_device_bytes(
            int(memory_before["free_bytes"]),
            int(memory_after_close["free_bytes"]),
        )
        if final_boundary is None:
            raise KimiCudaError("Kimi PREPARE fixture emitted no boundary")
        finite = bool(np.isfinite(final_boundary).all())
        if not finite:
            raise KimiCudaError("Kimi PREPARE fixture emitted non-finite values")
        if transport_evidence is None or thread_quiescence is None:
            raise KimiCudaError("Kimi PREPARE omitted transport/thread initialization")
        final_warm_thread_ids = sorted(item.id for item in psutil.Process().threads())
        transport_evidence["final_warm_iteration"] = iterations
        transport_evidence["os_thread_count_after_final_warm"] = len(
            final_warm_thread_ids
        )
        transport_evidence["os_thread_ids_added_by_final_warm"] = sorted(
            set(final_warm_thread_ids)
            - set(thread_quiescence["final_thread_ids"])
        )
        measured_wall_ms = (time.perf_counter_ns() - wall_started) / 1e6
        del self.execution_records[records_before:]
        self._execute_count = execute_count_before
        return {
            "iterations": iterations,
            "input": ("int64 token ID 0" if self._owns_embeddings else "float32 zero [1,9,7168]"),
            "device": _timing(device_ms),
            "device_ms_values": [float(value) for value in device_ms],
            "measured_device_ms": sum(device_ms),
            "measured_wall_ms": measured_wall_ms,
            "temporary_device_bytes": max(
                0,
                int(memory_before["free_bytes"]) - int(memory_after_open["free_bytes"]),
            ),
            "free_bytes_before": memory_before["free_bytes"],
            "free_bytes_after_close": memory_after_close["free_bytes"],
            "retained_after_close_bytes": retained_after_close_bytes,
            "memory_recovery_tolerance_bytes": (
                PREPARE_MEMORY_RECOVERY_TOLERANCE_BYTES
            ),
            "temporary_memory_recovered": (
                retained_after_close_bytes
                <= PREPARE_MEMORY_RECOVERY_TOLERANCE_BYTES
            ),
            "active_sessions_after": len(self._sessions),
            "serving_execute_count_restored": self._execute_count == execute_count_before,
            "research_records_removed": len(self.execution_records) == records_before,
            "output_fingerprint": _array_fingerprint(final_boundary),
            "output_finite": finite,
            "last_selected_expert_ids": selected_expert_ids,
            "host_thread_native_ids": host_thread_native_ids,
            "host_thread_idents": host_thread_idents,
            "single_host_thread": (
                len(set(host_thread_native_ids)) == 1
                and len(set(host_thread_idents)) == 1
            ),
            "canonical_transport_round_trip": transport_evidence,
            "thread_quiescence": thread_quiescence,
            "cpu_transport_threads": self.cpu_transport_thread_contract,
            "cpu_mathematical_fallbacks": 0,
        }

    def _allocate(
        self,
        resources: _LayerResources,
        elements: int,
    ) -> ctypes.c_void_p:
        self._persistent_buffer_allocation_count += 1
        return resources.allocate(elements)

    def _allocate_batch_workspace(self) -> None:
        """Allocate the fixed row-cooperative workspace before READY."""
        capacity = self._batch_capacity
        hidden = self.config.hidden
        latent = self.config.latent
        topk = self.config.topk
        self._batch_hidden_input = self._allocate(self._batch_resources, capacity * hidden)
        self._batch_hidden_output = self._allocate(self._batch_resources, capacity * hidden)
        self._batch_shared_output = self._allocate(self._batch_resources, capacity * hidden)
        self._batch_boundary = self._allocate(
            self._batch_resources, capacity * 9 * hidden
        )
        self._batch_latent_rows = self._allocate(self._batch_resources, capacity * latent)
        self._batch_expert_input = self._allocate(self._batch_resources, capacity * topk * latent)
        self._batch_expert_output = self._allocate(self._batch_resources, capacity * topk * latent)
        self._batch_gather_indices = self._allocate(
            self._batch_resources, capacity * topk
        )
        self._batch_scatter_indices = self._allocate(
            self._batch_resources, capacity * topk
        )
        attention_elements = 0
        self._batch_attention_scratch: dict[str, ctypes.c_void_p] = {}

        def attention_buffer(name: str, row_elements: int) -> None:
            nonlocal attention_elements
            self._batch_attention_scratch[name] = self._allocate(
                self._batch_resources, capacity * row_elements
            )
            attention_elements += capacity * row_elements

        if self.layer in self.config.kda_layers:
            for name in ("q", "k", "v", "gate", "decay", "core"):
                attention_buffer(name, self.config.kda_projection)
            attention_buffer("decay_low", self.config.kda_head_dimension)
            attention_buffer("beta", self.config.kda_heads)
        else:
            attention_buffer("query_low", self.config.query_lora)
            attention_buffer("query", self.config.query_dimension)
            attention_buffer("compressed_kv", self.config.kv_lora + self.config.query_rope)
            attention_buffer("mla_gate", self.config.context_dimension)
            attention_buffer("context", self.config.context_dimension)
        self._batch_workspace_bytes = (
            capacity * (
                12 * hidden + latent + 2 * topk * latent + 2 * topk
            )
            + attention_elements
        ) * np.dtype(np.float32).itemsize

    def _supported_batch_chunks(self, count: int) -> tuple[int, ...]:
        if count <= 0:
            raise ValueError("Kimi row-cooperative chunk count must be positive")
        supported = tuple(
            sorted(
                (
                    size
                    for size in self.runtime.expert_supported_batches
                    if size <= self._batch_capacity
                ),
                reverse=True,
            )
        )
        chunks: list[int] = []
        remaining = count
        while remaining:
            try:
                chunk = next(size for size in supported if size <= remaining)
            except StopIteration as exc:
                raise KimiCudaError(
                    f"cannot partition Kimi batch {count} into certified sizes {supported}"
                ) from exc
            chunks.append(chunk)
            remaining -= chunk
        return tuple(chunks)

    def _route_batch_chunked(
        self,
        activation: ctypes.c_void_p,
        *,
        batch: int,
        accumulate_stats: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Route any prepared row count through certified native chunks."""

        if self.runtime.router_batch_function is None:
            raise KimiCudaError("Kimi CUDA binary has no batched-router export")
        ids_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        effective_parts: list[np.ndarray] = []
        offset = 0
        for chunk in self._supported_batch_chunks(batch):
            ids, route_weights, effective = self.runtime.route_batch(
                _pointer_offset(activation, offset * self.config.hidden),
                self._weights["router"],
                self._weights["router_bias"],
                batch=chunk,
                hidden=self.config.hidden,
                experts=self.config.experts,
                topk=self.config.topk,
                accumulate_stats=accumulate_stats,
            )
            ids_parts.append(ids)
            weight_parts.append(route_weights)
            effective_parts.append(effective)
            offset += chunk
        return (
            np.concatenate(ids_parts, axis=0),
            np.concatenate(weight_parts, axis=0),
            np.concatenate(effective_parts, axis=0),
        )

    def open_session(self, session_id: str, *, maximum_context_override: int | None = None) -> None:
        if self._closed:
            raise RuntimeError("Kimi final-stage executor is closed")
        if not session_id or session_id in self._sessions:
            raise ValueError("Kimi final-stage session is empty or already open")
        maximum_context = max(
            3,
            int(
                self.request.fast_path_context_bucket
                if maximum_context_override is None
                else maximum_context_override
            ),
        )
        resources = _LayerResources(self.runtime)
        if self.layer in self.config.kda_layers:
            attention_state = {
                "state": self._allocate_zero(
                    resources,
                    (
                        self.config.kda_heads,
                        self.config.kda_head_dimension,
                        self.config.kda_head_dimension,
                    ),
                ),
                "window_q": self._allocate_zero(
                    resources,
                    (self.config.kda_projection, self.config.convolution_width),
                ),
                "window_k": self._allocate_zero(
                    resources,
                    (self.config.kda_projection, self.config.convolution_width),
                ),
                "window_v": self._allocate_zero(
                    resources,
                    (self.config.kda_projection, self.config.convolution_width),
                ),
            }
            attention_scratch = {
                "q": self._allocate(resources, self.config.kda_projection),
                "k": self._allocate(resources, self.config.kda_projection),
                "v": self._allocate(resources, self.config.kda_projection),
                "gate": self._allocate(resources, self.config.kda_projection),
                "decay_low": self._allocate(resources, self.config.kda_head_dimension),
                "decay": self._allocate(resources, self.config.kda_projection),
                "beta": self._allocate(resources, self.config.kda_heads),
                "core": self._allocate(resources, self.config.kda_projection),
            }
        else:
            attention_state = {
                "latent_cache": self._allocate_zero(
                    resources, (maximum_context, self.config.kv_lora)
                ),
                "rope_cache": self._allocate_zero(
                    resources, (maximum_context, self.config.query_rope)
                ),
            }
            attention_scratch = {
                "query_low": self._allocate(resources, self.config.query_lora),
                "query": self._allocate(resources, self.config.query_dimension),
                "compressed_kv": self._allocate(
                    resources, self.config.kv_lora + self.config.query_rope
                ),
                "mla_gate": self._allocate(resources, self.config.context_dimension),
                "context": self._allocate(resources, self.config.context_dimension),
            }
        hidden = self.config.hidden
        latent = self.config.latent
        topk = self.config.topk
        session = _FinalStageSession(
            resources=resources,
            attention_state=attention_state,
            attention_scratch=attention_scratch,
            input_row=self._allocate(resources, hidden),
            prefix_row=self._allocate(resources, hidden),
            hidden_scratch=self._allocate(resources, hidden),
            normalized=self._allocate(resources, hidden),
            attention_output=self._allocate(resources, hidden),
            mixed=self._allocate(resources, hidden),
            mlp_input=self._allocate(resources, hidden),
            expert_rows=self._allocate(resources, topk * latent),
            route_weights=self._allocate(resources, topk),
            latent_input=self._allocate(resources, latent),
            reduced=self._allocate(resources, latent),
            routed_output=self._allocate(resources, hidden),
            shared_output=self._allocate(resources, hidden),
            residual_scratch=self._allocate(resources, 8 * hidden),
            final_mixed=self._allocate(resources, hidden),
            final_normalized=self._allocate(resources, hidden),
            logits=self._allocate(resources, self.config.vocab),
            token_id=self._allocate(resources, 1),
            maximum_context=maximum_context,
        )
        self._sessions[session_id] = session
        self._session_open_count += 1

    def _require_session(self, session_id: str) -> _FinalStageSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown Kimi stage session {session_id!r}") from exc

    def set_research_telemetry_mode(self, mode: str) -> None:
        """Select native counters without changing Kimi execution semantics."""
        if mode not in {"minimal", "production", "detailed"}:
            raise ValueError(f"unsupported Kimi telemetry mode {mode!r}")
        self.runtime.set_telemetry(mode)
        self._telemetry_mode = mode

    def _execute_one(
        self,
        *,
        session_id: str,
        cache_position_start: int,
        boundary: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
        external_expert_dispatch: Callable[
            [np.ndarray, np.ndarray], tuple[np.ndarray, dict[str, Any]]
        ]
        | None = None,
    ) -> StageExecutionResult:
        session = self._require_session(session_id)
        if cache_position_start != session.cache_length:
            raise ValueError("Kimi final-stage cache position is not contiguous")
        if cache_position_start >= session.maximum_context:
            raise ValueError("Kimi final-stage context exceeds its prepared cache")
        if self._owns_embeddings:
            if boundary is not None or token_ids is None:
                raise ValueError("Kimi stage zero requires token IDs and no hidden boundary")
            if token_ids.dtype != torch.int64 or token_ids.numel() != 1:
                raise ValueError("Kimi stage zero requires one int64 token ID")
            token_id = int(token_ids.reshape(-1)[0].item())
            if not 0 <= token_id < self.config.vocab:
                raise ValueError("Kimi stage-zero token ID is outside the vocabulary")
            hidden: np.ndarray | None = None
            residuals = np.zeros((8, self.config.hidden), dtype=np.float32)
        else:
            if token_ids is not None or boundary is None:
                raise ValueError("non-embedding Kimi stages require a hidden boundary")
            if boundary.dtype != torch.float32 or tuple(boundary.shape) != (
                1,
                9,
                self.config.hidden,
            ):
                raise ValueError("Kimi stage boundary must be float32 [1,9,7168]")
            values = np.ascontiguousarray(boundary.detach().cpu().numpy(), dtype=np.float32)
            hidden = values[0, 0]
            residuals = values[0, 1:].copy()
            token_id = None
        block_count = (self.layer + self.config.residual_block - 1) // (self.config.residual_block)
        is_snapshot = self.layer % self.config.residual_block == 0
        next_block_count = block_count + int(is_snapshot)
        if not 0 <= block_count <= 8 or not 1 <= next_block_count <= 8:
            raise KimiCudaError("persistent stage AttnRes block count is invalid")
        resources = session.resources
        weights = self._weights
        runtime = self.runtime
        allocation_before = self._persistent_buffer_allocation_count
        weight_load_before = self._weight_load_count
        materialization_before = self._model_materialization_count
        wall_started = time.perf_counter_ns()
        whole_stage_profile = self._telemetry_mode != "detailed"
        if whole_stage_profile:
            runtime.profile_begin()
        routed_expert_device_ms: float | None = None
        shared_expert_device_ms: float | None = None
        external_expert_record: dict[str, Any] | None = None
        routed_expert_execution_count = 0
        if token_id is not None:
            if self._embedding is None:
                raise KimiCudaError("stage-zero embedding is not resident")
            runtime.upload_bytes(session.token_id, np.ascontiguousarray([token_id], dtype=np.int32))
            runtime.execute_embedding(self._embedding, session.input_row, session.token_id, 1)
            runtime.execute_copy(session.prefix_row, session.input_row, self.config.hidden)
        else:
            assert hidden is not None
            runtime.upload_activation(session.input_row, hidden)
            runtime.upload_activation(session.prefix_row, hidden)
        if block_count:
            runtime.upload_activation(session.residual_scratch, residuals[:block_count])
            runtime.execute_attnres_mix(
                session.hidden_scratch,
                session.prefix_row,
                session.residual_scratch,
                weights["attention_residual_score"],
                block_count=block_count,
                dimension=self.config.hidden,
                epsilon=self.config.epsilon,
            )
            attention_input = session.hidden_scratch
        else:
            attention_input = session.input_row
        if is_snapshot:
            if token_id is not None:
                runtime.execute_copy(
                    session.residual_scratch, session.input_row, self.config.hidden
                )
            else:
                assert hidden is not None
                residuals[block_count] = hidden
        runtime.execute_rmsnorm(
            session.normalized,
            attention_input,
            weights["input_norm"],
            batch=1,
            dimension=self.config.hidden,
            epsilon=self.config.epsilon,
        )
        self.runner._execute_attention(
            resources,
            weights,
            session.attention_state,
            session.normalized,
            session.attention_output,
            layer=self.layer,
            position=cache_position_start,
            scratch=session.attention_scratch,
        )
        if is_snapshot:
            runtime.execute_copy(session.prefix_row, session.attention_output, self.config.hidden)
        else:
            runtime.execute_add(session.prefix_row, session.attention_output, self.config.hidden)
        if token_id is None:
            runtime.upload_activation(session.residual_scratch, residuals[:next_block_count])
        runtime.execute_attnres_mix(
            session.mixed,
            session.prefix_row,
            session.residual_scratch,
            weights["mlp_residual_score"],
            block_count=next_block_count,
            dimension=self.config.hidden,
            epsilon=self.config.epsilon,
        )
        runtime.execute_rmsnorm(
            session.mlp_input,
            session.mixed,
            weights["post_norm"],
            batch=1,
            dimension=self.config.hidden,
            epsilon=self.config.epsilon,
        )
        if weights["mlp_type"] == "moe":
            ids, selected_weights, effective = runtime.route(
                session.mlp_input,
                weights["router"],
                weights["router_bias"],
                hidden=self.config.hidden,
                experts=self.config.experts,
                topk=self.config.topk,
                accumulate_stats=self._telemetry_mode == "detailed",
            )
            if effective != self.config.topk:
                raise KimiCudaError("persistent Kimi route did not retain top-16")
            runtime.execute_dense(
                weights["latent_down"], session.latent_input, session.mlp_input, 1
            )
            if external_expert_dispatch is None:
                if self._telemetry_mode == "detailed":
                    runtime.profile_begin()
                for slot, expert in enumerate(ids.tolist()):
                    try:
                        expert_tensors = self._experts[int(expert)]
                    except KeyError as exc:
                        raise KimiCudaError(
                            f"selected expert {int(expert)} is not resident"
                        ) from exc
                    runtime.execute_resident(
                        expert_tensors,
                        _pointer_offset(session.expert_rows, slot * self.config.latent),
                        session.latent_input,
                        1,
                    )
                    routed_expert_execution_count += 1
                if self._telemetry_mode == "detailed":
                    routed_expert_device_ms = runtime.profile_end()
            else:
                external_started = time.perf_counter_ns()
                latent_download_started = time.perf_counter_ns()
                latent_host = runtime.download_activation(
                    session.latent_input, (self.config.latent,)
                )
                latent_download_ms = (time.perf_counter_ns() - latent_download_started) / 1e6
                external_rows, external_expert_record = external_expert_dispatch(
                    ids.copy(), latent_host
                )
                routed_expert_execution_count = len(ids)
                external_rows = np.ascontiguousarray(external_rows, dtype=np.float32)
                if external_rows.shape != (self.config.topk, self.config.latent):
                    raise KimiCudaError(
                        "external expert dispatch returned an invalid output geometry"
                    )
                if not np.isfinite(external_rows).all():
                    raise KimiCudaError("external expert dispatch returned non-finite output")
                expert_upload_started = time.perf_counter_ns()
                runtime.upload_activation(session.expert_rows, external_rows)
                external_expert_record = dict(external_expert_record)
                external_expert_record.update(
                    {
                        "parent_latent_d2h_ms": latent_download_ms,
                        "parent_expert_rows_h2d_enqueue_ms": (
                            time.perf_counter_ns() - expert_upload_started
                        )
                        / 1e6,
                        "parent_external_path_wall_ms": (time.perf_counter_ns() - external_started)
                        / 1e6,
                    }
                )
            runtime.upload_activation(
                session.route_weights,
                np.ascontiguousarray(selected_weights, dtype=np.float32),
            )
            runtime.execute_moe_reduction(
                session.reduced,
                session.expert_rows,
                session.route_weights,
                count=self.config.topk,
                dimension=self.config.latent,
            )
            runtime.execute_rmsnorm(
                session.reduced,
                session.reduced,
                weights["routed_norm"],
                batch=1,
                dimension=self.config.latent,
                epsilon=self.config.epsilon,
            )
            runtime.execute_dense(weights["latent_up"], session.routed_output, session.reduced, 1)
            if self._telemetry_mode == "detailed":
                runtime.profile_begin()
            runtime.execute_resident(
                weights["shared_mlp"], session.shared_output, session.mlp_input, 1
            )
            if self._telemetry_mode == "detailed":
                shared_expert_device_ms = runtime.profile_end()
            runtime.execute_add(session.routed_output, session.shared_output, self.config.hidden)
            runtime.execute_add(session.prefix_row, session.routed_output, self.config.hidden)
        else:
            ids = np.empty((0,), dtype=np.int32)
            selected_weights = np.empty((0,), dtype=np.float32)
            self.runner._execute_dense_mlp(weights, session.mlp_input, session.routed_output)
            runtime.execute_add(session.prefix_row, session.routed_output, self.config.hidden)
        if self._owns_final_endpoint:
            runtime.upload_activation(session.residual_scratch, residuals[:next_block_count])
            runtime.execute_attnres_mix(
                session.final_mixed,
                session.prefix_row,
                session.residual_scratch,
                self._final_score,
                block_count=next_block_count,
                dimension=self.config.hidden,
                epsilon=self.config.epsilon,
            )
            runtime.execute_rmsnorm(
                session.final_normalized,
                session.final_mixed,
                self._final_norm,
                batch=1,
                dimension=self.config.hidden,
                epsilon=self.config.epsilon,
            )
            runtime.execute_dense(self._head, session.logits, session.final_normalized, 1)
        runtime.synchronize()
        device_ms = runtime.profile_end() if whole_stage_profile else None
        layer_output = runtime.download_activation(session.prefix_row, (self.config.hidden,))
        if token_id is not None:
            residuals[block_count] = runtime.download_activation(
                session.residual_scratch, (self.config.hidden,)
            )
        boundary_output = np.empty((1, 9, self.config.hidden), dtype=np.float32)
        boundary_output[0, 0] = layer_output
        boundary_output[0, 1:] = residuals
        final_hidden = (
            runtime.download_activation(session.final_normalized, (self.config.hidden,))
            if self._owns_final_endpoint
            else None
        )
        logits = (
            runtime.download_activation(session.logits, (self.config.vocab,))
            if self._owns_final_endpoint
            else None
        )
        wall_ns = time.perf_counter_ns() - wall_started
        sampled_id = int(np.argmax(logits)) if logits is not None else None
        session.cache_length += 1
        self._execute_count += 1
        record = {
            "position": cache_position_start,
            "host_thread_native_id": threading.get_native_id(),
            "host_thread_ident": threading.get_ident(),
            "host_thread_name": threading.current_thread().name,
            "selected_expert_ids": [int(value) for value in ids],
            "selected_weights": [float(value) for value in selected_weights],
            "routed_expert_execution_count": routed_expert_execution_count,
            "all_selected_experts_executed_once": (
                routed_expert_execution_count == len(ids)
            ),
            "layer_output": layer_output,
            "boundary_output": boundary_output,
            "final_hidden": final_hidden,
            "logits": logits,
            "sampled_token_id": sampled_id,
            "wall_ms": wall_ns / 1e6,
            "device_ms": device_ms,
            "routed_expert_device_ms": routed_expert_device_ms,
            "shared_expert_device_ms": shared_expert_device_ms,
            "external_expert_dispatch": external_expert_record,
            "device_measurement_scope": (
                "whole_stage" if whole_stage_profile else "routed_and_shared_expert_phases_only"
            ),
            "telemetry_mode": self._telemetry_mode,
            "persistent_buffer_allocations_during_execute": (
                self._persistent_buffer_allocation_count - allocation_before
            ),
            "weight_loads_during_execute": self._weight_load_count - weight_load_before,
            "materializations_during_execute": (
                self._model_materialization_count - materialization_before
            ),
        }
        self.execution_records.append(record)
        sampled = torch.tensor([sampled_id], dtype=torch.int64) if sampled_id is not None else None
        layer_tensor = torch.from_numpy(layer_output.copy()).reshape(1, 1, -1)
        boundary_tensor = torch.from_numpy(boundary_output.copy())
        final_tensor = (
            torch.from_numpy(final_hidden.copy()).reshape(1, 1, -1)
            if final_hidden is not None
            else None
        )
        logits_tensor = (
            torch.from_numpy(logits.copy()).reshape(1, 1, -1) if logits is not None else None
        )
        return StageExecutionResult(
            hidden_states=layer_tensor,
            stage_boundary_hidden_states=boundary_tensor,
            router_logits=(),
            final_hidden_states=final_tensor,
            logits=logits_tensor,
            sampled_token_ids=sampled,
            all_sampled_token_ids=sampled,
            cache_sequence_length=session.cache_length,
            compute_ns=wall_ns,
            expert_events=(
                (
                    {
                        "layer_id": self.layer,
                        "selected_expert_ids": [int(value) for value in ids],
                        "selected_weights": [float(value) for value in selected_weights],
                    },
                )
                if weights["mlp_type"] == "moe"
                else ()
            ),
            expert_metrics={
                "backend_identity": "nvidia_cuda_persistent_kimi_stage",
                "cpu_mathematical_fallbacks": 0,
                "resident_expert_count": len(self._experts),
                "device_ms": device_ms,
                "persistent_buffer_allocations_during_execute": 0,
                "weight_loads_during_execute": 0,
                "model_materializations_during_execute": 0,
            },
        )

    def _execute_dense_attention_batch(
        self,
        tensor: ctypes.c_void_p,
        output: ctypes.c_void_p,
        source: ctypes.c_void_p,
        *,
        batch: int,
        input_dimension: int,
        output_dimension: int,
    ) -> None:
        """Execute one attention projection with exact row-reuse chunks."""
        if self.runtime.dense_rows_reuse_function is None:
            self.runtime.execute_dense(tensor, output, source, batch)
            return
        offset = 0
        remaining = batch
        for size in (8, 4, 2, 1):
            while remaining >= size:
                self.runtime.execute_dense_rows_reuse(
                    tensor,
                    _pointer_offset(output, offset * output_dimension),
                    _pointer_offset(source, offset * input_dimension),
                    size,
                )
                offset += size
                remaining -= size
        if remaining:
            raise KimiCudaError("attention batch could not be partitioned safely")

    def _execute_attention_batch(
        self,
        sessions: list[_FinalStageSession],
        normalized_rows: ctypes.c_void_p,
        output_rows: ctypes.c_void_p,
        *,
        batch: int,
        positions: tuple[int, ...],
        dcp_degree: int = 1,
    ) -> None:
        """Batch stateless projections while preserving session-owned state."""
        if len(positions) != batch:
            raise ValueError("Kimi attention batch positions must match the batch")
        if dcp_degree not in (1, 2, 4, 8):
            raise ValueError("Kimi DCP degree must be one of 1, 2, 4 or 8")
        runtime = self.runtime
        config = self.config
        weights = self._weights
        scratch = self._batch_attention_scratch

        def dense(
            role: str,
            destination: str | ctypes.c_void_p,
            source: str | ctypes.c_void_p,
            input_dimension: int,
            output_dimension: int,
        ) -> None:
            output = scratch[destination] if isinstance(destination, str) else destination
            inputs = scratch[source] if isinstance(source, str) else source
            self._execute_dense_attention_batch(
                weights[role],
                output,
                inputs,
                batch=batch,
                input_dimension=input_dimension,
                output_dimension=output_dimension,
            )

        if self.layer in config.kda_layers:
            projection = config.kda_projection
            for role in ("q", "k", "v", "gate"):
                dense(role, role, normalized_rows, config.hidden, projection)
            dense(
                "decay_a",
                "decay_low",
                normalized_rows,
                config.hidden,
                config.kda_head_dimension,
            )
            dense(
                "decay_b",
                "decay",
                "decay_low",
                config.kda_head_dimension,
                projection,
            )
            dense(
                "beta",
                "beta",
                normalized_rows,
                config.hidden,
                config.kda_heads,
            )
            if self._kda_short_window_enabled:
                if any(session is not sessions[0] for session in sessions[1:]):
                    raise KimiCudaError(
                        "short-window KDA requires one session with contiguous rows"
                    )
                runtime.execute_kda_short_window(
                    scratch["core"],
                    scratch["q"],
                    scratch["k"],
                    scratch["v"],
                    scratch["gate"],
                    scratch["decay"],
                    scratch["beta"],
                    weights["conv_q"],
                    weights["conv_k"],
                    weights["conv_v"],
                    sessions[0].attention_state["window_q"],
                    sessions[0].attention_state["window_k"],
                    sessions[0].attention_state["window_v"],
                    sessions[0].attention_state["state"],
                    weights["dt"],
                    weights["a"],
                    weights["output_norm"],
                    rows=batch,
                    heads=config.kda_heads,
                    head_dimension=config.kda_head_dimension,
                    convolution_width=config.convolution_width,
                    gate_lower_bound=config.gate_lower_bound,
                    epsilon=config.epsilon,
                )
            else:
                for row, session in enumerate(sessions):
                    runtime.execute_kda_core(
                        _pointer_offset(scratch["core"], row * projection),
                        _pointer_offset(scratch["q"], row * projection),
                        _pointer_offset(scratch["k"], row * projection),
                        _pointer_offset(scratch["v"], row * projection),
                        _pointer_offset(scratch["gate"], row * projection),
                        _pointer_offset(scratch["decay"], row * projection),
                        _pointer_offset(scratch["beta"], row * config.kda_heads),
                        weights["conv_q"],
                        weights["conv_k"],
                        weights["conv_v"],
                        session.attention_state["window_q"],
                        session.attention_state["window_k"],
                        session.attention_state["window_v"],
                        session.attention_state["state"],
                        weights["dt"],
                        weights["a"],
                        weights["output_norm"],
                        heads=config.kda_heads,
                        head_dimension=config.kda_head_dimension,
                        convolution_width=config.convolution_width,
                        gate_lower_bound=config.gate_lower_bound,
                        epsilon=config.epsilon,
                    )
            dense(
                "output",
                output_rows,
                "core",
                projection,
                config.hidden,
            )
            return

        dense(
            "query_a",
            "query_low",
            normalized_rows,
            config.hidden,
            config.query_lora,
        )
        runtime.execute_rmsnorm(
            scratch["query_low"],
            scratch["query_low"],
            weights["query_norm"],
            batch=batch,
            dimension=config.query_lora,
            epsilon=config.epsilon,
        )
        dense(
            "query_b",
            "query",
            "query_low",
            config.query_lora,
            config.query_dimension,
        )
        dense(
            "kv_a",
            "compressed_kv",
            normalized_rows,
            config.hidden,
            config.kv_lora + config.query_rope,
        )
        dense(
            "gate",
            "mla_gate",
            normalized_rows,
            config.hidden,
            config.context_dimension,
        )
        for row, session in enumerate(sessions):
            position = positions[row]
            runtime.execute_mla_cache_append(
                _pointer_offset(session.attention_state["latent_cache"], position * config.kv_lora),
                _pointer_offset(
                    session.attention_state["rope_cache"], position * config.query_rope
                ),
                _pointer_offset(
                    scratch["compressed_kv"],
                    row * (config.kv_lora + config.query_rope),
                ),
                weights["kv_norm"],
                kv_lora=config.kv_lora,
                rope_dimension=config.query_rope,
                epsilon=config.epsilon,
            )
        one_session = all(session is sessions[0] for session in sessions)
        contiguous = positions == tuple(range(positions[0], positions[0] + batch))
        final_context_length = positions[-1] + 1
        use_dcp_kernel = dcp_degree > 1 or final_context_length > 16_385
        if (
            batch > 1
            and one_session
            and contiguous
            and use_dcp_kernel
        ):
            if getattr(runtime, "mla_absorb_dcp_function", None) is None:
                raise KimiCudaError("Kimi CUDA binary has no DCP MLA export")
            session = sessions[0]
            runtime.execute_mla_absorb_dcp(
                weights["kv_b"],
                scratch["context"],
                scratch["query"],
                session.attention_state["latent_cache"],
                session.attention_state["rope_cache"],
                batch=batch,
                degree=dcp_degree,
                heads=config.heads,
                query_nope=config.query_nope,
                query_rope=config.query_rope,
                value_dimension=config.value_dimension,
                kv_lora=config.kv_lora,
                final_context_length=final_context_length,
                attention_scale=config.attention_scale,
            )
        elif (
            batch > 1
            and one_session
            and contiguous
            and getattr(runtime, "mla_absorb_triangular_function", None) is not None
        ):
            session = sessions[0]
            runtime.execute_mla_absorb_triangular(
                weights["kv_b"],
                scratch["context"],
                scratch["query"],
                session.attention_state["latent_cache"],
                session.attention_state["rope_cache"],
                batch=batch,
                heads=config.heads,
                query_nope=config.query_nope,
                query_rope=config.query_rope,
                value_dimension=config.value_dimension,
                kv_lora=config.kv_lora,
                final_context_length=final_context_length,
                attention_scale=config.attention_scale,
            )
        else:
            for row, session in enumerate(sessions):
                runtime.execute_mla_absorb(
                    weights["kv_b"],
                    _pointer_offset(scratch["context"], row * config.context_dimension),
                    _pointer_offset(scratch["query"], row * config.query_dimension),
                    session.attention_state["latent_cache"],
                    session.attention_state["rope_cache"],
                    heads=config.heads,
                    query_nope=config.query_nope,
                    query_rope=config.query_rope,
                    value_dimension=config.value_dimension,
                    kv_lora=config.kv_lora,
                    context_length=positions[row] + 1,
                    attention_scale=config.attention_scale,
                )
        runtime.execute_mla_gate(
            scratch["context"],
            scratch["mla_gate"],
            batch * config.context_dimension,
        )
        dense(
            "output",
            output_rows,
            "context",
            config.context_dimension,
            config.hidden,
        )

    def execute_decode_batch(
        self,
        *,
        session_ids: tuple[str, ...],
        hidden_states: torch.Tensor,
        cache_position_start: int | None = None,
        cache_position_starts: tuple[int, ...] | None = None,
        profile_phases: bool = False,
        external_expert_dispatch: Any | None = None,
        overlap_parent_shared: bool = False,
    ) -> dict[str, Any]:
        """Execute one state-isolated static Kimi batch with real-route weight reuse.

        Attention remains session-owned. Rows selecting the same routed expert are
        gathered into exact native batch sizes, run once per group, and scattered
        back before the unchanged deterministic per-row reduction.
        """
        batch = len(session_ids)
        if self._owns_embeddings or self._owns_final_endpoint:
            raise RuntimeError(
                "row-cooperative decode currently requires a non-endpoint Kimi layer"
            )
        if self._weights["mlp_type"] != "moe":
            raise RuntimeError("row-cooperative decode requires a routed-MoE layer")
        if overlap_parent_shared and external_expert_dispatch is None:
            raise ValueError("parent shared overlap requires an external collective")
        if overlap_parent_shared and not all(
            callable(getattr(external_expert_dispatch, name, None))
            for name in ("start_batch", "collect_batch")
        ):
            raise TypeError("parent shared overlap requires persistent start_batch/collect_batch")
        if batch <= 0 or batch > self._batch_capacity:
            raise KimiCudaError(
                "Kimi complete-stage batch rejected before CUDA work: "
                f"requested={batch}, certified_max={self._batch_capacity}"
            )
        if len(set(session_ids)) != batch:
            raise ValueError("Kimi complete-stage batch session IDs must be unique")
        if (cache_position_start is None) == (cache_position_starts is None):
            raise ValueError("provide exactly one of cache_position_start or cache_position_starts")
        positions = (
            (int(cache_position_start),) * batch
            if cache_position_starts is None
            else tuple(int(value) for value in cache_position_starts)
        )
        if len(positions) != batch:
            raise ValueError("Kimi batch cache positions must match the session count")
        if hidden_states.dtype != torch.float32 or tuple(hidden_states.shape) != (
            batch,
            9,
            self.config.hidden,
        ):
            raise ValueError(f"Kimi stage batch boundary must be float32 [{batch},9,7168]")
        sessions = [self._require_session(session_id) for session_id in session_ids]
        for session, position in zip(sessions, positions, strict=True):
            if position != session.cache_length:
                raise ValueError("Kimi stage batch cache position is not contiguous")
            if position >= session.maximum_context:
                raise ValueError("Kimi stage batch context exceeds its prepared cache")

        values = np.ascontiguousarray(hidden_states.detach().cpu().numpy(), dtype=np.float32)
        block_count = (self.layer + self.config.residual_block - 1) // (self.config.residual_block)
        is_snapshot = self.layer % self.config.residual_block == 0
        next_block_count = block_count + int(is_snapshot)
        if not 0 <= block_count <= 8 or not 1 <= next_block_count <= 8:
            raise KimiCudaError("persistent stage AttnRes block count is invalid")

        runtime = self.runtime
        weights = self._weights
        hidden = self.config.hidden
        latent = self.config.latent
        topk = self.config.topk
        allocation_before = self._persistent_buffer_allocation_count
        weight_load_before = self._weight_load_count
        materialization_before = self._model_materialization_count
        wall_started = time.perf_counter_ns()
        phase_device_ms: dict[str, float] = {}
        phase_wall_ms: dict[str, float] = {}

        def run_phase(name: str, action: Callable[[], None]) -> None:
            phase_started = time.perf_counter_ns()
            if profile_phases:
                runtime.profile_begin()
            try:
                action()
            except BaseException:
                # A transport/cancellation failure can interrupt a phase after
                # the enclosing CUDA event interval has begun. Close that
                # interval before propagating the original error so a safe
                # request can reuse the persistent runtime without rebuilding
                # its CUDA context.
                with contextlib.suppress(BaseException):
                    runtime.profile_end()
                raise
            if profile_phases:
                phase_device_ms[name] = runtime.profile_end()
            phase_wall_ms[name] = (time.perf_counter_ns() - phase_started) / 1e6

        if not profile_phases:
            runtime.profile_begin()

        residual_rows = [values[row, 1:].copy() for row in range(batch)]

        def attention_and_pre_moe() -> None:
            for row, session in enumerate(sessions):
                row_hidden = values[row, 0]
                residuals = residual_rows[row]
                runtime.upload_activation(session.input_row, row_hidden)
                runtime.upload_activation(session.prefix_row, row_hidden)
                if block_count:
                    runtime.upload_activation(session.residual_scratch, residuals[:block_count])
                    runtime.execute_attnres_mix(
                        session.hidden_scratch,
                        session.prefix_row,
                        session.residual_scratch,
                        weights["attention_residual_score"],
                        block_count=block_count,
                        dimension=hidden,
                        epsilon=self.config.epsilon,
                    )
                    attention_input = session.hidden_scratch
                else:
                    attention_input = session.input_row
                if is_snapshot:
                    residuals[block_count] = row_hidden
                runtime.execute_rmsnorm(
                    _pointer_offset(self._batch_hidden_input, row * hidden),
                    attention_input,
                    weights["input_norm"],
                    batch=1,
                    dimension=hidden,
                    epsilon=self.config.epsilon,
                )
            self._execute_attention_batch(
                sessions,
                self._batch_hidden_input,
                self._batch_hidden_output,
                batch=batch,
                positions=positions,
            )
            for row, session in enumerate(sessions):
                attention_output = _pointer_offset(self._batch_hidden_output, row * hidden)
                residuals = residual_rows[row]
                if is_snapshot:
                    runtime.execute_copy(session.prefix_row, attention_output, hidden)
                else:
                    runtime.execute_add(session.prefix_row, attention_output, hidden)
                runtime.upload_activation(session.residual_scratch, residuals[:next_block_count])
                runtime.execute_attnres_mix(
                    session.mixed,
                    session.prefix_row,
                    session.residual_scratch,
                    weights["mlp_residual_score"],
                    block_count=next_block_count,
                    dimension=hidden,
                    epsilon=self.config.epsilon,
                )
                runtime.execute_rmsnorm(
                    session.mlp_input,
                    session.mixed,
                    weights["post_norm"],
                    batch=1,
                    dimension=hidden,
                    epsilon=self.config.epsilon,
                )

        run_phase("attention_and_pre_moe", attention_and_pre_moe)

        selected_ids: list[np.ndarray] = []
        selected_weights: list[np.ndarray] = []
        router_mode = (
            "certified_native_chunks"
            if runtime.router_batch_function is not None
            else "row_serial_native"
        )

        def router() -> None:
            for row, session in enumerate(sessions):
                runtime.execute_copy(
                    _pointer_offset(self._batch_hidden_input, row * hidden),
                    session.mlp_input,
                    hidden,
                )
            if runtime.router_batch_function is not None:
                ids_batch, weights_batch, effective_batch = self._route_batch_chunked(
                    self._batch_hidden_input,
                    batch=batch,
                    accumulate_stats=profile_phases,
                )
                for row in range(batch):
                    ids = ids_batch[row]
                    route_weights = weights_batch[row]
                    if int(effective_batch[row]) != topk or len(set(ids.tolist())) != topk:
                        raise KimiCudaError(
                            "row-cooperative Kimi route did not retain 16 unique experts"
                        )
                    selected_ids.append(ids.copy())
                    selected_weights.append(route_weights.copy())
                return
            for session in sessions:
                ids, route_weights, effective = runtime.route(
                    session.mlp_input,
                    weights["router"],
                    weights["router_bias"],
                    hidden=hidden,
                    experts=self.config.experts,
                    topk=topk,
                    accumulate_stats=profile_phases,
                )
                if effective != topk or len(set(ids.tolist())) != topk:
                    raise KimiCudaError(
                        "row-cooperative Kimi route did not retain 16 unique experts"
                    )
                selected_ids.append(ids.copy())
                selected_weights.append(route_weights.copy())

        run_phase("router", router)

        grouped_plan = build_grouped_expert_plan(
            np.stack(selected_ids, axis=0),
            supported_batch_sizes=(
                size
                for size in self.runtime.expert_supported_batches
                if size <= self._batch_capacity
            ),
        )
        task_plan = [
            (
                assignment.expert_id,
                assignment.row,
                assignment.slot,
                assignment.work_index,
            )
            for assignment in grouped_plan.assignments
        ]
        expert_groups = [
            {
                "expert_id": group.expert_id,
                "start": group.start,
                "rows": group.count,
                "chunks": group.native_chunks,
            }
            for group in grouped_plan.groups
        ]
        native_expert_calls = grouped_plan.native_call_count
        if len(task_plan) != batch * topk:
            raise KimiCudaError("row-cooperative Kimi dispatch omitted selected experts")

        def latent_down() -> None:
            runtime.execute_dense(
                weights["latent_down"],
                self._batch_latent_rows,
                self._batch_hidden_input,
                batch,
            )

        run_phase("latent_down", latent_down)

        shared_chunks = self._supported_batch_chunks(batch)

        def shared_expert() -> None:
            offset = 0
            for chunk in shared_chunks:
                runtime.execute_resident(
                    weights["shared_mlp"],
                    _pointer_offset(self._batch_hidden_output, offset * hidden),
                    _pointer_offset(self._batch_hidden_input, offset * hidden),
                    chunk,
                )
                offset += chunk
            for row, session in enumerate(sessions):
                runtime.execute_copy(
                    session.shared_output,
                    _pointer_offset(self._batch_hidden_output, row * hidden),
                    hidden,
                )

        external_record: dict[str, Any] | None = None
        shared_expert_executed = False
        if external_expert_dispatch is None:
            if len(self._expert_ownership) != self.config.experts:
                raise KimiCudaError(
                    "Kimi batch routed execution requires local experts or an external dispatcher"
                )

            def expert_dispatch() -> None:
                for _expert, row, _slot, work_index in task_plan:
                    runtime.execute_copy(
                        _pointer_offset(self._batch_expert_input, work_index * latent),
                        _pointer_offset(self._batch_latent_rows, row * latent),
                        latent,
                    )

            run_phase("expert_dispatch", expert_dispatch)

            def routed_expert_compute() -> None:
                for group in expert_groups:
                    offset = int(group["start"])
                    for chunk in group["chunks"]:
                        runtime.execute_resident(
                            self._experts[int(group["expert_id"])],
                            _pointer_offset(self._batch_expert_output, offset * latent),
                            _pointer_offset(self._batch_expert_input, offset * latent),
                            int(chunk),
                        )
                        offset += int(chunk)

            run_phase("routed_expert_compute", routed_expert_compute)

            def expert_collection() -> None:
                for _expert, row, slot, work_index in task_plan:
                    runtime.execute_copy(
                        _pointer_offset(sessions[row].expert_rows, slot * latent),
                        _pointer_offset(self._batch_expert_output, work_index * latent),
                        latent,
                    )

            run_phase("expert_collection", expert_collection)
        else:

            def install_external_rows(expert_rows: np.ndarray, record: dict[str, Any]) -> None:
                nonlocal external_record, native_expert_calls
                external_record = record
                observed = np.ascontiguousarray(expert_rows, dtype=np.float32)
                if observed.shape != (batch, topk, latent) or not np.isfinite(observed).all():
                    raise KimiCudaError(
                        "external Kimi batch expert collective returned invalid geometry"
                    )
                for row, session in enumerate(sessions):
                    runtime.upload_activation(session.expert_rows, observed[row])
                native_expert_calls = int(
                    external_record.get("native_expert_calls", native_expert_calls)
                )

            if overlap_parent_shared:
                external_handle: Any | None = None

                def external_expert_start() -> None:
                    nonlocal external_handle
                    latent_rows = runtime.download_activation(
                        self._batch_latent_rows, (batch, latent)
                    )
                    external_handle = external_expert_dispatch.start_batch(
                        np.stack(selected_ids, axis=0), latent_rows
                    )

                run_phase("external_expert_start", external_expert_start)
                run_phase("shared_expert_overlap", shared_expert)
                shared_expert_executed = True

                def external_expert_collect() -> None:
                    if external_handle is None:
                        raise KimiCudaError("external expert collective was not started")
                    expert_rows, record = external_expert_dispatch.collect_batch(external_handle)
                    install_external_rows(expert_rows, record)

                run_phase("external_expert_collect", external_expert_collect)
            else:

                def external_expert_collective() -> None:
                    latent_rows = runtime.download_activation(
                        self._batch_latent_rows, (batch, latent)
                    )
                    expert_rows, record = external_expert_dispatch(
                        np.stack(selected_ids, axis=0), latent_rows
                    )
                    install_external_rows(expert_rows, record)

                run_phase("external_expert_collective", external_expert_collective)

        def reduction() -> None:
            for row, session in enumerate(sessions):
                runtime.upload_activation(
                    session.route_weights,
                    np.ascontiguousarray(selected_weights[row], dtype=np.float32),
                )
                runtime.execute_moe_reduction(
                    session.reduced,
                    session.expert_rows,
                    session.route_weights,
                    count=topk,
                    dimension=latent,
                )
                runtime.execute_rmsnorm(
                    session.reduced,
                    session.reduced,
                    weights["routed_norm"],
                    batch=1,
                    dimension=latent,
                    epsilon=self.config.epsilon,
                )

        run_phase("reduction", reduction)

        def latent_up() -> None:
            for row, session in enumerate(sessions):
                runtime.execute_copy(
                    _pointer_offset(self._batch_latent_rows, row * latent),
                    session.reduced,
                    latent,
                )
            runtime.execute_dense(
                weights["latent_up"],
                self._batch_hidden_output,
                self._batch_latent_rows,
                batch,
            )
            for row, session in enumerate(sessions):
                runtime.execute_copy(
                    session.routed_output,
                    _pointer_offset(self._batch_hidden_output, row * hidden),
                    hidden,
                )

        run_phase("latent_up", latent_up)

        if not shared_expert_executed:
            run_phase("shared_expert", shared_expert)

        def residual() -> None:
            for session in sessions:
                runtime.execute_add(session.routed_output, session.shared_output, hidden)
                runtime.execute_add(session.prefix_row, session.routed_output, hidden)

        run_phase("residual", residual)
        runtime.synchronize()
        device_ms = None if profile_phases else runtime.profile_end()

        output_rows: list[np.ndarray] = []
        for row, session in enumerate(sessions):
            layer_output = runtime.download_activation(session.prefix_row, (hidden,))
            boundary_output = np.empty((9, hidden), dtype=np.float32)
            boundary_output[0] = layer_output
            boundary_output[1:] = residual_rows[row]
            output_rows.append(boundary_output)
            session.cache_length += 1
        boundaries = np.stack(output_rows, axis=0)
        wall_ns = time.perf_counter_ns() - wall_started
        self._execute_count += batch
        self._batch_execute_count += 1
        total_selections = batch * topk
        unique_experts = len(expert_groups)
        group_histogram: dict[str, int] = {}
        for group in expert_groups:
            key = str(group["rows"])
            group_histogram[key] = group_histogram.get(key, 0) + 1
        record: dict[str, Any] = {
            "position": positions[0] if len(set(positions)) == 1 else None,
            "positions": list(positions),
            "host_thread_native_id": threading.get_native_id(),
            "host_thread_ident": threading.get_ident(),
            "host_thread_name": threading.current_thread().name,
            "batch_size": batch,
            "session_ids": list(session_ids),
            "selected_expert_ids": [[int(value) for value in ids] for ids in selected_ids],
            "selected_weights": [
                [float(value) for value in route_weights] for route_weights in selected_weights
            ],
            "boundary_output": boundaries,
            "wall_ms": wall_ns / 1e6,
            "device_ms": device_ms,
            "phase_device_ms": phase_device_ms,
            "phase_wall_ms": phase_wall_ms,
            "profile_phases": profile_phases,
            "router_mode": router_mode,
            "routing": {
                "total_selections": total_selections,
                "unique_experts": unique_experts,
                "repeated_expert_hits": total_selections - unique_experts,
                "maximum_rows_for_one_expert": max(int(group["rows"]) for group in expert_groups),
                "expert_group_size_histogram": group_histogram,
                "native_routed_expert_calls": native_expert_calls,
                "effective_weight_reuse_rows_per_native_call": (
                    total_selections / native_expert_calls
                ),
                "avoided_routed_expert_weight_launches": (total_selections - native_expert_calls),
                "all_selected_experts_executed_once": len(task_plan) == total_selections,
            },
            "shared_expert_native_calls": len(shared_chunks),
            "parent_shared_overlap": overlap_parent_shared,
            "external_expert_collective": external_record,
            "transfers": {
                "expert_dispatch_d2d_bytes": total_selections * latent * 4,
                "expert_collection_d2d_bytes": total_selections * latent * 4,
                "batch_gather_scatter_d2d_bytes": batch * (3 * hidden + latent) * 4,
                "boundary_d2h_bytes": batch * hidden * 4,
                "route_weight_h2d_bytes": batch * topk * 4,
                "message_count": int(
                    external_record.get("messages", 0) if external_record is not None else 0
                ),
                "network_bytes": int(
                    external_record.get("total_transport_bytes", 0)
                    if external_record is not None
                    else 0
                ),
            },
            "persistent_buffer_allocations_during_execute": (
                self._persistent_buffer_allocation_count - allocation_before
            ),
            "weight_loads_during_execute": self._weight_load_count - weight_load_before,
            "materializations_during_execute": self._model_materialization_count
            - materialization_before,
        }
        self.execution_records.append(record)
        return record

    def execute_verification_block(
        self,
        *,
        block: VerificationBlock,
        hidden_states: torch.Tensor,
        expert_strategy: str = "expert_major",
        profile_phases: bool = False,
        dcp_degree: int = 1,
    ) -> dict[str, Any]:
        """Execute contiguous candidate positions as one exact scheduling unit.

        Unlike :meth:`execute_decode_batch`, all rows belong to one session and
        therefore update one KDA/MLA state in position order.  Stateless
        projections and MoE work are row-cooperative. KDA retains its canonical
        sequential state dependency; MLA may schedule the known contiguous
        queries together, including an exact context-sharded reduction.
        """

        if not self._verification_major_enabled:
            raise RuntimeError(
                "verification-major execution requires a verification-major fast path"
            )
        if self._owns_embeddings or self._owns_final_endpoint:
            raise RuntimeError(
                "verification-major decode currently requires a non-endpoint Kimi layer"
            )
        if self._weights["mlp_type"] != "moe":
            raise RuntimeError("verification-major decode requires a routed-MoE layer")
        if expert_strategy not in {"token_major", "expert_major"}:
            raise ValueError("expert strategy must be 'token_major' or 'expert_major'")
        if dcp_degree not in (1, 2, 4, 8):
            raise ValueError("verification DCP degree must be one of 1, 2, 4 or 8")
        rows = block.row_count
        if rows > self._batch_capacity:
            raise KimiCudaError(
                "Kimi verification block rejected before CUDA work: "
                f"requested_rows={rows}, prepared_rows={self._batch_capacity}"
            )
        if hidden_states.dtype != torch.float32 or tuple(hidden_states.shape) != (
            rows,
            9,
            self.config.hidden,
        ):
            raise ValueError(
                f"Kimi verification boundary must be float32 [{rows},9,7168]"
            )
        session = self._require_session(block.session_id)
        positions = block.positions
        if block.cache_position_start != session.cache_length:
            raise ValueError("Kimi verification block cache position is not contiguous")
        if positions[-1] >= session.maximum_context:
            raise ValueError("Kimi verification block exceeds its prepared cache")

        values = np.ascontiguousarray(hidden_states.detach().cpu().numpy(), dtype=np.float32)
        block_count = (self.layer + self.config.residual_block - 1) // (
            self.config.residual_block
        )
        is_snapshot = self.layer % self.config.residual_block == 0
        next_block_count = block_count + int(is_snapshot)
        if not 0 <= block_count <= 8 or not 1 <= next_block_count <= 8:
            raise KimiCudaError("persistent stage AttnRes block count is invalid")

        runtime = self.runtime
        weights = self._weights
        hidden = self.config.hidden
        latent = self.config.latent
        topk = self.config.topk
        boundary_row_elements = 9 * hidden
        allocation_before = self._persistent_buffer_allocation_count
        weight_load_before = self._weight_load_count
        materialization_before = self._model_materialization_count
        memory_before = runtime.mem_info()
        wall_started = time.perf_counter_ns()
        phase_device_ms: dict[str, float] = {}
        phase_wall_ms: dict[str, float] = {}
        result_boundary: np.ndarray | None = None
        whole_profile_open = False

        def run_phase(name: str, action: Callable[[], None]) -> None:
            phase_started = time.perf_counter_ns()
            if profile_phases:
                runtime.profile_begin()
            try:
                action()
            except BaseException:
                if profile_phases:
                    with contextlib.suppress(BaseException):
                        runtime.profile_end()
                raise
            if profile_phases:
                phase_device_ms[name] = runtime.profile_end()
            phase_wall_ms[name] = (time.perf_counter_ns() - phase_started) / 1e6

        if not profile_phases:
            runtime.profile_begin()
            whole_profile_open = True

        try:
            run_phase(
                "boundary_h2d",
                lambda: runtime.upload_activation(self._batch_boundary, values),
            )

            def attention_and_pre_moe() -> None:
                for row in range(rows):
                    row_boundary = _pointer_offset(
                        self._batch_boundary, row * boundary_row_elements
                    )
                    row_hidden = row_boundary
                    runtime.execute_copy(session.prefix_row, row_hidden, hidden)
                    if block_count:
                        runtime.execute_attnres_mix(
                            session.hidden_scratch,
                            session.prefix_row,
                            _pointer_offset(row_boundary, hidden),
                            weights["attention_residual_score"],
                            block_count=block_count,
                            dimension=hidden,
                            epsilon=self.config.epsilon,
                        )
                        attention_input = session.hidden_scratch
                    else:
                        attention_input = row_hidden
                    if is_snapshot:
                        runtime.execute_copy(
                            _pointer_offset(
                                row_boundary, (1 + block_count) * hidden
                            ),
                            row_hidden,
                            hidden,
                        )
                    runtime.execute_rmsnorm(
                        _pointer_offset(self._batch_hidden_input, row * hidden),
                        attention_input,
                        weights["input_norm"],
                        batch=1,
                        dimension=hidden,
                        epsilon=self.config.epsilon,
                    )

                self._execute_attention_batch(
                    [session] * rows,
                    self._batch_hidden_input,
                    self._batch_hidden_output,
                    batch=rows,
                    positions=positions,
                    dcp_degree=dcp_degree,
                )

                for row in range(rows):
                    row_boundary = _pointer_offset(
                        self._batch_boundary, row * boundary_row_elements
                    )
                    attention_output = _pointer_offset(
                        self._batch_hidden_output, row * hidden
                    )
                    if is_snapshot:
                        runtime.execute_copy(session.prefix_row, attention_output, hidden)
                    else:
                        runtime.execute_copy(session.prefix_row, row_boundary, hidden)
                        runtime.execute_add(session.prefix_row, attention_output, hidden)
                    runtime.execute_attnres_mix(
                        session.mixed,
                        session.prefix_row,
                        _pointer_offset(row_boundary, hidden),
                        weights["mlp_residual_score"],
                        block_count=next_block_count,
                        dimension=hidden,
                        epsilon=self.config.epsilon,
                    )
                    runtime.execute_rmsnorm(
                        _pointer_offset(self._batch_hidden_input, row * hidden),
                        session.mixed,
                        weights["post_norm"],
                        batch=1,
                        dimension=hidden,
                        epsilon=self.config.epsilon,
                    )
                    runtime.execute_copy(attention_output, session.prefix_row, hidden)

            run_phase("attention_and_pre_moe", attention_and_pre_moe)

            selected_ids: list[np.ndarray] = []
            selected_weights: list[np.ndarray] = []

            def router() -> None:
                if runtime.router_batch_function is not None:
                    ids_batch, weights_batch, effective_batch = self._route_batch_chunked(
                        self._batch_hidden_input,
                        batch=rows,
                        accumulate_stats=profile_phases,
                    )
                    for row in range(rows):
                        ids = ids_batch[row]
                        route_weights = weights_batch[row]
                        if int(effective_batch[row]) != topk or len(set(ids.tolist())) != topk:
                            raise KimiCudaError(
                                "verification-major Kimi route did not retain 16 unique experts"
                            )
                        selected_ids.append(ids.copy())
                        selected_weights.append(route_weights.copy())
                    return
                for row in range(rows):
                    ids, route_weights, effective = runtime.route(
                        _pointer_offset(self._batch_hidden_input, row * hidden),
                        weights["router"],
                        weights["router_bias"],
                        hidden=hidden,
                        experts=self.config.experts,
                        topk=topk,
                        accumulate_stats=profile_phases,
                    )
                    if effective != topk or len(set(ids.tolist())) != topk:
                        raise KimiCudaError(
                            "verification-major Kimi route did not retain 16 unique experts"
                        )
                    selected_ids.append(ids.copy())
                    selected_weights.append(route_weights.copy())

            run_phase("router", router)
            route_matrix = np.stack(selected_ids, axis=0)
            grouped_plan = build_grouped_expert_plan(
                route_matrix,
                supported_batch_sizes=(
                    size
                    for size in runtime.expert_supported_batches
                    if size <= self._batch_capacity
                ),
            )

            run_phase(
                "latent_down",
                lambda: runtime.execute_dense(
                    weights["latent_down"],
                    self._batch_latent_rows,
                    self._batch_hidden_input,
                    rows,
                ),
            )

            shared_chunks = self._supported_batch_chunks(rows)

            def shared_expert() -> None:
                offset = 0
                for chunk in shared_chunks:
                    runtime.execute_resident(
                        weights["shared_mlp"],
                        _pointer_offset(self._batch_shared_output, offset * hidden),
                        _pointer_offset(self._batch_hidden_input, offset * hidden),
                        chunk,
                    )
                    offset += chunk

            run_phase("shared_expert", shared_expert)

            dispatch_copies = 0
            collection_copies = 0
            fused_indexed_dispatch = runtime.indexed_copy_rows_function is not None
            if len(self._expert_ownership) != self.config.experts:
                raise KimiCudaError(
                    "verification-major execution requires all experts to be locally resident"
                )

            if expert_strategy == "expert_major":

                if fused_indexed_dispatch:
                    gather_indices = np.ascontiguousarray(
                        [assignment.row for assignment in grouped_plan.assignments],
                        dtype=np.int32,
                    )

                    def expert_dispatch() -> None:
                        nonlocal dispatch_copies
                        runtime.upload_bytes(
                            self._batch_gather_indices, gather_indices
                        )
                        runtime.execute_indexed_copy_rows(
                            self._batch_expert_input,
                            self._batch_latent_rows,
                            self._batch_gather_indices,
                            rows=grouped_plan.total_assignments,
                            dimension=latent,
                        )
                        dispatch_copies = grouped_plan.total_assignments

                else:

                    def expert_dispatch() -> None:
                        nonlocal dispatch_copies
                        for assignment in grouped_plan.assignments:
                            runtime.execute_copy(
                                _pointer_offset(
                                    self._batch_expert_input,
                                    assignment.work_index * latent,
                                ),
                                _pointer_offset(
                                    self._batch_latent_rows,
                                    assignment.row * latent,
                                ),
                                latent,
                            )
                            dispatch_copies += 1

                run_phase("expert_dispatch", expert_dispatch)

                def routed_expert_compute() -> None:
                    for group in grouped_plan.groups:
                        offset = group.start
                        for chunk in group.native_chunks:
                            runtime.execute_resident(
                                self._experts[group.expert_id],
                                _pointer_offset(
                                    self._batch_expert_output, offset * latent
                                ),
                                _pointer_offset(
                                    self._batch_expert_input, offset * latent
                                ),
                                chunk,
                            )
                            offset += chunk

                run_phase("routed_expert_compute", routed_expert_compute)

                if fused_indexed_dispatch:
                    scatter_indices = np.ascontiguousarray(
                        grouped_plan.row_slot_to_work, dtype=np.int32
                    ).reshape(-1)

                    def expert_collection() -> None:
                        nonlocal collection_copies
                        runtime.upload_bytes(
                            self._batch_scatter_indices, scatter_indices
                        )
                        runtime.execute_indexed_copy_rows(
                            self._batch_expert_input,
                            self._batch_expert_output,
                            self._batch_scatter_indices,
                            rows=grouped_plan.total_assignments,
                            dimension=latent,
                        )
                        collection_copies = grouped_plan.total_assignments

                else:

                    def expert_collection() -> None:
                        nonlocal collection_copies
                        for assignment in grouped_plan.assignments:
                            runtime.execute_copy(
                                _pointer_offset(
                                    self._batch_expert_input,
                                    (
                                        assignment.row * topk
                                        + assignment.slot
                                    )
                                    * latent,
                                ),
                                _pointer_offset(
                                    self._batch_expert_output,
                                    assignment.work_index * latent,
                                ),
                                latent,
                            )
                            collection_copies += 1

                run_phase("expert_collection", expert_collection)
                native_expert_calls = grouped_plan.native_call_count
            else:

                def token_major_experts() -> None:
                    for row, ids in enumerate(selected_ids):
                        for slot, expert_value in enumerate(ids.tolist()):
                            runtime.execute_resident(
                                self._experts[int(expert_value)],
                                _pointer_offset(
                                    self._batch_expert_input,
                                    (row * topk + slot) * latent,
                                ),
                                _pointer_offset(self._batch_latent_rows, row * latent),
                                1,
                            )

                run_phase("routed_expert_compute", token_major_experts)
                native_expert_calls = rows * topk

            def reduction() -> None:
                for row in range(rows):
                    runtime.execute_copy(
                        session.expert_rows,
                        _pointer_offset(
                            self._batch_expert_input, row * topk * latent
                        ),
                        topk * latent,
                    )
                    runtime.upload_activation(
                        session.route_weights,
                        np.ascontiguousarray(selected_weights[row], dtype=np.float32),
                    )
                    runtime.execute_moe_reduction(
                        session.reduced,
                        session.expert_rows,
                        session.route_weights,
                        count=topk,
                        dimension=latent,
                    )
                    runtime.execute_rmsnorm(
                        session.reduced,
                        session.reduced,
                        weights["routed_norm"],
                        batch=1,
                        dimension=latent,
                        epsilon=self.config.epsilon,
                    )
                    runtime.execute_copy(
                        _pointer_offset(self._batch_latent_rows, row * latent),
                        session.reduced,
                        latent,
                    )

            run_phase("scatter_reduction", reduction)
            run_phase(
                "latent_up",
                lambda: runtime.execute_dense(
                    weights["latent_up"],
                    self._batch_hidden_input,
                    self._batch_latent_rows,
                    rows,
                ),
            )

            def residual_and_boundary() -> None:
                for row in range(rows):
                    routed = _pointer_offset(self._batch_hidden_input, row * hidden)
                    prefix = _pointer_offset(self._batch_hidden_output, row * hidden)
                    shared = _pointer_offset(self._batch_shared_output, row * hidden)
                    runtime.execute_add(routed, shared, hidden)
                    runtime.execute_add(prefix, routed, hidden)
                    runtime.execute_copy(
                        _pointer_offset(
                            self._batch_boundary, row * boundary_row_elements
                        ),
                        prefix,
                        hidden,
                    )

            run_phase("residual", residual_and_boundary)

            def download_boundary() -> None:
                nonlocal result_boundary
                result_boundary = runtime.download_activation(
                    self._batch_boundary, (rows, 9, hidden)
                )

            run_phase("boundary_d2h", download_boundary)
            runtime.synchronize()
            device_ms = None
            if whole_profile_open:
                device_ms = runtime.profile_end()
                whole_profile_open = False
        except BaseException:
            if whole_profile_open:
                with contextlib.suppress(BaseException):
                    runtime.profile_end()
            # Stateful attention cannot be rolled back cheaply.  Invalidating
            # the session is the only exact failure-recovery policy.
            with contextlib.suppress(BaseException):
                self.close_session(block.session_id)
            raise

        if result_boundary is None:
            raise KimiCudaError("verification-major execution emitted no boundary")
        session.cache_length += rows
        self._execute_count += rows
        self._batch_execute_count += 1
        wall_ns = time.perf_counter_ns() - wall_started
        memory_after = runtime.mem_info()
        reuse = expert_reuse_statistics(route_matrix)
        total_assignments = rows * topk
        boundary_bytes = rows * boundary_row_elements * np.dtype(np.float32).itemsize
        router_metadata_bytes = rows * (2 * topk + 1) * np.dtype(np.float32).itemsize
        record: dict[str, Any] = {
            "verification_block": {
                "session_id": block.session_id,
                "candidate_count": block.candidate_count,
                "include_bonus_token": block.include_bonus_token,
                "row_count": rows,
                "positions": list(positions),
            },
            "expert_strategy": expert_strategy,
            "dcp_degree": dcp_degree,
            "selected_expert_ids": route_matrix.astype(int).tolist(),
            "selected_weights": [
                [float(value) for value in row] for row in selected_weights
            ],
            "boundary_output": result_boundary,
            "wall_ms": wall_ns / 1e6,
            "device_ms": device_ms,
            "phase_device_ms": phase_device_ms,
            "phase_wall_ms": phase_wall_ms,
            "profile_phases": profile_phases,
            "routing": {
                **reuse,
                "native_routed_expert_calls": native_expert_calls,
                "effective_weight_reuse_rows_per_native_call": (
                    total_assignments / native_expert_calls
                ),
                "avoided_routed_expert_weight_launches": (
                    total_assignments - native_expert_calls
                ),
            },
            "dispatch": {
                "implementation": (
                    "fused_indexed_device_copy"
                    if fused_indexed_dispatch and expert_strategy == "expert_major"
                    else "individual_device_copies"
                ),
                "count": dispatch_copies + collection_copies,
                "gather_count": dispatch_copies,
                "scatter_count": collection_copies,
                "gather_payload_bytes": dispatch_copies * latent * 4,
                "scatter_payload_bytes": collection_copies * latent * 4,
                "index_h2d_bytes": (
                    2 * total_assignments * np.dtype(np.int32).itemsize
                    if fused_indexed_dispatch and expert_strategy == "expert_major"
                    else 0
                ),
                "group_dimensions": [
                    {
                        "expert_id": group.expert_id,
                        "assignments": group.count,
                        "native_chunks": list(group.native_chunks),
                    }
                    for group in grouped_plan.groups
                ],
            },
            "transfers": {
                "boundary_h2d_bytes": boundary_bytes,
                "boundary_d2h_bytes": boundary_bytes,
                "router_metadata_d2h_bytes": router_metadata_bytes,
                "route_weight_h2d_bytes": rows * topk * 4,
                "expert_gather_d2d_bytes": dispatch_copies * latent * 4,
                "expert_scatter_d2d_bytes": collection_copies * latent * 4,
                "host_device_transfer_count": 2 + 2 * len(
                    self._supported_batch_chunks(rows)
                ) + rows,
            },
            "synchronization_count": (
                len(phase_wall_ms) if profile_phases else 2
            ),
            "persistent_buffer_allocations_during_execute": (
                self._persistent_buffer_allocation_count - allocation_before
            ),
            "weight_loads_during_execute": self._weight_load_count - weight_load_before,
            "materializations_during_execute": (
                self._model_materialization_count - materialization_before
            ),
            "free_device_bytes_before": memory_before["free_bytes"],
            "free_device_bytes_after": memory_after["free_bytes"],
            "device_memory_growth_bytes": max(
                0,
                int(memory_before["free_bytes"]) - int(memory_after["free_bytes"]),
            ),
        }
        self.execution_records.append(record)
        return record

    def execute_prefill(
        self,
        *,
        session_id: str,
        token_ids: torch.Tensor,
        cache_position_start: int,
    ) -> StageExecutionResult:
        if not self._owns_embeddings:
            raise RuntimeError("this Kimi stage does not own embeddings")
        return self._execute_one(
            session_id=session_id,
            token_ids=token_ids,
            cache_position_start=cache_position_start,
        )

    def execute_decode(
        self,
        *,
        session_id: str,
        hidden_states: torch.Tensor,
        cache_position_start: int,
    ) -> StageExecutionResult:
        return self._execute_one(
            session_id=session_id,
            boundary=hidden_states,
            cache_position_start=cache_position_start,
        )

    def execute_decode_with_external_experts(
        self,
        *,
        session_id: str,
        hidden_states: torch.Tensor,
        cache_position_start: int,
        dispatch: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, dict[str, Any]]],
    ) -> StageExecutionResult:
        """Execute the exact layer while selected routed experts are worker-owned."""
        if self._owns_embeddings or self._owns_final_endpoint:
            raise RuntimeError("external expert execution requires a non-endpoint MoE stage")
        if self._weights["mlp_type"] != "moe":
            raise RuntimeError("external expert execution requires a routed-MoE layer")
        return self._execute_one(
            session_id=session_id,
            boundary=hidden_states,
            cache_position_start=cache_position_start,
            external_expert_dispatch=dispatch,
        )

    def kv_cache_bytes(self, session_id: str) -> int:
        session = self._require_session(session_id)
        if self.layer in self.config.kda_layers:
            return (
                self.config.kda_heads
                * self.config.kda_head_dimension
                * self.config.kda_head_dimension
                + 3 * self.config.kda_projection * self.config.convolution_width
            ) * np.dtype(np.float32).itemsize
        return (
            session.maximum_context
            * (self.config.kv_lora + self.config.query_rope)
            * np.dtype(np.float32).itemsize
        )

    def clone_session_state(
        self,
        source_session_id: str,
        destination_session_id: str,
        *,
        maximum_context_override: int | None = None,
    ) -> dict[str, Any]:
        """Clone exact attention state device-to-device for a prepared branch."""

        source = self._require_session(source_session_id)
        maximum_context = (
            source.maximum_context
            if maximum_context_override is None
            else int(maximum_context_override)
        )
        if maximum_context < source.cache_length:
            raise ValueError("cloned Kimi session cannot truncate active attention state")
        copied_bytes = 0
        self.open_session(
            destination_session_id,
            maximum_context_override=maximum_context,
        )
        try:
            destination = self._require_session(destination_session_id)
            if self.layer in self.config.kda_layers:
                elements = {
                    "state": (
                        self.config.kda_heads
                        * self.config.kda_head_dimension
                        * self.config.kda_head_dimension
                    ),
                    "window_q": (
                        self.config.kda_projection * self.config.convolution_width
                    ),
                    "window_k": (
                        self.config.kda_projection * self.config.convolution_width
                    ),
                    "window_v": (
                        self.config.kda_projection * self.config.convolution_width
                    ),
                }
            else:
                elements = {
                    "latent_cache": source.cache_length * self.config.kv_lora,
                    "rope_cache": source.cache_length * self.config.query_rope,
                }
            for name, count in elements.items():
                if count:
                    self.runtime.execute_copy(
                        destination.attention_state[name],
                        source.attention_state[name],
                        count,
                    )
                    copied_bytes += count * np.dtype(np.float32).itemsize
            self.runtime.synchronize()
            destination.cache_length = source.cache_length
        except BaseException:
            with contextlib.suppress(BaseException):
                self.close_session(destination_session_id)
            raise
        return {
            "source_session_id": source_session_id,
            "destination_session_id": destination_session_id,
            "cache_sequence_length": source.cache_length,
            "copied_device_to_device_bytes": copied_bytes,
            "attention_type": (
                "KDA" if self.layer in self.config.kda_layers else "Gated_MLA"
            ),
        }

    def close_session(self, session_id: str) -> int:
        session = self._require_session(session_id)
        released = self.kv_cache_bytes(session_id)
        del self._sessions[session_id]
        session.resources.close()
        self._session_close_count += 1
        return released

    def cancel_session(self, session_id: str) -> int:
        return self.close_session(session_id)

    def lifecycle_snapshot(self) -> dict[str, Any]:
        return {
            "weight_load_count": self._weight_load_count,
            "model_materialization_count": self._model_materialization_count,
            "persistent_buffer_allocation_count": self._persistent_buffer_allocation_count,
            "execute_count": self._execute_count,
            "batch_execute_count": self._batch_execute_count,
            "session_open_count": self._session_open_count,
            "session_close_count": self._session_close_count,
            "prepare_warmup_count": self._prepare_warmup_count,
            "prepare_warmup": self.prepare_warmup,
            "cpu_transport_threads": self.cpu_transport_thread_contract,
            "active_sessions": len(self._sessions),
            "resident_device_bytes": self.resident_device_bytes,
            "tracked_device_bytes": self.tracked_device_bytes,
            "batch_capacity": self._batch_capacity,
            "verification_major_enabled": self._verification_major_enabled,
            "verification_max_rows": VERIFICATION_MAX_ROWS,
            "batch_supported_sizes": [
                size
                for size in self.runtime.expert_supported_batches
                if size <= self._batch_capacity
            ],
            "native_batch_supported_sizes": list(self.runtime.expert_supported_batches),
            "batch_workspace_bytes": self._batch_workspace_bytes,
            "measured_free_memory_delta_bytes": max(
                0,
                int(self.memory_before["free_bytes"]) - int(self.memory_after["free_bytes"]),
            ),
            "resident_expert_count": len(self._experts),
            "owned_expert_ids": sorted(self._expert_ownership),
            "complete_expert_ownership": len(self._expert_ownership) == self.config.experts,
            "cuda_library_sha256": self.cuda_library_sha256,
            "shared_memory_limits": self.runtime.shared_memory_limits,
            "weight_fingerprint": self.weight_fingerprint,
            "backend_identity": "nvidia_cuda_persistent_kimi_stage",
            "cpu_mathematical_fallbacks": 0,
        }

    def session_state_evidence(self, session_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        if self.layer in self.config.kda_layers:
            shapes = {
                "state": (
                    self.config.kda_heads,
                    self.config.kda_head_dimension,
                    self.config.kda_head_dimension,
                ),
                "window_q": (
                    self.config.kda_projection,
                    self.config.convolution_width,
                ),
                "window_k": (
                    self.config.kda_projection,
                    self.config.convolution_width,
                ),
                "window_v": (
                    self.config.kda_projection,
                    self.config.convolution_width,
                ),
            }
        else:
            shapes = {
                "latent_cache": (session.maximum_context, self.config.kv_lora),
                "rope_cache": (session.maximum_context, self.config.query_rope),
            }
        arrays = {
            name: self.runtime.download_activation(session.attention_state[name], shape)
            for name, shape in shapes.items()
        }
        digest = hashlib.sha256()
        active_prefix_digest = hashlib.sha256()
        for name, values in arrays.items():
            _digest_array(digest, name, values)
            active_values = (
                values if self.layer in self.config.kda_layers else values[: session.cache_length]
            )
            _digest_array(active_prefix_digest, name, active_values)
        finite = all(bool(np.isfinite(values).all()) for values in arrays.values())
        nonzero = any(bool(np.any(values)) for values in arrays.values())
        zero_suffix = True
        if self.layer not in self.config.kda_layers:
            zero_suffix = all(
                not np.any(values[session.cache_length :]) for values in arrays.values()
            )
        return {
            "cache_sequence_length": session.cache_length,
            "bytes": sum(values.nbytes for values in arrays.values()),
            "fingerprint": "sha256:" + digest.hexdigest(),
            "active_prefix_fingerprint": "sha256:" + active_prefix_digest.hexdigest(),
            "finite": finite,
            "nonzero_prefix": nonzero,
            "zero_suffix": zero_suffix,
            "attention_type": ("KDA" if self.layer in self.config.kda_layers else "Gated_MLA"),
        }

    def close(self) -> None:
        if self._closed:
            return
        for session_id in list(self._sessions):
            self.close_session(session_id)
        self._batch_resources.close()
        self._endpoint_resources.close()
        self._layer_resources.close()
        self.runner.close()
        self._closed = True


# Backward-compatible experiment name retained for immutable H014-026a/026b
# callers. Registered product loading instantiates the production entry class.
PersistentKimiFinalStageExecutor = PersistentKimiStageExecutor
KimiK3StageExecutor = PersistentKimiStageExecutor


def _boundary_fixtures(
    checkpoint: Path,
    oracle_trace: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    from swarm_inference.execution.kimi_k3_graph_runtime import _CheckpointReader

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
    boundaries: list[np.ndarray] = []
    expected_layer: list[np.ndarray] = []
    expected_final: list[np.ndarray] = []
    snapshot_layers = tuple(range(0, config.layers, config.residual_block))
    if snapshot_layers != (0, 12, 24, 36, 48, 60, 72, 84):
        raise KimiCudaError(f"unexpected Kimi AttnRes snapshot layers {snapshot_layers}")
    for position, token_id in enumerate(token_ids):
        base = position * (config.layers + 1)
        boundary = np.empty((1, 9, config.hidden), dtype=np.float32)
        boundary[0, 0] = trace[base + 91]
        token_bits = np.asarray(embedding[token_id], dtype=np.uint16)
        boundary[0, 1] = (token_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
        for slot, layer in enumerate(snapshot_layers[1:], start=2):
            boundary[0, slot] = trace[base + layer - 1]
        boundaries.append(boundary)
        expected_layer.append(np.ascontiguousarray(trace[base + 92]))
        expected_final.append(np.ascontiguousarray(trace[base + 93]))
    return boundaries, expected_layer, expected_final


def _process_thread_metadata(process_threads: list[Any]) -> list[dict[str, Any]]:
    """Return read-only thread attribution used by the H014 lifecycle audit."""

    rows = [
        {
            "thread_id": int(item.id),
            "user_time_s": float(item.user_time),
            "system_time_s": float(item.system_time),
            "description": "",
            "start_address": None,
            "start_module": None,
        }
        for item in process_threads
    ]
    if os.name != "nt":
        return rows
    kernel32 = ctypes.windll.kernel32
    ntdll = ctypes.windll.ntdll
    psapi = ctypes.windll.psapi
    kernel32.OpenThread.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenThread.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    ntdll.NtQueryInformationThread.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    ntdll.NtQueryInformationThread.restype = ctypes.c_long
    psapi.GetMappedFileNameW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_ulong,
    ]
    psapi.GetMappedFileNameW.restype = ctypes.c_ulong
    get_description = getattr(kernel32, "GetThreadDescription", None)
    if get_description is not None:
        get_description.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_wchar_p),
        ]
        get_description.restype = ctypes.c_long
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    process_handle = kernel32.GetCurrentProcess()
    for row in rows:
        handle = kernel32.OpenThread(
            0x0040 | 0x0800,
            False,
            int(row["thread_id"]),
        )
        if not handle:
            continue
        try:
            start_address = ctypes.c_void_p()
            status = ntdll.NtQueryInformationThread(
                handle,
                9,
                ctypes.byref(start_address),
                ctypes.sizeof(start_address),
                None,
            )
            if status == 0 and start_address.value:
                row["start_address"] = f"0x{start_address.value:016x}"
                buffer = ctypes.create_unicode_buffer(1024)
                if psapi.GetMappedFileNameW(
                    process_handle,
                    start_address,
                    buffer,
                    len(buffer),
                ):
                    row["start_module"] = buffer.value
            if get_description is not None:
                description = ctypes.c_wchar_p()
                if get_description(handle, ctypes.byref(description)) == 0:
                    row["description"] = description.value or ""
                    if description:
                        kernel32.LocalFree(ctypes.cast(description, ctypes.c_void_p))
        finally:
            kernel32.CloseHandle(handle)
    return rows


def _process_snapshot(
    runtime: PersistentStageRuntime,
    executor: PersistentKimiFinalStageExecutor,
) -> dict[str, Any]:
    process = psutil.Process()
    process_threads = process.threads()
    connection = runtime.connection_pool.snapshot()
    return {
        "process_id": os.getpid(),
        "child_process_ids": sorted(child.pid for child in process.children(recursive=True)),
        "os_thread_ids": sorted(item.id for item in process_threads),
        "os_thread_metadata": _process_thread_metadata(process_threads),
        "python_thread_ids": sorted(
            int(item.ident) for item in threading.enumerate() if item.ident is not None
        ),
        "python_native_thread_ids": sorted(
            int(item.native_id)
            for item in threading.enumerate()
            if item.native_id is not None
        ),
        "python_threads": [
            {
                "name": item.name,
                "ident": int(item.ident) if item.ident is not None else None,
                "native_id": int(item.native_id) if item.native_id is not None else None,
            }
            for item in threading.enumerate()
        ],
        "asyncio_task_ids": sorted(id(item) for item in asyncio.all_tasks()),
        "route_generation": (
            runtime.installed_route.route_generation if runtime.installed_route else None
        ),
        "topology_id": (runtime.installed_route.topology_id if runtime.installed_route else None),
        "runtime_load_count": runtime.load_count,
        "connection_metrics": {
            key: int(value) for key, value in connection.items() if isinstance(value, int)
        },
        "connection_count": int(connection["active_connections"]),
        "compute_executor": runtime.compute_executor_snapshot(),
        "executor": executor.lifecycle_snapshot(),
    }


def _warm_lifecycle_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    before_executor = before["executor"]
    after_executor = after["executor"]
    before_connections = before["connection_metrics"]
    after_connections = after["connection_metrics"]
    return {
        "topology_rebuilds": int(
            (before["topology_id"], before["route_generation"])
            != (after["topology_id"], after["route_generation"])
        ),
        "process_creation": len(set(after["child_process_ids"]) - set(before["child_process_ids"])),
        "thread_creation": len(set(after["os_thread_ids"]) - set(before["os_thread_ids"])),
        "task_creation": len(set(after["asyncio_task_ids"]) - set(before["asyncio_task_ids"])),
        "connection_establishment": int(after_connections["connections_created"])
        - int(before_connections["connections_created"]),
        "weight_loading": int(after_executor["weight_load_count"])
        - int(before_executor["weight_load_count"]),
        "model_materialization": int(after_executor["model_materialization_count"])
        - int(before_executor["model_materialization_count"]),
        "persistent_buffer_allocation": int(after_executor["persistent_buffer_allocation_count"])
        - int(before_executor["persistent_buffer_allocation_count"]),
    }


def _added_thread_metadata(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[dict[str, Any]]:
    added = set(after["os_thread_ids"]) - set(before["os_thread_ids"])
    return [
        row for row in after.get("os_thread_metadata", []) if row["thread_id"] in added
    ]


def _stage_assignment() -> StageAssignment:
    return StageAssignment(
        stage_id=92,
        layer_start=92,
        layer_end=93,
        layer_ids=(92,),
        weight_bytes=18_915_537_408,
        estimated_compute_ns=1,
        measured_compute_ns=None,
        kv_cache_bytes_per_token=(1536 + 64) * 4,
        peak_temporary_bytes=512 * 1024**2,
        activation_bytes=9 * 7168 * 4,
        device="native-cuda:0",
        owns_embeddings=False,
        owns_final_norm=True,
        owns_output_projection=True,
    )


def _previous_assignment() -> StageAssignment:
    return StageAssignment(
        stage_id=91,
        layer_start=91,
        layer_end=92,
        layer_ids=(91,),
        weight_bytes=16_566_684_160,
        estimated_compute_ns=1,
        measured_compute_ns=None,
        kv_cache_bytes_per_token=(1536 + 64) * 4,
        peak_temporary_bytes=512 * 1024**2,
        activation_bytes=9 * 7168 * 4,
        device="native-cuda:0",
        owns_embeddings=False,
        owns_final_norm=False,
        owns_output_projection=False,
    )


def _message(
    *,
    boundary: np.ndarray,
    position: int,
    sequence: int,
    session_id: str,
    request_id: str,
    model_revision: str,
    tokenizer_revision: str,
    topology_id: str,
) -> StageMessage:
    packed = pack_tensor(torch.from_numpy(boundary.copy()), requested_mode="none")
    return StageMessage(
        operation=Operation.PREFILL if position == 0 else Operation.DECODE,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        topology_id=topology_id,
        stage_id=92,
        layer_start=92,
        layer_end=93,
        session_id=session_id,
        request_id=request_id,
        sequence_number=sequence,
        token_position=position,
        source_stage=91,
        destination_stage=92,
        tensor_shape=packed.shape,
        tensor_dtype=packed.dtype,
        compression_mode=packed.compression_mode,
        payload=packed.payload,
        attributes={
            "model_id": "moonshotai/Kimi-K3",
            "route_generation": 1,
            "request_generation": 1,
            "replay_only": False,
            "source_worker_id": "k3-worker-091",
            "destination_worker_id": "k3-worker-092",
            "cache_position_start": position,
            "deadline_ns": time.time_ns() + 600_000_000_000,
            "tensor": packed.attributes(),
        },
    )


async def _benchmark_persistent_final_stage(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    oracle_logits: Path,
    *,
    registered: bool = False,
    identity_manifest: Path | None = None,
    cycle_id: str | None = None,
    execution_phase_observer: Callable[[str, StageMessage], Awaitable[None]]
    | None = None,
    readiness_phase_observer: Callable[[str], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    cuda_library = cuda_library.expanduser().resolve()
    oracle_trace = oracle_trace.expanduser().resolve()
    oracle_routes = oracle_routes.expanduser().resolve()
    oracle_logits = oracle_logits.expanduser().resolve()
    boundaries, expected_layers, expected_finals = _boundary_fixtures(checkpoint, oracle_trace)
    logits = np.memmap(
        oracle_logits,
        mode="r",
        dtype="<f4",
        shape=(2, 163840),
    )
    routes = _parse_oracle_routes(oracle_routes)[92]
    assignment = _stage_assignment()
    cycle_id = cycle_id or ("H014-026b" if registered else "H014-026a")
    request_prefix = cycle_id.lower()
    topology_id = f"{request_prefix}-final-stage"
    model_revision = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
    tokenizer_revision = (
        "sha256:49f733745c76dbd69bd90fa109a66929af087e881bcffadbd4068b68635fd526"
        if registered
        else "sha256:cf5dfd2c5a41890b946a6d28c317b8e4"
    )
    if registered and identity_manifest is None:
        raise ValueError("registered Kimi loading requires a worker-pinned identity manifest")
    executor_holder: dict[str, PersistentKimiFinalStageExecutor] = {}

    def loader(request: LoadStageRequest, resolved: Path | None) -> Any:
        if resolved is None:
            raise FileNotFoundError("Kimi persistent fixture requires its checkpoint")
        executor = PersistentKimiFinalStageExecutor(
            request=request,
            checkpoint=resolved,
            cuda_library=cuda_library,
        )
        executor_holder["executor"] = executor
        return executor

    runtime = PersistentStageRuntime(
        worker_id="k3-worker-092",
        device="native-cuda:0",
        dtype="float32",
        memory_limit_bytes=31 * 1024**3,
        maximum_sessions=2,
        configured_model_path=checkpoint,
        configured_model_identity_path=identity_manifest if registered else None,
        loader=None if registered else loader,
        execution_phase_observer=execution_phase_observer,
    )
    load_request = LoadStageRequest(
        worker_id="k3-worker-092",
        request_id=f"{request_prefix}-load",
        model_id="moonshotai/Kimi-K3",
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        topology_id=topology_id,
        route_generation=1,
        stage_count=93,
        assignment=assignment,
        adapter_id="kimi_k3_cuda" if registered else "kimi-k3-cuda",
        fast_path_id="colibri-kimi-k3-cuda",
        fast_path_mode="resident",
        fast_path_context_bucket=3,
        model_content_fingerprint=(
            "25162130a11904bac1220a7d654a3f7dfd616ea5f4035d488e40ac74ddea8f94"
            if registered
            else None
        ),
        native_runtime_library=str(cuda_library) if registered else None,
        native_runtime_library_sha256=(_sha256_file(cuda_library) if registered else None),
        device="native-cuda:0",
        dtype="float32",
        model_path=str(checkpoint),
    )
    try:
        load_response = await runtime.load_stage(load_request)
        loaded_executor = runtime.loaded_executor
        if not isinstance(loaded_executor, PersistentKimiFinalStageExecutor):
            raise TypeError("canonical loader did not return the Kimi final-stage executor")
        executor = loaded_executor
        if readiness_phase_observer is not None:
            await readiness_phase_observer("load_prepare_complete")
        lease_expiry = time.time_ns() + 3_600_000_000_000
        route_response = await runtime.install_route(
            InstallStageRouteRequest(
                worker_id="k3-worker-092",
                request_id=f"{request_prefix}-route",
                model_id=load_request.model_id,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                topology_id=load_request.topology_id,
                route_generation=1,
                assignment=assignment,
                device="native-cuda:0",
                dtype="float32",
                previous_stage=StageRouteEndpoint(
                    worker_id="k3-worker-091",
                    stage_id=91,
                    data_endpoint="127.0.0.1:19091",
                    assignment=_previous_assignment(),
                ),
                next_stage=None,
                stage_count=93,
                lease_expiry_unix_ns=lease_expiry,
            )
        )
        if readiness_phase_observer is not None:
            await readiness_phase_observer("route_installed")

        async def open_session(session_id: str, request_id: str) -> None:
            await runtime.open_session(
                OpenStageSessionRequest(
                    worker_id="k3-worker-092",
                    request_id=request_id,
                    model_id=load_request.model_id,
                    model_revision=model_revision,
                    tokenizer_revision=tokenizer_revision,
                    topology_id=load_request.topology_id,
                    route_generation=1,
                    stage_id=92,
                    device="native-cuda:0",
                    dtype="float32",
                    session_id=session_id,
                    request_generation=1,
                    lease_expiry_unix_ns=lease_expiry,
                )
            )

        async def close_session(session_id: str, request_id: str) -> None:
            await runtime.close_session(
                CloseStageSessionRequest(
                    worker_id="k3-worker-092",
                    request_id=request_id,
                    model_id=load_request.model_id,
                    model_revision=model_revision,
                    tokenizer_revision=tokenizer_revision,
                    topology_id=load_request.topology_id,
                    route_generation=1,
                    stage_id=92,
                    device="native-cuda:0",
                    dtype="float32",
                    session_id=session_id,
                    request_generation=1,
                    lease_expiry_unix_ns=lease_expiry,
                )
            )

        await open_session("warmup", f"{request_prefix}-open-warmup")
        if readiness_phase_observer is not None:
            await readiness_phase_observer("warmup_session_opened")
        warmup_message = _message(
            boundary=boundaries[0],
            position=0,
            sequence=0,
            session_id="warmup",
            request_id=f"{request_prefix}-warmup",
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            topology_id=topology_id,
        )
        if readiness_phase_observer is not None:
            await readiness_phase_observer("warmup_message_built")
        await runtime.handle_message(warmup_message)
        await close_session("warmup", f"{request_prefix}-close-warmup")
        executor.execution_records.clear()

        await open_session("retained", f"{request_prefix}-open-retained")
        before = _process_snapshot(runtime, executor)
        responses: list[StageMessage] = []
        wire: list[dict[str, int]] = []
        per_call_snapshots: list[dict[str, Any]] = []
        for position, boundary in enumerate(boundaries):
            incoming = _message(
                boundary=boundary,
                position=position,
                sequence=position,
                session_id="retained",
                request_id=f"{request_prefix}-position-{position}",
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                topology_id=topology_id,
            )
            call_before = _process_snapshot(runtime, executor)
            response = await runtime.handle_message(incoming)
            call_after = _process_snapshot(runtime, executor)
            responses.append(response)
            incoming_frame = encode_message(incoming)
            response_frame = encode_message(response)
            wire.append(
                {
                    "input_payload_bytes": incoming_frame.payload_bytes,
                    "input_wire_bytes": incoming_frame.wire_bytes,
                    "output_payload_bytes": response_frame.payload_bytes,
                    "output_wire_bytes": response_frame.wire_bytes,
                }
            )
            per_call_snapshots.append(
                {
                    "position": position,
                    "lifecycle_delta": _warm_lifecycle_delta(call_before, call_after),
                    "added_thread_metadata": _added_thread_metadata(
                        call_before,
                        call_after,
                    ),
                }
            )
        immediate_after = _process_snapshot(runtime, executor)
        thread_observation: list[dict[str, Any]] = []
        observation_before = immediate_after
        for sample in range(1, 21):
            time.sleep(0.05)
            observation_after = _process_snapshot(runtime, executor)
            if (
                observation_before["os_thread_ids"]
                != observation_after["os_thread_ids"]
            ):
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
        records = executor.execution_records
        expected_tokens = (220, 11, 374)
        correctness: list[dict[str, Any]] = []
        for position, record in enumerate(records):
            token_tensor, _ = unpack_tensor(
                responses[position].payload,
                dict(responses[position].attributes["tensor"]),
            )
            row = {
                "position": position,
                "host_thread_native_id": int(record["host_thread_native_id"]),
                "host_thread_ident": int(record["host_thread_ident"]),
                "host_thread_name": str(record["host_thread_name"]),
                "input_fingerprint": _array_fingerprint(boundaries[position]),
                "layer_output_fingerprint": _array_fingerprint(record["layer_output"]),
                "expected_layer_output_fingerprint": _array_fingerprint(expected_layers[position]),
                "layer_output_metrics": _numerical_metrics(
                    expected_layers[position], record["layer_output"]
                ),
                "final_hidden_fingerprint": _array_fingerprint(record["final_hidden"]),
                "expected_final_hidden_fingerprint": _array_fingerprint(expected_finals[position]),
                "final_hidden_metrics": _numerical_metrics(
                    expected_finals[position], record["final_hidden"]
                ),
                "expected_expert_ids": routes[position],
                "selected_expert_ids": record["selected_expert_ids"],
                "routing_equality": record["selected_expert_ids"] == routes[position],
                "sampled_token_id": int(token_tensor.item()),
                "expected_sampled_token_id": expected_tokens[position],
                "sampled_token_equality": int(token_tensor.item()) == expected_tokens[position],
            }
            if position < 2:
                row["logits_fingerprint"] = _array_fingerprint(record["logits"])
                row["expected_logits_fingerprint"] = _array_fingerprint(logits[position])
                row["logits_metrics"] = _numerical_metrics(
                    np.ascontiguousarray(logits[position]), record["logits"]
                )
            correctness.append(row)
        state = executor.session_state_evidence("retained")
        maximum_relative_error = max(
            max(
                float(row["layer_output_metrics"]["relative_l2_error"]),
                float(row["final_hidden_metrics"]["relative_l2_error"]),
            )
            for row in correctness
        )
        maximum_logits_relative_error = max(
            float(row["logits_metrics"]["relative_l2_error"]) for row in correctness[:2]
        )
        compute_executor = before["compute_executor"]
        compute_thread_native_id = compute_executor["thread_native_id"]
        compute_thread_ident = compute_executor["thread_ident"]
        prepare_warmup = executor.prepare_warmup or {}
        prepare_stage_fixture = prepare_warmup.get("stage_fixture", {})
        prepare_transport = prepare_stage_fixture.get(
            "canonical_transport_round_trip", {}
        )
        prepare_quiescence = prepare_stage_fixture.get("thread_quiescence", {})
        prepare_cpu_threads = prepare_stage_fixture.get("cpu_transport_threads", {})
        prepare_pass = (
            prepare_warmup.get("count") == 1
            and prepare_warmup.get("primitive")
            == "isolated assigned-stage Kimi CUDA fixture"
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
            and prepare_quiescence.get("pass") is True
            and prepare_quiescence.get("minimum_observation_ms") == 3_500.0
            and prepare_quiescence.get("required_stable_samples") == 20
            and prepare_quiescence.get("stable_samples_observed", 0) >= 20
            and prepare_cpu_threads.get("intraop_threads") == 1
            and prepare_cpu_threads.get("interop_threads") == 1
        )
        compute_thread_consistent = (
            compute_thread_native_id is not None
            and compute_thread_ident is not None
            and compute_executor["max_workers"] == 1
            and compute_executor["thread_start_count"] == 1
            and prepare_stage_fixture.get("single_host_thread") is True
            and set(prepare_stage_fixture.get("host_thread_native_ids", []))
            == {compute_thread_native_id}
            and set(prepare_stage_fixture.get("host_thread_idents", []))
            == {compute_thread_ident}
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
                for item in per_call_snapshots
            )
        )
        correctness_pass = (
            maximum_relative_error <= 3e-5
            and maximum_logits_relative_error <= 3e-5
            and all(row["routing_equality"] for row in correctness)
            and all(row["sampled_token_equality"] for row in correctness)
            and state["finite"]
            and state["nonzero_prefix"]
            and state["zero_suffix"]
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "cycle_id": cycle_id,
            "status": (
                "PASS" if correctness_pass and lifecycle_pass and prepare_pass else "FAIL"
            ),
            "hypothesis": (
                "The registered canonical persistent runtime loads the exact worker-pinned "
                "real Kimi CUDA stage once and executes three state-contiguous positions "
                "with no warm reconstruction."
                if registered
                else "The canonical persistent runtime loads the real final Kimi CUDA stage "
                "once and executes three state-contiguous positions with no warm reconstruction."
            ),
            "backend": {
                "identity": "nvidia_cuda_persistent_kimi_final_stage",
                "cpu_mathematical_fallbacks": 0,
                "cuda_library": str(cuda_library),
                "cuda_library_sha256": _sha256_file(cuda_library),
                "device_control_identity": "native-cuda:0",
                "physical_device": "NVIDIA GeForce RTX 5090 sm_120",
                "target_binary": "sm_86+compute_86 PTX",
                "registered_adapter": registered,
            },
            "checkpoint": {
                "path": str(checkpoint),
                "model_revision": model_revision,
                "layer": 92,
                "source_weight_bytes": assignment.weight_bytes,
                "owned_tensor_count": executor.ownership.parameter_count,
                "ownership_hash": executor.ownership.ownership_hash,
            },
            "lifecycle": {
                "install": {"topology_id": load_request.topology_id},
                "materialize": {
                    "count": 1,
                    "checkpoint": str(checkpoint),
                },
                "load": {
                    "accepted": load_response.accepted,
                    "count": runtime.load_count,
                    "elapsed_ms": executor.load_ns / 1e6,
                    "resident_device_bytes": executor.resident_device_bytes,
                    "tracked_device_bytes": executor.tracked_device_bytes,
                    "measured_free_memory_delta_bytes": max(
                        0,
                        int(executor.memory_before["free_bytes"])
                        - int(executor.memory_after["free_bytes"]),
                    ),
                    "resident_experts": len(executor._experts),
                    "memory_before": executor.memory_before,
                    "memory_after": executor.memory_after,
                },
                "prepare": {
                    "pass": prepare_pass,
                    "maximum_context": 3,
                    "persistent_buffer_allocations_before_warm": before["executor"][
                        "persistent_buffer_allocation_count"
                    ],
                    "warmup": prepare_warmup,
                },
                "ready": {
                    "route_accepted": route_response.accepted,
                    "route_generation": 1,
                    "stage_identity": "92:[92,93)",
                },
                "warm_delta": lifecycle_delta,
                "per_generation_deltas": per_call_snapshots,
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
            "correctness": {
                "relative_error_gate": 3e-5,
                "maximum_layer_or_final_relative_l2_error": maximum_relative_error,
                "maximum_logits_relative_l2_error": maximum_logits_relative_error,
                "routing_equality": all(row["routing_equality"] for row in correctness),
                "sampled_token_equality": all(row["sampled_token_equality"] for row in correctness),
                "positions": correctness,
                "state": state,
                "pass": correctness_pass,
            },
            "benchmark": {
                "retained_generations": len(records),
                "warmup_generations": 1,
                "wall": _timing([float(item["wall_ms"]) for item in records]),
                "device": _timing([float(item["device_ms"]) for item in records]),
                "wire": wire,
                "instrumentation_mode": "production-minimal plus lifecycle counters",
            },
            "inspection": {
                "actual_bottleneck": (
                    "pending retained measurement"
                    if not records
                    else "resident final-stage compute"
                ),
                "streamed_weight_loading_in_warm_path": False,
                "all_experts_preloaded": len(executor._experts) == executor.config.experts,
                "canonical_default_loader": registered,
                "worker_pinned_identity": (
                    str(identity_manifest.expanduser().resolve())
                    if identity_manifest is not None
                    else None
                ),
            },
        }
        await close_session("retained", f"{request_prefix}-close-retained")
        return result
    finally:
        await runtime.close()


def benchmark_persistent_final_stage(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    oracle_logits: Path,
    output_path: Path,
    *,
    registered: bool = False,
    identity_manifest: Path | None = None,
    cycle_id: str | None = None,
) -> dict[str, Any]:
    result = asyncio.run(
        _benchmark_persistent_final_stage(
            checkpoint,
            cuda_library,
            oracle_trace,
            oracle_routes,
            oracle_logits,
            registered=registered,
            identity_manifest=identity_manifest,
            cycle_id=cycle_id,
        )
    )
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return {
        "status": result["status"],
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "summary": {
            "resident_device_bytes": result["lifecycle"]["load"]["resident_device_bytes"],
            "maximum_relative_l2_error": result["correctness"][
                "maximum_layer_or_final_relative_l2_error"
            ],
            "routing_equality": result["correctness"]["routing_equality"],
            "warm_delta": result["lifecycle"]["warm_delta"],
        },
    }


__all__ = [
    "KimiK3StageExecutor",
    "PersistentKimiFinalStageExecutor",
    "PersistentKimiStageExecutor",
    "benchmark_persistent_final_stage",
]
