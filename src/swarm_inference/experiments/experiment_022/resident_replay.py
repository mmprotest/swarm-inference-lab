"""Resident exact sharded K3 layer replay for the E022 timing repair."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _CudaRuntime, _numerical_metrics
from swarm_inference.experiments.experiment_018.analysis import parse_oracle_routes
from swarm_inference.experiments.experiment_019.attention import (
    DeviceResources,
    execute_kda_attention,
    execute_mla_attention,
)
from swarm_inference.experiments.experiment_019.checkpoint import DirectShardLoader
from swarm_inference.experiments.experiment_019.physical import (
    LATENT,
    RELATIVE_L2_GATE,
    ROUTED_EXPERTS,
    TOPK,
    _ResidentHandles,
    striped_latent_down,
    timing,
)
from swarm_inference.experiments.experiment_019.quantization import GpuShardQuantizer
from swarm_inference.experiments.experiment_019.sharded_graph import (
    EPSILON,
    HIDDEN,
    LAYERS,
    ShardedK3Graph,
)
from swarm_inference.experiments.experiment_020.expert_grouped import (
    GroupedTop16Runtime,
    _execute_grouped,
)
from swarm_inference.experiments.experiment_022.resident_primitives import (
    prepare_complete_expert_stripe_banks,
)


class _ResidentReplayRuntime:
    """Record allocation/upload handles once, then replay the same native DAG."""

    def __init__(self, base: _CudaRuntime) -> None:
        self.base = base
        self.mode = "record"
        self._sequences: dict[str, list[Any]] = {}
        self._cursors: dict[str, int] = {}
        self._persistent_state: list[tuple[Any, np.ndarray]] = []
        self.weight_uploads = 0
        self.allocations = 0
        self.replay_stats: dict[str, float | int] = {}

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.base, name)
        if not callable(value) or not name.startswith("execute_"):
            return value

        def measured(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter_ns()
            result = value(*args, **kwargs)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            self.replay_stats["launch_submit_ms"] = float(
                self.replay_stats.get("launch_submit_ms", 0.0)
            ) + elapsed
            self.replay_stats["launch_count"] = int(
                self.replay_stats.get("launch_count", 0)
            ) + 1
            return result

        return measured

    def begin_replay(self) -> None:
        self.mode = "replay"
        self._cursors = {name: 0 for name in self._sequences}
        self.replay_stats = {
            "handle_lookup_ms": 0.0,
            "launch_submit_ms": 0.0,
            "launch_count": 0,
            "cuda_sync_wall_ms": 0.0,
            "cuda_sync_count": 0,
            "h2d_wall_ms": 0.0,
            "h2d_bytes": 0,
            "d2h_wall_ms": 0.0,
            "d2h_bytes": 0,
        }

    def register_persistent_state(
        self, pointer: Any, initial: np.ndarray
    ) -> None:
        """Register fixture state during preparation and initialize it once."""

        if self.mode != "record":
            return
        values = np.ascontiguousarray(initial, dtype=np.float32).copy()
        self.base.upload_activation(pointer, values)
        self._persistent_state.append((pointer, values))

    def reset_persistent_states(self) -> float:
        """Restore an identical validation fixture outside the measured wall."""

        started = time.perf_counter_ns()
        for pointer, values in self._persistent_state:
            self.base.upload_activation(pointer, values)
        return (time.perf_counter_ns() - started) / 1e6

    def _handle(self, name: str, creator: Callable[[], Any]) -> Any:
        if self.mode == "record":
            value = creator()
            self._sequences.setdefault(name, []).append(value)
            if name == "allocate":
                self.allocations += 1
            else:
                self.weight_uploads += 1
            return value
        started = time.perf_counter_ns()
        index = self._cursors.get(name, 0)
        values = self._sequences.get(name, [])
        if index >= len(values):
            raise RuntimeError(f"resident replay introduced an unprepared {name}")
        self._cursors[name] = index + 1
        self.replay_stats["handle_lookup_ms"] = float(
            self.replay_stats.get("handle_lookup_ms", 0.0)
        ) + (time.perf_counter_ns() - started) / 1e6
        return values[index]

    def allocate(self, bytes_: int) -> Any:
        return self._handle("allocate", lambda: self.base.allocate(bytes_))

    def free(self, _pointer: Any) -> None:
        # Prepared buffers remain resident until close().
        return None

    def upload(self, tensor: Any) -> Any:
        return self._handle("upload", lambda: self.base.upload(tensor))

    def upload_grouped_int4(self, tensor: Any) -> Any:
        return self._handle(
            "upload_grouped_int4", lambda: self.base.upload_grouped_int4(tensor)
        )

    def upload_float32(self, matrix: np.ndarray) -> Any:
        return self._handle(
            "upload_float32", lambda: self.base.upload_float32(matrix)
        )

    def upload_bf16_embedding(self, table: np.ndarray) -> Any:
        return self._handle(
            "upload_bf16_embedding", lambda: self.base.upload_bf16_embedding(table)
        )

    def upload_int8(self, tensor: Any) -> Any:
        return self._handle("upload_int8", lambda: self.base.upload_int8(tensor))

    def release_tensor(self, _handle: Any) -> None:
        # Prepared immutable tensors remain resident until close().
        return None

    def upload_activation(self, destination: Any, activation: np.ndarray) -> None:
        started = time.perf_counter_ns()
        self.base.upload_activation(destination, activation)
        self.replay_stats["h2d_wall_ms"] = float(
            self.replay_stats.get("h2d_wall_ms", 0.0)
        ) + (time.perf_counter_ns() - started) / 1e6
        self.replay_stats["h2d_bytes"] = int(
            self.replay_stats.get("h2d_bytes", 0)
        ) + int(np.ascontiguousarray(activation).nbytes)

    def download_activation(self, source: Any, shape: tuple[int, ...]) -> np.ndarray:
        started = time.perf_counter_ns()
        result = self.base.download_activation(source, shape)
        self.replay_stats["d2h_wall_ms"] = float(
            self.replay_stats.get("d2h_wall_ms", 0.0)
        ) + (time.perf_counter_ns() - started) / 1e6
        self.replay_stats["d2h_bytes"] = int(
            self.replay_stats.get("d2h_bytes", 0)
        ) + int(result.nbytes)
        return result

    def synchronize(self) -> None:
        started = time.perf_counter_ns()
        self.base.synchronize()
        self.replay_stats["cuda_sync_wall_ms"] = float(
            self.replay_stats.get("cuda_sync_wall_ms", 0.0)
        ) + (time.perf_counter_ns() - started) / 1e6
        self.replay_stats["cuda_sync_count"] = int(
            self.replay_stats.get("cuda_sync_count", 0)
        ) + 1

    def close(self, *, shutdown: bool = True) -> None:
        # Weight handles are owned by _CudaRuntime and released here.  Pipe
        # allocations are not tracked by it, so release unique prepared ones.
        for pointer in reversed(self._sequences.get("allocate", [])):
            self.base.free(pointer)
        self._sequences["allocate"] = []
        self.base.close(shutdown=shutdown)


class _ResidentReplayLoader:
    def __init__(self, base: DirectShardLoader) -> None:
        self.base = base
        self.catalog = base.catalog
        self._cache: dict[tuple[Any, ...], Any] = {}
        self.cache_hits = 0
        self.replay_lookup_ms = 0.0

    def begin_replay(self) -> None:
        self.replay_lookup_ms = 0.0

    @property
    def audit(self) -> list[dict[str, Any]]:
        return self.base.audit

    def load(self, name: str, **kwargs: Any) -> np.ndarray:
        started = time.perf_counter_ns()
        key = ("load", name, *sorted(kwargs.items()))
        if key in self._cache:
            self.cache_hits += 1
            value = self._cache[key]
        else:
            value = self.base.load(name, **kwargs)
            self._cache[key] = value
        self.replay_lookup_ms += (time.perf_counter_ns() - started) / 1e6
        return value

    def reviewed_small(self, name: str, **kwargs: Any) -> np.ndarray:
        return self.load(name, allow_reviewed_full_tensor=True, **kwargs)

    def expert_stripe(self, **kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        key = ("expert_stripe", *sorted(kwargs.items()))
        if key in self._cache:
            self.cache_hits += 1
            value = self._cache[key]
        else:
            value = self.base.expert_stripe(**kwargs)
            self._cache[key] = value
        self.replay_lookup_ms += (time.perf_counter_ns() - started) / 1e6
        return value


class _ResidentReplayQuantizer:
    def __init__(self, base: GpuShardQuantizer) -> None:
        self.base = base
        self._cache: dict[tuple[Any, ...], Any] = {}
        self.cache_hits = 0
        self.replay_lookup_ms = 0.0

    def begin_replay(self) -> None:
        self.replay_lookup_ms = 0.0

    @property
    def audit(self) -> list[dict[str, Any]]:
        return self.base.audit

    def _cached(
        self,
        key: tuple[Any, ...],
        operation: Callable[[], Any],
    ) -> Any:
        started = time.perf_counter_ns()
        if key in self._cache:
            self.cache_hits += 1
            value = self._cache[key]
        else:
            value = operation()
            self._cache[key] = value
        self.replay_lookup_ms += (time.perf_counter_ns() - started) / 1e6
        return value

    def grouped_int4(self, source: np.ndarray, *, owner: str) -> Any:
        return self._cached(
            ("grouped_int4", id(source), owner),
            lambda: self.base.grouped_int4(source, owner=owner),
        )

    def row_int8(self, source: np.ndarray, *, owner: str) -> Any:
        return self._cached(
            ("row_int8", id(source), owner),
            lambda: self.base.row_int8(source, owner=owner),
        )

    def row_maximum(self, source: np.ndarray, *, owner: str) -> np.ndarray:
        return self._cached(
            ("row_maximum", id(source), owner),
            lambda: self.base.row_maximum(source, owner=owner),
        )

    def row_int8_with_scales(
        self,
        source: np.ndarray,
        scales: np.ndarray,
        *,
        owner: str,
    ) -> Any:
        digest = hashlib.sha256(np.ascontiguousarray(scales).tobytes()).hexdigest()
        return self._cached(
            ("row_int8_with_scales", id(source), owner, digest),
            lambda: self.base.row_int8_with_scales(source, scales, owner=owner),
        )

    def rmsnorm(self, *args: Any, **kwargs: Any) -> np.ndarray:
        # RMSNorm is real hot-path compute, not startup conversion.
        return self.base.rmsnorm(*args, **kwargs)


class _ResidentGroupedGraph(ShardedK3Graph):
    def __init__(
        self,
        checkpoint: Path,
        cuda_library: Path,
        shard_library: Path,
        grouped_library: Path,
        *,
        degree: int,
        capture_state_arrays: bool = False,
        shutdown_runtime_on_close: bool = True,
    ) -> None:
        super().__init__(checkpoint, cuda_library, shard_library, degree=degree)
        self.capture_state_arrays = capture_state_arrays
        self.shutdown_runtime_on_close = shutdown_runtime_on_close
        self.grouped = GroupedTop16Runtime(grouped_library)
        self.runtime = _ResidentReplayRuntime(self.runtime)  # type: ignore[assignment]
        self.loader = _ResidentReplayLoader(self.loader)  # type: ignore[assignment]
        self.quantizer = _ResidentReplayQuantizer(self.quantizer)  # type: ignore[assignment]
        self.reduction_records: list[dict[str, Any]] = []
        self._full_expert_banks: dict[tuple[int, int], _ResidentHandles] = {}
        self._residual_scores: dict[tuple[int, str], np.ndarray] = {}
        self._resident_small_tensors: dict[tuple[str, str, str], np.ndarray] = {}
        self.last_phase_timings: dict[str, float] = {}

    def _small(self, name: str, *, worker: str, purpose: str) -> np.ndarray:
        """Prepare reviewed immutable tensors once, including BF16 conversion.

        The E019 reference helper converts BF16 tensors to FP32 on every call.
        That is useful for a one-shot reference graph, but it made the 12.8 MiB
        K3 router matrix conversion part of every resident replay row.  A real
        persistent worker prepares that representation at startup, so retain
        the converted host tensor alongside the resident device handle.
        """

        key = (name, worker, purpose)
        value = self._resident_small_tensors.get(key)
        if value is None:
            value = np.ascontiguousarray(
                super()._small(name, worker=worker, purpose=purpose),
                dtype=np.float32,
            )
            self._resident_small_tensors[key] = value
        return value

    def _residual_score(self, layer: int, stem: str) -> np.ndarray:
        """Prepare immutable AttnRes score vectors once, outside replay wall."""

        key = (layer, stem)
        if key not in self._residual_scores:
            prefix = f"language_model.model.layers.{layer}"
            self._residual_scores[key] = np.ascontiguousarray(
                self._small(
                    f"{prefix}.{stem}_res_norm.weight",
                    worker=f"layer-{layer:02d}.attnres-worker",
                    purpose=f"resident_{stem}_residual_norm",
                )
                * self._small(
                    f"{prefix}.{stem}_res_proj.weight",
                    worker=f"layer-{layer:02d}.attnres-worker",
                    purpose=f"resident_{stem}_residual_projection",
                ),
                dtype=np.float32,
            )
        return self._residual_scores[key]

    def prepare_full_expert_banks(self, layer: int) -> None:
        """Prepare arbitrary-route expert pointer maps outside timed replay."""

        if self._full_expert_banks:
            raise RuntimeError("resident replay expert banks were already prepared")
        banks = prepare_complete_expert_stripe_banks(
            self.runtime.base,
            self.loader.base,
            layer=layer,
            degree=self.degree,
            worker_prefix=f"layer-{layer:02d}.expert-bank",
        )
        self._full_expert_banks.update(
            {(layer, stripe): bank for stripe, bank in enumerate(banks)}
        )

    def close(self) -> None:
        for bank in self._full_expert_banks.values():
            bank.close()
        self._full_expert_banks.clear()
        self.grouped.close()
        self.runtime.close(shutdown=self.shutdown_runtime_on_close)

    def _reduce_partials(
        self,
        partials: list[np.ndarray] | tuple[np.ndarray, ...],
        *,
        layer: int | str,
        operator: str,
    ) -> np.ndarray:
        if not partials:
            raise ValueError("resident reduction requires worker partials")
        shape = tuple(int(value) for value in partials[0].shape)
        resources = DeviceResources(self.runtime)
        try:
            inputs = [resources.upload(value) for value in partials]
            elements = int(np.prod(shape, dtype=np.int64))
            output = resources.allocate(elements)
            started = time.perf_counter_ns()
            self.runtime.profile_begin()
            self.runtime.execute_copy(output, inputs[0], elements)
            for source in inputs[1:]:
                self.runtime.execute_add(output, source, elements)
            self.runtime.synchronize()
            cuda_ms = self.runtime.profile_end()
            wall_ms = (time.perf_counter_ns() - started) / 1e6
            values = self.runtime.download_activation(output, shape)
        finally:
            resources.close()
        record = {
            "worker_id": f"layer-{layer}.{operator}.reducer",
            "operator": f"{operator}_native_reduction",
            "resource_type": "microworker",
            "implementation": "production_cuda_copy_add_reduction",
            "participants": len(partials),
            "duration_ms": wall_ms,
            "cuda_ms": cuda_ms,
            "host_overhead_ms": max(0.0, wall_ms - cuda_ms),
            "physical_launches": len(partials),
            "payload_bytes_per_participant": int(partials[0].nbytes),
        }
        self.reduction_records.append(record)
        self.worker_operations.append(record)
        return values

    def _native_boundary_merge(
        self,
        partials: list[np.ndarray] | tuple[np.ndarray, ...],
        *,
        layer: int,
        operator: str,
    ) -> np.ndarray:
        """Execute a canonical CUDA copy/add boundary and retain its exact name."""

        if not partials:
            raise ValueError("native boundary merge requires at least one input")
        shape = tuple(int(value) for value in partials[0].shape)
        if any(tuple(value.shape) != shape for value in partials):
            raise ValueError("native boundary merge geometry differs")
        resources = DeviceResources(self.runtime)
        try:
            inputs = [resources.upload(value) for value in partials]
            elements = int(np.prod(shape, dtype=np.int64))
            output = resources.allocate(elements)
            started = time.perf_counter_ns()
            self.runtime.profile_begin()
            self.runtime.execute_copy(output, inputs[0], elements)
            for source in inputs[1:]:
                self.runtime.execute_add(output, source, elements)
            self.runtime.synchronize()
            cuda_ms = self.runtime.profile_end()
            wall_ms = (time.perf_counter_ns() - started) / 1e6
            values = self.runtime.download_activation(output, shape)
        finally:
            resources.close()
        self.worker_operations.append(
            {
                "worker_id": f"layer-{layer:02d}.boundary-worker",
                "operator": operator,
                "resource_type": "microworker",
                "implementation": "production_cuda_copy_add_boundary",
                "participants": len(partials),
                "duration_ms": wall_ms,
                "cuda_ms": cuda_ms,
                "host_overhead_ms": max(0.0, wall_ms - cuda_ms),
                "physical_launches": len(partials),
                "payload_bytes_per_participant": int(partials[0].nbytes),
            }
        )
        return values

    def _attnres_rows(
        self,
        prefix: np.ndarray,
        residuals: list[np.ndarray],
        query: np.ndarray,
        *,
        worker: str,
        operator: str,
    ) -> np.ndarray:
        rows = prefix.shape[0]
        return np.ascontiguousarray(
            np.stack(
                [
                    self._attnres(
                        prefix[row],
                        [value[row] for value in residuals],
                        query,
                        worker=worker,
                        operator=operator,
                    )
                    for row in range(rows)
                ]
            ),
            dtype=np.float32,
        )

    def _rmsnorm_rows(
        self,
        values: np.ndarray,
        weight: np.ndarray,
        *,
        worker: str,
        operator: str,
    ) -> np.ndarray:
        """Execute and event-time the canonical CUDA RMSNorm hot path."""

        source = np.ascontiguousarray(values, dtype=np.float32)
        scale = np.ascontiguousarray(weight, dtype=np.float32).reshape(-1)
        if source.ndim != 2 or scale.shape != (source.shape[1],):
            raise ValueError("resident RMSNorm geometry differs")
        resources = DeviceResources(self.runtime)
        started = time.perf_counter_ns()
        try:
            source_device = resources.upload(source)
            scale_device = resources.allocate(scale.size)
            if self.runtime.mode == "record":
                self.runtime.upload_activation(scale_device, scale)
            output_device = resources.allocate(source.size)
            self.runtime.profile_begin()
            self.runtime.execute_rmsnorm(
                output_device,
                source_device,
                scale_device,
                batch=source.shape[0],
                dimension=source.shape[1],
                epsilon=EPSILON,
            )
            self.runtime.synchronize()
            cuda_ms = self.runtime.profile_end()
            output = self.runtime.download_activation(output_device, source.shape)
        finally:
            resources.close()
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        self.worker_operations.append(
            {
                "worker_id": worker,
                "operator": operator,
                "resource_type": "microworker",
                "implementation": "coli_cuda_execute_rmsnorm",
                "duration_ms": wall_ms,
                "cuda_ms": cuda_ms,
                "host_overhead_ms": max(0.0, wall_ms - cuda_ms),
                "physical_launches": 1,
            }
        )
        return output

    def _route_rows(
        self, values: np.ndarray, *, layer: int
    ) -> tuple[np.ndarray, np.ndarray]:
        source = np.ascontiguousarray(values, dtype=np.float32)
        if source.ndim != 2 or source.shape[1] != HIDDEN:
            raise ValueError("resident router geometry differs")
        prefix = f"language_model.model.layers.{layer}.block_sparse_moe.gate"
        worker = f"layer-{layer:02d}.router-worker"
        router = self._small(
            f"{prefix}.weight",
            worker=worker,
            purpose="reviewed_small_router_replica",
        )
        bias = self._small(
            f"{prefix}.e_score_correction_bias",
            worker=worker,
            purpose="reviewed_small_router_bias_replica",
        )
        resources = DeviceResources(self.runtime)
        started = time.perf_counter_ns()
        try:
            input_device = resources.upload(source)
            router_device = resources.allocate(router.size)
            bias_device = resources.allocate(bias.size)
            if self.runtime.mode == "record":
                self.runtime.upload_activation(router_device, router)
                self.runtime.upload_activation(bias_device, bias)
            self.runtime.profile_begin()
            ids, weights, effective = self.runtime.route_batch(
                input_device,
                router_device,
                bias_device,
                batch=source.shape[0],
                hidden=HIDDEN,
                experts=896,
                topk=TOPK,
            )
            if not np.all(effective == TOPK):
                raise RuntimeError(
                    f"layer {layer} batched router retained {effective.tolist()} experts"
                )
            self.runtime.synchronize()
            cuda_ms = self.runtime.profile_end()
        finally:
            resources.close()
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        self.worker_operations.append(
            {
                "worker_id": worker,
                "operator": "router",
                "resource_type": "microworker",
                "implementation": "coli_cuda_pipe_router_batch",
                "duration_ms": wall_ms,
                "cuda_ms": cuda_ms,
                "host_overhead_ms": max(0.0, wall_ms - cuda_ms),
                "physical_launches": 1,
                "batch_rows": source.shape[0],
                "route_metadata_bytes": source.shape[0] * TOPK * 8,
                "resident_router_weight_bytes": int(router.nbytes + bias.nbytes),
                "weight_preparation_in_timed_region": False,
            }
        )
        return (
            np.ascontiguousarray(ids, dtype=np.int32),
            np.ascontiguousarray(weights, dtype=np.float32),
        )

    def execute_layer_rows(
        self,
        layer: int,
        hidden: np.ndarray,
        residuals: list[np.ndarray],
    ) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
        """Execute a contiguous chunk while retaining row/state semantics."""

        phase_timings: dict[str, float] = {}
        phase_started = time.perf_counter_ns()
        phase_operation_start = len(self.worker_operations)

        def finish_phase(name: str) -> None:
            nonlocal phase_operation_start, phase_started
            finished = time.perf_counter_ns()
            phase_timings[name] = (finished - phase_started) / 1e6
            for operation in self.worker_operations[phase_operation_start:]:
                operation["phase"] = name
            phase_operation_start = len(self.worker_operations)
            phase_started = finished

        incoming = np.ascontiguousarray(hidden, dtype=np.float32)
        if incoming.ndim != 2 or incoming.shape[1] != HIDDEN:
            raise ValueError("resident layer chunk must be rows x hidden")
        is_snapshot = layer % 12 == 0
        prefix = f"language_model.model.layers.{layer}"
        attention_score = self._residual_score(layer, "self_attention")
        finish_phase("input_and_attention_metadata")
        attention_input = self._attnres_rows(
            incoming,
            residuals,
            attention_score,
            worker=f"layer-{layer:02d}.attnres-worker",
            operator="attention_attnres_mix",
        )
        if is_snapshot:
            residuals = [*residuals, incoming.copy()]
        input_norm = self._small(
            f"{prefix}.input_layernorm.weight",
            worker=f"layer-{layer:02d}.norm-worker",
            purpose="replicated_small_input_norm",
        )
        normalized = self._rmsnorm_rows(
            attention_input,
            input_norm,
            worker=f"layer-{layer:02d}.norm-worker",
            operator="input_rmsnorm",
        )
        finish_phase("attention_preprocess")
        attention_executor = (
            execute_kda_attention
            if f"{prefix}.self_attn.q_proj.weight" in self.catalog.records()
            else execute_mla_attention
        )
        arguments: dict[str, Any] = {
            "quantizer": self.quantizer,
            "native_reduction": True,
            # The encompassing resident DAG clock must see one invocation of
            # every logical native task.  E019's standalone benchmark keeps a
            # separate initialization pass, which is intentionally disabled
            # here rather than mislabeled as distributed overhead.
            "resident_single_execution": True,
            "capture_state_arrays": self.capture_state_arrays,
        }
        if attention_executor is execute_kda_attention:
            arguments["shard_kernel"] = self.kda_kernel
        attention_output, attention_record = attention_executor(
            self.runtime,
            self.loader,
            normalized,
            layer=layer,
            degree=self.degree,
            warmup=0,
            iterations=1,
            **arguments,
        )
        common = attention_record["common_projection"]
        self.worker_operations.append(
            {
                "worker_id": attention_record["common_owner"],
                "operator": f"{attention_record['attention_type']}_common_projection",
                "resource_type": "microworker",
                "duration_ms": common["wall"]["p50_ms"],
                "cuda_ms": common["cuda"]["p50_ms"],
                "host_overhead_ms": common["host_overhead"]["p50_ms"],
                "physical_launches": 3
                if attention_record["attention_type"] == "Gated_MLA"
                else 1,
            }
        )
        for worker in attention_record["workers"]:
            self.worker_operations.append(
                {
                    "worker_id": worker["worker_id"],
                    "operator": f"{attention_record['attention_type']}_attention_stripe",
                    "resource_type": "microworker",
                    "stripe_index": worker["stripe_index"],
                    "duration_ms": worker["wall"]["p50_ms"],
                    "cuda_ms": worker["cuda"]["p50_ms"],
                    "host_overhead_ms": worker["host_overhead"]["p50_ms"],
                    "runtime_weight_bytes": worker["runtime_weight_bytes"],
                    "persistent_state_bytes": worker["persistent_state_bytes"],
                    "physical_launches": worker["physical_launches"],
                }
            )
        validation_state_capture_ms = float(
            attention_record.get("validation_state_capture_ms", 0.0)
        )
        if validation_state_capture_ms > 0.0:
            self.worker_operations.append(
                {
                    "worker_id": "experiment.validation",
                    "operator": "state_validation_capture",
                    "resource_type": "experiment_harness",
                    "duration_ms": validation_state_capture_ms,
                    "cuda_ms": 0.0,
                    "device_copy_ms": float(
                        attention_record.get(
                            "validation_state_capture_d2h_ms", 0.0
                        )
                    ),
                    "host_overhead_ms": validation_state_capture_ms,
                    "physical_launches": 0,
                    "cost_classification": "EXPERIMENT_ONLY",
                    "production_model_eligible": False,
                }
            )
        attention_reduction = dict(attention_record["reduction"])
        attention_reduction.update(
            {
                "worker_id": f"layer-{layer:02d}.attention.reducer",
                "operator": "attention_native_reduction",
                "resource_type": "microworker",
                "duration_ms": attention_reduction["wall_ms"],
            }
        )
        self.reduction_records.append(attention_reduction)
        self.worker_operations.append(attention_reduction)
        layer_prefix = self._native_boundary_merge(
            [attention_output] if is_snapshot else [incoming, attention_output],
            layer=layer,
            operator="attention_residual_merge",
        )
        finish_phase("attention_workers_and_collective")
        mlp_score = self._residual_score(layer, "mlp")
        mixed = self._attnres_rows(
            layer_prefix,
            residuals,
            mlp_score,
            worker=f"layer-{layer:02d}.attnres-worker",
            operator="mlp_attnres_mix",
        )
        post_norm = self._small(
            f"{prefix}.post_attention_layernorm.weight",
            worker=f"layer-{layer:02d}.norm-worker",
            purpose="replicated_small_post_attention_norm",
        )
        mlp_input = self._rmsnorm_rows(
            mixed,
            post_norm,
            worker=f"layer-{layer:02d}.norm-worker",
            operator="post_attention_rmsnorm",
        )
        finish_phase("post_attention_preprocess")
        if layer == 0:
            mlp_output, mlp_workers = self._intermediate_mlp_stripes(
                f"{prefix}.mlp",
                mlp_input,
                layer=layer,
                intermediate=33792,
                operator="dense_mlp_stripe",
            )
            routes_record = None
            finish_phase("dense_mlp_workers_and_collective")
        else:
            routes, route_weights = self._route_rows(mlp_input, layer=layer)
            finish_phase("router")
            latent, latent_workers = striped_latent_down(
                self.runtime,
                self.loader,
                mlp_input,
                layer=layer,
                degree=self.degree,
                warmup=0,
                iterations=1,
                quantizer=self.quantizer,
            )
            for worker in latent_workers:
                self.worker_operations.append(
                    {
                        "worker_id": worker["worker_id"],
                        "operator": "latent_down_projection_stripe",
                        "resource_type": "microworker",
                        "stripe_index": worker["stripe_index"],
                        "duration_ms": worker["wall"]["p50_ms"],
                        "cuda_ms": worker["cuda"]["p50_ms"],
                        "host_overhead_ms": max(
                            0.0,
                            float(worker["wall"]["p50_ms"])
                            - float(worker["cuda"]["p50_ms"]),
                        ),
                        "runtime_weight_bytes": worker["runtime_weight_bytes"],
                        "physical_launches": 1,
                    }
                )
            finish_phase("latent_down_workers")
            expert_output, expert_workers = self._routed_experts(
                latent, routes, route_weights, layer=layer
            )
            finish_phase("expert_workers_and_collective")
            routed_norm = self._small(
                f"{prefix}.block_sparse_moe.routed_expert_norm.weight",
                worker=f"layer-{layer:02d}.routed-norm-worker",
                purpose="replicated_small_routed_expert_norm",
            )
            routed = self._rmsnorm_rows(
                expert_output,
                routed_norm,
                worker=f"layer-{layer:02d}.routed-norm-worker",
                operator="routed_expert_rmsnorm",
            )
            finish_phase("routed_normalization")
            routed_up, routed_up_workers = self._column_projection(
                f"{prefix}.block_sparse_moe.routed_expert_up_proj.weight",
                routed,
                layer=layer,
                operator="latent_up_projection_stripe",
                input_dimension=LATENT,
            )
            finish_phase("latent_up_workers_and_collective")
            shared, shared_workers = self._intermediate_mlp_stripes(
                f"{prefix}.block_sparse_moe.shared_experts",
                mlp_input,
                layer=layer,
                intermediate=6144,
                operator="shared_expert_stripe",
            )
            finish_phase("shared_workers_and_collective")
            # This is a real coordinator-local reduction in the distributed
            # task graph.  Execute it with the same production CUDA copy/add
            # primitive as every other reduction so its service is measured
            # directly instead of disappearing into Python array arithmetic.
            mlp_output = self._reduce_partials(
                [routed_up, shared], layer=layer, operator="routed_shared"
            )
            mlp_workers = [
                *latent_workers,
                *expert_workers,
                *routed_up_workers,
                *shared_workers,
            ]
            routes_record = {
                "selected_expert_ids": routes.tolist(),
                "selected_weights": route_weights.tolist(),
            }
        finish_phase("routed_shared_collective")
        output = self._native_boundary_merge(
            [layer_prefix, mlp_output],
            layer=layer,
            operator="layer_output_state_commit",
        )
        finish_phase("output_state_commit")
        record = {
            "layer": layer,
            "rows": incoming.shape[0],
            "attention_type": attention_record["attention_type"],
            "attention": attention_record,
            "routes": routes_record,
            "mlp_worker_count": len(mlp_workers),
        }
        finish_phase("receipt_assembly")
        self.last_phase_timings = phase_timings
        return output, residuals, record

    def _routed_experts(
        self,
        latent: np.ndarray,
        routes: np.ndarray,
        route_weights: np.ndarray,
        *,
        layer: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        partials: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        experts = sorted({int(value) for value in routes.reshape(-1)})
        for stripe in range(self.degree):
            worker = f"layer-{layer:02d}.expert-bank.worker-{stripe:02d}"
            try:
                resident = self._full_expert_banks[(layer, stripe)]
            except KeyError as exc:
                raise RuntimeError("complete resident expert bank was not prepared") from exc
            measurement, partial = _execute_grouped(
                self.runtime,
                self.grouped,
                resident,
                latent,
                routes,
                route_weights,
                warmup=0,
                iterations=1,
            )
            partials.append(partial)
            record = {
                "worker_id": worker,
                "operator": "grouped_expert_stripe_bank_top16",
                "resource_type": "microworker",
                "stripe_index": stripe,
                "active_experts": len(experts),
                "resident_experts": len(resident.handles),
                "arbitrary_route_complete_bank": len(resident.handles)
                == ROUTED_EXPERTS,
                "runtime_weight_bytes": resident.runtime_bytes,
                "duration_ms": measurement["wall"]["p50_ms"],
                "cuda_ms": measurement["cuda"]["p50_ms"],
                "host_overhead_ms": measurement["wall"]["p50_ms"]
                - measurement["cuda"]["p50_ms"],
                "logical_expert_operations": routes.shape[0] * TOPK,
                "physical_launches": self.grouped.physical_launches,
                "network_visible_partial_outputs": 1,
            }
            records.append(record)
            self.worker_operations.append(record)
        return self._reduce_partials(
            partials, layer=layer, operator="expert_stripe_bank"
        ), records


def _trace_rows(
    trace: np.memmap[Any, Any], boundary: int, rows: int
) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack([trace[(row % 3) * (LAYERS + 1) + boundary] for row in range(rows)]),
        dtype=np.float32,
    )


def _residual_rows(
    trace: np.memmap[Any, Any],
    layer: int,
    embeddings: np.ndarray,
) -> list[np.ndarray]:
    rows = embeddings.shape[0]
    return [
        np.ascontiguousarray(embeddings, dtype=np.float32)
        if snapshot == 0
        else _trace_rows(trace, snapshot - 1, rows)
        for snapshot in range(0, layer, 12)
    ]


def replay_resident_layer(
    checkpoint: Path,
    cuda_library: Path,
    shard_library: Path,
    grouped_library: Path,
    oracle_root: Path,
    *,
    layer: int,
    degree: int,
    rows: int = 1,
    warmup: int = 1,
    iterations: int = 5,
    exact_reference_output: np.ndarray | None = None,
    exact_reference_routes: list[list[int]] | None = None,
    exact_reference_state_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Prepare once, then physically replay the same exact ordered shard DAG."""

    if layer <= 0 or layer >= LAYERS:
        raise ValueError("resident sharded replay requires a routed K3 layer")
    trace = np.memmap(
        oracle_root / "hidden-trace.f32",
        mode="r",
        dtype="<f4",
        shape=(3 * (LAYERS + 1), HIDDEN),
    )
    graph = _ResidentGroupedGraph(
        checkpoint,
        cuda_library,
        shard_library,
        grouped_library,
        degree=degree,
    )
    try:
        if rows not in (1, 2, 4):
            raise ValueError("resident completion replay supports chunk rows 1, 2, or 4")
        hidden = _trace_rows(trace, layer - 1, rows)
        token_ids = (163584, 18699, 11)
        embeddings = np.ascontiguousarray(
            np.stack(
                [graph.embedding(token_ids[row % len(token_ids)])[0][0] for row in range(rows)]
            ),
            dtype=np.float32,
        )
        immutable_residuals = _residual_rows(trace, layer, embeddings)
        graph.prepare_full_expert_banks(layer)
        graph.worker_operations.clear()
        graph.reduction_records.clear()
        startup_started = time.perf_counter_ns()
        prepared_output, _state, _record = graph.execute_layer_rows(
            layer, hidden, immutable_residuals
        )
        startup_ms = (time.perf_counter_ns() - startup_started) / 1e6
        read_count = len(graph.loader.audit)
        quantization_count = sum(
            row.get("quantization") != "none_rmsnorm_compute"
            for row in graph.quantizer.audit
        )
        upload_count = graph.runtime.weight_uploads
        allocation_count = graph.runtime.allocations
        memory = graph.runtime.mem_info()
        wall: list[float] = []
        outputs: list[np.ndarray] = []
        operation_records: list[list[dict[str, Any]]] = []
        instrumentation: list[dict[str, Any]] = []
        layer_records: list[dict[str, Any]] = []
        for index in range(warmup + iterations):
            graph.runtime.begin_replay()
            fixture_reset_ms = graph.runtime.reset_persistent_states()
            graph.loader.begin_replay()
            graph.quantizer.begin_replay()
            start_operation = len(graph.worker_operations)
            quantizer_audit_start = len(graph.quantizer.audit)
            started = time.perf_counter_ns()
            output, _state, _record = graph.execute_layer_rows(
                layer, hidden, immutable_residuals
            )
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if index >= warmup:
                wall.append(elapsed)
                outputs.append(output)
                operation_records.append(
                    [dict(value) for value in graph.worker_operations[start_operation:]]
                )
                layer_records.append(_record)
                rmsnorm_ms = sum(
                    float(row.get("wall_ms", 0.0))
                    for row in graph.quantizer.audit[quantizer_audit_start:]
                    if row.get("quantization") == "none_rmsnorm_compute"
                )
                instrumentation.append(
                    {
                        **dict(graph.runtime.replay_stats),
                        "loader_lookup_ms": graph.loader.replay_lookup_ms,
                        "quantizer_lookup_ms": graph.quantizer.replay_lookup_ms,
                        "rmsnorm_host_ms": rmsnorm_ms,
                        "fixture_state_reset_ms_outside_outer": fixture_reset_ms,
                        "fixture_state_reset_classification": "EXPERIMENT_ONLY",
                        "phase_wall_ms": dict(graph.last_phase_timings),
                    }
                )
        oracle_rows = min(rows, 3)
        if exact_reference_output is not None:
            expected = np.ascontiguousarray(exact_reference_output, dtype=np.float32)
            if expected.shape != outputs[-1].shape:
                raise ValueError("exact layer reference does not match replay output")
            correctness_output = outputs[-1]
            correctness_rows = rows
            correctness_source = "physical whole-layer execution of the same K3 layer inputs/state"
        else:
            expected = _trace_rows(trace, layer, oracle_rows)
            correctness_output = outputs[-1][:oracle_rows]
            correctness_rows = oracle_rows
            correctness_source = "immutable E014 oracle boundaries"
        correctness = _numerical_metrics(expected, correctness_output)
        expected_routes = parse_oracle_routes(oracle_root / "routes.txt")[layer]
        actual_routes = (
            layer_records[-1]["routes"]["selected_expert_ids"]
            if layer_records[-1].get("routes") is not None
            else []
        )
        oracle_routes_exact = all(
            tuple(int(value) for value in actual_routes[row])
            == tuple(int(value) for value in expected_routes[row])
            for row in range(oracle_rows)
        )
        reference_routes_exact = (
            exact_reference_routes is None
            or (len(exact_reference_routes) == rows
            and all(
                tuple(int(value) for value in actual_routes[row])
                == tuple(int(value) for value in exact_reference_routes[row])
                for row in range(rows)
            ))
        )
        state_output = layer_records[-1]["attention"]["state_output"]
        state_exact = (
            exact_reference_state_fingerprint is None
            or state_output["fingerprint"] == exact_reference_state_fingerprint
        )
        replay_reads = len(graph.loader.audit) - read_count
        replay_quantizations = (
            sum(
                row.get("quantization") != "none_rmsnorm_compute"
                for row in graph.quantizer.audit
            )
            - quantization_count
        )
        replay_uploads = graph.runtime.weight_uploads - upload_count
        replay_allocations = graph.runtime.allocations - allocation_count
        operation_names = sorted(
            {str(row.get("operator")) for records in operation_records for row in records}
        )
        stable = all(np.array_equal(prepared_output, output) for output in outputs)
        complete_expert_bank_residency = (
            len(graph._full_expert_banks) == degree
            and all(
                len(bank.handles) == ROUTED_EXPERTS
                for bank in graph._full_expert_banks.values()
            )
        )
        passed = (
            float(correctness["relative_l2_error"]) <= RELATIVE_L2_GATE
            and oracle_routes_exact
            and reference_routes_exact
            and state_exact
            and bool(np.isfinite(outputs[-1]).all())
            and stable
            and complete_expert_bank_residency
            and replay_reads == replay_quantizations == replay_uploads == replay_allocations == 0
        )
        return {
            "schema_version": "experiment-022-resident-ordered-shard-layer-v1",
            "status": "PASS" if passed else "FAIL",
            "evidence_class": "PHYSICAL RTX 5090 resident ordered shard DAG",
            "layer": layer,
            "attention_type": "KDA" if "KDA_attention_stripe" in operation_names else "Gated_MLA",
            "degree": degree,
            "rows": rows,
            "startup_ms": startup_ms,
            "startup_checkpoint_reads": read_count,
            "startup_quantizations": quantization_count,
            "startup_weight_uploads": upload_count,
            "startup_buffer_allocations": allocation_count,
            "startup_excluded_from_service": True,
            "resident_free_bytes_after_prepare": memory["free_bytes"],
            "resident_total_bytes": memory["total_bytes"],
            "all_active_partition_shards_simultaneously_resident": True,
            "arbitrary_route_full_expert_bank_resident": complete_expert_bank_residency,
            "resident_expert_bank_count": len(graph._full_expert_banks),
            "expected_resident_expert_bank_count": degree,
            "resident_experts_per_bank": sorted(
                len(bank.handles) for bank in graph._full_expert_banks.values()
            ),
            "timed_checkpoint_reads": replay_reads,
            "timed_quantizations": replay_quantizations,
            "timed_weight_uploads": replay_uploads,
            "timed_buffer_allocations": replay_allocations,
            "wall": timing(wall),
            "wall_samples_ms": wall,
            "correctness": correctness,
            "correctness_source": correctness_source,
            "correctness_rows": correctness_rows,
            "correctness_oracle_rows": oracle_rows,
            "rows_without_immutable_oracle_boundary": max(0, rows - oracle_rows),
            "routes_exact_on_oracle_rows": oracle_routes_exact,
            "routes_exact_against_physical_reference": reference_routes_exact,
            "state_output": state_output,
            "state_exact_against_physical_reference": state_exact,
            "all_output_values_finite": bool(np.isfinite(outputs[-1]).all()),
            "replay_output_stable": stable,
            "operation_names": operation_names,
            "operation_count": len(operation_records[-1]),
            "operation_records": operation_records,
            "instrumentation": instrumentation,
            "reduction_records": [dict(value) for value in graph.reduction_records],
            "layer_records": layer_records,
            "state_fingerprints": [
                worker.get("state_fingerprint")
                for worker in layer_records[-1]["attention"]["workers"]
            ],
            "prepared_output_sha256": "sha256:"
            + hashlib.sha256(np.ascontiguousarray(prepared_output).tobytes()).hexdigest(),
        }
    finally:
        graph.close()


ResidentMixedLayerGraph = _ResidentGroupedGraph


__all__ = ["ResidentMixedLayerGraph", "replay_resident_layer"]
