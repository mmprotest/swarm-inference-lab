"""Resident exact sharded K3 layer replay for the E022 timing repair."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _CudaRuntime, _numerical_metrics
from swarm_inference.experiments.experiment_019.checkpoint import DirectShardLoader
from swarm_inference.experiments.experiment_019.physical import (
    RELATIVE_L2_GATE,
    TOPK,
    _upload_stripe_experts,
    timing,
)
from swarm_inference.experiments.experiment_019.quantization import GpuShardQuantizer
from swarm_inference.experiments.experiment_019.sharded_graph import (
    HIDDEN,
    LAYERS,
    ShardedK3Graph,
)
from swarm_inference.experiments.experiment_020.expert_grouped import (
    GroupedTop16Runtime,
    _execute_grouped,
)


class _ResidentReplayRuntime:
    """Record allocation/upload handles once, then replay the same native DAG."""

    def __init__(self, base: _CudaRuntime) -> None:
        self.base = base
        self.mode = "record"
        self._sequences: dict[str, list[Any]] = {}
        self._cursors: dict[str, int] = {}
        self.weight_uploads = 0
        self.allocations = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def begin_replay(self) -> None:
        self.mode = "replay"
        self._cursors = {name: 0 for name in self._sequences}

    def _handle(self, name: str, creator: Callable[[], Any]) -> Any:
        if self.mode == "record":
            value = creator()
            self._sequences.setdefault(name, []).append(value)
            if name == "allocate":
                self.allocations += 1
            else:
                self.weight_uploads += 1
            return value
        index = self._cursors.get(name, 0)
        values = self._sequences.get(name, [])
        if index >= len(values):
            raise RuntimeError(f"resident replay introduced an unprepared {name}")
        self._cursors[name] = index + 1
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

    def close(self) -> None:
        # Weight handles are owned by _CudaRuntime and released here.  Pipe
        # allocations are not tracked by it, so release unique prepared ones.
        for pointer in reversed(self._sequences.get("allocate", [])):
            self.base.free(pointer)
        self._sequences["allocate"] = []
        self.base.close()


class _ResidentReplayLoader:
    def __init__(self, base: DirectShardLoader) -> None:
        self.base = base
        self.catalog = base.catalog
        self._cache: dict[tuple[Any, ...], Any] = {}
        self.cache_hits = 0

    @property
    def audit(self) -> list[dict[str, Any]]:
        return self.base.audit

    def load(self, name: str, **kwargs: Any) -> np.ndarray:
        key = ("load", name, *sorted(kwargs.items()))
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        value = self.base.load(name, **kwargs)
        self._cache[key] = value
        return value

    def reviewed_small(self, name: str, **kwargs: Any) -> np.ndarray:
        return self.load(name, allow_reviewed_full_tensor=True, **kwargs)

    def expert_stripe(self, **kwargs: Any) -> Any:
        key = ("expert_stripe", *sorted(kwargs.items()))
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        value = self.base.expert_stripe(**kwargs)
        self._cache[key] = value
        return value


class _ResidentReplayQuantizer:
    def __init__(self, base: GpuShardQuantizer) -> None:
        self.base = base
        self._cache: dict[tuple[Any, ...], Any] = {}
        self.cache_hits = 0

    @property
    def audit(self) -> list[dict[str, Any]]:
        return self.base.audit

    def _cached(
        self,
        key: tuple[Any, ...],
        operation: Callable[[], Any],
    ) -> Any:
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        value = operation()
        self._cache[key] = value
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
    ) -> None:
        super().__init__(checkpoint, cuda_library, shard_library, degree=degree)
        self.grouped = GroupedTop16Runtime(grouped_library)
        self.runtime = _ResidentReplayRuntime(self.runtime)  # type: ignore[assignment]
        self.loader = _ResidentReplayLoader(self.loader)  # type: ignore[assignment]
        self.quantizer = _ResidentReplayQuantizer(self.quantizer)  # type: ignore[assignment]

    def close(self) -> None:
        self.grouped.close()
        self.runtime.close()

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
            resident = _upload_stripe_experts(
                self.runtime,
                self.loader,
                layer=layer,
                experts=experts,
                degree=self.degree,
                stripe=stripe,
                worker_id=worker,
            )
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
            resident.close()
        output = np.sum(np.stack(partials), axis=0, dtype=np.float64).astype(np.float32)
        return output, records


def _residuals(trace: np.memmap[Any, Any], layer: int, embedding: np.ndarray) -> list[np.ndarray]:
    return [
        np.ascontiguousarray(embedding, dtype=np.float32)
        if snapshot == 0
        else np.ascontiguousarray(trace[snapshot - 1], dtype=np.float32)
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
    warmup: int = 1,
    iterations: int = 5,
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
        hidden = np.ascontiguousarray(trace[layer - 1], dtype=np.float32)
        embedding = graph.embedding(163584)[0][0]
        graph.worker_operations.clear()
        startup_started = time.perf_counter_ns()
        prepared_output, _state, _record = graph.execute_layer(
            layer, hidden, _residuals(trace, layer, embedding)
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
        for index in range(warmup + iterations):
            graph.runtime.begin_replay()
            start_operation = len(graph.worker_operations)
            started = time.perf_counter_ns()
            output, _state, _record = graph.execute_layer(
                layer, hidden, _residuals(trace, layer, embedding)
            )
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if index >= warmup:
                wall.append(elapsed)
                outputs.append(output)
                operation_records.append(
                    [dict(value) for value in graph.worker_operations[start_operation:]]
                )
        expected = np.ascontiguousarray(trace[layer], dtype=np.float32)
        correctness = _numerical_metrics(expected, outputs[-1])
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
        passed = (
            float(correctness["relative_l2_error"]) <= RELATIVE_L2_GATE
            and stable
            and replay_reads == replay_quantizations == replay_uploads == replay_allocations == 0
        )
        return {
            "schema_version": "experiment-022-resident-ordered-shard-layer-v1",
            "status": "PASS" if passed else "FAIL",
            "evidence_class": "PHYSICAL RTX 5090 resident ordered shard DAG",
            "layer": layer,
            "attention_type": "KDA" if "KDA_attention_stripe" in operation_names else "Gated_MLA",
            "degree": degree,
            "rows": 1,
            "startup_ms": startup_ms,
            "startup_checkpoint_reads": read_count,
            "startup_quantizations": quantization_count,
            "startup_weight_uploads": upload_count,
            "startup_buffer_allocations": allocation_count,
            "startup_excluded_from_service": True,
            "resident_free_bytes_after_prepare": memory["free_bytes"],
            "resident_total_bytes": memory["total_bytes"],
            "all_active_partition_shards_simultaneously_resident": True,
            "arbitrary_route_full_expert_bank_resident": False,
            "timed_checkpoint_reads": replay_reads,
            "timed_quantizations": replay_quantizations,
            "timed_weight_uploads": replay_uploads,
            "timed_buffer_allocations": replay_allocations,
            "wall": timing(wall),
            "correctness": correctness,
            "replay_output_stable": stable,
            "operation_names": operation_names,
            "operation_count": len(operation_records[-1]),
            "operation_records": operation_records,
            "prepared_output_sha256": "sha256:"
            + hashlib.sha256(np.ascontiguousarray(prepared_output).tobytes()).hexdigest(),
        }
    finally:
        graph.close()


__all__ = ["replay_resident_layer"]
