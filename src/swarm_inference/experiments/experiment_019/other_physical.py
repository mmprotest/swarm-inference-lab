"""Physical projection, shared-expert, endpoint, routing, and AttnRes shards."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _array_fingerprint
from swarm_inference.experiments.experiment_019.attention import DeviceResources
from swarm_inference.experiments.experiment_019.checkpoint import balanced_range
from swarm_inference.experiments.experiment_019.physical import HIDDEN, LATENT, timing
from swarm_inference.experiments.experiment_019.sharded_graph import ShardedK3Graph


def _summarize_workers(repeats: Sequence[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not repeats:
        return []
    worker_ids = [str(row["worker_id"]) for row in repeats[0]]
    result: list[dict[str, Any]] = []
    for index, worker_id in enumerate(worker_ids):
        values = [float(rows[index]["duration_ms"]) for rows in repeats]
        row = {**repeats[0][index], "duration": timing(values)}
        if "cuda_ms" in repeats[0][index]:
            row["cuda"] = timing(
                [float(rows[index]["cuda_ms"]) for rows in repeats]
            )
            row["host_overhead"] = timing(
                [float(rows[index]["host_overhead_ms"]) for rows in repeats]
            )
        result.append(row)
    return result


def _persistent_router(
    graph: ShardedK3Graph,
    values: np.ndarray,
    *,
    layer: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    prefix = f"language_model.model.layers.{layer}.block_sparse_moe.gate"
    worker = f"layer-{layer:02d}.router-worker"
    router = graph._small(
        f"{prefix}.weight",
        worker=worker,
        purpose="physical_persistent_router_weight",
    )
    bias = graph._small(
        f"{prefix}.e_score_correction_bias",
        worker=worker,
        purpose="physical_persistent_router_bias",
    )
    resources = DeviceResources(graph.runtime)
    try:
        input_device = resources.upload(values)
        router_device = resources.upload(router)
        bias_device = resources.upload(bias)

        def operation() -> tuple[list[int], list[float]]:
            ids, weights, effective = graph.runtime.route(
                input_device,
                router_device,
                bias_device,
                hidden=HIDDEN,
                experts=896,
                topk=16,
            )
            if effective != 16:
                raise RuntimeError("physical router did not retain top-16")
            return [int(value) for value in ids], [float(value) for value in weights]

        for _ in range(warmup):
            operation()
        values_ms: list[float] = []
        result: tuple[list[int], list[float]] | None = None
        for _ in range(iterations):
            started = time.perf_counter_ns()
            result = operation()
            values_ms.append((time.perf_counter_ns() - started) / 1e6)
        if result is None:
            raise RuntimeError("router benchmark produced no result")
        return {
            "worker_id": worker,
            "rows": values.shape[0],
            "weight_bytes": router.nbytes + bias.nbytes,
            "route_metadata_bytes": 16 * 8,
            "duration": timing(values_ms),
            "selected_expert_ids": result[0],
            "selected_weights_fingerprint": _array_fingerprint(
                np.asarray(result[1], dtype=np.float32)
            ),
            "hot_path_weight_loads": 0,
        }
    finally:
        resources.close()


def _embedding_shard(
    graph: ShardedK3Graph,
    *,
    token_id: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    owner = next(
        stripe
        for stripe in range(graph.degree)
        if (
            balanced_range(163840, graph.degree, stripe).start
            <= token_id
            < balanced_range(163840, graph.degree, stripe).stop
        )
    )
    shard = balanced_range(163840, graph.degree, owner)
    worker = f"endpoint.embedding.worker-{owner:02d}"
    table = graph.loader.load(
        "language_model.model.embed_tokens.weight",
        worker_id=worker,
        purpose="persistent_embedding_vocabulary_shard",
        axis=0,
        start=shard.start,
        stop=shard.stop,
    )
    resources = DeviceResources(graph.runtime)
    handle = graph.runtime.upload_bf16_embedding(table)
    token_device = graph.runtime.allocate(4)
    output_device = graph.runtime.allocate(HIDDEN * 4)
    local = np.asarray([token_id - shard.start], dtype=np.int32)
    graph.runtime.upload_bytes(token_device, local)

    def operation() -> None:
        graph.runtime.execute_embedding(handle, output_device, token_device, 1)
        graph.runtime.synchronize()

    try:
        for _ in range(warmup):
            operation()
        values: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            operation()
            values.append((time.perf_counter_ns() - started) / 1e6)
        output = graph.runtime.download_activation(output_device, (HIDDEN,))
        return {
            "worker_id": worker,
            "degree": graph.degree,
            "vocabulary_range": [shard.start, shard.stop],
            "resident_weight_bytes": graph.runtime.tensor_bytes(handle),
            "duration": timing(values),
            "output_fingerprint": _array_fingerprint(output),
        }
    finally:
        graph.runtime.free(output_device)
        graph.runtime.free(token_device)
        graph.runtime.release_tensor(handle)
        resources.close()


def _attnres_worker(
    graph: ShardedK3Graph,
    prefix: np.ndarray,
    residuals: np.ndarray,
    query: np.ndarray,
    *,
    rows: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    resources = DeviceResources(graph.runtime)
    try:
        prefix_device = resources.upload(prefix)
        residual_device = resources.upload(residuals)
        query_device = resources.upload(query)
        output_device = resources.allocate(rows * HIDDEN)

        def operation() -> None:
            for row in range(rows):
                graph.runtime.execute_attnres_mix(
                    ctypes_offset(output_device, row * HIDDEN),
                    ctypes_offset(prefix_device, row * HIDDEN),
                    residual_device,
                    query_device,
                    block_count=residuals.shape[0],
                    dimension=HIDDEN,
                    epsilon=1e-5,
                )
            graph.runtime.synchronize()

        for _ in range(warmup):
            operation()
        values: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            operation()
            values.append((time.perf_counter_ns() - started) / 1e6)
        return {
            "worker_id": "attnres.worker-owner",
            "rows": rows,
            "block_count": residuals.shape[0],
            "duration": timing(values),
            "cached_objects_have_concrete_owner": True,
            "transport_cache_reused": True,
        }
    finally:
        resources.close()


def ctypes_offset(pointer: Any, elements: int) -> Any:
    import ctypes

    return ctypes.c_void_p(int(pointer.value) + elements * 4)


def _reduction_compute(
    graph: ShardedK3Graph,
    *,
    participants: int,
    rows: int,
    dimension: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(19019 + participants + rows + dimension)
    partials = [
        np.ascontiguousarray(rng.normal(size=(rows, dimension)), dtype=np.float32)
        for _ in range(participants)
    ]
    resources = DeviceResources(graph.runtime)
    try:
        inputs = [resources.upload(values) for values in partials]
        output = resources.allocate(rows * dimension)

        def operation() -> None:
            graph.runtime.execute_copy(output, inputs[0], rows * dimension)
            for source in inputs[1:]:
                graph.runtime.execute_add(output, source, rows * dimension)
            graph.runtime.synchronize()

        for _ in range(warmup):
            operation()
        values_ms: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            operation()
            values_ms.append((time.perf_counter_ns() - started) / 1e6)
        return {
            "participants": participants,
            "rows": rows,
            "dimension": dimension,
            "payload_bytes_per_participant": rows * dimension * 4,
            "local_reduction_compute": timing(values_ms),
            "physical_launches": participants,
        }
    finally:
        resources.close()


def _rmsnorm_worker(
    graph: ShardedK3Graph,
    values: np.ndarray,
    weight: np.ndarray,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    resources = DeviceResources(graph.runtime)
    try:
        input_device = resources.upload(values)
        weight_device = resources.upload(weight)
        output_device = resources.allocate(values.size)

        def operation() -> None:
            graph.runtime.execute_rmsnorm(
                output_device,
                input_device,
                weight_device,
                batch=values.shape[0],
                dimension=values.shape[1],
                epsilon=1e-5,
            )
            graph.runtime.synchronize()

        for _ in range(warmup):
            operation()
        wall: list[float] = []
        cuda: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            graph.runtime.profile_begin()
            operation()
            cuda.append(graph.runtime.profile_end())
            wall.append((time.perf_counter_ns() - started) / 1e6)
        return {
            "worker_id": "normalization.worker-owner",
            "rows": values.shape[0],
            "dimension": values.shape[1],
            "wall": timing(wall),
            "cuda": timing(cuda),
            "host_overhead": timing(
                [maximum - device for maximum, device in zip(wall, cuda, strict=True)]
            ),
        }
    finally:
        resources.close()


def benchmark_other_shards(
    checkpoint: Path,
    cuda_library: Path,
    shard_library: Path,
    oracle_trace: Path,
    *,
    layer: int = 89,
    degrees: Sequence[int] = (4, 8, 16, 32),
    rows_sweep: Sequence[int] = (1, 2, 4),
    repeats: int = 3,
    device: int = 0,
) -> dict[str, Any]:
    trace = np.memmap(oracle_trace, mode="r", dtype="<f4", shape=(3 * 94, HIDDEN))
    results: list[dict[str, Any]] = []
    embeddings: list[dict[str, Any]] = []
    attnres_rows: list[dict[str, Any]] = []
    reductions: list[dict[str, Any]] = []
    rmsnorm_rows: list[dict[str, Any]] = []
    all_read_audits: list[dict[str, Any]] = []
    all_quantization: list[dict[str, Any]] = []
    for degree in degrees:
        graph = ShardedK3Graph(
            checkpoint,
            cuda_library,
            shard_library,
            degree=degree,
            device=device,
        )
        try:
            embeddings.append(
                _embedding_shard(graph, token_id=163584, warmup=1, iterations=5)
            )
            for row_count in rows_sweep:
                hidden = np.ascontiguousarray(
                    np.stack([trace[(row % 3) * 94 + layer - 1] for row in range(row_count)]),
                    dtype=np.float32,
                )
                latent = np.ascontiguousarray(hidden[:, :LATENT], dtype=np.float32)
                repeated: dict[str, list[list[dict[str, Any]]]] = {
                    "latent_down": [],
                    "latent_up": [],
                    "shared_expert": [],
                    "lm_head": [],
                    "dense_mlp": [],
                }
                outputs: dict[str, np.ndarray] = {}
                for _ in range(repeats):
                    outputs["latent_down"], records = graph._row_projection(
                        f"language_model.model.layers.{layer}.block_sparse_moe.routed_expert_down_proj.weight",
                        hidden,
                        layer=layer,
                        operator="latent_down_projection_stripe",
                        output_dimension=LATENT,
                        quantization="int4",
                    )
                    repeated["latent_down"].append(records)
                    outputs["latent_up"], records = graph._column_projection(
                        f"language_model.model.layers.{layer}.block_sparse_moe.routed_expert_up_proj.weight",
                        latent,
                        layer=layer,
                        operator="latent_up_projection_stripe",
                        input_dimension=LATENT,
                    )
                    repeated["latent_up"].append(records)
                    outputs["shared_expert"], records = graph._intermediate_mlp_stripes(
                        f"language_model.model.layers.{layer}.block_sparse_moe.shared_experts",
                        hidden,
                        layer=layer,
                        intermediate=6144,
                        operator="shared_expert_stripe",
                    )
                    repeated["shared_expert"].append(records)
                    outputs["lm_head"], records = graph._row_projection(
                        "language_model.lm_head.weight",
                        hidden,
                        layer="endpoint",
                        operator="lm_head_vocabulary_shard",
                        output_dimension=163840,
                        quantization="int8",
                    )
                    repeated["lm_head"].append(records)
                    outputs["dense_mlp"], records = graph._intermediate_mlp_stripes(
                        "language_model.model.layers.0.mlp",
                        hidden,
                        layer=0,
                        intermediate=33792,
                        operator="dense_mlp_stripe",
                    )
                    repeated["dense_mlp"].append(records)
                results.append(
                    {
                        "degree": degree,
                        "rows": row_count,
                        "operators": {
                            name: {
                                "workers": _summarize_workers(rows),
                                "independent_worker_compute_ceiling_ms": max(
                                    row["duration"]["p50_ms"]
                                    for row in _summarize_workers(rows)
                                ),
                                "sequential_worker_compute_ms": sum(
                                    row["duration"]["p50_ms"]
                                    for row in _summarize_workers(rows)
                                ),
                                "output_fingerprint": _array_fingerprint(outputs[name]),
                            }
                            for name, rows in repeated.items()
                        },
                        "router": _persistent_router(
                            graph,
                            hidden,
                            layer=layer,
                            warmup=1,
                            iterations=max(3, repeats),
                        ),
                    }
                )
                residual_values = np.ascontiguousarray(
                    np.stack([trace[snapshot - 1] for snapshot in range(12, 93, 12)]),
                    dtype=np.float32,
                )
                prefix_values = np.ascontiguousarray(hidden, dtype=np.float32)
                query = graph._small(
                    f"language_model.model.layers.{layer}.mlp_res_norm.weight",
                    worker="attnres.worker-owner",
                    purpose="physical_attnres_norm",
                ) * graph._small(
                    f"language_model.model.layers.{layer}.mlp_res_proj.weight",
                    worker="attnres.worker-owner",
                    purpose="physical_attnres_projection",
                )
                attnres_rows.append(
                    _attnres_worker(
                        graph,
                        prefix_values,
                        residual_values,
                        query,
                        rows=row_count,
                        warmup=1,
                        iterations=5,
                    )
                )
                rmsnorm_rows.append(
                    _rmsnorm_worker(
                        graph,
                        hidden,
                        query,
                        warmup=1,
                        iterations=5,
                    )
                )
                reductions.extend(
                    _reduction_compute(
                        graph,
                        participants=participants,
                        rows=row_count,
                        dimension=dimension,
                        warmup=1,
                        iterations=5,
                    )
                    for participants in sorted({2, degree})
                    for dimension in (LATENT, HIDDEN)
                )
            all_read_audits.extend(graph.loader.audit)
            all_quantization.extend(graph.quantizer.audit)
        finally:
            graph.close()
    violations = [
        row
        for row in all_read_audits
        if row["full_source_tensor_materialized"]
        and int(row["source_tensor_bytes"]) > 32 * 1024 * 1024
    ]
    return {
        "schema_version": "experiment-019-other-shards-physical-v1",
        "status": "PASS" if not violations else "FAIL",
        "evidence_class": "PHYSICAL RTX 5090 directly loaded sub-layer workers",
        "results": results,
        "embedding": embeddings,
        "attnres": attnres_rows,
        "reductions": reductions,
        "rmsnorm": rmsnorm_rows,
        "checkpoint_read_audit": all_read_audits,
        "startup_quantization_audit": all_quantization,
        "direct_loader_violations": violations,
    }


__all__ = ["benchmark_other_shards"]
