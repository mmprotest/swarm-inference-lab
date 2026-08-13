"""Exact Kimi K3 graph execution assembled only from sub-layer workers.

The single-GPU implementation runs logical workers sequentially.  That run is
an oracle/correctness experiment, never a distributed-throughput measurement.
Every material checkpoint read goes through :class:`DirectShardLoader`.
"""

from __future__ import annotations

import ctypes
import hashlib
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import (
    _array_fingerprint,
    _CudaRuntime,
    _numerical_metrics,
    _quantize_bf16_rows_int8,
    _rmsnorm_reference,
)
from swarm_inference.execution.kimi_k3_graph_runtime import (
    _quantize_bf16_grouped_int4,
)
from swarm_inference.experiments.experiment_018.analysis import (
    parse_oracle_routes,
)
from swarm_inference.experiments.experiment_019.attention import (
    DeviceResources,
    KdaShardKernel,
    _bf16_f32,
    execute_kda_attention,
    execute_mla_attention,
)
from swarm_inference.experiments.experiment_019.checkpoint import (
    CheckpointCatalog,
    DirectShardLoader,
    balanced_range,
)
from swarm_inference.experiments.experiment_019.physical import (
    HIDDEN,
    LATENT,
    _execute_local_route_sum,
    _upload_stripe_experts,
    striped_latent_down,
)
from swarm_inference.experiments.experiment_019.quantization import GpuShardQuantizer
from swarm_inference.experiments.experiment_019.placement import KDA_LAYERS

LAYERS = 93
VOCAB = 163840
TOPK = 16
EPSILON = 1e-5
RELATIVE_L2_GATE = 2e-5
SMALL_REPLICATION_LIMIT = 32 * 1024 * 1024


def _fingerprint_parts(parts: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        contiguous = np.ascontiguousarray(part)
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    return "sha256:" + digest.hexdigest()


class ShardedK3Graph:
    """Sequential physical oracle for a P-way sub-layer K3 placement."""

    def __init__(
        self,
        checkpoint: Path,
        cuda_library: Path,
        kda_shard_library: Path,
        *,
        degree: int = 4,
        device: int = 0,
    ) -> None:
        if degree < 4 or degree > 32:
            raise ValueError("full sharded graph requires stripe degree 4..32")
        self.checkpoint = checkpoint.resolve()
        self.catalog = CheckpointCatalog(self.checkpoint)
        self.loader = DirectShardLoader(self.catalog)
        self.runtime = _CudaRuntime(cuda_library.resolve(), device)
        self.runtime.set_telemetry("minimal")
        self.runtime.set_fused_gate_up(True)
        self.kda_kernel = KdaShardKernel(kda_shard_library.resolve(), device)
        self.quantizer = GpuShardQuantizer(kda_shard_library.resolve(), device)
        self.degree = degree
        self.worker_operations: list[dict[str, Any]] = []

    def close(self) -> None:
        self.runtime.close()

    def _small(self, name: str, *, worker: str, purpose: str) -> np.ndarray:
        source = self.loader.reviewed_small(
            name,
            worker_id=worker,
            purpose=purpose,
        )
        return _bf16_f32(source) if source.dtype == np.dtype("<u2") else np.asarray(source, dtype=np.float32)

    def _attnres(
        self,
        prefix: np.ndarray,
        residuals: Sequence[np.ndarray],
        query: np.ndarray,
        *,
        worker: str,
        operator: str,
    ) -> np.ndarray:
        if not residuals:
            return np.ascontiguousarray(prefix, dtype=np.float32)
        resources = DeviceResources(self.runtime)
        started = time.perf_counter_ns()
        try:
            prefix_device = resources.upload(prefix)
            residual_values = np.ascontiguousarray(np.stack(residuals), dtype=np.float32)
            residual_device = resources.upload(residual_values)
            query_device = resources.upload(query)
            output = resources.allocate(HIDDEN)
            self.runtime.execute_attnres_mix(
                output,
                prefix_device,
                residual_device,
                query_device,
                block_count=len(residuals),
                dimension=HIDDEN,
                epsilon=EPSILON,
            )
            self.runtime.synchronize()
            value = self.runtime.download_activation(output, (HIDDEN,))
        finally:
            resources.close()
        self.worker_operations.append(
            {
                "worker_id": worker,
                "operator": operator,
                "resource_type": "microworker",
                "duration_ms": (time.perf_counter_ns() - started) / 1e6,
                "input_bytes": prefix.nbytes + sum(row.nbytes for row in residuals),
                "output_bytes": value.nbytes,
            }
        )
        return value

    def _route(
        self,
        mlp_input: np.ndarray,
        *,
        layer: int,
    ) -> tuple[np.ndarray, np.ndarray]:
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
            input_device = resources.upload(mlp_input)
            router_device = resources.upload(router)
            bias_device = resources.upload(bias)
            ids, weights, effective = self.runtime.route(
                input_device,
                router_device,
                bias_device,
                hidden=HIDDEN,
                experts=896,
                topk=TOPK,
            )
            if effective != TOPK:
                raise RuntimeError(f"layer {layer} router retained {effective} experts")
        finally:
            resources.close()
        self.worker_operations.append(
            {
                "worker_id": worker,
                "operator": "router",
                "resource_type": "microworker",
                "duration_ms": (time.perf_counter_ns() - started) / 1e6,
                "route_metadata_bytes": TOPK * 8,
            }
        )
        return np.asarray(ids, dtype=np.int32), np.asarray(weights, dtype=np.float32)

    def _row_projection(
        self,
        name: str,
        values: np.ndarray,
        *,
        layer: int | str,
        operator: str,
        output_dimension: int,
        quantization: str,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        rows = int(values.shape[0])
        outputs: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        for stripe in range(self.degree):
            shard = balanced_range(output_dimension, self.degree, stripe)
            worker = f"layer-{layer}.{operator}.worker-{stripe:02d}"
            source = self.loader.load(
                name,
                worker_id=worker,
                purpose=f"{operator}_output_row_stripe",
                axis=0,
                start=shard.start,
                stop=shard.stop,
            )
            if quantization == "int4":
                handle = self.runtime.upload_grouped_int4(
                    self.quantizer.grouped_int4(source, owner=worker)
                )
            elif quantization == "int8":
                handle = self.runtime.upload_int8(
                    self.quantizer.row_int8(source, owner=worker)
                )
            else:
                raise ValueError(f"unsupported projection quantization: {quantization}")
            resources = DeviceResources(self.runtime)
            input_device = resources.upload(values)
            output_device = resources.allocate(rows * (shard.stop - shard.start))
            started = time.perf_counter_ns()
            try:
                self.runtime.profile_begin()
                self.runtime.execute_dense(handle, output_device, input_device, rows)
                self.runtime.synchronize()
                cuda_ms = self.runtime.profile_end()
                wall_ms = (time.perf_counter_ns() - started) / 1e6
                output = self.runtime.download_activation(
                    output_device, (rows, shard.stop - shard.start)
                )
                outputs.append(output)
                record = {
                    "worker_id": worker,
                    "operator": operator,
                    "resource_type": "microworker",
                    "stripe_index": stripe,
                    "output_range": [shard.start, shard.stop],
                    "checkpoint_bytes": int(source.nbytes),
                    "runtime_weight_bytes": int(self.runtime.tensor_bytes(handle)),
                    "duration_ms": wall_ms,
                    "cuda_ms": cuda_ms,
                    "host_overhead_ms": wall_ms - cuda_ms,
                    "input_bytes": int(values.nbytes),
                    "output_bytes": int(output.nbytes),
                }
                records.append(record)
                self.worker_operations.append(record)
            finally:
                resources.close()
                self.runtime.release_tensor(handle)
        return np.ascontiguousarray(np.concatenate(outputs, axis=1)), records

    def _column_projection(
        self,
        name: str,
        values: np.ndarray,
        *,
        layer: int | str,
        operator: str,
        input_dimension: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        record = self.catalog.record(name)
        output_dimension = record.shape[0]
        if record.shape != (output_dimension, input_dimension):
            raise ValueError("column projection geometry differs from checkpoint")
        partials: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        for stripe in range(self.degree):
            shard = balanced_range(input_dimension, self.degree, stripe, quantum=64)
            worker = f"layer-{layer}.{operator}.worker-{stripe:02d}"
            source = self.loader.load(
                name,
                worker_id=worker,
                purpose=f"{operator}_input_column_stripe",
                axis=1,
                start=shard.start,
                stop=shard.stop,
            )
            tensor = self.quantizer.grouped_int4(source, owner=worker)
            handle = self.runtime.upload_grouped_int4(tensor)
            resources = DeviceResources(self.runtime)
            input_values = np.ascontiguousarray(values[:, shard.start : shard.stop])
            input_device = resources.upload(input_values)
            output_device = resources.allocate(values.shape[0] * output_dimension)
            started = time.perf_counter_ns()
            try:
                self.runtime.profile_begin()
                self.runtime.execute_dense(
                    handle, output_device, input_device, values.shape[0]
                )
                self.runtime.synchronize()
                cuda_ms = self.runtime.profile_end()
                wall_ms = (time.perf_counter_ns() - started) / 1e6
                partial = self.runtime.download_activation(
                    output_device, (values.shape[0], output_dimension)
                )
                partials.append(partial)
                worker_record = {
                    "worker_id": worker,
                    "operator": operator,
                    "resource_type": "microworker",
                    "stripe_index": stripe,
                    "input_range": [shard.start, shard.stop],
                    "checkpoint_bytes": int(source.nbytes),
                    "runtime_weight_bytes": int(self.runtime.tensor_bytes(handle)),
                    "duration_ms": wall_ms,
                    "cuda_ms": cuda_ms,
                    "host_overhead_ms": wall_ms - cuda_ms,
                    "input_bytes": int(input_values.nbytes),
                    "output_bytes": int(partial.nbytes),
                    "network_visible_partial_outputs": 1,
                }
                records.append(worker_record)
                self.worker_operations.append(worker_record)
            finally:
                resources.close()
                self.runtime.release_tensor(handle)
        output = np.sum(
            np.stack(partials, axis=0), axis=0, dtype=np.float64
        ).astype(np.float32)
        return output, records

    def _intermediate_mlp_stripes(
        self,
        prefix: str,
        values: np.ndarray,
        *,
        layer: int,
        intermediate: int,
        operator: str,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        partials: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        for stripe in range(self.degree):
            # Dense layer 0 has checkpoint-authoritative width 33,792.  At
            # P=32 an equal 1,056 split cuts native 64-value MXFP4 groups;
            # balance the integer group count instead (17/16 groups).
            shard = balanced_range(intermediate, self.degree, stripe, quantum=64)
            if shard.start % 64 or shard.stop % 64:
                raise RuntimeError("MLP intermediate stripes must preserve MXFP4 groups")
            worker = f"layer-{layer:02d}.{operator}.worker-{stripe:02d}"
            sources = (
                self.loader.load(
                    f"{prefix}.gate_proj.weight",
                    worker_id=worker,
                    purpose=f"{operator}_gate_intermediate_rows",
                    axis=0,
                    start=shard.start,
                    stop=shard.stop,
                ),
                self.loader.load(
                    f"{prefix}.up_proj.weight",
                    worker_id=worker,
                    purpose=f"{operator}_up_intermediate_rows",
                    axis=0,
                    start=shard.start,
                    stop=shard.stop,
                ),
                self.loader.load(
                    f"{prefix}.down_proj.weight",
                    worker_id=worker,
                    purpose=f"{operator}_down_intermediate_columns",
                    axis=1,
                    start=shard.start,
                    stop=shard.stop,
                ),
            )
            handles = tuple(
                self.runtime.upload_grouped_int4(
                    self.quantizer.grouped_int4(source, owner=worker)
                )
                for source in sources
            )
            resources = DeviceResources(self.runtime)
            input_device = resources.upload(values)
            output_device = resources.allocate(values.shape[0] * HIDDEN)
            started = time.perf_counter_ns()
            try:
                self.runtime.profile_begin()
                self.runtime.execute_resident(handles, output_device, input_device, values.shape[0])
                self.runtime.synchronize()
                cuda_ms = self.runtime.profile_end()
                wall_ms = (time.perf_counter_ns() - started) / 1e6
                partial = self.runtime.download_activation(
                    output_device, (values.shape[0], HIDDEN)
                )
                partials.append(partial)
                record = {
                    "worker_id": worker,
                    "operator": operator,
                    "resource_type": "microworker",
                    "stripe_index": stripe,
                    "intermediate_range": [shard.start, shard.stop],
                    "checkpoint_bytes": int(sum(source.nbytes for source in sources)),
                    "runtime_weight_bytes": int(
                        sum(self.runtime.tensor_bytes(handle) for handle in handles)
                    ),
                    "duration_ms": wall_ms,
                    "cuda_ms": cuda_ms,
                    "host_overhead_ms": wall_ms - cuda_ms,
                    "network_visible_partial_outputs": 1,
                    "physical_launches": 1,
                }
                records.append(record)
                self.worker_operations.append(record)
            finally:
                resources.close()
                for handle in handles:
                    self.runtime.release_tensor(handle)
        # The collective uses a deterministic FP64 accumulator before the
        # exact FP32 boundary cast.  This avoids compounding reassociation
        # error across independently accumulated intermediate stripes.
        output = np.sum(
            np.stack(partials, axis=0), axis=0, dtype=np.float64
        ).astype(np.float32)
        return output, records

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
            try:
                measurement, partial = _execute_local_route_sum(
                    self.runtime,
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
                    "operator": "expert_stripe_bank",
                    "resource_type": "microworker",
                    "stripe_index": stripe,
                    "active_experts": len(experts),
                    "runtime_weight_bytes": resident.runtime_bytes,
                    "duration_ms": measurement["wall"]["p50_ms"],
                    "cuda_ms": measurement["cuda"]["p50_ms"],
                    "host_overhead_ms": measurement["wall"]["p50_ms"]
                    - measurement["cuda"]["p50_ms"],
                    "logical_expert_operations": TOPK,
                    "physical_launches": TOPK + 1,
                    "network_visible_partial_outputs": 1,
                }
                records.append(record)
                self.worker_operations.append(record)
            finally:
                resident.close()
        output = np.sum(
            np.stack(partials, axis=0), axis=0, dtype=np.float64
        ).astype(np.float32)
        return output, records

    def embedding(self, token_id: int) -> tuple[np.ndarray, dict[str, Any]]:
        shard = balanced_range(VOCAB, self.degree, 0)
        owner = None
        for stripe in range(self.degree):
            candidate = balanced_range(VOCAB, self.degree, stripe)
            if candidate.start <= token_id < candidate.stop:
                owner = stripe
                shard = candidate
                break
        if owner is None:
            raise RuntimeError("embedding token has no vocabulary owner")
        worker = f"endpoint.embedding.worker-{owner:02d}"
        source = self.loader.load(
            "language_model.model.embed_tokens.weight",
            worker_id=worker,
            purpose="embedding_vocabulary_row_owned_read",
            axis=0,
            start=token_id,
            stop=token_id + 1,
        )
        value = _bf16_f32(source)
        record = {
            "worker_id": worker,
            "operator": "embedding_vocabulary_shard",
            "resource_type": "microworker",
            "vocabulary_range": [shard.start, shard.stop],
            "token_id": token_id,
            "checkpoint_bytes": int(source.nbytes),
        }
        self.worker_operations.append(record)
        return value, record

    def execute_layer(
        self,
        layer: int,
        hidden: np.ndarray,
        residuals: list[np.ndarray],
    ) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
        incoming = np.ascontiguousarray(hidden.reshape(HIDDEN), dtype=np.float32)
        is_snapshot = layer % 12 == 0
        prefix = f"language_model.model.layers.{layer}"
        attention_score = self._small(
            f"{prefix}.self_attention_res_norm.weight",
            worker=f"layer-{layer:02d}.attnres-worker",
            purpose="replicated_small_attention_residual_norm",
        ) * self._small(
            f"{prefix}.self_attention_res_proj.weight",
            worker=f"layer-{layer:02d}.attnres-worker",
            purpose="replicated_small_attention_residual_projection",
        )
        attention_input = self._attnres(
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
        normalized = self.quantizer.rmsnorm(
            attention_input,
            input_norm,
            owner=f"layer-{layer:02d}.norm-worker",
            epsilon=EPSILON,
        )
        attention_executor = (
            execute_kda_attention
            if f"{prefix}.self_attn.q_proj.weight" in self.catalog.records()
            else execute_mla_attention
        )
        arguments: dict[str, Any] = {}
        if attention_executor is execute_kda_attention:
            arguments["shard_kernel"] = self.kda_kernel
        arguments["quantizer"] = self.quantizer
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
        for worker in attention_record["workers"]:
            self.worker_operations.append(
                {
                    "worker_id": worker["worker_id"],
                    "operator": f"{attention_record['attention_type']}_attention_stripe",
                    "resource_type": "microworker",
                    "duration_ms": worker["wall"]["p50_ms"],
                    "runtime_weight_bytes": worker["runtime_weight_bytes"],
                }
            )
        layer_prefix = attention_output[0].copy() if is_snapshot else incoming + attention_output[0]
        mlp_score = self._small(
            f"{prefix}.mlp_res_norm.weight",
            worker=f"layer-{layer:02d}.attnres-worker",
            purpose="replicated_small_mlp_residual_norm",
        ) * self._small(
            f"{prefix}.mlp_res_proj.weight",
            worker=f"layer-{layer:02d}.attnres-worker",
            purpose="replicated_small_mlp_residual_projection",
        )
        mixed = self._attnres(
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
        mlp_input = self.quantizer.rmsnorm(
            mixed,
            post_norm,
            owner=f"layer-{layer:02d}.norm-worker",
            epsilon=EPSILON,
        )
        routes_record: dict[str, Any] | None = None
        if layer == 0:
            mlp_output, mlp_workers = self._intermediate_mlp_stripes(
                f"{prefix}.mlp",
                mlp_input,
                layer=layer,
                intermediate=33792,
                operator="dense_mlp_stripe",
            )
        else:
            routes, route_weights = self._route(mlp_input, layer=layer)
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
                        "duration_ms": worker["wall"]["p50_ms"],
                        "runtime_weight_bytes": worker["runtime_weight_bytes"],
                    }
                )
            expert_output, expert_workers = self._routed_experts(
                latent, routes[None, :], route_weights[None, :], layer=layer
            )
            routed_norm = self._small(
                f"{prefix}.block_sparse_moe.routed_expert_norm.weight",
                worker=f"layer-{layer:02d}.routed-norm-worker",
                purpose="replicated_small_routed_expert_norm",
            )
            routed = self.quantizer.rmsnorm(
                expert_output[0],
                routed_norm,
                owner=f"layer-{layer:02d}.routed-norm-worker",
                epsilon=EPSILON,
            )
            routed_up, routed_up_workers = self._column_projection(
                f"{prefix}.block_sparse_moe.routed_expert_up_proj.weight",
                routed,
                layer=layer,
                operator="latent_up_projection_stripe",
                input_dimension=LATENT,
            )
            shared, shared_workers = self._intermediate_mlp_stripes(
                f"{prefix}.block_sparse_moe.shared_experts",
                mlp_input,
                layer=layer,
                intermediate=6144,
                operator="shared_expert_stripe",
            )
            mlp_output = routed_up + shared
            mlp_workers = [
                *latent_workers,
                *expert_workers,
                *routed_up_workers,
                *shared_workers,
            ]
            routes_record = {
                "selected_expert_ids": [int(value) for value in routes],
                "selected_weights": [float(value) for value in route_weights],
            }
        output = np.ascontiguousarray(layer_prefix + mlp_output[0], dtype=np.float32)
        return output, residuals, {
            "layer": layer,
            "attention_type": attention_record["attention_type"],
            "attention": attention_record,
            "routes": routes_record,
            "mlp_worker_count": len(mlp_workers),
            "input_fingerprint": _array_fingerprint(incoming),
            "output_fingerprint": _array_fingerprint(output),
        }

    def final_head(
        self,
        hidden: np.ndarray,
        residuals: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        score = self._small(
            "language_model.model.output_attn_res_norm.weight",
            worker="endpoint.attnres-worker",
            purpose="replicated_small_output_attnres_norm",
        ) * self._small(
            "language_model.model.output_attn_res_proj.weight",
            worker="endpoint.attnres-worker",
            purpose="replicated_small_output_attnres_projection",
        )
        mixed = self._attnres(
            hidden,
            residuals,
            score,
            worker="endpoint.attnres-worker",
            operator="final_attnres_mix",
        )
        norm = self._small(
            "language_model.model.norm.weight",
            worker="endpoint.norm-worker",
            purpose="replicated_small_final_norm",
        )
        final_hidden = self.quantizer.rmsnorm(
            mixed,
            norm,
            owner="endpoint.norm-worker",
            epsilon=EPSILON,
        )
        logits, workers = self._row_projection(
            "language_model.lm_head.weight",
            final_hidden,
            layer="endpoint",
            operator="lm_head_vocabulary_shard",
            output_dimension=VOCAB,
            quantization="int8",
        )
        local_candidates = []
        for worker in workers:
            start, stop = worker["output_range"]
            local = logits[0, start:stop]
            local_index = int(np.argmax(local))
            local_candidates.append(
                {
                    "worker_id": worker["worker_id"],
                    "token_id": start + local_index,
                    "logit": float(local[local_index]),
                }
            )
        winner = max(local_candidates, key=lambda row: (row["logit"], -row["token_id"]))
        return final_hidden, logits, {
            "distributed_exact_argmax": winner,
            "local_candidates": local_candidates,
            "final_hidden_fingerprint": _array_fingerprint(final_hidden),
            "logits_fingerprint": _array_fingerprint(logits),
        }

    def execute_full_oracle(
        self,
        oracle_trace: Path,
        oracle_routes_path: Path,
        oracle_logits_path: Path,
        *,
        token_id: int = 163584,
        layer_limit: int = LAYERS,
        progress: bool = True,
    ) -> dict[str, Any]:
        trace = np.memmap(
            oracle_trace,
            mode="r",
            dtype="<f4",
            shape=(3 * (LAYERS + 1), HIDDEN),
        )
        expected_routes = parse_oracle_routes(oracle_routes_path)
        hidden, embedding = self.embedding(token_id)
        hidden = hidden[0]
        residuals: list[np.ndarray] = []
        layers: list[dict[str, Any]] = []
        maximum_error = 0.0
        routes_exact = True
        started = time.perf_counter_ns()
        for layer in range(layer_limit):
            layer_started = time.perf_counter_ns()
            hidden, residuals, record = self.execute_layer(layer, hidden, residuals)
            expected = np.ascontiguousarray(trace[layer], dtype=np.float32)
            metrics = _numerical_metrics(expected, hidden)
            record["oracle_correctness"] = metrics
            record["wall_ms"] = (time.perf_counter_ns() - layer_started) / 1e6
            maximum_error = max(maximum_error, float(metrics["relative_l2_error"]))
            if record["routes"] is not None:
                expected_route = expected_routes[layer][0]
                equal = (
                    tuple(int(value) for value in record["routes"]["selected_expert_ids"])
                    == expected_route
                )
                record["routes"]["ordered_ids_match_oracle"] = equal
                routes_exact = routes_exact and equal
            layers.append(record)
            if progress:
                print(
                    f"[exp019 shard oracle] layer {layer + 1:02d}/{layer_limit} "
                    f"rel_l2={float(metrics['relative_l2_error']):.3e} "
                    f"wall={record['wall_ms']:.1f} ms",
                    flush=True,
                )
        head: dict[str, Any] | None = None
        if layer_limit == LAYERS:
            final_hidden, logits, head = self.final_head(hidden, residuals)
            reference_logits = np.memmap(
                oracle_logits_path,
                mode="r",
                dtype="<f4",
                shape=(2, VOCAB),
            )[0:1]
            head["oracle_logits"] = _numerical_metrics(reference_logits, logits)
            head["oracle_greedy_token"] = int(np.argmax(reference_logits[0]))
            head["greedy_token_match"] = (
                head["distributed_exact_argmax"]["token_id"]
                == head["oracle_greedy_token"]
            )
            maximum_error = max(
                maximum_error,
                float(head["oracle_logits"]["relative_l2_error"]),
            )
        full_material = [
            row
            for row in self.loader.audit
            if bool(row["full_source_tensor_materialized"])
            and int(row["source_tensor_bytes"]) > SMALL_REPLICATION_LIMIT
        ]
        status = (
            "PASS"
            if layer_limit == LAYERS
            and maximum_error <= RELATIVE_L2_GATE
            and routes_exact
            and head is not None
            and bool(head["greedy_token_match"])
            and not full_material
            else "FAIL"
        )
        return {
            "schema_version": "experiment-019-full-sharded-oracle-v1",
            "status": status,
            "evidence_class": "PHYSICAL sequential sub-layer workers on one RTX 5090",
            "stripe_degree": self.degree,
            "executed_layers": layer_limit,
            "complete_93_layer_graph": layer_limit == LAYERS,
            "embedding": embedding,
            "layers": layers,
            "head": head,
            "maximum_relative_l2_error": maximum_error,
            "routes_exact": routes_exact,
            "state_fingerprints": [
                {
                    "layer": row["layer"],
                    "workers": [
                        worker.get("state_fingerprint")
                        for worker in row["attention"]["workers"]
                        if worker.get("state_fingerprint")
                    ],
                }
                for row in layers
            ],
            "direct_loader_gate": {
                "full_material_tensors_over_review_threshold": len(full_material),
                "violations": full_material,
                "request_count": len(self.loader.audit),
                "bytes_read": sum(int(row["bytes_read"]) for row in self.loader.audit),
            },
            "worker_operation_count": len(self.worker_operations),
            "all_compute_resources_are_microworkers": all(
                row.get("resource_type") == "microworker"
                for row in self.worker_operations
            ),
            "wall_seconds": (time.perf_counter_ns() - started) / 1e9,
            "checkpoint_read_audit": self.loader.audit,
            "startup_quantization_audit": self.quantizer.audit,
        }


def validate_representative_layers(
    checkpoint: Path,
    cuda_library: Path,
    kda_shard_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    *,
    degrees: Sequence[int] = (4, 8, 16, 32),
    layers: Sequence[int] = (89, 91),
    device: int = 0,
) -> dict[str, Any]:
    trace = np.memmap(oracle_trace, mode="r", dtype="<f4", shape=(3 * 94, HIDDEN))
    results: list[dict[str, Any]] = []
    for degree in degrees:
        graph = ShardedK3Graph(
            checkpoint,
            cuda_library,
            kda_shard_library,
            degree=degree,
            device=device,
        )
        try:
            for layer in layers:
                hidden = np.ascontiguousarray(trace[layer - 1], dtype=np.float32)
                residuals = [
                    np.ascontiguousarray(trace[snapshot - 1], dtype=np.float32)
                    if snapshot
                    else graph.embedding(163584)[0][0]
                    for snapshot in range(0, layer + 1, 12)
                ]
                actual, _residuals, record = graph.execute_layer(layer, hidden, residuals)
                expected = np.ascontiguousarray(trace[layer], dtype=np.float32)
                metrics = _numerical_metrics(expected, actual)
                results.append(
                    {
                        "degree": degree,
                        "layer": layer,
                        "attention_type": record["attention_type"],
                        "metrics": metrics,
                        "routes": record["routes"],
                        "pass": float(metrics["relative_l2_error"]) <= RELATIVE_L2_GATE,
                    }
                )
        finally:
            graph.close()
    return {
        "schema_version": "experiment-019-representative-sharded-layers-v1",
        "status": "PASS" if all(row["pass"] for row in results) else "FAIL",
        "relative_l2_gate": RELATIVE_L2_GATE,
        "results": results,
    }


def validate_depth_spans(
    checkpoint: Path,
    cuda_library: Path,
    kda_shard_library: Path,
    oracle_trace: Path,
    oracle_routes_path: Path,
    *,
    degree: int = 4,
    spans: Sequence[int] = (2, 4, 8),
    end_layer_exclusive: int = 92,
    device: int = 0,
) -> dict[str, Any]:
    trace = np.memmap(oracle_trace, mode="r", dtype="<f4", shape=(3 * 94, HIDDEN))
    expected_routes = parse_oracle_routes(oracle_routes_path)
    results: list[dict[str, Any]] = []
    for span in spans:
        start = end_layer_exclusive - span
        graph = ShardedK3Graph(
            checkpoint,
            cuda_library,
            kda_shard_library,
            degree=degree,
            device=device,
        )
        try:
            hidden = np.ascontiguousarray(trace[start - 1], dtype=np.float32)
            embedding = graph.embedding(163584)[0][0]
            residuals = [
                embedding
                if snapshot == 0
                else np.ascontiguousarray(trace[snapshot - 1], dtype=np.float32)
                # A snapshot at ``start`` is appended by execute_layer only
                # after the attention-side AttnRes mix.  Preloading it here
                # would expose future state and then append it a second time.
                for snapshot in range(0, start, 12)
            ]
            layers: list[dict[str, Any]] = []
            routes_exact = True
            maximum_error = 0.0
            for layer in range(start, end_layer_exclusive):
                hidden, residuals, record = graph.execute_layer(layer, hidden, residuals)
                metrics = _numerical_metrics(
                    np.ascontiguousarray(trace[layer], dtype=np.float32), hidden
                )
                route_equal = True
                if record["routes"] is not None:
                    route_equal = (
                        tuple(
                            int(value)
                            for value in record["routes"]["selected_expert_ids"]
                        )
                        == expected_routes[layer][0]
                    )
                routes_exact = routes_exact and route_equal
                maximum_error = max(
                    maximum_error, float(metrics["relative_l2_error"])
                )
                layers.append(
                    {
                        "layer": layer,
                        "attention_type": record["attention_type"],
                        "metrics": metrics,
                        "ordered_routes_exact": route_equal,
                    }
                )
            results.append(
                {
                    "span": span,
                    "layer_start": start,
                    "layer_end_exclusive": end_layer_exclusive,
                    "contains_kda": any(layer in KDA_LAYERS for layer in range(start, end_layer_exclusive)),
                    "contains_mla": any(layer not in KDA_LAYERS for layer in range(start, end_layer_exclusive)),
                    "maximum_relative_l2_error": maximum_error,
                    "routes_exact": routes_exact,
                    "layers": layers,
                    "pass": maximum_error <= RELATIVE_L2_GATE and routes_exact,
                }
            )
        finally:
            graph.close()
    return {
        "schema_version": "experiment-019-depth-span-correctness-v1",
        "status": "PASS" if all(row["pass"] for row in results) else "FAIL",
        "stripe_degree": degree,
        "relative_l2_gate": RELATIVE_L2_GATE,
        "results": results,
    }


__all__ = [
    "ShardedK3Graph",
    "validate_depth_spans",
    "validate_representative_layers",
]
