"""Canonical-control diagnostics used during Experiment 019 redesign loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _numerical_metrics
from swarm_inference.execution.kimi_k3_graph_runtime import (
    KimiCudaGraphRunner,
    _LayerResources,
)
from swarm_inference.experiments.experiment_019.attention import (
    execute_kda_attention,
    execute_mla_attention,
)
from swarm_inference.experiments.experiment_019.sharded_graph import (
    EPSILON,
    HIDDEN,
    ShardedK3Graph,
)
from swarm_inference.execution.kimi_cuda_runtime import _rmsnorm_reference


def compare_attention_control(
    checkpoint: Path,
    cuda_library: Path,
    kda_shard_library: Path,
    oracle_trace: Path,
    *,
    layer: int,
    degree: int = 4,
    device: int = 0,
) -> dict[str, Any]:
    trace = np.memmap(oracle_trace, mode="r", dtype="<f4", shape=(3 * 94, HIDDEN))
    graph = ShardedK3Graph(
        checkpoint,
        cuda_library,
        kda_shard_library,
        degree=degree,
        device=device,
    )
    canonical = KimiCudaGraphRunner(checkpoint, cuda_library, device)
    canonical_resources = _LayerResources(canonical.runtime)
    try:
        prefix = f"language_model.model.layers.{layer}"
        hidden = np.ascontiguousarray(trace[layer - 1], dtype=np.float32)
        residuals = [
            graph.embedding(163584)[0][0]
            if snapshot == 0
            else np.ascontiguousarray(trace[snapshot - 1], dtype=np.float32)
            for snapshot in range(0, layer + 1, 12)
        ]
        score = graph._small(
            f"{prefix}.self_attention_res_norm.weight",
            worker="diagnostic.attnres",
            purpose="canonical_control_small_norm",
        ) * graph._small(
            f"{prefix}.self_attention_res_proj.weight",
            worker="diagnostic.attnres",
            purpose="canonical_control_small_projection",
        )
        mixed = graph._attnres(
            hidden,
            residuals,
            score,
            worker="diagnostic.attnres",
            operator="canonical_control_attnres",
        )
        norm = graph._small(
            f"{prefix}.input_layernorm.weight",
            worker="diagnostic.norm",
            purpose="canonical_control_small_input_norm",
        )
        normalized = _rmsnorm_reference(mixed, norm, EPSILON)[None, :]
        executor = (
            execute_kda_attention
            if f"{prefix}.self_attn.q_proj.weight" in graph.catalog.records()
            else execute_mla_attention
        )
        arguments: dict[str, Any] = {}
        if executor is execute_kda_attention:
            arguments["shard_kernel"] = graph.kda_kernel
        arguments["quantizer"] = graph.quantizer
        sharded, sharded_record = executor(
            graph.runtime,
            graph.loader,
            normalized,
            layer=layer,
            degree=degree,
            warmup=0,
            iterations=1,
            **arguments,
        )

        weights = canonical._load_layer_weights(canonical_resources, layer)
        state = canonical._prepare_attention_state(
            canonical_resources, layer, maximum_context=256
        )
        scratch = canonical._attention_scratch(canonical_resources, layer)
        normalized_device = canonical_resources.upload_data(normalized)
        output_device = canonical_resources.allocate(HIDDEN)
        canonical._execute_attention(
            canonical_resources,
            weights,
            state,
            normalized_device,
            output_device,
            layer=layer,
            position=0,
            scratch=scratch,
        )
        canonical.runtime.synchronize()
        reference = canonical.runtime.download_activation(output_device, (1, HIDDEN))
        return {
            "layer": layer,
            "degree": degree,
            "attention_type": sharded_record["attention_type"],
            "metrics": _numerical_metrics(reference, sharded),
            "sharded_record": sharded_record,
        }
    finally:
        canonical_resources.close()
        canonical.close()
        graph.close()


__all__ = ["compare_attention_control"]


def compare_moe_control(
    checkpoint: Path,
    cuda_library: Path,
    kda_shard_library: Path,
    oracle_trace: Path,
    *,
    layer: int,
    degree: int = 4,
    device: int = 0,
) -> dict[str, Any]:
    """Compare the striped complete MoE tail with the canonical whole tail."""
    trace = np.memmap(oracle_trace, mode="r", dtype="<f4", shape=(3 * 94, HIDDEN))
    graph = ShardedK3Graph(
        checkpoint,
        cuda_library,
        kda_shard_library,
        degree=degree,
        device=device,
    )
    canonical = KimiCudaGraphRunner(checkpoint, cuda_library, device)
    resources = _LayerResources(canonical.runtime)
    try:
        prefix = f"language_model.model.layers.{layer}"
        # Any real layer-boundary vector is sufficient: both paths receive the
        # same normalized value, so this isolates only MoE decomposition.
        norm = graph._small(
            f"{prefix}.post_attention_layernorm.weight",
            worker="diagnostic.moe-norm",
            purpose="canonical_control_post_attention_norm",
        )
        mlp_input = _rmsnorm_reference(
            np.ascontiguousarray(trace[layer - 1], dtype=np.float32), norm, EPSILON
        )[None, :]
        routes, route_weights = graph._route(mlp_input, layer=layer)
        latent, _ = graph._row_projection(
            f"{prefix}.block_sparse_moe.routed_expert_down_proj.weight",
            mlp_input,
            layer=layer,
            operator="diagnostic_latent_down",
            output_dimension=3584,
            quantization="int4",
        )
        expert_output, _ = graph._routed_experts(
            latent, routes[None, :], route_weights[None, :], layer=layer
        )
        routed_norm = graph._small(
            f"{prefix}.block_sparse_moe.routed_expert_norm.weight",
            worker="diagnostic.routed-norm",
            purpose="canonical_control_routed_norm",
        )
        routed = _rmsnorm_reference(expert_output[0], routed_norm, EPSILON)[None, :]
        routed_up, _ = graph._column_projection(
            f"{prefix}.block_sparse_moe.routed_expert_up_proj.weight",
            routed,
            layer=layer,
            operator="diagnostic_latent_up",
            input_dimension=3584,
        )
        shared, _ = graph._intermediate_mlp_stripes(
            f"{prefix}.block_sparse_moe.shared_experts",
            mlp_input,
            layer=layer,
            intermediate=6144,
            operator="diagnostic_shared_expert",
        )
        sharded = routed_up + shared

        weights = canonical._load_layer_weights(resources, layer)
        input_device = resources.upload_data(mlp_input)
        zero = np.zeros((1, HIDDEN), dtype=np.float32)
        prefix_device = resources.upload_data(zero)
        input_for_route = input_device
        ids, weights_values, effective = canonical.runtime.route(
            input_for_route,
            weights["router"],
            weights["router_bias"],
            hidden=HIDDEN,
            experts=896,
            topk=16,
        )
        if effective != 16:
            raise RuntimeError("canonical control route did not retain top-16")
        canonical._execute_sparse_mlp_rows(
            resources,
            weights,
            input_device,
            prefix_device,
            [
                {
                    "selected_expert_ids": [int(value) for value in ids],
                    "selected_weights": [float(value) for value in weights_values],
                }
            ],
            layer=layer,
        )
        canonical.runtime.synchronize()
        reference = canonical.runtime.download_activation(prefix_device, (1, HIDDEN))
        return {
            "layer": layer,
            "degree": degree,
            "route_ids_equal": [int(value) for value in ids]
            == [int(value) for value in routes],
            "route_weight_metrics": _numerical_metrics(
                np.asarray(weights_values, dtype=np.float32), route_weights
            ),
            "metrics": _numerical_metrics(reference, sharded),
        }
    finally:
        resources.close()
        canonical.close()
        graph.close()


__all__.append("compare_moe_control")


def compare_complete_layer_control(
    checkpoint: Path,
    cuda_library: Path,
    kda_shard_library: Path,
    oracle_trace: Path,
    *,
    layer: int,
    degree: int = 4,
    device: int = 0,
) -> dict[str, Any]:
    trace = np.memmap(oracle_trace, mode="r", dtype="<f4", shape=(3 * 94, HIDDEN))
    graph = ShardedK3Graph(
        checkpoint,
        cuda_library,
        kda_shard_library,
        degree=degree,
        device=device,
    )
    canonical = KimiCudaGraphRunner(checkpoint, cuda_library, device)
    try:
        hidden = np.ascontiguousarray(trace[layer - 1], dtype=np.float32)
        embedding = graph.embedding(163584)[0][0]
        residuals = [
            embedding
            if snapshot == 0
            else np.ascontiguousarray(trace[snapshot - 1], dtype=np.float32)
            for snapshot in range(0, layer + 1, 12)
        ]
        sharded, _next_residuals, _record = graph.execute_layer(
            layer, hidden, [row.copy() for row in residuals]
        )
        canonical_residuals = np.zeros((1, 8, HIDDEN), dtype=np.float32)
        canonical_residuals[0, : len(residuals)] = np.stack(residuals)
        canonical_output, _count, canonical_record = canonical.execute_layer(
            layer,
            hidden[None, :],
            canonical_residuals,
            len(residuals),
            [0],
            maximum_context=256,
        )
        oracle = np.ascontiguousarray(trace[layer], dtype=np.float32)[None, :]
        return {
            "layer": layer,
            "degree": degree,
            "sharded_vs_canonical": _numerical_metrics(canonical_output, sharded[None, :]),
            "canonical_vs_inherited_oracle": _numerical_metrics(oracle, canonical_output),
            "sharded_vs_inherited_oracle": _numerical_metrics(oracle, sharded[None, :]),
            "canonical_routes": canonical_record["routes"],
        }
    finally:
        canonical.close()
        graph.close()


__all__.append("compare_complete_layer_control")
