"""Frozen closed-loop decode workload helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_022.io import canonical_sha256

from .event_model import DeterministicEventScheduler, TaskGraph
from .freeze import (
    COMMODITY_WORKER_MEMORY_BYTES,
    DECODE_CONCURRENCY_LEVELS,
    sha256_file,
)
from .models import StageAArm
from .service import SERVICE_RELATIVE_PATH, E024ServiceTable
from .task_graph import ExecutionTopology, K3TaskGraphBuilder


def microbatch_sizes(active_sequence_count: int) -> tuple[int, ...]:
    if active_sequence_count not in DECODE_CONCURRENCY_LEVELS:
        raise ValueError("concurrency is outside the frozen E024 ladder")
    full, remainder = divmod(active_sequence_count, 4)
    values = [4] * full
    if remainder:
        values.append(remainder)
    return tuple(values)


def measured_steps_per_batch(active_sequence_count: int) -> int:
    if active_sequence_count not in DECODE_CONCURRENCY_LEVELS:
        raise ValueError("concurrency is outside the frozen E024 ladder")
    return max(4, math.ceil(256 / active_sequence_count))


def measured_output_token_count(active_sequence_count: int) -> int:
    return active_sequence_count * measured_steps_per_batch(active_sequence_count)


def token_latency_quantiles(
    step_latencies_ms: tuple[tuple[float, int], ...],
) -> tuple[float, float]:
    values: list[float] = []
    for latency, batch_size in step_latencies_ms:
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("decode-step latency must be finite and non-negative")
        if batch_size not in (1, 2, 4):
            raise ValueError("invalid E024 microbatch size")
        values.extend([latency] * batch_size)
    if not values:
        raise ValueError("at least one measured decode-step latency is required")
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.quantile(array, 0.50, method="linear")),
        float(np.quantile(array, 0.95, method="linear")),
    )


@dataclass(frozen=True, slots=True)
class DecodeMeasurementPlan:
    active_sequence_count: int
    microbatch_sizes: tuple[int, ...]
    warmup_steps_per_batch: int
    measured_steps_per_batch: int
    minimum_measured_output_tokens: int


class DecodeServingEngine:
    """Define the closed-loop run without target-pass terminology."""

    def measurement_plan(self, active_sequence_count: int) -> DecodeMeasurementPlan:
        sizes = microbatch_sizes(active_sequence_count)
        steps = measured_steps_per_batch(active_sequence_count)
        measured = steps * sum(sizes)
        if measured < 256:
            raise RuntimeError("INSUFFICIENT_TOKEN_SAMPLES")
        return DecodeMeasurementPlan(
            active_sequence_count=active_sequence_count,
            microbatch_sizes=sizes,
            warmup_steps_per_batch=2,
            measured_steps_per_batch=steps,
            minimum_measured_output_tokens=measured,
        )

    def execute(
        self,
        *,
        repo_root: Path,
        topology: ExecutionTopology,
        service: E024ServiceTable,
        active_sequence_count: int,
        arm: StageAArm,
        available_node_budget: int | None,
        placement_sha256: str,
        active_node_ids: tuple[str, ...],
        resident_model_bytes: int,
        peak_transient_bytes: int,
    ) -> dict[str, Any]:
        """Execute the frozen deterministic closed-loop autoregressive workload."""

        plan = self.measurement_plan(active_sequence_count)
        builder = K3TaskGraphBuilder(topology, service)
        graph = TaskGraph()
        warmup_finals: list[int] = []
        for rows in plan.microbatch_sizes:
            dependency: tuple[int, ...] = ()
            for _ in range(plan.warmup_steps_per_batch):
                final = builder.build_decode_step(
                    graph,
                    rows=rows,
                    arrival_dependencies=dependency,
                    measured=False,
                    arm=arm,
                )
                dependency = (final,)
            warmup_finals.append(dependency[0])

        compressed_repetitions = (
            plan.measured_steps_per_batch if len(plan.microbatch_sizes) == 1 else 1
        )
        simulated_steps = (
            1 if len(plan.microbatch_sizes) == 1 else plan.measured_steps_per_batch
        )
        measured_steps: list[tuple[int, tuple[int, ...], int]] = []
        barrier_dependencies = tuple(warmup_finals)
        for rows in plan.microbatch_sizes:
            dependency = barrier_dependencies
            for _ in range(simulated_steps):
                arrival = dependency
                final = builder.build_decode_step(
                    graph,
                    rows=rows,
                    arrival_dependencies=arrival,
                    measured=True,
                    arm=arm,
                )
                measured_steps.append((final, arrival, rows))
                dependency = (final,)

        run = DeterministicEventScheduler().run(graph)
        barrier_ms = max(run.task_finish(value) for value in warmup_finals)
        latencies: list[float] = []
        for final, arrival_dependencies, rows in measured_steps:
            arrival_ms = max(
                (run.task_finish(value) for value in arrival_dependencies),
                default=0.0,
            )
            latency = run.task_finish(final) - arrival_ms
            latencies.extend([latency] * rows * compressed_repetitions)
        latency_array = np.asarray(latencies, dtype=np.float64)
        if compressed_repetitions > 1:
            measured_window_ms = latencies[0] * plan.measured_steps_per_batch
        else:
            measured_window_ms = (
                max(run.task_finish(final) for final, _, _ in measured_steps)
                - barrier_ms
            )
        measured_output_tokens = plan.minimum_measured_output_tokens
        output_tps = measured_output_tokens / (measured_window_ms / 1000.0)
        scale = compressed_repetitions
        compute_ms = run.measured_compute_ms * scale
        network_bytes = run.measured_network_bytes * scale
        network_messages = run.measured_network_messages * scale
        compute_queue = run.measured_compute_queue_wait_ms * scale
        network_queue = run.measured_network_queue_wait_ms * scale
        compute_by_node = {
            key: value * scale for key, value in run.measured_compute_ms_by_node.items()
        }
        network_by_link = {
            key: value * scale for key, value in run.measured_network_ms_by_link.items()
        }
        compute_by_layer = {
            key: value * scale for key, value in run.measured_compute_ms_by_layer.items()
        }
        node_by_id = topology.node_map
        overall_denominator = sum(compute_by_layer.values())
        overall_incapable = (
            sum(
                value
                for layer, value in compute_by_layer.items()
                if layer >= 0
                and service.whole_resident_bytes_by_layer[layer]
                > COMMODITY_WORKER_MEMORY_BYTES
            )
            if topology.commodity_scenario is not None
            else 0.0
        )
        p8_compute = sum(
            value for layer, value in compute_by_layer.items() if 1 <= layer <= 92
        )
        p8_incapable = (
            p8_compute if topology.commodity_scenario is not None else 0.0
        )
        overall_share = (
            overall_incapable / overall_denominator if overall_denominator else 0.0
        )
        p8_share = p8_incapable / p8_compute if p8_compute else 0.0
        active_compute_equivalents = sum(
            node_by_id[node_id].compute_multiplier for node_id in active_node_ids
        )
        compute_utilization = max(
            (value / measured_window_ms for value in compute_by_node.values()),
            default=0.0,
        )
        link_utilization = max(
            (value / measured_window_ms for value in network_by_link.values()),
            default=0.0,
        )
        service_path = repo_root.resolve() / SERVICE_RELATIVE_PATH
        task_graph_descriptor = {
            "architecture": topology.architecture,
            "placement_sha256": placement_sha256,
            "active_sequence_count": active_sequence_count,
            "microbatch_sizes": list(plan.microbatch_sizes),
            "warmup_steps_per_batch": plan.warmup_steps_per_batch,
            "measured_steps_per_batch": plan.measured_steps_per_batch,
            "arm": arm.value,
            "task_count_simulated": len(graph.tasks),
            "compressed_repetitions": compressed_repetitions,
            "service_table_sha256": sha256_file(service_path),
        }
        scenario = (
            topology.commodity_scenario.value
            if topology.commodity_scenario is not None
            else "FAST_FABRIC"
        )
        return {
            "architecture": topology.architecture,
            "scenario": scenario,
            "available_node_budget": available_node_budget,
            "placement_sha256": placement_sha256,
            "concurrency": active_sequence_count,
            "status": "PASS",
            "active_sequence_count": active_sequence_count,
            "microbatch_count": len(plan.microbatch_sizes),
            "microbatch_sizes": "|".join(str(value) for value in plan.microbatch_sizes),
            "warmup_steps_per_batch": plan.warmup_steps_per_batch,
            "measured_decode_steps": plan.measured_steps_per_batch
            * len(plan.microbatch_sizes),
            "measured_output_tokens": measured_output_tokens,
            "measurement_window_ms": measured_window_ms,
            "aggregate_output_tokens_per_second": output_tps,
            "p50_token_latency_ms": float(np.percentile(latency_array, 50)),
            "p95_token_latency_ms": float(np.percentile(latency_array, 95)),
            "active_node_count": len(active_node_ids),
            "active_compute_equivalents": active_compute_equivalents,
            "network_bytes": network_bytes,
            "network_bytes_per_output_token": network_bytes / measured_output_tokens,
            "network_messages": network_messages,
            "worker_compute_ms": compute_ms,
            "compute_queue_wait_ms": compute_queue,
            "network_queue_wait_ms": network_queue,
            "maximum_compute_utilization": compute_utilization,
            "maximum_tx_utilization": link_utilization,
            "maximum_rx_utilization": link_utilization,
            "resident_model_bytes": resident_model_bytes,
            "peak_transient_bytes": peak_transient_bytes,
            "whole_layer_incapable_compute_share": overall_share,
            "overall_whole_layer_incapable_compute_share": overall_share,
            "p8_required_whole_layer_incapable_compute_share": p8_share,
            "p8_required_compute_ms": p8_compute,
            "p8_required_incapable_compute_ms": p8_incapable,
            "task_graph_sha256": canonical_sha256(task_graph_descriptor),
            "service_table_sha256": sha256_file(service_path),
            "task_count_simulated": len(graph.tasks),
            "task_graph_replication_factor": compressed_repetitions,
        }


__all__ = [
    "DecodeMeasurementPlan",
    "DecodeServingEngine",
    "measured_output_token_count",
    "measured_steps_per_batch",
    "microbatch_sizes",
    "token_latency_quantiles",
]
