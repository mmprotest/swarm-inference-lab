"""Stage A communication-mechanism benchmark for E024."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np

from .block_engine import measured_blocks_per_slot
from .event_model import DeterministicEventScheduler, EventTask, TaskGraph
from .freeze import (
    STAGE_A_CONCURRENCY,
    STAGE_A_NETWORK_STRESS_PERCENT,
    STAGE_A_ROWS,
)
from .geometry import GEOMETRY
from .models import CommodityScenario, StageAArm
from .placement import LayerPlacement
from .service import E024ServiceTable
from .task_graph import ExecutionTopology, K3TaskGraphBuilder, RuntimeNode

STAGE_A_LAYERS = (89, 91)


def _topology(
    scenario: CommodityScenario,
    *,
    network_stress_percent: float,
) -> ExecutionTopology:
    nodes = tuple(
        RuntimeNode(
            node_id=f"stage-a-{index:02d}",
            node_index=index,
            compute_multiplier=(1.0, 0.8, 0.6, 0.4)[index % 4],
        )
        for index in range(8)
    )
    node_ids = tuple(node.node_id for node in nodes)
    assignments = tuple(
        LayerPlacement(
            layer=layer,
            candidate_id=f"layer-{layer:02d}:FULL_MIXED_STRIPE:p8",
            candidate_type="FULL_MIXED_STRIPE",
            degree=8,
            node_ids=node_ids,
            resident_memory_bytes=(1,) * 8,
            checkpoint_bytes=(1,) * 8,
            coordinator_node_id=node_ids[0],
        )
        for layer in range(93)
    )
    return ExecutionTopology(
        architecture="STAGE_A_FIXED_P8",
        nodes=nodes,
        endpoint_node_ids=(node_ids[0],),
        assignments=assignments,
        commodity_scenario=scenario,
        network_stress_percent=network_stress_percent,
    )


def _arrival_marker(
    graph: TaskGraph,
    *,
    node_id: str,
    dependencies: tuple[int, ...],
) -> int:
    return graph.add(
        EventTask(
            resource_id=f"arrival:{node_id}",
            dependency_ids=dependencies,
            duration_ms=0.0,
            category="compute",
            measured=False,
            operation="stage_a_arrival_marker",
            node_id=node_id,
        )
    )


def run_stage_a_cell(
    *,
    service: E024ServiceTable,
    scenario: CommodityScenario,
    layer: int,
    rows: int,
    concurrency: int,
    arm: StageAArm,
    network_stress_percent: float = STAGE_A_NETWORK_STRESS_PERCENT,
) -> dict[str, Any]:
    topology = _topology(
        scenario,
        network_stress_percent=network_stress_percent,
    )
    builder = K3TaskGraphBuilder(topology, service)
    graph = TaskGraph()
    warmup_finals: list[int] = []
    for stream in range(concurrency):
        dependency: tuple[int, ...] = ()
        for _ in range(2):
            marker = _arrival_marker(
                graph,
                node_id=f"stage-a-{stream % 8:02d}",
                dependencies=dependency,
            )
            final = builder.build_stage_a_block(
                graph,
                layer=layer,
                rows=rows,
                arrival_dependencies=(marker,),
                measured=False,
                arm=arm,
            )
            dependency = (final,)
        warmup_finals.append(dependency[0])
    steps_per_stream = measured_blocks_per_slot(concurrency)
    compressed_repetitions = steps_per_stream if concurrency == 1 else 1
    simulated_steps = 1 if concurrency == 1 else steps_per_stream
    barrier_dependencies = tuple(warmup_finals)
    measured: list[tuple[int, tuple[int, ...]]] = []
    for stream in range(concurrency):
        dependency = barrier_dependencies
        for _ in range(simulated_steps):
            marker = _arrival_marker(
                graph,
                node_id=f"stage-a-{stream % 8:02d}",
                dependencies=dependency,
            )
            final = builder.build_stage_a_block(
                graph,
                layer=layer,
                rows=rows,
                arrival_dependencies=(marker,),
                measured=True,
                arm=arm,
            )
            measured.append((final, dependency))
            dependency = (final,)
    run = DeterministicEventScheduler().run(graph)
    barrier_ms = max(run.task_finish(value) for value in warmup_finals)
    latencies = [
        run.task_finish(final)
        - max((run.task_finish(value) for value in dependencies), default=0.0)
        for final, dependencies in measured
    ]
    if compressed_repetitions > 1:
        window_ms = latencies[0] * steps_per_stream
    else:
        window_ms = max(run.task_finish(final) for final, _ in measured) - barrier_ms
    measured_blocks = concurrency * steps_per_stream
    scale = compressed_repetitions
    moe_bytes = sum(
        task.payload_bytes
        for task in graph.tasks
        if task.measured
        and task.category == "network"
        and task.operation.startswith("moe_")
    ) * scale
    moe_messages = sum(
        1
        for task in graph.tasks
        if task.measured
        and task.category == "network"
        and task.operation.startswith("moe_")
    ) * scale
    expected_bytes = GEOMETRY[arm].bytes_per_row * rows * measured_blocks
    expected_messages = GEOMETRY[arm].messages_per_row * measured_blocks
    if moe_bytes != expected_bytes or moe_messages != expected_messages:
        raise RuntimeError(
            f"Stage A accounting changed for {arm}: "
            f"bytes {moe_bytes}!={expected_bytes}, messages {moe_messages}!={expected_messages}"
        )
    return {
        "scenario": scenario.value,
        "network_stress_percent": network_stress_percent,
        "layer": layer,
        "layer_type": service.layer_type_by_id[layer],
        "rows": rows,
        "concurrency": concurrency,
        "arm": arm.value,
        "status": "PASS",
        "warmup_blocks_per_stream": 2,
        "measured_blocks_per_stream": steps_per_stream,
        "measured_blocks": measured_blocks,
        "measured_rows": measured_blocks * rows,
        "measurement_window_ms": window_ms,
        "aggregate_blocks_per_second": measured_blocks / (window_ms / 1000.0),
        "aggregate_rows_per_second": measured_blocks * rows / (window_ms / 1000.0),
        "p50_block_latency_ms": float(np.percentile(latencies, 50)),
        "p95_block_latency_ms": float(np.percentile(latencies, 95)),
        "total_network_bytes": run.measured_network_bytes * scale,
        "moe_network_bytes": moe_bytes,
        "moe_network_bytes_per_row": moe_bytes / (measured_blocks * rows),
        "moe_network_messages": moe_messages,
        "moe_network_messages_per_block": moe_messages / measured_blocks,
        "worker_compute_ms": run.measured_compute_ms * scale,
        "compute_queue_wait_ms": run.measured_compute_queue_wait_ms * scale,
        "network_queue_wait_ms": run.measured_network_queue_wait_ms * scale,
        "task_count_simulated": len(graph.tasks),
        "task_graph_replication_factor": scale,
        "evidence_class": "PHYSICALLY_GROUNDED_MODEL_ON_SHAPED_NETWORK",
    }


@dataclass(frozen=True, slots=True)
class StageAResult:
    rows: tuple[dict[str, Any], ...]
    gap_closure_rows: tuple[dict[str, Any], ...]
    median_gap_closure_percent: float
    maximum_gap_closure_percent: float
    d_max_regression_percent: float


def run_stage_a(service: E024ServiceTable) -> StageAResult:
    rows = tuple(
        run_stage_a_cell(
            service=service,
            scenario=scenario,
            layer=layer,
            rows=chunk_rows,
            concurrency=concurrency,
            arm=arm,
        )
        for scenario in CommodityScenario
        for layer in STAGE_A_LAYERS
        for chunk_rows in STAGE_A_ROWS
        for concurrency in STAGE_A_CONCURRENCY
        for arm in StageAArm
    )
    lookup = {
        (
            row["scenario"],
            row["layer"],
            row["rows"],
            row["concurrency"],
            row["arm"],
        ): row
        for row in rows
    }
    gap_rows: list[dict[str, Any]] = []
    for scenario in CommodityScenario:
        for layer in STAGE_A_LAYERS:
            for chunk_rows in STAGE_A_ROWS:
                for concurrency in STAGE_A_CONCURRENCY:
                    key = (scenario.value, layer, chunk_rows, concurrency)
                    current = lookup[(*key, StageAArm.A_CURRENT.value)]
                    fused = lookup[(*key, StageAArm.D_FUSE_OUTPUT.value)]
                    current_latency = float(current["p50_block_latency_ms"])
                    fused_latency = float(fused["p50_block_latency_ms"])
                    closure = (current_latency - fused_latency) / current_latency * 100
                    gap_rows.append(
                        {
                            "scenario": scenario.value,
                            "layer": layer,
                            "rows": chunk_rows,
                            "concurrency": concurrency,
                            "a_p50_block_latency_ms": current_latency,
                            "d_p50_block_latency_ms": fused_latency,
                            "latency_gap_closure_percent": closure,
                            "d_regression_percent": max(0.0, -closure),
                        }
                    )
    closures = [float(row["latency_gap_closure_percent"]) for row in gap_rows]
    regressions = [float(row["d_regression_percent"]) for row in gap_rows]
    return StageAResult(
        rows=rows,
        gap_closure_rows=tuple(gap_rows),
        median_gap_closure_percent=float(statistics.median(closures)),
        maximum_gap_closure_percent=max(closures),
        d_max_regression_percent=max(regressions),
    )


__all__ = [
    "STAGE_A_LAYERS",
    "StageAResult",
    "run_stage_a",
    "run_stage_a_cell",
]
