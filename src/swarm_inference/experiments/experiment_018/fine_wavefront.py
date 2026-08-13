"""Fine-grain expert-microshard expansion of the Experiment 018 wavefront."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from swarm_inference.experiments.experiment_018.wavefront import (
    MICROCELL_COUNT,
    DeterministicEventEngine,
    EventRun,
    EventTask,
    MicrocellServiceProfile,
    TaskKind,
    VerificationBlockSpec,
    WavefrontModel,
    WavefrontNetworkAccounting,
    partition_rows,
)

LATENT_SIZE = 3584
TOPK = 16


@dataclass(frozen=True, slots=True)
class FineWavefrontResult:
    block_candidates: int
    accepted_rows: int
    chunk_size: int
    split_degree: int
    cache_enabled: bool
    total_ms: float
    oracle_tok_s_per_user: float
    speedup_vs_coarse_wavefront: float
    logical_task_count: int
    physical_event_count: int
    logical_to_physical_ratio: float
    physical_kernel_count: int
    message_count: int
    payload_bytes: int
    critical_path_waits: int
    critical_path_communication_ms: float
    scheduler_overhead_ms: float
    simulation_cpu_ms: float
    useful_parallelism: float
    critical_path_fraction: float
    peak_concurrency: int
    stage_utilization: Mapping[int, float]
    median_stage_utilization: float
    event_run: EventRun
    network: WavefrontNetworkAccounting

    def summary(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("event_run")
        return value


class FineWavefrontModel:
    """Replace measured whole-expert phases with explicit exact shard DAGs.

    Non-expert work stays calibrated to the physically measured coarse cell.
    Every expert/shard leaf is an independent future resource.  Fanout and each
    stable reduction level are explicit shaped-network events.  The model never
    interprets sequential execution on the development RTX 5090 as overlap.
    """

    def __init__(
        self,
        profiles: Sequence[MicrocellServiceProfile],
        *,
        expert_component_ms_by_cell_rows: Mapping[int, Mapping[int, float]],
        shard_service_ms_by_rows: Mapping[int, float],
        routes_by_layer_position: Mapping[int, Sequence[Sequence[int]]],
        route_weights_by_layer_position: Mapping[
            int, Sequence[Sequence[float]]
        ],
        split_degree: int,
        coarse_model: WavefrontModel | None = None,
    ) -> None:
        if split_degree not in (8, 16, 32):
            raise ValueError("fine wavefront split degree must be 8/16/32")
        if set(shard_service_ms_by_rows) != {1, 2, 4, 8}:
            raise ValueError("fine service requires measured rows 1/2/4/8")
        if set(expert_component_ms_by_cell_rows) != set(range(MICROCELL_COUNT)):
            raise ValueError("fine service requires an expert component for every cell")
        missing_layers = sorted(set(range(1, 93)) - set(routes_by_layer_position))
        if missing_layers:
            raise ValueError(f"fine route fixture is missing layers {missing_layers[:8]}")
        for layer, rows in routes_by_layer_position.items():
            if not rows or any(len(row) != TOPK or len(set(row)) != TOPK for row in rows):
                raise ValueError(f"layer {layer} route fixture is not exact top-16")
        if set(route_weights_by_layer_position) != set(routes_by_layer_position):
            raise ValueError("fine route weights do not cover the exact route fixture")
        for layer, rows in route_weights_by_layer_position.items():
            if len(rows) != len(routes_by_layer_position[layer]) or any(
                len(row) != TOPK or any(not math.isfinite(value) for value in row)
                for row in rows
            ):
                raise ValueError(f"layer {layer} route weights are not finite top-16")
        self.profiles = tuple(sorted(profiles, key=lambda item: item.microcell_id))
        self.expert_components = expert_component_ms_by_cell_rows
        self.shard_service = shard_service_ms_by_rows
        self.routes = routes_by_layer_position
        self.route_weights = route_weights_by_layer_position
        self.split_degree = split_degree
        self.coarse = coarse_model or WavefrontModel(self.profiles)
        self.engine = DeterministicEventEngine()

    @staticmethod
    def _pre_id(cell: int, chunk: int, layer: int) -> str:
        return f"fine-pre:c{cell:02d}:q{chunk:03d}:l{layer:02d}"

    @staticmethod
    def _post_id(cell: int, chunk: int, layer: int) -> str:
        return f"fine-post:c{cell:02d}:q{chunk:03d}:l{layer:02d}"

    @staticmethod
    def _fanout_id(cell: int, chunk: int, layer: int) -> str:
        return f"fine-fanout:c{cell:02d}:q{chunk:03d}:l{layer:02d}"

    @staticmethod
    def _leaf_id(cell: int, chunk: int, layer: int, expert: int, shard: int) -> str:
        return (
            f"fine-expert:c{cell:02d}:q{chunk:03d}:l{layer:02d}:"
            f"e{expert:03d}:s{shard:02d}"
        )

    @staticmethod
    def _reduce_id(cell: int, chunk: int, layer: int, depth: int) -> str:
        return f"fine-reduce:c{cell:02d}:q{chunk:03d}:l{layer:02d}:d{depth:02d}"

    def _route_rows(self, layer: int, positions: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        fixture = self.routes[layer]
        return tuple(tuple(fixture[position % len(fixture)]) for position in positions)

    def _route_weight_rows(
        self, layer: int, positions: Sequence[int]
    ) -> tuple[tuple[float, ...], ...]:
        fixture = self.route_weights[layer]
        return tuple(tuple(fixture[position % len(fixture)]) for position in positions)

    def _shard_duration(self, assignment_count: int) -> float:
        return sum(
            float(self.shard_service[rows])
            for rows in partition_rows(assignment_count, 8)
        )

    def build_tasks(
        self,
        block: VerificationBlockSpec,
        *,
        maximum_chunk_rows: int,
        cache_enabled: bool,
    ) -> tuple[
        tuple[EventTask, ...],
        WavefrontNetworkAccounting,
        dict[str, int],
    ]:
        coarse_tasks, chunks, accounting = self.coarse.build_tasks(
            block,
            maximum_chunk_rows=maximum_chunk_rows,
            cache_enabled=cache_enabled,
        )
        coarse_compute = {
            task.task_id: task for task in coarse_tasks if task.kind == TaskKind.COMPUTE
        }
        tasks = [task for task in coarse_tasks if task.kind != TaskKind.COMPUTE]
        logical_pair_reductions = 0
        reduction_level_events = 0
        physical_kernels = 0
        micro_messages = 0
        micro_bytes = 0

        for chunk in chunks:
            for profile in self.profiles:
                cell = profile.microcell_id
                coarse_id = self.coarse._compute_id(cell, chunk.chunk_id)
                original = coarse_compute[coarse_id]
                expert_component = float(
                    self.expert_components[cell][chunk.row_count]
                )
                compute_component = float(original.metadata["compute_ms"])
                if not 0.0 <= expert_component <= compute_component:
                    raise ValueError(
                        f"cell {cell} expert component exceeds measured compute service"
                    )
                layers = tuple(range(profile.layer_start, profile.layer_end))
                nonexpert = original.duration_ms - expert_component
                per_layer_nonexpert = nonexpert / len(layers)
                previous: tuple[str, ...] = original.dependencies

                for layer in layers:
                    pre_id = self._pre_id(cell, chunk.chunk_id, layer)
                    tasks.append(
                        EventTask(
                            task_id=pre_id,
                            kind=TaskKind.COMPUTE,
                            resource_id=f"microcell:{cell:02d}:nonexpert",
                            duration_ms=per_layer_nonexpert / 2.0,
                            dependencies=previous,
                            priority=30,
                            metadata={
                                "cell": cell,
                                "chunk": chunk.chunk_id,
                                "layer": layer,
                                "phase": "nonexpert_pre",
                                "rows": chunk.row_count,
                                "compute_ms": per_layer_nonexpert / 2.0,
                            },
                        )
                    )
                    physical_kernels += 1
                    if layer == 0:
                        reduction_dependency = (pre_id,)
                    else:
                        route_rows = self._route_rows(layer, chunk.positions)
                        route_weight_rows = self._route_weight_rows(
                            layer, chunk.positions
                        )
                        counts = Counter(expert for row in route_rows for expert in row)
                        expert_weights: dict[int, list[tuple[int, int, float]]] = {
                            expert: [] for expert in counts
                        }
                        for row_index, (expert_row, weight_row) in enumerate(
                            zip(route_rows, route_weight_rows, strict=True)
                        ):
                            for slot, (expert, weight) in enumerate(
                                zip(expert_row, weight_row, strict=True)
                            ):
                                expert_weights[expert].append(
                                    (row_index, slot, float(weight))
                                )
                        input_payload = (
                            sum(counts.values())
                            * self.split_degree
                            * LATENT_SIZE
                            * 4
                        )
                        fanout_id = self._fanout_id(cell, chunk.chunk_id, layer)
                        tasks.append(
                            EventTask(
                                task_id=fanout_id,
                                kind=TaskKind.HANDOFF,
                                resource_id=f"microcell:{cell:02d}:expert-fanout",
                                duration_ms=self.coarse.internal_network.service_ms(
                                    chunk.row_count * LATENT_SIZE * 4
                                ),
                                dependencies=(pre_id,),
                                priority=25,
                                metadata={
                                    "cell": cell,
                                    "chunk": chunk.chunk_id,
                                    "layer": layer,
                                    "phase": "microshard_fanout",
                                    "logical_messages": len(counts) * self.split_degree,
                                    "payload_bytes": input_payload,
                                    "independent_links": True,
                                },
                            )
                        )
                        leaves: list[str] = []
                        for expert, count in sorted(counts.items()):
                            for shard in range(self.split_degree):
                                leaf_id = self._leaf_id(
                                    cell,
                                    chunk.chunk_id,
                                    layer,
                                    expert,
                                    shard,
                                )
                                leaves.append(leaf_id)
                                tasks.append(
                                    EventTask(
                                        task_id=leaf_id,
                                        kind=TaskKind.EXPERT,
                                        resource_id=(
                                            f"microshard:l{layer:02d}:e{expert:03d}:"
                                            f"s{shard:02d}"
                                        ),
                                        duration_ms=self._shard_duration(count),
                                        dependencies=(fanout_id,),
                                        priority=35,
                                        metadata={
                                            "cell": cell,
                                            "chunk": chunk.chunk_id,
                                            "layer": layer,
                                            "expert": expert,
                                            "shard": shard,
                                            "assignment_count": count,
                                            "route_weight_applied_before_reduction": True,
                                            "route_weight_assignments": expert_weights[
                                                expert
                                            ],
                                            "rows": chunk.row_count,
                                            "compute_ms": self._shard_duration(count),
                                        },
                                    )
                                )
                        contribution_count = TOPK * self.split_degree
                        depth_count = math.ceil(math.log2(contribution_count))
                        dependency = tuple(leaves)
                        nodes = chunk.row_count * math.ceil(contribution_count / 2)
                        for depth in range(depth_count):
                            reduction_id = self._reduce_id(
                                cell, chunk.chunk_id, layer, depth
                            )
                            payload = chunk.row_count * LATENT_SIZE * 4
                            tasks.append(
                                EventTask(
                                    task_id=reduction_id,
                                    kind=TaskKind.REDUCTION,
                                    resource_id=(
                                        f"microcell:{cell:02d}:expert-reduction:"
                                        f"depth:{depth:02d}"
                                    ),
                                    duration_ms=self.coarse.internal_network.service_ms(
                                        payload
                                    ),
                                    dependencies=dependency,
                                    priority=40,
                                    metadata={
                                        "cell": cell,
                                        "chunk": chunk.chunk_id,
                                        "layer": layer,
                                        "depth": depth,
                                        "logical_pair_reductions": nodes,
                                        "logical_messages": nodes,
                                        "payload_bytes": nodes * LATENT_SIZE * 4,
                                        "physical_batched_levels": 1,
                                        "stable_order": "expert_id_then_shard_id",
                                    },
                                )
                            )
                            micro_messages += nodes
                            micro_bytes += nodes * LATENT_SIZE * 4
                            logical_pair_reductions += nodes
                            reduction_level_events += 1
                            nodes = chunk.row_count * math.ceil(nodes / (2 * chunk.row_count))
                            dependency = (reduction_id,)
                        reduction_dependency = dependency
                        micro_messages += len(counts) * self.split_degree
                        micro_bytes += input_payload
                        physical_kernels += len(leaves) + depth_count

                    post_id = self._post_id(cell, chunk.chunk_id, layer)
                    tasks.append(
                        EventTask(
                            task_id=post_id,
                            kind=TaskKind.COMPUTE,
                            resource_id=f"microcell:{cell:02d}:nonexpert",
                            duration_ms=per_layer_nonexpert / 2.0,
                            dependencies=reduction_dependency,
                            priority=45,
                            metadata={
                                "cell": cell,
                                "chunk": chunk.chunk_id,
                                "layer": layer,
                                "phase": "nonexpert_post",
                                "rows": chunk.row_count,
                                "compute_ms": per_layer_nonexpert / 2.0,
                            },
                        )
                    )
                    previous = (post_id,)
                    physical_kernels += 1
                tasks.append(
                    EventTask(
                        task_id=coarse_id,
                        kind=TaskKind.CONTROL,
                        resource_id=f"microcell:{cell:02d}:completion",
                        duration_ms=0.0,
                        dependencies=previous,
                        priority=50,
                        metadata={
                            "cell": cell,
                            "chunk": chunk.chunk_id,
                            "rows": chunk.row_count,
                            "phase": "cell_complete",
                        },
                    )
                )
        control = {
            "logical_task_count": (
                len(tasks) + logical_pair_reductions - reduction_level_events
            ),
            "physical_event_count": len(tasks),
            "physical_kernel_count": physical_kernels,
            "microshard_message_count": micro_messages,
            "microshard_payload_bytes": micro_bytes,
        }
        return tuple(tasks), accounting, control

    def run(
        self,
        *,
        block_candidates: int,
        maximum_chunk_rows: int,
        cache_enabled: bool,
        request_id: str = "experiment-018-fine",
    ) -> FineWavefrontResult:
        accepted = block_candidates + 1
        block = VerificationBlockSpec(
            block_id=f"fine-block-{block_candidates}",
            request_id=request_id,
            positions=tuple(range(accepted)),
        )
        tasks, accounting, control = self.build_tasks(
            block,
            maximum_chunk_rows=maximum_chunk_rows,
            cache_enabled=cache_enabled,
        )
        run = self.engine.run(tasks)
        coarse = self.coarse.run(
            block_candidates=block_candidates,
            maximum_chunk_rows=maximum_chunk_rows,
            cache_enabled=cache_enabled,
            request_id=f"{request_id}-coarse-comparison",
        )
        records = run.record_map()
        stage_start: dict[int, float] = {}
        stage_finish: dict[int, float] = {}
        for record in run.records:
            cell_value = record.metadata.get("cell")
            if cell_value is None:
                continue
            cell = int(cell_value)
            stage_start[cell] = min(stage_start.get(cell, record.start_ms), record.start_ms)
            stage_finish[cell] = max(
                stage_finish.get(cell, record.finish_ms), record.finish_ms
            )
        stage_utilization = {
            cell: (stage_finish[cell] - stage_start[cell]) / run.makespan_ms
            for cell in range(MICROCELL_COUNT)
        }
        critical_communication = sum(
            records[task_id].duration_ms
            for task_id in run.critical_path
            if records[task_id].kind
            in {TaskKind.HANDOFF.value, TaskKind.CACHE_SEED.value, TaskKind.REDUCTION.value}
        )
        critical_compute = sum(
            records[task_id].duration_ms
            for task_id in run.critical_path
            if records[task_id].kind
            in {TaskKind.COMPUTE.value, TaskKind.EXPERT.value}
        )
        total_compute = sum(
            record.duration_ms
            for record in run.records
            if record.kind in {TaskKind.COMPUTE.value, TaskKind.EXPERT.value}
        )
        coarse_messages = sum(
            task.kind in {TaskKind.HANDOFF, TaskKind.CACHE_SEED}
            and task.metadata.get("phase") != "microshard_fanout"
            for task in tasks
        )
        coarse_bytes = sum(
            int(task.metadata.get("payload_bytes", 0))
            for task in tasks
            if task.kind in {TaskKind.HANDOFF, TaskKind.CACHE_SEED}
            and task.metadata.get("phase") != "microshard_fanout"
        )
        return FineWavefrontResult(
            block_candidates=block_candidates,
            accepted_rows=accepted,
            chunk_size=maximum_chunk_rows,
            split_degree=self.split_degree,
            cache_enabled=cache_enabled,
            total_ms=run.makespan_ms,
            oracle_tok_s_per_user=accepted * 1000.0 / run.makespan_ms,
            speedup_vs_coarse_wavefront=coarse.total_ms / run.makespan_ms,
            logical_task_count=int(control["logical_task_count"]),
            physical_event_count=len(tasks),
            logical_to_physical_ratio=int(control["logical_task_count"])
            / max(1, len(tasks)),
            physical_kernel_count=int(control["physical_kernel_count"]),
            message_count=coarse_messages + int(control["microshard_message_count"]),
            payload_bytes=coarse_bytes + int(control["microshard_payload_bytes"]),
            critical_path_waits=len(run.critical_path),
            critical_path_communication_ms=critical_communication,
            scheduler_overhead_ms=self.coarse.control_setup_ms,
            simulation_cpu_ms=run.scheduler_cpu_ms,
            useful_parallelism=total_compute / max(critical_compute, 1e-12),
            critical_path_fraction=run.makespan_ms / max(run.serial_sum_ms, 1e-12),
            peak_concurrency=run.peak_concurrency,
            stage_utilization=stage_utilization,
            median_stage_utilization=statistics.median(stage_utilization.values()),
            event_run=run,
            network=accounting,
        )


__all__ = ["LATENT_SIZE", "TOPK", "FineWavefrontModel", "FineWavefrontResult"]
