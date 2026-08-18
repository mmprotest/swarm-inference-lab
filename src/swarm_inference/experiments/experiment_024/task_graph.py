"""Concrete Kimi K3 decode and P8 block task graphs for E024."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .commodity_pool import link_definition
from .event_model import EventTask, TaskGraph
from .freeze import (
    FAST_FABRIC_BANDWIDTH_GBPS,
    FAST_FABRIC_LATENCY_MS,
    FAST_FABRIC_SOFTWARE_OVERHEAD_MS,
    HIDDEN,
    LATENT,
    TOPK,
)
from .models import CommodityScenario, StageAArm
from .placement import CommodityPlacement
from .service import E024ServiceTable

KDA_COMMON_WIDTH = 128
MLA_COMMON_WIDTH = 1536 + 512 + 64
VOCABULARY = 163_840
ROUTE_BYTES_PER_ROW = TOPK * (4 + 4)

# Frozen E022 endpoint primitives retained by the original E024 design.
EMBEDDING_MS_PER_ROW = 0.08
FINAL_NORM_MS_PER_ROW = 0.04
LM_HEAD_SHARD_MS_PER_ROW = 0.55
GREEDY_ARGMAX_MS_PER_ROW = 0.03


class AssignmentLike(Protocol):
    layer: int
    candidate_id: str
    candidate_type: str
    degree: int
    node_ids: tuple[str, ...]
    coordinator_node_id: str


@dataclass(frozen=True, slots=True)
class RuntimeNode:
    node_id: str
    node_index: int
    compute_multiplier: float


@dataclass(frozen=True, slots=True)
class ExecutionTopology:
    architecture: str
    nodes: tuple[RuntimeNode, ...]
    endpoint_node_ids: tuple[str, ...]
    assignments: tuple[AssignmentLike, ...]
    commodity_scenario: CommodityScenario | None
    network_stress_percent: float = 0.0

    @property
    def node_map(self) -> dict[str, RuntimeNode]:
        return {node.node_id: node for node in self.nodes}


def commodity_topology(
    placement: CommodityPlacement,
    *,
    architecture: str,
    network_stress_percent: float = 0.0,
) -> ExecutionTopology:
    if not placement.feasible:
        raise ValueError("cannot execute an infeasible commodity placement")
    return ExecutionTopology(
        architecture=architecture,
        nodes=tuple(
            RuntimeNode(node.node_id, node.node_index, node.compute_multiplier)
            for node in placement.nodes
        ),
        endpoint_node_ids=tuple(placement.endpoint_memory_by_node),
        assignments=placement.assignments,
        commodity_scenario=placement.scenario,
        network_stress_percent=network_stress_percent,
    )


class K3TaskGraphBuilder:
    """Materialize logical work as concrete node and directed-link events."""

    def __init__(self, topology: ExecutionTopology, service: E024ServiceTable) -> None:
        self.topology = topology
        self.service = service
        self.nodes = topology.node_map
        if len(topology.assignments) != 93:
            raise ValueError("K3 execution requires exactly 93 transformer assignments")
        if not topology.endpoint_node_ids:
            raise ValueError("K3 execution requires at least one endpoint owner")

    def _compute(
        self,
        graph: TaskGraph,
        node_id: str,
        dependencies: tuple[int, ...],
        duration_ms: float,
        *,
        measured: bool,
        operation: str,
        layer: int | None,
    ) -> int:
        multiplier = self.nodes[node_id].compute_multiplier
        return graph.add(
            EventTask(
                resource_id=f"compute:{node_id}",
                dependency_ids=tuple(dict.fromkeys(dependencies)),
                duration_ms=duration_ms / multiplier,
                category="compute",
                measured=measured,
                operation=operation,
                node_id=node_id,
                layer_id=layer,
            )
        )

    def _network_duration_ms(
        self,
        source: str,
        destination: str,
        payload_bytes: int,
    ) -> float:
        stress = self.topology.network_stress_percent / 100.0
        if not 0 <= stress < 1:
            raise ValueError("network stress must be in [0, 100)")
        if self.topology.commodity_scenario is None:
            latency = FAST_FABRIC_LATENCY_MS * (1 + stress)
            bandwidth = FAST_FABRIC_BANDWIDTH_GBPS * (1 - stress)
            overhead = FAST_FABRIC_SOFTWARE_OVERHEAD_MS
        else:
            source_node = self.nodes[source]
            destination_node = self.nodes[destination]
            link = link_definition(
                self.topology.commodity_scenario,
                source_node.node_index,
                destination_node.node_index,
            )
            latency = link.latency_ms * (1 + stress)
            bandwidth = link.bandwidth_gbps * (1 - stress)
            overhead = link.software_overhead_ms
        return latency + overhead + payload_bytes * 8 / (bandwidth * 1_000_000)

    def _network(
        self,
        graph: TaskGraph,
        source: str,
        destination: str,
        dependency: int,
        payload_bytes: int,
        *,
        measured: bool,
        operation: str,
        layer: int | None,
    ) -> int:
        if source == destination:
            return dependency
        return graph.add(
            EventTask(
                resource_id=f"link:{source}->{destination}",
                dependency_ids=(dependency,),
                duration_ms=self._network_duration_ms(
                    source,
                    destination,
                    payload_bytes,
                ),
                category="network",
                measured=measured,
                operation=operation,
                source_node_id=source,
                destination_node_id=destination,
                payload_bytes=payload_bytes,
                layer_id=layer,
            )
        )

    def _service_compute(
        self,
        graph: TaskGraph,
        node_id: str,
        dependencies: tuple[int, ...],
        *,
        layer: int,
        operation: str,
        degree: int,
        rows: int,
        measured: bool,
    ) -> int:
        return self._compute(
            graph,
            node_id,
            dependencies,
            self.service.service_ms(layer, operation, degree, rows),
            measured=measured,
            operation=operation,
            layer=layer,
        )

    def _fanout(
        self,
        graph: TaskGraph,
        source: str,
        destinations: tuple[str, ...],
        dependency: int,
        payload_bytes: int,
        *,
        measured: bool,
        operation: str,
        layer: int,
    ) -> dict[str, int]:
        return {
            destination: self._network(
                graph,
                source,
                destination,
                dependency,
                payload_bytes,
                measured=measured,
                operation=operation,
                layer=layer,
            )
            for destination in destinations
        }

    def _gather(
        self,
        graph: TaskGraph,
        sources: tuple[tuple[str, int], ...],
        coordinator: str,
        payload_bytes: int,
        *,
        measured: bool,
        operation: str,
        layer: int,
    ) -> tuple[int, ...]:
        return tuple(
            self._network(
                graph,
                source,
                coordinator,
                dependency,
                payload_bytes,
                measured=measured,
                operation=operation,
                layer=layer,
            )
            for source, dependency in sources
        )

    def _whole_layer(
        self,
        graph: TaskGraph,
        assignment: AssignmentLike,
        dependency: int,
        *,
        rows: int,
        measured: bool,
    ) -> int:
        node = assignment.coordinator_node_id
        protocol = self._service_compute(
            graph,
            node,
            (dependency,),
            layer=assignment.layer,
            operation="worker_protocol",
            degree=1,
            rows=rows,
            measured=measured,
        )
        return self._service_compute(
            graph,
            node,
            (protocol,),
            layer=assignment.layer,
            operation="whole_layer",
            degree=1,
            rows=rows,
            measured=measured,
        )

    def _p8_layer(
        self,
        graph: TaskGraph,
        assignment: AssignmentLike,
        dependency: int,
        *,
        rows: int,
        measured: bool,
        arm: StageAArm,
    ) -> int:
        if assignment.degree != 8 or len(assignment.node_ids) != 8:
            raise ValueError("E024 P8 execution requires eight concrete workers")
        if assignment.candidate_type != "FULL_MIXED_STRIPE":
            raise ValueError("E024 A/D semantics require FULL_MIXED_STRIPE")
        layer = assignment.layer
        workers = assignment.node_ids
        coordinator = assignment.coordinator_node_id
        hidden_bytes = rows * HIDDEN * 4
        latent_bytes = rows * LATENT * 4
        latent_slice_bytes = latent_bytes // 8
        route_bytes = rows * ROUTE_BYTES_PER_ROW
        layer_type = self.service.layer_type_by_id[layer]
        common_width = KDA_COMMON_WIDTH if layer_type == "KDA" else MLA_COMMON_WIDTH

        coordinator_protocol = self._service_compute(
            graph,
            coordinator,
            (dependency,),
            layer=layer,
            operation="worker_protocol",
            degree=1,
            rows=rows,
            measured=measured,
        )
        preprocess = self._service_compute(
            graph,
            coordinator,
            (coordinator_protocol,),
            layer=layer,
            operation="attention_preprocess",
            degree=1,
            rows=rows,
            measured=measured,
        )
        common = self._service_compute(
            graph,
            coordinator,
            (preprocess,),
            layer=layer,
            operation="attention_common",
            degree=1,
            rows=rows,
            measured=measured,
        )
        attention_ready = self._fanout(
            graph,
            coordinator,
            workers,
            common,
            rows * (HIDDEN + common_width) * 4,
            measured=measured,
            operation="attention_invariant_fanout",
            layer=layer,
        )
        worker_protocol: dict[str, int] = {coordinator: common}
        attention_outputs: list[tuple[str, int]] = []
        for worker in workers:
            worker_dependency = attention_ready[worker]
            if worker != coordinator:
                worker_dependency = self._service_compute(
                    graph,
                    worker,
                    (worker_dependency,),
                    layer=layer,
                    operation="worker_protocol",
                    degree=1,
                    rows=rows,
                    measured=measured,
                )
            worker_protocol[worker] = worker_dependency
            attention = self._service_compute(
                graph,
                worker,
                (worker_dependency,),
                layer=layer,
                operation="attention_shard",
                degree=8,
                rows=rows,
                measured=measured,
            )
            attention_outputs.append((worker, attention))
        attention_gather = self._gather(
            graph,
            tuple(attention_outputs),
            coordinator,
            hidden_bytes,
            measured=measured,
            operation="attention_invariant_gather",
            layer=layer,
        )
        attention_reduced = self._service_compute(
            graph,
            coordinator,
            attention_gather,
            layer=layer,
            operation="attention_reduction",
            degree=8,
            rows=rows,
            measured=measured,
        )
        post_attention = self._service_compute(
            graph,
            coordinator,
            (attention_reduced,),
            layer=layer,
            operation="post_attention_preprocess",
            degree=1,
            rows=rows,
            measured=measured,
        )
        router = self._service_compute(
            graph,
            coordinator,
            (post_attention,),
            layer=layer,
            operation="router",
            degree=1,
            rows=rows,
            measured=measured,
        )

        if arm is StageAArm.A_CURRENT:
            routed_ready = self._fanout(
                graph,
                coordinator,
                workers,
                router,
                hidden_bytes + route_bytes,
                measured=measured,
                operation="moe_routed_hidden_metadata_fanout",
                layer=layer,
            )
            retained_hidden_ready: dict[str, int] | None = None
        else:
            retained_hidden_ready = self._fanout(
                graph,
                coordinator,
                workers,
                post_attention,
                hidden_bytes,
                measured=measured,
                operation="moe_retained_hidden_fanout",
                layer=layer,
            )
            route_ready = self._fanout(
                graph,
                coordinator,
                workers,
                router,
                route_bytes,
                measured=measured,
                operation="moe_route_metadata_fanout",
                layer=layer,
            )
            routed_ready = route_ready

        expert_outputs: list[tuple[str, int]] = []
        for worker in workers:
            dependencies = [routed_ready[worker], worker_protocol[worker]]
            if retained_hidden_ready is not None:
                dependencies.append(retained_hidden_ready[worker])
            down = self._service_compute(
                graph,
                worker,
                tuple(dependencies),
                layer=layer,
                operation="latent_down",
                degree=8,
                rows=rows,
                measured=measured,
            )
            expert = self._service_compute(
                graph,
                worker,
                (down,),
                layer=layer,
                operation="expert_stripe",
                degree=8,
                rows=rows,
                measured=measured,
            )
            expert_outputs.append((worker, expert))
        expert_gather = self._gather(
            graph,
            tuple(expert_outputs),
            coordinator,
            latent_bytes,
            measured=measured,
            operation="moe_expert_latent_gather",
            layer=layer,
        )
        expert_reduced = self._service_compute(
            graph,
            coordinator,
            expert_gather,
            layer=layer,
            operation="expert_reduction",
            degree=8,
            rows=rows,
            measured=measured,
        )
        normalized = self._service_compute(
            graph,
            coordinator,
            (expert_reduced,),
            layer=layer,
            operation="routed_norm",
            degree=1,
            rows=rows,
            measured=measured,
        )
        latent_payload = (
            latent_bytes
            if arm in {StageAArm.A_CURRENT, StageAArm.B_RETAIN_HIDDEN}
            else latent_slice_bytes
        )
        latent_ready = self._fanout(
            graph,
            coordinator,
            workers,
            normalized,
            latent_payload,
            measured=measured,
            operation="moe_normalized_latent_fanout",
            layer=layer,
        )
        routed_outputs: list[tuple[str, int]] = []
        for worker in workers:
            up = self._service_compute(
                graph,
                worker,
                (latent_ready[worker],),
                layer=layer,
                operation="latent_up",
                degree=8,
                rows=rows,
                measured=measured,
            )
            routed_outputs.append((worker, up))

        if arm is StageAArm.D_FUSE_OUTPUT:
            assert retained_hidden_ready is not None
            shared_outputs: dict[str, int] = {}
            for worker in workers:
                shared_outputs[worker] = self._service_compute(
                    graph,
                    worker,
                    (retained_hidden_ready[worker],),
                    layer=layer,
                    operation="shared_expert",
                    degree=8,
                    rows=rows,
                    measured=measured,
                )
            routed_by_worker = dict(routed_outputs)
            fused_outputs: list[tuple[str, int]] = []
            for worker in workers:
                fused = self._compute(
                    graph,
                    worker,
                    (routed_by_worker[worker], shared_outputs[worker]),
                    self.service.service_ms(-1, "d_worker_local_fusion", 1, rows),
                    measured=measured,
                    operation="d_worker_local_fusion",
                    layer=layer,
                )
                fused_outputs.append((worker, fused))
            fused_gather = self._gather(
                graph,
                tuple(fused_outputs),
                coordinator,
                hidden_bytes,
                measured=measured,
                operation="moe_combined_hidden_gather",
                layer=layer,
            )
            output = self._service_compute(
                graph,
                coordinator,
                fused_gather,
                layer=layer,
                operation="shared_reduction",
                degree=8,
                rows=rows,
                measured=measured,
            )
        else:
            routed_gather = self._gather(
                graph,
                tuple(routed_outputs),
                coordinator,
                hidden_bytes,
                measured=measured,
                operation="moe_routed_hidden_gather",
                layer=layer,
            )
            routed_reduced = self._service_compute(
                graph,
                coordinator,
                routed_gather,
                layer=layer,
                operation="latent_up_reduction",
                degree=8,
                rows=rows,
                measured=measured,
            )
            if arm is StageAArm.A_CURRENT:
                shared_ready = self._fanout(
                    graph,
                    coordinator,
                    workers,
                    routed_reduced,
                    hidden_bytes,
                    measured=measured,
                    operation="moe_shared_hidden_fanout",
                    layer=layer,
                )
            else:
                assert retained_hidden_ready is not None
                shared_ready = retained_hidden_ready
            shared_outputs = []
            for worker in workers:
                shared = self._service_compute(
                    graph,
                    worker,
                    (shared_ready[worker],),
                    layer=layer,
                    operation="shared_expert",
                    degree=8,
                    rows=rows,
                    measured=measured,
                )
                shared_outputs.append((worker, shared))
            shared_gather = self._gather(
                graph,
                tuple(shared_outputs),
                coordinator,
                hidden_bytes,
                measured=measured,
                operation="moe_shared_hidden_gather",
                layer=layer,
            )
            shared_reduced = self._service_compute(
                graph,
                coordinator,
                shared_gather,
                layer=layer,
                operation="shared_reduction",
                degree=8,
                rows=rows,
                measured=measured,
            )
            output = self._service_compute(
                graph,
                coordinator,
                (routed_reduced, shared_reduced),
                layer=layer,
                operation="routed_shared_reduction",
                degree=2,
                rows=rows,
                measured=measured,
            )
        return self._service_compute(
            graph,
            coordinator,
            (output,),
            layer=layer,
            operation="output_state_commit",
            degree=1,
            rows=rows,
            measured=measured,
        )

    def build_decode_step(
        self,
        graph: TaskGraph,
        *,
        rows: int,
        arrival_dependencies: tuple[int, ...],
        measured: bool,
        arm: StageAArm,
    ) -> int:
        if rows not in (1, 2, 4):
            raise ValueError("decode rows must be 1, 2, or 4")
        endpoint_owner = self.topology.endpoint_node_ids[0]
        embedding = self._compute(
            graph,
            endpoint_owner,
            arrival_dependencies,
            EMBEDDING_MS_PER_ROW * rows,
            measured=measured,
            operation="embedding",
            layer=None,
        )
        dependency = embedding
        source_node = endpoint_owner
        for assignment in self.topology.assignments:
            destination = assignment.coordinator_node_id
            dependency = self._network(
                graph,
                source_node,
                destination,
                dependency,
                rows * HIDDEN * 4,
                measured=measured,
                operation="transformer_activation_boundary",
                layer=assignment.layer,
            )
            if assignment.candidate_type == "WHOLE_LAYER":
                if assignment.degree != 1:
                    raise ValueError("whole-layer execution must use degree 1")
                if (
                    self.topology.commodity_scenario is not None
                    and assignment.layer != 0
                ):
                    raise ValueError("commodity E024 permits whole execution only at layer 0")
                dependency = self._whole_layer(
                    graph,
                    assignment,
                    dependency,
                    rows=rows,
                    measured=measured,
                )
            else:
                dependency = self._p8_layer(
                    graph,
                    assignment,
                    dependency,
                    rows=rows,
                    measured=measured,
                    arm=arm,
                )
            source_node = destination
        dependency = self._network(
            graph,
            source_node,
            endpoint_owner,
            dependency,
            rows * HIDDEN * 4,
            measured=measured,
            operation="final_hidden_boundary",
            layer=92,
        )
        final_norm = self._compute(
            graph,
            endpoint_owner,
            (dependency,),
            FINAL_NORM_MS_PER_ROW * rows,
            measured=measured,
            operation="final_norm",
            layer=None,
        )
        head_outputs: list[int] = []
        shard_count = len(self.topology.endpoint_node_ids)
        for node in self.topology.endpoint_node_ids:
            ready = self._network(
                graph,
                endpoint_owner,
                node,
                final_norm,
                rows * HIDDEN * 4,
                measured=measured,
                operation="lm_head_input",
                layer=None,
            )
            head = self._compute(
                graph,
                node,
                (ready,),
                LM_HEAD_SHARD_MS_PER_ROW * rows,
                measured=measured,
                operation="lm_head_vocabulary_shard",
                layer=None,
            )
            head = self._network(
                graph,
                node,
                endpoint_owner,
                head,
                rows * (VOCABULARY // shard_count) * 4,
                measured=measured,
                operation="logit_shard_gather",
                layer=None,
            )
            head_outputs.append(head)
        return self._compute(
            graph,
            endpoint_owner,
            tuple(head_outputs),
            GREEDY_ARGMAX_MS_PER_ROW * rows,
            measured=measured,
            operation="greedy_argmax_and_state_commit",
            layer=None,
        )

    def build_stage_a_block(
        self,
        graph: TaskGraph,
        *,
        layer: int,
        rows: int,
        arrival_dependencies: tuple[int, ...],
        measured: bool,
        arm: StageAArm,
    ) -> int:
        assignment = self.topology.assignments[layer]
        return self._p8_layer(
            graph,
            assignment,
            arrival_dependencies[0],
            rows=rows,
            measured=measured,
            arm=arm,
        )


__all__ = [
    "ExecutionTopology",
    "K3TaskGraphBuilder",
    "RuntimeNode",
    "commodity_topology",
]
