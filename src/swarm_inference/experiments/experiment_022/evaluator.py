"""Exact worker-task construction and common placement evaluation for E022.

The evaluator contains no planner-level branches.  Planner A and Planner E
both hand a concrete placement to this module and receive an objective from
the same event DAG, network model, cache policy, and wavefront schedule.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .event_model import DeterministicWorkerEventEngine, EventRun, EventTask
from .models import (
    Inventory,
    LayerAssignment,
    ModelGraph,
    PartitionKind,
    PlacementPlan,
)
from .service import ResidentServiceModel

HIDDEN = 7168
LATENT = 3584
TOPK = 16
TARGET_ROWS = 17
ROUTE_METADATA_BYTES_PER_ROW = TOPK * (4 + 4)
KDA_COMMON_WIDTH = 128
MLA_COMMON_WIDTH = 1536 + 512 + 64


@dataclass(frozen=True, slots=True)
class EndpointPolicy:
    """The common, planner-independent non-transformer placement."""

    node_ids: tuple[str, ...]
    memory_by_node: dict[str, int]
    checkpoint_bytes_by_node: dict[str, int]
    embedding_ms: float
    final_norm_ms: float
    lm_head_ms: float


def target_chunks(rows: int) -> tuple[int, ...]:
    if rows not in (1, 2, 4):
        raise ValueError("validated wavefront chunks are 1, 2, or 4 rows")
    result: list[int] = []
    remaining = TARGET_ROWS
    while remaining:
        value = min(rows, remaining)
        # The final remainder is always represented by a physically validated
        # row count: 17 % 2 == 1 and 17 % 4 == 1.
        result.append(value)
        remaining -= value
    return tuple(result)


def common_endpoint_policy(model: ModelGraph, inventory: Inventory) -> EndpointPolicy | None:
    """Derive one endpoint policy from capabilities, independent of action space."""

    available = [
        node
        for node in inventory.available_nodes
        if "PROJECTION_SHARD" in node.runtime_capabilities
    ]
    if not available:
        return None
    # Four vocabulary shards are the canonical local policy.  If fewer nodes
    # exist, use all of them.  Selection is capability-driven, never SKU-driven.
    selected = sorted(
        available,
        key=lambda node: (
            -node.compute_multiplier,
            -node.accelerator_memory_bytes,
            node.node_id,
        ),
    )[: min(4, len(available))]
    count = len(selected)
    memory_parts = _split(model.endpoint_resident_bytes, count)
    checkpoint_parts = _split(model.endpoint_checkpoint_bytes, count)
    if any(
        memory > node.accelerator_memory_bytes
        for node, memory in zip(selected, memory_parts, strict=True)
    ):
        return None
    return EndpointPolicy(
        node_ids=tuple(node.node_id for node in selected),
        memory_by_node={
            node.node_id: memory
            for node, memory in zip(selected, memory_parts, strict=True)
        },
        checkpoint_bytes_by_node={
            node.node_id: value
            for node, value in zip(selected, checkpoint_parts, strict=True)
        },
        # These are replaced by fresh E022 endpoint measurements when the
        # resident harness is materialized.  They are invariant across arms.
        embedding_ms=0.08,
        final_norm_ms=0.04,
        lm_head_ms=0.55,
    )


def _split(total: int, count: int) -> tuple[int, ...]:
    bounds = [(total * index) // count for index in range(count + 1)]
    return tuple(bounds[index + 1] - bounds[index] for index in range(count))


class PlacementEvaluator:
    def __init__(
        self,
        model: ModelGraph,
        inventory: Inventory,
        service: ResidentServiceModel,
        endpoint: EndpointPolicy,
    ) -> None:
        self.model = model
        self.inventory = inventory
        self.service = service
        self.endpoint = endpoint
        self.nodes = inventory.node_map()
        self.engine = DeterministicWorkerEventEngine()

    def _compute(
        self,
        task_id: str,
        node_id: str,
        dependencies: Iterable[str],
        duration_ms: float,
        *,
        layer: int | None,
        chunk: int | None,
        operation: str,
    ) -> EventTask:
        multiplier = self.nodes[node_id].compute_multiplier
        return EventTask(
            task_id=task_id,
            resource_id=f"compute:{node_id}",
            dependency_ids=tuple(dict.fromkeys(dependencies)),
            duration_ms=duration_ms / multiplier,
            category="compute",
            node_id=node_id,
            layer_id=layer,
            chunk_id=chunk,
            operation=operation,
        )

    def _network(
        self,
        task_id: str,
        source: str,
        destination: str,
        dependency: str,
        payload_bytes: int,
        *,
        layer: int,
        chunk: int,
        operation: str,
    ) -> EventTask | None:
        if source == destination:
            return None
        peer = self.nodes[source].peer(destination)
        return EventTask(
            task_id=task_id,
            resource_id=f"link:{source}->{destination}",
            dependency_ids=(dependency,),
            duration_ms=peer.transfer_ms(payload_bytes),
            category="network",
            source_node_id=source,
            destination_node_id=destination,
            payload_bytes=payload_bytes,
            layer_id=layer,
            chunk_id=chunk,
            operation=operation,
        )

    def _service(
        self,
        layer_id: int,
        operation: str,
        degree: int,
        rows: int,
    ) -> float:
        return self.service.service_ms(
            self.model.layers[layer_id], operation, degree, rows
        )

    def _sub_compute(
        self,
        task_id: str,
        node_id: str,
        dependencies: Iterable[str],
        duration_ms: float,
        *,
        layer: int,
        chunk: int,
        operation: str,
        rows: int,
    ) -> tuple[EventTask, ...]:
        """Create native compute inside an already authenticated worker DAG."""

        return (self._compute(
            task_id,
            node_id,
            dependencies,
            duration_ms,
            layer=layer,
            chunk=chunk,
            operation=operation,
        ),)

    def _worker_protocol(
        self,
        task_id: str,
        node_id: str,
        dependencies: Iterable[str],
        *,
        layer: int,
        chunk: int,
        rows: int,
        operation: str,
    ) -> EventTask:
        return EventTask(
            task_id=task_id,
            resource_id=f"compute:{node_id}",
            dependency_ids=tuple(dict.fromkeys(dependencies)),
            duration_ms=self._service(layer, "worker_protocol", 1, rows),
            category="compute",
            node_id=node_id,
            layer_id=layer,
            chunk_id=chunk,
            operation=f"worker_protocol:{operation}",
        )

    def _remote_sub_compute(
        self,
        task_id: str,
        node_id: str,
        coordinator: str,
        dependencies: Iterable[str],
        duration_ms: float,
        *,
        layer: int,
        chunk: int,
        operation: str,
        rows: int,
    ) -> tuple[EventTask, ...]:
        """Dispatch one production primitive outside the coordinator DAG."""

        if node_id == coordinator:
            return self._sub_compute(
                task_id,
                node_id,
                dependencies,
                duration_ms,
                layer=layer,
                chunk=chunk,
                operation=operation,
                rows=rows,
            )
        protocol_id = f"{task_id}.worker-protocol"
        return (
            self._worker_protocol(
                protocol_id,
                node_id,
                dependencies,
                layer=layer,
                chunk=chunk,
                rows=rows,
                operation=operation,
            ),
            self._compute(
                task_id,
                node_id,
                (protocol_id,),
                duration_ms,
                layer=layer,
                chunk=chunk,
                operation=operation,
            ),
        )

    def _fanout(
        self,
        tasks: list[EventTask],
        *,
        prefix: str,
        source: str,
        destinations: Iterable[str],
        dependency: str,
        payload_bytes: int,
        layer: int,
        chunk: int,
        operation: str,
    ) -> dict[str, str]:
        readiness: dict[str, str] = {source: dependency}
        for index, destination in enumerate(dict.fromkeys(destinations)):
            if destination == source:
                readiness[destination] = dependency
                continue
            identifier = f"{prefix}.send-{index:02d}.{source}.{destination}"
            task = self._network(
                identifier,
                source,
                destination,
                dependency,
                payload_bytes,
                layer=layer,
                chunk=chunk,
                operation=operation,
            )
            if task is not None:
                tasks.append(task)
                readiness[destination] = identifier
        return readiness

    def _gather(
        self,
        tasks: list[EventTask],
        *,
        prefix: str,
        sources: Iterable[tuple[str, str]],
        coordinator: str,
        payload_bytes: int,
        layer: int,
        chunk: int,
        operation: str,
    ) -> list[str]:
        ready: list[str] = []
        for index, (source, dependency) in enumerate(sources):
            if source == coordinator:
                ready.append(dependency)
                continue
            identifier = f"{prefix}.recv-{index:02d}.{source}.{coordinator}"
            task = self._network(
                identifier,
                source,
                coordinator,
                dependency,
                payload_bytes,
                layer=layer,
                chunk=chunk,
                operation=operation,
            )
            if task is not None:
                tasks.append(task)
                ready.append(identifier)
        return ready

    def _sub_layer(
        self,
        tasks: list[EventTask],
        assignment: LayerAssignment,
        *,
        chunk_id: int,
        rows: int,
        input_dependency: str,
        state_dependency: str | None,
    ) -> str:
        layer = assignment.layer_id
        prefix = f"c{chunk_id:02d}.l{layer:02d}"
        coordinator = assignment.coordinator_node_id
        workers = assignment.node_ids
        degree = assignment.degree
        hidden_bytes = rows * HIDDEN * 4
        attention_fanout_bytes = rows * (
            HIDDEN
            + (
                KDA_COMMON_WIDTH
                if self.model.layers[layer].layer_type.value == "KDA"
                else MLA_COMMON_WIDTH
            )
        ) * 4
        latent_bytes = rows * LATENT * 4
        route_bytes = rows * ROUTE_METADATA_BYTES_PER_ROW
        initial_dependencies = [input_dependency]
        if state_dependency is not None:
            initial_dependencies.append(state_dependency)
        coordinator_dispatch = f"{prefix}.ordered-dag.worker-protocol"
        tasks.append(
            self._worker_protocol(
                coordinator_dispatch,
                coordinator,
                initial_dependencies,
                layer=layer,
                chunk=chunk_id,
                rows=rows,
                operation="ordered_layer_dag",
            )
        )
        initial_dependencies = [coordinator_dispatch]

        split_attention = assignment.partition_kind in {
            PartitionKind.ATTENTION_PROJECTION_SHARD,
            PartitionKind.FULL_MIXED_STRIPE,
        }
        split_experts = assignment.partition_kind in {
            PartitionKind.WHOLE_EXPERT,
            PartitionKind.EXPERT_SHARD,
            PartitionKind.FULL_MIXED_STRIPE,
        }
        striped_experts = assignment.partition_kind in {
            PartitionKind.EXPERT_SHARD,
            PartitionKind.FULL_MIXED_STRIPE,
        }

        preprocess = f"{prefix}.attention-preprocess"
        tasks.extend(
            self._sub_compute(
                preprocess,
                coordinator,
                initial_dependencies,
                self._service(layer, "attention_preprocess", 1, rows),
                layer=layer,
                chunk=chunk_id,
                operation="attention_preprocess",
                rows=rows,
            )
        )
        if split_attention:
            common_projection = f"{prefix}.attention-common-projection"
            tasks.extend(
                self._sub_compute(
                    common_projection,
                    coordinator,
                    (preprocess,),
                    self._service(layer, "attention_common", 1, rows),
                    layer=layer,
                    chunk=chunk_id,
                    operation="attention_common_projection",
                    rows=rows,
                )
            )
            readiness = self._fanout(
                tasks,
                prefix=f"{prefix}.attention-fanout",
                source=coordinator,
                destinations=workers,
                dependency=common_projection,
                payload_bytes=attention_fanout_bytes,
                layer=layer,
                chunk=chunk_id,
                operation="attention_input_fanout",
            )
            partials: list[tuple[str, str]] = []
            for stripe, node_id in enumerate(workers):
                identifier = f"{prefix}.attention-shard-{stripe:02d}"
                tasks.extend(
                    self._remote_sub_compute(
                        identifier,
                        node_id,
                        coordinator,
                        (readiness[node_id],),
                        self._service(
                            layer,
                            "attention_shard"
                            if node_id == coordinator
                            else "attention_shard_remote",
                            degree,
                            rows,
                        ),
                        layer=layer,
                        chunk=chunk_id,
                        operation="attention_head_projection_shard",
                        rows=rows,
                    )
                )
                partials.append((node_id, identifier))
            gather = self._gather(
                tasks,
                prefix=f"{prefix}.attention-gather",
                sources=partials,
                coordinator=coordinator,
                payload_bytes=hidden_bytes,
                layer=layer,
                chunk=chunk_id,
                operation="attention_partial",
            )
            attention_reduced = f"{prefix}.attention-reduction"
            tasks.extend(
                self._sub_compute(
                    attention_reduced,
                    coordinator,
                    gather,
                    self._service(layer, "attention_reduction", degree, rows),
                    layer=layer,
                    chunk=chunk_id,
                    operation="attention_reduction",
                    rows=rows,
                )
            )
            attention_result = attention_reduced
        else:
            attention_result = f"{prefix}.attention-whole"
            tasks.extend(
                self._sub_compute(
                    attention_result,
                    coordinator,
                    (preprocess,),
                    self._service(layer, "attention_whole", 1, rows),
                    layer=layer,
                    chunk=chunk_id,
                    operation="attention_whole",
                    rows=rows,
                )
            )

        attention_done = f"{prefix}.post-attention-preprocess"
        tasks.extend(
            self._sub_compute(
                attention_done,
                coordinator,
                (attention_result,),
                self._service(layer, "post_attention_preprocess", 1, rows),
                layer=layer,
                chunk=chunk_id,
                operation="post_attention_preprocess",
                rows=rows,
            )
        )

        router_done = f"{prefix}.router"
        tasks.extend(
            self._sub_compute(
                router_done,
                coordinator,
                (attention_done,),
                self._service(layer, "router", 1, rows),
                layer=layer,
                chunk=chunk_id,
                operation="router_top16",
                rows=rows,
            )
        )

        if split_experts:
            if assignment.partition_kind is PartitionKind.FULL_MIXED_STRIPE:
                expert_ready = self._fanout(
                    tasks,
                    prefix=f"{prefix}.projection-fanout",
                    source=coordinator,
                    destinations=workers,
                    dependency=router_done,
                    payload_bytes=hidden_bytes + route_bytes,
                    layer=layer,
                    chunk=chunk_id,
                    operation="expert_hidden_input_fanout",
                )
            else:
                down_whole = f"{prefix}.latent-down-whole"
                tasks.extend(
                    self._sub_compute(
                        down_whole,
                        coordinator,
                        (router_done,),
                        self._service(layer, "latent_down_whole", 1, rows),
                        layer=layer,
                        chunk=chunk_id,
                        operation="latent_down_whole",
                        rows=rows,
                    )
                )
                expert_ready = self._fanout(
                    tasks,
                    prefix=f"{prefix}.expert-latent-fanout",
                    source=coordinator,
                    destinations=workers,
                    dependency=down_whole,
                    payload_bytes=latent_bytes + route_bytes,
                    layer=layer,
                    chunk=chunk_id,
                    operation="expert_latent_input_fanout",
                )
            expert_partials: list[tuple[str, str]] = []
            for stripe, node_id in enumerate(workers):
                dependency = expert_ready[node_id]
                if assignment.partition_kind is PartitionKind.FULL_MIXED_STRIPE:
                    down_id = f"{prefix}.latent-down-{stripe:02d}"
                    tasks.extend(
                        self._remote_sub_compute(
                            down_id,
                            node_id,
                            coordinator,
                            (dependency,),
                            self._service(
                                layer,
                                "latent_down"
                                if node_id == coordinator
                                else "latent_down_remote",
                                degree,
                                rows,
                            ),
                            layer=layer,
                            chunk=chunk_id,
                            operation="latent_down_projection_shard",
                            rows=rows,
                        )
                    )
                    dependency = down_id
                expert_id = f"{prefix}.expert-{stripe:02d}"
                expert_operation = "expert_stripe" if striped_experts else "expert_whole_group"
                tasks.extend(
                    self._remote_sub_compute(
                        expert_id,
                        node_id,
                        coordinator,
                        (dependency,),
                        self._service(
                            layer,
                            expert_operation
                            if node_id == coordinator
                            else f"{expert_operation}_remote",
                            degree,
                            rows,
                        ),
                        layer=layer,
                        chunk=chunk_id,
                        operation=expert_operation,
                        rows=rows,
                    )
                )
                expert_partials.append((node_id, expert_id))
            gather = self._gather(
                tasks,
                prefix=f"{prefix}.expert-gather",
                sources=expert_partials,
                coordinator=coordinator,
                payload_bytes=latent_bytes,
                layer=layer,
                chunk=chunk_id,
                operation="expert_partial",
            )
            expert_done = f"{prefix}.expert-reduction"
            tasks.extend(
                self._sub_compute(
                    expert_done,
                    coordinator,
                    gather,
                    self._service(layer, "expert_reduction", degree, rows),
                    layer=layer,
                    chunk=chunk_id,
                    operation="expert_reduction",
                    rows=rows,
                )
            )
        else:
            down_done = f"{prefix}.latent-down-whole"
            tasks.extend(
                self._sub_compute(
                    down_done,
                    coordinator,
                    (router_done,),
                    self._service(layer, "latent_down_whole", 1, rows),
                    layer=layer,
                    chunk=chunk_id,
                    operation="latent_down_whole",
                    rows=rows,
                )
            )
            expert_done = f"{prefix}.expert-whole"
            tasks.extend(
                self._sub_compute(
                    expert_done,
                    coordinator,
                    (down_done,),
                    self._service(layer, "expert_whole", 1, rows),
                    layer=layer,
                    chunk=chunk_id,
                    operation="expert_whole",
                    rows=rows,
                )
            )

        routed_normalized = f"{prefix}.routed-normalization"
        tasks.extend(
            self._sub_compute(
                routed_normalized,
                coordinator,
                (expert_done,),
                self._service(layer, "routed_norm", 1, rows),
                layer=layer,
                chunk=chunk_id,
                operation="routed_expert_normalization",
                rows=rows,
            )
        )

        if assignment.partition_kind is PartitionKind.FULL_MIXED_STRIPE:
            shared_ready = self._fanout(
                tasks,
                prefix=f"{prefix}.shared-fanout",
                source=coordinator,
                destinations=workers,
                dependency=attention_done,
                payload_bytes=hidden_bytes,
                layer=layer,
                chunk=chunk_id,
                operation="shared_expert_input_fanout",
            )
            shared_partials: list[tuple[str, str]] = []
            for stripe, node_id in enumerate(workers):
                shared_id = f"{prefix}.shared-{stripe:02d}"
                tasks.extend(
                    self._remote_sub_compute(
                        shared_id,
                        node_id,
                        coordinator,
                        (shared_ready[node_id],),
                        self._service(
                            layer,
                            "shared_expert"
                            if node_id == coordinator
                            else "shared_expert_remote",
                            degree,
                            rows,
                        ),
                        layer=layer,
                        chunk=chunk_id,
                        operation="shared_expert_shard",
                        rows=rows,
                    )
                )
                shared_partials.append((node_id, shared_id))
            shared_gather = self._gather(
                tasks,
                prefix=f"{prefix}.shared-gather",
                sources=shared_partials,
                coordinator=coordinator,
                payload_bytes=hidden_bytes,
                layer=layer,
                chunk=chunk_id,
                operation="shared_expert_partial",
            )
            shared_done = f"{prefix}.shared-reduction"
            tasks.extend(
                self._sub_compute(
                    shared_done,
                    coordinator,
                    shared_gather,
                    self._service(layer, "shared_reduction", degree, rows),
                    layer=layer,
                    chunk=chunk_id,
                    operation="shared_expert_reduction",
                    rows=rows,
                )
            )
        else:
            shared_done = f"{prefix}.shared-whole"
            tasks.extend(
                self._sub_compute(
                    shared_done,
                    coordinator,
                    (attention_done,),
                    self._service(layer, "shared_expert_whole", 1, rows),
                    layer=layer,
                    chunk=chunk_id,
                    operation="shared_expert_whole",
                    rows=rows,
                )
            )

        if assignment.partition_kind is PartitionKind.FULL_MIXED_STRIPE:
            up_ready = self._fanout(
                tasks,
                prefix=f"{prefix}.latent-up-fanout",
                source=coordinator,
                destinations=workers,
                dependency=routed_normalized,
                payload_bytes=latent_bytes,
                layer=layer,
                chunk=chunk_id,
                operation="latent_up_input_fanout",
            )
            up_partials: list[tuple[str, str]] = []
            for stripe, node_id in enumerate(workers):
                up_id = f"{prefix}.latent-up-{stripe:02d}"
                tasks.extend(
                    self._remote_sub_compute(
                        up_id,
                        node_id,
                        coordinator,
                        (up_ready[node_id],),
                        self._service(
                            layer,
                            "latent_up"
                            if node_id == coordinator
                            else "latent_up_remote",
                            degree,
                            rows,
                        ),
                        layer=layer,
                        chunk=chunk_id,
                        operation="latent_up_projection_shard",
                        rows=rows,
                    )
                )
                up_partials.append((node_id, up_id))
            up_gather = self._gather(
                tasks,
                prefix=f"{prefix}.latent-up-gather",
                sources=up_partials,
                coordinator=coordinator,
                payload_bytes=hidden_bytes,
                layer=layer,
                chunk=chunk_id,
                operation="latent_up_partial",
            )
            routed_up_done = f"{prefix}.latent-up-reduction"
            tasks.extend(
                self._sub_compute(
                    routed_up_done,
                    coordinator,
                    up_gather,
                    self._service(layer, "latent_up_reduction", degree, rows),
                    layer=layer,
                    chunk=chunk_id,
                    operation="latent_up_reduction",
                    rows=rows,
                )
            )
        else:
            routed_up_done = f"{prefix}.latent-up-whole"
            tasks.extend(
                self._sub_compute(
                    routed_up_done,
                    coordinator,
                    (routed_normalized,),
                    self._service(layer, "latent_up_whole", 1, rows),
                    layer=layer,
                    chunk=chunk_id,
                    operation="latent_up_whole",
                    rows=rows,
                )
            )
        up_done = f"{prefix}.exact-output-sum"
        tasks.extend(
            self._sub_compute(
                up_done,
                coordinator,
                (routed_up_done, shared_done),
                self._service(layer, "routed_shared_reduction", 2, rows),
                layer=layer,
                chunk=chunk_id,
                operation="routed_shared_exact_sum",
                rows=rows,
            )
        )
        finish = f"{prefix}.output-state-commit"
        tasks.extend(
            self._sub_compute(
                finish,
                coordinator,
                (up_done,),
                self._service(layer, "output_state_commit", 1, rows),
                layer=layer,
                chunk=chunk_id,
                operation="output_state_commit",
                rows=rows,
            )
        )
        return finish

    def build_tasks(self, plan: PlacementPlan) -> list[EventTask]:
        if len(plan.assignments) != len(self.model.layers):
            raise ValueError("feasible plan must assign all 93 transformer layers")
        tasks: list[EventTask] = []
        endpoint_owner = self.endpoint.node_ids[0]
        chunks = target_chunks(plan.chunk_rows)
        embeddings: dict[int, str] = {}
        for chunk_id, rows in enumerate(chunks):
            embedding = f"c{chunk_id:02d}.endpoint.embedding"
            tasks.append(
                self._compute(
                    embedding,
                    endpoint_owner,
                    (),
                    self.endpoint.embedding_ms * rows,
                    layer=None,
                    chunk=chunk_id,
                    operation="embedding",
                )
            )
            embeddings[chunk_id] = embedding
        layer_state: dict[int, str] = {}
        chunk_outputs: dict[tuple[int, int], str] = {}
        for chunk_id, rows in enumerate(chunks):
            for assignment in plan.assignments:
                layer = assignment.layer_id
                if layer == 0:
                    dependency = embeddings[chunk_id]
                    source_node = endpoint_owner
                else:
                    dependency = chunk_outputs[(chunk_id, layer - 1)]
                    source_node = plan.assignments[layer - 1].coordinator_node_id
                destination_node = assignment.coordinator_node_id
                if source_node != destination_node:
                    transfer_id = (
                        f"c{chunk_id:02d}.l{layer:02d}.activation."
                        f"{source_node}.{destination_node}"
                    )
                    transfer = self._network(
                        transfer_id,
                        source_node,
                        destination_node,
                        dependency,
                        rows * HIDDEN * 4,
                        layer=layer,
                        chunk=chunk_id,
                        operation="coarse_activation_boundary",
                    )
                    if transfer is not None:
                        tasks.append(transfer)
                        dependency = transfer_id
                state_dependency = layer_state.get(layer)
                if assignment.partition_kind is PartitionKind.WHOLE_LAYER:
                    identifier = f"c{chunk_id:02d}.l{layer:02d}.whole"
                    dependencies = [dependency]
                    if state_dependency is not None:
                        dependencies.append(state_dependency)
                    protocol_id = f"{identifier}.worker-protocol"
                    tasks.append(
                        self._worker_protocol(
                            protocol_id,
                            assignment.coordinator_node_id,
                            dependencies,
                            layer=layer,
                            chunk=chunk_id,
                            rows=rows,
                            operation="whole_layer",
                        )
                    )
                    tasks.extend(
                        self._sub_compute(
                            identifier,
                            assignment.coordinator_node_id,
                            (protocol_id,),
                            self._service(layer, "whole_layer", 1, rows),
                            layer=layer,
                            chunk=chunk_id,
                            operation="whole_layer",
                            rows=rows,
                        )
                    )
                    output = identifier
                else:
                    output = self._sub_layer(
                        tasks,
                        assignment,
                        chunk_id=chunk_id,
                        rows=rows,
                        input_dependency=dependency,
                        state_dependency=state_dependency,
                    )
                chunk_outputs[(chunk_id, layer)] = output
                layer_state[layer] = output
        final_dependencies: list[str] = []
        final_source = plan.assignments[-1].coordinator_node_id
        for chunk_id, rows in enumerate(chunks):
            dependency = chunk_outputs[(chunk_id, len(self.model.layers) - 1)]
            if final_source != endpoint_owner:
                transfer_id = (
                    f"c{chunk_id:02d}.endpoint-final."
                    f"{final_source}.{endpoint_owner}"
                )
                transfer = self._network(
                    transfer_id,
                    final_source,
                    endpoint_owner,
                    dependency,
                    rows * HIDDEN * 4,
                    layer=92,
                    chunk=chunk_id,
                    operation="final_hidden_boundary",
                )
                if transfer is not None:
                    tasks.append(transfer)
                    dependency = transfer_id
            final_dependencies.append(dependency)
        final_norm = "endpoint.final-norm"
        tasks.append(
            self._compute(
                final_norm,
                endpoint_owner,
                final_dependencies,
                self.endpoint.final_norm_ms * TARGET_ROWS,
                layer=None,
                chunk=None,
                operation="final_norm",
            )
        )
        # Each endpoint vocabulary owner computes exactly one resident shard.
        head_ids: list[str] = []
        for index, node_id in enumerate(self.endpoint.node_ids):
            dependency = final_norm
            if node_id != endpoint_owner:
                transfer_id = f"endpoint.hidden.{endpoint_owner}.{node_id}"
                task = self._network(
                    transfer_id,
                    endpoint_owner,
                    node_id,
                    final_norm,
                    TARGET_ROWS * HIDDEN * 4,
                    layer=92,
                    chunk=len(chunks) - 1,
                    operation="lm_head_input",
                )
                if task is not None:
                    tasks.append(task)
                    dependency = transfer_id
            identifier = f"endpoint.lm-head-{index:02d}"
            tasks.append(
                self._compute(
                    identifier,
                    node_id,
                    (dependency,),
                    self.endpoint.lm_head_ms * TARGET_ROWS,
                    layer=None,
                    chunk=None,
                    operation="lm_head_vocabulary_shard",
                )
            )
            if node_id != endpoint_owner:
                logits_transfer = f"endpoint.logits.{node_id}.{endpoint_owner}"
                task = self._network(
                    logits_transfer,
                    node_id,
                    endpoint_owner,
                    identifier,
                    TARGET_ROWS * (163840 // len(self.endpoint.node_ids)) * 4,
                    layer=92,
                    chunk=len(chunks) - 1,
                    operation="logit_shard_gather",
                )
                if task is not None:
                    tasks.append(task)
                    identifier = logits_transfer
            head_ids.append(identifier)
        logits_owner = f"endpoint.logits-owner.{endpoint_owner}"
        tasks.append(
            self._compute(
                logits_owner,
                endpoint_owner,
                tuple(head_ids),
                0.03 * TARGET_ROWS,
                layer=None,
                chunk=None,
                operation="greedy_argmax",
            )
        )
        return tasks

    def evaluate(self, plan: PlacementPlan, *, include_records: bool = False) -> EventRun:
        run = self.engine.run(self.build_tasks(plan))
        plan.exact_tok_s_per_user = TARGET_ROWS / (run.makespan_ms / 1000.0)
        plan.critical_path_ms = run.makespan_ms
        plan.total_worker_compute_ms = run.total_compute_ms
        plan.network_bytes = run.total_network_bytes
        plan.serial_waits = run.serial_waits
        plan.messages = run.messages
        plan.worker_utilization = run.worker_utilization
        plan.worker_seconds_per_token = run.total_compute_ms / 1000.0 / TARGET_ROWS
        plan.used_nodes = tuple(sorted(plan.memory_used_by_node))
        plan.objective_tuple = (
            plan.exact_tok_s_per_user,
            -run.makespan_ms,
            -run.total_compute_ms,
            -float(run.total_network_bytes),
            -float(len(plan.used_nodes)),
        )
        plan.event_receipt = run.as_dict(include_records=include_records)
        return run


__all__ = [
    "TARGET_ROWS",
    "EndpointPolicy",
    "PlacementEvaluator",
    "common_endpoint_policy",
    "target_chunks",
]
