"""Canonical composition of complete execution plans from engine fragments."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict, deque

from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    ComponentDataEdge,
    ComponentPlanFragment,
    CompositeExecutionPlan,
    ExecutionComponent,
    ExecutionComponentProvider,
    ExecutionComponentType,
    ExecutionPlan,
    ExecutionRequest,
    PhasePlan,
)
from swarm_inference.model.descriptor import ResolvedModelDescriptor

_ALWAYS_REQUIRED = frozenset(
    {
        ExecutionComponentType.TOKENIZATION,
        ExecutionComponentType.EMBEDDING,
        ExecutionComponentType.ATTENTION,
        ExecutionComponentType.KV_CACHE,
        ExecutionComponentType.NORMALIZATION,
        ExecutionComponentType.LM_HEAD,
        ExecutionComponentType.SAMPLING,
        ExecutionComponentType.TOKEN_PUBLICATION,
    }
)


def _complete(component_types: set[ExecutionComponentType]) -> bool:
    return (
        _ALWAYS_REQUIRED.issubset(component_types)
        and bool(
            component_types
            & {ExecutionComponentType.DENSE_MLP, ExecutionComponentType.ROUTED_EXPERTS}
        )
        and (
            ExecutionComponentType.ROUTED_EXPERTS not in component_types
            or ExecutionComponentType.ROUTER in component_types
        )
    )


def _topological_path(components: tuple[ExecutionComponent, ...]) -> tuple[str, ...]:
    by_id = {item.component_id: item for item in components}
    incoming = {key: set(item.depends_on) for key, item in by_id.items()}
    followers: dict[str, set[str]] = defaultdict(set)
    for target, dependencies in incoming.items():
        for source in dependencies:
            followers[source].add(target)
    ready = deque(sorted(key for key, dependencies in incoming.items() if not dependencies))
    ordered: list[str] = []
    while ready:
        source = ready.popleft()
        ordered.append(source)
        for target in sorted(followers[source]):
            incoming[target].discard(source)
            if not incoming[target]:
                ready.append(target)
    if len(ordered) != len(components):
        raise ValueError("execution component graph contains a dependency cycle")
    return tuple(ordered)


def _validate_colocation(components: tuple[ExecutionComponent, ...]) -> None:
    """Reject plans whose fragment placement cannot be honored by execution."""

    groups: dict[str, list[ExecutionComponent]] = defaultdict(list)
    for component in components:
        group = component.placement.colocation_group
        if group is not None:
            groups[group].append(component)
    for group, members in groups.items():
        common_workers = set(members[0].placement.worker_ids)
        for member in members[1:]:
            common_workers.intersection_update(member.placement.worker_ids)
        if not common_workers:
            raise ValueError(
                f"component colocation group {group!r} has no common logical worker"
            )
        if any(member.placement.require_same_device for member in members):
            devices = {member.placement.device for member in members}
            if len(devices) != 1:
                raise ValueError(
                    f"component colocation group {group!r} spans multiple devices"
                )


def _cross_fragment_edges(
    components: tuple[ExecutionComponent, ...],
    existing: tuple[ComponentDataEdge, ...],
) -> tuple[ComponentDataEdge, ...]:
    """Connect every declared dependency using a semantically exact boundary."""

    by_id = {item.component_id: item for item in components}
    if len(by_id) != len(components):
        raise ValueError("execution component IDs must be unique")
    for target in components:
        missing = sorted(set(target.depends_on) - set(by_id))
        if missing:
            raise ValueError(
                f"component {target.component_id!r} depends on missing components {missing}"
            )
    edges = list(existing)
    connected: set[tuple[str, str]] = set()
    for edge in existing:
        try:
            source = by_id[edge.source_component_id]
            target = by_id[edge.target_component_id]
        except KeyError as exc:
            raise ValueError("component edge references an unknown component") from exc
        if source.component_id not in target.depends_on:
            raise ValueError(
                f"component edge {source.component_id!r} -> {target.component_id!r} "
                "does not correspond to a declared dependency"
            )
        producers = [
            item
            for item in source.output_contracts
            if item.boundary_id == edge.source_boundary_id
        ]
        consumers = [
            item
            for item in target.input_contracts
            if item.boundary_id == edge.target_boundary_id
        ]
        if len(producers) != 1 or len(consumers) != 1:
            raise ValueError("component edge boundary IDs must resolve exactly once")
        producer, consumer = producers[0], consumers[0]
        mismatches = producer.semantic_mismatches(
            consumer,
            allow_device_transfer=edge.device_transfer,
        )
        if mismatches:
            raise ValueError(
                f"component edge {source.component_id!r} -> {target.component_id!r} "
                f"has semantic boundary mismatches: {', '.join(mismatches)}"
            )
        requires_transfer = producer.device != consumer.device
        if edge.device_transfer != requires_transfer:
            raise ValueError("component edge device-transfer declaration is not exact")
        shares_worker = bool(
            set(source.placement.worker_ids).intersection(target.placement.worker_ids)
        )
        if edge.transport == "in-process" and not shares_worker:
            raise ValueError("in-process component edge has no shared logical worker")
        if edge.transport == "direct-worker" and not (
            source.placement.direct_data_path and target.placement.direct_data_path
        ):
            raise ValueError("direct-worker edge requires direct-data-path placements")
        if edge.transport == "coordinator-control" and target.component_type not in {
            ExecutionComponentType.TOKEN_PUBLICATION
        }:
            raise ValueError("model activations cannot relay through coordinator control")
        dependency = (target.component_id, source.component_id)
        if dependency in connected:
            raise ValueError("component dependency has duplicate data edges")
        connected.add(dependency)
    for target in components:
        for dependency in target.depends_on:
            if (target.component_id, dependency) in connected:
                continue
            source = by_id[dependency]
            matches = []
            for producer in source.output_contracts:
                for consumer in target.input_contracts:
                    mismatch = producer.semantic_mismatches(
                        consumer,
                        allow_device_transfer=True,
                    )
                    if not mismatch:
                        matches.append((producer, consumer))
            if len(matches) != 1:
                raise ValueError(
                    f"dependency {dependency!r} -> {target.component_id!r} requires exactly "
                    f"one compatible boundary, found {len(matches)}"
                )
            producer, consumer = matches[0]
            same_worker = bool(
                set(source.placement.worker_ids).intersection(target.placement.worker_ids)
            )
            edges.append(
                ComponentDataEdge(
                    source_component_id=source.component_id,
                    source_boundary_id=producer.boundary_id,
                    target_component_id=target.component_id,
                    target_boundary_id=consumer.boundary_id,
                    transport="in-process" if same_worker else "direct-worker",
                    device_transfer=producer.device != consumer.device,
                    bounded_asynchronous=True,
                    estimated_bytes=max(
                        source.estimated_network_bytes,
                        target.estimated_network_bytes,
                    ),
                )
            )
    return tuple(edges)


def _stable_identity(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _combine(
    fragments: tuple[ComponentPlanFragment, ...],
    request: ExecutionRequest,
) -> CompositeExecutionPlan:
    components = tuple(item for fragment in fragments for item in fragment.components)
    _validate_colocation(components)
    edges = _cross_fragment_edges(
        components,
        tuple(item for fragment in fragments for item in fragment.edges),
    )
    path = _topological_path(components)
    publication = next(
        item
        for item in components
        if item.component_type == ExecutionComponentType.TOKEN_PUBLICATION
    )
    controller = next(
        fragment.controller_engine_id
        for fragment in fragments
        if publication in fragment.components
    )
    worker_roles: dict[str, str] = {}
    for fragment in fragments:
        worker_roles.update(fragment.worker_roles)
    rates = [item.predicted_decode_tokens_s for item in fragments]
    aggregate_rates = [item.predicted_aggregate_tokens_s for item in fragments]
    decode_rate = 1.0 / sum(1.0 / max(item, 1e-12) for item in rates)
    aggregate_rate = 1.0 / sum(1.0 / max(item, 1e-12) for item in aggregate_rates)
    messages = sum(item.predicted_messages_per_token or 0.0 for item in fragments)
    model_edges = [item for item in edges if item.transport == "direct-worker"]
    messages += 2.0 * len(model_edges)
    bytes_per_token = sum(item.predicted_bytes_per_token or 0.0 for item in fragments)
    bytes_per_token += sum(item.estimated_bytes for item in model_edges)
    serial_waits = sum(item.predicted_serial_waits_per_token or 0.0 for item in fragments)
    serial_waits += len(model_edges)
    optional: dict[str, bool] = {}
    for fragment in fragments:
        for mechanism, enabled in fragment.optional_mechanisms.items():
            optional[mechanism] = optional.get(mechanism, False) or enabled
    identity_payload = {
        "fragments": [item.execution_identity for item in fragments],
        "components": [item.model_dump(mode="json") for item in components],
        "edges": [item.model_dump(mode="json") for item in edges],
    }
    execution_identity = _stable_identity(identity_payload)
    plan_id = "composite-" + execution_identity.removeprefix("sha256:")[:20]
    parameters = {
        "composite": True,
        "component_fragments": {item.fragment_id: item.engine_parameters for item in fragments},
        "component_engines": sorted({item.engine_id for item in components}),
        "direct_model_data_edges": len(model_edges),
        "coordinator_activation_bytes_per_token": 0,
    }
    return CompositeExecutionPlan(
        plan_id=plan_id,
        engine_id=controller,
        model_fingerprint=fragments[0].model_fingerprint,
        execution_identity=execution_identity,
        objective=request.objective,
        topology="composite-direct-component-graph",
        worker_roles=worker_roles,
        components=components,
        component_edges=edges,
        critical_path=path,
        optional_mechanisms=optional,
        engine_parameters=parameters,
        prefill_plan=PhasePlan(phase="prefill", worker_roles=worker_roles),
        decode_plan=PhasePlan(phase="decode", worker_roles=worker_roles),
        predicted_ttft_ms=sum(item.predicted_ttft_ms for item in fragments),
        predicted_decode_tokens_s=decode_rate,
        predicted_aggregate_tokens_s=aggregate_rate,
        predicted_network_bytes=int(bytes_per_token * request.max_new_tokens),
        predicted_messages_per_token=messages,
        predicted_bytes_per_token=bytes_per_token,
        predicted_serial_waits_per_token=serial_waits,
        number_of_wan_stage_boundaries=0,
        persistent_connections=all(component.placement.persistent for component in components),
        network_cost_confidence="estimated",
        network_cost_provenance="component contracts plus direct-data edges",
        required_memory_bytes=sum(item.required_memory_bytes for item in fragments),
        score=min(item.score for item in fragments),
        explanation=(
            "complete plan composed from independently probed engine components",
            "all activation edges are direct worker-to-worker or in-process",
            "coordinator owns control and token publication, not hidden-state relay",
            *(reason for item in fragments for reason in item.explanation),
        ),
    )


async def compose_candidate_plans(
    providers: tuple[ExecutionComponentProvider, ...],
    model: ResolvedModelDescriptor,
    cluster: ClusterCapabilities,
    request: ExecutionRequest,
) -> tuple[ExecutionPlan, ...]:
    """Compose all minimal complete fragment sets without family dispatch."""

    fragments: list[ComponentPlanFragment] = []
    for provider in providers:
        fragments.extend(await provider.candidate_components(model, cluster, request))
    plans: list[ExecutionPlan] = []
    for size in range(2, len(fragments) + 1):
        plans_at_size: list[ExecutionPlan] = []
        for selection in itertools.combinations(fragments, size):
            if len({item.model_fingerprint for item in selection}) != 1:
                continue
            components = tuple(item for fragment in selection for item in fragment.components)
            types = [item.component_type for item in components]
            if len(types) != len(set(types)) or not _complete(set(types)):
                continue
            if request.requested_engine is not None and request.requested_engine not in {
                item.engine_id for item in components
            }:
                continue
            try:
                plan = _combine(selection, request)
            except ValueError:
                continue
            plans_at_size.append(plan)
        if plans_at_size:
            # Enumerate every valid composition at the first complete fragment
            # cardinality so engine competition can compare measured options.
            # Larger sets would duplicate component work because component
            # types are unique within a selectable plan.
            plans.extend(plans_at_size)
            break
    return tuple(sorted(plans, key=lambda item: item.plan_id))


__all__ = ["compose_candidate_plans"]
