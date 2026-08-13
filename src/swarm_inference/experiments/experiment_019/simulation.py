"""Worker-level wavefront model assembled from measured shard primitives."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from swarm_inference.experiments.experiment_019.events import (
    CommunicationEdge,
    DeterministicMicroworkerEngine,
    EventRun,
    MicroworkerTask,
    NetworkProfile,
    NetworkTask,
    assert_headline_trace,
    tree_collective,
)
from swarm_inference.experiments.experiment_019.placement import (
    KDA_LAYERS,
    PlacementSpec,
)

HIDDEN = 7168
LATENT = 3584
LAYERS = 93

NETWORK_PROFILES: dict[str, NetworkProfile] = {
    "very_fast_local": NetworkProfile("very_fast_local", 0.05, 100.0, 0.0),
    "canonical_fast_local": NetworkProfile("canonical_fast_local", 0.25, 25.0, 0.0),
    "commodity_fast_lan": NetworkProfile("commodity_fast_lan", 1.0, 10.0, 0.0),
    "regional": NetworkProfile("regional", 5.0, 1.0, 0.0),
    "residential_wan": NetworkProfile("residential_wan", 20.0, 0.1, 0.0),
    "canonical_inter_pod": NetworkProfile("canonical_inter_pod", 5.0, 10.0, 0.0),
}


@dataclass(frozen=True, slots=True)
class MeasuredShardService:
    degree: int
    rows: int
    kda_common_ms: float
    kda_worker_ms: tuple[float, ...]
    mla_common_ms: float
    mla_worker_ms: tuple[float, ...]
    moe_pre_ms: float
    latent_down_worker_ms: tuple[float, ...]
    expert_worker_ms: tuple[float, ...]
    routed_norm_ms: float
    latent_up_shared_worker_ms: tuple[float, ...]
    dense_worker_ms: tuple[float, ...]
    layer_finalize_ms: float
    endpoint_worker_ms: tuple[float, ...]
    embedding_ms: float
    hidden_pair_reduce_ms: float
    latent_pair_reduce_ms: float
    source: str = "RTX 5090 physically measured shard service"

    def __post_init__(self) -> None:
        for values in (
            self.kda_worker_ms,
            self.mla_worker_ms,
            self.latent_down_worker_ms,
            self.expert_worker_ms,
            self.latent_up_shared_worker_ms,
            self.dense_worker_ms,
            self.endpoint_worker_ms,
        ):
            if len(values) != self.degree or any(value < 0 for value in values):
                raise ValueError("service vectors must name every stripe worker")


@dataclass(frozen=True, slots=True)
class SimulationConfiguration:
    placement: PlacementSpec
    block: int
    chunk: int
    local_profile: NetworkProfile
    inter_pod_profile: NetworkProfile
    compute_slowdown: float = 1.0
    heterogeneity: str = "homogeneous"
    jitter_fraction: float = 0.0
    activation_strategy: str = "hybrid"
    seed: int = 19019


def _worker(pod: int, stripe: int) -> str:
    return f"pod-{pod:03d}.worker-{stripe:02d}"


def _task(
    task_id: str,
    worker: str,
    *,
    chunk: int,
    layer: int,
    operator: str,
    dependencies: Sequence[str],
    service_ms: float,
) -> MicroworkerTask:
    pod = worker.split(".", maxsplit=1)[0]
    return MicroworkerTask(
        task_id=task_id,
        worker_id=worker,
        pod_id=pod,
        request_id="zero-draft-target-oracle",
        block_id="verification",
        chunk_id=chunk,
        layer_id=layer,
        operator=operator,
        shard_id=worker.rsplit("-", maxsplit=1)[-1],
        dependency_ids=tuple(dependencies),
        input_refs=(f"hidden.chunk-{chunk}.layer-{layer}",),
        state_refs=(f"state.{worker}.layer-{layer}",),
        service_time_ms=service_ms,
    )


def _multipliers(configuration: SimulationConfiguration) -> dict[str, float]:
    spec = configuration.placement
    workers = [_worker(pod, stripe) for pod in range(spec.pod_count) for stripe in range(spec.stripe_degree)]
    values = {worker: configuration.compute_slowdown for worker in workers}
    rng = random.Random(configuration.seed)
    shuffled = workers.copy()
    rng.shuffle(shuffled)
    if configuration.heterogeneity == "ten_percent_1p5x":
        for worker in shuffled[: math.ceil(len(workers) * 0.10)]:
            values[worker] *= 1.5
    elif configuration.heterogeneity == "ten_percent_2x":
        for worker in shuffled[: math.ceil(len(workers) * 0.10)]:
            values[worker] *= 2.0
    elif configuration.heterogeneity == "twentyfive_percent_1p5x":
        for worker in shuffled[: math.ceil(len(workers) * 0.25)]:
            values[worker] *= 1.5
    elif configuration.heterogeneity == "random_plus_minus_20":
        for worker in workers:
            values[worker] *= rng.uniform(0.8, 1.2)
    elif configuration.heterogeneity != "homogeneous":
        raise ValueError(f"unknown heterogeneity scenario {configuration.heterogeneity}")
    return values


def _jittered(profile: NetworkProfile, fraction: float, rng: random.Random) -> NetworkProfile:
    if fraction <= 0:
        return profile
    factor = rng.uniform(1.0 - fraction, 1.0 + fraction)
    return replace(profile, rtt_ms=profile.rtt_ms * factor)


def _recursive_doubling_allgather(
    *,
    task_prefix: str,
    participants: tuple[str, ...],
    total_payload_bytes: int,
    dependency_ids: tuple[str, ...],
    profile: NetworkProfile,
    chunk_id: int,
    layer_id: int,
) -> tuple[list[NetworkTask], str]:
    count = len(participants)
    if count & (count - 1):
        raise ValueError("recursive doubling requires a power-of-two participant count")
    tasks: list[NetworkTask] = []
    dependencies = dependency_ids
    shard_bytes = math.ceil(total_payload_bytes / count)
    for step in range(int(math.log2(count))):
        edges = tuple(
            CommunicationEdge(
                participants[index],
                participants[index ^ (1 << step)],
                min(total_payload_bytes, shard_bytes * (1 << step)),
            )
            for index in range(count)
        )
        identifier = f"{task_prefix}.step-{step:02d}"
        tasks.append(
            NetworkTask(
                task_id=identifier,
                request_id="zero-draft-target-oracle",
                block_id="verification",
                chunk_id=chunk_id,
                layer_id=layer_id,
                operator="latent_recursive_doubling_allgather",
                dependency_ids=dependencies,
                edges=edges,
                profile=profile,
                collective_algorithm="recursive_doubling_allgather",
                collective_participants=participants,
                collective_steps=1,
            )
        )
        dependencies = (identifier,)
    return tasks, dependencies[0]


def build_worker_dag(
    service: MeasuredShardService,
    configuration: SimulationConfiguration,
) -> list[MicroworkerTask | NetworkTask]:
    spec = configuration.placement
    if service.degree != spec.stripe_degree or service.rows != configuration.chunk:
        raise ValueError("physical service and placement geometry differ")
    multipliers = _multipliers(configuration)
    rng = random.Random(configuration.seed + 1)
    tasks: list[MicroworkerTask | NetworkTask] = []
    chunk_rows = [
        min(configuration.chunk, configuration.block + 1 - start)
        for start in range(0, configuration.block + 1, configuration.chunk)
    ]
    layer_complete: dict[tuple[int, int], str] = {}
    attention_state: dict[tuple[int, int], str] = {}
    for chunk_id, actual_rows in enumerate(chunk_rows):
        first_worker = _worker(0, 0)
        embed_id = f"c{chunk_id:03d}.embedding"
        tasks.append(
            _task(
                embed_id,
                first_worker,
                chunk=chunk_id,
                layer=-1,
                operator="embedding_vocabulary_shard",
                dependencies=(),
                service_ms=service.embedding_ms * multipliers[first_worker],
            )
        )
        for layer in range(LAYERS):
            pod = layer // spec.depth_span
            participants = tuple(_worker(pod, stripe) for stripe in range(spec.stripe_degree))
            root = participants[0]
            if layer == 0:
                input_ready = embed_id
            else:
                input_ready = layer_complete[(chunk_id, layer - 1)]
                previous_pod = (layer - 1) // spec.depth_span
                if previous_pod != pod:
                    handoff_id = f"c{chunk_id:03d}.l{layer:03d}.interpod"
                    tasks.append(
                        NetworkTask(
                            task_id=handoff_id,
                            request_id="zero-draft-target-oracle",
                            block_id="verification",
                            chunk_id=chunk_id,
                            layer_id=layer,
                            operator="explicit_inter_pod_hidden_handoff",
                            dependency_ids=(input_ready,),
                            edges=(
                                CommunicationEdge(
                                    _worker(previous_pod, 0),
                                    root,
                                    actual_rows * HIDDEN * 4,
                                ),
                            ),
                            profile=_jittered(
                                configuration.inter_pod_profile,
                                configuration.jitter_fraction,
                                rng,
                            ),
                        )
                    )
                    input_ready = handoff_id
            attention_type = "KDA" if layer in KDA_LAYERS else "Gated_MLA"
            common_ms = (
                service.kda_common_ms if attention_type == "KDA" else service.mla_common_ms
            )
            common_id = f"c{chunk_id:03d}.l{layer:03d}.attention-common"
            tasks.append(
                _task(
                    common_id,
                    root,
                    chunk=chunk_id,
                    layer=layer,
                    operator="attnres_norm_and_low_projection",
                    dependencies=(input_ready,),
                    service_ms=common_ms * multipliers[root],
                )
            )
            fanout_id = f"c{chunk_id:03d}.l{layer:03d}.attention-fanout"
            if configuration.activation_strategy == "hidden_sharded":
                allgather_tasks, fanout_id = _recursive_doubling_allgather(
                    task_prefix=fanout_id,
                    participants=participants,
                    total_payload_bytes=actual_rows * HIDDEN * 4,
                    dependency_ids=(common_id,),
                    profile=_jittered(
                        configuration.local_profile,
                        configuration.jitter_fraction,
                        rng,
                    ),
                    chunk_id=chunk_id,
                    layer_id=layer,
                )
                tasks.extend(allgather_tasks)
            else:
                if configuration.activation_strategy not in {
                    "replicated_small",
                    "hybrid",
                }:
                    raise ValueError("unknown activation strategy")
                tasks.append(NetworkTask(
                    task_id=fanout_id,
                    request_id="zero-draft-target-oracle",
                    block_id="verification",
                    chunk_id=chunk_id,
                    layer_id=layer,
                    operator="replicated_activation_fanout",
                    dependency_ids=(common_id,),
                    edges=tuple(
                        CommunicationEdge(root, worker, actual_rows * HIDDEN * 4)
                        for worker in participants[1:]
                    ),
                    profile=_jittered(
                        configuration.local_profile,
                        configuration.jitter_fraction,
                        rng,
                    ),
                ))
            attention_tasks: list[str] = []
            attention_values = (
                service.kda_worker_ms
                if attention_type == "KDA"
                else service.mla_worker_ms
            )
            for stripe, worker in enumerate(participants):
                identifier = f"c{chunk_id:03d}.l{layer:03d}.attention.p{stripe:02d}"
                dependencies = [fanout_id]
                prior_state = attention_state.get((layer, stripe))
                if prior_state is not None:
                    dependencies.append(prior_state)
                tasks.append(
                    _task(
                        identifier,
                        worker,
                        chunk=chunk_id,
                        layer=layer,
                        operator=f"{attention_type}_head_projection_stripe",
                        dependencies=dependencies,
                        service_ms=attention_values[stripe] * multipliers[worker],
                    )
                )
                attention_state[(layer, stripe)] = identifier
                attention_tasks.append(identifier)
            attention_reduce = f"c{chunk_id:03d}.l{layer:03d}.attention-reduce"
            tasks.append(
                tree_collective(
                    task_id=attention_reduce,
                    participants=participants,
                    payload_bytes=actual_rows * HIDDEN * 4,
                    dependency_ids=tuple(attention_tasks),
                    profile=_jittered(
                        configuration.local_profile,
                        configuration.jitter_fraction,
                        rng,
                    ),
                    request_id="zero-draft-target-oracle",
                    block_id="verification",
                    chunk_id=chunk_id,
                    layer_id=layer,
                    operator="attention_partial_allreduce",
                )
            )
            attention_finalize: list[str] = []
            for stripe, worker in enumerate(participants):
                identifier = (
                    f"c{chunk_id:03d}.l{layer:03d}.attention-reduce-compute.p{stripe:02d}"
                )
                tasks.append(
                    _task(
                        identifier,
                        worker,
                        chunk=chunk_id,
                        layer=layer,
                        operator="attention_allreduce_pairwise_accumulation",
                        dependencies=(attention_reduce,),
                        service_ms=service.hidden_pair_reduce_ms
                        * math.ceil(math.log2(spec.stripe_degree))
                        * multipliers[worker],
                    )
                )
                attention_finalize.append(identifier)
            moe_common_id = f"c{chunk_id:03d}.l{layer:03d}.moe-common"
            tasks.append(
                _task(
                    moe_common_id,
                    root,
                    chunk=chunk_id,
                    layer=layer,
                    operator="post_attention_attnres_norm_router",
                    dependencies=(attention_finalize[0],),
                    service_ms=service.moe_pre_ms * multipliers[root],
                )
            )
            moe_fanout = f"c{chunk_id:03d}.l{layer:03d}.moe-fanout"
            if configuration.activation_strategy == "hidden_sharded":
                route_broadcast = moe_fanout + ".routes"
                tasks.append(NetworkTask(
                    task_id=route_broadcast,
                    request_id="zero-draft-target-oracle",
                    block_id="verification",
                    chunk_id=chunk_id,
                    layer_id=layer,
                    operator="route_metadata_broadcast",
                    dependency_ids=(moe_common_id,),
                    edges=tuple(
                        CommunicationEdge(root, worker, actual_rows * 16 * 8)
                        for worker in participants[1:]
                    ),
                    profile=_jittered(
                        configuration.local_profile,
                        configuration.jitter_fraction,
                        rng,
                    ),
                ))
                allgather_tasks, moe_fanout = _recursive_doubling_allgather(
                    task_prefix=moe_fanout + ".hidden",
                    participants=participants,
                    total_payload_bytes=actual_rows * HIDDEN * 4,
                    dependency_ids=(route_broadcast,),
                    profile=_jittered(
                        configuration.local_profile,
                        configuration.jitter_fraction,
                        rng,
                    ),
                    chunk_id=chunk_id,
                    layer_id=layer,
                )
                tasks.extend(allgather_tasks)
            else:
                tasks.append(NetworkTask(
                    task_id=moe_fanout,
                    request_id="zero-draft-target-oracle",
                    block_id="verification",
                    chunk_id=chunk_id,
                    layer_id=layer,
                    operator="latent_route_and_hidden_fanout",
                    dependency_ids=(moe_common_id,),
                    edges=tuple(
                        CommunicationEdge(
                            root,
                            worker,
                            actual_rows * (HIDDEN * 4 + 16 * 8),
                        )
                        for worker in participants[1:]
                    ),
                    profile=_jittered(
                        configuration.local_profile,
                        configuration.jitter_fraction,
                        rng,
                    ),
                ))
            moe_tasks: list[str] = []
            if layer == 0:
                for stripe, worker in enumerate(participants):
                    identifier = f"c{chunk_id:03d}.l{layer:03d}.dense.p{stripe:02d}"
                    tasks.append(
                        _task(
                            identifier,
                            worker,
                            chunk=chunk_id,
                            layer=layer,
                            operator="dense_mlp_intermediate_stripe",
                            dependencies=(moe_fanout,),
                            service_ms=service.dense_worker_ms[stripe]
                            * multipliers[worker],
                        )
                    )
                    moe_tasks.append(identifier)
            else:
                latent_tasks: list[str] = []
                for stripe, worker in enumerate(participants):
                    identifier = f"c{chunk_id:03d}.l{layer:03d}.latent-down.p{stripe:02d}"
                    tasks.append(
                        _task(
                            identifier,
                            worker,
                            chunk=chunk_id,
                            layer=layer,
                            operator="latent_down_row_projection_stripe",
                            dependencies=(moe_fanout,),
                            service_ms=service.latent_down_worker_ms[stripe]
                            * multipliers[worker],
                        )
                    )
                    latent_tasks.append(identifier)
                allgather_tasks, latent_ready = _recursive_doubling_allgather(
                    task_prefix=f"c{chunk_id:03d}.l{layer:03d}.latent-allgather",
                    participants=participants,
                    total_payload_bytes=actual_rows * LATENT * 4,
                    dependency_ids=tuple(latent_tasks),
                    profile=_jittered(
                        configuration.local_profile,
                        configuration.jitter_fraction,
                        rng,
                    ),
                    chunk_id=chunk_id,
                    layer_id=layer,
                )
                tasks.extend(allgather_tasks)
                expert_tasks: list[str] = []
                for stripe, worker in enumerate(participants):
                    identifier = f"c{chunk_id:03d}.l{layer:03d}.expert.p{stripe:02d}"
                    tasks.append(
                        _task(
                            identifier,
                            worker,
                            chunk=chunk_id,
                            layer=layer,
                            operator="expert_stripe_local_top16_accumulation",
                            dependencies=(latent_ready,),
                            service_ms=service.expert_worker_ms[stripe]
                            * multipliers[worker],
                        )
                    )
                    expert_tasks.append(identifier)
                expert_reduce = f"c{chunk_id:03d}.l{layer:03d}.expert-reduce"
                tasks.append(
                    tree_collective(
                        task_id=expert_reduce,
                        participants=participants,
                        payload_bytes=actual_rows * LATENT * 4,
                        dependency_ids=tuple(expert_tasks),
                        profile=_jittered(
                            configuration.local_profile,
                            configuration.jitter_fraction,
                            rng,
                        ),
                        request_id="zero-draft-target-oracle",
                        block_id="verification",
                        chunk_id=chunk_id,
                        layer_id=layer,
                        operator="one_expert_stripe_partial_allreduce",
                    )
                )
                expert_finalize: list[str] = []
                for stripe, worker in enumerate(participants):
                    identifier = f"c{chunk_id:03d}.l{layer:03d}.expert-reduce-compute.p{stripe:02d}"
                    tasks.append(
                        _task(
                            identifier,
                            worker,
                            chunk=chunk_id,
                            layer=layer,
                            operator="expert_allreduce_pairwise_accumulation",
                            dependencies=(expert_reduce,),
                            service_ms=service.latent_pair_reduce_ms
                            * math.ceil(math.log2(spec.stripe_degree))
                            * multipliers[worker],
                        )
                    )
                    expert_finalize.append(identifier)
                routed_norm_tasks: list[str] = []
                for stripe, worker in enumerate(participants):
                    routed_norm_id = (
                        f"c{chunk_id:03d}.l{layer:03d}.routed-norm.p{stripe:02d}"
                    )
                    tasks.append(
                        _task(
                            routed_norm_id,
                            worker,
                            chunk=chunk_id,
                            layer=layer,
                            operator="replicated_routed_expert_norm",
                            dependencies=(expert_finalize[stripe],),
                            service_ms=service.routed_norm_ms * multipliers[worker],
                        )
                    )
                    routed_norm_tasks.append(routed_norm_id)
                for stripe, worker in enumerate(participants):
                    identifier = f"c{chunk_id:03d}.l{layer:03d}.up-shared.p{stripe:02d}"
                    tasks.append(
                        _task(
                            identifier,
                            worker,
                            chunk=chunk_id,
                            layer=layer,
                            operator="colocated_latent_up_and_shared_expert_partial",
                            dependencies=(routed_norm_tasks[stripe], moe_fanout),
                            service_ms=service.latent_up_shared_worker_ms[stripe]
                            * multipliers[worker],
                        )
                    )
                    moe_tasks.append(identifier)
            moe_reduce = f"c{chunk_id:03d}.l{layer:03d}.moe-reduce"
            tasks.append(
                tree_collective(
                    task_id=moe_reduce,
                    participants=participants,
                    payload_bytes=actual_rows * HIDDEN * 4,
                    dependency_ids=tuple(moe_tasks),
                    profile=_jittered(
                        configuration.local_profile,
                        configuration.jitter_fraction,
                        rng,
                    ),
                    request_id="zero-draft-target-oracle",
                    block_id="verification",
                    chunk_id=chunk_id,
                    layer_id=layer,
                    operator="routed_and_shared_partial_allreduce",
                )
            )
            moe_finalize: list[str] = []
            for stripe, worker in enumerate(participants):
                identifier = (
                    f"c{chunk_id:03d}.l{layer:03d}.moe-reduce-compute.p{stripe:02d}"
                )
                tasks.append(
                    _task(
                        identifier,
                        worker,
                        chunk=chunk_id,
                        layer=layer,
                        operator="moe_allreduce_pairwise_accumulation",
                        dependencies=(moe_reduce,),
                        service_ms=service.hidden_pair_reduce_ms
                        * math.ceil(math.log2(spec.stripe_degree))
                        * multipliers[worker],
                    )
                )
                moe_finalize.append(identifier)
            finalize_id = f"c{chunk_id:03d}.l{layer:03d}.finalize"
            tasks.append(
                _task(
                    finalize_id,
                    root,
                    chunk=chunk_id,
                    layer=layer,
                    operator="residual_finalize",
                    dependencies=(moe_finalize[0],),
                    service_ms=service.layer_finalize_ms * multipliers[root],
                )
            )
            layer_complete[(chunk_id, layer)] = finalize_id
        final_pod = spec.pod_count - 1
        participants = tuple(
            _worker(final_pod, stripe) for stripe in range(spec.stripe_degree)
        )
        endpoint_tasks: list[str] = []
        for stripe, worker in enumerate(participants):
            identifier = f"c{chunk_id:03d}.endpoint.p{stripe:02d}"
            tasks.append(
                _task(
                    identifier,
                    worker,
                    chunk=chunk_id,
                    layer=LAYERS,
                    operator="lm_head_vocabulary_shard",
                    dependencies=(layer_complete[(chunk_id, LAYERS - 1)],),
                    service_ms=service.endpoint_worker_ms[stripe] * multipliers[worker],
                )
            )
            endpoint_tasks.append(identifier)
        argmax_network = f"c{chunk_id:03d}.endpoint.argmax-network"
        tasks.append(
            tree_collective(
                task_id=argmax_network,
                participants=participants,
                payload_bytes=16,
                dependency_ids=tuple(endpoint_tasks),
                profile=_jittered(
                    configuration.local_profile,
                    configuration.jitter_fraction,
                    rng,
                ),
                request_id="zero-draft-target-oracle",
                block_id="verification",
                chunk_id=chunk_id,
                layer_id=LAYERS,
                operator="distributed_exact_argmax_reduce",
                all_reduce=False,
            )
        )
        tasks.append(
            _task(
                f"c{chunk_id:03d}.endpoint.argmax-finalize",
                participants[0],
                chunk=chunk_id,
                layer=LAYERS,
                operator="distributed_exact_argmax_finalize",
                dependencies=(argmax_network,),
                service_ms=service.layer_finalize_ms * multipliers[participants[0]],
            )
        )
    return tasks


def simulate(
    service: MeasuredShardService,
    configuration: SimulationConfiguration,
) -> tuple[dict[str, Any], EventRun]:
    tasks = build_worker_dag(service, configuration)
    run = DeterministicMicroworkerEngine().run(tasks)
    assert_headline_trace(run.records)
    utilizations = list(run.worker_utilization.values())
    accepted = configuration.block + 1
    tok_s = accepted / (run.makespan_ms / 1000.0)
    active_workers = sum(value > 0 for value in utilizations)
    compute_records = [row for row in run.records if row.resource_type == "microworker"]
    active = 0
    peak_simultaneous = 0
    boundaries = sorted(
        [
            boundary
            for row in compute_records
            for boundary in ((row.start_time, 1), (row.finish_time, -1))
        ],
        key=lambda item: (item[0], item[1]),
    )
    for _time, change in boundaries:
        active += change
        peak_simultaneous = max(peak_simultaneous, active)
    result = {
        "schema_version": "experiment-019-worker-simulation-result-v1",
        **asdict(configuration.placement),
        "block": configuration.block,
        "chunk": configuration.chunk,
        "local_profile": configuration.local_profile.name,
        "inter_pod_profile": configuration.inter_pod_profile.name,
        "compute_slowdown": configuration.compute_slowdown,
        "heterogeneity": configuration.heterogeneity,
        "jitter_fraction": configuration.jitter_fraction,
        "activation_strategy": configuration.activation_strategy,
        "placement_family": (
            "C_full_layer_stripes_plus_worker_wavefront"
            if configuration.placement.depth_span == 1
            else "D_depth_stripes_plus_E_exact_worker_wavefront"
        ),
        "target_pass_ms": run.makespan_ms,
        "exact_tok_s_per_user": tok_s,
        "total_physical_compute_work_ms": run.total_compute_work_ms,
        "critical_path_compute_ms": run.critical_path_compute_ms,
        "network_critical_path_ms": run.network_critical_path_ms,
        "total_network_bytes": run.total_network_bytes,
        "network_bytes_per_accepted_token": run.total_network_bytes / accepted,
        "worker_seconds_per_accepted_token": run.total_compute_work_ms / 1000 / accepted,
        "average_worker_utilization": statistics.fmean(utilizations),
        "p50_worker_utilization": float(np.percentile(utilizations, 50)),
        "p95_worker_utilization": float(np.percentile(utilizations, 95)),
        "active_workers": active_workers,
        "active_workers_per_token": active_workers,
        "active_workers_per_chunk": active_workers,
        "peak_simultaneous_workers": peak_simultaneous,
        "total_task_count": len(run.records),
        "all_compute_resources_are_microworkers": True,
        "evidence_class": service.source + " + deterministic explicit-worker event model",
    }
    return result, run


__all__ = [
    "MeasuredShardService",
    "NETWORK_PROFILES",
    "SimulationConfiguration",
    "build_worker_dag",
    "simulate",
]
