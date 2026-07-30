"""Complete deterministic synthetic-swarm simulation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, cast

import numpy as np

from swarm_inference.config.models import (
    Backend,
    ExperimentConfig,
    OperationKind,
    RequestStatus,
    SchedulerMode,
    StageBenchmark,
    StageDefinition,
    StageReplica,
    VerificationState,
    WorkerCapability,
)
from swarm_inference.coordinator.placement import PlacementPlan, place_replicas
from swarm_inference.exceptions import BackpressureError, ConfigurationError
from swarm_inference.simulation.clock import SimClock
from swarm_inference.simulation.events import EventRecord
from swarm_inference.simulation.model import build_synthetic_stages
from swarm_inference.simulation.network import NetworkEmulator
from swarm_inference.simulation.node import PlannedOperation, SimWorker
from swarm_inference.simulation.queueing import percentile, utilisation


@dataclass(slots=True)
class SimRequest:
    request_id: str
    arrival_s: float
    prompt_tokens: int
    target_output_tokens: int
    status: RequestStatus = RequestStatus.PENDING
    verification_state: VerificationState = VerificationState.UNVERIFIED
    committed_tokens: int = 0
    first_token_s: float | None = None
    completed_s: float | None = None
    queueing_s: float = 0.0
    network_s: float = 0.0
    execution_s: float = 0.0
    replay_s: float = 0.0
    replay_bytes: int = 0
    retries: int = 0
    route_changes: int = 0
    corrupted: bool = False
    failure_reason: str | None = None
    static_route: dict[int, str] = field(default_factory=dict)
    stage_routes: list[dict[str, Any]] = field(default_factory=list)
    replay_inputs: dict[int, list[int]] = field(default_factory=dict)

    def to_metrics(self) -> dict[str, Any]:
        end_to_end = self.completed_s - self.arrival_s if self.completed_s is not None else None
        ttft = self.first_token_s - self.arrival_s if self.first_token_s is not None else None
        if (
            self.completed_s is not None
            and self.first_token_s is not None
            and self.committed_tokens > 1
            and self.completed_s > self.first_token_s
        ):
            decode_tps = (self.committed_tokens - 1) / (self.completed_s - self.first_token_s)
        else:
            decode_tps = 0.0
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "verification_state": self.verification_state.value,
            "arrival_s": self.arrival_s,
            "first_token_s": self.first_token_s,
            "completed_s": self.completed_s,
            "time_to_first_token_s": ttft,
            "decode_tokens_s": decode_tps,
            "end_to_end_s": end_to_end,
            "queueing_s": self.queueing_s,
            "network_s": self.network_s,
            "stage_execution_s": self.execution_s,
            "replay_s": self.replay_s,
            "replay_bytes": self.replay_bytes,
            "retry_count": self.retries,
            "route_changes": self.route_changes,
            "committed_output_tokens": self.committed_tokens,
            "failure_reason": self.failure_reason,
        }


@dataclass(slots=True)
class SimulationResult:
    seed: int
    execution_mode: str
    node_count: int
    concurrent_requests: int
    simulated_duration_s: float
    stages: list[StageDefinition]
    placement: PlacementPlan
    events: list[EventRecord]
    requests: list[dict[str, Any]]
    workers: list[dict[str, Any]]
    stage_metrics: list[dict[str, Any]]
    network_metrics: list[dict[str, Any]]
    summary: dict[str, Any]

    def deterministic_fingerprint(self) -> str:
        import json

        deterministic = {
            "seed": self.seed,
            "node_count": self.node_count,
            "concurrent_requests": self.concurrent_requests,
            "events": [event.to_dict() for event in self.events],
            "requests": self.requests,
            "workers": self.workers,
            "stage_metrics": self.stage_metrics,
            "network_metrics": self.network_metrics,
            "summary": self.summary,
        }
        encoded = json.dumps(
            deterministic, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


class Simulator:
    """Seeded event-queue simulator; never sleeps or reads wall time for execution."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        node_count: int | None = None,
        concurrent_requests: int | None = None,
    ) -> None:
        if config.execution_mode.value != "simulation":
            raise ConfigurationError("Simulator requires execution_mode=simulation")
        self.config = config
        self.node_count = node_count or sum(profile.count for profile in config.nodes)
        self.concurrent_requests = concurrent_requests or config.workload.concurrent_requests
        if self.node_count < config.model.stage_count:
            raise ConfigurationError(
                f"node count {self.node_count} is smaller than stage count "
                f"{config.model.stage_count}"
            )
        seed_sequence = np.random.SeedSequence(config.seed)
        simulation_seed, network_seed = seed_sequence.spawn(2)
        self.rng = np.random.default_rng(simulation_seed)
        self.clock = SimClock()
        self.network = NetworkEmulator(config.network, seed=int(network_seed.generate_state(1)[0]))
        self.stages = build_synthetic_stages(config.model)
        self.workers = self._build_workers()
        self.capabilities = self._build_capabilities()
        self.placement = place_replicas(stages=self.stages, workers=self.capabilities)
        self._install_placement()
        self.replicas_by_stage: dict[int, list[StageReplica]] = {
            stage.stage_id: sorted(
                [
                    replica
                    for replica in self.placement.replicas
                    if replica.stage_id == stage.stage_id
                ],
                key=lambda replica: replica.worker_id,
            )
            for stage in self.stages
        }
        self.events: list[EventRecord] = []
        self.requests: dict[str, SimRequest] = {}
        self._event_sequence = 0
        self._operation_routes: dict[tuple[str, int, int], str] = {}
        self._stage_busy: dict[int, float] = {stage.stage_id: 0.0 for stage in self.stages}
        self._stage_failures: dict[int, int] = {stage.stage_id: 0 for stage in self.stages}
        self._stage_replay_s: dict[int, float] = {stage.stage_id: 0.0 for stage in self.stages}
        self._stage_route_counts: dict[tuple[int, str], int] = {}
        self._failures_during_active_requests = 0

    def _expanded_profiles(self) -> list[Any]:
        templates: list[Any] = []
        for profile in self.config.nodes:
            templates.extend([profile] * profile.count)
        if not templates:
            raise ConfigurationError("no node profiles configured")
        return [templates[index % len(templates)] for index in range(self.node_count)]

    def _build_workers(self) -> dict[str, SimWorker]:
        workers: dict[str, SimWorker] = {}
        profiles = self._expanded_profiles()
        corrupt_count = math.floor(self.node_count * self.config.faults.corrupt_worker_fraction)
        corrupt_indices = set(
            self.rng.choice(self.node_count, size=corrupt_count, replace=False).tolist()
            if corrupt_count
            else []
        )
        slow_count = math.floor(self.node_count * self.config.faults.slow_worker_fraction)
        remaining_indices = [
            index for index in range(self.node_count) if index not in corrupt_indices
        ]
        slow_indices = set(
            self.rng.choice(
                remaining_indices, size=min(slow_count, len(remaining_indices)), replace=False
            ).tolist()
            if slow_count
            else []
        )
        for index, profile in enumerate(profiles):
            worker_id = f"worker-{index:03d}-{profile.name}"
            rate = profile.compute_rate_layers_s
            if index in slow_indices:
                rate /= self.config.faults.slow_worker_multiplier
            workers[worker_id] = SimWorker(
                worker_id=worker_id,
                profile_name=profile.name,
                memory_bytes=profile.memory_bytes,
                compute_rate_layers_s=rate,
                reliability=profile.reliability,
                max_concurrent_operations=profile.max_concurrent_stage_operations,
                queue_capacity=self.config.queue.capacity,
                corrupt=index in corrupt_indices,
            )
        return workers

    def _build_capabilities(self) -> list[WorkerCapability]:
        capabilities: list[WorkerCapability] = []
        for worker in self.workers.values():
            benchmarks = []
            for stage in self.stages:
                stage_layers = stage.layer_end - stage.layer_start
                mean_ms = (
                    stage_layers
                    * self.config.model.compute_work_per_layer
                    / worker.compute_rate_layers_s
                    * 1000
                )
                benchmarks.append(
                    StageBenchmark(
                        stage_id=stage.stage_id,
                        worker_class=worker.profile_name,
                        operation=OperationKind.DECODE,
                        sequence_length=1,
                        batch_size=1,
                        mean_ms=mean_ms,
                        p95_ms=mean_ms,
                        samples=1,
                        measured=False,
                    )
                )
            capabilities.append(
                WorkerCapability(
                    worker_id=worker.worker_id,
                    public_key="simulation-no-identity",
                    hostname=worker.worker_id,
                    operating_system="simulation",
                    architecture="simulation",
                    backend=Backend.SYNTHETIC,
                    cpu_model=worker.profile_name,
                    logical_cpu_count=1,
                    physical_cpu_count=1,
                    total_ram_bytes=worker.memory_bytes,
                    available_ram_bytes=worker.memory_bytes,
                    supported_dtypes=[self.config.model.activation_dtype],
                    supported_quantisation_formats=[],
                    stage_benchmarks=benchmarks,
                    upload_bandwidth_bytes_s=self.config.network.upload_bandwidth_bytes_s,
                    download_bandwidth_bytes_s=self.config.network.download_bandwidth_bytes_s,
                    coordinator_latency_ms=self.config.network.base_latency_ms,
                    reliability_score=worker.reliability,
                    memory_limit_bytes=worker.memory_bytes,
                    max_concurrent_stage_operations=worker.max_concurrent_operations,
                    profile_source="measured"
                    if any(
                        profile.name == worker.profile_name and profile.measured
                        for profile in self.config.nodes
                    )
                    else "assumed",
                )
            )
        return capabilities

    def _install_placement(self) -> None:
        for replica in self.placement.replicas:
            stage = self.stages[replica.stage_id]
            self.workers[replica.worker_id].assign(
                stage_id=stage.stage_id,
                required_bytes=stage.required_memory_bytes,
            )

    def _record(
        self,
        event_type: str,
        *,
        request_id: str | None = None,
        worker_id: str | None = None,
        stage_id: int | None = None,
        **details: Any,
    ) -> None:
        self.events.append(
            EventRecord(
                sequence=self._event_sequence,
                simulated_time_s=self.clock.now_s,
                event_type=event_type,
                request_id=request_id,
                worker_id=worker_id,
                stage_id=stage_id,
                details=details,
            )
        )
        self._event_sequence += 1

    def _schedule_faults(self) -> None:
        duration = max(
            self.config.steady_state_s,
            self.config.workload.duration_s or 0,
        )
        rate = self.config.faults.churn_rate_per_hour / 3600.0
        if rate > 0:
            for worker_id in sorted(self.workers):
                failure_time = float(self.rng.exponential(1.0 / rate))
                if failure_time <= duration:
                    self.clock.schedule_at(
                        failure_time,
                        partial(self._fail_worker, worker_id, reason="churn"),
                        name=f"churn:{worker_id}",
                    )
        burst_count = math.floor(self.node_count * self.config.faults.burst_failure_fraction)
        if burst_count:
            worker_ids = sorted(self.workers)
            selected = self.rng.choice(
                worker_ids, size=min(burst_count, len(worker_ids)), replace=False
            ).tolist()
            at_s = duration / 2.0
            for worker_id in selected:
                self.clock.schedule_at(
                    at_s,
                    partial(self._fail_worker, str(worker_id), reason="burst"),
                    name=f"burst:{worker_id}",
                )

    def _fail_worker(self, worker_id: str, *, reason: str) -> None:
        worker = self.workers[worker_id]
        if not worker.healthy:
            return
        worker.healthy = False
        worker.failures += 1
        active_request_count = sum(
            request.status == RequestStatus.RUNNING for request in self.requests.values()
        )
        if active_request_count:
            self._failures_during_active_requests += 1
        self._record(
            "worker_failed",
            worker_id=worker_id,
            reason=reason,
            active_request_count=active_request_count,
        )

    def _start_requests(self) -> None:
        interval = self.config.workload.arrival_interval_ms / 1000.0
        for index in range(self.concurrent_requests):
            request_id = f"request-{index:05d}"
            arrival = index * interval
            request = SimRequest(
                request_id=request_id,
                arrival_s=arrival,
                prompt_tokens=self.config.workload.prompt_tokens,
                target_output_tokens=self.config.workload.output_tokens,
            )
            self.requests[request_id] = request
            self.clock.schedule_at(
                arrival,
                partial(self._begin_request, request),
                name=f"request_start:{request_id}",
            )

    def _begin_request(self, request: SimRequest) -> None:
        request.status = RequestStatus.RUNNING
        self._record("request_started", request_id=request.request_id)
        if self.config.scheduler == SchedulerMode.STATIC:
            for stage in self.stages:
                replicas = self._eligible_replicas(stage.stage_id)
                if not replicas:
                    self._fail_request(
                        request, f"no healthy static replica for stage {stage.stage_id}"
                    )
                    return
                request.static_route[stage.stage_id] = replicas[0].worker_id
        self._dispatch_stage(
            request,
            stage_id=0,
            operation=OperationKind.PREFILL,
            token_position=0,
            sequence_length=request.prompt_tokens,
            previous_worker_id="coordinator",
            excluded_workers=set(),
        )

    def _eligible_replicas(self, stage_id: int) -> list[StageReplica]:
        return [
            replica
            for replica in self.replicas_by_stage[stage_id]
            if self.workers[replica.worker_id].healthy
            and not self.workers[replica.worker_id].quarantined
        ]

    def _select_replica(
        self,
        request: SimRequest,
        *,
        stage: StageDefinition,
        ready_s: float,
        sequence_length: int,
        excluded_workers: set[str],
    ) -> StageReplica | None:
        candidates = [
            replica
            for replica in self._eligible_replicas(stage.stage_id)
            if replica.worker_id not in excluded_workers
        ]
        if self.config.scheduler == SchedulerMode.STATIC:
            selected_id = request.static_route.get(stage.stage_id)
            candidates = [replica for replica in candidates if replica.worker_id == selected_id]
        if not candidates:
            return None
        if self.config.scheduler == SchedulerMode.STATIC:
            return candidates[0]
        work = self.config.model.compute_work_per_layer
        if sequence_length > 1:
            work *= sequence_length
        return min(
            candidates,
            key=lambda replica: (
                self.workers[replica.worker_id].predicted_completion(
                    now_s=ready_s,
                    stage_layers=stage.layer_end - stage.layer_start,
                    work_multiplier=work,
                )
                + (1.0 - self.workers[replica.worker_id].reliability)
                * (3.0 if self.config.workload.workload_class.value == "interactive" else 1.0),
                replica.worker_id,
            ),
        )

    def _dispatch_stage(
        self,
        request: SimRequest,
        *,
        stage_id: int,
        operation: OperationKind,
        token_position: int,
        sequence_length: int,
        previous_worker_id: str,
        excluded_workers: set[str],
    ) -> None:
        if request.status not in {RequestStatus.RUNNING, RequestStatus.RECOVERING}:
            return
        stage = self.stages[stage_id]
        payload_bytes = (
            sequence_length * 8
            if stage_id == 0
            else sequence_length * self.config.model.activation_bytes
        )
        replica = self._select_replica(
            request,
            stage=stage,
            ready_s=self.clock.now_s,
            sequence_length=sequence_length,
            excluded_workers=excluded_workers,
        )
        if replica is None:
            self._stage_failures[stage_id] += 1
            self._fail_request(
                request,
                f"no healthy route for stage {stage_id} after excluding {sorted(excluded_workers)}",
            )
            return
        worker = self.workers[replica.worker_id]
        transmission = self.network.transmit(
            source=previous_worker_id,
            destination=worker.worker_id,
            now_s=self.clock.now_s,
            payload_bytes=payload_bytes,
        )
        request.network_s += (
            transmission.queueing_s + transmission.latency_s + transmission.serialization_s
        )
        worker.bytes_received += payload_bytes
        if previous_worker_id in self.workers:
            self.workers[previous_worker_id].bytes_sent += payload_bytes
        self._record(
            "network_transmission",
            request_id=request.request_id,
            worker_id=worker.worker_id,
            stage_id=stage_id,
            **transmission.to_dict(),
        )
        if transmission.lost:
            request.retries += 1
            if request.retries > 8:
                self._fail_request(request, "transport retry budget exhausted")
                return
            self.clock.schedule_at(
                transmission.completed_at_s,
                lambda: self._dispatch_stage(
                    request,
                    stage_id=stage_id,
                    operation=operation,
                    token_position=token_position,
                    sequence_length=sequence_length,
                    previous_worker_id=previous_worker_id,
                    excluded_workers=excluded_workers,
                ),
                name=f"network_retry:{request.request_id}:{stage_id}",
            )
            return
        request.replay_inputs.setdefault(stage_id, []).append(payload_bytes)
        work_multiplier = self.config.model.compute_work_per_layer
        if operation == OperationKind.PREFILL:
            work_multiplier *= sequence_length
        try:
            planned = worker.plan(
                now_s=transmission.completed_at_s,
                stage_id=stage_id,
                stage_layers=stage.layer_end - stage.layer_start,
                operation=operation,
                work_multiplier=work_multiplier,
                coordinator_overhead_s=0.0001,
            )
        except BackpressureError as exc:
            self._record(
                "backpressure",
                request_id=request.request_id,
                worker_id=worker.worker_id,
                stage_id=stage_id,
                reason=str(exc),
            )
            request.retries += 1
            alternatives = set(excluded_workers)
            alternatives.add(worker.worker_id)
            if self._eligible_replicas(stage_id):
                self.clock.schedule_in(
                    0.0001,
                    lambda: self._dispatch_stage(
                        request,
                        stage_id=stage_id,
                        operation=operation,
                        token_position=token_position,
                        sequence_length=sequence_length,
                        previous_worker_id=previous_worker_id,
                        excluded_workers=alternatives,
                    ),
                    name=f"backpressure_reroute:{request.request_id}:{stage_id}",
                )
            else:
                self._fail_request(request, str(exc))
            return
        request.queueing_s += planned.queueing_s
        request.execution_s += planned.execution_s
        self._stage_busy[stage_id] += planned.execution_s
        self._stage_route_counts[(stage_id, worker.worker_id)] = (
            self._stage_route_counts.get((stage_id, worker.worker_id), 0) + 1
        )
        request.stage_routes.append(
            {
                "token_position": token_position,
                "stage_id": stage_id,
                "worker_id": worker.worker_id,
                "predicted_completion_s": planned.completed_at_s,
                "queue_depth": worker.queue_depth(transmission.completed_at_s),
            }
        )
        self._record(
            "stage_scheduled",
            request_id=request.request_id,
            worker_id=worker.worker_id,
            stage_id=stage_id,
            operation=operation.value,
            token_position=token_position,
            queueing_s=planned.queueing_s,
            execution_s=planned.execution_s,
        )
        self.clock.schedule_at(
            planned.completed_at_s,
            lambda: self._complete_stage(
                request,
                stage=stage,
                worker=worker,
                planned=planned,
                operation=operation,
                token_position=token_position,
                sequence_length=sequence_length,
                excluded_workers=excluded_workers,
            ),
            name=f"stage_complete:{request.request_id}:{stage_id}:{token_position}",
        )

    def _complete_stage(
        self,
        request: SimRequest,
        *,
        stage: StageDefinition,
        worker: SimWorker,
        planned: PlannedOperation,
        operation: OperationKind,
        token_position: int,
        sequence_length: int,
        excluded_workers: set[str],
    ) -> None:
        if request.status not in {RequestStatus.RUNNING, RequestStatus.RECOVERING}:
            return
        if not worker.healthy:
            self._recover_stage(
                request,
                stage=stage,
                failed_worker=worker,
                operation=operation,
                token_position=token_position,
                sequence_length=sequence_length,
                excluded_workers=excluded_workers,
                reason="worker failed during operation",
            )
            return
        audit = bool(
            self.config.integrity.enabled
            and self.rng.random()
            < max(
                self.config.integrity.audit_fraction,
                self.config.faults.audit_fraction,
            )
        )
        if worker.corrupt:
            request.corrupted = True
            self._record(
                "corrupt_result",
                request_id=request.request_id,
                worker_id=worker.worker_id,
                stage_id=stage.stage_id,
                audited=audit,
            )
            if audit:
                worker.audit_count += 1
                worker.audit_disagreements += 1
                worker.reputation = max(
                    0.0,
                    worker.reputation - self.config.integrity.disagreement_penalty,
                )
                if worker.reputation < self.config.integrity.quarantine_threshold:
                    worker.quarantined = True
                    self._record(
                        "worker_quarantined",
                        worker_id=worker.worker_id,
                        stage_id=stage.stage_id,
                        reputation=worker.reputation,
                    )
                request.corrupted = False
                self._recover_stage(
                    request,
                    stage=stage,
                    failed_worker=worker,
                    operation=operation,
                    token_position=token_position,
                    sequence_length=sequence_length,
                    excluded_workers=excluded_workers,
                    reason="audit disagreement",
                )
                return
        elif audit:
            worker.audit_count += 1
            worker.reputation = min(1.0, worker.reputation + self.config.integrity.agreement_reward)
            self._record(
                "audit_agreement",
                request_id=request.request_id,
                worker_id=worker.worker_id,
                stage_id=stage.stage_id,
            )
        self._record(
            "stage_completed",
            request_id=request.request_id,
            worker_id=worker.worker_id,
            stage_id=stage.stage_id,
            operation=operation.value,
            token_position=token_position,
        )
        if stage.stage_id + 1 < len(self.stages):
            self._dispatch_stage(
                request,
                stage_id=stage.stage_id + 1,
                operation=operation,
                token_position=token_position,
                sequence_length=sequence_length,
                previous_worker_id=worker.worker_id,
                excluded_workers=set(),
            )
            return
        self._commit_output_token(request, worker=worker, token_position=token_position)

    def _recover_stage(
        self,
        request: SimRequest,
        *,
        stage: StageDefinition,
        failed_worker: SimWorker,
        operation: OperationKind,
        token_position: int,
        sequence_length: int,
        excluded_workers: set[str],
        reason: str,
    ) -> None:
        request.status = RequestStatus.RECOVERING
        request.retries += 1
        request.route_changes += 1
        self._stage_failures[stage.stage_id] += 1
        new_excluded = set(excluded_workers)
        new_excluded.add(failed_worker.worker_id)
        replay_bytes = sum(request.replay_inputs.get(stage.stage_id, []))
        bandwidth = min(
            self.config.network.upload_bandwidth_bytes_s,
            self.config.network.download_bandwidth_bytes_s,
        )
        replay_transport_s = replay_bytes / bandwidth
        completed_positions = max(request.committed_tokens, 1)
        replay_compute_s = (
            (stage.layer_end - stage.layer_start)
            * completed_positions
            * self.config.model.compute_work_per_layer
            / max(
                (
                    replica.measured_service_rate
                    for replica in self._eligible_replicas(stage.stage_id)
                    if replica.worker_id not in new_excluded
                ),
                default=1.0,
            )
        )
        replay_s = replay_transport_s + replay_compute_s
        request.replay_bytes += replay_bytes
        request.replay_s += replay_s
        self._stage_replay_s[stage.stage_id] += replay_s
        self._record(
            "stage_recovery_started",
            request_id=request.request_id,
            worker_id=failed_worker.worker_id,
            stage_id=stage.stage_id,
            reason=reason,
            replay_bytes=replay_bytes,
            replay_duration_s=replay_s,
            additional_computation_s=replay_compute_s,
        )
        if not [
            replica
            for replica in self._eligible_replicas(stage.stage_id)
            if replica.worker_id not in new_excluded
        ]:
            self._fail_request(
                request,
                f"stage {stage.stage_id} failed and no compatible replica can replay state",
            )
            return
        self.clock.schedule_in(
            replay_s,
            lambda: self._resume_after_replay(
                request,
                stage=stage,
                operation=operation,
                token_position=token_position,
                sequence_length=sequence_length,
                previous_worker_id="coordinator",
                excluded_workers=new_excluded,
            ),
            name=f"replay:{request.request_id}:{stage.stage_id}",
        )

    def _resume_after_replay(
        self,
        request: SimRequest,
        *,
        stage: StageDefinition,
        operation: OperationKind,
        token_position: int,
        sequence_length: int,
        previous_worker_id: str,
        excluded_workers: set[str],
    ) -> None:
        request.status = RequestStatus.RUNNING
        self._record(
            "stage_recovery_completed",
            request_id=request.request_id,
            stage_id=stage.stage_id,
        )
        self._dispatch_stage(
            request,
            stage_id=stage.stage_id,
            operation=operation,
            token_position=token_position,
            sequence_length=sequence_length,
            previous_worker_id=previous_worker_id,
            excluded_workers=excluded_workers,
        )

    def _commit_output_token(
        self,
        request: SimRequest,
        *,
        worker: SimWorker,
        token_position: int,
    ) -> None:
        request.committed_tokens += 1
        worker.tokens_contributed += 1
        if request.first_token_s is None:
            request.first_token_s = self.clock.now_s
        self._record(
            "token_committed",
            request_id=request.request_id,
            worker_id=worker.worker_id,
            token_position=token_position,
            verified=not request.corrupted,
        )
        if request.committed_tokens >= request.target_output_tokens:
            request.completed_s = self.clock.now_s
            if request.corrupted:
                request.status = RequestStatus.FAILED
                request.verification_state = VerificationState.REJECTED
                request.failure_reason = "undetected corrupt stage output reached completion"
            else:
                request.status = RequestStatus.COMPLETED
                request.verification_state = VerificationState.VERIFIED
            self._record(
                "request_finished",
                request_id=request.request_id,
                status=request.status.value,
                committed_tokens=request.committed_tokens,
            )
            return
        self._dispatch_stage(
            request,
            stage_id=0,
            operation=OperationKind.DECODE,
            token_position=request.committed_tokens,
            sequence_length=1,
            previous_worker_id="coordinator",
            excluded_workers=set(),
        )

    def _fail_request(self, request: SimRequest, reason: str) -> None:
        if request.status in {
            RequestStatus.COMPLETED,
            RequestStatus.CANCELLED,
            RequestStatus.FAILED,
        }:
            return
        request.status = RequestStatus.FAILED
        request.verification_state = VerificationState.REJECTED
        request.completed_s = self.clock.now_s
        request.failure_reason = reason
        self._record(
            "request_failed",
            request_id=request.request_id,
            reason=reason,
        )

    def run(self) -> SimulationResult:
        self._schedule_faults()
        self._start_requests()
        maximum_events = max(
            10_000,
            self.concurrent_requests * self.config.workload.output_tokens * len(self.stages) * 30,
        )
        executed = self.clock.run(maximum_events=maximum_events)
        if self.clock.pending_events:
            for request in self.requests.values():
                self._fail_request(
                    request,
                    f"simulation event budget {maximum_events} exhausted after {executed} events",
                )
        return self._result()

    def _result(self) -> SimulationResult:
        elapsed = max(self.clock.now_s, 1e-12)
        request_metrics = [
            request.to_metrics()
            for request in sorted(self.requests.values(), key=lambda item: item.request_id)
        ]
        verified_requests = [
            request
            for request in self.requests.values()
            if request.status == RequestStatus.COMPLETED
            and request.verification_state == VerificationState.VERIFIED
        ]
        verified_tokens = sum(request.committed_tokens for request in verified_requests)
        aggregate_tps = verified_tokens / elapsed
        completed_fraction = len(verified_requests) / max(len(self.requests), 1)
        worker_metrics: list[dict[str, Any]] = []
        for worker in sorted(self.workers.values(), key=lambda item: item.worker_id):
            service = worker.service_times_s
            worker_metrics.append(
                {
                    "worker_id": worker.worker_id,
                    "profile": worker.profile_name,
                    "assigned_stage_id": worker.assigned_stage_id,
                    "busy_time_s": worker.busy_time_s,
                    "idle_time_s": max(
                        0.0,
                        elapsed * worker.max_concurrent_operations - worker.busy_time_s,
                    ),
                    "utilisation": utilisation(
                        busy_time_s=worker.busy_time_s,
                        elapsed_s=elapsed,
                        parallelism=worker.max_concurrent_operations,
                    ),
                    "stage_operations": worker.operations,
                    "tokens_contributed": worker.tokens_contributed,
                    "bytes_sent": worker.bytes_sent,
                    "bytes_received": worker.bytes_received,
                    "queue_depth": worker.queue_depth(elapsed),
                    "mean_service_time_s": (sum(service) / len(service) if service else 0.0),
                    "p50_service_time_s": percentile(service, 50),
                    "p95_service_time_s": percentile(service, 95),
                    "p99_service_time_s": percentile(service, 99),
                    "memory_usage_bytes": worker.assigned_stage_bytes,
                    "vram_usage_bytes": 0,
                    "failure_count": worker.failures,
                    "audit_count": worker.audit_count,
                    "audit_disagreement_count": worker.audit_disagreements,
                    "reputation": worker.reputation,
                    "healthy": worker.healthy,
                    "quarantined": worker.quarantined,
                    "corrupt": worker.corrupt,
                }
            )
        stage_metrics: list[dict[str, Any]] = []
        for stage in self.stages:
            replicas = self.replicas_by_stage[stage.stage_id]
            healthy = [
                replica
                for replica in replicas
                if self.workers[replica.worker_id].healthy
                and not self.workers[replica.worker_id].quarantined
            ]
            capacity = sum(replica.measured_service_rate for replica in healthy)
            parallelism = sum(
                self.workers[replica.worker_id].max_concurrent_operations for replica in replicas
            )
            stage_metrics.append(
                {
                    "stage_id": stage.stage_id,
                    "replica_count": len(replicas),
                    "healthy_replica_count": len(healthy),
                    "aggregate_service_rate": capacity,
                    "queue_depth": sum(
                        self.workers[replica.worker_id].queue_depth(elapsed) for replica in replicas
                    ),
                    "utilisation": utilisation(
                        busy_time_s=self._stage_busy[stage.stage_id],
                        elapsed_s=elapsed,
                        parallelism=max(parallelism, 1),
                    ),
                    "bottleneck_duration_s": (
                        elapsed
                        if capacity
                        == min(
                            (
                                sum(
                                    item.measured_service_rate
                                    for item in self.replicas_by_stage[other.stage_id]
                                    if self.workers[item.worker_id].healthy
                                    and not self.workers[item.worker_id].quarantined
                                )
                                for other in self.stages
                            ),
                            default=0.0,
                        )
                        else 0.0
                    ),
                    "route_distribution": {
                        worker_id: count
                        for (stage_id, worker_id), count in sorted(self._stage_route_counts.items())
                        if stage_id == stage.stage_id
                    },
                    "failure_count": self._stage_failures[stage.stage_id],
                    "replay_overhead_s": self._stage_replay_s[stage.stage_id],
                }
            )
        network_metrics = [transmission.to_dict() for transmission in self.network.transmissions]
        stage_utilisations = [metric["utilisation"] for metric in stage_metrics]
        service_capacities = [metric["aggregate_service_rate"] for metric in stage_metrics]
        capacity_imbalance = (
            (max(service_capacities) - min(service_capacities))
            / max(max(service_capacities), 1e-12)
            if service_capacities
            else 1.0
        )
        summary = {
            "execution_mode": self.config.execution_mode.value,
            "values": "emulated",
            "model": self.config.model_id,
            "node_count": self.node_count,
            "concurrent_request_count": self.concurrent_requests,
            "simulated_duration_s": elapsed,
            "aggregate_verified_output_tokens_s": aggregate_tps,
            "verified_output_tokens": verified_tokens,
            "completed_verified_requests": len(verified_requests),
            "accepted_requests": len(self.requests),
            "completion_fraction": completed_fraction,
            "mean_request_tokens_s": (
                sum(metric["decode_tokens_s"] for metric in request_metrics)
                / max(len(request_metrics), 1)
            ),
            "mean_time_to_first_token_s": _mean_present(
                metric["time_to_first_token_s"] for metric in request_metrics
            ),
            "mean_end_to_end_s": _mean_present(
                metric["end_to_end_s"] for metric in request_metrics
            ),
            "minimum_stage_utilisation": min(stage_utilisations, default=0.0),
            "mean_stage_utilisation": (
                sum(stage_utilisations) / len(stage_utilisations) if stage_utilisations else 0.0
            ),
            "network_bytes": sum(
                int(cast(int, metric["payload_bytes"])) for metric in network_metrics
            ),
            "network_lost_transmissions": sum(1 for metric in network_metrics if metric["lost"]),
            "capacity_imbalance": capacity_imbalance,
            "failed_requests": sum(
                1 for request in self.requests.values() if request.status == RequestStatus.FAILED
            ),
            "failures_during_active_requests": self._failures_during_active_requests,
            "recovered_route_changes": sum(
                request.route_changes for request in self.requests.values()
            ),
            "replay_bytes": sum(request.replay_bytes for request in self.requests.values()),
            "replay_duration_s": sum(request.replay_s for request in self.requests.values()),
            "quarantined_workers": sum(1 for worker in self.workers.values() if worker.quarantined),
            "idle_workers": sum(
                1 for worker in self.workers.values() if worker.assigned_stage_id is None
            ),
        }
        return SimulationResult(
            seed=self.config.seed,
            execution_mode=self.config.execution_mode.value,
            node_count=self.node_count,
            concurrent_requests=self.concurrent_requests,
            simulated_duration_s=elapsed,
            stages=self.stages,
            placement=self.placement,
            events=self.events,
            requests=request_metrics,
            workers=worker_metrics,
            stage_metrics=stage_metrics,
            network_metrics=network_metrics,
            summary=summary,
        )


def _mean_present(values: Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None
