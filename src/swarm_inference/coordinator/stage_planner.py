"""Measured, memory-aware contiguous product stage planning."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from swarm_inference.cluster.models import NetworkLinkMeasurement
from swarm_inference.cluster.network import network_measurement_is_fresh
from swarm_inference.config.models import OperationKind, WorkerCapability, WorkerRole
from swarm_inference.coordinator.expert_planner import (
    ExpertPolicy,
    ExpertStrategy,
    ExpertStrategyCandidate,
    ExpertUtilityInputs,
    ExpertUtilityPlanner,
)
from swarm_inference.coordinator.model_catalog import InspectedProductModel
from swarm_inference.model.partition import (
    ModelPartitionMetadata,
    PartitionMethod,
    StageAssignment,
    equal_ranges,
)
from swarm_inference.protocol.expert import ReductionMode, TransportCodec
from swarm_inference.protocol.product import (
    DirectedLinkSelection,
    ModelPlanRequest,
    NodeUtilityReport,
    PlanCandidateReport,
    PlanWorkerAssignment,
    ProductExpertPlacement,
    ProductStageExpertPlan,
    ProductStagePlan,
    StagePlanReport,
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    report: PlanCandidateReport
    assignments: tuple[PlanWorkerAssignment, ...]


@dataclass(frozen=True, slots=True)
class _SearchState:
    next_layer: int
    selected_worker_ids: tuple[str, ...]
    assignments: tuple[PlanWorkerAssignment, ...]
    compute: dict[str, float]
    memory: dict[str, int]
    waits: dict[str, float]
    network: dict[str, float]
    links: tuple[DirectedLinkSelection, ...]
    unmeasured_assumptions: tuple[str, ...]
    total_ms: float
    headroom_ratios: tuple[float, ...]
    headroom_bytes: tuple[int, ...]
    reliability_scores: tuple[float, ...]


def _normalised_device(worker: WorkerCapability) -> str:
    if worker.device_identifier is None:
        raise ValueError(f"worker {worker.worker_id} has no stage device")
    return worker.device_identifier


def _measured_compute_ms(
    worker: WorkerCapability,
    assignment: StageAssignment,
    *,
    mean_layer_execution_ns: float,
) -> float | None:
    measured = [
        benchmark
        for benchmark in worker.stage_benchmarks
        if benchmark.measured
        and benchmark.mean_ms > 0
        and benchmark.operation == OperationKind.DECODE
        and benchmark.stage_id in {None, assignment.stage_id}
    ]
    if not measured:
        return None
    baseline = min(measured, key=lambda item: (item.mean_ms, item.p95_ms))
    if mean_layer_execution_ns <= 0:
        scale = float(len(assignment.layer_ids))
    else:
        scale = max(1e-9, assignment.estimated_compute_ns / mean_layer_execution_ns)
    return float(baseline.mean_ms * scale)


def _worker_node_id(worker: WorkerCapability) -> str:
    if worker.node_id:
        return worker.node_id
    if "/" in worker.worker_id:
        return worker.worker_id.split("/", 1)[0]
    return worker.worker_id


def _stage_name(stage_count: int, method: PartitionMethod) -> str:
    if stage_count == 1:
        return "local-monolithic"
    prefix = {2: "two", 3: "three", 4: "four", 8: "eight"}.get(stage_count, str(stage_count))
    return f"{prefix}-stage-{method}-ring"


def _assignment_for_range(
    metadata: ModelPartitionMetadata,
    *,
    stage_id: int,
    stage_count: int,
    start: int,
    end: int,
    device: str,
) -> StageAssignment:
    selected = metadata.layer_costs[start:end]
    if not selected:
        raise ValueError("every product stage must own at least one layer")
    overhead = (metadata.embedding_weight_bytes if stage_id == 0 else 0) + (
        metadata.final_weight_bytes if stage_id == stage_count - 1 else 0
    )
    return StageAssignment(
        stage_id=stage_id,
        layer_start=start,
        layer_end=end,
        layer_ids=tuple(range(start, end)),
        weight_bytes=sum(item.weight_bytes for item in selected) + overhead,
        estimated_compute_ns=sum(item.execution_ns for item in selected),
        measured_compute_ns=(
            sum(item.execution_ns for item in selected)
            if all(item.measured for item in selected)
            else None
        ),
        kv_cache_bytes_per_token=sum(item.kv_bytes_per_token for item in selected),
        peak_temporary_bytes=max(item.peak_temporary_bytes for item in selected),
        activation_bytes=max(item.activation_bytes for item in selected),
        device=device,
        owns_embeddings=stage_id == 0,
        owns_final_norm=stage_id == stage_count - 1,
        owns_output_projection=stage_id == stage_count - 1,
    )


class ProductStagePlanner:
    """Bounded deterministic beam search over contiguous N-stage topologies."""

    def __init__(
        self,
        *,
        maximum_candidate_workers: int = 64,
        maximum_stage_count: int = 32,
        beam_width: int = 512,
        network_measurement_ttl_seconds: int = 900,
        allow_unmeasured_links_for_explicit_plans: bool = True,
        balanced_throughput_weight: float = 0.45,
        balanced_memory_headroom_weight: float = 0.25,
        balanced_reliability_weight: float = 0.20,
        balanced_participation_weight: float = 0.10,
        network_measurement_provider: Callable[[], list[NetworkLinkMeasurement]] | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not 1 <= maximum_candidate_workers <= 256:
            raise ValueError("maximum candidate workers must be in [1, 256]")
        if not 1 <= maximum_stage_count <= 128:
            raise ValueError("maximum stage count must be in [1, 128]")
        if not 1 <= beam_width <= 8192:
            raise ValueError("planning beam width must be in [1, 8192]")
        if network_measurement_ttl_seconds <= 0:
            raise ValueError("network measurement TTL must be positive")
        weights = (
            balanced_throughput_weight,
            balanced_memory_headroom_weight,
            balanced_reliability_weight,
            balanced_participation_weight,
        )
        if any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("balanced planning weights must be non-negative with positive sum")
        self.maximum_candidate_workers = maximum_candidate_workers
        self.maximum_stage_count = maximum_stage_count
        self.beam_width = beam_width
        self.network_measurement_ttl_seconds = network_measurement_ttl_seconds
        self.allow_unmeasured_links_for_explicit_plans = allow_unmeasured_links_for_explicit_plans
        total_weight = sum(weights)
        self.balanced_weights = tuple(value / total_weight for value in weights)
        self.network_measurement_provider = network_measurement_provider or (lambda: [])
        self.clock_ns = clock_ns

    def _network_links(self) -> dict[tuple[str, str], NetworkLinkMeasurement]:
        selected: dict[tuple[str, str], NetworkLinkMeasurement] = {}
        for measurement in self.network_measurement_provider():
            key = (measurement.source_worker_id, measurement.destination_worker_id)
            previous = selected.get(key)
            if previous is None or measurement.measured_at_unix_ns > previous.measured_at_unix_ns:
                selected[key] = measurement
        return selected

    def _beam_key(
        self,
        state: _SearchState,
        mode: str,
        partition_method: PartitionMethod,
    ) -> tuple[object, ...]:
        minimum_headroom = min(state.headroom_ratios, default=0.0)
        average_reliability = (
            sum(state.reliability_scores) / len(state.reliability_scores)
            if state.reliability_scores
            else 0.0
        )
        participation = len({worker_id.split("/", 1)[0] for worker_id in state.selected_worker_ids})
        boundaries = tuple(item.assignment.layer_end for item in state.assignments)
        stable = (state.selected_worker_ids, boundaries)
        stage_service_times = [
            state.compute.get(f"stage-{index}", 0.0)
            + state.waits.get(f"stage-{index}-queue", 0.0)
            + state.waits.get(f"stage-{index}-reliability", 0.0)
            for index in range(len(state.assignments))
        ]
        search_time = (
            max(stage_service_times, default=0.0) + sum(state.network.values())
            if partition_method == "balanced"
            else state.total_ms
        )
        if mode == "capacity":
            return (-minimum_headroom, -sum(state.headroom_bytes), search_time, *stable)
        if mode == "balanced":
            throughput, memory, reliability, participation_weight = self.balanced_weights
            score = (
                throughput * search_time
                + memory * (1.0 - minimum_headroom) * 100.0
                + reliability * (1.0 - average_reliability) * 100.0
                - participation_weight * participation
            )
            return (score, search_time, *stable)
        return (search_time, len(state.assignments), *stable)

    def _definitions(
        self, effective_maximum: int, request: ModelPlanRequest
    ) -> list[tuple[str, int, PartitionMethod]]:
        definitions: list[tuple[str, int, PartitionMethod]] = [("local-monolithic", 1, "equal")]
        methods: tuple[PartitionMethod, ...] = ("equal", "balanced")
        for stage_count in range(2, effective_maximum + 1):
            for method in methods:
                definitions.append((_stage_name(stage_count, method), stage_count, method))
        if request.stage_count is not None and request.stage_count > effective_maximum:
            for method in methods:
                definitions.append(
                    (_stage_name(request.stage_count, method), request.stage_count, method)
                )
        return definitions

    def build_plan(
        self,
        request: ModelPlanRequest,
        inspected: InspectedProductModel,
    ) -> ProductStagePlan:
        all_workers = sorted(inspected.capabilities.values(), key=lambda item: item.worker_id)
        required_nodes = set(request.required_node_ids)
        excluded_nodes = set(request.excluded_node_ids)
        available_nodes = {_worker_node_id(worker) for worker in all_workers}
        missing_required = sorted(required_nodes - available_nodes)
        if missing_required:
            raise RuntimeError(
                "required nodes are not healthy and eligible: " + ", ".join(missing_required)
            )
        eligible_workers = [
            worker for worker in all_workers if _worker_node_id(worker) not in excluded_nodes
        ]
        workers = sorted(
            eligible_workers,
            key=lambda item: (
                _worker_node_id(item) not in required_nodes,
                item.active_session_count,
                item.current_queue_depth,
                item.worker_id,
            ),
        )[: self.maximum_candidate_workers]
        worker_nodes = {_worker_node_id(worker) for worker in workers}
        missing_after_bound = sorted(required_nodes - worker_nodes)
        if missing_after_bound:
            raise RuntimeError(
                "required nodes exceed the bounded candidate set: " + ", ".join(missing_after_bound)
            )
        effective_maximum = min(
            inspected.spec.layer_count,
            len(workers),
            self.maximum_stage_count,
        )
        candidates: list[_Candidate] = []
        expert_workers = [
            worker
            for worker in inspected.all_capabilities.values()
            if set(worker.roles)
            & {
                WorkerRole.WHOLE_EXPERT,
                WorkerRole.EXPERT_MICROSHARD,
                WorkerRole.REDUCER,
            }
        ]
        explicit = request.stage_count is not None or request.partition_method != "auto"
        use_dense_memory_floor = (
            request.expert_policy != ExpertPolicy.LOCAL.value
            and bool(expert_workers)
            and inspected.metadata.expert_count > 0
        )
        links = self._network_links()
        excluded_workers = {
            worker.worker_id: (
                "node explicitly excluded"
                if _worker_node_id(worker) in excluded_nodes
                else "outside configured maximum candidate workers"
            )
            for worker in all_workers
            if worker not in workers
        }
        for name, stage_count, method in self._definitions(effective_maximum, request):
            candidates.append(
                self._evaluate_candidate(
                    name=name,
                    stage_count=stage_count,
                    method=method,
                    request=request,
                    inspected=inspected,
                    workers=workers,
                    explicit=explicit,
                    use_dense_memory_floor=use_dense_memory_floor,
                    required_nodes=required_nodes,
                    links=links,
                    base_excluded_workers=excluded_workers,
                )
            )
        local_throughputs = [
            float(candidate.report.expected_utility_tokens_s)
            for candidate in candidates
            if candidate.report.feasible
            and candidate.report.stage_count == 1
            and candidate.report.expected_utility_tokens_s is not None
        ]
        if not local_throughputs and workers:
            baseline_request = request.model_copy(
                update={
                    "stage_count": None,
                    "partition_method": "auto",
                    "require_distributed": False,
                    "required_node_ids": [],
                }
            )
            baseline_candidate = self._evaluate_candidate(
                name="local-baseline-diagnostic",
                stage_count=1,
                method="equal",
                request=baseline_request,
                inspected=inspected,
                workers=workers,
                explicit=False,
                use_dense_memory_floor=use_dense_memory_floor,
                required_nodes=set(),
                links=links,
                base_excluded_workers=excluded_workers,
            )
            if baseline_candidate.report.expected_utility_tokens_s is not None:
                local_throughputs.append(float(baseline_candidate.report.expected_utility_tokens_s))
        local_baseline = max(local_throughputs, default=None)
        enriched: list[_Candidate] = []
        maximum_throughput = max(
            (
                float(candidate.report.expected_utility_tokens_s)
                for candidate in candidates
                if candidate.report.feasible
                and candidate.report.expected_utility_tokens_s is not None
            ),
            default=0.0,
        )
        total_nodes = max(1, len({_worker_node_id(worker) for worker in workers}))
        for candidate in candidates:
            report = candidate.report
            throughput = report.expected_utility_tokens_s
            distributed_throughput = throughput if report.stage_count > 1 else None
            delta = (
                float(distributed_throughput) - local_baseline
                if distributed_throughput is not None and local_baseline is not None
                else None
            )
            headroom_ratios = [
                report.per_stage_headroom_bytes[key] / max(report.memory_estimates_bytes[key], 1)
                for key in report.per_stage_headroom_bytes
            ]
            minimum_headroom = min(headroom_ratios, default=0.0)
            selected_capabilities = [
                inspected.capabilities[worker_id]
                for worker_id in report.worker_ids
                if worker_id in inspected.capabilities
            ]
            reliability = (
                sum(worker.reliability_score for worker in selected_capabilities)
                / len(selected_capabilities)
                if selected_capabilities
                else 0.0
            )
            participation = (
                len({_worker_node_id(worker) for worker in selected_capabilities}) / total_nodes
            )
            throughput_penalty = (
                1.0 - float(throughput) / maximum_throughput
                if throughput is not None and maximum_throughput > 0
                else 1.0
            )
            components = {
                "throughput_penalty": throughput_penalty,
                "memory_headroom_penalty": 1.0 / (1.0 + minimum_headroom),
                "reliability_penalty": 1.0 - reliability,
                "participation_penalty": 1.0 - participation,
            }
            if request.mode == "capacity":
                score = (
                    -minimum_headroom * 1_000_000
                    - sum(report.per_stage_headroom_bytes.values())
                    + float(report.expected_critical_path_ms or 0.0)
                )
            elif request.mode == "balanced":
                score = sum(
                    weight * component
                    for weight, component in zip(
                        self.balanced_weights,
                        components.values(),
                        strict=True,
                    )
                )
            else:
                score = -float(throughput) if throughput is not None else float("inf")
            update: dict[str, object] = {
                "objective_mode": request.mode,
                "objective_score": score,
                "local_baseline_throughput_tokens_s": local_baseline,
                "distributed_expected_throughput_tokens_s": distributed_throughput,
                "throughput_delta_tokens_s": delta,
                "objective_components": components,
            }
            if (
                request.mode == "speed"
                and not explicit
                and report.feasible
                and report.stage_count > 1
                and local_baseline is not None
                and throughput is not None
                and float(throughput) <= local_baseline
                and not required_nodes
                and not request.require_distributed
            ):
                update["feasible"] = False
                update["rejection_reasons"] = [
                    *report.rejection_reasons,
                    "distributed throughput does not exceed the fastest local baseline",
                ]
            enriched.append(
                _Candidate(
                    report=report.model_copy(update=update),
                    assignments=candidate.assignments,
                )
            )
        candidates = enriched
        feasible = [candidate for candidate in candidates if candidate.report.feasible]
        if not feasible:
            reasons = "; ".join(
                f"{candidate.report.name}: "
                f"{', '.join(candidate.report.rejection_reasons) or 'not feasible'}"
                for candidate in candidates
            )
            raise RuntimeError(f"no valid product topology candidate; {reasons}")

        selectable = feasible
        if not explicit:
            selectable = [
                candidate
                for candidate in feasible
                if candidate.report.expected_utility_tokens_s is not None
            ]
        if not selectable:
            raise RuntimeError(
                "automatic planning has no candidate with complete measured compute and "
                "directed network inputs"
            )
        selected = min(
            selectable,
            key=lambda candidate: (
                float(candidate.report.objective_score or 0.0),
                candidate.report.stage_count,
                tuple(candidate.report.worker_ids),
                candidate.report.name,
            ),
        )
        reason = (
            "selected the deterministic lowest objective score for the explicit constraints"
            if explicit
            else "selected the feasible candidate with the highest measured expected token "
            "utility after compute, active-load, reliability, memory, and directed network waits"
            if request.mode == "speed"
            else f"selected the deterministic {request.mode} objective optimum"
        )

        reports = [
            candidate.report.model_copy(
                update={"selected": candidate.report.name == selected.report.name}
            )
            for candidate in candidates
        ]
        selected_report = next(item for item in reports if item.selected)
        selected_worker_ids = set(selected_report.worker_ids)
        utility_rows = [
            NodeUtilityReport(
                node_id=_worker_node_id(worker),
                worker_id=worker.worker_id,
                included=worker.worker_id in selected_worker_ids,
                reason=(
                    "selected by the objective"
                    if worker.worker_id in selected_worker_ids
                    else selected_report.excluded_workers.get(
                        worker.worker_id,
                        "healthy but expected utility did not justify participation",
                    )
                ),
                expected_throughput_delta_tokens_s=(
                    selected_report.throughput_delta_tokens_s
                    if worker.worker_id in selected_worker_ids
                    else None
                ),
                memory_headroom_bytes=next(
                    (
                        value
                        for key, value in selected_report.per_stage_headroom_bytes.items()
                        if key.endswith(f"@{worker.worker_id}")
                    ),
                    None,
                ),
                reliability_score=worker.reliability_score,
            )
            for worker in all_workers
        ]
        plan_report = StagePlanReport(
            selected_topology=selected_report.topology,
            rejected_candidates=[
                (
                    f"{item.name}: {', '.join(item.rejection_reasons)}"
                    if not item.feasible
                    else f"{item.name}: lower measured utility than {selected_report.name}"
                )
                for item in reports
                if not item.selected
            ],
            worker_assignments=list(selected.assignments),
            memory_estimates_bytes=dict(selected_report.memory_estimates_bytes),
            compute_estimates_ms=dict(selected_report.compute_estimates_ms),
            network_estimates_ms=dict(selected_report.network_estimates_ms),
            expected_critical_path_waits_ms=dict(selected_report.expected_critical_path_waits_ms),
            reason_for_selection=reason,
            candidates=reports,
            worker_eligibility=list(inspected.eligibility),
            objective_mode=request.mode,
            local_baseline_throughput_tokens_s=local_baseline,
            distributed_expected_throughput_tokens_s=(
                selected_report.distributed_expected_throughput_tokens_s
            ),
            throughput_delta_tokens_s=selected_report.throughput_delta_tokens_s,
            directed_links_selected=selected_report.directed_links_selected,
            excluded_workers=selected_report.excluded_workers,
            per_stage_headroom_bytes=selected_report.per_stage_headroom_bytes,
            search_method=selected_report.search_method,
            beam_width=self.beam_width,
            measurement_freshness=selected_report.measurement_freshness,
            unmeasured_assumptions=selected_report.unmeasured_assumptions,
            confidence=selected_report.confidence,
            objective_components=selected_report.objective_components,
            node_utility=utility_rows,
        )
        expert_plans, final_assignments = self._plan_experts(
            request=request,
            inspected=inspected,
            assignments=selected.assignments,
            expert_workers=expert_workers,
            use_dense_memory_floor=use_dense_memory_floor,
        )
        plan_report.worker_assignments = list(final_assignments)
        plan_report.memory_estimates_bytes = {
            f"stage-{item.stage_id}@{item.worker_id}": item.required_memory_bytes
            for item in final_assignments
        }
        identity = {
            "model": inspected.spec.model_dump(mode="json"),
            "partition_method": selected_report.partition_method,
            "max_sequence_tokens": request.max_sequence_tokens,
            "assignments": [item.model_dump(mode="json") for item in final_assignments],
            "expert_plans": [item.model_dump(mode="json") for item in expert_plans],
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return ProductStagePlan(
            plan_id=f"plan-{digest[:20]}",
            topology_id=f"topology-{digest[:20]}",
            generation=1,
            created_monotonic_ns=time.monotonic_ns(),
            model=inspected.spec,
            stage_count=selected_report.stage_count,
            partition_method=selected_report.partition_method,
            max_sequence_tokens=request.max_sequence_tokens,
            assignments=list(final_assignments),
            expert_plans=list(expert_plans),
            expert_model_fingerprint=inspected.metadata.model_fingerprint,
            expert_quantization_fingerprint=inspected.metadata.quantization_fingerprint,
            report=plan_report,
        )

    def _evaluate_candidate(
        self,
        *,
        name: str,
        stage_count: int,
        method: PartitionMethod,
        request: ModelPlanRequest,
        inspected: InspectedProductModel,
        workers: list[WorkerCapability],
        explicit: bool,
        use_dense_memory_floor: bool,
        required_nodes: set[str],
        links: dict[tuple[str, str], NetworkLinkMeasurement],
        base_excluded_workers: dict[str, str],
    ) -> _Candidate:
        topology = name
        rejections: list[str] = []
        if request.stage_count is not None and request.stage_count != stage_count:
            rejections.append(
                f"stage-count override requires {request.stage_count}, candidate has {stage_count}"
            )
        if request.partition_method != "auto" and request.partition_method != method:
            rejections.append(
                f"partition override requires {request.partition_method}, candidate is {method}"
            )
        if request.require_distributed and stage_count == 1:
            rejections.append("request requires distributed stage ownership")
        if stage_count > len(workers):
            rejections.append(
                f"{stage_count}-stage ring requires {stage_count} eligible workers; "
                f"only {len(workers)} are available"
            )
        if stage_count > inspected.spec.layer_count:
            rejections.append("stage count exceeds the model layer count")
        if stage_count > self.maximum_stage_count:
            rejections.append("stage count exceeds the configured maximum")
        if len(required_nodes) > stage_count:
            rejections.append("stage count cannot include every required node")
        if not any(worker.effective_memory_bytes > 0 for worker in workers):
            rejections.append("no worker advertises positive effective memory")
        if rejections:
            return _Candidate(
                report=PlanCandidateReport(
                    name=name,
                    topology=topology,
                    stage_count=stage_count,
                    partition_method=method,
                    feasible=False,
                    rejection_reasons=rejections,
                    objective_mode=request.mode,
                    beam_width=self.beam_width,
                    excluded_workers=dict(base_excluded_workers),
                ),
                assignments=(),
            )

        planning_metadata = inspected.metadata.to_partition_metadata(
            model_revision=inspected.spec.model_revision,
            tokenizer_revision=inspected.spec.tokenizer_revision,
        )
        if use_dense_memory_floor:
            planning_metadata = replace(
                planning_metadata,
                layer_costs=tuple(
                    replace(
                        cost,
                        weight_bytes=cost.weight_bytes - cost.expert_weight_bytes,
                    )
                    for cost in planning_metadata.layer_costs
                ),
            )

        layer_total = sum(cost.execution_ns for cost in inspected.metadata.layer_costs)
        mean_layer_ns = layer_total / max(1, len(inspected.metadata.layer_costs))
        combination_reasons: set[str] = set()
        equal_boundaries = equal_ranges(inspected.spec.layer_count, stage_count)
        beam = [
            _SearchState(
                next_layer=0,
                selected_worker_ids=(),
                assignments=(),
                compute={},
                memory={},
                waits={},
                network={},
                links=(),
                unmeasured_assumptions=(),
                total_ms=0.0,
                headroom_ratios=(),
                headroom_bytes=(),
                reliability_scores=(),
            )
        ]
        worker_by_id = {worker.worker_id: worker for worker in workers}
        for stage_id in range(stage_count):
            expanded: list[_SearchState] = []
            remaining_stages = stage_count - stage_id - 1
            for state in beam:
                ends: tuple[int, ...]
                if method == "equal":
                    expected_start, expected_end = equal_boundaries[stage_id]
                    if state.next_layer != expected_start:
                        continue
                    ends = (expected_end,)
                elif remaining_stages == 0:
                    ends = (inspected.spec.layer_count,)
                else:
                    maximum_end = inspected.spec.layer_count - remaining_stages
                    ends = tuple(range(state.next_layer + 1, maximum_end + 1))
                selected_nodes = {
                    _worker_node_id(worker_by_id[worker_id])
                    for worker_id in state.selected_worker_ids
                }
                missing_nodes = required_nodes - selected_nodes
                if len(missing_nodes) > stage_count - stage_id:
                    continue
                for end in ends:
                    for worker in workers:
                        if worker.worker_id in state.selected_worker_ids:
                            continue
                        assignment = _assignment_for_range(
                            planning_metadata,
                            stage_id=stage_id,
                            stage_count=stage_count,
                            start=state.next_layer,
                            end=end,
                            device=_normalised_device(worker),
                        )
                        required = (
                            assignment.weight_bytes
                            + assignment.peak_temporary_bytes
                            + assignment.kv_cache_bytes_per_token * request.max_sequence_tokens
                        )
                        key = f"stage-{stage_id}@{worker.worker_id}"
                        if required > worker.effective_memory_bytes:
                            combination_reasons.add(
                                f"{key} requires {required} bytes but worker has "
                                f"{worker.effective_memory_bytes}"
                            )
                            continue
                        compute_ms = _measured_compute_ms(
                            worker,
                            assignment,
                            mean_layer_execution_ns=mean_layer_ns,
                        )
                        if compute_ms is None:
                            combination_reasons.add(f"{key} has no measured compute estimate")
                            continue
                        queue_ms = (
                            (worker.current_queue_depth + worker.active_session_count)
                            * compute_ms
                            / max(1, worker.max_concurrent_stage_operations)
                        )
                        reliability_ms = (1.0 - worker.reliability_score) * compute_ms
                        added_network_ms = 0.0
                        added_links: tuple[DirectedLinkSelection, ...] = ()
                        assumptions = state.unmeasured_assumptions
                        network = dict(state.network)
                        waits = dict(state.waits)
                        if stage_id > 0:
                            source_id = state.selected_worker_ids[-1]
                            boundary = f"stage-{stage_id - 1}-to-{stage_id}"
                            measurement = links.get((source_id, worker.worker_id))
                            fresh = bool(
                                measurement
                                and measurement.measured
                                and measurement.upload_bytes_per_s > 0
                                and network_measurement_is_fresh(
                                    measurement,
                                    ttl_seconds=self.network_measurement_ttl_seconds,
                                    now_unix_ns=self.clock_ns(),
                                )
                            )
                            if not fresh:
                                assumption = (
                                    f"{source_id}->{worker.worker_id} has no fresh directed "
                                    "measurement"
                                )
                                if not (
                                    explicit and self.allow_unmeasured_links_for_explicit_plans
                                ):
                                    combination_reasons.add(assumption)
                                    continue
                                assumptions = (*assumptions, assumption)
                                waits[f"{boundary}-unmeasured"] = 0.0
                                added_links = (
                                    DirectedLinkSelection(
                                        source_worker_id=source_id,
                                        destination_worker_id=worker.worker_id,
                                        latency_ms=0,
                                        transfer_ms=0,
                                        fresh=False,
                                        measured=False,
                                        assumption=assumption,
                                    ),
                                )
                            else:
                                assert measurement is not None
                                latency_ms = float(
                                    measurement.one_way_estimate_ms
                                    if measurement.one_way_estimate_ms is not None
                                    else measurement.round_trip_latency_ms / 2
                                )
                                source_assignment = state.assignments[-1].assignment
                                transfer_ms = (
                                    source_assignment.activation_bytes
                                    / measurement.upload_bytes_per_s
                                    * 1000
                                )
                                added_network_ms = latency_ms + transfer_ms
                                network[boundary] = added_network_ms
                                waits[f"{boundary}-transfer"] = transfer_ms
                                waits[f"{boundary}-latency"] = latency_ms
                                added_links = (
                                    DirectedLinkSelection(
                                        source_worker_id=source_id,
                                        destination_worker_id=worker.worker_id,
                                        measured_at_unix_ns=measurement.measured_at_unix_ns,
                                        latency_ms=latency_ms,
                                        transfer_ms=transfer_ms,
                                        upload_bytes_per_s=measurement.upload_bytes_per_s,
                                        fresh=True,
                                        measured=True,
                                        source_endpoint=measurement.source_endpoint,
                                        destination_endpoint=measurement.destination_endpoint,
                                    ),
                                )
                        compute = dict(state.compute)
                        compute[f"stage-{stage_id}"] = compute_ms
                        waits[f"stage-{stage_id}-queue"] = queue_ms
                        waits[f"stage-{stage_id}-reliability"] = reliability_ms
                        memory = dict(state.memory)
                        memory[key] = required
                        headroom = worker.effective_memory_bytes - required
                        assignment_record = PlanWorkerAssignment(
                            stage_id=stage_id,
                            worker_id=worker.worker_id,
                            control_endpoint=worker.control_endpoint or worker.endpoint or "",
                            data_endpoint=worker.data_plane_endpoint or "",
                            device=assignment.device,
                            effective_memory_bytes=worker.effective_memory_bytes,
                            required_memory_bytes=required,
                            assignment=assignment,
                        )
                        expanded.append(
                            _SearchState(
                                next_layer=end,
                                selected_worker_ids=(
                                    *state.selected_worker_ids,
                                    worker.worker_id,
                                ),
                                assignments=(*state.assignments, assignment_record),
                                compute=compute,
                                memory=memory,
                                waits=waits,
                                network=network,
                                links=(*state.links, *added_links),
                                unmeasured_assumptions=assumptions,
                                total_ms=(
                                    state.total_ms
                                    + compute_ms
                                    + queue_ms
                                    + reliability_ms
                                    + added_network_ms
                                ),
                                headroom_ratios=(
                                    *state.headroom_ratios,
                                    headroom / worker.effective_memory_bytes,
                                ),
                                headroom_bytes=(*state.headroom_bytes, headroom),
                                reliability_scores=(
                                    *state.reliability_scores,
                                    worker.reliability_score,
                                ),
                            )
                        )
            expanded.sort(key=lambda state: self._beam_key(state, request.mode, method))
            beam = expanded[: self.beam_width]
            if not beam:
                break
        complete = [
            state
            for state in beam
            if state.next_layer == inspected.spec.layer_count
            and required_nodes
            <= {_worker_node_id(worker_by_id[worker_id]) for worker_id in state.selected_worker_ids}
        ]
        if not complete:
            return _Candidate(
                report=PlanCandidateReport(
                    name=name,
                    topology=topology,
                    stage_count=stage_count,
                    partition_method=method,
                    feasible=False,
                    rejection_reasons=sorted(combination_reasons)
                    or ["no unique memory-compatible worker assignment"],
                    objective_mode=request.mode,
                    beam_width=self.beam_width,
                    excluded_workers=dict(base_excluded_workers),
                ),
                assignments=(),
            )
        best = min(
            complete,
            key=lambda state: self._beam_key(state, request.mode, method),
        )
        complete_network = not best.unmeasured_assumptions
        utility = (
            1000.0 / best.total_ms
            if best.total_ms > 0 and (stage_count == 1 or complete_network)
            else None
        )
        selected_ids = set(best.selected_worker_ids)
        candidate_excluded = {
            **base_excluded_workers,
            **{
                worker.worker_id: "not selected by bounded objective search"
                for worker in workers
                if worker.worker_id not in selected_ids
            },
        }
        per_stage_headroom = {
            f"stage-{assignment.stage_id}@{assignment.worker_id}": (
                assignment.effective_memory_bytes - assignment.required_memory_bytes
            )
            for assignment in best.assignments
        }
        freshness = {
            f"{link.source_worker_id}->{link.destination_worker_id}": (
                "fresh" if link.fresh else "unmeasured-explicit-assumption"
            )
            for link in best.links
        }
        confidence = (
            "measured"
            if not best.unmeasured_assumptions
            else "unmeasured"
            if len(best.unmeasured_assumptions) == len(best.links)
            else "mixed"
        )
        raw_objective_score = self._beam_key(best, request.mode, method)[0]
        if not isinstance(raw_objective_score, (int, float)):
            raise TypeError("planner objective score is not numeric")
        report = PlanCandidateReport(
            name=name,
            topology=topology,
            stage_count=stage_count,
            partition_method=method,
            feasible=True,
            worker_ids=list(best.selected_worker_ids),
            memory_estimates_bytes=best.memory,
            compute_estimates_ms=best.compute,
            network_estimates_ms=best.network,
            expected_critical_path_waits_ms=best.waits,
            expected_critical_path_ms=best.total_ms,
            expected_utility_tokens_s=utility,
            objective_mode=request.mode,
            objective_score=float(raw_objective_score),
            directed_links_selected=list(best.links),
            excluded_workers=candidate_excluded,
            per_stage_headroom_bytes=per_stage_headroom,
            beam_width=self.beam_width,
            measurement_freshness=freshness,
            unmeasured_assumptions=list(best.unmeasured_assumptions),
            confidence=confidence,
        )
        return _Candidate(report=report, assignments=best.assignments)

    @staticmethod
    def _service_ms(worker: WorkerCapability, operation: str, fallback: float) -> float:
        rates = worker.measured_expert_service_rates
        explicit_ms = rates.get(f"{operation}_ms")
        if explicit_ms is not None and explicit_ms > 0:
            return float(explicit_ms)
        per_second = rates.get(f"{operation}_calls_per_second") or rates.get(
            "expert_calls_per_second"
        )
        if per_second is not None and per_second > 0:
            return 1000.0 / float(per_second)
        return fallback

    @staticmethod
    def _network_ms(
        source: WorkerCapability,
        destinations: list[WorkerCapability],
        activation_bytes: int,
    ) -> float:
        total = 0.0
        for destination in destinations:
            outbound = min(
                source.upload_bandwidth_bytes_s,
                destination.download_bandwidth_bytes_s,
            )
            inbound = min(
                destination.upload_bandwidth_bytes_s,
                source.download_bandwidth_bytes_s,
            )
            if outbound <= 0 or inbound <= 0:
                return float("inf")
            total += activation_bytes / outbound * 1000
            total += activation_bytes / inbound * 1000
        return total

    def _plan_experts(
        self,
        *,
        request: ModelPlanRequest,
        inspected: InspectedProductModel,
        assignments: tuple[PlanWorkerAssignment, ...],
        expert_workers: list[WorkerCapability],
        use_dense_memory_floor: bool,
    ) -> tuple[tuple[ProductStageExpertPlan, ...], tuple[PlanWorkerAssignment, ...]]:
        metadata = inspected.metadata
        if metadata.expert_count <= 0:
            if request.require_remote_experts:
                raise RuntimeError("forced-remote expert validation requires MoE metadata")
            return (), assignments
        utility_planner = ExpertUtilityPlanner()
        product_plans: list[ProductStageExpertPlan] = []
        final_assignments: list[PlanWorkerAssignment] = []
        for stage_assignment in assignments:
            stage_worker = inspected.capabilities[stage_assignment.worker_id]
            base_required = stage_assignment.required_memory_bytes
            local_reserved = 0
            placements: list[ProductExpertPlacement] = []
            for layer_id in stage_assignment.assignment.layer_ids:
                layer_cost = metadata.layer_costs[layer_id]
                expert_bytes = (
                    layer_cost.expert_weight_bytes // metadata.expert_count
                    if metadata.expert_count
                    else 0
                )
                local_ms = max(
                    layer_cost.expert_execution_ns / max(metadata.expert_count, 1) / 1e6,
                    1e-9,
                )
                for expert_id in range(metadata.expert_count):
                    candidates = [
                        ExpertStrategyCandidate(
                            candidate_id=f"local:{stage_worker.worker_id}",
                            strategy=ExpertStrategy.LOCAL,
                            memory_required_bytes=(
                                base_required + local_reserved + expert_bytes
                                if use_dense_memory_floor
                                else base_required
                            ),
                            memory_available_bytes=stage_worker.effective_memory_bytes,
                            explanation=["expert remains resident in its contiguous stage"],
                        )
                    ]
                    whole_owners = [
                        worker
                        for worker in expert_workers
                        if WorkerRole.WHOLE_EXPERT in worker.roles
                        if expert_id in worker.owned_experts.get(str(layer_id), [])
                    ]
                    for owner in whole_owners:
                        content_hash = owner.expert_content_hashes.get(
                            f"{layer_id}:{expert_id}", ""
                        )
                        remote_ms = self._service_ms(owner, "whole_expert", local_ms)
                        network_ms = self._network_ms(
                            stage_worker, [owner], stage_assignment.assignment.activation_bytes
                        )
                        hits = owner.expert_cache_hits
                        misses = owner.expert_cache_misses
                        cache_hit_rate = hits / (hits + misses) if hits + misses else 0.0
                        identity_matches = (
                            not metadata.model_fingerprint
                            or owner.model_fingerprint == metadata.model_fingerprint
                        )
                        quantization_matches = (
                            not metadata.quantization_fingerprint
                            or owner.quantisation_fingerprint == metadata.quantization_fingerprint
                        )
                        exact_transport = (
                            TransportCodec.RAW_FP32.value in owner.supported_expert_codecs
                            and ReductionMode.FIXED_ORDER_FP32.value
                            in owner.supported_reduction_modes
                        )
                        candidates.append(
                            ExpertStrategyCandidate(
                                candidate_id=f"whole:{owner.worker_id}",
                                strategy=ExpertStrategy.WHOLE_REMOTE,
                                worker_ids=[owner.worker_id],
                                utility=ExpertUtilityInputs(
                                    measured_local_expert_ms=local_ms,
                                    measured_remote_expert_ms=remote_ms,
                                    serialization_ms=max(
                                        0.0,
                                        stage_assignment.assignment.activation_bytes
                                        / max(
                                            stage_worker.measured_memory_bandwidth_bytes_s or 1,
                                            1,
                                        )
                                        * 2000,
                                    ),
                                    network_transfer_ms=network_ms,
                                    queue_delay_ms=owner.current_queue_depth * remote_ms,
                                    reduction_ms=0.0,
                                    cache_hit_rate=cache_hit_rate,
                                    cache_miss_cost_ms=remote_ms,
                                    memory_pressure_cost_ms=(
                                        expert_bytes
                                        / max(owner.expert_memory_budget_bytes or 1, 1)
                                        * remote_ms
                                    ),
                                    memory_relief_value_ms=(
                                        local_ms
                                        if base_required + expert_bytes
                                        > stage_worker.effective_memory_bytes
                                        else 0
                                    ),
                                    failure_risk=1.0 - owner.reliability_score,
                                    expected_fallback_cost_ms=local_ms,
                                ),
                                feasible=(
                                    network_ms != float("inf")
                                    and owner.expert_data_plane_endpoint is not None
                                ),
                                correctness_validated=(
                                    content_hash.startswith("sha256:") and exact_transport
                                ),
                                model_identity_matches=identity_matches,
                                quantization_identity_matches=quantization_matches,
                                memory_required_bytes=expert_bytes,
                                memory_available_bytes=owner.expert_memory_budget_bytes or 0,
                                worker_memory_required_bytes={owner.worker_id: expert_bytes},
                                worker_memory_available_bytes={
                                    owner.worker_id: owner.expert_memory_budget_bytes or 0
                                },
                                explanation=["worker advertises physical whole-expert ownership"],
                            )
                        )
                    shard_entries = [
                        (worker, shard)
                        for worker in expert_workers
                        if WorkerRole.EXPERT_MICROSHARD in worker.roles
                        for shard in worker.owned_microshards
                        if int(shard.get("layer_id", -1)) == layer_id
                        and int(shard.get("expert_id", -1)) == expert_id
                    ]
                    ordered_shards = sorted(
                        shard_entries,
                        key=lambda item: (
                            int(item[1].get("hidden_start", -1)),
                            item[0].worker_id,
                        ),
                    )
                    if ordered_shards:
                        cursor = 0
                        owner_ids = [item[0].worker_id for item in ordered_shards]
                        valid_union = (
                            len(owner_ids) > 1
                            and len(set(owner_ids)) == len(owner_ids)
                            and all(
                                TransportCodec.RAW_FP32.value in owner.supported_expert_codecs
                                and ReductionMode.FIXED_ORDER_FP32.value
                                in owner.supported_reduction_modes
                                for owner, _ in ordered_shards
                            )
                        )
                        for shard_owner, shard in ordered_shards:
                            start = int(shard.get("hidden_start", -1))
                            end = int(shard.get("hidden_end", -1))
                            logical_width = int(shard.get("logical_intermediate_dimension", -1))
                            if (
                                start != cursor
                                or end <= start
                                or end > metadata.expert_intermediate_size
                                or logical_width != metadata.expert_intermediate_size
                            ):
                                valid_union = False
                            group_size = int(shard.get("quantization_group_size", 0) or 0)
                            if group_size and (
                                start % group_size
                                or (end != metadata.expert_intermediate_size and end % group_size)
                            ):
                                valid_union = False
                            content_hash = str(
                                shard.get("content_hash")
                                or shard_owner.expert_content_hashes.get(
                                    f"{layer_id}:{expert_id}", ""
                                )
                            )
                            if not content_hash.startswith("sha256:"):
                                valid_union = False
                            cursor = max(cursor, end)
                        valid_union = valid_union and cursor == metadata.expert_intermediate_size
                        owners = [item[0] for item in ordered_shards]
                        remote_ms = max(
                            self._service_ms(owner, "microshard", local_ms) for owner in owners
                        )
                        network_ms = self._network_ms(
                            stage_worker,
                            owners,
                            stage_assignment.assignment.activation_bytes,
                        )
                        cache_hits = sum(owner.expert_cache_hits for owner in owners)
                        cache_misses = sum(owner.expert_cache_misses for owner in owners)
                        cache_hit_rate = (
                            cache_hits / (cache_hits + cache_misses)
                            if cache_hits + cache_misses
                            else 0.0
                        )
                        shard_required_bytes = [
                            (expert_bytes * (int(shard["hidden_end"]) - int(shard["hidden_start"])))
                            // max(metadata.expert_intermediate_size, 1)
                            for _, shard in ordered_shards
                        ]
                        candidates.append(
                            ExpertStrategyCandidate(
                                candidate_id="microshard:"
                                + ",".join(item.worker_id for item in owners),
                                strategy=ExpertStrategy.MICROSHARD_REMOTE,
                                worker_ids=[item.worker_id for item in owners],
                                utility=ExpertUtilityInputs(
                                    measured_local_expert_ms=local_ms,
                                    measured_remote_expert_ms=remote_ms,
                                    serialization_ms=max(
                                        0.0,
                                        stage_assignment.assignment.activation_bytes
                                        / max(
                                            stage_worker.measured_memory_bandwidth_bytes_s or 1,
                                            1,
                                        )
                                        * 2000,
                                    ),
                                    network_transfer_ms=network_ms,
                                    queue_delay_ms=max(
                                        owner.current_queue_depth * remote_ms for owner in owners
                                    ),
                                    reduction_ms=max(
                                        self._service_ms(owner, "reduction", 0.01)
                                        for owner in owners
                                    ),
                                    cache_hit_rate=cache_hit_rate,
                                    cache_miss_cost_ms=remote_ms,
                                    memory_pressure_cost_ms=max(
                                        required
                                        / max(owner.expert_memory_budget_bytes or 1, 1)
                                        * remote_ms
                                        for owner, required in zip(
                                            owners, shard_required_bytes, strict=True
                                        )
                                    ),
                                    memory_relief_value_ms=(
                                        local_ms
                                        if base_required + expert_bytes
                                        > stage_worker.effective_memory_bytes
                                        else 0
                                    ),
                                    failure_risk=max(
                                        1.0 - owner.reliability_score for owner in owners
                                    ),
                                    expected_fallback_cost_ms=local_ms,
                                ),
                                feasible=(
                                    valid_union
                                    and network_ms != float("inf")
                                    and all(
                                        owner.expert_data_plane_endpoint is not None
                                        for owner in owners
                                    )
                                ),
                                correctness_validated=valid_union,
                                model_identity_matches=all(
                                    not metadata.model_fingerprint
                                    or owner.model_fingerprint == metadata.model_fingerprint
                                    for owner in owners
                                ),
                                quantization_identity_matches=all(
                                    not metadata.quantization_fingerprint
                                    or owner.quantisation_fingerprint
                                    == metadata.quantization_fingerprint
                                    for owner in owners
                                ),
                                memory_required_bytes=sum(shard_required_bytes),
                                memory_available_bytes=sum(
                                    owner.expert_memory_budget_bytes or 0 for owner in owners
                                ),
                                worker_memory_required_bytes={
                                    owner.worker_id: required
                                    for owner, required in zip(
                                        owners, shard_required_bytes, strict=True
                                    )
                                },
                                worker_memory_available_bytes={
                                    owner.worker_id: owner.expert_memory_budget_bytes or 0
                                    for owner in owners
                                },
                                explanation=[
                                    "workers advertise a gap-free matched native microshard union"
                                ],
                            )
                        )
                    decision = utility_planner.choose(
                        stage_id=stage_assignment.stage_id,
                        layer_id=layer_id,
                        expert_id=expert_id,
                        candidates=candidates,
                        policy=request.expert_policy,
                        require_remote=request.require_remote_experts,
                        allow_local_fallback=request.allow_expert_local_fallback,
                    )
                    if (
                        decision.selected_strategy == ExpertStrategy.LOCAL
                        or decision.local_fallback_permitted
                    ) and use_dense_memory_floor:
                        local_reserved += expert_bytes
                    selected_workers = {worker.worker_id: worker for worker in expert_workers}
                    selected_shards = (
                        [
                            {
                                **dict(shard),
                                "worker_id": worker.worker_id,
                                "logical_intermediate_dimension": metadata.expert_intermediate_size,
                                "content_hash": str(
                                    shard.get("content_hash")
                                    or worker.expert_content_hashes.get(
                                        f"{layer_id}:{expert_id}", ""
                                    )
                                ),
                            }
                            for worker, shard in ordered_shards
                            if worker.worker_id in decision.selected_workers
                        ]
                        if decision.selected_strategy == ExpertStrategy.MICROSHARD_REMOTE
                        else []
                    )
                    placements.append(
                        ProductExpertPlacement(
                            layer_id=layer_id,
                            expert_id=expert_id,
                            strategy=decision.selected_strategy.value,
                            worker_ids=decision.selected_workers,
                            worker_endpoints={
                                worker_id: selected_workers[worker_id].expert_data_plane_endpoint
                                or ""
                                for worker_id in decision.selected_workers
                            },
                            expert_hashes={
                                worker_id: selected_workers[worker_id].expert_content_hashes.get(
                                    f"{layer_id}:{expert_id}", ""
                                )
                                for worker_id in decision.selected_workers
                            },
                            microshards=selected_shards,
                            measured_utility_ms=decision.measured_utility_ms,
                            capacity_required=decision.capacity_required,
                            local_fallback_permitted=decision.local_fallback_permitted,
                            forced_remote=decision.forced_remote,
                            explanation=decision.explanation,
                            rejected=[item.model_dump(mode="json") for item in decision.rejected],
                        )
                    )
            final_required = base_required + local_reserved
            updated_assignment = replace(
                stage_assignment.assignment,
                weight_bytes=stage_assignment.assignment.weight_bytes + local_reserved,
            )
            final_assignments.append(
                stage_assignment.model_copy(
                    update={
                        "assignment": updated_assignment,
                        "required_memory_bytes": final_required,
                    }
                )
            )
            product_plans.append(
                ProductStageExpertPlan(
                    stage_id=stage_assignment.stage_id,
                    policy=request.expert_policy,
                    require_remote_experts=request.require_remote_experts,
                    placements=placements,
                )
            )
        if request.require_remote_experts and not any(
            placement.strategy != ExpertStrategy.LOCAL.value
            for plan in product_plans
            for placement in plan.placements
        ):
            raise RuntimeError("forced-remote expert planning selected no remote expert")
        return tuple(product_plans), tuple(final_assignments)


__all__ = ["ProductStagePlanner"]
