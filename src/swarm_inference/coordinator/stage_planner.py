"""Measured, memory-aware contiguous product stage planning."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

from swarm_inference.config.models import OperationKind, WorkerCapability, WorkerRole
from swarm_inference.coordinator.expert_planner import (
    ExpertPolicy,
    ExpertStrategy,
    ExpertStrategyCandidate,
    ExpertUtilityInputs,
    ExpertUtilityPlanner,
)
from swarm_inference.coordinator.model_catalog import InspectedProductModel
from swarm_inference.model.partition import PartitionMethod, StageAssignment, build_stage_plan
from swarm_inference.protocol.expert import ReductionMode, TransportCodec
from swarm_inference.protocol.product import (
    ModelPlanRequest,
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


class ProductStagePlanner:
    """Evaluate local, equal-ring, and balanced-ring candidates from measurements."""

    def build_plan(
        self,
        request: ModelPlanRequest,
        inspected: InspectedProductModel,
    ) -> ProductStagePlan:
        workers = sorted(
            inspected.capabilities.values(),
            key=lambda item: (
                item.active_session_count,
                item.current_queue_depth,
                item.worker_id,
            ),
        )[:16]
        maximum_memory = max(
            (worker.effective_memory_bytes for worker in workers),
            default=0,
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
        definitions: tuple[tuple[str, int, PartitionMethod], ...] = (
            ("local-monolithic", 1, "equal"),
            ("two-stage-equal-ring", 2, "equal"),
            ("two-stage-balanced-ring", 2, "balanced"),
        )
        explicit = request.stage_count is not None or request.partition_method != "auto"
        use_dense_memory_floor = (
            request.expert_policy != ExpertPolicy.LOCAL.value
            and bool(expert_workers)
            and inspected.metadata.expert_count > 0
        )
        for name, stage_count, method in definitions:
            candidates.append(
                self._evaluate_candidate(
                    name=name,
                    stage_count=stage_count,
                    method=method,
                    request=request,
                    inspected=inspected,
                    workers=workers,
                    maximum_memory=maximum_memory,
                    explicit=explicit,
                    use_dense_memory_floor=use_dense_memory_floor,
                )
            )
        feasible = [candidate for candidate in candidates if candidate.report.feasible]
        if not feasible:
            reasons = "; ".join(
                f"{candidate.report.name}: "
                f"{', '.join(candidate.report.rejection_reasons) or 'not feasible'}"
                for candidate in candidates
            )
            raise RuntimeError(f"no valid product topology candidate; {reasons}")

        if explicit:
            selected = min(
                feasible,
                key=lambda candidate: (
                    candidate.report.expected_critical_path_ms
                    if candidate.report.expected_critical_path_ms is not None
                    else float("inf"),
                    candidate.report.stage_count,
                    candidate.report.name,
                ),
            )
            reason = (
                "selected the lowest measured critical path among candidates allowed by the "
                "explicit stage/partition override"
            )
        else:
            measured = [
                candidate
                for candidate in feasible
                if candidate.report.expected_utility_tokens_s is not None
            ]
            if not measured:
                raise RuntimeError(
                    "automatic planning has no candidate with complete measured compute and "
                    "network inputs"
                )
            selected = max(
                measured,
                key=lambda candidate: (
                    float(candidate.report.expected_utility_tokens_s or 0.0),
                    -float(candidate.report.expected_critical_path_ms or 0),
                    -candidate.report.stage_count,
                ),
            )
            reason = (
                "selected the feasible candidate with the highest measured expected token "
                "utility after compute, active-load, memory, and network waits"
            )

        reports = [
            candidate.report.model_copy(
                update={"selected": candidate.report.name == selected.report.name}
            )
            for candidate in candidates
        ]
        selected_report = next(item for item in reports if item.selected)
        report = StagePlanReport(
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
        )
        expert_plans, final_assignments = self._plan_experts(
            request=request,
            inspected=inspected,
            assignments=selected.assignments,
            expert_workers=expert_workers,
            use_dense_memory_floor=use_dense_memory_floor,
        )
        report.worker_assignments = list(final_assignments)
        report.memory_estimates_bytes = {
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
            report=report,
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
        maximum_memory: int,
        explicit: bool,
        use_dense_memory_floor: bool,
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
        if stage_count == 2 and len(workers) < 2:
            rejections.append("two-stage ring requires two eligible workers")
        if maximum_memory <= 0:
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
                ),
                assignments=(),
            )

        try:
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
            generic = build_stage_plan(
                Path("."),
                metadata=planning_metadata,
                stage_count=stage_count,
                method=method,
                memory_limit_bytes=maximum_memory,
                device="cpu",
            )
        except (MemoryError, ValueError) as exc:
            return _Candidate(
                report=PlanCandidateReport(
                    name=name,
                    topology=topology,
                    stage_count=stage_count,
                    partition_method=method,
                    feasible=False,
                    rejection_reasons=[str(exc)],
                ),
                assignments=(),
            )

        layer_total = sum(cost.execution_ns for cost in inspected.metadata.layer_costs)
        mean_layer_ns = layer_total / max(1, len(inspected.metadata.layer_costs))
        best: tuple[float, tuple[PlanWorkerAssignment, ...], PlanCandidateReport] | None = None
        combination_reasons: set[str] = set()
        for selected_workers in itertools.permutations(workers, stage_count):
            plan_assignments: list[PlanWorkerAssignment] = []
            compute: dict[str, float] = {}
            memory: dict[str, int] = {}
            waits: dict[str, float] = {}
            network: dict[str, float] = {}
            total_ms = 0.0
            complete_network = True
            valid = True
            for generic_assignment, worker in zip(
                generic.assignments, selected_workers, strict=True
            ):
                assignment = replace(
                    generic_assignment,
                    device=_normalised_device(worker),
                )
                required = (
                    assignment.weight_bytes
                    + assignment.peak_temporary_bytes
                    + assignment.kv_cache_bytes_per_token * request.max_sequence_tokens
                )
                key = f"stage-{assignment.stage_id}@{worker.worker_id}"
                memory[key] = required
                if required > worker.effective_memory_bytes:
                    combination_reasons.add(
                        f"{key} requires {required} bytes but worker has "
                        f"{worker.effective_memory_bytes}"
                    )
                    valid = False
                    break
                compute_ms = _measured_compute_ms(
                    worker,
                    assignment,
                    mean_layer_execution_ns=mean_layer_ns,
                )
                if compute_ms is None:
                    combination_reasons.add(f"{key} has no measured compute estimate")
                    valid = False
                    break
                queue_ms = (
                    worker.current_queue_depth
                    * compute_ms
                    / max(1, worker.max_concurrent_stage_operations)
                )
                compute[f"stage-{assignment.stage_id}"] = compute_ms
                waits[f"stage-{assignment.stage_id}-queue"] = queue_ms
                total_ms += compute_ms + queue_ms
                assert worker.control_endpoint or worker.endpoint
                assert worker.data_plane_endpoint
                plan_assignments.append(
                    PlanWorkerAssignment(
                        stage_id=assignment.stage_id,
                        worker_id=worker.worker_id,
                        control_endpoint=worker.control_endpoint or worker.endpoint or "",
                        data_endpoint=worker.data_plane_endpoint,
                        device=assignment.device,
                        effective_memory_bytes=worker.effective_memory_bytes,
                        required_memory_bytes=required,
                        assignment=assignment,
                    )
                )
            if not valid:
                continue
            for boundary in range(stage_count - 1):
                source = selected_workers[boundary]
                destination = selected_workers[boundary + 1]
                bandwidth = min(
                    source.upload_bandwidth_bytes_s,
                    destination.download_bandwidth_bytes_s,
                )
                boundary_key = f"stage-{boundary}-to-{boundary + 1}"
                latency_ms = source.coordinator_latency_ms + destination.coordinator_latency_ms
                if bandwidth <= 0:
                    complete_network = False
                    network[boundary_key] = latency_ms
                    waits[f"{boundary_key}-bandwidth-unmeasured"] = 0.0
                    total_ms += latency_ms
                else:
                    transfer_ms = generic.assignments[boundary].activation_bytes / bandwidth * 1000
                    network[boundary_key] = latency_ms + transfer_ms
                    waits[f"{boundary_key}-transfer"] = transfer_ms
                    waits[f"{boundary_key}-latency"] = latency_ms
                    total_ms += latency_ms + transfer_ms
            if stage_count > 1 and not complete_network and not explicit:
                combination_reasons.add(
                    "distributed candidate lacks measured advertised peer bandwidth"
                )
                continue
            utility = 1000.0 / total_ms if total_ms > 0 and complete_network else None
            report = PlanCandidateReport(
                name=name,
                topology=topology,
                stage_count=stage_count,
                partition_method=method,
                feasible=True,
                worker_ids=[item.worker_id for item in plan_assignments],
                memory_estimates_bytes=memory,
                compute_estimates_ms=compute,
                network_estimates_ms=network,
                expected_critical_path_waits_ms=waits,
                expected_critical_path_ms=total_ms,
                expected_utility_tokens_s=utility,
            )
            score = total_ms
            if best is None or score < best[0]:
                best = (score, tuple(plan_assignments), report)
        if best is None:
            return _Candidate(
                report=PlanCandidateReport(
                    name=name,
                    topology=topology,
                    stage_count=stage_count,
                    partition_method=method,
                    feasible=False,
                    rejection_reasons=sorted(combination_reasons)
                    or ["no unique memory-compatible worker assignment"],
                ),
                assignments=(),
            )
        return _Candidate(report=best[2], assignments=best[1])

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
            total += source.coordinator_latency_ms + destination.coordinator_latency_ms
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
