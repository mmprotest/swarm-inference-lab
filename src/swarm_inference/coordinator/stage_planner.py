"""Measured, memory-aware contiguous product stage planning."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

from swarm_inference.config.models import OperationKind, WorkerCapability
from swarm_inference.coordinator.model_catalog import InspectedProductModel
from swarm_inference.model.partition import PartitionMethod, StageAssignment, build_stage_plan
from swarm_inference.protocol.product import (
    ModelPlanRequest,
    PlanCandidateReport,
    PlanWorkerAssignment,
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
        definitions: tuple[tuple[str, int, PartitionMethod], ...] = (
            ("local-monolithic", 1, "equal"),
            ("two-stage-equal-ring", 2, "equal"),
            ("two-stage-balanced-ring", 2, "balanced"),
        )
        explicit = request.stage_count is not None or request.partition_method != "auto"
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
        identity = {
            "model": inspected.spec.model_dump(mode="json"),
            "partition_method": selected_report.partition_method,
            "max_sequence_tokens": request.max_sequence_tokens,
            "assignments": [item.model_dump(mode="json") for item in selected.assignments],
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
            assignments=list(selected.assignments),
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
            generic = build_stage_plan(
                Path("."),
                metadata=inspected.metadata.to_partition_metadata(
                    model_revision=inspected.spec.model_revision,
                    tokenizer_revision=inspected.spec.tokenizer_revision,
                ),
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


__all__ = ["ProductStagePlanner"]
