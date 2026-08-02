"""Expert dispatch, real-path recovery, and deterministic failure scheduling."""

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_010.schemas import (
    ExpertExecutionRequest,
    ExpertExecutionResponse,
    FailureType,
    RecoveryStrategy,
)
from swarm_inference.experiments.experiment_010.transport import ExpertTransportClient
from swarm_inference.experiments.experiment_010.verification import (
    TrustController,
    VerificationDecision,
)

LocalExecutor = Callable[
    [ExpertExecutionRequest, np.ndarray], tuple[ExpertExecutionResponse, np.ndarray]
]


@dataclass(frozen=True, slots=True)
class FailureEvent:
    event_id: str
    failure_type: FailureType
    worker_id: str
    token_index: int
    layer_id: int
    parameters: dict[str, Any]


@dataclass(slots=True)
class RecoveryMetrics:
    request_id: str
    recovery_strategy: str
    primary_worker: str
    selected_worker: str | None = None
    recovered: bool = False
    failed_explicitly: bool = False
    correctness: bool = False
    failure_detection_ns: int = 0
    recovery_latency_ns: int = 0
    recomputed_work: int = 0
    extra_bytes: int = 0
    extra_compute_ns: int = 0
    duplicate_results: int = 0
    lost_tokens: int = 0
    duplicate_tokens: int = 0
    planner_reaction_ns: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DispatchResult:
    response: ExpertExecutionResponse
    result: np.ndarray
    verification: VerificationDecision
    metrics: dict[str, Any]


class FailureController:
    def __init__(self, schedule: list[FailureEvent]) -> None:
        self.schedule = sorted(
            schedule, key=lambda item: (item.token_index, item.layer_id, item.event_id)
        )
        self.applied: set[str] = set()

    def apply_due(
        self,
        *,
        token_index: int,
        layer_id: int,
        clients: dict[str, ExpertTransportClient],
    ) -> list[dict[str, Any]]:
        applied = []
        for event in self.schedule:
            if (
                event.event_id in self.applied
                or event.token_index != token_index
                or event.layer_id != layer_id
            ):
                continue
            client = clients.get(event.worker_id)
            if client is None:
                applied.append(
                    {"event_id": event.event_id, "applied": False, "reason": "unknown worker"}
                )
                self.applied.add(event.event_id)
                continue
            payload = {
                "fault_type": event.failure_type.value,
                "remaining": int(event.parameters.get("remaining", 1)),
                "fixed_delay_ms": float(event.parameters.get("fixed_delay_ms", 0)),
                "random_delay_ms": float(event.parameters.get("random_delay_ms", 0)),
                "storage_slowdown_ms": float(event.parameters.get("storage_slowdown_ms", 0)),
            }
            client.control("configure_fault", **payload)
            self.applied.add(event.event_id)
            applied.append(
                {
                    "event_id": event.event_id,
                    "applied": True,
                    "worker_id": event.worker_id,
                    "failure_type": event.failure_type.value,
                    "timestamp_ns": time.time_ns(),
                }
            )
        return applied


class ExpertDispatcher:
    def __init__(
        self,
        clients: dict[str, ExpertTransportClient],
        *,
        trust: TrustController,
        local_executor: LocalExecutor | None = None,
    ) -> None:
        self.clients = clients
        self.trust = trust
        self.local_executor = local_executor

    def _remote(
        self,
        worker_id: str,
        request: ExpertExecutionRequest,
        activation: np.ndarray,
        *,
        reference: np.ndarray | None,
    ) -> DispatchResult:
        started = time.perf_counter_ns()
        response, result, transport = self.clients[worker_id].execute(request, activation)
        elapsed = time.perf_counter_ns() - started
        verification = self.trust.verify(
            request,
            response,
            result,
            reference=reference,
            latency_ns=elapsed,
            verification_network_bytes=int(result.nbytes) if reference is not None else 0,
        )
        if not verification.accepted:
            raise ValueError("worker verification failed: " + ", ".join(verification.reasons))
        return DispatchResult(
            response=response,
            result=result,
            verification=verification,
            metrics={"worker_id": worker_id, "transport": transport, "elapsed_ns": elapsed},
        )

    def execute(
        self,
        request: ExpertExecutionRequest,
        activation: np.ndarray,
        *,
        primary_worker: str,
        recovery_strategy: RecoveryStrategy | str,
        alternate_workers: tuple[str, ...] = (),
        reference: np.ndarray | None = None,
        hedge_delay_ms: float = 0.0,
    ) -> DispatchResult:
        strategy = RecoveryStrategy(recovery_strategy)
        metrics = RecoveryMetrics(
            request_id=request.request_id,
            recovery_strategy=strategy.value,
            primary_worker=primary_worker,
        )
        started = time.perf_counter_ns()
        try:
            if strategy == RecoveryStrategy.HEDGED_DUPLICATE:
                result = self._hedged(
                    request,
                    activation,
                    primary_worker=primary_worker,
                    alternate_workers=alternate_workers,
                    reference=reference,
                    hedge_delay_ms=hedge_delay_ms,
                    metrics=metrics,
                )
            elif strategy == RecoveryStrategy.SAMPLED_REPLICATION:
                result = self._replicated(
                    request,
                    activation,
                    primary_worker=primary_worker,
                    alternate_workers=alternate_workers,
                    reference=reference,
                    metrics=metrics,
                )
            else:
                result = self._remote(primary_worker, request, activation, reference=reference)
                metrics.selected_worker = primary_worker
                metrics.correctness = result.verification.accepted
            return DispatchResult(
                response=result.response,
                result=result.result,
                verification=result.verification,
                metrics={**asdict(metrics), **result.metrics},
            )
        except Exception as primary_error:
            detected = time.perf_counter_ns()
            metrics.failure_detection_ns = detected - started
            self.trust.timeout(primary_worker)
            metrics.error = f"{type(primary_error).__name__}: {primary_error}"
            if strategy in {
                RecoveryStrategy.TIMEOUT_ALTERNATE_WORKER,
                RecoveryStrategy.SMALL_TILE_WORK_STEALING,
            }:
                reaction_started = time.perf_counter_ns()
                for alternate in alternate_workers:
                    if not self.trust.eligible(alternate, minimum_confidence=0.0):
                        continue
                    metrics.planner_reaction_ns += time.perf_counter_ns() - reaction_started
                    try:
                        result = self._remote(alternate, request, activation, reference=reference)
                    except Exception:
                        self.trust.timeout(alternate)
                        continue
                    metrics.selected_worker = alternate
                    metrics.recovered = True
                    metrics.correctness = result.verification.accepted
                    metrics.recomputed_work += 1
                    metrics.extra_bytes += int(activation.nbytes + result.result.nbytes)
                    metrics.recovery_latency_ns = time.perf_counter_ns() - detected
                    return DispatchResult(
                        response=result.response,
                        result=result.result,
                        verification=result.verification,
                        metrics={**asdict(metrics), **result.metrics},
                    )
            if strategy == RecoveryStrategy.TIMEOUT_LOCAL_FALLBACK:
                if self.local_executor is None:
                    raise RuntimeError(
                        "local fallback requested but no local executor exists"
                    ) from primary_error
                fallback_started = time.perf_counter_ns()
                response, output = self.local_executor(request, activation)
                verification = self.trust.verify(
                    request,
                    response,
                    output,
                    reference=reference if reference is not None else output,
                )
                metrics.selected_worker = response.worker_id
                metrics.recovered = verification.accepted
                metrics.correctness = verification.accepted
                metrics.recomputed_work = 1
                metrics.extra_compute_ns = time.perf_counter_ns() - fallback_started
                metrics.recovery_latency_ns = time.perf_counter_ns() - detected
                return DispatchResult(
                    response=response,
                    result=output,
                    verification=verification,
                    metrics=asdict(metrics),
                )
            metrics.failed_explicitly = True
            metrics.lost_tokens = 1
            raise RuntimeError(
                f"expert request failed explicitly under {strategy.value}: {primary_error}"
            ) from primary_error

    def _hedged(
        self,
        request: ExpertExecutionRequest,
        activation: np.ndarray,
        *,
        primary_worker: str,
        alternate_workers: tuple[str, ...],
        reference: np.ndarray | None,
        hedge_delay_ms: float,
        metrics: RecoveryMetrics,
    ) -> DispatchResult:
        if not alternate_workers:
            raise ValueError("hedged execution requires an alternate worker")

        def issue(worker_id: str, delay_ms: float) -> DispatchResult:
            if delay_ms:
                time.sleep(delay_ms / 1000)
            return self._remote(worker_id, request, activation, reference=reference)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        try:
            futures = {
                executor.submit(issue, primary_worker, 0.0): primary_worker,
                executor.submit(issue, alternate_workers[0], hedge_delay_ms): alternate_workers[0],
            }
            failures = []
            for future in concurrent.futures.as_completed(futures):
                worker = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    failures.append(error)
                    self.trust.timeout(worker)
                    continue
                metrics.selected_worker = worker
                metrics.recovered = worker != primary_worker
                metrics.correctness = result.verification.accepted
                metrics.recomputed_work = 1
                metrics.extra_bytes = int(activation.nbytes + result.result.nbytes)
                metrics.duplicate_results = 1
                for pending in futures:
                    if pending is not future:
                        pending.cancel()
                return result
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        raise RuntimeError(f"both hedged workers failed: {failures}")

    def _replicated(
        self,
        request: ExpertExecutionRequest,
        activation: np.ndarray,
        *,
        primary_worker: str,
        alternate_workers: tuple[str, ...],
        reference: np.ndarray | None,
        metrics: RecoveryMetrics,
    ) -> DispatchResult:
        if not alternate_workers:
            raise ValueError("sampled replication requires an alternate worker")
        primary = self._remote(primary_worker, request, activation, reference=reference)
        duplicate = self._remote(alternate_workers[0], request, activation, reference=reference)
        comparison = self.trust.compare_duplicate(
            primary_worker,
            primary.result,
            alternate_workers[0],
            duplicate.result,
            exact=request.determinism_mode.value == "exact",
        )
        if not comparison.accepted:
            raise ValueError("sampled duplicate verification disagreed")
        metrics.selected_worker = primary_worker
        metrics.correctness = True
        metrics.recomputed_work = 1
        metrics.extra_bytes = int(activation.nbytes + duplicate.result.nbytes)
        metrics.duplicate_results = 1
        return primary
