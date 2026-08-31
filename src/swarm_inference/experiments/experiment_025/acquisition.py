"""Resilient controller-only acquisition for the E025 physical headline fleet.

This module is deliberately outside the worker startup/import path.  It keeps one
supervisor alive per paid instance group until a simultaneous-health freeze, starts
the Layer 89 parent from public fragment mappings, refreshes compatible Vast offers,
and supports tightly bounded deadline-aware hedges.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .constants import HEADLINE_BACKBONE_GPU_NAMES, PREFERRED_SMALL_GPU_NAMES
from .io import append_jsonl, canonical_sha256, utc_now
from .provisioning import (
    LiveWorker,
    WorkerCompatibilityError,
    create_worker_group_instance,
    probe_worker_liveness,
    wait_for_public_endpoint,
    wait_for_worker,
)
from .vast_lifecycle import AppendOnlyLifecycleLedger, Offer, VastClient

EventCallback = Callable[..., None]


class NoCompatibleOffer(RuntimeError):
    """No currently compatible, unexcluded offer can serve a group."""


class AcquisitionBudgetExceeded(RuntimeError):
    """A primary or hedge launch would exceed the declared acquisition guard."""


@dataclass(frozen=True, slots=True)
class AcquisitionPolicy:
    acquisition_window_seconds: float
    inference_cleanup_reserve_seconds: float
    stability_barrier_seconds: float
    minimum_stability_health_rounds: int
    provider_poll_seconds: float
    ready_health_probe_seconds: float
    provider_missing_poll_limit: int
    health_failure_limit: int
    public_mapping_timeout_seconds: float
    dud_no_progress_timeout_seconds: float
    progress_log_probe_after_seconds: float
    progress_log_probe_interval_seconds: float
    dynamic_alternate_refresh_enabled: bool
    maximum_dynamic_refreshes_per_group: int
    dynamic_refresh_retry_seconds: float
    hedge_enabled: bool
    hedge_warning_no_progress_seconds: float
    hedge_minimum_elapsed_seconds: float
    hedge_parent_minimum_elapsed_seconds: float
    hedge_parent_ready_role_threshold: int
    hedge_probability_threshold: float
    maximum_concurrent_hedges: int
    maximum_total_acquisition_cost_usd: float
    hedge_projected_cost_reserve_usd: float
    maximum_create_concurrency: int
    health_probe_concurrency: int
    offer_scoring: dict[str, Any]
    persistent_hard_exclusion_machine_ids: frozenset[int]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AcquisitionPolicy:
        hedge = dict(value.get("hedge_policy", {}))
        refresh = dict(value.get("alternate_refresh_policy", {}))
        health = dict(value.get("ready_health_policy", {}))
        cost = dict(value.get("cost_guard", {}))
        return cls(
            acquisition_window_seconds=float(value["acquisition_window_seconds"]),
            inference_cleanup_reserve_seconds=float(
                value["inference_cleanup_reserve_seconds"]
            ),
            stability_barrier_seconds=float(health["stability_barrier_seconds"]),
            minimum_stability_health_rounds=int(
                health["minimum_stability_health_rounds"]
            ),
            provider_poll_seconds=float(health["provider_poll_seconds"]),
            ready_health_probe_seconds=float(health["authenticated_probe_seconds"]),
            provider_missing_poll_limit=int(health["provider_missing_poll_limit"]),
            health_failure_limit=int(health["health_failure_limit"]),
            public_mapping_timeout_seconds=float(
                value["parent_endpoint_generation_policy"][
                    "public_mapping_timeout_seconds"
                ]
            ),
            dud_no_progress_timeout_seconds=float(value["dud_timeout_seconds"]),
            progress_log_probe_after_seconds=float(
                value["progress_log_probe_after_seconds"]
            ),
            progress_log_probe_interval_seconds=float(
                value["progress_log_probe_interval_seconds"]
            ),
            dynamic_alternate_refresh_enabled=bool(refresh["enabled"]),
            maximum_dynamic_refreshes_per_group=int(
                refresh["maximum_refreshes_per_group"]
            ),
            dynamic_refresh_retry_seconds=float(refresh["retry_seconds"]),
            hedge_enabled=bool(hedge["enabled"]),
            hedge_warning_no_progress_seconds=float(
                hedge["warning_no_progress_seconds"]
            ),
            hedge_minimum_elapsed_seconds=float(hedge["minimum_elapsed_seconds"]),
            hedge_parent_minimum_elapsed_seconds=float(
                hedge["parent_minimum_elapsed_seconds"]
            ),
            hedge_parent_ready_role_threshold=int(
                hedge["parent_ready_role_threshold"]
            ),
            hedge_probability_threshold=float(hedge["probability_threshold"]),
            maximum_concurrent_hedges=int(hedge["maximum_concurrent_hedges"]),
            maximum_total_acquisition_cost_usd=float(
                cost["maximum_total_acquisition_cost_usd"]
            ),
            hedge_projected_cost_reserve_usd=float(
                cost["hedge_projected_cost_reserve_usd"]
            ),
            maximum_create_concurrency=int(value["maximum_create_concurrency"]),
            health_probe_concurrency=int(health["probe_concurrency"]),
            offer_scoring=dict(value["offer_score"]),
            persistent_hard_exclusion_machine_ids=frozenset(
                int(machine_id)
                for machine_id in value.get(
                    "persistent_hard_exclusion_machine_ids", []
                )
            ),
        )


@dataclass(slots=True)
class Candidate:
    group_id: str
    sequence: int
    offer: Offer
    score: dict[str, Any]
    endpoint_generation: int | None
    expert_endpoints: list[dict[str, Any]] | None
    hedge: bool
    started_epoch: float
    abort_event: threading.Event = field(default_factory=threading.Event)
    instance_id: int | None = None
    created_epoch: float | None = None
    completed_epoch: float | None = None
    abort_reason: str | None = None
    fragment_endpoint: dict[str, Any] | None = None
    destroyed: bool = False
    destroy_lock: threading.Lock = field(default_factory=threading.Lock)
    create_finished_event: threading.Event = field(default_factory=threading.Event)

    @property
    def candidate_id(self) -> str:
        return f"{self.group_id}-candidate-{self.sequence:03d}"


@dataclass(frozen=True, slots=True)
class CandidateResult:
    candidate: Candidate
    workers: tuple[LiveWorker, ...]


@dataclass(slots=True)
class ProgressState:
    instance_id: int
    first_seen_epoch: float
    last_progress_epoch: float
    signature: tuple[Any, ...] | None = None
    last_log_query_epoch: float = 0.0
    last_log_sha256: str | None = None
    fatal_marker: str | None = None
    row: dict[str, Any] | None = None


class FragmentEndpointCoordinator:
    """Version the four public expert mappings independently of worker readiness."""

    def __init__(self, fragment_ids: list[str], emit: EventCallback) -> None:
        if len(fragment_ids) != 4 or len(set(fragment_ids)) != 4:
            raise ValueError("E025 requires exactly four distinct fragment identities")
        self.fragment_ids = tuple(sorted(fragment_ids))
        self.emit = emit
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.generation = 1
        self.endpoints: dict[str, dict[str, Any]] = {}
        self.parent_invalidator: Callable[[str], None] | None = None

    def set_parent_invalidator(self, callback: Callable[[str], None]) -> None:
        with self.lock:
            self.parent_invalidator = callback

    def publish(
        self,
        *,
        worker_id: str,
        worker_index: int,
        instance_id: int,
        machine_id: int,
        host: str,
        port: int,
    ) -> int:
        with self.condition:
            current = self.endpoints.get(worker_id)
            if current is not None and int(current["instance_id"]) != instance_id:
                raise RuntimeError(
                    "fragment endpoint changed without retiring its prior generation"
                )
            self.endpoints[worker_id] = {
                "worker_id": worker_id,
                "worker_index": worker_index,
                "host": host,
                "port": port,
                "timeout_seconds": 180.0,
                "instance_id": instance_id,
                "machine_id": machine_id,
                "fragment_endpoint_generation": self.generation,
            }
            generation = self.generation
            complete = set(self.endpoints) == set(self.fragment_ids)
            self.condition.notify_all()
        self.emit(
            "FRAGMENT_ENDPOINT_PUBLISHED",
            worker_id=worker_id,
            worker_index=worker_index,
            instance_id=instance_id,
            machine_id=machine_id,
            host=host,
            port=port,
            fragment_endpoint_generation=generation,
            endpoint_set_complete=complete,
        )
        return generation

    def retire(self, *, worker_id: str, instance_id: int, reason: str) -> bool:
        invalidator: Callable[[str], None] | None = None
        with self.condition:
            current = self.endpoints.get(worker_id)
            if current is None or int(current["instance_id"]) != instance_id:
                return False
            old_generation = self.generation
            self.endpoints.pop(worker_id, None)
            self.generation += 1
            new_generation = self.generation
            invalidator = self.parent_invalidator
            self.condition.notify_all()
        self.emit(
            "FRAGMENT_ENDPOINT_GENERATION_CHANGED",
            worker_id=worker_id,
            retired_instance_id=instance_id,
            old_fragment_endpoint_generation=old_generation,
            fragment_endpoint_generation=new_generation,
            reason=reason,
        )
        if invalidator is not None:
            invalidator(
                f"fragment {worker_id} retired from endpoint generation {old_generation}"
            )
        return True

    def complete_snapshot(self) -> tuple[int, list[dict[str, Any]]] | None:
        with self.lock:
            if set(self.endpoints) != set(self.fragment_ids):
                return None
            generation = self.generation
            endpoints = [dict(self.endpoints[worker_id]) for worker_id in self.fragment_ids]
        payload = [
            {
                "worker_id": row["worker_id"],
                "worker_index": row["worker_index"],
                "host": row["host"],
                "port": row["port"],
                "timeout_seconds": row["timeout_seconds"],
            }
            for row in endpoints
        ]
        return generation, payload

    def wait_for_complete(
        self, *, deadline_epoch: float, abort_event: threading.Event
    ) -> tuple[int, list[dict[str, Any]]]:
        with self.condition:
            while time.time() < deadline_epoch:
                if abort_event.is_set():
                    raise RuntimeError("fragment endpoint wait was aborted")
                snapshot = self.complete_snapshot_unlocked()
                if snapshot is not None:
                    return snapshot
                self.condition.wait(timeout=min(1.0, max(0.0, deadline_epoch - time.time())))
        raise TimeoutError("four current fragment public mappings never converged")

    def complete_snapshot_unlocked(self) -> tuple[int, list[dict[str, Any]]] | None:
        if set(self.endpoints) != set(self.fragment_ids):
            return None
        return self.generation, [
            {
                "worker_id": self.endpoints[worker_id]["worker_id"],
                "worker_index": self.endpoints[worker_id]["worker_index"],
                "host": self.endpoints[worker_id]["host"],
                "port": self.endpoints[worker_id]["port"],
                "timeout_seconds": self.endpoints[worker_id]["timeout_seconds"],
            }
            for worker_id in self.fragment_ids
        ]

    def matches(self, generation: int | None) -> bool:
        with self.lock:
            return (
                generation is not None
                and generation == self.generation
                and set(self.endpoints) == set(self.fragment_ids)
            )

    def receipt(self) -> dict[str, Any]:
        with self.lock:
            return {
                "fragment_endpoint_generation": self.generation,
                "complete": set(self.endpoints) == set(self.fragment_ids),
                "endpoints": [
                    dict(self.endpoints[worker_id])
                    for worker_id in self.fragment_ids
                    if worker_id in self.endpoints
                ],
            }


class ReadyRegistry:
    def __init__(self, expected_worker_ids: set[str], parent_group_id: str) -> None:
        self.expected_worker_ids = frozenset(expected_worker_ids)
        self.parent_group_id = parent_group_id
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.by_group: dict[str, tuple[LiveWorker, ...]] = {}
        self.parent_generation: int | None = None
        self.version = 0

    def publish(
        self,
        group_id: str,
        workers: tuple[LiveWorker, ...],
        *,
        parent_generation: int | None,
    ) -> None:
        with self.condition:
            self.by_group[group_id] = workers
            if group_id == self.parent_group_id:
                self.parent_generation = parent_generation
            self.version += 1
            self.condition.notify_all()

    def remove(self, group_id: str) -> tuple[LiveWorker, ...]:
        with self.condition:
            workers = self.by_group.pop(group_id, ())
            if group_id == self.parent_group_id:
                self.parent_generation = None
            if workers:
                self.version += 1
                self.condition.notify_all()
            return workers

    def snapshot(self) -> tuple[dict[str, tuple[LiveWorker, ...]], int | None, int]:
        with self.lock:
            return dict(self.by_group), self.parent_generation, self.version

    def workers(self) -> dict[str, LiveWorker]:
        groups, _generation, _version = self.snapshot()
        return {
            worker.worker_id: worker
            for group_workers in groups.values()
            for worker in group_workers
        }

    def current_ready_count(self) -> int:
        return len(self.workers())


class CostGuard:
    """Conservative real-time cap using committed ingress plus paid lifetimes."""

    def __init__(self, maximum_usd: float, hedge_reserve_usd: float) -> None:
        self.maximum_usd = maximum_usd
        self.hedge_reserve_usd = hedge_reserve_usd
        self.lock = threading.Lock()
        self.created: dict[str, dict[str, float | None]] = {}
        self.authorized_projection_usd: dict[str, float] = {}
        self.blockers: list[dict[str, Any]] = []

    @staticmethod
    def _launch_projection(
        offer: Offer, *, disk_gb: float, download_bytes: int, seconds: float
    ) -> float:
        return (
            offer.effective_rate_per_hour(disk_gb) * seconds / 3600.0
            + offer.inet_down_cost_per_gb * download_bytes / 1e9
        )

    def _accrued_unlocked(self, now: float) -> float:
        total = 0.0
        for row in self.created.values():
            end = float(row["destroyed_epoch"] or now)
            elapsed = max(0.0, end - float(row["created_epoch"] or now))
            total += float(row["effective_rate_per_hour"] or 0.0) * elapsed / 3600.0
            total += float(row["committed_ingress_usd"] or 0.0)
        return total

    def authorize(
        self,
        *,
        candidate_id: str,
        offer: Offer,
        disk_gb: float,
        download_bytes: int,
        projected_seconds: float,
        hedge: bool,
    ) -> tuple[bool, float]:
        with self.lock:
            now = time.time()
            committed = 0.0
            for candidate_id_value, value in self.authorized_projection_usd.items():
                created = self.created.get(candidate_id_value)
                if created is None:
                    committed += value
                elif created["destroyed_epoch"] is None:
                    created_elapsed_cost = (
                        float(created["effective_rate_per_hour"] or 0.0)
                        * max(0.0, now - float(created["created_epoch"] or now))
                        / 3600.0
                        + float(created["committed_ingress_usd"] or 0.0)
                    )
                    committed += max(value, created_elapsed_cost)
            committed += sum(
                float(row["effective_rate_per_hour"] or 0.0)
                * max(
                    0.0,
                    float(row["destroyed_epoch"] or now)
                    - float(row["created_epoch"] or now),
                )
                / 3600.0
                + float(row["committed_ingress_usd"] or 0.0)
                for candidate_id_value, row in self.created.items()
                if row["destroyed_epoch"] is not None
                and candidate_id_value not in self.authorized_projection_usd
            )
            launch_projection = self._launch_projection(
                offer,
                disk_gb=disk_gb,
                download_bytes=download_bytes,
                seconds=projected_seconds,
            )
            projected = committed + launch_projection
            reserve = self.hedge_reserve_usd if hedge else 0.0
            allowed = projected + reserve <= self.maximum_usd
            if not allowed:
                self.blockers.append(
                    {
                        "timestamp_utc": utc_now(),
                        "candidate_id": candidate_id,
                        "hedge": hedge,
                        "projected_total_usd": projected,
                        "required_reserve_usd": reserve,
                        "maximum_usd": self.maximum_usd,
                    }
                )
            else:
                self.authorized_projection_usd[candidate_id] = launch_projection
            return allowed, projected

    def created_instance(
        self,
        candidate: Candidate,
        *,
        disk_gb: float,
        download_bytes: int,
    ) -> None:
        with self.lock:
            self.created[candidate.candidate_id] = {
                "created_epoch": candidate.created_epoch,
                "destroyed_epoch": None,
                "effective_rate_per_hour": candidate.offer.effective_rate_per_hour(
                    disk_gb
                ),
                "committed_ingress_usd": (
                    candidate.offer.inet_down_cost_per_gb * download_bytes / 1e9
                ),
            }

    def destroyed_instance(self, candidate: Candidate) -> None:
        with self.lock:
            row = self.created.get(candidate.candidate_id)
            if row is not None and row["destroyed_epoch"] is None:
                row["destroyed_epoch"] = time.time()
            self.authorized_projection_usd.pop(candidate.candidate_id, None)

    def release_authorization(self, candidate: Candidate) -> None:
        with self.lock:
            self.authorized_projection_usd.pop(candidate.candidate_id, None)

    def receipt(self) -> dict[str, Any]:
        with self.lock:
            return {
                "maximum_total_acquisition_cost_usd": self.maximum_usd,
                "hedge_projected_cost_reserve_usd": self.hedge_reserve_usd,
                "estimated_committed_cost_usd": self._accrued_unlocked(time.time()),
                "outstanding_projected_cost_usd": sum(
                    self.authorized_projection_usd.values()
                ),
                "created_candidate_count": len(self.created),
                "authorization_blockers": list(self.blockers),
            }


class ProviderProgressTracker:
    fatal_markers = (
        "native mxfp4 cuda tensor upload failed",
        "tensor allocation: out of memory",
        "no space left on device",
        "refusing to replace activated snapshot",
        "port is already allocated",
    )

    def __init__(self, policy: AcquisitionPolicy, emit: EventCallback) -> None:
        self.policy = policy
        self.emit = emit
        self.lock = threading.Lock()
        self.states: dict[int, ProgressState] = {}
        self.last_rows: dict[int, dict[str, Any]] = {}

    def register(self, instance_id: int) -> None:
        now = time.time()
        with self.lock:
            self.states[instance_id] = ProgressState(
                instance_id=instance_id,
                first_seen_epoch=now,
                last_progress_epoch=now,
            )

    def unregister(self, instance_id: int) -> None:
        with self.lock:
            self.states.pop(instance_id, None)
            self.last_rows.pop(instance_id, None)

    def poll(self, client: VastClient) -> dict[int, dict[str, Any]]:
        rows = client.show_instances()
        now = time.time()
        indexed = {
            int(row.get("id", row.get("instance_id", -1))): row
            for row in rows
            if int(row.get("id", row.get("instance_id", -1))) >= 0
        }
        log_queries: list[tuple[int, ProgressState, dict[str, Any]]] = []
        with self.lock:
            self.last_rows = indexed
            for instance_id, state in self.states.items():
                row = indexed.get(instance_id)
                state.row = row
                if row is None:
                    continue
                signature = (
                    row.get("actual_status"),
                    row.get("status_msg"),
                    row.get("disk_usage"),
                    row.get("inet_down_billed"),
                    row.get("vmem_usage"),
                )
                if signature != state.signature:
                    state.signature = signature
                    state.last_progress_epoch = now
                if (
                    now - state.last_progress_epoch
                    >= self.policy.progress_log_probe_after_seconds
                    and now - state.last_log_query_epoch
                    >= self.policy.progress_log_probe_interval_seconds
                ):
                    state.last_log_query_epoch = now
                    log_queries.append((instance_id, state, row))
        for instance_id, state, row in log_queries:
            try:
                log_text = client.logs(instance_id, tail=160)
            except BaseException as exc:
                self.emit(
                    "RETRY",
                    operation="bootstrap_progress_log_query",
                    instance_id=instance_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                continue
            lowered = log_text.lower()
            substantive = bool(log_text.strip()) and not any(
                marker in lowered
                for marker in ("error response from daemon", "no such container")
            )
            if not substantive:
                continue
            digest = hashlib.sha256(log_text.encode("utf-8")).hexdigest()
            with self.lock:
                changed = state.last_log_sha256 is None or state.last_log_sha256 != digest
                state.last_log_sha256 = digest
                if changed:
                    state.last_progress_epoch = time.time()
                marker = next(
                    (value for value in self.fatal_markers if value in lowered), None
                )
                if marker is not None:
                    state.fatal_marker = marker
            self.emit(
                "BOOTSTRAP_PROGRESS",
                instance_id=instance_id,
                machine_id=row.get("machine_id"),
                log_changed=changed,
                log_substantive=True,
                raw_log_retained=False,
                fatal_marker=marker,
            )
        return indexed

    def risk(self, candidate: Candidate) -> dict[str, Any]:
        now = time.time()
        if candidate.instance_id is None:
            return {
                "elapsed_seconds": now - candidate.started_epoch,
                "no_progress_seconds": 0.0,
                "fatal_marker": None,
                "provider_row": None,
            }
        with self.lock:
            state = self.states.get(candidate.instance_id)
            if state is None:
                return {
                    "elapsed_seconds": now - candidate.started_epoch,
                    "no_progress_seconds": 0.0,
                    "fatal_marker": None,
                    "provider_row": None,
                }
            return {
                "elapsed_seconds": now - candidate.started_epoch,
                "no_progress_seconds": max(0.0, now - state.last_progress_epoch),
                "fatal_marker": state.fatal_marker,
                "provider_row": dict(state.row) if state.row is not None else None,
            }


class OfferSelector:
    """Reserve frozen candidates first, then auditable compatible live refreshes."""

    def __init__(
        self,
        *,
        client: VastClient,
        policy: AcquisitionPolicy,
        groups: Mapping[str, dict[str, Any]],
        history_by_machine: Mapping[int, Mapping[str, Any]],
        audit_path: Path,
        emit: EventCallback,
    ) -> None:
        self.client = client
        self.policy = policy
        self.groups = groups
        self.history_by_machine = history_by_machine
        self.audit_path = audit_path
        self.emit = emit
        self.lock = threading.Lock()
        self.query_lock = threading.Lock()
        self.audit_lock = threading.Lock()
        self.active_offer_ids: set[int] = set()
        self.active_machine_ids: set[int] = set()
        self.failed_offer_ids: set[int] = set()
        self.failed_machine_ids: set[int] = set(
            policy.persistent_hard_exclusion_machine_ids
        )
        self.initial: dict[str, list[Offer]] = {
            group_id: [
                Offer(**group["selected_offer"]),
                *(Offer(**row["offer"]) for row in group.get("alternates", [])),
            ]
            for group_id, group in groups.items()
        }
        self.refresh_counts: dict[str, int] = {group_id: 0 for group_id in groups}
        self.last_blocker: dict[str, str] = {}
        self.selected_scores: list[dict[str, Any]] = []

    def _compatible(self, group_id: str, offer: Offer) -> bool:
        group = self.groups[group_id]
        workers = list(group["workers"])
        capacity = len(workers)
        role = str(workers[0]["role"])
        return (
            offer.gpu_count >= capacity
            and offer.direct_port_count >= capacity
            and offer.qualifies(
                role,
                disk_gb=float(group["disk_gb"]),
                allow_multi_gpu=offer.gpu_count > 1,
            )
            and (
                role == "SUB_LAYER_WORKER"
                or any(
                    name.upper() in offer.gpu_name.upper()
                    for name in HEADLINE_BACKBONE_GPU_NAMES
                )
            )
        )

    def _score(self, group_id: str, offer: Offer) -> dict[str, Any]:
        group = self.groups[group_id]
        required_download_bytes = sum(
            int(worker["download_bytes_cold_cache"])
            for worker in group["workers"]
        )
        history = self.history_by_machine.get(offer.machine_id)
        return offer.acquisition_score(
            required_download_bytes=required_download_bytes,
            disk_gb=float(group["disk_gb"]),
            history=history,
            scoring_policy=self.policy.offer_scoring,
        )

    def _available_unlocked(self, offer: Offer) -> bool:
        return (
            offer.offer_id not in self.active_offer_ids
            and offer.offer_id not in self.failed_offer_ids
            and offer.machine_id not in self.active_machine_ids
            and offer.machine_id not in self.failed_machine_ids
        )

    def reserve(self, group_id: str) -> tuple[Offer, dict[str, Any], str]:
        with self.lock:
            frozen = self.initial[group_id]
            while frozen:
                offer = frozen.pop(0)
                if not self._available_unlocked(offer) or not self._compatible(
                    group_id, offer
                ):
                    continue
                score = self._score(group_id, offer)
                self.active_offer_ids.add(offer.offer_id)
                self.active_machine_ids.add(offer.machine_id)
                self.selected_scores.append(
                    {
                        "timestamp_utc": utc_now(),
                        "instance_group_id": group_id,
                        "selection_source": "FROZEN_INITIAL_PLAN",
                        "offer_id": offer.offer_id,
                        "machine_id": offer.machine_id,
                        "score": score,
                    }
                )
                return offer, score, "FROZEN_INITIAL_PLAN"
        return self._refresh_and_reserve(group_id)

    def _refresh_and_reserve(self, group_id: str) -> tuple[Offer, dict[str, Any], str]:
        if not self.policy.dynamic_alternate_refresh_enabled:
            raise NoCompatibleOffer(
                f"frozen alternates exhausted and refresh disabled for {group_id}"
            )
        with self.query_lock:
            with self.lock:
                if (
                    self.refresh_counts[group_id]
                    >= self.policy.maximum_dynamic_refreshes_per_group
                ):
                    self.last_blocker[group_id] = "bounded live refresh limit exhausted"
                    raise NoCompatibleOffer(
                        f"bounded live refresh limit exhausted for {group_id}"
                    )
                self.refresh_counts[group_id] += 1
                refresh_number = self.refresh_counts[group_id]
                excluded_offer_ids = sorted(
                    self.active_offer_ids | self.failed_offer_ids
                )
                excluded_machine_ids = sorted(
                    self.active_machine_ids | self.failed_machine_ids
                )
            group = self.groups[group_id]
            workers = list(group["workers"])
            capacity = len(workers)
            role = str(workers[0]["role"])
            gpu_names = (
                PREFERRED_SMALL_GPU_NAMES
                if role == "SUB_LAYER_WORKER"
                else HEADLINE_BACKBONE_GPU_NAMES
            )
            offers = self.client.search_offers(
                gpu_names=gpu_names,
                storage_gb=float(group["disk_gb"]),
                single_gpu_only=capacity == 1,
            )
            scored: list[tuple[float, Offer, dict[str, Any]]] = []
            with self.lock:
                for offer in offers:
                    if not self._available_unlocked(offer) or not self._compatible(
                        group_id, offer
                    ):
                        continue
                    score = self._score(group_id, offer)
                    scored.append(
                        (float(score["short_run_acquisition_score"]), offer, score)
                    )
                scored.sort(
                    key=lambda row: (
                        row[0],
                        -float(row[2].get("empirical_probability_healthy_ready", 0.0)),
                        int(row[1].offer_id),
                    )
                )
                selected = scored[0] if scored else None
                if selected is not None:
                    _value, offer, score = selected
                    self.active_offer_ids.add(offer.offer_id)
                    self.active_machine_ids.add(offer.machine_id)
                    self.selected_scores.append(
                        {
                            "timestamp_utc": utc_now(),
                            "instance_group_id": group_id,
                            "selection_source": "DYNAMIC_LIVE_REFRESH",
                            "refresh_number": refresh_number,
                            "offer_id": offer.offer_id,
                            "machine_id": offer.machine_id,
                            "score": score,
                        }
                    )
            snapshot = {
                "schema_version": "experiment-025-dynamic-offer-refresh-v1",
                "timestamp_utc": utc_now(),
                "instance_group_id": group_id,
                "refresh_number": refresh_number,
                "exact_constraints": {
                    "role": role,
                    "gpu_names": list(gpu_names),
                    "required_gpu_slots": capacity,
                    "required_disk_gb": group["disk_gb"],
                    "required_direct_ports": capacity,
                    "consumer_only": True,
                },
                "excluded_offer_ids": excluded_offer_ids,
                "excluded_machine_ids": excluded_machine_ids,
                "raw_compatible_candidate_count": len(scored),
                "compatible_candidates": [
                    {"offer": asdict(offer), "score": score}
                    for _value, offer, score in scored
                ],
                "selected_offer_id": selected[1].offer_id if selected else None,
                "selected_machine_id": selected[1].machine_id if selected else None,
            }
            snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
            with self.audit_lock:
                append_jsonl(self.audit_path, snapshot)
            self.emit(
                "DYNAMIC_OFFER_REFRESH",
                instance_group_id=group_id,
                refresh_number=refresh_number,
                compatible_candidate_count=len(scored),
                selected_offer_id=snapshot["selected_offer_id"],
                selected_machine_id=snapshot["selected_machine_id"],
                offer_snapshot_sha256=snapshot["snapshot_sha256"],
            )
            if selected is None:
                with self.lock:
                    self.last_blocker[group_id] = (
                        "live marketplace returned no compatible unexcluded offer"
                    )
                raise NoCompatibleOffer(
                    f"live marketplace has no compatible offer for {group_id}"
                )
            return selected[1], selected[2], "DYNAMIC_LIVE_REFRESH"

    def release(self, offer: Offer) -> None:
        with self.lock:
            self.active_offer_ids.discard(offer.offer_id)
            self.active_machine_ids.discard(offer.machine_id)

    def fail(self, offer: Offer) -> None:
        with self.lock:
            self.active_offer_ids.discard(offer.offer_id)
            self.active_machine_ids.discard(offer.machine_id)
            self.failed_offer_ids.add(offer.offer_id)
            self.failed_machine_ids.add(offer.machine_id)

    def receipt(self) -> dict[str, Any]:
        with self.lock:
            return {
                "failed_offer_ids": sorted(self.failed_offer_ids),
                "failed_machine_ids": sorted(self.failed_machine_ids),
                "active_offer_ids": sorted(self.active_offer_ids),
                "active_machine_ids": sorted(self.active_machine_ids),
                "refresh_counts": dict(sorted(self.refresh_counts.items())),
                "last_marketplace_blockers": dict(sorted(self.last_blocker.items())),
                "selections": list(self.selected_scores),
                "audit_path": str(self.audit_path.resolve()),
            }


class HedgeLimiter:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.lock = threading.Lock()
        self.current = 0
        self.peak = 0
        self.started = 0

    def acquire(self) -> bool:
        with self.lock:
            if self.current >= self.maximum:
                return False
            self.current += 1
            self.started += 1
            self.peak = max(self.peak, self.current)
            return True

    def release(self) -> None:
        with self.lock:
            self.current = max(0, self.current - 1)

    def receipt(self) -> dict[str, int]:
        with self.lock:
            return {
                "maximum_concurrent_hedges": self.maximum,
                "hedges_started": self.started,
                "peak_concurrent_hedges": self.peak,
                "current_hedges": self.current,
            }


class GroupSupervisor:
    def __init__(
        self,
        *,
        controller: FleetAcquisitionController,
        group_id: str,
    ) -> None:
        self.controller = controller
        self.group_id = group_id
        self.group = controller.groups[group_id]
        self.worker_rows = sorted(
            self.group["workers"], key=lambda row: int(row["gpu_slot"])
        )
        self.is_parent = group_id == controller.parent_group_id
        self.fragment_row = next(
            (
                row
                for row in self.worker_rows
                if str(row["role"]) == "SUB_LAYER_WORKER"
            ),
            None,
        )
        self.lock = threading.Lock()
        self.invalidate_event = threading.Event()
        self.invalidation_reason: str | None = None
        self.invalidation_attributable = True
        self.current_candidate: Candidate | None = None
        self.active_candidates: dict[int, Candidate] = {}
        self.sequence = 0
        self.state = "NO_INSTANCE"
        self.thread = threading.Thread(
            target=self.run,
            name=f"e025-group-supervisor-{group_id}",
            daemon=False,
        )
        self.last_error: dict[str, Any] | None = None

    def transition(self, state: str, **fields: Any) -> None:
        with self.lock:
            previous = self.state
            self.state = state
        self.controller.emit(
            "GROUP_STATE_CHANGED",
            instance_group_id=self.group_id,
            previous_state=previous,
            state=state,
            **fields,
        )

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout=timeout)

    def invalidate(self, reason: str, *, attributable: bool = True) -> None:
        with self.lock:
            candidate = self.current_candidate
            active_candidates = list(self.active_candidates.values())
            already = self.invalidate_event.is_set()
            self.invalidation_reason = reason
            self.invalidation_attributable = attributable
            self.invalidate_event.set()
        removed = self.controller.registry.remove(self.group_id)
        if removed and not already:
            for worker in removed:
                self.controller.emit(
                    "WORKER_READY_LOST",
                    instance_group_id=self.group_id,
                    worker_id=worker.worker_id,
                    instance_id=worker.instance_id,
                    machine_id=worker.machine_id,
                    reason=reason,
                )
            self.controller.emit(
                "READY_WORKER_REPLACEMENT_STARTED",
                instance_group_id=self.group_id,
                lost_worker_count=len(removed),
                reason=reason,
            )
        if self.fragment_row is not None and candidate is not None and candidate.instance_id:
            self.controller.fragment_endpoints.retire(
                worker_id=str(self.fragment_row["worker_id"]),
                instance_id=candidate.instance_id,
                reason=reason,
            )
        for active_candidate in active_candidates:
            active_candidate.abort_reason = reason
            active_candidate.abort_event.set()

    def active_candidate_snapshot(self) -> list[Candidate]:
        with self.lock:
            return list(self.active_candidates.values())

    def _next_candidate(
        self,
        *,
        hedge: bool,
        endpoint_generation: int | None,
        expert_endpoints: list[dict[str, Any]] | None,
    ) -> Candidate:
        offer, score, source = self.controller.selector.reserve(self.group_id)
        with self.lock:
            self.sequence += 1
            sequence = self.sequence
        candidate = Candidate(
            group_id=self.group_id,
            sequence=sequence,
            offer=offer,
            score=score,
            endpoint_generation=endpoint_generation,
            expert_endpoints=expert_endpoints,
            hedge=hedge,
            started_epoch=time.time(),
        )
        projected_seconds = min(
            max(
                60.0,
                float(score.get("expected_ready_seconds", 0.0))
                or float(score.get("expected_bootstrap_seconds", 0.0)),
            ),
            max(60.0, self.controller.deadline_epoch - time.time()),
        )
        allowed, projected_total = self.controller.cost_guard.authorize(
            candidate_id=candidate.candidate_id,
            offer=offer,
            disk_gb=float(self.group["disk_gb"]),
            download_bytes=sum(
                int(row["download_bytes_cold_cache"]) for row in self.worker_rows
            ),
            projected_seconds=projected_seconds,
            hedge=hedge,
        )
        if not allowed:
            self.controller.selector.release(offer)
            raise AcquisitionBudgetExceeded(
                f"{candidate.candidate_id} would project acquisition cost to "
                f"${projected_total:.4f}"
            )
        with self.lock:
            self.active_candidates[sequence] = candidate
        self.controller.emit(
            "OFFER_SELECTED",
            instance_group_id=self.group_id,
            candidate_id=candidate.candidate_id,
            selection_source=source,
            hedge=hedge,
            offer_id=offer.offer_id,
            machine_id=offer.machine_id,
            gpu_model=offer.gpu_name,
            acquisition_score=score,
            fragment_endpoint_generation=endpoint_generation,
        )
        return candidate

    def _destroy_candidate(self, candidate: Candidate, reason: str) -> None:
        with candidate.destroy_lock:
            if candidate.destroyed:
                return
            # A hedge can be cancelled while create_instance is still in flight.
            # Leave it armed for cleanup once the provider ID becomes known.
            if candidate.instance_id is None:
                if candidate.create_finished_event.is_set():
                    candidate.destroyed = True
                    self.controller.cost_guard.release_authorization(candidate)
                return
            candidate.destroyed = True
            try:
                self.controller.client.destroy_instance(
                    candidate.instance_id,
                    reason=reason,
                )
            finally:
                self.controller.progress.unregister(candidate.instance_id)
                self.controller.cost_guard.destroyed_instance(candidate)
                if self.fragment_row is not None:
                    self.controller.fragment_endpoints.retire(
                        worker_id=str(self.fragment_row["worker_id"]),
                        instance_id=candidate.instance_id,
                        reason=reason,
                    )

    def _bootstrap(self, candidate: Candidate) -> CandidateResult:
        self.transition(
            "CREATING",
            candidate_id=candidate.candidate_id,
            hedge=candidate.hedge,
        )
        try:
            worker_specs = [
                {
                    "worker_id": str(worker["worker_id"]),
                    "gpu_slot": int(worker["gpu_slot"]),
                    "port": int(worker["container_port"]),
                    "maximum_context": 64,
                    "expert_endpoints": (
                        candidate.expert_endpoints
                        if str(worker["worker_id"]) == self.controller.parent_id
                        else None
                    ),
                }
                for worker in self.worker_rows
            ]
            try:
                with self.controller.create_semaphore:
                    if (
                        self.controller.abort_event.is_set()
                        or candidate.abort_event.is_set()
                    ):
                        raise RuntimeError("candidate aborted before paid create")
                    instance_id = self.controller.create_group_instance_fn(
                        client=self.controller.client,
                        run_id=self.controller.run_id,
                        group_id=candidate.candidate_id,
                        group_index=self.controller.group_indexes[self.group_id],
                        offer=candidate.offer,
                        image_reference=self.controller.image_reference,
                        image_digest=self.controller.image_digest,
                        disk_gb=int(self.group["disk_gb"]),
                        material=self.controller.material,
                        watchdog_receipt=self.controller.watchdog_receipt,
                        go_receipt=self.controller.go_receipt,
                        worker_specs=worker_specs,
                    )
            finally:
                candidate.create_finished_event.set()
            candidate.instance_id = instance_id
            candidate.created_epoch = time.time()
            self.controller.progress.register(instance_id)
            self.controller.cost_guard.created_instance(
                candidate,
                disk_gb=float(self.group["disk_gb"]),
                download_bytes=sum(
                    int(row["download_bytes_cold_cache"])
                    for row in self.worker_rows
                ),
            )
            self.controller.record_created(candidate)
            if self.controller.abort_event.is_set() or candidate.abort_event.is_set():
                raise RuntimeError(candidate.abort_reason or "candidate aborted after create")
            self.transition(
                "BOOTSTRAPPING",
                candidate_id=candidate.candidate_id,
                instance_id=instance_id,
                hedge=candidate.hedge,
            )
            if self.fragment_row is not None:
                host, port = self.controller.wait_for_public_endpoint_fn(
                    client=self.controller.client,
                    ledger=self.controller.ledger,
                    worker_id=str(self.fragment_row["worker_id"]),
                    offer=candidate.offer,
                    instance_id=instance_id,
                    deadline_epoch=min(
                        self.controller.deadline_epoch,
                        time.time()
                        + self.controller.policy.public_mapping_timeout_seconds,
                    ),
                    container_port=int(self.fragment_row["container_port"]),
                    abort_event=candidate.abort_event,
                )
                candidate.fragment_endpoint = {
                    "worker_id": str(self.fragment_row["worker_id"]),
                    "worker_index": int(
                        self.controller.specs[str(self.fragment_row["worker_id"])][
                            "worker_index"
                        ]
                    ),
                    "instance_id": instance_id,
                    "machine_id": candidate.offer.machine_id,
                    "host": host,
                    "port": port,
                }
                # The primary mapping is immediately useful to the parent.  A
                # fragment hedge remains private until it wins, at which point
                # promotion atomically advances the endpoint generation.
                if not candidate.hedge:
                    self.controller.fragment_endpoints.publish(
                        **candidate.fragment_endpoint
                    )
            credential = self.controller.credential
            certificate = self.controller.certificate

            def await_worker(row: dict[str, Any]) -> LiveWorker:
                worker_id = str(row["worker_id"])
                spec = self.controller.specs[worker_id]
                return self.controller.wait_for_worker_fn(
                    client=self.controller.client,
                    ledger=self.controller.ledger,
                    run_id=self.controller.run_id,
                    worker_id=worker_id,
                    role=spec["role"],
                    layer=spec["layer"],
                    worker_index=spec["worker_index"],
                    offer=candidate.offer,
                    instance_id=instance_id,
                    credential=credential,
                    certificate=certificate,
                    image_digest=self.controller.image_digest,
                    deadline_epoch=self.controller.deadline_epoch,
                    container_port=int(row["container_port"]),
                    gpu_slot=int(row["gpu_slot"]),
                    abort_event=candidate.abort_event,
                )

            with ThreadPoolExecutor(max_workers=len(self.worker_rows)) as pool:
                workers = tuple(pool.map(await_worker, self.worker_rows))
            if candidate.abort_event.is_set():
                raise RuntimeError(candidate.abort_reason or "candidate became stale")
            if len(workers) != len(self.worker_rows):
                raise RuntimeError("instance group returned partial readiness")
            candidate.completed_epoch = time.time()
            return CandidateResult(candidate=candidate, workers=workers)
        except BaseException:
            self._destroy_candidate(
                candidate,
                candidate.abort_reason or "candidate-bootstrap-failed",
            )
            raise

    def _promote_fragment_winner(self, winner: Candidate) -> None:
        if self.fragment_row is None or winner.fragment_endpoint is None:
            return
        snapshot = self.controller.fragment_endpoints.receipt()
        current = next(
            (
                row
                for row in snapshot["endpoints"]
                if row["worker_id"] == winner.fragment_endpoint["worker_id"]
            ),
            None,
        )
        if current is not None and int(current["instance_id"]) == winner.instance_id:
            return
        if current is not None:
            self.controller.fragment_endpoints.retire(
                worker_id=str(current["worker_id"]),
                instance_id=int(current["instance_id"]),
                reason=f"fragment hedge promoted {winner.candidate_id}",
            )
        self.controller.fragment_endpoints.publish(**winner.fragment_endpoint)

    def _candidate_risk(self, candidate: Candidate) -> tuple[bool, bool, dict[str, Any]]:
        risk = self.controller.progress.risk(candidate)
        hard_dud = bool(risk["fatal_marker"]) or (
            float(risk["no_progress_seconds"])
            >= self.controller.policy.dud_no_progress_timeout_seconds
        )
        if hard_dud:
            return True, False, risk
        if not self.controller.policy.hedge_enabled:
            return False, False, risk
        elapsed = float(risk["elapsed_seconds"])
        expected = float(
            candidate.score.get("expected_ready_seconds", 0.0)
            or candidate.score.get("expected_bootstrap_seconds", 0.0)
        )
        deadline_threat = (
            time.time() + max(60.0, expected - elapsed)
            + self.controller.policy.stability_barrier_seconds
            >= self.controller.deadline_epoch
        )
        no_progress_warning = (
            elapsed >= self.controller.policy.hedge_minimum_elapsed_seconds
            and float(risk["no_progress_seconds"])
            >= self.controller.policy.hedge_warning_no_progress_seconds
        )
        empirical_risk = (
            elapsed >= self.controller.policy.hedge_minimum_elapsed_seconds
            and float(
                candidate.score.get("empirical_probability_healthy_ready", 1.0)
            )
            < self.controller.policy.hedge_probability_threshold
        )
        parent_behind = (
            self.is_parent
            and elapsed >= self.controller.policy.hedge_parent_minimum_elapsed_seconds
            and self.controller.registry.current_ready_count()
            >= self.controller.policy.hedge_parent_ready_role_threshold
        )
        return False, bool(
            deadline_threat or no_progress_warning or empirical_risk or parent_behind
        ), {
            **risk,
            "deadline_threat": deadline_threat,
            "no_progress_warning": no_progress_warning,
            "empirical_risk": empirical_risk,
            "parent_behind": parent_behind,
        }

    def _acquire_generation(
        self,
        *,
        endpoint_generation: int | None,
        expert_endpoints: list[dict[str, Any]] | None,
    ) -> CandidateResult:
        executor = ThreadPoolExecutor(max_workers=2)
        futures: dict[Future[CandidateResult], Candidate] = {}
        hedge_slot_held = False
        try:
            primary = self._next_candidate(
                hedge=False,
                endpoint_generation=endpoint_generation,
                expert_endpoints=expert_endpoints,
            )
            futures[executor.submit(self._bootstrap, primary)] = primary
            while futures and not self.controller.abort_event.is_set():
                done, _pending = wait(
                    futures,
                    timeout=1.0,
                    return_when=FIRST_COMPLETED,
                )
                successful: list[CandidateResult] = []
                for future in done:
                    candidate = futures.pop(future)
                    try:
                        successful.append(future.result())
                    except BaseException as exc:
                        fatal = isinstance(exc, WorkerCompatibilityError)
                        stale = candidate.abort_reason is not None and (
                            "generation" in candidate.abort_reason
                            or "hedge loser" in candidate.abort_reason
                            or "fleet freeze" in candidate.abort_reason
                        )
                        self.controller.record_failure(
                            candidate,
                            exc,
                            phase="BOOTSTRAP",
                            attributable=not stale,
                        )
                        self._destroy_candidate(
                            candidate,
                            candidate.abort_reason
                            or f"bootstrap-failed-{self.group_id}",
                        )
                        if stale:
                            self.controller.selector.release(candidate.offer)
                        else:
                            self.controller.selector.fail(candidate.offer)
                        with self.lock:
                            self.active_candidates.pop(candidate.sequence, None)
                        if candidate.hedge and hedge_slot_held:
                            self.controller.hedges.release()
                            hedge_slot_held = False
                        if fatal:
                            self.controller.abort(
                                "worker compatibility contract failed", exc
                            )
                            raise
                if successful:
                    winner = min(
                        successful,
                        key=lambda result: float(
                            result.candidate.completed_epoch or float("inf")
                        ),
                    )
                    self._promote_fragment_winner(winner.candidate)
                    for result in successful:
                        if result is winner:
                            continue
                        result.candidate.abort_reason = "simultaneous hedge loser"
                        result.candidate.abort_event.set()
                        self._destroy_candidate(
                            result.candidate, "simultaneous-healthy-hedge-loser"
                        )
                        self.controller.selector.release(result.candidate.offer)
                        self.controller.emit(
                            "HEDGE_LOSER_DESTROYED",
                            instance_group_id=self.group_id,
                            candidate_id=result.candidate.candidate_id,
                            instance_id=result.candidate.instance_id,
                            losing_candidate_was_primary=(
                                not result.candidate.hedge
                            ),
                            winning_candidate_id=winner.candidate.candidate_id,
                        )
                    for _future, loser in list(futures.items()):
                        loser.abort_reason = "hedge loser after first healthy READY"
                        loser.abort_event.set()
                        self._destroy_candidate(loser, "hedge-loser-after-ready")
                        self.controller.selector.release(loser.offer)
                        self.controller.emit(
                            "HEDGE_LOSER_DESTROYED",
                            instance_group_id=self.group_id,
                            candidate_id=loser.candidate_id,
                            instance_id=loser.instance_id,
                            losing_candidate_was_primary=not loser.hedge,
                            winning_candidate_id=winner.candidate.candidate_id,
                        )
                    if winner.candidate.hedge:
                        self.controller.emit(
                            "HEDGE_WON",
                            instance_group_id=self.group_id,
                            candidate_id=winner.candidate.candidate_id,
                            instance_id=winner.candidate.instance_id,
                        )
                    elif any(candidate.hedge for candidate in futures.values()):
                        self.controller.emit(
                            "HEDGE_WON",
                            instance_group_id=self.group_id,
                            candidate_id=winner.candidate.candidate_id,
                            instance_id=winner.candidate.instance_id,
                            winner_was_primary=True,
                        )
                    if hedge_slot_held:
                        self.controller.hedges.release()
                        hedge_slot_held = False
                    return winner
                primary_candidate = next(
                    (candidate for candidate in futures.values() if not candidate.hedge),
                    None,
                )
                hedge_present = any(candidate.hedge for candidate in futures.values())
                if primary_candidate is not None:
                    hard_dud, hedge_needed, risk = self._candidate_risk(primary_candidate)
                    if hard_dud:
                        primary_candidate.abort_reason = (
                            "fatal bootstrap marker"
                            if risk.get("fatal_marker")
                            else "hard no-progress timeout"
                        )
                        primary_candidate.abort_event.set()
                        self.controller.emit(
                            "TIMEOUT",
                            instance_group_id=self.group_id,
                            candidate_id=primary_candidate.candidate_id,
                            instance_id=primary_candidate.instance_id,
                            machine_id=primary_candidate.offer.machine_id,
                            operation="bootstrap_readiness",
                            no_progress_seconds=risk.get("no_progress_seconds"),
                            reason=primary_candidate.abort_reason,
                        )
                        self._destroy_candidate(
                            primary_candidate,
                            primary_candidate.abort_reason,
                        )
                    elif hedge_needed and not hedge_present and not hedge_slot_held:
                        if self.controller.hedges.acquire():
                            try:
                                hedge = self._next_candidate(
                                    hedge=True,
                                    endpoint_generation=endpoint_generation,
                                    expert_endpoints=expert_endpoints,
                                )
                            except (NoCompatibleOffer, AcquisitionBudgetExceeded) as exc:
                                self.controller.hedges.release()
                                self.controller.emit(
                                    "HEDGE_SKIPPED",
                                    instance_group_id=self.group_id,
                                    primary_candidate_id=primary_candidate.candidate_id,
                                    reason=f"{type(exc).__name__}: {exc}",
                                    trigger=risk,
                                )
                            else:
                                hedge_slot_held = True
                                futures[executor.submit(self._bootstrap, hedge)] = hedge
                                self.controller.emit(
                                    "HEDGE_STARTED",
                                    instance_group_id=self.group_id,
                                    primary_candidate_id=primary_candidate.candidate_id,
                                    hedge_candidate_id=hedge.candidate_id,
                                    primary_instance_id=primary_candidate.instance_id,
                                    primary_machine_id=primary_candidate.offer.machine_id,
                                    hedge_offer_id=hedge.offer.offer_id,
                                    hedge_machine_id=hedge.offer.machine_id,
                                    trigger=risk,
                                )
            raise RuntimeError("group acquisition aborted before a healthy winner")
        finally:
            for candidate in futures.values():
                candidate.abort_reason = candidate.abort_reason or "group acquisition ending"
                candidate.abort_event.set()
            executor.shutdown(wait=True, cancel_futures=False)
            if hedge_slot_held:
                self.controller.hedges.release()

    def run(self) -> None:
        try:
            while (
                time.time() < self.controller.deadline_epoch
                and not self.controller.abort_event.is_set()
                and not self.controller.freeze_event.is_set()
            ):
                self.invalidate_event.clear()
                self.invalidation_reason = None
                self.invalidation_attributable = True
                endpoint_generation: int | None = None
                expert_endpoints: list[dict[str, Any]] | None = None
                if self.is_parent:
                    endpoint_generation, expert_endpoints = (
                        self.controller.fragment_endpoints.wait_for_complete(
                            deadline_epoch=self.controller.deadline_epoch,
                            abort_event=self.controller.abort_event,
                        )
                    )
                    self.controller.emit(
                        "PARENT_GENERATION_STARTED",
                        instance_group_id=self.group_id,
                        fragment_endpoint_generation=endpoint_generation,
                        expert_endpoints=expert_endpoints,
                        dependency_boundary="FOUR_CURRENT_PUBLIC_FRAGMENT_MAPPINGS",
                    )
                try:
                    winner = self._acquire_generation(
                        endpoint_generation=endpoint_generation,
                        expert_endpoints=expert_endpoints,
                    )
                except NoCompatibleOffer as exc:
                    self.last_error = {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "timestamp_utc": utc_now(),
                    }
                    self.transition("NO_INSTANCE", marketplace_blocker=str(exc))
                    if self.controller.abort_event.wait(
                        self.controller.policy.dynamic_refresh_retry_seconds
                    ):
                        break
                    continue
                except AcquisitionBudgetExceeded as exc:
                    self.controller.abort("acquisition cost guard refused launch", exc)
                    break
                except WorkerCompatibilityError:
                    raise
                except BaseException as exc:
                    self.last_error = {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "timestamp_utc": utc_now(),
                    }
                    self.transition("REPLACING", reason=str(exc))
                    if self.controller.abort_event.wait(0.25):
                        break
                    continue
                if self.is_parent and not self.controller.fragment_endpoints.matches(
                    winner.candidate.endpoint_generation
                ):
                    winner.candidate.abort_reason = "parent endpoint generation became stale"
                    self.controller.emit(
                        "PARENT_GENERATION_STALE",
                        instance_group_id=self.group_id,
                        instance_id=winner.candidate.instance_id,
                        parent_fragment_endpoint_generation=(
                            winner.candidate.endpoint_generation
                        ),
                        current_fragment_endpoint_generation=(
                            self.controller.fragment_endpoints.generation
                        ),
                    )
                    self._destroy_candidate(winner.candidate, "stale-parent-generation")
                    self.controller.selector.release(winner.candidate.offer)
                    continue
                with self.lock:
                    self.current_candidate = winner.candidate
                    self.active_candidates = {
                        winner.candidate.sequence: winner.candidate
                    }
                self.controller.registry.publish(
                    self.group_id,
                    winner.workers,
                    parent_generation=winner.candidate.endpoint_generation,
                )
                self.transition(
                    "READY_HEALTHY",
                    candidate_id=winner.candidate.candidate_id,
                    instance_id=winner.candidate.instance_id,
                    worker_ids=[worker.worker_id for worker in winner.workers],
                    fragment_endpoint_generation=winner.candidate.endpoint_generation,
                )
                while not (
                    self.invalidate_event.wait(timeout=0.5)
                    or self.controller.abort_event.is_set()
                    or self.controller.freeze_event.is_set()
                ):
                    pass
                if self.controller.freeze_event.is_set():
                    return
                reason = self.invalidation_reason or "fleet acquisition aborted"
                attributable = self.invalidation_attributable
                self.controller.registry.remove(self.group_id)
                if self.is_parent and "generation" in reason:
                    self.controller.emit(
                        "PARENT_GENERATION_STALE",
                        instance_group_id=self.group_id,
                        instance_id=winner.candidate.instance_id,
                        parent_fragment_endpoint_generation=(
                            winner.candidate.endpoint_generation
                        ),
                        current_fragment_endpoint_generation=(
                            self.controller.fragment_endpoints.generation
                        ),
                        reason=reason,
                    )
                self._destroy_candidate(winner.candidate, reason)
                if attributable:
                    self.controller.selector.fail(winner.candidate.offer)
                else:
                    self.controller.selector.release(winner.candidate.offer)
                with self.lock:
                    self.current_candidate = None
                    self.active_candidates.clear()
                self.transition("REPLACING", reason=reason)
        except BaseException as exc:
            self.last_error = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp_utc": utc_now(),
            }
            if not isinstance(exc, NoCompatibleOffer):
                self.controller.abort(f"group supervisor failed: {self.group_id}", exc)
        finally:
            if not self.controller.freeze_event.is_set():
                self.controller.registry.remove(self.group_id)


class FleetAcquisitionController:
    """Coordinate long-lived group supervisors through a simultaneous freeze."""

    def __init__(
        self,
        *,
        run_id: str,
        client: VastClient,
        ledger: AppendOnlyLifecycleLedger,
        rows: Mapping[str, dict[str, Any]],
        groups: Mapping[str, dict[str, Any]],
        specs: Mapping[str, dict[str, Any]],
        parent_id: str,
        image_reference: str,
        image_digest: str,
        material: Mapping[str, Any],
        watchdog_receipt: Path,
        go_receipt: Path,
        stage_root: Path,
        policy: AcquisitionPolicy,
        history_by_machine: Mapping[int, Mapping[str, Any]],
        emit: EventCallback | None = None,
        create_worker_group_instance_fn: Callable[..., int] = create_worker_group_instance,
        wait_for_public_endpoint_fn: Callable[..., tuple[str, int]] = wait_for_public_endpoint,
        wait_for_worker_fn: Callable[..., LiveWorker] = wait_for_worker,
        probe_worker_liveness_fn: Callable[..., dict[str, Any]] = probe_worker_liveness,
        time_fn: Callable[[], float] = time.time,
        deadline_epoch: float | None = None,
    ) -> None:
        self.run_id = run_id
        self.client = client
        self.ledger = ledger
        self.rows = dict(rows)
        self.groups = dict(groups)
        self.specs = dict(specs)
        self.parent_id = parent_id
        self.parent_group_id = str(rows[parent_id]["instance_group_id"])
        self.image_reference = image_reference
        self.image_digest = image_digest
        self.material = dict(material)
        self.credential = Path(str(material["credential_path"])).read_bytes()
        self.certificate = Path(str(material["certificate_path"]))
        self.watchdog_receipt = watchdog_receipt
        self.go_receipt = go_receipt
        self.stage_root = stage_root
        self.policy = policy
        self.emit_callback = emit or (lambda _event_type, **_fields: None)
        self.create_group_instance_fn = create_worker_group_instance_fn
        self.wait_for_public_endpoint_fn = wait_for_public_endpoint_fn
        self.wait_for_worker_fn = wait_for_worker_fn
        self.probe_worker_liveness_fn = probe_worker_liveness_fn
        self.time_fn = time_fn
        self.started_epoch = time_fn()
        self.deadline_epoch = min(
            self.started_epoch + policy.acquisition_window_seconds,
            float(deadline_epoch)
            if deadline_epoch is not None
            else self.started_epoch + policy.acquisition_window_seconds,
        )
        self.abort_event = threading.Event()
        self.freeze_event = threading.Event()
        self.abort_reason: str | None = None
        self.abort_error: dict[str, Any] | None = None
        self.create_semaphore = threading.Semaphore(policy.maximum_create_concurrency)
        self.group_indexes = {
            group_id: index for index, group_id in enumerate(sorted(groups))
        }
        fragment_ids = sorted(
            worker_id
            for worker_id, spec in specs.items()
            if spec["role"] == "SUB_LAYER_WORKER"
        )
        self.fragment_endpoints = FragmentEndpointCoordinator(fragment_ids, self.emit)
        self.registry = ReadyRegistry(set(rows), self.parent_group_id)
        self.cost_guard = CostGuard(
            policy.maximum_total_acquisition_cost_usd,
            policy.hedge_projected_cost_reserve_usd,
        )
        self.progress = ProviderProgressTracker(policy, self.emit)
        self.hedges = HedgeLimiter(policy.maximum_concurrent_hedges)
        self.selector = OfferSelector(
            client=client,
            policy=policy,
            groups=self.groups,
            history_by_machine=history_by_machine,
            audit_path=stage_root / "dynamic-offer-refreshes.jsonl",
            emit=self.emit,
        )
        self.supervisors = {
            group_id: GroupSupervisor(controller=self, group_id=group_id)
            for group_id in sorted(groups)
        }
        self.fragment_endpoints.set_parent_invalidator(
            lambda reason: self.supervisors[self.parent_group_id].invalidate(
                reason, attributable=False
            )
        )
        self.created_lock = threading.Lock()
        self.created_candidates: list[dict[str, Any]] = []
        self.failure_lock = threading.Lock()
        self.failures: list[dict[str, Any]] = []
        self.health_failures: dict[str, int] = {}
        self.missing_provider_polls: dict[str, int] = {}
        self.health_rounds = 0
        self.healthy_rounds_during_barrier = 0
        self.peak_ready_count = 0
        self.barrier_started_epoch: float | None = None
        self.barrier_resets = 0

    def emit(self, event_type: str, **fields: Any) -> None:
        self.emit_callback(event_type, **fields)

    def abort(self, reason: str, exc: BaseException | None = None) -> None:
        if self.abort_event.is_set():
            return
        self.abort_reason = reason
        if exc is not None:
            self.abort_error = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        self.abort_event.set()
        self.emit("ACQUISITION_ABORT_REQUESTED", reason=reason, error=self.abort_error)

    def record_created(self, candidate: Candidate) -> None:
        with self.created_lock:
            self.created_candidates.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "instance_group_id": candidate.group_id,
                    "instance_id": candidate.instance_id,
                    "offer_id": candidate.offer.offer_id,
                    "machine_id": candidate.offer.machine_id,
                    "gpu_count": candidate.offer.gpu_count,
                    "hedge": candidate.hedge,
                    "fragment_endpoint_generation": candidate.endpoint_generation,
                    "created_epoch": candidate.created_epoch,
                }
            )

    def record_failure(
        self,
        candidate: Candidate,
        exc: BaseException,
        *,
        phase: str,
        attributable: bool,
    ) -> None:
        row = {
            "timestamp_utc": utc_now(),
            "phase": phase,
            "instance_group_id": candidate.group_id,
            "candidate_id": candidate.candidate_id,
            "instance_id": candidate.instance_id,
            "offer_id": candidate.offer.offer_id,
            "machine_id": candidate.offer.machine_id,
            "hedge": candidate.hedge,
            "fragment_endpoint_generation": candidate.endpoint_generation,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "machine_attributable": attributable,
        }
        with self.failure_lock:
            self.failures.append(row)
        self.emit("ERROR", operation="fleet_acquisition", **row)
        if candidate.instance_id is not None:
            failure_log = self.stage_root / "failed-group-logs"
            failure_log.mkdir(parents=True, exist_ok=True)
            try:
                log = self.client.logs(candidate.instance_id, tail=1000)
            except BaseException as log_exc:
                log = f"LOG_RETRIEVAL_FAILED: {type(log_exc).__name__}: {log_exc}\n"
            (failure_log / f"{candidate.candidate_id}.log").write_text(
                log, encoding="utf-8"
            )

    def _validate_ready_health(
        self, provider_rows: dict[int, dict[str, Any]]
    ) -> bool:
        groups, _parent_generation, _version = self.registry.snapshot()
        if not groups:
            return False
        invalid: dict[str, tuple[str, bool]] = {}
        probes: dict[Future[dict[str, Any]], tuple[str, LiveWorker]] = {}
        with ThreadPoolExecutor(
            max_workers=max(1, self.policy.health_probe_concurrency)
        ) as pool:
            for group_id, workers in groups.items():
                instance_ids = {worker.instance_id for worker in workers}
                if len(instance_ids) != 1:
                    invalid[group_id] = ("group spans multiple provider instances", True)
                    continue
                instance_id = next(iter(instance_ids))
                row = provider_rows.get(instance_id)
                if row is None:
                    missing = self.missing_provider_polls.get(group_id, 0) + 1
                    self.missing_provider_polls[group_id] = missing
                    if missing >= self.policy.provider_missing_poll_limit:
                        invalid[group_id] = (
                            f"provider omitted READY instance for {missing} polls",
                            True,
                        )
                    continue
                self.missing_provider_polls[group_id] = 0
                for worker in workers:
                    future = pool.submit(
                        self.probe_worker_liveness_fn,
                        client=self.client,
                        worker=worker,
                        credential=self.credential,
                        certificate=self.certificate,
                        image_digest=self.image_digest,
                        provider_row=row,
                        timeout_seconds=self.policy.ready_health_probe_seconds,
                    )
                    probes[future] = (group_id, worker)
            successful_by_group: dict[str, int] = {}
            for future in as_completed(probes):
                group_id, worker = probes[future]
                try:
                    future.result()
                    successful_by_group[group_id] = (
                        successful_by_group.get(group_id, 0) + 1
                    )
                except BaseException as exc:
                    immediate = isinstance(exc, WorkerCompatibilityError)
                    failures = self.health_failures.get(group_id, 0) + 1
                    self.health_failures[group_id] = failures
                    if immediate or failures >= self.policy.health_failure_limit:
                        invalid[group_id] = (
                            f"authenticated READY liveness failed: {type(exc).__name__}: {exc}",
                            True,
                        )
            for group_id, workers in groups.items():
                if group_id in invalid or self.missing_provider_polls.get(group_id, 0):
                    continue
                if successful_by_group.get(group_id, 0) == len(workers):
                    self.health_failures[group_id] = 0
        for group_id, (reason, attributable) in invalid.items():
            self.supervisors[group_id].invalidate(reason, attributable=attributable)
        self.health_rounds += 1
        fully_healthy = bool(groups) and all(
            group_id not in invalid
            and self.missing_provider_polls.get(group_id, 0) == 0
            and successful_by_group.get(group_id, 0) == len(workers)
            for group_id, workers in groups.items()
        )
        return not invalid and fully_healthy

    def _simultaneously_ready(self) -> bool:
        workers = self.registry.workers()
        _groups, parent_generation, _version = self.registry.snapshot()
        return (
            set(workers) == set(self.rows)
            and len(workers) == len(self.rows)
            and self.fragment_endpoints.matches(parent_generation)
        )

    def _barrier_update(self, health_round_succeeded: bool) -> bool:
        ready_count = self.registry.current_ready_count()
        self.peak_ready_count = max(self.peak_ready_count, ready_count)
        simultaneous = self._simultaneously_ready()
        now = self.time_fn()
        if not simultaneous:
            if self.barrier_started_epoch is not None:
                self.barrier_resets += 1
                self.emit(
                    "FLEET_STABILITY_BARRIER_RESET",
                    current_ready_role_count=ready_count,
                    reason="simultaneous readiness lost",
                )
            self.barrier_started_epoch = None
            self.healthy_rounds_during_barrier = 0
            return False
        if self.barrier_started_epoch is None:
            self.barrier_started_epoch = now
            self.healthy_rounds_during_barrier = 0
            self.emit(
                "FLEET_STABILITY_BARRIER_STARTED",
                current_ready_role_count=ready_count,
                required_role_count=len(self.rows),
                barrier_duration_seconds=self.policy.stability_barrier_seconds,
                required_authenticated_health_rounds=(
                    self.policy.minimum_stability_health_rounds
                ),
                fragment_endpoint_generation=self.fragment_endpoints.generation,
            )
        if health_round_succeeded:
            self.healthy_rounds_during_barrier += 1
        elapsed = now - self.barrier_started_epoch
        return (
            elapsed >= self.policy.stability_barrier_seconds
            and self.healthy_rounds_during_barrier
            >= self.policy.minimum_stability_health_rounds
            and self._simultaneously_ready()
        )

    def acquire(self) -> tuple[list[LiveWorker], dict[str, Any]]:
        self.emit(
            "ACQUISITION_POLICY_APPLIED",
            state_machine=[
                "NO_INSTANCE",
                "CREATING",
                "BOOTSTRAPPING",
                "READY",
                "READY_HEALTHY",
                "REPLACING",
            ],
            acquisition_deadline_epoch=self.deadline_epoch,
            acquisition_window_seconds=self.policy.acquisition_window_seconds,
            stability_barrier_seconds=self.policy.stability_barrier_seconds,
            parent_dependency_boundary="FOUR_CURRENT_PUBLIC_FRAGMENT_MAPPINGS",
            dynamic_alternate_refresh=self.policy.dynamic_alternate_refresh_enabled,
            maximum_concurrent_hedges=self.policy.maximum_concurrent_hedges,
        )
        for supervisor in self.supervisors.values():
            supervisor.start()
        next_provider_poll = 0.0
        success = False
        try:
            while self.time_fn() < self.deadline_epoch and not self.abort_event.is_set():
                now = self.time_fn()
                health_succeeded = False
                if now >= next_provider_poll:
                    next_provider_poll = now + self.policy.provider_poll_seconds
                    try:
                        provider_rows = self.progress.poll(self.client)
                    except BaseException as exc:
                        self.emit(
                            "RETRY",
                            operation="provider_fleet_liveness_query",
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                    else:
                        health_succeeded = self._validate_ready_health(provider_rows)
                if self._barrier_update(health_succeeded):
                    success = True
                    self.freeze_event.set()
                    workers = self.registry.workers()
                    self.emit(
                        "FLEET_SIMULTANEOUS_READY",
                        current_ready_role_count=len(workers),
                        required_role_count=len(self.rows),
                        stability_barrier_seconds=self.policy.stability_barrier_seconds,
                        authenticated_health_rounds=(
                            self.healthy_rounds_during_barrier
                        ),
                        fragment_endpoint_generation=(
                            self.fragment_endpoints.generation
                        ),
                    )
                    break
                self.abort_event.wait(0.25)
            if not success and not self.abort_event.is_set():
                self.abort("evidence-based acquisition window expired")
        finally:
            if not success:
                self.abort_event.set()
                for supervisor in self.supervisors.values():
                    supervisor.invalidate(
                        self.abort_reason or "fleet acquisition ended", attributable=False
                    )
            for supervisor in self.supervisors.values():
                supervisor.join()
        workers = self.registry.workers()
        receipt = self.receipt(success=success)
        if not success:
            raise RuntimeError(
                "E025 acquisition did not reach 97 simultaneous healthy roles: "
                + json.dumps(receipt["terminal_blocker"], sort_keys=True)
            )
        return sorted(workers.values(), key=lambda worker: worker.worker_id), receipt

    def receipt(self, *, success: bool) -> dict[str, Any]:
        groups, parent_generation, _version = self.registry.snapshot()
        workers = self.registry.workers()
        supervisor_states = {
            group_id: {
                "state": supervisor.state,
                "last_error": supervisor.last_error,
                "current_candidate_id": (
                    supervisor.current_candidate.candidate_id
                    if supervisor.current_candidate is not None
                    else None
                ),
            }
            for group_id, supervisor in sorted(self.supervisors.items())
        }
        terminal_blocker = None
        if not success:
            terminal_blocker = {
                "reason": self.abort_reason or "acquisition deadline expired",
                "error": self.abort_error,
                "current_ready_role_count": len(workers),
                "missing_worker_ids": sorted(set(self.rows) - set(workers)),
                "marketplace_blockers": self.selector.receipt()[
                    "last_marketplace_blockers"
                ],
            }
        return {
            "schema_version": "experiment-025-resilient-acquisition-v1",
            "generated_at_utc": utc_now(),
            "status": "PASS" if success else "INCOMPLETE",
            "run_id": self.run_id,
            "acquisition_started_epoch": self.started_epoch,
            "acquisition_deadline_epoch": self.deadline_epoch,
            "acquisition_window_seconds": self.policy.acquisition_window_seconds,
            "current_ready_role_count": len(workers),
            "peak_ready_role_count": self.peak_ready_count,
            "ready_group_count": len(groups),
            "required_role_count": len(self.rows),
            "fragment_endpoints": self.fragment_endpoints.receipt(),
            "parent_fragment_endpoint_generation": parent_generation,
            "parent_generation_matches_fragments": self.fragment_endpoints.matches(
                parent_generation
            ),
            "stability_barrier": {
                "duration_seconds": self.policy.stability_barrier_seconds,
                "minimum_health_rounds": self.policy.minimum_stability_health_rounds,
                "completed_health_rounds": self.healthy_rounds_during_barrier,
                "resets": self.barrier_resets,
            },
            "total_health_rounds": self.health_rounds,
            "supervisors": supervisor_states,
            "offer_selection": self.selector.receipt(),
            "hedging": self.hedges.receipt(),
            "cost_guard": self.cost_guard.receipt(),
            "created_candidates": list(self.created_candidates),
            "failures": list(self.failures),
            "terminal_blocker": terminal_blocker,
        }


__all__ = [
    "AcquisitionBudgetExceeded",
    "AcquisitionPolicy",
    "FleetAcquisitionController",
    "FragmentEndpointCoordinator",
    "NoCompatibleOffer",
    "ReadyRegistry",
]
