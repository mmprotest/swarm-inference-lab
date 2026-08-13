"""Exact E021 provisioning/rollback state machine exercised with fakes in E020."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .vast import atomic_write_json


class ProvisioningState(StrEnum):
    DISCOVER_OFFERS = "DISCOVER_OFFERS"
    PLAN_FLEET = "PLAN_FLEET"
    USER_BUDGET_GATE = "USER_BUDGET_GATE"
    RENT_CONTROLLER_POD = "RENT_CONTROLLER_POD"
    WAIT_CONTROLLER = "WAIT_CONTROLLER"
    DISCOVER_CONTROLLER_ADDRESS = "DISCOVER_CONTROLLER_ADDRESS"
    RENT_WORKER_PODS = "RENT_WORKER_PODS"
    WAIT_INSTANCES = "WAIT_INSTANCES"
    BOOTSTRAP = "BOOTSTRAP"
    DOWNLOAD_SHARDS = "DOWNLOAD_SHARDS"
    VERIFY_MODEL = "VERIFY_MODEL"
    REGISTER_WORKERS = "REGISTER_WORKERS"
    MEASURE_NETWORK = "MEASURE_NETWORK"
    VALIDATE_TOPOLOGY = "VALIDATE_TOPOLOGY"
    READY = "READY"
    RUN_EXPERIMENT = "RUN_EXPERIMENT"
    COLLECT_RESULTS = "COLLECT_RESULTS"
    DESTROY_ALL = "DESTROY_ALL"
    VERIFY_DESTROYED = "VERIFY_DESTROYED"


@dataclass(frozen=True, slots=True)
class RentalRecord:
    instance_id: int
    offer_id: int
    machine_id: int
    creation_time_unix_ns: int
    hourly_rate: float
    label: str
    pod_assignment: str


class RentalLedger:
    """Append-only hash-chained ownership ledger for E021 resources."""

    def __init__(self) -> None:
        self.records: list[RentalRecord] = []
        self.events: list[dict[str, Any]] = []
        self._last_hash = "0" * 64

    def append_created(self, record: RentalRecord) -> None:
        if not record.label.startswith("swarm-e021-"):
            raise ValueError("E021 instance label is mandatory")
        if record.instance_id in self.owned_ids:
            raise ValueError("instance already exists in immutable ledger")
        self.records.append(record)
        self._append_event("CREATED", record.instance_id)

    def append_action(self, action: str, instance_id: int) -> None:
        self.assert_owned(instance_id)
        self._append_event(action, instance_id)

    def _append_event(self, action: str, instance_id: int) -> None:
        body = {
            "sequence": len(self.events),
            "action": action,
            "instance_id": instance_id,
            "timestamp_unix_ns": time.time_ns(),
            "previous_sha256": self._last_hash,
        }
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        body["event_sha256"] = digest
        self.events.append(body)
        self._last_hash = digest

    @property
    def owned_ids(self) -> frozenset[int]:
        return frozenset(record.instance_id for record in self.records)

    def assert_owned(self, instance_id: int) -> None:
        if instance_id not in self.owned_ids:
            raise PermissionError("E021_UNRELATED_INSTANCE_PROTECTED")

    def validate(self) -> bool:
        previous = "0" * 64
        for index, event in enumerate(self.events):
            body = {key: value for key, value in event.items() if key != "event_sha256"}
            if body["sequence"] != index or body["previous_sha256"] != previous:
                return False
            digest = hashlib.sha256(
                json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if digest != event["event_sha256"]:
                return False
            previous = digest
        return all(event["instance_id"] in self.owned_ids for event in self.events)

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "experiment-021-rental-ledger-v1",
            "immutable": True,
            "records": [asdict(record) for record in self.records],
            "events": self.events,
            "valid": self.validate(),
        }


class FakeVastBackend:
    def __init__(self, failure: str | None = None, pods: int = 12) -> None:
        self.failure = failure
        self.pods = pods
        self.created: dict[int, dict[str, Any]] = {}
        self.destroyed: set[int] = set()
        self._next = 10_000

    def create(self, pod: int, *, controller: bool = False) -> RentalRecord:
        if self.failure == "offer_disappears" and not self.created:
            raise RuntimeError("offer disappeared between plan and create")
        if self.failure == "one_pod_fails" and pod == 3:
            raise RuntimeError("pod provisioning failed")
        instance_id = self._next
        self._next += 1
        record = RentalRecord(
            instance_id=instance_id,
            offer_id=20_000 + pod,
            machine_id=30_000 + pod,
            creation_time_unix_ns=time.time_ns(),
            hourly_rate=4.0,
            label=f"swarm-e021-mock-run-pod-{pod:03d}",
            pod_assignment=f"pod-{pod:03d}",
        )
        self.created[instance_id] = {"record": record, "controller": controller}
        return record

    def gate(self, state: ProvisioningState) -> None:
        mapping = {
            "instance_never_ready": ProvisioningState.WAIT_INSTANCES,
            "wrong_gpu_count": ProvisioningState.WAIT_INSTANCES,
            "insufficient_disk": ProvisioningState.BOOTSTRAP,
            "bootstrap_failure": ProvisioningState.BOOTSTRAP,
            "model_download_failure": ProvisioningState.DOWNLOAD_SHARDS,
            "hash_mismatch": ProvisioningState.VERIFY_MODEL,
            "network_too_slow": ProvisioningState.VALIDATE_TOPOLOGY,
            "worker_health_failure": ProvisioningState.REGISTER_WORKERS,
            "controller_crash": ProvisioningState.RUN_EXPERIMENT,
        }
        if mapping.get(self.failure) == state:
            raise RuntimeError(self.failure.replace("_", " "))

    def destroy(self, instance_id: int) -> None:
        if instance_id not in self.created:
            raise PermissionError("attempted to destroy unrelated instance")
        self.destroyed.add(instance_id)


class ProvisioningStateMachine:
    def __init__(
        self,
        backend: FakeVastBackend,
        *,
        max_budget_usd: float = 500.0,
        ledger_path: Path | None = None,
    ) -> None:
        self.backend = backend
        self.max_budget_usd = max_budget_usd
        self.ledger = RentalLedger()
        self.states: list[str] = []
        self.failure: str | None = None
        self.ledger_path = ledger_path

    def _persist_ledger(self) -> None:
        if self.ledger_path is not None:
            atomic_write_json(self.ledger_path, self.ledger.receipt())

    def _state(self, state: ProvisioningState) -> None:
        self.states.append(state.value)
        self.backend.gate(state)

    def run(self) -> dict[str, Any]:
        try:
            for state in (
                ProvisioningState.DISCOVER_OFFERS,
                ProvisioningState.PLAN_FLEET,
                ProvisioningState.USER_BUDGET_GATE,
            ):
                self._state(state)
            self._state(ProvisioningState.RENT_CONTROLLER_POD)
            controller = self.backend.create(0, controller=True)
            self.ledger.append_created(controller)
            self._persist_ledger()
            self._state(ProvisioningState.WAIT_CONTROLLER)
            self._state(ProvisioningState.DISCOVER_CONTROLLER_ADDRESS)
            self._state(ProvisioningState.RENT_WORKER_PODS)
            for pod in range(1, self.backend.pods):
                self.ledger.append_created(self.backend.create(pod))
                self._persist_ledger()
            for state in (
                ProvisioningState.WAIT_INSTANCES,
                ProvisioningState.BOOTSTRAP,
                ProvisioningState.DOWNLOAD_SHARDS,
                ProvisioningState.VERIFY_MODEL,
                ProvisioningState.REGISTER_WORKERS,
                ProvisioningState.MEASURE_NETWORK,
                ProvisioningState.VALIDATE_TOPOLOGY,
                ProvisioningState.READY,
                ProvisioningState.RUN_EXPERIMENT,
                ProvisioningState.COLLECT_RESULTS,
            ):
                self._state(state)
        except Exception as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
        finally:
            self.states.append(ProvisioningState.DESTROY_ALL.value)
            for instance_id in sorted(self.ledger.owned_ids):
                self.backend.destroy(instance_id)
                self.ledger.append_action("DESTROYED", instance_id)
                self._persist_ledger()
            self.states.append(ProvisioningState.VERIFY_DESTROYED.value)
        created = set(self.backend.created)
        destroyed = set(self.backend.destroyed)
        clean = created == destroyed == set(self.ledger.owned_ids)
        return {
            "schema_version": "experiment-020-mock-provisioning-v1",
            "scenario": self.backend.failure or "success",
            "status": (
                "PASS" if self.failure is None and clean else "ABORTED_CLEAN" if clean else "FAIL"
            ),
            "states": self.states,
            "failure": self.failure,
            "created_instance_ids": sorted(created),
            "destroyed_instance_ids": sorted(destroyed),
            "no_orphans": clean,
            "ledger_valid": self.ledger.validate(),
            "ledger": self.ledger.receipt(),
        }


class CostKillSwitch:
    def __init__(self, max_budget_usd: float, teardown: Callable[[], None]) -> None:
        if max_budget_usd <= 0:
            raise ValueError("positive budget is required")
        self.max_budget_usd = float(max_budget_usd)
        self.teardown = teardown
        self.triggered = False

    def check(self, rates_per_hour: list[float], elapsed_hours: float) -> float:
        estimated = sum(rates_per_hour) * max(0.0, elapsed_hours)
        if estimated >= self.max_budget_usd and not self.triggered:
            self.triggered = True
            self.teardown()
        return estimated


FAILURE_SCENARIOS = (
    "offer_disappears",
    "one_pod_fails",
    "instance_never_ready",
    "wrong_gpu_count",
    "insufficient_disk",
    "bootstrap_failure",
    "model_download_failure",
    "hash_mismatch",
    "network_too_slow",
    "worker_health_failure",
    "controller_crash",
)


def exercise_mock_provisioning() -> dict[str, Any]:
    success = ProvisioningStateMachine(FakeVastBackend()).run()
    failures = [
        ProvisioningStateMachine(FakeVastBackend(scenario)).run()
        for scenario in FAILURE_SCENARIOS
    ]
    return {
        "schema_version": "experiment-020-mock-provisioning-suite-v1",
        "status": (
            "PASS"
            if success["status"] == "PASS"
            and all(row["status"] == "ABORTED_CLEAN" for row in failures)
            and all(row["ledger_valid"] and row["no_orphans"] for row in failures)
            else "FAIL"
        ),
        "success": success,
        "failures": failures,
    }


__all__ = [
    "FAILURE_SCENARIOS",
    "CostKillSwitch",
    "FakeVastBackend",
    "ProvisioningState",
    "ProvisioningStateMachine",
    "RentalLedger",
    "RentalRecord",
    "exercise_mock_provisioning",
]
