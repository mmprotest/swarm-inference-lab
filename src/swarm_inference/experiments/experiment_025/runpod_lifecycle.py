"""RunPod Pod-group lifecycle state machine and deterministic E025 simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .io import canonical_sha256, utc_now
from .runpod_planning import RUN_ID


class PodLifecycleState(StrEnum):
    PLANNED = "PLANNED"
    CREATE_REQUESTED = "CREATE_REQUESTED"
    CREATED = "CREATED"
    CONTAINER_STARTING = "CONTAINER_STARTING"
    MODEL_ACQUISITION = "MODEL_ACQUISITION"
    PACKAGE_READY = "PACKAGE_READY"
    GPU_LOAD = "GPU_LOAD"
    WORKER_LISTENING = "WORKER_LISTENING"
    WORKER_READY = "WORKER_READY"
    READY_HEALTHY = "READY_HEALTHY"
    FROZEN_FOR_INFERENCE = "FROZEN_FOR_INFERENCE"
    STALLED = "STALLED"
    UNHEALTHY = "UNHEALTHY"
    REPLACEMENT_REQUIRED = "REPLACEMENT_REQUIRED"
    TERMINATING = "TERMINATING"
    TERMINATED = "TERMINATED"


FORWARD_STATES = (
    PodLifecycleState.PLANNED,
    PodLifecycleState.CREATE_REQUESTED,
    PodLifecycleState.CREATED,
    PodLifecycleState.CONTAINER_STARTING,
    PodLifecycleState.MODEL_ACQUISITION,
    PodLifecycleState.PACKAGE_READY,
    PodLifecycleState.GPU_LOAD,
    PodLifecycleState.WORKER_LISTENING,
    PodLifecycleState.WORKER_READY,
    PodLifecycleState.READY_HEALTHY,
    PodLifecycleState.FROZEN_FOR_INFERENCE,
)
RECOVERY_STATES = frozenset(
    {
        PodLifecycleState.STALLED,
        PodLifecycleState.UNHEALTHY,
        PodLifecycleState.REPLACEMENT_REQUIRED,
    }
)


@dataclass(slots=True)
class WorkerLifecycle:
    worker_id: str
    gpu_slot: int
    port: int
    state: PodLifecycleState = PodLifecycleState.PLANNED
    progress: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PodLifecycle:
    logical_pod_id: str
    worker_ids: tuple[str, ...]
    state: PodLifecycleState = PodLifecycleState.PLANNED
    pod_id: str | None = None
    machine_id: str | None = None
    datacenter_id: str | None = None
    parent_endpoint_generation: int | None = None
    workers: dict[str, WorkerLifecycle] = field(default_factory=dict)
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def transition(
        self,
        state: PodLifecycleState,
        *,
        detail: dict[str, Any] | None = None,
    ) -> None:
        if self.state is PodLifecycleState.TERMINATED:
            raise ValueError("terminated RunPod Pod cannot transition")
        if state in FORWARD_STATES and self.state in FORWARD_STATES:
            if FORWARD_STATES.index(state) < FORWARD_STATES.index(self.state):
                raise ValueError(f"RunPod lifecycle regression {self.state} -> {state}")
        elif state not in RECOVERY_STATES and state not in {
            PodLifecycleState.TERMINATING,
            PodLifecycleState.TERMINATED,
        }:
            raise ValueError(f"invalid RunPod lifecycle transition {self.state} -> {state}")
        self.state = state
        self.transitions.append(
            {"timestamp": utc_now(), "state": state.value, "detail": detail or {}}
        )

    def mark_worker(
        self,
        worker_id: str,
        state: PodLifecycleState,
        **progress: Any,
    ) -> None:
        worker = self.workers[worker_id]
        worker.state = state
        worker.progress.update(progress)
        if state in {PodLifecycleState.UNHEALTHY, PodLifecycleState.STALLED}:
            # The existing supervisor terminates every sibling when one child exits.
            for sibling in self.workers.values():
                sibling.state = PodLifecycleState.UNHEALTHY
            self.transition(
                PodLifecycleState.UNHEALTHY,
                detail={
                    "failed_worker_id": worker_id,
                    "semantics": "ONE_CHILD_FAILURE_INVALIDATES_WHOLE_POD_GROUP",
                },
            )
            self.transition(PodLifecycleState.REPLACEMENT_REQUIRED)

    @property
    def fully_ready(self) -> bool:
        return bool(self.workers) and all(
            worker.state is PodLifecycleState.READY_HEALTHY for worker in self.workers.values()
        )


@dataclass(frozen=True, slots=True)
class FragmentIdentity:
    worker_id: str
    pod_id: str
    machine_id: str
    host: str
    port: int


class FragmentEndpointCoordinator:
    """Track four physical fragment machines and parent endpoint generations."""

    def __init__(self) -> None:
        self.identities: dict[str, FragmentIdentity] = {}
        self.generation = 0
        self.replacements: list[dict[str, Any]] = []

    def publish(self, identity: FragmentIdentity) -> tuple[bool, str | None]:
        duplicate = next(
            (
                value
                for worker_id, value in self.identities.items()
                if worker_id != identity.worker_id and value.machine_id == identity.machine_id
            ),
            None,
        )
        if duplicate is not None:
            self.replacements.append(
                {
                    "rejected_worker_id": identity.worker_id,
                    "rejected_pod_id": identity.pod_id,
                    "duplicate_machine_id": identity.machine_id,
                    "conflicting_worker_id": duplicate.worker_id,
                    "expensive_model_download_started": False,
                }
            )
            return False, duplicate.worker_id
        prior = self.identities.get(identity.worker_id)
        if prior != identity:
            self.identities[identity.worker_id] = identity
            self.generation += 1
            if prior is not None:
                self.replacements.append(
                    {
                        "replaced_worker_id": identity.worker_id,
                        "old_pod_id": prior.pod_id,
                        "old_machine_id": prior.machine_id,
                        "new_pod_id": identity.pod_id,
                        "new_machine_id": identity.machine_id,
                        "generation": self.generation,
                    }
                )
        return True, None

    def retire(self, worker_id: str) -> None:
        if worker_id in self.identities:
            del self.identities[worker_id]
            self.generation += 1

    @property
    def complete(self) -> bool:
        return (
            len(self.identities) == 4
            and len({identity.machine_id for identity in self.identities.values()}) == 4
        )

    def endpoints(self) -> list[dict[str, Any]]:
        if not self.complete:
            raise RuntimeError("four distinct RunPod fragment machines are not frozen")
        return [
            {
                "worker_id": identity.worker_id,
                "worker_index": int(identity.worker_id.rsplit("-", maxsplit=1)[1]),
                "host": identity.host,
                "port": identity.port,
                "fragment_endpoint_generation": self.generation,
                "timeout_seconds": 180.0,
                "pod_id": identity.pod_id,
                "machine_id": identity.machine_id,
            }
            for identity in sorted(self.identities.values(), key=lambda value: value.worker_id)
        ]


class FakeRunPodProvider:
    """In-memory provider; deliberately has no HTTP transport or credentials."""

    def __init__(self) -> None:
        self.sequence = 0
        self.live: dict[str, dict[str, Any]] = {}
        self.created: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.cost_usd = 0.0

    def create(
        self,
        manifest: dict[str, Any],
        *,
        forced_machine_id: str | None = None,
    ) -> dict[str, Any]:
        self.sequence += 1
        pod_id = f"sim-pod-{self.sequence:04d}"
        machine_id = forced_machine_id or f"sim-machine-{self.sequence:04d}"
        hourly_rate = int(manifest["requested_gpu_count"]) * (
            0.18 if manifest["pod_class"] == "FRAGMENT" else 0.50
        )
        simulated_elapsed_hours = 1 / 60
        row = {
            "pod_id": pod_id,
            "machine_id": machine_id,
            "pod_name": manifest["pod_name"],
            "logical_pod_id": manifest["logical_pod_id"],
            "gpu_count": manifest["requested_gpu_count"],
            "datacenter_id": (manifest.get("preferred_datacenters") or ["SIM-DC-1"])[0],
            "public_ip": f"192.0.2.{self.sequence % 250 + 1}",
            "private_identity": f"{pod_id}.runpod.internal",
            "port_mappings": {
                str(port): 50_000 + self.sequence * 10 + index
                for index, port in enumerate(manifest["internal_serving_ports"])
            },
            "status": "CREATED",
            "hourly_rate_usd": hourly_rate,
            "simulated_elapsed_hours": simulated_elapsed_hours,
            "simulated_cost_usd": hourly_rate * simulated_elapsed_hours,
        }
        self.cost_usd += float(row["simulated_cost_usd"])
        self.live[pod_id] = row
        self.created.append(dict(row))
        return row

    def delete(self, pod_id: str) -> None:
        self.live.pop(pod_id, None)
        if pod_id not in self.deleted:
            self.deleted.append(pod_id)


def _event(events: list[dict[str, Any]], event_type: str, **fields: Any) -> None:
    events.append(
        {
            "timestamp": utc_now(),
            "event_type": event_type,
            "provider": "runpod-simulator",
            "evidence_class": "SIMULATED_PROVIDER_CONTROL_FLOW",
            **fields,
        }
    )


def _lifecycle_from_manifest(manifest: dict[str, Any]) -> PodLifecycle:
    workers = {
        str(spec["worker_id"]): WorkerLifecycle(
            worker_id=str(spec["worker_id"]),
            gpu_slot=int(spec["gpu_slot"]),
            port=int(spec["port"]),
        )
        for spec in manifest["worker_specs"]
    }
    return PodLifecycle(
        logical_pod_id=str(manifest["logical_pod_id"]),
        worker_ids=tuple(workers),
        workers=workers,
    )


def _advance_ready(
    lifecycle: PodLifecycle,
    events: list[dict[str, Any]],
) -> None:
    for state in FORWARD_STATES[3:10]:
        lifecycle.transition(state)
        _event(
            events,
            state.value,
            logical_pod_id=lifecycle.logical_pod_id,
            pod_id=lifecycle.pod_id,
        )
        if state is PodLifecycleState.MODEL_ACQUISITION:
            _event(
                events,
                "MODEL_DOWNLOAD_PROGRESS",
                logical_pod_id=lifecycle.logical_pod_id,
                downloaded_bytes=1,
                progress_is_synthetic=True,
            )
    for worker in lifecycle.workers.values():
        worker.state = PodLifecycleState.READY_HEALTHY


def simulate_full_control_flow(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Exercise the planned headline lifecycle without model or provider execution."""

    provider = FakeRunPodProvider()
    events: list[dict[str, Any]] = []
    lifecycles = {
        str(manifest["logical_pod_id"]): _lifecycle_from_manifest(manifest)
        for manifest in manifests
    }
    _event(events, "INVENTORY_CAPTURED", source="DETERMINISTIC_FAKE_INVENTORY")
    _event(events, "PACKING_FROZEN", planned_pod_count=len(manifests))
    fragments = [manifest for manifest in manifests if manifest["pod_class"] == "FRAGMENT"]
    parent = next(manifest for manifest in manifests if manifest["pod_class"] == "LAYER89_PARENT")
    backbone = [
        manifest
        for manifest in manifests
        if manifest["pod_class"] not in {"FRAGMENT", "LAYER89_PARENT"}
    ]
    endpoint_coordinator = FragmentEndpointCoordinator()

    # First fragment allocation succeeds. The second deliberately lands on the
    # same physical machine, is rejected before acquisition, and is replaced.
    first_machine: str | None = None
    for fragment_index, manifest in enumerate(fragments):
        lifecycle = lifecycles[manifest["logical_pod_id"]]
        lifecycle.transition(PodLifecycleState.CREATE_REQUESTED)
        _event(events, "PLANNED_CREATE", logical_pod_id=lifecycle.logical_pod_id)
        forced = first_machine if fragment_index == 1 else None
        allocation = provider.create(manifest, forced_machine_id=forced)
        if fragment_index == 0:
            first_machine = str(allocation["machine_id"])
        lifecycle.pod_id = str(allocation["pod_id"])
        lifecycle.machine_id = str(allocation["machine_id"])
        lifecycle.datacenter_id = str(allocation["datacenter_id"])
        lifecycle.transition(PodLifecycleState.CREATED)
        worker_id = str(manifest["worker_specs"][0]["worker_id"])
        accepted, conflicting = endpoint_coordinator.publish(
            FragmentIdentity(
                worker_id=worker_id,
                pod_id=str(allocation["pod_id"]),
                machine_id=str(allocation["machine_id"]),
                host=str(allocation["private_identity"]),
                port=int(manifest["internal_serving_ports"][0]),
            )
        )
        if not accepted:
            _event(
                events,
                "FRAGMENT_DUPLICATE_MACHINE_REJECTED",
                pod_id=lifecycle.pod_id,
                machine_id=lifecycle.machine_id,
                conflicting_worker_id=conflicting,
                expensive_model_download_started=False,
            )
            lifecycle.transition(PodLifecycleState.REPLACEMENT_REQUIRED)
            lifecycle.transition(PodLifecycleState.TERMINATING)
            provider.delete(str(lifecycle.pod_id))
            lifecycle.transition(PodLifecycleState.TERMINATED)
            lifecycle = _lifecycle_from_manifest(manifest)
            lifecycles[manifest["logical_pod_id"]] = lifecycle
            lifecycle.transition(PodLifecycleState.CREATE_REQUESTED)
            allocation = provider.create(manifest)
            lifecycle.pod_id = str(allocation["pod_id"])
            lifecycle.machine_id = str(allocation["machine_id"])
            lifecycle.datacenter_id = str(allocation["datacenter_id"])
            lifecycle.transition(PodLifecycleState.CREATED)
            accepted, _ = endpoint_coordinator.publish(
                FragmentIdentity(
                    worker_id=worker_id,
                    pod_id=str(allocation["pod_id"]),
                    machine_id=str(allocation["machine_id"]),
                    host=str(allocation["private_identity"]),
                    port=int(manifest["internal_serving_ports"][0]),
                )
            )
            if not accepted:  # pragma: no cover - deterministic simulator invariant
                raise RuntimeError("fake replacement retained a duplicate machine")
        _advance_ready(lifecycle, events)

    if not endpoint_coordinator.complete:
        raise RuntimeError("simulated fragment acquisition did not establish four machines")
    _event(
        events,
        "FRAGMENT_IDENTITIES_FROZEN",
        generation=endpoint_coordinator.generation,
        distinct_machine_ids=4,
    )

    # Parent can be created from stable endpoint identity before waiting on an
    # additional fragment readiness serialization step.
    parent_lifecycle = lifecycles[parent["logical_pod_id"]]
    parent_lifecycle.transition(PodLifecycleState.CREATE_REQUESTED)
    parent_allocation = provider.create(parent)
    parent_lifecycle.pod_id = str(parent_allocation["pod_id"])
    parent_lifecycle.machine_id = str(parent_allocation["machine_id"])
    parent_lifecycle.datacenter_id = str(parent_allocation["datacenter_id"])
    parent_lifecycle.parent_endpoint_generation = endpoint_coordinator.generation
    parent_lifecycle.transition(PodLifecycleState.CREATED)
    _event(
        events,
        "PARENT_ENDPOINT_GENERATION_BOUND",
        generation=endpoint_coordinator.generation,
        endpoints=endpoint_coordinator.endpoints(),
    )
    _advance_ready(parent_lifecycle, events)

    for manifest in backbone:
        lifecycle = lifecycles[manifest["logical_pod_id"]]
        lifecycle.transition(PodLifecycleState.CREATE_REQUESTED)
        allocation = provider.create(manifest)
        lifecycle.pod_id = str(allocation["pod_id"])
        lifecycle.machine_id = str(allocation["machine_id"])
        lifecycle.datacenter_id = str(allocation["datacenter_id"])
        lifecycle.transition(PodLifecycleState.CREATED)
        _advance_ready(lifecycle, events)

    # Exercise whole-Pod replacement when one child fails.
    churn_manifest = next(
        (manifest for manifest in backbone if len(manifest["worker_specs"]) > 1),
        backbone[0],
    )
    churn = lifecycles[churn_manifest["logical_pod_id"]]
    failed_worker = str(churn_manifest["worker_specs"][0]["worker_id"])
    old_pod = str(churn.pod_id)
    churn.mark_worker(failed_worker, PodLifecycleState.UNHEALTHY, reason="SIMULATED_EXIT")
    _event(
        events,
        "WORKER_FAILURE_INVALIDATED_POD_GROUP",
        worker_id=failed_worker,
        pod_id=old_pod,
        sibling_count=len(churn.worker_ids) - 1,
        multi_worker_group_available=len(churn.worker_ids) > 1,
    )
    churn.transition(PodLifecycleState.TERMINATING)
    provider.delete(old_pod)
    churn.transition(PodLifecycleState.TERMINATED)
    replacement = _lifecycle_from_manifest(churn_manifest)
    lifecycles[churn_manifest["logical_pod_id"]] = replacement
    replacement.transition(PodLifecycleState.CREATE_REQUESTED)
    allocation = provider.create(churn_manifest)
    replacement.pod_id = str(allocation["pod_id"])
    replacement.machine_id = str(allocation["machine_id"])
    replacement.datacenter_id = str(allocation["datacenter_id"])
    replacement.transition(PodLifecycleState.CREATED)
    _advance_ready(replacement, events)

    # Exercise fragment churn after a parent is bound. Only the dedicated parent
    # is recycled; healthy unrelated backbone Pods are retained.
    fragment_manifest = fragments[0]
    fragment_lifecycle = lifecycles[fragment_manifest["logical_pod_id"]]
    old_fragment_pod = str(fragment_lifecycle.pod_id)
    worker_id = str(fragment_manifest["worker_specs"][0]["worker_id"])
    endpoint_coordinator.retire(worker_id)
    fragment_lifecycle.transition(PodLifecycleState.REPLACEMENT_REQUIRED)
    fragment_lifecycle.transition(PodLifecycleState.TERMINATING)
    provider.delete(old_fragment_pod)
    fragment_lifecycle.transition(PodLifecycleState.TERMINATED)
    fragment_replacement = _lifecycle_from_manifest(fragment_manifest)
    lifecycles[fragment_manifest["logical_pod_id"]] = fragment_replacement
    fragment_replacement.transition(PodLifecycleState.CREATE_REQUESTED)
    allocation = provider.create(fragment_manifest)
    fragment_replacement.pod_id = str(allocation["pod_id"])
    fragment_replacement.machine_id = str(allocation["machine_id"])
    fragment_replacement.datacenter_id = str(allocation["datacenter_id"])
    fragment_replacement.transition(PodLifecycleState.CREATED)
    endpoint_coordinator.publish(
        FragmentIdentity(
            worker_id=worker_id,
            pod_id=str(allocation["pod_id"]),
            machine_id=str(allocation["machine_id"]),
            host=str(allocation["private_identity"]),
            port=int(fragment_manifest["internal_serving_ports"][0]),
        )
    )
    _advance_ready(fragment_replacement, events)
    if parent_lifecycle.parent_endpoint_generation != endpoint_coordinator.generation:
        old_parent_pod = str(parent_lifecycle.pod_id)
        parent_lifecycle.transition(PodLifecycleState.REPLACEMENT_REQUIRED)
        parent_lifecycle.transition(PodLifecycleState.TERMINATING)
        provider.delete(old_parent_pod)
        parent_lifecycle.transition(PodLifecycleState.TERMINATED)
        parent_lifecycle = _lifecycle_from_manifest(parent)
        lifecycles[parent["logical_pod_id"]] = parent_lifecycle
        parent_lifecycle.transition(PodLifecycleState.CREATE_REQUESTED)
        allocation = provider.create(parent)
        parent_lifecycle.pod_id = str(allocation["pod_id"])
        parent_lifecycle.machine_id = str(allocation["machine_id"])
        parent_lifecycle.datacenter_id = str(allocation["datacenter_id"])
        parent_lifecycle.parent_endpoint_generation = endpoint_coordinator.generation
        parent_lifecycle.transition(PodLifecycleState.CREATED)
        _advance_ready(parent_lifecycle, events)
        _event(
            events,
            "PARENT_REPLACED_FOR_ENDPOINT_GENERATION",
            old_parent_pod_id=old_parent_pod,
            new_parent_pod_id=parent_lifecycle.pod_id,
            generation=endpoint_coordinator.generation,
            unrelated_backbone_pods_recycled=0,
        )

    current_workers = [
        worker
        for lifecycle in lifecycles.values()
        for worker in lifecycle.workers.values()
        if worker.state is PodLifecycleState.READY_HEALTHY
    ]
    if len(current_workers) != 97:
        raise RuntimeError(f"simulated fleet has {len(current_workers)} healthy roles")
    for lifecycle in lifecycles.values():
        if not lifecycle.fully_ready:
            raise RuntimeError(f"simulated Pod is not fully ready: {lifecycle.logical_pod_id}")
        lifecycle.transition(PodLifecycleState.FROZEN_FOR_INFERENCE)
    _event(
        events,
        "FLEET_FROZEN",
        simultaneously_healthy_logical_roles=97,
        current_pod_count=len(provider.live),
        fragment_machine_ids_distinct=True,
        parent_endpoint_generation=endpoint_coordinator.generation,
    )
    for token_index, label in ((1, "TOKEN_1"), (2, "TOKEN_2")):
        _event(
            events,
            "TOKEN_EMITTED",
            token_index=token_index,
            token_label=label,
            simulated=True,
            model_execution_performed=False,
        )
    _event(
        events,
        "PUBLIC_GENERATION_SIMULATED",
        simulated=True,
        model_text_fabricated=False,
        purpose="CONTROL_FLOW_ONLY",
    )

    for lifecycle in lifecycles.values():
        lifecycle.transition(PodLifecycleState.TERMINATING)
        provider.delete(str(lifecycle.pod_id))
        lifecycle.transition(PodLifecycleState.TERMINATED)
    _event(events, "CLEANUP_CONFIRMED", zero_live_simulated_pods=not provider.live)
    result = {
        "schema_version": "experiment-025-runpod-provider-simulator-v1",
        "generated_at_utc": utc_now(),
        "run_id": RUN_ID,
        "status": "PASS",
        "evidence_class": "SIMULATED_PROVIDER_CONTROL_FLOW",
        "physical_execution": False,
        "model_execution": False,
        "provider_network_calls": 0,
        "provider_mutations": [],
        "simulated_cost_accrual_usd": provider.cost_usd,
        "simulated_cost_is_not_provider_invoice": True,
        "planned_pod_count": len(manifests),
        "created_simulated_pod_count_including_replacements": len(provider.created),
        "deleted_simulated_pod_count": len(provider.deleted),
        "zero_live_simulated_pods": not provider.live,
        "simultaneously_healthy_logical_roles": 97,
        "fragment_duplicate_machine_rejected": any(
            event["event_type"] == "FRAGMENT_DUPLICATE_MACHINE_REJECTED" for event in events
        ),
        "parent_generation_restart_exercised": any(
            event["event_type"] == "PARENT_REPLACED_FOR_ENDPOINT_GENERATION" for event in events
        ),
        "pod_group_churn_exercised": any(
            event["event_type"] == "WORKER_FAILURE_INVALIDATED_POD_GROUP" for event in events
        ),
        "events": events,
    }
    result["event_log_sha256"] = canonical_sha256(events)
    return result


__all__ = [
    "FakeRunPodProvider",
    "FragmentEndpointCoordinator",
    "FragmentIdentity",
    "PodLifecycle",
    "PodLifecycleState",
    "WorkerLifecycle",
    "simulate_full_control_flow",
]
