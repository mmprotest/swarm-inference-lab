from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from swarm_inference.experiments.experiment_025 import (
    provisioning as provisioning_module,
)
from swarm_inference.experiments.experiment_025.acquisition import (
    AcquisitionPolicy,
    Candidate,
    CostGuard,
    FleetAcquisitionController,
    FragmentEndpointCoordinator,
    HedgeLimiter,
    OfferSelector,
    ReadyRegistry,
)
from swarm_inference.experiments.experiment_025.provisioning import (
    LiveWorker,
    WorkerCompatibilityError,
)
from swarm_inference.experiments.experiment_025.vast_lifecycle import (
    AppendOnlyLifecycleLedger,
    Offer,
)

IMAGE_DIGEST = "sha256:" + "4" * 64


def _scoring_policy() -> dict[str, Any]:
    return {
        "formula_version": "test-v1",
        "provider_reliability_prior_weight": 2.0,
        "public_mapping_prior_probability": 0.95,
        "public_mapping_prior_weight": 2.0,
        "post_ready_health_prior_probability": 0.95,
        "post_ready_health_prior_weight": 2.0,
        "probability_floor": 0.05,
        "global_median_ready_seconds": 1.0,
        "history_time_prior_weight": 1.0,
        "fleet_delay_cost_usd_per_minute": 0.01,
        "repeated_attributable_failure_penalty_usd": 1.0,
        "maximum_historical_success_credit_usd": 0.5,
        "historical_success_credit_usd": 0.1,
        "hard_exclusion_penalty_usd": 1_000.0,
    }


def _policy(**overrides: Any) -> AcquisitionPolicy:
    values: dict[str, Any] = {
        "acquisition_window_seconds": 8.0,
        "inference_cleanup_reserve_seconds": 1.0,
        "stability_barrier_seconds": 0.05,
        "minimum_stability_health_rounds": 1,
        "provider_poll_seconds": 0.02,
        "ready_health_probe_seconds": 0.05,
        "provider_missing_poll_limit": 1,
        "health_failure_limit": 1,
        "public_mapping_timeout_seconds": 2.0,
        "dud_no_progress_timeout_seconds": 4.0,
        "progress_log_probe_after_seconds": 10.0,
        "progress_log_probe_interval_seconds": 10.0,
        "dynamic_alternate_refresh_enabled": True,
        "maximum_dynamic_refreshes_per_group": 2,
        "dynamic_refresh_retry_seconds": 0.02,
        "hedge_enabled": False,
        "hedge_warning_no_progress_seconds": 0.05,
        "hedge_minimum_elapsed_seconds": 0.05,
        "hedge_parent_minimum_elapsed_seconds": 0.05,
        "hedge_parent_ready_role_threshold": 1,
        "hedge_probability_threshold": 0.99,
        "maximum_concurrent_hedges": 1,
        "maximum_total_acquisition_cost_usd": 100.0,
        "hedge_projected_cost_reserve_usd": 0.01,
        "maximum_create_concurrency": 8,
        "health_probe_concurrency": 8,
        "offer_scoring": _scoring_policy(),
        "persistent_hard_exclusion_machine_ids": frozenset(),
    }
    values.update(overrides)
    return AcquisitionPolicy(**values)


def _offer(
    offer_id: int,
    *,
    gpu_name: str = "RTX 3090",
    dph: float = 0.25,
    machine_id: int | None = None,
    direct_ports: int = 1,
) -> Offer:
    return Offer(
        offer_id=offer_id,
        machine_id=machine_id if machine_id is not None else offer_id + 10_000,
        gpu_name=gpu_name,
        gpu_count=1,
        gpu_ram_gib=24.0 if "3090" in gpu_name else 12.0,
        reliability=0.995,
        verified=True,
        rentable=True,
        dph_total=dph,
        storage_cost_per_gb_month=0.01,
        inet_down_mbps=1_000.0,
        inet_up_mbps=500.0,
        inet_down_cost_per_gb=0.01,
        inet_up_cost_per_gb=0.0,
        disk_bw_mbps=1_000.0,
        disk_space_gb=500.0,
        direct_port_count=direct_ports,
        static_ip=True,
        driver_version="570.00",
        cuda_max_version=13.0,
        raw={"dph_base": dph, "geolocation": "test-region"},
    )


def _worker(
    worker_id: str,
    *,
    role: str,
    instance_id: int,
    machine_id: int,
    offer_id: int,
    worker_index: int | None = None,
) -> LiveWorker:
    ready = {
        "worker_id": worker_id,
        "role": role,
        "machine_id": str(machine_id),
        "instance_id": str(instance_id),
        "image_digest": IMAGE_DIGEST,
        "checkpoint_fingerprint": "c" * 64,
        "assignment_sha256": "a" * 64,
        "gpu": {"gpu_uuid": f"GPU-{instance_id}"},
    }
    return LiveWorker(
        worker_id=worker_id,
        role=role,
        layer=89 if "089" in worker_id else 0,
        worker_index=worker_index,
        gpu_slot=0,
        offer_id=offer_id,
        instance_id=instance_id,
        machine_id=machine_id,
        label=f"test-{instance_id}",
        host=f"host-{instance_id}",
        port=50_000 + instance_id,
        gpu_name="RTX 3060" if role == "SUB_LAYER_WORKER" else "RTX 3090",
        ready=ready,
    )


class _FakeClient:
    def __init__(self, refresh_offers: list[Offer] | None = None) -> None:
        self.lock = threading.Lock()
        self.next_instance_id = 1
        self.instances: dict[int, dict[str, Any]] = {}
        self.destroyed: list[tuple[int, str]] = []
        self.refresh_offers = list(refresh_offers or [])
        self.peak_live = 0

    def create(self, *, offer: Offer, candidate_id: str) -> int:
        with self.lock:
            instance_id = self.next_instance_id
            self.next_instance_id += 1
            self.instances[instance_id] = {
                "id": instance_id,
                "machine_id": offer.machine_id,
                "offer_id": offer.offer_id,
                "candidate_id": candidate_id,
                "actual_status": "running",
                "status_msg": "booting",
                "public_ipaddr": f"host-{instance_id}",
                "ports": {
                    "42525/tcp": [{"HostPort": str(50_000 + instance_id)}]
                },
            }
            self.peak_live = max(self.peak_live, len(self.instances))
            return instance_id

    def show_instances(self) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(row) for row in self.instances.values()]

    def destroy_instance(self, instance_id: int, *, reason: str) -> None:
        with self.lock:
            self.instances.pop(instance_id, None)
            self.destroyed.append((instance_id, reason))

    def logs(self, instance_id: int, *, tail: int) -> str:
        return f"instance {instance_id} still bootstrapping ({tail})"

    def search_offers(self, **_kwargs: Any) -> list[Offer]:
        return list(self.refresh_offers)


def _fleet_inputs(
    *,
    alternate_worker_ids: set[str] | None = None,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    str,
]:
    rows: dict[str, dict[str, Any]] = {}
    specs: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    identities = [
        *(f"e025-layer-089-sub-{index:02d}" for index in range(4)),
        "e025-stage-089-parent",
        "e025-stage-000",
    ]
    for index, worker_id in enumerate(identities):
        fragment = "sub-" in worker_id
        parent = worker_id.endswith("parent")
        role = (
            "SUB_LAYER_WORKER"
            if fragment
            else "SUB_LAYER_PARENT"
            if parent
            else "BACKBONE_STAGE"
        )
        group_id = f"group-{worker_id}"
        selected = _offer(
            100 + index,
            gpu_name="RTX 3060" if fragment else "RTX 3090",
        )
        alternates = []
        if worker_id in (alternate_worker_ids or set()):
            alternate_id = 900 + index
            alternates = [
                {
                    "offer": asdict(
                        _offer(
                            alternate_id,
                            gpu_name="RTX 3060" if fragment else "RTX 3090",
                        )
                    )
                }
            ]
        row = {
            "worker_id": worker_id,
            "role": role,
            "instance_group_id": group_id,
            "gpu_slot": 0,
            "container_port": 42525,
            "download_bytes_cold_cache": 1_000_000,
        }
        rows[worker_id] = row
        specs[worker_id] = {
            "role": role,
            "layer": 89 if (fragment or parent) else 0,
            "worker_index": index if fragment else None,
        }
        groups[group_id] = {
            "instance_group_id": group_id,
            "role": role,
            "disk_gb": 80,
            "workers": [row],
            "selected_offer": asdict(selected),
            "alternates": alternates,
        }
    return rows, groups, specs, "e025-stage-089-parent"


def _controller(
    tmp_path: Path,
    *,
    client: _FakeClient,
    policy: AcquisitionPolicy,
    wait_for_worker: Any,
    probe: Any,
    emit: Any,
    alternate_worker_ids: set[str] | None = None,
) -> FleetAcquisitionController:
    rows, groups, specs, parent_id = _fleet_inputs(
        alternate_worker_ids=alternate_worker_ids
    )
    credential = tmp_path / "credential.bin"
    certificate = tmp_path / "certificate.pem"
    credential.write_bytes(b"test-credential")
    certificate.write_text("test-certificate", encoding="utf-8")

    def create_group(**kwargs: Any) -> int:
        return client.create(
            offer=kwargs["offer"], candidate_id=str(kwargs["group_id"])
        )

    def public_endpoint(**kwargs: Any) -> tuple[str, int]:
        instance_id = int(kwargs["instance_id"])
        return f"host-{instance_id}", 50_000 + instance_id

    return FleetAcquisitionController(
        run_id="e025-test",
        client=client,  # type: ignore[arg-type]
        ledger=AppendOnlyLifecycleLedger(tmp_path / "ledger.jsonl", "e025-test"),
        rows=rows,
        groups=groups,
        specs=specs,
        parent_id=parent_id,
        image_reference="example.invalid/e025:test",
        image_digest=IMAGE_DIGEST,
        material={
            "credential_path": str(credential),
            "certificate_path": str(certificate),
        },
        watchdog_receipt=tmp_path / "watchdog.json",
        go_receipt=tmp_path / "go.json",
        stage_root=tmp_path,
        policy=policy,
        history_by_machine={},
        emit=emit,
        create_worker_group_instance_fn=create_group,
        wait_for_public_endpoint_fn=public_endpoint,
        wait_for_worker_fn=wait_for_worker,
        probe_worker_liveness_fn=probe,
    )


def test_parent_bootstrap_starts_from_mappings_before_fragments_ready(
    tmp_path: Path,
) -> None:
    client = _FakeClient()
    release_fragments = threading.Event()
    parent_started = threading.Event()
    fragment_waiters = 0
    waiter_lock = threading.Lock()

    def wait_for_worker(**kwargs: Any) -> LiveWorker:
        nonlocal fragment_waiters
        worker_id = str(kwargs["worker_id"])
        role = str(kwargs["role"])
        if role == "SUB_LAYER_WORKER":
            with waiter_lock:
                fragment_waiters += 1
            assert release_fragments.wait(timeout=4.0)
        elif role == "SUB_LAYER_PARENT":
            parent_started.set()
        offer = kwargs["offer"]
        return _worker(
            worker_id,
            role=role,
            instance_id=int(kwargs["instance_id"]),
            machine_id=offer.machine_id,
            offer_id=offer.offer_id,
            worker_index=kwargs["worker_index"],
        )

    controller = _controller(
        tmp_path,
        client=client,
        policy=_policy(),
        wait_for_worker=wait_for_worker,
        probe=lambda **_kwargs: {"status": "READY_HEALTHY"},
        emit=lambda *_args, **_kwargs: None,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(controller.acquire)
        assert parent_started.wait(timeout=3.0)
        with waiter_lock:
            assert fragment_waiters == 4
        assert release_fragments.is_set() is False
        release_fragments.set()
        workers, receipt = future.result(timeout=5.0)
    assert len(workers) == 6
    assert receipt["parent_generation_matches_fragments"] is True
    parent_creation = next(
        row
        for row in controller.created_candidates
        if row["instance_group_id"].endswith("stage-089-parent")
    )
    assert parent_creation["fragment_endpoint_generation"] == 1


def test_fragment_replacement_advances_generation_and_stales_parent() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    stale: list[str] = []
    coordinator = FragmentEndpointCoordinator(
        [f"fragment-{index}" for index in range(4)],
        lambda event_type, **fields: events.append((event_type, fields)),
    )
    coordinator.set_parent_invalidator(stale.append)
    for index in range(4):
        coordinator.publish(
            worker_id=f"fragment-{index}",
            worker_index=index,
            instance_id=10 + index,
            machine_id=100 + index,
            host=f"old-{index}",
            port=42_525 + index,
        )
    generation_1, endpoints_1 = coordinator.complete_snapshot() or (None, None)
    assert generation_1 == 1
    assert endpoints_1 is not None
    assert coordinator.retire(
        worker_id="fragment-2", instance_id=12, reason="provider disappeared"
    )
    assert stale and coordinator.matches(generation_1) is False
    coordinator.publish(
        worker_id="fragment-2",
        worker_index=2,
        instance_id=99,
        machine_id=999,
        host="replacement",
        port=49_999,
    )
    generation_2, endpoints_2 = coordinator.complete_snapshot() or (None, None)
    assert generation_2 == 2
    assert endpoints_2 is not None and len(endpoints_2) == 4
    assert {row["host"] for row in endpoints_2} == {
        "old-0",
        "old-1",
        "replacement",
        "old-3",
    }
    assert any(event == "FRAGMENT_ENDPOINT_GENERATION_CHANGED" for event, _ in events)


def test_fragment_ready_loss_restarts_parent_on_current_generation(
    tmp_path: Path,
) -> None:
    client = _FakeClient()
    events: list[tuple[str, dict[str, Any]]] = []
    initial_parent_ready = threading.Event()

    def wait_for_worker(**kwargs: Any) -> LiveWorker:
        offer = kwargs["offer"]
        worker = _worker(
            str(kwargs["worker_id"]),
            role=str(kwargs["role"]),
            instance_id=int(kwargs["instance_id"]),
            machine_id=offer.machine_id,
            offer_id=offer.offer_id,
            worker_index=kwargs["worker_index"],
        )
        if worker.role == "SUB_LAYER_PARENT" and worker.offer_id == 104:
            initial_parent_ready.set()
        return worker

    def probe(**kwargs: Any) -> dict[str, Any]:
        worker = kwargs["worker"]
        if worker.worker_id == "e025-layer-089-sub-00" and worker.offer_id == 100:
            assert initial_parent_ready.wait(timeout=2.0)
            raise RuntimeError("fragment disappeared after READY")
        return {"status": "READY_HEALTHY"}

    controller = _controller(
        tmp_path,
        client=client,
        policy=_policy(),
        wait_for_worker=wait_for_worker,
        probe=probe,
        emit=lambda event_type, **fields: events.append((event_type, fields)),
        alternate_worker_ids={
            "e025-layer-089-sub-00",
            "e025-stage-089-parent",
        },
    )
    workers, receipt = controller.acquire()
    fragment = next(
        worker for worker in workers if worker.worker_id == "e025-layer-089-sub-00"
    )
    parent = next(
        worker for worker in workers if worker.worker_id == "e025-stage-089-parent"
    )
    assert fragment.offer_id == 900
    assert parent.offer_id == 904
    assert receipt["fragment_endpoints"]["fragment_endpoint_generation"] >= 2
    assert receipt["parent_generation_matches_fragments"] is True
    assert len(receipt["fragment_endpoints"]["endpoints"]) == 4
    fragment_endpoint = next(
        endpoint
        for endpoint in receipt["fragment_endpoints"]["endpoints"]
        if endpoint["worker_id"] == "e025-layer-089-sub-00"
    )
    assert fragment_endpoint["machine_id"] == 10_900
    assert any(event == "PARENT_GENERATION_STALE" for event, _fields in events)
    parent_creations = [
        row
        for row in controller.created_candidates
        if row["instance_group_id"].endswith("stage-089-parent")
    ]
    assert len(parent_creations) == 2


def test_ready_worker_loss_removes_count_and_replaces_only_affected_group(
    tmp_path: Path,
) -> None:
    client = _FakeClient()
    events: list[tuple[str, dict[str, Any]]] = []
    holder: dict[str, FleetAcquisitionController] = {}

    def wait_for_worker(**kwargs: Any) -> LiveWorker:
        offer = kwargs["offer"]
        return _worker(
            str(kwargs["worker_id"]),
            role=str(kwargs["role"]),
            instance_id=int(kwargs["instance_id"]),
            machine_id=offer.machine_id,
            offer_id=offer.offer_id,
            worker_index=kwargs["worker_index"],
        )

    failed_once = False

    def probe(**kwargs: Any) -> dict[str, Any]:
        nonlocal failed_once
        worker = kwargs["worker"]
        if worker.worker_id == "e025-stage-000" and worker.offer_id != 905:
            failed_once = True
            raise RuntimeError("worker became unreachable after READY")
        return {"status": "READY_HEALTHY"}

    def emit(event_type: str, **fields: Any) -> None:
        if event_type == "WORKER_READY_LOST":
            fields["observed_current_ready_count"] = holder[
                "controller"
            ].registry.current_ready_count()
        events.append((event_type, fields))

    controller = _controller(
        tmp_path,
        client=client,
        policy=_policy(),
        wait_for_worker=wait_for_worker,
        probe=probe,
        emit=emit,
        alternate_worker_ids={"e025-stage-000"},
    )
    holder["controller"] = controller
    workers, _receipt = controller.acquire()
    assert failed_once is True
    target = next(worker for worker in workers if worker.worker_id == "e025-stage-000")
    assert target.offer_id == 905
    target_creations = [
        row
        for row in controller.created_candidates
        if row["instance_group_id"].endswith("stage-000")
    ]
    assert len(target_creations) == 2
    assert all(
        len(
            [
                row
                for row in controller.created_candidates
                if row["instance_group_id"] == group_id
            ]
        )
        == 1
        for group_id in controller.groups
        if not group_id.endswith("stage-000")
    )
    lost = [fields for event, fields in events if event == "WORKER_READY_LOST"]
    assert len(lost) == 1
    assert lost[0]["observed_current_ready_count"] == 5


def test_dynamic_refresh_is_compatible_exclusive_and_auditable(tmp_path: Path) -> None:
    initial = _offer(1)
    active = _offer(2)
    failed_machine = _offer(3)
    invalid_hardware = _offer(4, gpu_name="RTX A6000")
    selected = _offer(5)
    client = _FakeClient([active, failed_machine, invalid_hardware, selected])
    group_id = "group-e025-stage-000"
    worker_row = {
        "worker_id": "e025-stage-000",
        "role": "BACKBONE_STAGE",
        "download_bytes_cold_cache": 1_000_000,
    }
    selector = OfferSelector(
        client=client,  # type: ignore[arg-type]
        policy=_policy(),
        groups={
            group_id: {
                "disk_gb": 80,
                "workers": [worker_row],
                "selected_offer": asdict(initial),
                "alternates": [],
            }
        },
        history_by_machine={},
        audit_path=tmp_path / "refresh.jsonl",
        emit=lambda *_args, **_kwargs: None,
    )
    selector.initial[group_id].clear()
    selector.active_offer_ids.add(active.offer_id)
    selector.active_machine_ids.add(active.machine_id)
    selector.failed_machine_ids.add(failed_machine.machine_id)
    offer, _score, source = selector.reserve(group_id)
    assert source == "DYNAMIC_LIVE_REFRESH"
    assert offer.offer_id == selected.offer_id
    audit = (tmp_path / "refresh.jsonl").read_text(encoding="utf-8")
    assert '"selected_offer_id":5' in audit
    assert f'"offer_id":{invalid_hardware.offer_id}' not in audit
    assert str(active.machine_id) in audit and str(failed_machine.machine_id) in audit


def test_slow_primary_hedges_first_healthy_wins_and_loser_is_destroyed(
    tmp_path: Path,
) -> None:
    client = _FakeClient()
    events: list[tuple[str, dict[str, Any]]] = []

    def wait_for_worker(**kwargs: Any) -> LiveWorker:
        worker_id = str(kwargs["worker_id"])
        offer = kwargs["offer"]
        if worker_id == "e025-stage-000" and offer.offer_id != 905:
            abort_event = kwargs["abort_event"]
            if not abort_event.wait(timeout=4.0):
                raise TimeoutError("test primary unexpectedly survived")
            raise RuntimeError("slow primary cancelled after hedge won")
        return _worker(
            worker_id,
            role=str(kwargs["role"]),
            instance_id=int(kwargs["instance_id"]),
            machine_id=offer.machine_id,
            offer_id=offer.offer_id,
            worker_index=kwargs["worker_index"],
        )

    controller = _controller(
        tmp_path,
        client=client,
        policy=_policy(
            hedge_enabled=True,
            hedge_warning_no_progress_seconds=0.01,
            hedge_minimum_elapsed_seconds=0.01,
        ),
        wait_for_worker=wait_for_worker,
        probe=lambda **_kwargs: {"status": "READY_HEALTHY"},
        emit=lambda event_type, **fields: events.append((event_type, fields)),
        alternate_worker_ids={"e025-stage-000"},
    )
    workers, receipt = controller.acquire()
    target = next(worker for worker in workers if worker.worker_id == "e025-stage-000")
    assert target.offer_id == 905
    assert receipt["hedging"]["peak_concurrent_hedges"] == 1
    assert any(event == "HEDGE_STARTED" for event, _fields in events)
    assert any(event == "HEDGE_WON" for event, _fields in events)
    assert any(event == "HEDGE_LOSER_DESTROYED" for event, _fields in events)
    assert any("hedge-loser" in reason for _instance, reason in client.destroyed)
    assert len({worker.worker_id for worker in workers}) == len(workers) == 6
    assert client.peak_live <= len(controller.groups) + 1


def test_hedge_limiter_and_cost_guard_enforce_caps() -> None:
    limiter = HedgeLimiter(1)
    assert limiter.acquire() is True
    assert limiter.acquire() is False
    limiter.release()
    assert limiter.acquire() is True

    guard = CostGuard(maximum_usd=0.001, hedge_reserve_usd=0.0005)
    expensive = _offer(50, dph=10.0)
    allowed, _projection = guard.authorize(
        candidate_id="hedge",
        offer=expensive,
        disk_gb=100,
        download_bytes=1_000_000_000,
        projected_seconds=600,
        hedge=True,
    )
    assert allowed is False
    assert guard.receipt()["authorization_blockers"]


def test_reliability_score_prefers_proven_ready_machine_over_cheaper_dud() -> None:
    policy = _scoring_policy()
    proven = _offer(60, dph=0.30)
    dud = _offer(61, dph=0.05)
    proven_score = proven.acquisition_score(
        required_download_bytes=18_000_000_000,
        disk_gb=80,
        history={
            "attempt_count": 3,
            "ready_success_count": 3,
            "public_mapping_observation_count": 3,
            "public_mapping_success_count": 3,
            "post_ready_health_observation_count": 3,
            "ready_healthy_count": 3,
            "attributable_failure_count": 0,
            "median_time_to_ready_seconds": 600,
            "median_download_throughput_mbps": 500,
        },
        scoring_policy=policy,
    )
    dud_score = dud.acquisition_score(
        required_download_bytes=18_000_000_000,
        disk_gb=80,
        history={
            "attempt_count": 3,
            "ready_success_count": 0,
            "public_mapping_observation_count": 3,
            "public_mapping_success_count": 0,
            "post_ready_health_observation_count": 0,
            "ready_healthy_count": 0,
            "attributable_failure_count": 3,
            "hard_excluded": True,
        },
        scoring_policy=policy,
    )
    assert proven.gpu_rental_rate_per_hour > dud.gpu_rental_rate_per_hour
    assert (
        proven_score["short_run_acquisition_score"]
        < dud_score["short_run_acquisition_score"]
    )


def test_ready_liveness_probe_is_authenticated_and_identity_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker = _worker(
        "e025-stage-000",
        role="BACKBONE_STAGE",
        instance_id=5,
        machine_id=10_005,
        offer_id=5,
    )
    provider_row = {
        "id": 5,
        "machine_id": 10_005,
        "actual_status": "running",
        "public_ipaddr": "host-5",
        "ports": {"42525/tcp": [{"HostPort": "50005"}]},
    }
    monkeypatch.setattr(
        provisioning_module,
        "_probe_register",
        lambda **_kwargs: dict(worker.ready),
    )
    result = provisioning_module.probe_worker_liveness(
        client=SimpleNamespace(show_instances=lambda: [provider_row]),
        worker=worker,
        credential=b"credential",
        certificate=tmp_path / "certificate.pem",
        image_digest=IMAGE_DIGEST,
        provider_row=provider_row,
    )
    assert result["status"] == "READY_HEALTHY"

    changed = dict(worker.ready)
    changed["image_digest"] = "sha256:" + "f" * 64
    monkeypatch.setattr(
        provisioning_module,
        "_probe_register",
        lambda **_kwargs: changed,
    )
    with pytest.raises(WorkerCompatibilityError, match="identity changed"):
        provisioning_module.probe_worker_liveness(
            client=SimpleNamespace(show_instances=lambda: [provider_row]),
            worker=worker,
            credential=b"credential",
            certificate=tmp_path / "certificate.pem",
            image_digest=IMAGE_DIGEST,
            provider_row=provider_row,
        )


def test_hard_dud_is_destroyed_then_replaced(tmp_path: Path) -> None:
    client = _FakeClient()

    def wait_for_worker(**kwargs: Any) -> LiveWorker:
        worker_id = str(kwargs["worker_id"])
        offer = kwargs["offer"]
        if worker_id == "e025-stage-000" and offer.offer_id != 905:
            if not kwargs["abort_event"].wait(timeout=4.0):
                raise TimeoutError("dud was not cancelled")
            raise RuntimeError("hard dud cancelled")
        return _worker(
            worker_id,
            role=str(kwargs["role"]),
            instance_id=int(kwargs["instance_id"]),
            machine_id=offer.machine_id,
            offer_id=offer.offer_id,
            worker_index=kwargs["worker_index"],
        )

    controller = _controller(
        tmp_path,
        client=client,
        policy=_policy(dud_no_progress_timeout_seconds=0.05),
        wait_for_worker=wait_for_worker,
        probe=lambda **_kwargs: {"status": "READY_HEALTHY"},
        emit=lambda *_args, **_kwargs: None,
        alternate_worker_ids={"e025-stage-000"},
    )
    workers, receipt = controller.acquire()
    target = next(worker for worker in workers if worker.worker_id == "e025-stage-000")
    assert target.offer_id == 905
    assert any("hard no-progress timeout" in reason for _id, reason in client.destroyed)
    assert 105 in receipt["offer_selection"]["failed_offer_ids"]


def test_abort_cleans_every_created_candidate(tmp_path: Path) -> None:
    client = _FakeClient()

    def never_ready(**kwargs: Any) -> LiveWorker:
        kwargs["abort_event"].wait(timeout=3.0)
        raise RuntimeError("acquisition aborted")

    controller = _controller(
        tmp_path,
        client=client,
        policy=_policy(acquisition_window_seconds=0.4),
        wait_for_worker=never_ready,
        probe=lambda **_kwargs: {"status": "READY_HEALTHY"},
        emit=lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="did not reach"):
        controller.acquire()
    assert client.instances == {}
    assert len(client.destroyed) == len(controller.created_candidates)


def test_simultaneous_ready_and_stability_barrier_reject_historical_readiness() -> None:
    expected = {f"worker-{index:03d}" for index in range(97)}
    parent_group = "group-parent"
    controller = object.__new__(FleetAcquisitionController)
    controller.rows = {worker_id: {} for worker_id in expected}
    controller.parent_group_id = parent_group
    controller.registry = ReadyRegistry(expected, parent_group)
    controller.fragment_endpoints = FragmentEndpointCoordinator(
        [f"fragment-{index}" for index in range(4)],
        lambda *_args, **_kwargs: None,
    )
    for index in range(4):
        controller.fragment_endpoints.publish(
            worker_id=f"fragment-{index}",
            worker_index=index,
            instance_id=1_000 + index,
            machine_id=2_000 + index,
            host=f"fragment-{index}",
            port=43_000 + index,
        )
    for index, worker_id in enumerate(sorted(expected)[:96]):
        group_id = parent_group if index == 0 else f"group-{index}"
        controller.registry.publish(
            group_id,
            (
                _worker(
                    worker_id,
                    role="SUB_LAYER_PARENT" if index == 0 else "BACKBONE_STAGE",
                    instance_id=10_000 + index,
                    machine_id=20_000 + index,
                    offer_id=30_000 + index,
                ),
            ),
            parent_generation=1 if index == 0 else None,
        )
    assert controller._simultaneously_ready() is False
    missing = sorted(expected)[96]
    controller.registry.publish(
        "group-final",
        (
            _worker(
                missing,
                role="BACKBONE_STAGE",
                instance_id=99_999,
                machine_id=88_888,
                offer_id=77_777,
            ),
        ),
        parent_generation=None,
    )
    assert controller._simultaneously_ready() is True

    clock = [100.0]
    controller.time_fn = lambda: clock[0]
    controller.policy = SimpleNamespace(
        stability_barrier_seconds=2.0, minimum_stability_health_rounds=2
    )
    controller.emit = lambda *_args, **_kwargs: None
    controller.barrier_started_epoch = None
    controller.healthy_rounds_during_barrier = 0
    controller.barrier_resets = 0
    controller.peak_ready_count = 0
    assert controller._barrier_update(True) is False
    controller.registry.remove("group-final")
    clock[0] += 1.0
    assert controller._barrier_update(True) is False
    assert controller.barrier_resets == 1
    controller.registry.publish(
        "group-final",
        (
            _worker(
                missing,
                role="BACKBONE_STAGE",
                instance_id=99_998,
                machine_id=88_887,
                offer_id=77_776,
            ),
        ),
        parent_generation=None,
    )
    assert controller._barrier_update(True) is False
    clock[0] += 2.1
    assert controller._barrier_update(True) is True
    controller.registry.parent_generation = 0
    assert controller._simultaneously_ready() is False


def test_cancelled_create_is_destroyed_after_provider_id_becomes_known(
    tmp_path: Path,
) -> None:
    client = _FakeClient()
    controller = SimpleNamespace(
        client=client,
        progress=SimpleNamespace(unregister=lambda _instance_id: None),
        cost_guard=CostGuard(10.0, 0.0),
        fragment_endpoints=SimpleNamespace(retire=lambda **_kwargs: None),
    )
    supervisor = object.__new__(
        __import__(
            "swarm_inference.experiments.experiment_025.acquisition",
            fromlist=["GroupSupervisor"],
        ).GroupSupervisor
    )
    supervisor.controller = controller
    supervisor.fragment_row = None
    candidate = Candidate(
        group_id="group",
        sequence=1,
        offer=_offer(70),
        score={},
        endpoint_generation=None,
        expert_endpoints=None,
        hedge=True,
        started_epoch=time.time(),
    )
    supervisor._destroy_candidate(candidate, "cancel-before-create-completes")
    assert candidate.destroyed is False
    instance_id = client.create(offer=candidate.offer, candidate_id="late-create")
    candidate.instance_id = instance_id
    candidate.create_finished_event.set()
    supervisor._destroy_candidate(candidate, "late-created-hedge-loser")
    assert candidate.destroyed is True
    assert client.instances == {}
