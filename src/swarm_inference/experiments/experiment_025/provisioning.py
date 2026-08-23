"""Bounded E025 worker provisioning and authenticated readiness probes."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_020.transport import Frame, MessageType

from .io import atomic_write_json, utc_now
from .secrets import vast_environment_options, vast_environment_options_for_workers
from .vast_lifecycle import AppendOnlyLifecycleLedger, Offer, VastClient
from .wire import Action, AuthenticatedConnection, pack_payload, unpack_payload


class WorkerCompatibilityError(RuntimeError):
    """A responding worker violates the frozen image/runtime contract."""


@dataclass(frozen=True, slots=True)
class LiveWorker:
    worker_id: str
    role: str
    layer: int
    worker_index: int | None
    gpu_slot: int
    offer_id: int
    instance_id: int
    machine_id: int
    label: str
    host: str
    port: int
    gpu_name: str
    ready: dict[str, Any]

    def endpoint(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "role": self.role,
            "host": self.host,
            "port": self.port,
            "layer": self.layer,
            "worker_index": self.worker_index,
            "gpu_slot": self.gpu_slot,
            "machine_id": str(self.machine_id),
            "instance_id": str(self.instance_id),
        }

    def expert_endpoint(self) -> dict[str, Any]:
        if self.worker_index is None:
            raise ValueError("only a physical sub-layer worker has an expert endpoint")
        return {
            "worker_id": self.worker_id,
            "worker_index": self.worker_index,
            "host": self.host,
            "port": self.port,
            "timeout_seconds": 180.0,
        }


def _instance_id(row: dict[str, Any]) -> int:
    return int(row.get("id", row.get("instance_id", -1)))


def _instance_machine_id(row: dict[str, Any]) -> int:
    return int(row.get("machine_id", row.get("machine", -1)))


def _public_endpoint(row: dict[str, Any], container_port: int = 42525) -> tuple[str, int]:
    host = str(
        row.get("public_ipaddr")
        or row.get("public_ip")
        or row.get("ssh_host")
        or ""
    )
    ports = row.get("ports", {})
    candidates: list[Any] = []
    if isinstance(ports, dict):
        for key in (f"{container_port}/tcp", str(container_port), container_port):
            if key in ports:
                value = ports[key]
                candidates.extend(value if isinstance(value, list) else [value])
    for candidate in candidates:
        if isinstance(candidate, dict):
            host = str(candidate.get("HostIp") or candidate.get("host_ip") or host)
            port = int(candidate.get("HostPort") or candidate.get("host_port") or -1)
            if host in {"0.0.0.0", "::", ""}:
                host = str(row.get("public_ipaddr") or row.get("public_ip") or "")
            if host and port > 0:
                return host, port
        elif isinstance(candidate, (str, int)) and host:
            return host, int(candidate)
    direct_start = int(row.get("direct_port_start", -1) or -1)
    direct_end = int(row.get("direct_port_end", -1) or -1)
    if host and direct_start <= container_port <= direct_end:
        return host, container_port
    raise ValueError("Vast instance has no public mapping for E025 port 42525")


def _probe_register(
    *,
    host: str,
    port: int,
    worker_id: str,
    credential: bytes,
    certificate: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    with AuthenticatedConnection(
        host,
        port,
        credential,
        certificate,
        timeout_seconds=timeout_seconds,
    ) as connection:
        frame = Frame(
            MessageType.EXECUTE_SHARD,
            f"e025-ready-{worker_id}",
            0,
            worker_id,
            "e025-readiness",
            pack_payload(Action.REGISTER, {"readiness_probe": True}),
        )
        response = connection.request(frame)
        action, metadata, _ = unpack_payload(response.payload)
        if response.message_type is not MessageType.SHARD_RESULT or action is not Action.REGISTER:
            raise WorkerCompatibilityError(
                "E025 readiness probe received a non-registration response"
            )
        ready = metadata.get("ready")
        if not isinstance(ready, dict) or ready.get("status") != "READY":
            raise WorkerCompatibilityError("E025 readiness probe received no READY receipt")
        return ready


def wait_for_public_endpoint(
    *,
    client: VastClient,
    ledger: AppendOnlyLifecycleLedger,
    worker_id: str,
    offer: Offer,
    instance_id: int,
    deadline_epoch: float,
    container_port: int = 42525,
    poll_seconds: float = 4.0,
    abort_event: threading.Event | None = None,
    initial_presence_timeout_seconds: float = 60.0,
    missing_after_seen_polls: int = 3,
) -> tuple[str, int]:
    """Freeze a created worker's public endpoint before it becomes READY.

    Stage 2 uses this narrow pre-readiness receipt so the parent can bootstrap in
    parallel with the fragment workers.  Endpoint publication is not worker
    readiness and never satisfies a physical-compute gate.
    """

    if initial_presence_timeout_seconds < 0:
        raise ValueError("initial presence timeout must be non-negative")
    if missing_after_seen_polls < 1:
        raise ValueError("missing-after-seen poll limit must be positive")
    seen_live_instance = False
    consecutive_missing_polls = 0
    endpoint_wait_started = time.monotonic()
    last_error = "instance has not appeared"
    while time.time() < deadline_epoch:
        if abort_event is not None and abort_event.is_set():
            raise RuntimeError(f"E025 worker {worker_id} endpoint wait was aborted")
        rows = client.show_instances()
        row = next((value for value in rows if _instance_id(value) == instance_id), None)
        if row is None:
            last_error = "instance absent from live query"
            consecutive_missing_polls += 1
            if seen_live_instance and consecutive_missing_polls >= missing_after_seen_polls:
                raise RuntimeError(
                    f"Vast instance {instance_id} disappeared after appearing in "
                    f"{consecutive_missing_polls} consecutive live queries"
                )
            if (
                not seen_live_instance
                and time.monotonic() - endpoint_wait_started
                >= initial_presence_timeout_seconds
            ):
                raise TimeoutError(
                    f"Vast instance {instance_id} never appeared within the "
                    "initial presence timeout"
                )
        else:
            seen_live_instance = True
            consecutive_missing_polls = 0
            actual_machine = _instance_machine_id(row)
            if actual_machine != offer.machine_id:
                raise RuntimeError("Vast instance machine ID differs from selected offer")
            status = str(row.get("actual_status", row.get("status", ""))).lower()
            if status in {"exited", "offline", "error"}:
                raise RuntimeError(f"Vast worker entered terminal status {status}")
            try:
                host, port = _public_endpoint(row, container_port)
            except ValueError as exc:
                last_error = str(exc)
            else:
                ledger.append(
                    "ENDPOINT_PUBLISHED",
                    command_category="show instances",
                    offer_id=offer.offer_id,
                    instance_id=instance_id,
                    machine_id=actual_machine,
                    worker_id=worker_id,
                    serving_port=port,
                    final_status="ENDPOINT_ONLY_NOT_READY",
                )
                return host, port
        if abort_event is not None:
            if abort_event.wait(poll_seconds):
                raise RuntimeError(f"E025 worker {worker_id} endpoint wait was aborted")
        else:
            time.sleep(poll_seconds)
    raise TimeoutError(f"E025 worker {worker_id} published no endpoint: {last_error}")


def wait_for_worker(
    *,
    client: VastClient,
    ledger: AppendOnlyLifecycleLedger,
    run_id: str,
    worker_id: str,
    role: str,
    layer: int,
    worker_index: int | None,
    offer: Offer,
    instance_id: int,
    credential: bytes,
    certificate: Path,
    image_digest: str,
    deadline_epoch: float,
    container_port: int = 42525,
    gpu_slot: int = 0,
    poll_seconds: float = 4.0,
    abort_event: threading.Event | None = None,
    initial_presence_timeout_seconds: float = 60.0,
    missing_after_seen_polls: int = 3,
) -> LiveWorker:
    if initial_presence_timeout_seconds < 0:
        raise ValueError("initial presence timeout must be non-negative")
    if missing_after_seen_polls < 1:
        raise ValueError("missing-after-seen poll limit must be positive")
    label = f"e025-{run_id}-{role.lower()}-{worker_index if worker_index is not None else layer:03d}"
    running_recorded = False
    seen_live_instance = False
    consecutive_missing_polls = 0
    readiness_started = time.monotonic()
    last_error = "instance has not appeared"
    while time.time() < deadline_epoch:
        if abort_event is not None and abort_event.is_set():
            raise RuntimeError(f"E025 worker {worker_id} readiness was aborted")
        rows = client.show_instances()
        row = next((value for value in rows if _instance_id(value) == instance_id), None)
        if row is None:
            last_error = "instance absent from live query"
            consecutive_missing_polls += 1
            if seen_live_instance and consecutive_missing_polls >= missing_after_seen_polls:
                raise RuntimeError(
                    f"Vast instance {instance_id} disappeared after appearing in "
                    f"{consecutive_missing_polls} consecutive live queries"
                )
            if (
                not seen_live_instance
                and time.monotonic() - readiness_started
                >= initial_presence_timeout_seconds
            ):
                raise TimeoutError(
                    f"Vast instance {instance_id} never appeared within the "
                    "initial presence timeout"
                )
            time.sleep(poll_seconds)
            continue
        seen_live_instance = True
        consecutive_missing_polls = 0
        actual_machine = _instance_machine_id(row)
        if actual_machine != offer.machine_id:
            raise RuntimeError("Vast instance machine ID differs from selected offer")
        status = str(row.get("actual_status", row.get("status", ""))).lower()
        if status in {"exited", "offline", "error"}:
            raise RuntimeError(f"Vast worker entered terminal status {status}")
        if status in {"running", "loading"} and not running_recorded:
            ledger.append(
                "INSTANCE_RUNNING",
                command_category="show instances",
                offer_id=offer.offer_id,
                instance_id=instance_id,
                machine_id=actual_machine,
                instance_label=str(row.get("label", label)),
                running_time=utc_now(),
                final_status=status.upper(),
            )
            running_recorded = True
        try:
            host, port = _public_endpoint(row, container_port)
            ready = _probe_register(
                host=host,
                port=port,
                worker_id=worker_id,
                credential=credential,
                certificate=certificate,
                timeout_seconds=min(15.0, max(2.0, deadline_epoch - time.time())),
            )
            if str(ready.get("image_digest")) != image_digest:
                raise WorkerCompatibilityError(
                    "physical worker image digest differs from the frozen image"
                )
            gpu = ready.get("gpu", {})
            if not bool(gpu.get("torch_cuda_available")):
                raise WorkerCompatibilityError("physical worker READY has no CUDA device")
            ledger.append(
                "WORKER_READY",
                command_category="authenticated readiness",
                offer_id=offer.offer_id,
                instance_id=instance_id,
                machine_id=actual_machine,
                gpu_model=gpu.get("gpu_name"),
                advertised_vram_gib=offer.gpu_ram_gib,
                measured_vram_mib=gpu.get("vram_mib"),
                physical_gpu_slot=gpu_slot,
                worker_id=worker_id,
                serving_port=port,
                instance_label=str(row.get("label", label)),
                worker_ready_time=utc_now(),
                final_status="READY",
            )
            return LiveWorker(
                worker_id=worker_id,
                role=role,
                layer=layer,
                worker_index=worker_index,
                gpu_slot=gpu_slot,
                offer_id=offer.offer_id,
                instance_id=instance_id,
                machine_id=actual_machine,
                label=str(row.get("label", label)),
                host=host,
                port=port,
                gpu_name=str(gpu.get("gpu_name")),
                ready=ready,
            )
        except WorkerCompatibilityError:
            raise
        except (ConnectionError, OSError, TimeoutError, ValueError, RuntimeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if abort_event is not None:
            if abort_event.wait(poll_seconds):
                raise RuntimeError(f"E025 worker {worker_id} readiness was aborted")
        else:
            time.sleep(poll_seconds)
    raise TimeoutError(f"E025 worker {worker_id} was not READY: {last_error}")


def provision_worker(
    *,
    client: VastClient,
    ledger: AppendOnlyLifecycleLedger,
    run_id: str,
    worker_id: str,
    role: str,
    layer: int,
    worker_index: int | None,
    offer: Offer,
    image_reference: str,
    image_digest: str,
    disk_gb: int,
    material: dict[str, Any],
    watchdog_receipt: Path,
    go_receipt: Path,
    deadline_epoch: float,
    maximum_context: int,
    expert_endpoints: list[dict[str, Any]] | None = None,
) -> LiveWorker:
    instance_id = create_worker_instance(
        client=client,
        run_id=run_id,
        worker_id=worker_id,
        role=role,
        layer=layer,
        worker_index=worker_index,
        offer=offer,
        image_reference=image_reference,
        image_digest=image_digest,
        disk_gb=disk_gb,
        material=material,
        watchdog_receipt=watchdog_receipt,
        go_receipt=go_receipt,
        maximum_context=maximum_context,
        expert_endpoints=expert_endpoints,
    )
    return wait_for_worker(
        client=client,
        ledger=ledger,
        run_id=run_id,
        worker_id=worker_id,
        role=role,
        layer=layer,
        worker_index=worker_index,
        offer=offer,
        instance_id=instance_id,
        credential=Path(material["credential_path"]).read_bytes(),
        certificate=Path(material["certificate_path"]),
        image_digest=image_digest,
        deadline_epoch=deadline_epoch,
        container_port=42525,
        gpu_slot=0,
    )


def create_worker_instance(
    *,
    client: VastClient,
    run_id: str,
    worker_id: str,
    role: str,
    layer: int,
    worker_index: int | None,
    offer: Offer,
    image_reference: str,
    image_digest: str,
    disk_gb: int,
    material: dict[str, Any],
    watchdog_receipt: Path,
    go_receipt: Path,
    maximum_context: int,
    expert_endpoints: list[dict[str, Any]] | None = None,
) -> int:
    environment = vast_environment_options(
        material=material,
        worker_id=worker_id,
        run_id=run_id,
        image_digest=image_digest,
        instance_id=None,
        machine_id=offer.machine_id,
        maximum_context=maximum_context,
        expert_endpoints=expert_endpoints,
    )
    return client.create_instance(
        run_id=run_id,
        role=role,
        index=worker_index if worker_index is not None else layer,
        offer=offer,
        image=image_reference,
        disk_gb=disk_gb,
        env_options=environment,
        watchdog_receipt=watchdog_receipt,
        go_receipt=go_receipt,
    )


def create_worker_group_instance(
    *,
    client: VastClient,
    run_id: str,
    group_id: str,
    group_index: int,
    offer: Offer,
    image_reference: str,
    image_digest: str,
    disk_gb: int,
    material: dict[str, Any],
    watchdog_receipt: Path,
    go_receipt: Path,
    worker_specs: list[dict[str, Any]],
) -> int:
    if len(worker_specs) > offer.gpu_count:
        raise ValueError("E025 worker group exceeds its selected offer GPU count")
    environment = vast_environment_options_for_workers(
        material=material,
        worker_specs=worker_specs,
        run_id=run_id,
        image_digest=image_digest,
        instance_id=None,
        machine_id=offer.machine_id,
    )
    return client.create_instance(
        run_id=run_id,
        role=group_id,
        index=group_index,
        offer=offer,
        image=image_reference,
        disk_gb=disk_gb,
        env_options=environment,
        watchdog_receipt=watchdog_receipt,
        go_receipt=go_receipt,
    )


def write_live_endpoints(path: Path, workers: list[LiveWorker]) -> dict[str, Any]:
    payload = {
        "schema_version": "experiment-025-live-endpoints-v1",
        "generated_at_utc": utc_now(),
        "workers": [worker.endpoint() for worker in workers],
    }
    atomic_write_json(path, payload)
    return payload


__all__ = [
    "LiveWorker",
    "WorkerCompatibilityError",
    "create_worker_group_instance",
    "create_worker_instance",
    "provision_worker",
    "wait_for_worker",
    "write_live_endpoints",
]
