"""Run the frozen E025 headline stage with an append-only semantic observer."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# The local controller uses Transformers only for checkpoint tokenization.  Keeping
# framework imports disabled is positive evidence that it performs no model compute.
os.environ.setdefault("USE_TORCH", "0")

import numpy as np
import psutil

from swarm_inference.experiments.experiment_025 import controller, headline
from swarm_inference.experiments.experiment_025.io import atomic_write_json, read_json, utc_now
from swarm_inference.experiments.experiment_025.vast_lifecycle import (
    destroy_all_from_ledger,
)
from swarm_inference.experiments.experiment_025.wire import Action

RUN_ID = "20260819T013016Z"
RUN_ROOT = Path("artifacts/runs/experiment-025-20260819T013016Z").resolve()
STAGE_ROOT = RUN_ROOT / "rental" / "stage-4-headline"
EVENT_PATH = RUN_ROOT / "telemetry" / "physical-run-events.jsonl"
PROMPT = (
    'Repeat exactly: "I am the swarm. 2.8 trillion parameters. '
    'Consumer GPUs. No datacenter required."'
)
TARGET_TEXT = (
    "I am the swarm. 2.8 trillion parameters. Consumer GPUs. "
    "No datacenter required."
)
DEFAULT_PUBLIC_TOKEN_BUDGET = 16
# The authoritative tokenizer encodes TARGET_TEXT as 20 tokens.  Four is the
# minimum extension needed to complete the exact sentence.
PUBLIC_TOKEN_BUDGET = 20
CHECKPOINT = Path(r"F:\models\Kimi-K3").resolve()


class HardReserveAcquisitionClock:
    """Select the watchdog hard acquisition limit in the frozen controller.

    The immutable controller computes ``min(stage deadline - 15 minutes,
    now + 25 minutes)``.  The first clock read after watchdog startup is
    deliberately seeded so both terms equal ``stage deadline - 15 minutes``.
    All later reads are real wall-clock time.  This keeps the 45-minute hard
    TTL and its 15-minute inference/cleanup reserve while removing the earlier
    cutoff that aborted demonstrably progressing replacements.
    """

    def __init__(self, real_time_module: Any) -> None:
        self.real_time_module = real_time_module
        self.lock = threading.Lock()
        self.seed: float | None = None

    def arm(self, stage_deadline_epoch: float) -> None:
        with self.lock:
            if self.seed is not None:
                raise RuntimeError("E025 acquisition clock was armed twice")
            # The frozen controller adds 25 minutes to this one clock read.
            self.seed = float(stage_deadline_epoch) - 40 * 60

    def time(self) -> float:
        with self.lock:
            if self.seed is not None:
                value = self.seed
                self.seed = None
                return value
        return float(self.real_time_module.time())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.real_time_module, name)


def ready_groups_with_full_backbone_concurrency(
    *,
    fragment_group_ids: list[str],
    backbone_group_ids: list[str],
    ready_group: Any,
    ready_parent_after_fragments: Any,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Start rolling readiness/replacement for every backbone group at once."""

    with ThreadPoolExecutor(max_workers=max(1, len(backbone_group_ids))) as pool:
        backbone_futures = {
            pool.submit(ready_group, group_id): group_id
            for group_id in backbone_group_ids
        }
        with ThreadPoolExecutor(max_workers=max(1, len(fragment_group_ids))) as fragment_pool:
            fragment_groups = list(fragment_pool.map(ready_group, fragment_group_ids))
        fragments = [worker for group in fragment_groups for worker in group]
        parent = ready_parent_after_fragments(fragments)
        backbone: list[Any] = []
        for future in as_completed(backbone_futures):
            backbone.extend(future.result())
    return fragments, parent, backbone


class EventSink:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size:
            raise RuntimeError(f"refusing non-empty canonical event stream: {path}")
        self.path = path
        self.origin_ns = time.perf_counter_ns()
        self.lock = threading.Lock()
        self.counter = 0
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def emit(self, event_type: str, **fields: Any) -> None:
        with self.lock:
            self.counter += 1
            event = {
                "schema_version": "experiment-025-physical-run-event-v1",
                "run_id": RUN_ID,
                "event_id": f"e025-{self.counter:09d}-{uuid.uuid4().hex}",
                "timestamp_utc": utc_now(),
                "monotonic_ns": time.perf_counter_ns() - self.origin_ns,
                "stage": "full-headline",
                "event_type": event_type,
                **fields,
            }
            self.handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            self.handle.flush()
            if self.counter % 64 == 0:
                os.fsync(self.handle.fileno())

    def close(self) -> None:
        with self.lock:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()


def _location(value: dict[str, Any]) -> dict[str, Any]:
    raw = value.get("raw", value)
    keys = (
        "country",
        "region",
        "state",
        "province",
        "city",
        "geolocation",
        "geolocode",
        "location",
        "latitude",
        "longitude",
        "datacenter",
    )
    return {key: raw[key] for key in keys if raw.get(key) not in (None, "")}


def _payload_bytes(arrays: dict[str, np.ndarray] | None) -> int:
    return sum(int(np.asarray(value).nbytes) for value in (arrays or {}).values())


def main() -> int:
    go = read_json(RUN_ROOT / "preflight" / "FULL_FLEET_GO.json")
    tokenizer, prompt_ids, tokenizer_identity = controller.tokenize_prompt(CHECKPOINT, PROMPT)
    target_ids = tokenizer.encode(TARGET_TEXT, add_special_tokens=False)
    target_roundtrip = tokenizer.decode(target_ids)
    if (
        max(DEFAULT_PUBLIC_TOKEN_BUDGET, len(target_ids)) != PUBLIC_TOKEN_BUDGET
        or target_roundtrip != TARGET_TEXT
    ):
        raise RuntimeError("E025 exact public sentence token-budget proof changed")
    atomic_write_json(
        RUN_ROOT / "preflight" / "headline-token-budget.json",
        {
            "schema_version": "experiment-025-headline-token-budget-v1",
            "status": "PASS",
            "run_id": RUN_ID,
            "prompt": PROMPT,
            "prompt_token_count": len(prompt_ids),
            "target_text": TARGET_TEXT,
            "default_max_new_tokens": DEFAULT_PUBLIC_TOKEN_BUDGET,
            "minimum_additional_tokens_required": (
                PUBLIC_TOKEN_BUDGET - DEFAULT_PUBLIC_TOKEN_BUDGET
            ),
            "target_token_count": len(target_ids),
            "target_token_ids": target_ids,
            "target_roundtrip": target_roundtrip,
            "tokenizer_identity": tokenizer_identity,
        },
    )
    sink = EventSink(EVENT_PATH)
    stop_observers = threading.Event()
    phase_state: dict[str, Any] = {
        "phase": "fleet",
        "prompt_length": None,
        "phase_started_ns": None,
        "token": None,
        "tokenizer": None,
    }
    worker_rows = {str(row["worker_id"]): row for row in go["fleet_plan"]["workers"]}
    instance_workers: dict[int, set[str]] = {}
    instance_workers_lock = threading.Lock()
    progress_lock = threading.Lock()
    progress_signature: dict[int, tuple[Any, ...]] = {}
    progress_monotonic: dict[int, float] = {}
    progress_log_hash: dict[int, str] = {}
    progress_log_query: dict[int, float] = {}
    instance_stall_events: dict[int, threading.Event] = {}
    ready_worker_ids: set[str] = set()

    observer_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sink.emit(
        "RUN_STARTED",
        immutable_image=go["image"]["immutable_reference"],
        source_tree_sha256=go["current_source_tree_sha256"],
        observer_sha256=observer_sha256,
        evidence_class="PHYSICAL",
        controller_compute_resource=False,
    )

    frozen_deadline_expression = (
        "acquisition_deadline = min(stage_deadline - 15 * 60, time.time() + 25 * 60)"
    )
    headline_source = Path(headline.__file__).read_text(encoding="utf-8")
    if frozen_deadline_expression not in headline_source:
        raise RuntimeError("E025 acquisition policy overlay no longer matches frozen source")

    real_headline_time = headline.time
    acquisition_clock = HardReserveAcquisitionClock(real_headline_time)
    original_start_watchdog = headline.start_watchdog

    def progress_aware_start_watchdog(*args: Any, **kwargs: Any) -> dict[str, Any]:
        receipt = original_start_watchdog(*args, **kwargs)
        hard_acquisition_deadline = float(receipt["deadline_epoch"]) - 15 * 60
        acquisition_clock.arm(float(receipt["deadline_epoch"]))
        sink.emit(
            "ACQUISITION_POLICY_APPLIED",
            policy="ROLLING_REPLACEMENT_WITH_HARD_TTL_RESERVE",
            legacy_fixed_25_minute_cutoff=False,
            per_instance_no_progress_timeout_seconds=240,
            hard_acquisition_deadline_epoch=hard_acquisition_deadline,
            inference_and_cleanup_reserve_seconds=15 * 60,
        )
        return receipt

    def observed_ready_groups(**kwargs: Any) -> tuple[list[Any], list[Any], list[Any]]:
        sink.emit(
            "READINESS_MONITORING_STARTED",
            policy="ALL_GROUPS_CONCURRENT_ROLLING_REPLACEMENT",
            backbone_group_count=len(kwargs["backbone_group_ids"]),
            fragment_group_count=len(kwargs["fragment_group_ids"]),
        )
        return ready_groups_with_full_backbone_concurrency(**kwargs)

    headline.time = acquisition_clock
    headline.start_watchdog = progress_aware_start_watchdog
    headline._ready_groups_with_concurrent_backbone = observed_ready_groups

    for worker_id, row in sorted(worker_rows.items()):
        offer = row["selected_offer"]
        spec = headline._spec(worker_id, str(row["role"]))
        common = {
            "worker_id": worker_id,
            "instance_group_id": row["instance_group_id"],
            "offer_id": offer["offer_id"],
            "machine_id": offer["machine_id"],
            "gpu_model": offer["gpu_name"],
            "gpu_slot": row["gpu_slot"],
            "vram_gib": offer["gpu_ram_gib"],
            "rental_rate_usd_per_hour": offer["dph_total"],
            "vast_reported_host_location_metadata": _location(offer),
        }
        sink.emit("OFFER_SELECTED", **common)
        sink.emit(
            "SHARD_ASSIGNED",
            **common,
            role=row["role"],
            layer=spec["layer"],
            fragment_id=(worker_id if row["role"] == "SUB_LAYER_WORKER" else None),
            assigned_bytes=row["assigned_tensor_bytes"],
            download_bytes=row["download_bytes_cold_cache"],
        )

    ledger_seen = 0
    group_instances: dict[str, int] = {}

    def observe_ledger() -> None:
        nonlocal ledger_seen
        ledger_path = STAGE_ROOT / "instance-ledger.jsonl"
        mapping = {
            "CREATE_REQUESTED": "INSTANCE_CREATE_REQUESTED",
            "CREATE_CONFIRMED": "INSTANCE_CREATED",
            "INSTANCE_RUNNING": "INSTANCE_BOOTING",
            "ENDPOINT_PUBLISHED": "WORKER_CONNECTING",
            "DESTROY_REQUESTED": "INSTANCE_DESTROY_REQUESTED",
            "DESTROY_CONFIRMED": "INSTANCE_DESTROYED",
        }
        while True:
            stopped = stop_observers.wait(1.0)
            if not ledger_path.is_file():
                if stopped:
                    break
                continue
            lines = ledger_path.read_text(encoding="utf-8").splitlines()
            for line in lines[ledger_seen:]:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                ledger_seen += 1
                event_type = mapping.get(str(row.get("event")))
                if event_type is None:
                    continue
                group_id = str(row.get("instance_group_id") or row.get("instance_label") or "")
                instance_id = row.get("instance_id")
                if event_type == "INSTANCE_CREATED" and group_id and instance_id is not None:
                    previous = group_instances.get(group_id)
                    if previous is not None and previous != int(instance_id):
                        sink.emit(
                            "INSTANCE_REPLACED",
                            instance_group_id=group_id,
                            previous_instance_id=previous,
                            replacement_instance_id=int(instance_id),
                            machine_id=row.get("machine_id"),
                        )
                    group_instances[group_id] = int(instance_id)
                sink.emit(
                    event_type,
                    lifecycle_ledger_sequence=row.get("sequence"),
                    instance_group_id=group_id or None,
                    instance_id=instance_id,
                    machine_id=row.get("machine_id"),
                    worker_id=row.get("worker_id"),
                    gpu_model=row.get("gpu_model"),
                    provider_status=row.get("final_status"),
                )
                if event_type == "INSTANCE_DESTROYED" and instance_id is not None:
                    with instance_workers_lock:
                        disconnected_worker_ids = sorted(
                            instance_workers.get(int(instance_id), set())
                        )
                    for disconnected_worker_id in disconnected_worker_ids:
                        sink.emit(
                            "WORKER_DISCONNECTED",
                            worker_id=disconnected_worker_id,
                            instance_id=int(instance_id),
                            reason="instance_destroyed",
                        )
            if stopped:
                break

    provider_previous: dict[int, tuple[Any, ...]] = {}

    def bootstrap_stage(row: dict[str, Any], log_text: str = "") -> str:
        lowered = log_text.lower()
        if "worker_ready" in lowered or "listening" in lowered:
            return "WORKER_LISTENING"
        if "k3-persistent:load" in lowered or "[cuda] device" in lowered:
            return "GPU_LOAD"
        if "snapshot" in lowered or "materialize" in lowered or "package" in lowered:
            return "PACKAGE_READY_OR_ACTIVATING"
        if "download" in lowered or "acquisition" in lowered:
            return "MODEL_DOWNLOAD"
        if row.get("vmem_usage") not in (None, 0, 0.0, "0"):
            return "GPU_LOAD"
        if str(row.get("actual_status", "")).lower() in {"running", "loading"}:
            return "CONTAINER_RUNNING_OR_MODEL_DOWNLOAD"
        return "CONTAINER_BOOTSTRAP"

    fatal_bootstrap_markers = (
        "native mxfp4 cuda tensor upload failed",
        "tensor allocation: out of memory",
        "no space left on device",
        "refusing to replace activated snapshot",
        "port is already allocated",
    )

    def observe_provider() -> None:
        while not stop_observers.wait(30.0):
            try:
                process = subprocess.run(
                    ["vastai", "show", "instances", "--raw"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=45,
                    check=False,
                )
                if process.returncode != 0:
                    sink.emit(
                        "RETRY",
                        operation="provider_progress_query",
                        error=process.stderr.strip()[:500],
                    )
                    continue
                rows = json.loads(process.stdout)
                for row in rows:
                    label = str(row.get("label", ""))
                    if not label.lower().startswith(f"e025-{RUN_ID.lower()}-"):
                        continue
                    instance_id = int(row["id"])
                    signature = (
                        row.get("actual_status"),
                        row.get("status_msg"),
                        row.get("disk_usage"),
                        row.get("inet_down_billed"),
                        row.get("vmem_usage"),
                    )
                    now = time.monotonic()
                    with instance_workers_lock:
                        expected_workers = set(instance_workers.get(instance_id, set()))
                    with progress_lock:
                        if progress_signature.get(instance_id) != signature:
                            progress_signature[instance_id] = signature
                            progress_monotonic[instance_id] = now
                        last_progress = progress_monotonic.setdefault(instance_id, now)
                        all_expected_ready = bool(expected_workers) and expected_workers.issubset(
                            ready_worker_ids
                        )
                        should_query_log = (
                            bool(expected_workers)
                            and
                            not all_expected_ready
                            and now - last_progress >= 120.0
                            and now - progress_log_query.get(instance_id, 0.0) >= 60.0
                        )
                        if should_query_log:
                            progress_log_query[instance_id] = now
                    log_text = ""
                    if should_query_log:
                        try:
                            log_text = subprocess.run(
                                ["vastai", "logs", str(instance_id), "--tail", "160"],
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                errors="replace",
                                timeout=45,
                                check=False,
                            ).stdout
                        except BaseException as exc:
                            sink.emit(
                                "RETRY",
                                operation="bootstrap_progress_log_query",
                                instance_id=instance_id,
                                error_type=type(exc).__name__,
                                error=str(exc),
                            )
                        lowered_log = log_text.lower()
                        substantive = bool(log_text.strip()) and not any(
                            marker in lowered_log
                            for marker in (
                                "error response from daemon",
                                "no such container",
                            )
                        )
                        if substantive:
                            log_digest = hashlib.sha256(log_text.encode("utf-8")).hexdigest()
                            with progress_lock:
                                previous_log_digest = progress_log_hash.get(instance_id)
                                progress_log_hash[instance_id] = log_digest
                                if previous_log_digest is not None and previous_log_digest != log_digest:
                                    progress_monotonic[instance_id] = time.monotonic()
                            sink.emit(
                                "BOOTSTRAP_PROGRESS",
                                instance_id=instance_id,
                                machine_id=row.get("machine_id"),
                                bootstrap_stage=bootstrap_stage(row, log_text),
                                log_changed=(
                                    previous_log_digest is not None
                                    and previous_log_digest != log_digest
                                ),
                                log_substantive=True,
                                raw_log_retained=False,
                            )
                            if any(marker in lowered_log for marker in fatal_bootstrap_markers):
                                with progress_lock:
                                    instance_stall_events.setdefault(
                                        instance_id, threading.Event()
                                    ).set()
                                sink.emit(
                                    "WORKER_UNHEALTHY",
                                    instance_id=instance_id,
                                    machine_id=row.get("machine_id"),
                                    bootstrap_stage=bootstrap_stage(row, log_text),
                                    reason="fatal_bootstrap_marker",
                                )
                    with progress_lock:
                        stalled_for = time.monotonic() - progress_monotonic.get(
                            instance_id, time.monotonic()
                        )
                        all_expected_ready = bool(expected_workers) and expected_workers.issubset(
                            ready_worker_ids
                        )
                        stall_event = instance_stall_events.get(instance_id)
                        if not all_expected_ready and stalled_for >= 240.0 and stall_event is not None:
                            newly_stalled = not stall_event.is_set()
                            stall_event.set()
                        else:
                            newly_stalled = False
                    if newly_stalled:
                        sink.emit(
                            "TIMEOUT",
                            instance_id=instance_id,
                            machine_id=row.get("machine_id"),
                            operation="bootstrap_readiness",
                            bootstrap_stage=bootstrap_stage(row, log_text),
                            no_progress_seconds=stalled_for,
                        )
                    if provider_previous.get(instance_id) == signature:
                        continue
                    provider_previous[instance_id] = signature
                    sink.emit(
                        "MODEL_DOWNLOAD_PROGRESS",
                        instance_id=instance_id,
                        machine_id=row.get("machine_id"),
                        instance_label=label,
                        provider_status=row.get("actual_status"),
                        provider_status_message=row.get("status_msg"),
                        bootstrap_stage=bootstrap_stage(row),
                        disk_usage_gb=row.get("disk_usage"),
                        provider_billed_download_counter=row.get("inet_down_billed"),
                        gpu_memory_usage=row.get("vmem_usage"),
                        gpu_utilization=row.get("gpu_util"),
                        gpu_model=row.get("gpu_name"),
                        driver_version=row.get("driver_version"),
                        cuda_max=row.get("cuda_max_good"),
                        vast_reported_host_location_metadata=_location(row),
                    )
            except BaseException as exc:
                sink.emit(
                    "RETRY",
                    operation="provider_progress_query",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

    ledger_thread = threading.Thread(target=observe_ledger, name="e025-event-ledger", daemon=True)
    provider_thread = threading.Thread(
        target=observe_provider, name="e025-event-provider", daemon=True
    )
    ledger_thread.start()
    provider_thread.start()

    original_wait = headline.wait_for_worker

    class CombinedAbort:
        def __init__(self, shared: threading.Event | None, stalled: threading.Event) -> None:
            self.shared = shared
            self.stalled = stalled

        def is_set(self) -> bool:
            return self.stalled.is_set() or (
                self.shared is not None and self.shared.is_set()
            )

        def wait(self, timeout: float | None = None) -> bool:
            deadline = None if timeout is None else time.monotonic() + timeout
            while not self.is_set():
                if deadline is not None and time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
            return self.is_set()

    def observed_wait_for_worker(**kwargs: Any) -> Any:
        worker_id = str(kwargs["worker_id"])
        instance_id = int(kwargs["instance_id"])
        with instance_workers_lock:
            instance_workers.setdefault(instance_id, set()).add(worker_id)
        with progress_lock:
            stall_event = instance_stall_events.setdefault(instance_id, threading.Event())
            progress_monotonic[instance_id] = time.monotonic()
            progress_signature.pop(instance_id, None)
            progress_log_hash.pop(instance_id, None)
            progress_log_query.pop(instance_id, None)
        original_abort = kwargs.get("abort_event")
        kwargs["abort_event"] = CombinedAbort(original_abort, stall_event)
        sink.emit(
            "WORKER_CONNECTING",
            worker_id=worker_id,
            instance_id=instance_id,
            machine_id=kwargs["offer"].machine_id,
            role=kwargs["role"],
        )
        sink.emit(
            "MODEL_DOWNLOAD_STARTED",
            worker_id=worker_id,
            instance_id=instance_id,
            model_id="moonshotai/Kimi-K3",
        )
        sink.emit(
            "SHARD_LOAD_STARTED",
            worker_id=worker_id,
            instance_id=instance_id,
            layer=kwargs["layer"],
        )
        try:
            worker = original_wait(**kwargs)
        except BaseException as exc:
            sink.emit(
                "TIMEOUT"
                if isinstance(exc, TimeoutError) or stall_event.is_set()
                else "WORKER_UNHEALTHY",
                worker_id=worker_id,
                instance_id=instance_id,
                operation="bootstrap_readiness",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            sink.emit(
                "ERROR",
                worker_id=worker_id,
                instance_id=instance_id,
                operation="bootstrap_readiness",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        ready = worker.ready
        with progress_lock:
            ready_worker_ids.add(worker_id)
        bootstrap = ready.get("bootstrap", {})
        acquisition = bootstrap.get("acquisition", {})
        sink.emit(
            "MODEL_DOWNLOAD_COMPLETED",
            worker_id=worker_id,
            instance_id=instance_id,
            downloaded_bytes=acquisition.get("downloaded_bytes"),
            all_cache_hits=acquisition.get("all_cache_hits"),
        )
        sink.emit(
            "MODEL_HASH_VERIFIED",
            worker_id=worker_id,
            instance_id=instance_id,
            checkpoint_fingerprint=ready.get("checkpoint_fingerprint"),
            snapshot_activation_sha256=ready.get("snapshot_activation_sha256"),
        )
        sink.emit(
            "SHARD_LOADED",
            worker_id=worker_id,
            instance_id=instance_id,
            machine_id=worker.machine_id,
            gpu_model=ready.get("gpu", {}).get("gpu_name"),
            gpu_uuid=ready.get("gpu", {}).get("gpu_uuid"),
            layer=worker.layer,
            native_backend=ready.get("executor", {}).get("native_primitive"),
        )
        sink.emit(
            "WORKER_CONNECTED",
            worker_id=worker_id,
            instance_id=instance_id,
            machine_id=worker.machine_id,
            host=worker.host,
            port=worker.port,
        )
        sink.emit(
            "WORKER_READY",
            worker_id=worker_id,
            instance_id=instance_id,
            machine_id=worker.machine_id,
            gpu=ready.get("gpu"),
            role=worker.role,
            layer=worker.layer,
            image_digest=ready.get("image_digest"),
            checkpoint_fingerprint=ready.get("checkpoint_fingerprint"),
            assignment=ready.get("assignment"),
        )
        return worker

    headline.wait_for_worker = observed_wait_for_worker

    original_request = controller.PhysicalController._request

    def observed_request(
        self: Any,
        endpoint: Any,
        action: Action,
        metadata: dict[str, Any],
        arrays: dict[str, np.ndarray] | None = None,
    ) -> Any:
        action_name = action.name
        position = metadata.get("position")
        layer = metadata.get("layer", endpoint.layer)
        task_id = (
            f"{self.session_id}:token-{position}:layer-{layer}"
            if action is Action.EXECUTE_STAGE
            else f"{self.session_id or 'control'}:{action_name}:{endpoint.worker_id}"
        )
        started_ns = time.perf_counter_ns()
        sink.emit(
            "MESSAGE_SEND_STARTED",
            phase=phase_state["phase"],
            sender_worker_id="e025-controller",
            receiver_worker_id=endpoint.worker_id,
            task_id=task_id,
            token_index=position,
            layer=layer,
            message_class=action_name,
            payload_bytes=_payload_bytes(arrays),
        )
        if action is Action.EXECUTE_STAGE:
            sink.emit(
                "LAYER_EXECUTION_STARTED",
                phase=phase_state["phase"],
                task_id=task_id,
                token_index=position,
                layer=layer,
                worker_id=endpoint.worker_id,
                machine_id=endpoint.machine_id,
            )
            if int(layer) == 92:
                sink.emit(
                    "TOKEN_SAMPLE_STARTED",
                    phase=phase_state["phase"],
                    task_id=task_id,
                    token_index=position,
                    worker_id=endpoint.worker_id,
                )
        try:
            result = original_request(self, endpoint, action, metadata, arrays)
        except BaseException as exc:
            sink.emit(
                "ERROR",
                phase=phase_state["phase"],
                task_id=task_id,
                worker_id=endpoint.worker_id,
                operation=action_name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        response_metadata, _response_arrays, transport = result
        sink.emit(
            "MESSAGE_SEND_COMPLETED",
            phase=phase_state["phase"],
            sender_worker_id="e025-controller",
            receiver_worker_id=endpoint.worker_id,
            task_id=task_id,
            token_index=position,
            layer=layer,
            message_class=action_name,
            payload_bytes=transport.get("request_wire_bytes"),
            controller_observed_roundtrip_ns=transport.get("wall_ns"),
        )
        sink.emit(
            "MESSAGE_RECEIVED",
            phase=phase_state["phase"],
            sender_worker_id="e025-controller",
            receiver_worker_id=endpoint.worker_id,
            task_id=task_id,
            token_index=position,
            layer=layer,
            message_class=action_name,
            payload_bytes=transport.get("request_wire_bytes"),
            observation_scope="confirmed_by_authenticated_response",
        )
        sink.emit(
            "MESSAGE_SEND_STARTED",
            phase=phase_state["phase"],
            sender_worker_id=endpoint.worker_id,
            receiver_worker_id="e025-controller",
            task_id=task_id,
            token_index=position,
            layer=layer,
            message_class=f"{action_name}_RESULT",
            payload_bytes=transport.get("response_wire_bytes"),
            observation_scope="post_response_controller_receipt",
        )
        sink.emit(
            "MESSAGE_SEND_COMPLETED",
            phase=phase_state["phase"],
            sender_worker_id=endpoint.worker_id,
            receiver_worker_id="e025-controller",
            task_id=task_id,
            token_index=position,
            layer=layer,
            message_class=f"{action_name}_RESULT",
            payload_bytes=transport.get("response_wire_bytes"),
            observation_scope="post_response_controller_receipt",
        )
        sink.emit(
            "MESSAGE_RECEIVED",
            phase=phase_state["phase"],
            sender_worker_id=endpoint.worker_id,
            receiver_worker_id="e025-controller",
            task_id=task_id,
            token_index=position,
            layer=layer,
            message_class=f"{action_name}_RESULT",
            payload_bytes=transport.get("response_wire_bytes"),
        )
        if action is Action.EXECUTE_STAGE:
            execution = response_metadata.get("execution", {})
            selected = execution.get("selected_expert_ids", [])
            sink.emit(
                "ROUTE_COMPUTED",
                phase=phase_state["phase"],
                task_id=task_id,
                token_index=position,
                layer=layer,
                selected_expert_ids=selected,
                route_source="physical_worker_execution_receipt",
            )
            dispatch = execution.get("external_expert_dispatch")
            if isinstance(dispatch, dict):
                sink.emit(
                    "REDUCTION_STARTED",
                    phase=phase_state["phase"],
                    task_id=task_id,
                    token_index=position,
                    layer=layer,
                    worker_id=endpoint.worker_id,
                    observation_scope="post_response_physical_receipt",
                )
                for fragment in dispatch.get("workers", []):
                    fragment_id = str(fragment.get("worker_id"))
                    fragment_task = f"{task_id}:fragment:{fragment_id}"
                    common = {
                        "phase": phase_state["phase"],
                        "task_id": fragment_task,
                        "parent_task_id": task_id,
                        "token_index": position,
                        "layer": layer,
                        "fragment_id": fragment_id,
                        "worker_id": fragment_id,
                        "expert_ids": fragment.get("owned_selected_expert_ids"),
                        "observation_scope": "post_response_physical_receipt",
                    }
                    sink.emit("EXPERT_DISPATCHED", **common)
                    sink.emit(
                        "MESSAGE_SEND_STARTED",
                        **common,
                        sender_worker_id=endpoint.worker_id,
                        receiver_worker_id=fragment_id,
                        message_class="EXPERT_DISPATCH",
                        payload_bytes=fragment.get("request_wire_bytes"),
                    )
                    sink.emit(
                        "MESSAGE_SEND_COMPLETED",
                        **common,
                        sender_worker_id=endpoint.worker_id,
                        receiver_worker_id=fragment_id,
                        message_class="EXPERT_DISPATCH",
                        payload_bytes=fragment.get("request_wire_bytes"),
                    )
                    sink.emit(
                        "MESSAGE_RECEIVED",
                        **common,
                        sender_worker_id=endpoint.worker_id,
                        receiver_worker_id=fragment_id,
                        message_class="EXPERT_DISPATCH",
                        payload_bytes=fragment.get("request_wire_bytes"),
                    )
                    sink.emit("FRAGMENT_EXECUTION_STARTED", **common)
                    sink.emit(
                        "FRAGMENT_EXECUTION_COMPLETED",
                        **common,
                        native_backend=fragment.get("native_primitive"),
                        native_expert_calls=fragment.get("native_expert_calls"),
                        cuda_ms=fragment.get("cuda_ms"),
                        wall_ms=fragment.get("wall_ms"),
                        input_fingerprint=fragment.get("input_fingerprint"),
                        output_fingerprint=fragment.get("output_fingerprint"),
                    )
                    sink.emit(
                        "MESSAGE_SEND_STARTED",
                        **common,
                        sender_worker_id=fragment_id,
                        receiver_worker_id=endpoint.worker_id,
                        message_class="EXPERT_RESULT",
                        payload_bytes=fragment.get("response_wire_bytes"),
                    )
                    sink.emit(
                        "MESSAGE_SEND_COMPLETED",
                        **common,
                        sender_worker_id=fragment_id,
                        receiver_worker_id=endpoint.worker_id,
                        message_class="EXPERT_RESULT",
                        payload_bytes=fragment.get("response_wire_bytes"),
                    )
                    sink.emit(
                        "MESSAGE_RECEIVED",
                        **common,
                        sender_worker_id=fragment_id,
                        receiver_worker_id=endpoint.worker_id,
                        message_class="EXPERT_RESULT",
                        payload_bytes=fragment.get("response_wire_bytes"),
                    )
                    sink.emit("EXPERT_RESULT_RETURNED", **common)
                sink.emit(
                    "REDUCTION_COMPLETED",
                    phase=phase_state["phase"],
                    task_id=task_id,
                    token_index=position,
                    layer=layer,
                    worker_id=endpoint.worker_id,
                    workers_invoked=dispatch.get("workers_invoked"),
                    every_selected_expert_executed_once=dispatch.get(
                        "every_selected_expert_executed_once"
                    ),
                    whole_layer_fallback=dispatch.get("whole_layer_fallback"),
                )
            sink.emit(
                "LAYER_EXECUTION_COMPLETED",
                phase=phase_state["phase"],
                task_id=task_id,
                token_index=position,
                layer=layer,
                worker_id=endpoint.worker_id,
                machine_id=endpoint.machine_id,
                duration_ns=time.perf_counter_ns() - started_ns,
                native_backend=execution.get("native_primitive"),
                input_bytes=_payload_bytes(arrays),
                output_bytes=transport.get("response_wire_bytes"),
                selected_expert_ids=selected,
                controller_compute_fallback=False,
            )
        return result

    controller.PhysicalController._request = observed_request
    original_execute = controller.PhysicalController.execute_token

    def observed_execute(self: Any, token_id: int, position: int, **kwargs: Any) -> Any:
        task_id = f"{self.session_id}:token-{position}"
        phase_state["token"] = {"position": position, "task_id": task_id}
        sink.emit(
            "TOKEN_EXECUTION_STARTED",
            phase=phase_state["phase"],
            task_id=task_id,
            token_index=position,
            input_token_id=int(token_id),
        )
        started_ns = time.perf_counter_ns()
        record = original_execute(self, token_id, position, **kwargs)
        elapsed_ns = time.perf_counter_ns() - started_ns
        sink.emit(
            "TOKEN_EXECUTION_COMPLETED",
            phase=phase_state["phase"],
            task_id=task_id,
            token_index=position,
            input_token_id=int(token_id),
            sampled_token_id=record.get("sampled_token_id"),
            duration_ns=elapsed_ns,
            layer_count=len(record.get("stages", [])),
            controller_model_compute=False,
        )
        prompt_length = phase_state.get("prompt_length")
        if isinstance(prompt_length, int) and position >= prompt_length - 1:
            emitted_index = position - (prompt_length - 1)
            tokenizer = phase_state.get("tokenizer")
            sampled_token_id = record.get("sampled_token_id")
            decoded_fragment = (
                tokenizer.decode([int(sampled_token_id)], skip_special_tokens=True)
                if tokenizer is not None and sampled_token_id is not None
                else None
            )
            sink.emit(
                "TOKEN_EMITTED",
                phase=phase_state["phase"],
                task_id=task_id,
                token_index=emitted_index,
                model_position=position,
                token_id=record.get("sampled_token_id"),
                decoded_text_fragment=decoded_fragment,
                latency_seconds=record.get("elapsed_seconds"),
                cumulative_elapsed_seconds=(
                    (time.perf_counter_ns() - int(phase_state["phase_started_ns"])) / 1e9
                    if phase_state.get("phase_started_ns") is not None
                    else None
                ),
            )
        return record

    controller.PhysicalController.execute_token = observed_execute
    original_generation = headline.run_physical_generation

    def observed_generation(**kwargs: Any) -> dict[str, Any]:
        phase = "correctness" if kwargs["output_path"] == RUN_ROOT / "correctness" / "physical-two-token.json" else "headline"
        if phase == "headline" and int(kwargs["max_new_tokens"]) < PUBLIC_TOKEN_BUDGET:
            raise TimeoutError(
                "E025 remaining TTL cannot fit the tokenizer-proven exact sentence"
            )
        phase_state["phase"] = phase
        tokenizer, prompt_ids, _identity = controller.tokenize_prompt(
            kwargs["stage_zero_snapshot"], kwargs["prompt"]
        )
        phase_state["tokenizer"] = tokenizer
        phase_state["prompt_length"] = len(prompt_ids)
        phase_state["phase_started_ns"] = time.perf_counter_ns()
        sink.emit(
            "GENERATION_STARTED",
            phase=phase,
            prompt=kwargs["prompt"],
            prompt_token_count=len(prompt_ids),
            max_new_tokens=kwargs["max_new_tokens"],
            decoding="greedy_argmax",
        )
        value = original_generation(**kwargs)
        if phase == "headline" and str(value.get("decoded_text", "")).strip() != TARGET_TEXT:
            sink.emit(
                "ERROR",
                phase=phase,
                operation="exact_public_sentence_gate",
                expected_text=TARGET_TEXT,
                observed_text=value.get("decoded_text"),
            )
            raise RuntimeError("E025 physical generation did not reproduce the exact public sentence")
        sink.emit(
            "GENERATION_COMPLETED",
            phase=phase,
            status=value.get("status"),
            generated_token_ids=value.get("generated_token_ids"),
            decoded_text=value.get("decoded_text"),
            elapsed_seconds=value.get("elapsed_seconds"),
            decode_tokens_per_second=value.get("decode_tokens_per_second"),
            network_wire_bytes=value.get("network_wire_bytes"),
        )
        return value

    headline.run_physical_generation = observed_generation

    original_create_instance = headline.VastClient.create_instance

    def guarded_create_instance(self: Any, *args: Any, **kwargs: Any) -> int:
        receipt_path = Path(kwargs["watchdog_receipt"])
        trigger_path = receipt_path.parent / "WATCHDOG_TRIGGER"
        stop_path = receipt_path.parent / "WATCHDOG_STOP"
        receipt = read_json(receipt_path)
        watchdog_pid = int(receipt.get("watchdog_pid", -1))
        if trigger_path.exists() or stop_path.exists():
            raise RuntimeError("E025 create refused after watchdog trigger or stop")
        if receipt.get("status") != "RUNNING" or time.time() >= float(
            receipt.get("deadline_epoch", 0.0)
        ):
            raise RuntimeError("E025 create refused without a live watchdog window")
        if watchdog_pid <= 0 or not psutil.pid_exists(watchdog_pid):
            raise RuntimeError("E025 create refused because the independent watchdog exited")
        return original_create_instance(self, *args, **kwargs)

    headline.VastClient.create_instance = guarded_create_instance

    receipt: dict[str, Any] | None = None
    exit_code = 2
    try:
        receipt = headline.run_headline_stage(
            run_id=RUN_ID,
            stage_root=STAGE_ROOT,
            full_fleet_go_path=RUN_ROOT / "preflight" / "FULL_FLEET_GO.json",
            image_receipt_path=RUN_ROOT / "preflight" / "deployment-image.json",
            private_root=Path(".e025-private").resolve() / RUN_ID,
            physical_placement_path=RUN_ROOT / "preflight" / "physical-placement.json",
            checkpoint=CHECKPOINT,
            oracle_trace=Path(
                "artifacts/experiment-014/oracle-full-93-idot0/hidden-trace.f32"
            ).resolve(),
            oracle_routes=Path(
                "artifacts/experiment-014/oracle-full-93-idot0/routes.txt"
            ).resolve(),
            correctness_output_path=RUN_ROOT / "correctness" / "physical-two-token.json",
            generation_output_path=RUN_ROOT / "generation" / "headline-generation.json",
            summary_output_path=RUN_ROOT / "final" / "summary.json",
            prompt=PROMPT,
            max_new_tokens=PUBLIC_TOKEN_BUDGET,
            disk_gb=60,
        )
        atomic_write_json(RUN_ROOT / "final" / "headline-stage.json", receipt)
        sink.emit(
            "RUN_COMPLETED" if receipt.get("status") == "PASS" else "ABORT",
            status=receipt.get("status"),
            cleanup=receipt.get("cleanup"),
            gates=receipt.get("gates"),
        )
        exit_code = 0 if receipt.get("status") == "PASS" else 2
    except BaseException as exc:
        sink.emit(
            "ERROR",
            phase=phase_state["phase"],
            operation="headline_stage",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        sink.emit("ABORT", reason="headline_stage_exception")
        ledger_path = STAGE_ROOT / "instance-ledger.jsonl"
        if ledger_path.is_file():
            cleanup = destroy_all_from_ledger(
                ledger_path=ledger_path,
                run_id=RUN_ID,
                reason="observed-headline-wrapper-exception",
                executable="vastai",
                attempts=6,
            )
            atomic_write_json(STAGE_ROOT / "observer-exception-cleanup.json", cleanup)
        raise
    finally:
        stop_observers.set()
        ledger_thread.join(timeout=5.0)
        provider_thread.join(timeout=55.0)
        if (STAGE_ROOT / "WATCHDOG_TRIGGER").is_file():
            sink.emit("WATCHDOG_TRIGGERED", reason="independent_watchdog_trigger_file_present")
        sink.close()
    print(
        json.dumps(
            {
                "status": (receipt or {}).get("status"),
                "cleanup": (receipt or {}).get("cleanup"),
                "failure": (receipt or {}).get("failure"),
                "event_path": str(EVENT_PATH),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
