from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import itertools
import json
import math
import random
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = (
    REPO_ROOT / "artifacts" / "runs" / "experiment-025-20260819T013016Z"
)
ATTEMPT_005_EVENT_SHA256 = (
    "84fef174ca83e99c7c227c5cfe2a1eb646280c9dc07c0522a39c937f10d7225e"
)
WORKER_IMAGE_DIGEST = (
    "sha256:46cad5a031b98aeecdee9ba471e2e8338eb8e9e485d1d2e90a1505f8890ee8e4"
)
CONTROLLER_BUG_MARKERS = (
    "watchdog-startup-failure",
    "late-finalizer-cleanup",
    "private-image-pull-failure",
    "observer-false-stall",
    "controller-schema",
    "before-parser-fix",
    "before-bootstrap-retry-fix",
    "before-headline-readiness-fix",
)
PHYSICAL_FAILURE_MARKERS = (
    "host-disappeared",
    "host-never-ready",
    "worker-ready-timeout",
    "transient-machine",
    "gpu-load-oom",
    "container-running-stall",
    "container-bootstrap-stall",
    "gpu-load-stall",
    "bootstrap-stall",
    "model-download-stall",
    "create-id-lost",
)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def parse_time(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def seconds_between(left: str | None, right: str | None) -> float | None:
    start = parse_time(left)
    end = parse_time(right)
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    payload["artifact_sha256"] = canonical_sha256(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def quantile(values: Iterable[float], probability: float) -> float | None:
    rows = sorted(float(value) for value in values)
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    position = (len(rows) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return rows[lower]
    weight = position - lower
    return rows[lower] * (1.0 - weight) + rows[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    return {
        "count": len(rows),
        "minimum": min(rows) if rows else None,
        "median": statistics.median(rows) if rows else None,
        "p90": quantile(rows, 0.90),
        "maximum": max(rows) if rows else None,
    }


def _ledger_observations(run_root: Path) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for ledger_path in sorted((run_root / "rental").glob("*/instance-ledger.jsonl")):
        rows = read_jsonl(ledger_path)
        stage = ledger_path.parent.name
        requests = [row for row in rows if row.get("event") == "CREATE_REQUESTED"]
        used_request_sequences: set[int] = set()
        for confirmed in [row for row in rows if row.get("event") == "CREATE_CONFIRMED"]:
            matches = [
                row
                for row in requests
                if int(row["sequence"]) not in used_request_sequences
                and int(row.get("offer_id", -1)) == int(confirmed.get("offer_id", -2))
                and int(row.get("machine_id", -1))
                == int(confirmed.get("machine_id", -2))
                and str(row.get("instance_label", ""))
                == str(confirmed.get("instance_label", ""))
                and int(row["sequence"]) < int(confirmed["sequence"])
            ]
            request = max(matches, key=lambda row: int(row["sequence"])) if matches else {}
            if request:
                used_request_sequences.add(int(request["sequence"]))
            instance_id = int(confirmed["instance_id"])
            related = [row for row in rows if row.get("instance_id") == instance_id]
            ready = [row for row in related if row.get("event") == "WORKER_READY"]
            endpoint = [row for row in related if row.get("event") == "ENDPOINT_PUBLISHED"]
            running = [row for row in related if row.get("event") == "INSTANCE_RUNNING"]
            destroyed = [row for row in related if row.get("event") == "DESTROY_CONFIRMED"]
            logs_failed = [
                row for row in related if row.get("event") == "LOGS_RETRIEVAL_FAILED"
            ]
            create_time = str(confirmed.get("creation_time") or confirmed["timestamp"])
            ready_times = [
                str(row.get("worker_ready_time") or row["timestamp"]) for row in ready
            ]
            gpu_count = int(confirmed.get("gpu_count") or request.get("gpu_count") or 1)
            group_ready = len(ready) >= gpu_count
            lowered_stage = stage.lower()
            controller_side = any(marker in lowered_stage for marker in CONTROLLER_BUG_MARKERS)
            named_physical = any(marker in lowered_stage for marker in PHYSICAL_FAILURE_MARKERS)
            machine_attributable = bool(not group_ready and named_physical and not controller_side)
            observations.append(
                {
                    "observation_id": f"{stage}:{instance_id}",
                    "stage_directory": stage,
                    "attempt_number": (
                        5
                        if stage == "stage-4-headline"
                        else int(stage.split("attempt-")[1][:3])
                        if "attempt-" in stage
                        else None
                    ),
                    "instance_id": instance_id,
                    "machine_id": int(confirmed.get("machine_id", -1)),
                    "offer_id": int(confirmed.get("offer_id", -1)),
                    "gpu_model": str(
                        confirmed.get("gpu_model") or request.get("gpu_model") or ""
                    ),
                    "gpu_count": gpu_count,
                    "region": None,
                    "advertised_reliability": request.get("advertised_reliability"),
                    "advertised_internet_down_mbps": request.get(
                        "advertised_internet_down_mbps"
                    ),
                    "advertised_internet_up_mbps": request.get(
                        "advertised_internet_up_mbps"
                    ),
                    "advertised_disk_bandwidth_mbps": request.get(
                        "advertised_disk_bandwidth_mbps"
                    ),
                    "active_rental_rate_usd_per_hour": request.get(
                        "active_rental_rate_usd_per_hour"
                    ),
                    "storage_rate_usd_per_gb_month": request.get(
                        "storage_rate_usd_per_gb_month"
                    ),
                    "internet_ingress_rate_usd_per_gb": request.get(
                        "internet_ingress_rate_usd_per_gb"
                    ),
                    "internet_egress_rate_usd_per_gb": request.get(
                        "internet_egress_rate_usd_per_gb"
                    ),
                    "requested_disk_gb": request.get("requested_disk_gb"),
                    "assigned_package_bytes": None,
                    "create_timestamp_utc": create_time,
                    "first_substantive_progress_timestamp_utc": (
                        str(running[0]["timestamp"]) if running else None
                    ),
                    "model_download_completed_timestamp_utc": None,
                    "gpu_load_timestamp_utc": None,
                    "ready_timestamp_utc": max(ready_times) if ready_times else None,
                    "destroyed_timestamp_utc": (
                        str(destroyed[0].get("destroy_confirmed_time") or destroyed[0]["timestamp"])
                        if destroyed
                        else None
                    ),
                    "time_to_first_substantive_progress_seconds": seconds_between(
                        create_time, str(running[0]["timestamp"]) if running else None
                    ),
                    "time_to_model_download_completion_seconds": None,
                    "time_to_gpu_load_seconds": None,
                    "time_to_ready_seconds": (
                        seconds_between(create_time, max(ready_times))
                        if ready_times
                        else None
                    ),
                    "download_throughput_mbps": None,
                    "public_mapping_observed": bool(endpoint or ready),
                    "public_mapping_failed": bool(not endpoint and not ready),
                    "ssl_or_connectivity_failed": bool(logs_failed),
                    "disappeared": "disappeared" in lowered_stage,
                    "terminal_or_exited": False,
                    "no_progress_timeout": bool(
                        "timeout" in lowered_stage or "stall" in lowered_stage
                    ),
                    "worker_ready_count": len(ready),
                    "expected_worker_count": gpu_count,
                    "reached_group_ready": group_ready,
                    "remained_healthy_after_ready": group_ready,
                    "controller_side_failure": controller_side,
                    "machine_attributable_failure": machine_attributable,
                    "classification_basis": (
                        "controller/image defect encoded by retained stage directory"
                        if controller_side
                        else "physical provider/bootstrap failure encoded by retained stage directory"
                        if machine_attributable
                        else "authenticated group READY"
                        if group_ready
                        else "right-censored or unattributed failure"
                    ),
                }
            )
        for lost in [row for row in rows if row.get("event") == "CREATE_ID_LOST"]:
            machine_id = int(lost.get("machine_id", -1))
            observations.append(
                {
                    "observation_id": f"{stage}:create-id-lost:{lost['sequence']}",
                    "stage_directory": stage,
                    "attempt_number": None,
                    "instance_id": None,
                    "machine_id": machine_id,
                    "offer_id": int(lost.get("offer_id", -1)),
                    "gpu_model": "",
                    "gpu_count": 1,
                    "region": None,
                    "assigned_package_bytes": None,
                    "create_timestamp_utc": lost.get("timestamp"),
                    "ready_timestamp_utc": None,
                    "time_to_ready_seconds": None,
                    "public_mapping_observed": False,
                    "public_mapping_failed": True,
                    "ssl_or_connectivity_failed": False,
                    "disappeared": False,
                    "terminal_or_exited": False,
                    "no_progress_timeout": False,
                    "worker_ready_count": 0,
                    "expected_worker_count": 1,
                    "reached_group_ready": False,
                    "remained_healthy_after_ready": False,
                    "controller_side_failure": False,
                    "machine_attributable_failure": True,
                    "classification_basis": "provider create returned no attributable instance ID",
                }
            )
    return observations


def _enrich_attempt_005(
    run_root: Path, observations: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_path = run_root / "telemetry" / "physical-run-events.jsonl"
    if sha256_file(event_path) != ATTEMPT_005_EVENT_SHA256:
        raise RuntimeError("Attempt 005 canonical event log digest changed")
    events = read_jsonl(event_path)
    inventory = read_json(run_root / "telemetry" / "attempt-005-node-inventory.json")
    terminal = read_json(run_root / "final" / "attempt-005-terminal-result.json")
    deadline = parse_time(terminal["deadline"]["hard_acquisition_deadline_utc"])
    if deadline is None:
        raise AssertionError("Attempt 005 deadline missing")
    by_instance: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for event in events:
        if event.get("instance_id") is not None:
            by_instance[int(event["instance_id"])].append(event)
    inventory_by_instance = {
        int(row["vast_instance_id"]): row for row in inventory["instances"]
    }
    attempt_observations = {
        int(row["instance_id"]): row
        for row in observations
        if row["stage_directory"] == "stage-4-headline"
        and row["instance_id"] is not None
    }
    for instance_id, inventory_row in inventory_by_instance.items():
        row = attempt_observations.get(instance_id)
        if row is None:
            continue
        related = by_instance[instance_id]
        event_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for event in related:
            event_groups[str(event["event_type"])].append(event)
        created_event = next(iter(event_groups["INSTANCE_CREATED"]), None)
        created = str(
            created_event["timestamp_utc"]
            if created_event is not None
            else inventory_row["created_at_utc"]
        )
        destroyed_event = next(iter(event_groups["INSTANCE_DESTROYED"]), None)
        destroyed = (
            str(destroyed_event["timestamp_utc"])
            if destroyed_event is not None
            else str(inventory_row["destroyed_at_utc"])
        )
        ready_events = event_groups["WORKER_READY"]
        complete_events = event_groups["MODEL_DOWNLOAD_COMPLETED"]
        load_events = event_groups["SHARD_LOAD_STARTED"]
        progress_events = [
            event
            for event in (
                event_groups["MODEL_DOWNLOAD_PROGRESS"]
                + event_groups["BOOTSTRAP_PROGRESS"]
            )
            if event.get("provider_status_message")
            or event.get("log_substantive") is True
            or event.get("disk_usage_gb") not in (None, -1)
        ]
        errors = " ".join(
            str(event.get("error") or event.get("reason") or "").lower()
            for event in related
            if event["event_type"] in {"ERROR", "WORKER_UNHEALTHY", "TIMEOUT"}
        )
        downloaded_bytes = sum(
            int(event.get("downloaded_bytes", 0)) for event in complete_events
        )
        full_ready = bool(inventory_row["worker_ids"]) and len(
            inventory_row["ready_worker_ids"]
        ) == len(inventory_row["worker_ids"])
        destroyed_time = parse_time(destroyed)
        early_destroy = bool(
            destroyed_time is not None
            and destroyed_time < deadline - dt.timedelta(seconds=30)
        )
        physical_failure = bool(
            early_destroy
            and not full_ready
            and any(
                marker in errors
                for marker in (
                    "disappeared",
                    "terminal",
                    "exited",
                    "no public mapping",
                    "no-progress",
                    "no progress",
                    "timed out",
                )
            )
        )
        first_progress = min(
            (str(event["timestamp_utc"]) for event in progress_events), default=None
        )
        download_completed = max(
            (str(event["timestamp_utc"]) for event in complete_events), default=None
        )
        gpu_load = min(
            (str(event["timestamp_utc"]) for event in load_events), default=None
        )
        ready_timestamp = max(
            (str(event["timestamp_utc"]) for event in ready_events), default=None
        )
        duration = seconds_between(created, download_completed)
        row.update(
            {
                "role_class": (
                    "FRAGMENT"
                    if any("sub-" in worker_id for worker_id in inventory_row["worker_ids"])
                    else "PARENT"
                    if any("parent" in worker_id for worker_id in inventory_row["worker_ids"])
                    else "BACKBONE"
                ),
                "instance_group_id": inventory_row.get("assigned_group"),
                "region": inventory_row["vast_reported_host_location_metadata"].get(
                    "geolocation"
                ),
                "advertised_reliability": inventory_row.get(
                    "advertised_reliability"
                ),
                "advertised_internet_down_mbps": inventory_row["network"].get(
                    "internet_down_mbps"
                ),
                "advertised_internet_up_mbps": inventory_row["network"].get(
                    "internet_up_mbps"
                ),
                "advertised_disk_bandwidth_mbps": inventory_row["network"].get(
                    "disk_bandwidth_mbps"
                ),
                "active_rental_rate_usd_per_hour": inventory_row.get(
                    "active_rental_rate_usd_per_hour"
                ),
                "storage_rate_usd_per_gb_month": inventory_row.get(
                    "storage_rate_usd_per_gb_month"
                ),
                "requested_disk_gb": inventory_row.get("requested_disk_gb"),
                "assigned_package_bytes": sum(
                    int(attempt.get("download_bytes", 0))
                    for attempt in inventory["worker_attempts"]
                    if int(attempt["vast_instance_id"]) == instance_id
                ),
                "create_timestamp_utc": created,
                "first_substantive_progress_timestamp_utc": first_progress,
                "model_download_completed_timestamp_utc": download_completed,
                "gpu_load_timestamp_utc": gpu_load,
                "ready_timestamp_utc": ready_timestamp,
                "destroyed_timestamp_utc": destroyed,
                "time_to_first_substantive_progress_seconds": seconds_between(
                    created, first_progress
                ),
                "time_to_model_download_completion_seconds": duration,
                "time_to_gpu_load_seconds": seconds_between(created, gpu_load),
                "time_to_ready_seconds": seconds_between(created, ready_timestamp),
                "download_throughput_mbps": (
                    downloaded_bytes * 8 / duration / 1e6
                    if downloaded_bytes and duration and duration > 0
                    else None
                ),
                "public_mapping_observed": bool(
                    ready_events or event_groups["WORKER_CONNECTED"]
                ),
                "public_mapping_failed": "public mapping" in errors,
                "ssl_or_connectivity_failed": any(
                    marker in errors for marker in ("ssl", "connect", "unreachable")
                ),
                "disappeared": "disappeared" in errors,
                "terminal_or_exited": any(
                    marker in errors for marker in ("terminal", "exited")
                ),
                "no_progress_timeout": any(
                    event["event_type"] == "TIMEOUT" for event in related
                ),
                "worker_ready_count": len(inventory_row["ready_worker_ids"]),
                "expected_worker_count": len(inventory_row["worker_ids"]),
                "reached_group_ready": full_ready,
                "remained_healthy_after_ready": full_ready and not early_destroy,
                "controller_side_failure": False,
                "machine_attributable_failure": physical_failure,
                "right_censored_at_attempt_deadline": bool(
                    not full_ready and not early_destroy
                ),
                "classification_basis": (
                    "all assigned workers reached READY and instance survived to cutoff"
                    if full_ready and not early_destroy
                    else "physical failure before cutoff with retained provider/timeout evidence"
                    if physical_failure
                    else "partial group readiness before early replacement"
                    if early_destroy
                    else "still progressing or unresolved at the hard cutoff"
                ),
            }
        )
    return events, terminal


def _machine_history(
    run_root: Path, observations: list[dict[str, Any]]
) -> dict[str, Any]:
    prior_exclusions = set(
        int(machine_id)
        for machine_id in read_json(
            run_root / "preflight" / "headline-failed-machine-exclusions.json"
        )["machine_ids"]
    )
    grouped: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in observations:
        if int(row["machine_id"]) >= 0:
            grouped[int(row["machine_id"])].append(row)
    machines: list[dict[str, Any]] = []
    removed_prior_exclusions: list[dict[str, Any]] = []
    for machine_id, rows in sorted(grouped.items()):
        successes = [row for row in rows if row["reached_group_ready"]]
        healthy = [row for row in rows if row["remained_healthy_after_ready"]]
        attributable = [row for row in rows if row["machine_attributable_failure"]]
        controller = [row for row in rows if row["controller_side_failure"]]
        mapping_observed = [row for row in rows if row["public_mapping_observed"]]
        mapping_failed = [row for row in rows if row["public_mapping_failed"]]
        ready_seconds = [
            float(row["time_to_ready_seconds"])
            for row in successes
            if row.get("time_to_ready_seconds") is not None
        ]
        download_rates = [
            float(row["download_throughput_mbps"])
            for row in rows
            if row.get("download_throughput_mbps") is not None
        ]
        previously_excluded = machine_id in prior_exclusions
        controller_only_prior_failure = bool(
            previously_excluded
            and controller
            and not attributable
            and not any(
                row["machine_attributable_failure"]
                for row in rows
                if row["stage_directory"] == "stage-4-headline"
            )
        )
        hard_excluded = bool(
            (previously_excluded and not controller_only_prior_failure)
            or (
                len(attributable) >= 2
                and not healthy
            )
            or sum(bool(row["disappeared"]) for row in rows) >= 2
            or sum(bool(row["public_mapping_failed"]) for row in rows) >= 2
        )
        if controller_only_prior_failure:
            removed_prior_exclusions.append(
                {
                    "machine_id": machine_id,
                    "reason": "all retained failures were controller/image-attributable",
                    "observation_ids": [row["observation_id"] for row in controller],
                }
            )
        def observed_values(
            field: str, source_rows: list[dict[str, Any]] = rows
        ) -> list[Any]:
            return sorted(
                {
                    row[field]
                    for row in source_rows
                    if row.get(field) not in (None, "")
                },
                key=str,
            )

        machines.append(
            {
                "machine_id": machine_id,
                "offer_ids": observed_values("offer_id"),
                "gpu_models": observed_values("gpu_model"),
                "regions": observed_values("region"),
                "advertised_reliability_values": observed_values(
                    "advertised_reliability"
                ),
                "advertised_internet_down_mbps_values": observed_values(
                    "advertised_internet_down_mbps"
                ),
                "advertised_internet_up_mbps_values": observed_values(
                    "advertised_internet_up_mbps"
                ),
                "advertised_disk_bandwidth_mbps_values": observed_values(
                    "advertised_disk_bandwidth_mbps"
                ),
                "active_rental_rate_usd_per_hour_values": observed_values(
                    "active_rental_rate_usd_per_hour"
                ),
                "storage_rate_usd_per_gb_month_values": observed_values(
                    "storage_rate_usd_per_gb_month"
                ),
                "internet_ingress_rate_usd_per_gb_values": observed_values(
                    "internet_ingress_rate_usd_per_gb"
                ),
                "assigned_package_bytes_values": observed_values(
                    "assigned_package_bytes"
                ),
                "attempt_count": len(rows),
                "ready_success_count": len(successes),
                "ready_healthy_count": len(healthy),
                "attributable_failure_count": len(attributable),
                "controller_side_failure_count": len(controller),
                "public_mapping_observation_count": len(rows),
                "public_mapping_success_count": len(mapping_observed),
                "public_mapping_failure_count": len(mapping_failed),
                "post_ready_health_observation_count": len(successes),
                "disappearance_count": sum(bool(row["disappeared"]) for row in rows),
                "terminal_or_exited_count": sum(
                    bool(row["terminal_or_exited"]) for row in rows
                ),
                "ssl_or_connectivity_failure_count": sum(
                    bool(row["ssl_or_connectivity_failed"]) for row in rows
                ),
                "no_progress_timeout_count": sum(
                    bool(row["no_progress_timeout"]) for row in rows
                ),
                "median_time_to_first_substantive_progress_seconds": quantile(
                    (
                        float(row["time_to_first_substantive_progress_seconds"])
                        for row in rows
                        if row.get("time_to_first_substantive_progress_seconds")
                        is not None
                    ),
                    0.50,
                ),
                "median_time_to_model_download_completion_seconds": quantile(
                    (
                        float(row["time_to_model_download_completion_seconds"])
                        for row in rows
                        if row.get("time_to_model_download_completion_seconds")
                        is not None
                    ),
                    0.50,
                ),
                "median_time_to_gpu_load_seconds": quantile(
                    (
                        float(row["time_to_gpu_load_seconds"])
                        for row in rows
                        if row.get("time_to_gpu_load_seconds") is not None
                    ),
                    0.50,
                ),
                "median_time_to_ready_seconds": quantile(ready_seconds, 0.50),
                "p90_time_to_ready_seconds": quantile(ready_seconds, 0.90),
                "median_download_throughput_mbps": quantile(download_rates, 0.50),
                "previous_e025_success_observation_ids": [
                    row["observation_id"] for row in successes
                ],
                "previous_e025_failure_observation_ids": [
                    row["observation_id"]
                    for row in rows
                    if not row["reached_group_ready"]
                ],
                "prior_persistent_hard_exclusion": previously_excluded,
                "controller_only_prior_exclusion_removed": controller_only_prior_failure,
                "hard_excluded": hard_excluded,
                "classification": (
                    "HARD_EXCLUDE"
                    if hard_excluded
                    else "STRONG_POSITIVE"
                    if healthy and not attributable
                    else "DEPRIORITIZE"
                    if attributable
                    else "UNPROVEN"
                ),
            }
        )
    coverage_fields = (
        "region",
        "advertised_reliability",
        "advertised_internet_down_mbps",
        "storage_rate_usd_per_gb_month",
        "internet_ingress_rate_usd_per_gb",
        "assigned_package_bytes",
        "time_to_first_substantive_progress_seconds",
        "time_to_model_download_completion_seconds",
        "time_to_gpu_load_seconds",
        "time_to_ready_seconds",
        "download_throughput_mbps",
    )
    coverage = {
        field: {
            "non_null_count": sum(row.get(field) is not None for row in observations),
            "observation_count": len(observations),
        }
        for field in coverage_fields
    }
    return {
        "schema_version": "experiment-025-machine-reliability-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "evidence_class": "PHYSICAL_ACQUISITION_HISTORY",
        "run_id": "20260819T013016Z",
        "immutable_worker_image": WORKER_IMAGE_DIGEST,
        "source_scope": "all retained E025 rental ledgers, enriched by the canonical Attempt 005 event log and node inventory",
        "observation_count": len(observations),
        "machine_count": len(machines),
        "field_coverage": coverage,
        "classification_policy": {
            "positive": "authenticated full-group READY plus survival until its experiment cutoff",
            "negative": "provider disappearance/terminal state, missing mapping, repeated physical bootstrap stall, or create-ID loss",
            "controller_bug": "retained stage naming/evidence identifies a watchdog, observer, schema, image-pull, or fixed-parser defect",
            "hard_exclusion": "retain prior hard exclusion unless all evidence is controller/image-attributable; also exclude repeated attributable physical failures",
            "censoring": "deadline-aborted progressing instances are not classified as machine failures",
        },
        "prior_hard_exclusion_count": len(prior_exclusions),
        "controller_only_prior_exclusions_removed": removed_prior_exclusions,
        "hard_exclusion_machine_ids": [
            row["machine_id"] for row in machines if row["hard_excluded"]
        ],
        "machines": machines,
        "observations": observations,
    }


def _postmortem(
    run_root: Path,
    events: list[dict[str, Any]],
    terminal: dict[str, Any],
    history: dict[str, Any],
) -> dict[str, Any]:
    counts = collections.Counter(str(event["event_type"]) for event in events)
    run_started = next(event for event in events if event["event_type"] == "RUN_STARTED")
    origin = parse_time(str(run_started["timestamp_utc"]))
    if origin is None:
        raise AssertionError("Attempt 005 RUN_STARTED timestamp missing")
    ready_events = [event for event in events if event["event_type"] == "WORKER_READY"]
    ready_from_run = [
        (parse_time(str(event["timestamp_utc"])) - origin).total_seconds()
        for event in ready_events
        if parse_time(str(event["timestamp_utc"])) is not None
    ]
    created_by_instance = {
        int(event["instance_id"]): parse_time(str(event["timestamp_utc"]))
        for event in events
        if event["event_type"] == "INSTANCE_CREATED"
    }
    create_to_ready = [
        (
            parse_time(str(event["timestamp_utc"]))
            - created_by_instance[int(event["instance_id"])]
        ).total_seconds()
        for event in ready_events
        if int(event["instance_id"]) in created_by_instance
        and created_by_instance[int(event["instance_id"])] is not None
        and parse_time(str(event["timestamp_utc"])) is not None
    ]
    fragment_ready = [
        event
        for event in ready_events
        if event.get("role") == "SUB_LAYER_WORKER"
    ]
    last_fragment_ready = max(
        (str(event["timestamp_utc"]) for event in fragment_ready), default=None
    )
    parent_created = next(
        (
            event
            for event in events
            if event["event_type"] == "INSTANCE_CREATED"
            and "layer-089-parent" in str(event.get("instance_group_id", ""))
        ),
        None,
    )
    timeline = terminal["readiness"]["timeline"]
    drops = [
        {
            "from": int(left["ready_workers"]),
            "to": int(right["ready_workers"]),
            "timestamp_utc": right["timestamp_utc"],
        }
        for left, right in itertools.pairwise(timeline)
        if int(right["ready_workers"]) < int(left["ready_workers"])
        and float(right["elapsed_minutes"]) < 30.0
    ]
    assertions = {
        "status_incomplete": terminal["status"] == "INCOMPLETE",
        "event_sha256_exact": terminal["event_log_sha256"]
        == ATTEMPT_005_EVENT_SHA256,
        "ready_events_56": counts["WORKER_READY"] == 56,
        "peak_simultaneous_46": terminal["readiness"][
            "peak_live_ready_worker_count"
        ]
        == 46,
        "instances_created_65": counts["INSTANCE_CREATED"] == 65,
        "replacements_19": counts["INSTANCE_REPLACED"] == 19,
        "four_fragments_ready": len(fragment_ready) == 4,
        "parent_not_ready": terminal["layer89_parent_ready"] is False,
        "no_token_execution": sum(
            counts[name]
            for name in (
                "TOKEN_EXECUTION_STARTED",
                "TOKEN_EXECUTION_COMPLETED",
                "TOKEN_EMITTED",
            )
        )
        == 0,
        "zero_live_cleanup": terminal["cleanup"]["zero_live_e025_instances"]
        is True,
    }
    return {
        "schema_version": "experiment-025-attempt-006-acquisition-postmortem-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "evidence_class": "PHYSICAL_ACQUISITION_ONLY",
        "run_id": "20260819T013016Z",
        "attempt": 5,
        "authoritative_result": "INCOMPLETE",
        "e025_pass_earned": False,
        "inference_failure": False,
        "inference_reached": False,
        "source_assertions": assertions,
        "source_files": {
            "canonical_event_log": str(
                (run_root / "telemetry" / "physical-run-events.jsonl").resolve()
            ),
            "canonical_event_log_sha256": ATTEMPT_005_EVENT_SHA256,
            "terminal_result": str(
                (run_root / "final" / "attempt-005-terminal-result.json").resolve()
            ),
            "node_inventory": str(
                (run_root / "telemetry" / "attempt-005-node-inventory.json").resolve()
            ),
        },
        "verified_counts": dict(sorted(counts.items())),
        "readiness": {
            "total_ready_events": counts["WORKER_READY"],
            "peak_simultaneous_current_ready_roles": terminal["readiness"][
                "peak_live_ready_worker_count"
            ],
            "net_ready_events_not_retained_at_peak": counts["WORKER_READY"]
            - terminal["readiness"]["peak_live_ready_worker_count"],
            "pre_cleanup_readiness_drops": drops,
            "ready_elapsed_from_run_start_seconds": distribution(ready_from_run),
            "create_to_ready_seconds": distribution(create_to_ready),
        },
        "parent_serialization": {
            "last_fragment_ready_timestamp_utc": last_fragment_ready,
            "parent_created_timestamp_utc": (
                parent_created["timestamp_utc"] if parent_created else None
            ),
            "seconds_from_last_fragment_ready_to_parent_create": seconds_between(
                last_fragment_ready,
                str(parent_created["timestamp_utc"]) if parent_created else None,
            ),
            "parent_ready": False,
            "diagnosis": "the parent paid bootstrap began only after all four fragment workers were READY, leaving only the acquisition tail before cutoff",
        },
        "cost_estimate_usd": terminal["cost"],
        "root_causes": [
            {
                "priority": 1,
                "cause": "Layer 89 parent bootstrap serialized behind four fragment READY completions",
                "repair": "create parent from a versioned set of four public fragment mappings before fragment model readiness",
            },
            {
                "priority": 2,
                "cause": "one-shot group acquisition did not own READY liveness until fleet freeze",
                "repair": "long-lived group supervisors remove lost roles from current readiness and replace only the affected group",
            },
            {
                "priority": 3,
                "cause": "finite frozen alternates and price-led selection amplified provider tail failures",
                "repair": "bounded live refresh plus reliability/time/cost scoring",
            },
            {
                "priority": 4,
                "cause": "serial replacement exposed the 97-role fleet to extreme-value tail latency",
                "repair": "at most one evidence-triggered duplicate per group with a global hedge cap and immediate loser cleanup",
            },
        ],
        "machine_history_artifact": "attempt-006-machine-reliability.json",
        "machine_history_observation_count": history["observation_count"],
        "decision_implication": "Attempt 006 must not use the Attempt 005 acquisition controller unchanged.",
    }


@dataclass(frozen=True, slots=True)
class CandidateDatum:
    machine_id: int
    capacity: int
    role_class: str
    outcome: str
    duration_seconds: float
    active_rate_usd_per_hour: float
    storage_rate_usd_per_hour: float
    effective_rate_usd_per_hour: float
    ingress_cost_usd: float
    full_ingress_cost_usd: float
    prior_history_class: str
    attempt_006_selection_history_class: str


@dataclass(frozen=True, slots=True)
class GroupSpec:
    group_id: str
    role_class: str
    capacity: int


@dataclass(slots=True)
class GroupResult:
    success: bool
    ready_time: float
    cost_usd: float
    active_cost_usd: float
    storage_cost_usd: float
    ingress_cost_usd: float
    launches: int
    hedges: int
    last_candidate_start: float
    retained_rate_usd_per_hour: float
    retained_active_rate_usd_per_hour: float
    retained_storage_rate_usd_per_hour: float


class HedgeCalendar:
    def __init__(self, maximum_concurrent: int) -> None:
        self.maximum_concurrent = maximum_concurrent
        self.intervals: list[tuple[float, float]] = []

    def reserve(self, requested_start: float, duration_seconds: float) -> float:
        start = requested_start
        while True:
            overlapping = [
                (left, right)
                for left, right in self.intervals
                if left <= start < right
            ]
            if len(overlapping) < self.maximum_concurrent:
                self.intervals.append((start, start + duration_seconds))
                return start
            start = min(right for _left, right in overlapping)

    def cancel(self, start: float, duration_seconds: float) -> None:
        self.intervals.remove((start, start + duration_seconds))

    def peak(self) -> int:
        points: list[tuple[float, int]] = []
        for left, right in self.intervals:
            points.extend(((left, 1), (right, -1)))
        current = 0
        peak = 0
        for _timestamp, delta in sorted(points, key=lambda row: (row[0], row[1])):
            current += delta
            peak = max(peak, current)
        return peak


def _simulation_inputs(
    run_root: Path,
    observations: list[dict[str, Any]],
) -> tuple[list[CandidateDatum], list[GroupSpec], dict[str, Any]]:
    prior_by_machine: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in observations:
        if row["stage_directory"] != "stage-4-headline":
            prior_by_machine[int(row["machine_id"])].append(row)
    candidate_data: list[CandidateDatum] = []
    for row in observations:
        if row["stage_directory"] != "stage-4-headline":
            continue
        start = str(row.get("create_timestamp_utc") or "")
        if row["reached_group_ready"]:
            outcome = "SUCCESS"
            duration = row.get("time_to_ready_seconds")
        else:
            outcome = (
                "RIGHT_CENSORED"
                if row.get("right_censored_at_attempt_deadline")
                else "FAILURE"
            )
            duration = seconds_between(start, row.get("destroyed_timestamp_utc"))
        if duration is None or float(duration) <= 0:
            continue
        prior = prior_by_machine.get(int(row["machine_id"]), [])
        prior_success = any(item["remained_healthy_after_ready"] for item in prior)
        prior_failure = any(item["machine_attributable_failure"] for item in prior)
        prior_class = (
            "POSITIVE"
            if prior_success
            else "NEGATIVE"
            if prior_failure
            else "UNSEEN"
        )
        selection_class = (
            "POSITIVE"
            if row["reached_group_ready"] or prior_success
            else "NEGATIVE"
            if row["machine_attributable_failure"] or prior_failure
            else "UNSEEN"
        )
        active_rate = float(row.get("active_rental_rate_usd_per_hour") or 0.25)
        storage_rate = float(row.get("storage_rate_usd_per_gb_month") or 0.20)
        disk_gb = float(row.get("requested_disk_gb") or 60)
        storage_rate_per_hour = storage_rate * disk_gb / (30 * 24)
        effective_rate = active_rate + storage_rate_per_hour
        package_bytes = float(row.get("assigned_package_bytes") or 0)
        ingress_rate = float(row.get("internet_ingress_rate_usd_per_gb") or 0.006)
        completed_fraction = (
            float(row.get("worker_ready_count") or 0)
            / max(1.0, float(row.get("expected_worker_count") or 1))
            if outcome != "SUCCESS"
            else 1.0
        )
        candidate_data.append(
            CandidateDatum(
                machine_id=int(row["machine_id"]),
                capacity=int(row["expected_worker_count"]),
                role_class=str(row.get("role_class") or "BACKBONE"),
                outcome=outcome,
                duration_seconds=float(duration),
                active_rate_usd_per_hour=active_rate,
                storage_rate_usd_per_hour=storage_rate_per_hour,
                effective_rate_usd_per_hour=effective_rate,
                ingress_cost_usd=(
                    package_bytes / 1e9 * ingress_rate * completed_fraction
                ),
                full_ingress_cost_usd=package_bytes / 1e9 * ingress_rate,
                prior_history_class=prior_class,
                attempt_006_selection_history_class=selection_class,
            )
        )
    go = read_json(run_root / "preflight" / "FULL_FLEET_GO-attempt-005.json")
    groups = []
    for group in go["fleet_plan"]["instance_groups"]:
        worker_ids = [str(row["worker_id"]) for row in group["workers"]]
        role_class = (
            "FRAGMENT"
            if any("sub-" in worker_id for worker_id in worker_ids)
            else "PARENT"
            if any("parent" in worker_id for worker_id in worker_ids)
            else "BACKBONE"
        )
        groups.append(
            GroupSpec(
                group_id=str(group["instance_group_id"]),
                role_class=role_class,
                capacity=len(worker_ids),
            )
        )
    predictive = {
        "attempt_005_candidate_count": len(candidate_data),
        "prior_positive_attempt_005_candidates": sum(
            row.prior_history_class == "POSITIVE" for row in candidate_data
        ),
        "prior_positive_success_count": sum(
            row.prior_history_class == "POSITIVE" and row.outcome == "SUCCESS"
            for row in candidate_data
        ),
        "prior_negative_attempt_005_candidates": sum(
            row.prior_history_class == "NEGATIVE" for row in candidate_data
        ),
        "prior_negative_success_count": sum(
            row.prior_history_class == "NEGATIVE" and row.outcome == "SUCCESS"
            for row in candidate_data
        ),
        "prior_unseen_attempt_005_candidates": sum(
            row.prior_history_class == "UNSEEN" for row in candidate_data
        ),
        "prior_unseen_success_count": sum(
            row.prior_history_class == "UNSEEN" and row.outcome == "SUCCESS"
            for row in candidate_data
        ),
    }
    return candidate_data, groups, predictive


def _draw_candidate(
    rng: random.Random,
    data: list[CandidateDatum],
    group: GroupSpec,
    *,
    reliability_scoring: bool,
    censored_eventual_success_probability: float,
) -> CandidateDatum:
    role = "BACKBONE" if group.role_class == "PARENT" else group.role_class
    exact_pool = [
        row
        for row in data
        if row.capacity == group.capacity and row.role_class == role
    ]
    pool = list(exact_pool)
    stratum_weights = [1.0] * len(pool)
    if len(exact_pool) < 5 or not any(
        row.outcome == "SUCCESS" for row in exact_pool
    ):
        partial_pool = [
            row
            for row in data
            if row.role_class == role
            and row.capacity >= 2
            and row.capacity != group.capacity
        ]
        pool.extend(partial_pool)
        stratum_weights.extend([0.20] * len(partial_pool))
    if not pool:
        pool = [row for row in data if row.capacity == group.capacity]
        stratum_weights = [1.0] * len(pool)
    if not pool:
        pool = list(data)
        stratum_weights = [1.0] * len(pool)
    weights = []
    for row, stratum_weight in zip(pool, stratum_weights, strict=True):
        if not reliability_scoring:
            weights.append(stratum_weight)
        elif row.attempt_006_selection_history_class == "POSITIVE":
            weights.append(3.0 * stratum_weight)
        elif row.attempt_006_selection_history_class == "NEGATIVE":
            weights.append(0.25 * stratum_weight)
        else:
            weights.append(stratum_weight)
    selected = rng.choices(pool, weights=weights, k=1)[0]
    if selected.outcome != "RIGHT_CENSORED":
        return selected
    if rng.random() < censored_eventual_success_probability:
        tail = rng.triangular(60.0, 600.0, 240.0)
        outcome = "SUCCESS"
    else:
        tail = rng.triangular(120.0, 480.0, 240.0)
        outcome = "FAILURE"
    return CandidateDatum(
        machine_id=selected.machine_id,
        capacity=selected.capacity,
        role_class=selected.role_class,
        outcome=outcome,
        duration_seconds=selected.duration_seconds + tail,
        active_rate_usd_per_hour=selected.active_rate_usd_per_hour,
        storage_rate_usd_per_hour=selected.storage_rate_usd_per_hour,
        effective_rate_usd_per_hour=selected.effective_rate_usd_per_hour,
        ingress_cost_usd=(
            selected.full_ingress_cost_usd
            if outcome == "SUCCESS"
            else selected.ingress_cost_usd
        ),
        full_ingress_cost_usd=selected.full_ingress_cost_usd,
        prior_history_class=selected.prior_history_class,
        attempt_006_selection_history_class=(
            selected.attempt_006_selection_history_class
        ),
    )


def _candidate_cost(
    datum: CandidateDatum, active_seconds: float, *, completed: bool
) -> tuple[float, float, float]:
    fraction = 1.0 if completed else min(
        1.0, active_seconds / max(1.0, datum.duration_seconds)
    )
    return (
        datum.active_rate_usd_per_hour * max(0.0, active_seconds) / 3600.0,
        datum.storage_rate_usd_per_hour * max(0.0, active_seconds) / 3600.0,
        datum.ingress_cost_usd * fraction,
    )


def _acquire_group(
    rng: random.Random,
    data: list[CandidateDatum],
    group: GroupSpec,
    *,
    start_time: float,
    window_seconds: float,
    live_refresh: bool,
    reliability_scoring: bool,
    hedging: bool,
    censored_probability: float,
    hedge_calendar: HedgeCalendar,
) -> GroupResult:
    now = start_time
    active_cost = 0.0
    storage_cost = 0.0
    ingress_cost = 0.0

    def charge(datum: CandidateDatum, seconds: float, *, completed: bool) -> None:
        nonlocal active_cost, storage_cost, ingress_cost
        active, storage, ingress = _candidate_cost(
            datum, seconds, completed=completed
        )
        active_cost += active
        storage_cost += storage
        ingress_cost += ingress

    def total_cost() -> float:
        return active_cost + storage_cost + ingress_cost
    launches = 0
    hedges = 0
    last_start = start_time
    maximum_attempts = 12 if live_refresh else 4
    while now < window_seconds and launches < maximum_attempts:
        primary_start = now
        last_start = primary_start
        primary = _draw_candidate(
            rng,
            data,
            group,
            reliability_scoring=reliability_scoring,
            censored_eventual_success_probability=censored_probability,
        )
        launches += 1
        primary_finish = primary_start + primary.duration_seconds
        at_risk = primary.outcome != "SUCCESS" or primary.duration_seconds >= 20 * 60
        detected = at_risk and rng.random() < 0.70
        false_positive = not at_risk and rng.random() < 0.03
        hedge: CandidateDatum | None = None
        hedge_start: float | None = None
        hedge_finish: float | None = None
        if hedging and (detected or false_positive):
            hedge = _draw_candidate(
                rng,
                data,
                group,
                reliability_scoring=reliability_scoring,
                censored_eventual_success_probability=censored_probability,
            )
            requested_start = primary_start + 120.0
            hedge_start = hedge_calendar.reserve(
                requested_start, hedge.duration_seconds
            )
            if hedge_start < min(primary_finish, window_seconds):
                hedge_finish = hedge_start + hedge.duration_seconds
                launches += 1
                hedges += 1
            else:
                hedge_calendar.cancel(hedge_start, hedge.duration_seconds)
                hedge = None
                hedge_start = None
        successful_finishes = []
        if primary.outcome == "SUCCESS":
            successful_finishes.append((primary_finish, primary, primary_start))
        if hedge is not None and hedge.outcome == "SUCCESS" and hedge_finish is not None:
            successful_finishes.append((hedge_finish, hedge, float(hedge_start)))
        if successful_finishes:
            winner_finish, winner, _winner_start = min(
                successful_finishes, key=lambda row: row[0]
            )
            if winner_finish > window_seconds:
                # The candidate would eventually succeed, but not inside this
                # policy window. Charge only the paid interval to cutoff.
                cutoff_primary = max(0.0, window_seconds - primary_start)
                charge(
                    primary,
                    min(primary.duration_seconds, cutoff_primary),
                    completed=False,
                )
                if hedge is not None and hedge_start is not None:
                    cutoff_hedge = max(0.0, window_seconds - hedge_start)
                    charge(
                        hedge,
                        min(hedge.duration_seconds, cutoff_hedge),
                        completed=False,
                    )
                return GroupResult(
                    success=False,
                    ready_time=window_seconds,
                    cost_usd=total_cost(),
                    active_cost_usd=active_cost,
                    storage_cost_usd=storage_cost,
                    ingress_cost_usd=ingress_cost,
                    launches=launches,
                    hedges=hedges,
                    last_candidate_start=last_start,
                    retained_rate_usd_per_hour=0.0,
                    retained_active_rate_usd_per_hour=0.0,
                    retained_storage_rate_usd_per_hour=0.0,
                )
            primary_active = min(primary_finish, winner_finish) - primary_start
            charge(
                primary,
                primary_active,
                completed=primary_finish <= winner_finish,
            )
            if hedge is not None and hedge_start is not None and hedge_finish is not None:
                hedge_active = max(0.0, min(hedge_finish, winner_finish) - hedge_start)
                charge(
                    hedge,
                    hedge_active,
                    completed=hedge_finish <= winner_finish,
                )
            return GroupResult(
                success=True,
                ready_time=winner_finish,
                cost_usd=total_cost(),
                active_cost_usd=active_cost,
                storage_cost_usd=storage_cost,
                ingress_cost_usd=ingress_cost,
                launches=launches,
                hedges=hedges,
                last_candidate_start=last_start,
                retained_rate_usd_per_hour=winner.effective_rate_usd_per_hour,
                retained_active_rate_usd_per_hour=winner.active_rate_usd_per_hour,
                retained_storage_rate_usd_per_hour=winner.storage_rate_usd_per_hour,
            )
        charge(
            primary,
            primary.duration_seconds,
            completed=primary.outcome == "SUCCESS",
        )
        finishes = [primary_finish]
        if hedge is not None and hedge_start is not None and hedge_finish is not None:
            charge(
                hedge,
                hedge.duration_seconds,
                completed=hedge.outcome == "SUCCESS",
            )
            finishes.append(hedge_finish)
        now = min(max(finishes), window_seconds)
    return GroupResult(
        success=False,
        ready_time=window_seconds,
        cost_usd=total_cost(),
        active_cost_usd=active_cost,
        storage_cost_usd=storage_cost,
        ingress_cost_usd=ingress_cost,
        launches=launches,
        hedges=hedges,
        last_candidate_start=last_start,
        retained_rate_usd_per_hour=0.0,
        retained_active_rate_usd_per_hour=0.0,
        retained_storage_rate_usd_per_hour=0.0,
    )


def _one_simulation(
    rng: random.Random,
    data: list[CandidateDatum],
    groups: list[GroupSpec],
    *,
    window_seconds: float,
    liveness_recovery: bool,
    early_parent: bool,
    live_refresh: bool,
    reliability_scoring: bool,
    hedging: bool,
    censored_probability: float,
    churn_probability_per_ten_minutes: float,
) -> dict[str, Any]:
    calendar = HedgeCalendar(3)
    parent = next(group for group in groups if group.role_class == "PARENT")
    non_parent = [group for group in groups if group.role_class != "PARENT"]
    rng.shuffle(non_parent)
    results = {
        group.group_id: _acquire_group(
            rng,
            data,
            group,
            start_time=0.0,
            window_seconds=window_seconds,
            live_refresh=live_refresh,
            reliability_scoring=reliability_scoring,
            hedging=hedging,
            censored_probability=censored_probability,
            hedge_calendar=calendar,
        )
        for group in non_parent
    }
    fragments = [group for group in non_parent if group.role_class == "FRAGMENT"]
    fragment_results = [results[group.group_id] for group in fragments]
    parent_extra_active_cost = 0.0
    parent_extra_storage_cost = 0.0
    parent_extra_launches = 0
    if early_parent:
        stable_mapping_time = max(
            (result.last_candidate_start + 45.0 for result in fragment_results),
            default=45.0,
        )
        if stable_mapping_time > 45.0:
            parent_active_rate = statistics.median(
                row.active_rate_usd_per_hour
                for row in data
                if row.capacity == 1 and row.role_class == "BACKBONE"
            )
            parent_storage_rate = statistics.median(
                row.storage_rate_usd_per_hour
                for row in data
                if row.capacity == 1 and row.role_class == "BACKBONE"
            )
            parent_extra_active_cost = (
                parent_active_rate * (stable_mapping_time - 45.0) / 3600.0
            )
            parent_extra_storage_cost = (
                parent_storage_rate * (stable_mapping_time - 45.0) / 3600.0
            )
            parent_extra_launches = 1
        parent_start = stable_mapping_time
    else:
        parent_start = max(
            (result.ready_time for result in fragment_results), default=window_seconds
        )
    parent_result = _acquire_group(
        rng,
        data,
        parent,
        start_time=parent_start,
        window_seconds=window_seconds,
        live_refresh=live_refresh,
        reliability_scoring=reliability_scoring,
        hedging=hedging,
        censored_probability=censored_probability,
        hedge_calendar=calendar,
    )
    parent_result.active_cost_usd += parent_extra_active_cost
    parent_result.storage_cost_usd += parent_extra_storage_cost
    parent_result.cost_usd += parent_extra_active_cost + parent_extra_storage_cost
    parent_result.launches += parent_extra_launches
    results[parent.group_id] = parent_result

    all_initial_success = all(result.success for result in results.values())
    if all_initial_success:
        provisional_ready = max(result.ready_time for result in results.values())
        for _round in range(3):
            losses: list[tuple[GroupSpec, float]] = []
            for group in groups:
                result = results[group.group_id]
                exposure = max(0.0, provisional_ready - result.ready_time)
                loss_probability = 1.0 - math.exp(
                    -churn_probability_per_ten_minutes * exposure / 600.0
                )
                if rng.random() < loss_probability:
                    loss_time = result.ready_time + rng.random() * exposure
                    losses.append((group, loss_time))
            if not losses:
                break
            if not liveness_recovery:
                all_initial_success = False
                break
            fragment_recovered = False
            for group, loss_time in losses:
                previous = results[group.group_id]
                retention_seconds = max(0.0, loss_time - previous.ready_time)
                previous.active_cost_usd += (
                    previous.retained_active_rate_usd_per_hour
                    * retention_seconds
                    / 3600.0
                )
                previous.storage_cost_usd += (
                    previous.retained_storage_rate_usd_per_hour
                    * retention_seconds
                    / 3600.0
                )
                previous.cost_usd = (
                    previous.active_cost_usd
                    + previous.storage_cost_usd
                    + previous.ingress_cost_usd
                )
                replacement = _acquire_group(
                    rng,
                    data,
                    group,
                    start_time=loss_time,
                    window_seconds=window_seconds,
                    live_refresh=live_refresh,
                    reliability_scoring=reliability_scoring,
                    hedging=hedging,
                    censored_probability=censored_probability,
                    hedge_calendar=calendar,
                )
                replacement.cost_usd += previous.cost_usd
                replacement.active_cost_usd += previous.active_cost_usd
                replacement.storage_cost_usd += previous.storage_cost_usd
                replacement.ingress_cost_usd += previous.ingress_cost_usd
                replacement.launches += previous.launches
                replacement.hedges += previous.hedges
                results[group.group_id] = replacement
                fragment_recovered = fragment_recovered or group.role_class == "FRAGMENT"
            if fragment_recovered and all(
                results[group.group_id].success for group in fragments
            ):
                parent_loss_time = max(
                    results[group.group_id].last_candidate_start + 45.0
                    for group in fragments
                )
                previous_parent = results[parent.group_id]
                replacement_parent = _acquire_group(
                    rng,
                    data,
                    parent,
                    start_time=parent_loss_time,
                    window_seconds=window_seconds,
                    live_refresh=live_refresh,
                    reliability_scoring=reliability_scoring,
                    hedging=hedging,
                    censored_probability=censored_probability,
                    hedge_calendar=calendar,
                )
                replacement_parent.cost_usd += previous_parent.cost_usd
                replacement_parent.active_cost_usd += previous_parent.active_cost_usd
                replacement_parent.storage_cost_usd += (
                    previous_parent.storage_cost_usd
                )
                replacement_parent.ingress_cost_usd += (
                    previous_parent.ingress_cost_usd
                )
                replacement_parent.launches += previous_parent.launches
                replacement_parent.hedges += previous_parent.hedges
                results[parent.group_id] = replacement_parent
            if not all(result.success for result in results.values()):
                all_initial_success = False
                break
            provisional_ready = max(result.ready_time for result in results.values())

    fleet_ready_time = (
        max(result.ready_time for result in results.values()) + 30.0
        if all_initial_success
        else window_seconds
    )
    success = all_initial_success and fleet_ready_time <= window_seconds
    active_cost = sum(result.active_cost_usd for result in results.values())
    storage_cost = sum(result.storage_cost_usd for result in results.values())
    ingress_cost = sum(result.ingress_cost_usd for result in results.values())
    if success:
        active_cost += sum(
            result.retained_active_rate_usd_per_hour
            * max(0.0, fleet_ready_time - result.ready_time)
            / 3600.0
            for result in results.values()
        )
        storage_cost += sum(
            result.retained_storage_rate_usd_per_hour
            * max(0.0, fleet_ready_time - result.ready_time)
            / 3600.0
            for result in results.values()
        )
    total_cost = active_cost + storage_cost + ingress_cost
    ready_roles = sum(
        group.capacity
        for group in groups
        if results[group.group_id].success
        and results[group.group_id].ready_time <= window_seconds
    )
    launches = sum(result.launches for result in results.values())
    return {
        "success": success,
        "acquisition_seconds": fleet_ready_time,
        "cost_usd": total_cost,
        "active_rental_cost_usd": active_cost,
        "storage_cost_usd": storage_cost,
        "ingress_cost_usd": ingress_cost,
        "replacements": max(0, launches - len(groups)),
        "hedges": sum(result.hedges for result in results.values()),
        "peak_paid_instances": (46 if early_parent else 45) + calendar.peak(),
        "peak_ready_roles": ready_roles,
    }


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.96
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _summarize_simulations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["success"]]
    successes = len(successful)
    lower, upper = _wilson_interval(successes, len(rows))
    probability = successes / len(rows)
    return {
        "simulation_count": len(rows),
        "simulated_full_simultaneous_readiness_count": successes,
        "estimated_probability_full_simultaneous_readiness": probability,
        "monte_carlo_95_percent_interval": [lower, upper],
        "likelihood_band": (
            "VERY_LOW"
            if probability < 0.10
            else "LOW"
            if probability < 0.35
            else "UNCERTAIN"
            if probability < 0.60
            else "PLAUSIBLE"
            if probability < 0.80
            else "HIGH"
        ),
        "acquisition_time_seconds_conditional_on_success": distribution(
            row["acquisition_seconds"] for row in successful
        ),
        "active_storage_ingress_cost_usd": distribution(
            row["cost_usd"] for row in rows
        ),
        "active_rental_cost_usd": distribution(
            row["active_rental_cost_usd"] for row in rows
        ),
        "storage_cost_usd": distribution(
            row["storage_cost_usd"] for row in rows
        ),
        "ingress_cost_usd": distribution(
            row["ingress_cost_usd"] for row in rows
        ),
        "ingress_plus_storage_cost_usd": distribution(
            row["ingress_cost_usd"] + row["storage_cost_usd"] for row in rows
        ),
        "replacement_count": distribution(row["replacements"] for row in rows),
        "hedge_count": distribution(row["hedges"] for row in rows),
        "peak_paid_instance_count": distribution(
            row["peak_paid_instances"] for row in rows
        ),
        "peak_ready_role_count": distribution(
            row["peak_ready_roles"] for row in rows
        ),
    }


def _simulate_policies(
    run_root: Path,
    observations: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    data, groups, predictive = _simulation_inputs(run_root, observations)
    attempt_005_terminal = read_json(
        run_root / "final" / "attempt-005-terminal-result.json"
    )
    arms = [
        {
            "policy_id": "attempt-005-current",
            "liveness_recovery": False,
            "early_parent": False,
            "live_refresh": False,
            "reliability_scoring": False,
            "hedging": False,
        },
        {
            "policy_id": "continuous-ready-liveness",
            "liveness_recovery": True,
            "early_parent": False,
            "live_refresh": False,
            "reliability_scoring": False,
            "hedging": False,
        },
        {
            "policy_id": "early-parallel-parent",
            "liveness_recovery": False,
            "early_parent": True,
            "live_refresh": False,
            "reliability_scoring": False,
            "hedging": False,
        },
        {
            "policy_id": "live-alternate-refresh",
            "liveness_recovery": False,
            "early_parent": False,
            "live_refresh": True,
            "reliability_scoring": False,
            "hedging": False,
        },
        {
            "policy_id": "reliability-aware-scoring",
            "liveness_recovery": False,
            "early_parent": False,
            "live_refresh": False,
            "reliability_scoring": True,
            "hedging": False,
        },
        {
            "policy_id": "bounded-hedging",
            "liveness_recovery": False,
            "early_parent": False,
            "live_refresh": False,
            "reliability_scoring": False,
            "hedging": True,
        },
        {
            "policy_id": "combined-without-hedging",
            "liveness_recovery": True,
            "early_parent": True,
            "live_refresh": True,
            "reliability_scoring": True,
            "hedging": False,
        },
        {
            "policy_id": "combined-bounded-hedging",
            "liveness_recovery": True,
            "early_parent": True,
            "live_refresh": True,
            "reliability_scoring": True,
            "hedging": True,
        },
    ]
    scenarios = {
        "PESSIMISTIC": {
            "censored_probability": 0.25,
            "churn_probability": 0.03,
        },
        "BASE": {
            "censored_probability": 0.50,
            "churn_probability": 0.015,
        },
        "OPTIMISTIC": {
            "censored_probability": 0.70,
            "churn_probability": 0.005,
        },
    }
    results: list[dict[str, Any]] = []
    for window_seconds in (1800.0, 2100.0, 2400.0):
        for arm_index, arm in enumerate(arms):
            scenario_results: dict[str, Any] = {}
            for scenario_index, (scenario_name, scenario) in enumerate(
                scenarios.items()
            ):
                rng = random.Random(
                    seed
                    + int(window_seconds) * 10_000
                    + arm_index * 1_000
                    + scenario_index * 100_000
                )
                rows = [
                    _one_simulation(
                        rng,
                        data,
                        groups,
                        window_seconds=window_seconds,
                        liveness_recovery=bool(arm["liveness_recovery"]),
                        early_parent=bool(arm["early_parent"]),
                        live_refresh=bool(arm["live_refresh"]),
                        reliability_scoring=bool(arm["reliability_scoring"]),
                        hedging=bool(arm["hedging"]),
                        censored_probability=float(
                            scenario["censored_probability"]
                        ),
                        churn_probability_per_ten_minutes=float(
                            scenario["churn_probability"]
                        ),
                    )
                    for _ in range(iterations)
                ]
                scenario_results[scenario_name] = _summarize_simulations(rows)
            base = scenario_results["BASE"]
            pessimistic = scenario_results["PESSIMISTIC"]
            cost_p90 = base["active_storage_ingress_cost_usd"]["p90"]
            credible = bool(
                base["estimated_probability_full_simultaneous_readiness"] >= 0.60
                and pessimistic[
                    "estimated_probability_full_simultaneous_readiness"
                ]
                >= 0.25
                and cost_p90 is not None
                and float(cost_p90) <= 45.0
            )
            results.append(
                {
                    **arm,
                    "acquisition_window_seconds": int(window_seconds),
                    "scenario_results": scenario_results,
                    "credible_path": credible,
                }
            )
    credible_rows = [row for row in results if row["credible_path"]]
    selected = min(
        credible_rows,
        key=lambda row: (
            int(row["acquisition_window_seconds"]),
            float(
                row["scenario_results"]["BASE"][
                    "active_storage_ingress_cost_usd"
                ]["median"]
            ),
        ),
        default=None,
    )
    current_validation = next(
        row
        for row in results
        if row["policy_id"] == "attempt-005-current"
        and row["acquisition_window_seconds"] == 1800
    )
    observed_peak = 46
    modeled_peak = current_validation["scenario_results"]["BASE"][
        "peak_ready_role_count"
    ]["median"]
    modeled_cost = current_validation["scenario_results"]["BASE"][
        "active_storage_ingress_cost_usd"
    ]["median"]
    modeled_active_cost = current_validation["scenario_results"]["BASE"][
        "active_rental_cost_usd"
    ]["median"]
    modeled_storage_cost = current_validation["scenario_results"]["BASE"][
        "storage_cost_usd"
    ]["median"]
    modeled_ingress_cost = current_validation["scenario_results"]["BASE"][
        "ingress_cost_usd"
    ]["median"]
    modeled_replacements = current_validation["scenario_results"]["BASE"][
        "replacement_count"
    ]["median"]
    observed_cost = attempt_005_terminal["cost"]
    observed_total_cost = float(
        observed_cost[
            "active_storage_plus_completed_download_lower_bound_usd"
        ]
    )
    observed_active_cost = float(observed_cost["estimated_active_rental_usd"])
    observed_storage_cost = float(observed_cost["estimated_storage_usd"])
    observed_ingress_cost = float(
        observed_cost["completed_download_ingress_lower_bound_usd"]
    )
    observed_replacements = 19
    return {
        "schema_version": "experiment-025-attempt-006-acquisition-simulation-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if selected is not None else "NO_GO",
        "evidence_class": "PHYSICALLY_GROUNDED_MODEL",
        "run_id": "20260819T013016Z",
        "seed": seed,
        "iterations_per_policy_window_scenario": iterations,
        "physical_input_summary": {
            "attempt_005_candidate_instance_count": len(data),
            "required_instance_group_count": len(groups),
            "required_physical_role_count": sum(group.capacity for group in groups),
            "candidate_outcomes": dict(
                collections.Counter(row.outcome for row in data)
            ),
            "candidate_success_duration_seconds": distribution(
                row.duration_seconds for row in data if row.outcome == "SUCCESS"
            ),
            "candidate_failure_duration_seconds": distribution(
                row.duration_seconds for row in data if row.outcome == "FAILURE"
            ),
            "history_predictiveness_on_attempt_005": predictive,
        },
        "assumptions": {
            "candidate_sampling": "nonparametric bootstrap stratified by instance capacity and fragment/backbone role",
            "reliability_scoring_reweighting": "3.0x for candidates whose machine reached healthy READY in retained E025 history (including Attempt 005), 0.25x for candidates with attributable physical failure and no positive evidence, 1.0x otherwise; the pre-Attempt-005 subset is reported separately as a historical predictiveness check",
            "right_censoring": scenarios,
            "parent_mapping_delay_seconds": 45,
            "stability_barrier_seconds": 30,
            "hedge_warning_seconds": 120,
            "hedge_detection_true_positive_probability": 0.70,
            "hedge_false_positive_probability": 0.03,
            "maximum_concurrent_hedges": 3,
            "finite_frozen_candidate_limit": 4,
            "live_refresh_candidate_limit": 12,
            "full_group_post_ready_churn": "conservative scenario assumption; Attempt 005 retained zero observed complete-group losses but did lose partial READY roles",
            "cost": "observed per-instance active/storage rates plus assigned bytes times advertised ingress; partial transfers prorated by observed lifetime",
        },
        "validation": {
            "policy": "Attempt 005 policy at 1800 seconds",
            "observed_peak_ready_roles": observed_peak,
            "modeled_median_peak_ready_roles": modeled_peak,
            "absolute_peak_ready_role_error": (
                abs(float(modeled_peak) - observed_peak)
                if modeled_peak is not None
                else None
            ),
            "observed_cost_estimate_lower_bound_usd": {
                "active_rental": observed_active_cost,
                "storage": observed_storage_cost,
                "completed_download_ingress": observed_ingress_cost,
                "total": observed_total_cost,
            },
            "modeled_median_cost_usd": {
                "active_rental": modeled_active_cost,
                "storage": modeled_storage_cost,
                "ingress": modeled_ingress_cost,
                "total": modeled_cost,
            },
            "absolute_total_cost_error_usd": (
                abs(float(modeled_cost) - observed_total_cost)
                if modeled_cost is not None
                else None
            ),
            "observed_replacement_count": observed_replacements,
            "modeled_median_replacement_count": modeled_replacements,
            "absolute_replacement_count_error": (
                abs(float(modeled_replacements) - observed_replacements)
                if modeled_replacements is not None
                else None
            ),
            "observed_attempt_status": "INCOMPLETE",
            "modeled_likelihood_band": current_validation["scenario_results"][
                "BASE"
            ]["likelihood_band"],
            "validation_note": "No correction factor or outcome normalization was applied.",
        },
        "credibility_rule_declared_before_live_preflight": {
            "base_probability_minimum": 0.60,
            "pessimistic_probability_minimum": 0.25,
            "base_p90_acquisition_cost_usd_maximum": 45.0,
            "selection": "shortest acquisition window, then lowest base median cost",
        },
        "policy_comparisons": results,
        "selected_policy": (
            {
                "policy_id": selected["policy_id"],
                "acquisition_window_seconds": selected[
                    "acquisition_window_seconds"
                ],
                "base": selected["scenario_results"]["BASE"],
                "pessimistic": selected["scenario_results"]["PESSIMISTIC"],
                "optimistic": selected["scenario_results"]["OPTIMISTIC"],
            }
            if selected is not None
            else None
        ),
        "decision": (
            "PROCEED_TO_FRESH_READ_ONLY_PREFLIGHT"
            if selected is not None
            else "NO_GO_NO_PAID_INSTANCES"
        ),
        "limitations": [
            "The fleet-level result is a projection, not a physical swarm execution.",
            "Only one 97-role acquisition trace exists under the final Attempt 005 image/controller lineage.",
            "Right-censored hosts require explicit sensitivity assumptions; probabilities are likelihood estimates, not guarantees.",
            "Marketplace capacity and account budget are intentionally excluded until the fresh read-only GO gate.",
        ],
    }


def _source_boundary(run_root: Path) -> dict[str, Any]:
    freeze = read_json(run_root / "preflight" / "code-freeze.json")
    frozen = {str(row["path"]): str(row["sha256"]) for row in freeze["files"]}
    worker_paths = [
        "src/swarm_inference/experiments/experiment_025/supervisor.py",
        "src/swarm_inference/experiments/experiment_025/bootstrap.py",
        "src/swarm_inference/experiments/experiment_025/worker.py",
        "src/swarm_inference/experiments/experiment_025/expert_partition.py",
        "src/swarm_inference/experiments/experiment_025/wire.py",
        "src/swarm_inference/experiments/experiment_025/constants.py",
        "src/swarm_inference/experiments/experiment_025/io.py",
        "src/swarm_inference/execution/kimi_k3_stage.py",
        "src/swarm_inference/execution/kimi_cuda_runtime.py",
        "src/swarm_inference/experiments/experiment_020/transport.py",
        "src/swarm_inference/model/partition.py",
        "src/swarm_inference/protocol/stage_worker.py",
    ]
    worker_files = []
    for relative in worker_paths:
        path = REPO_ROOT / relative
        current = sha256_file(path)
        worker_files.append(
            {
                "path": relative,
                "frozen_sha256": frozen.get(relative),
                "current_sha256": current,
                "unchanged": frozen.get(relative) == current,
            }
        )
    controller_paths = [
        "src/swarm_inference/experiments/experiment_025/acquisition.py",
        "src/swarm_inference/experiments/experiment_025/headline.py",
        "src/swarm_inference/experiments/experiment_025/provisioning.py",
        "src/swarm_inference/experiments/experiment_025/vast_lifecycle.py",
    ]
    controller_files = [
        {"path": relative, "sha256": sha256_file(REPO_ROOT / relative)}
        for relative in controller_paths
    ]
    return {
        "immutable_worker_image_digest": WORKER_IMAGE_DIGEST,
        "worker_entrypoint_import_closure": worker_files,
        "all_worker_executed_files_unchanged": all(
            row["unchanged"] for row in worker_files
        ),
        "controller_files": controller_files,
        "controller_source_sha256": canonical_sha256(controller_files),
        "boundary_statement": "The paid image starts experiment_025.supervisor -> bootstrap -> worker. Acquisition, headline, provisioning, and Vast planning execute only on the local controller; the worker import closure and immutable image are unchanged.",
    }


def _policy(
    simulation: dict[str, Any], history: dict[str, Any], source_boundary: dict[str, Any]
) -> dict[str, Any]:
    selected = simulation["selected_policy"]
    status = "PASS" if selected is not None else "NO_GO"
    window = int(selected["acquisition_window_seconds"]) if selected else 2400
    selected_id = str(selected["policy_id"]) if selected else None
    hedge_enabled = bool(selected_id and "hedging" in selected_id)
    success_durations = simulation["physical_input_summary"][
        "candidate_success_duration_seconds"
    ]
    cost_p90 = (
        float(selected["base"]["active_storage_ingress_cost_usd"]["p90"])
        if selected
        else 45.0
    )
    cost_cap = min(45.0, math.ceil(cost_p90 * 1.10 / 5.0) * 5.0)
    offer_score = {
        "formula_version": "e025-attempt-006-expected-healthy-ready-cost-v1",
        "objective": "minimize expected cost and fleet-tail time to one healthy READY instance group",
        "formula": "expected_direct_attempt_cost / P(healthy_READY) + expected_ready_minutes / P(healthy_READY) * fleet_delay_cost_per_minute + attributable_failures * repeated_failure_penalty - bounded_healthy_success_credit + hard_exclusion_penalty",
        "provider_reliability_prior_weight": 2.0,
        "public_mapping_prior_probability": 0.95,
        "public_mapping_prior_weight": 2.0,
        "post_ready_health_prior_probability": 0.95,
        "post_ready_health_prior_weight": 2.0,
        "probability_floor": 0.05,
        "global_median_ready_seconds": float(success_durations["median"]),
        "history_time_prior_weight": 2.0,
        "fleet_delay_cost_usd_per_minute": 0.02,
        "repeated_attributable_failure_penalty_usd": 2.0,
        "historical_success_credit_usd": 0.25,
        "maximum_historical_success_credit_usd": 0.75,
        "hard_exclusion_penalty_usd": 1_000_000.0,
        "tie_breakers": [
            "higher empirical healthy READY probability",
            "lower expected READY seconds",
            "lower offer ID for determinism",
        ],
    }
    return {
        "schema_version": "experiment-025-attempt-006-acquisition-policy-v1",
        "generated_at_utc": utc_now(),
        "status": status,
        "run_id": "20260819T013016Z",
        "attempt": 6,
        "selected_simulation_policy_id": selected_id,
        "worker_image": {
            "immutable_digest": WORKER_IMAGE_DIGEST,
            "changed": False,
            "stage_1_stage_2_canary_chain_remains_valid": source_boundary[
                "all_worker_executed_files_unchanged"
            ],
        },
        "controller_source_sha256": source_boundary["controller_source_sha256"],
        "state_machine": [
            "NO_INSTANCE",
            "CREATING",
            "BOOTSTRAPPING",
            "READY",
            "READY_HEALTHY",
            "REPLACING",
            "BOOTSTRAPPING",
            "READY_HEALTHY",
        ],
        "global_ready_definition": "all 97 required worker identities simultaneously occupy the current READY registry, every group passes provider/mapping/authenticated identity health, and parent endpoint generation equals the complete four-fragment generation",
        "acquisition_window_seconds": window,
        "inference_cleanup_reserve_seconds": 900,
        "hard_stage_ttl_seconds": window + 900,
        "dud_timeout_seconds": 240,
        "progress_log_probe_after_seconds": 90,
        "progress_log_probe_interval_seconds": 45,
        "maximum_create_concurrency": 16,
        "offer_score": offer_score,
        "alternate_refresh_policy": {
            "enabled": True,
            "controller_capability_required": True,
            "launch_authorized": status == "PASS",
            "maximum_refreshes_per_group": 10,
            "retry_seconds": 20,
            "exact_constraints_frozen": [
                "consumer GeForce GPU allowlists",
                "minimum VRAM",
                "required GPU slots",
                "required direct public ports",
                "disk capacity",
                "CUDA/driver contract",
                "four distinct sub-layer machines",
            ],
            "exclusions": [
                "active offers",
                "active machines",
                "attempt-failed offers and machines",
                "persistent evidence-backed hard exclusions",
            ],
            "audit_artifact": "dynamic-offer-refreshes.jsonl",
        },
        "parent_endpoint_generation_policy": {
            "initial_generation": 1,
            "dependency_boundary": "all four current fragment instances have stable public host/port mappings",
            "waits_for_fragment_worker_ready": False,
            "public_mapping_timeout_seconds": 180,
            "replacement": "retire changed fragment mapping, increment generation, abort/destroy stale parent, and immediately bootstrap a parent from the newly complete mapping generation",
            "final_consistency_required": True,
        },
        "ready_health_policy": {
            "provider_poll_seconds": 10,
            "authenticated_probe_seconds": 8,
            "provider_missing_poll_limit": 2,
            "health_failure_limit": 2,
            "probe_concurrency": 32,
            "checks": [
                "Vast instance exists",
                "provider state running/loading",
                "public mapping unchanged",
                "authenticated REGISTER succeeds",
                "worker/role/machine/instance/image/checkpoint/assignment/GPU UUID unchanged",
            ],
            "stability_barrier_seconds": 30,
            "minimum_stability_health_rounds": 3,
            "loss_action": "remove the group from current readiness and replace only that group; fragment loss also invalidates its parent generation",
        },
        "hedge_policy": {
            "enabled": hedge_enabled,
            "controller_capability_implemented_and_tested": True,
            "selection_note": (
                "enabled by the selected credible simulation arm"
                if hedge_enabled
                else "not launch-enabled because no credible simulation arm was selected"
            ),
            "warning_no_progress_seconds": 120,
            "minimum_elapsed_seconds": 120,
            "parent_minimum_elapsed_seconds": 120,
            "parent_ready_role_threshold": 70,
            "probability_threshold": 0.55,
            "maximum_concurrent_hedges": 3,
            "maximum_candidates_per_group": 2,
            "winner": "first candidate generation to pass full authenticated healthy READY",
            "loser": "abort and immediately destroy; never publish duplicate role ownership",
            "triggers": [
                "warning no-progress interval",
                "expected ETA plus barrier threatens deadline",
                "empirical healthy READY probability below threshold",
                "Layer 89 parent materially behind a mostly ready fleet",
            ],
        },
        "cost_guard": {
            "maximum_total_acquisition_cost_usd": cost_cap,
            "hedge_projected_cost_reserve_usd": 1.0,
            "authorization_basis": "accrued destroyed cost plus the larger of accrued/projected cost for every active or in-flight authorization",
            "simulation_base_p90_cost_usd": cost_p90,
            "fresh_provider_budget_safety_multiplier": 1.15,
            "inference_cleanup_reserve_is_separate": True,
        },
        "persistent_hard_exclusion_machine_ids": history[
            "hard_exclusion_machine_ids"
        ],
        "watchdog_cleanup_policy": {
            "independent_watchdog_required": True,
            "ledger_all_paid_creates_before_mutation": True,
            "destroy_dud_immediately": True,
            "destroy_hedge_loser_immediately": True,
            "destroy_stale_parent_immediately": True,
            "abort_cleanup_from_hash_chained_ledger": True,
            "terminal_requirement": "zero live E025 instances",
        },
        "offline_evidence": {
            "simulation_status": simulation["status"],
            "selected_policy": selected,
            "decision": simulation["decision"],
        },
        "launch_configuration_valid": status == "PASS",
        "unselected_no_go_defaults_authorized_for_rental": False,
        "live_go_gate": "PENDING_FRESH_READ_ONLY_PREFLIGHT"
        if status == "PASS"
        else "NO_GO",
    }


def generate_all(
    run_root: Path = DEFAULT_RUN_ROOT,
    *,
    iterations: int = 1000,
    seed: int = 25006,
) -> dict[str, Any]:
    preflight = run_root / "preflight"
    observations = _ledger_observations(run_root)
    events, terminal = _enrich_attempt_005(run_root, observations)
    history = _machine_history(run_root, observations)
    postmortem = _postmortem(run_root, events, terminal, history)
    simulation = _simulate_policies(
        run_root, observations, iterations=iterations, seed=seed
    )
    source_boundary = _source_boundary(run_root)
    policy = _policy(simulation, history, source_boundary)
    paths = {
        "history": preflight / "attempt-006-machine-reliability.json",
        "postmortem": preflight / "attempt-006-acquisition-postmortem.json",
        "simulation": preflight / "attempt-006-acquisition-simulation.json",
        "policy": preflight / "attempt-006-acquisition-policy.json",
        "source_boundary": preflight / "attempt-006-source-boundary.json",
    }
    for name, payload in (
        ("history", history),
        ("postmortem", postmortem),
        ("simulation", simulation),
        ("policy", policy),
        ("source_boundary", source_boundary),
    ):
        write_json(paths[name], payload)
    return {
        "status": policy["status"],
        "decision": simulation["decision"],
        "selected_policy": simulation["selected_policy"],
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=25006)
    args = parser.parse_args()
    result = generate_all(
        args.run_root.resolve(), iterations=args.iterations, seed=args.seed
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
