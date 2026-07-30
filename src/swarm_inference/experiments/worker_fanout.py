"""Complete Experiment 003 orchestration and evidence generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import shutil
import statistics
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from swarm_inference.config.worker_fanout import (
    IMMUTABLE_QWEN3_REVISION,
    FanoutExperimentConfig,
    load_fanout_experiment_config,
)
from swarm_inference.exceptions import IntegrityError
from swarm_inference.experiments.experiment_002 import (
    _environment_probe,
    _git_evidence,
)
from swarm_inference.experiments.fanout_acquisition import acquire_shard_directory
from swarm_inference.experiments.fanout_analysis import (
    adaptive_search_summary,
    coefficient_of_variation,
    config_fingerprint,
    count_is_runnable,
    count_is_stable,
    initial_sweep_order,
    next_adaptive_count,
    performance_optima,
    session_evidence_digest,
    valid_completed_counts,
)
from swarm_inference.experiments.fanout_economics import (
    minimum_lease_duration,
    productive_fraction,
    productive_tokens,
)
from swarm_inference.experiments.fanout_lifecycle import (
    validate_lifecycle_events,
    write_lifecycle_events,
)
from swarm_inference.experiments.fanout_reporting import (
    TABLE_FILES,
    render_report,
    write_all_tables,
    write_charts,
)
from swarm_inference.experiments.fanout_session import (
    FanoutSessionResult,
    run_fanout_session,
)
from swarm_inference.experiments.runner import write_artifact_manifest
from swarm_inference.model.manifest import load_manifest, verify_manifest_shards
from swarm_inference.model.reference import run_reference_suite_subprocess
from swarm_inference.model.shard_builder import (
    inspect_qwen3_model,
    model_inspection_payload,
    resolve_model,
    shard_model,
)

_LAYOUT_BUILDER_VERSION = "experiment-003-balanced-bytes-execution-kv-v1"


@dataclass(frozen=True, slots=True)
class Experiment003Run:
    run_directory: Path
    report_path: Path
    summary: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.summary.get("overall_status") == "PASS"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _maximum(values: list[float]) -> float | None:
    return max(values) if values else None


def _build_profiling_evidence(
    *,
    profile_requested: bool,
    maximum_stable: int,
    tables: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    available_counts = {
        int(row["worker_count"])
        for row in tables["worker_count_results.csv"]
        if row.get("worker_count") is not None
    }
    requested_counts = {1, 4, 14}
    if maximum_stable:
        requested_counts.add(maximum_stable)
    representative_counts = sorted(requested_counts & available_counts)
    profiles: list[dict[str, Any]] = []
    for count in representative_counts:
        warm = [
            row
            for row in tables["warm_inference.csv"]
            if int(row["worker_count"]) == count
            and row.get("phase") == "warm"
            and int(row.get("concurrency", 1)) == 1
        ]
        per_request_contributors: list[dict[str, float]] = []
        total_messages = 0
        total_request_seconds = 0.0
        for row in warm:
            per_stage = row.get("per_stage")
            stage_rows = per_stage if isinstance(per_stage, list) else []
            execution = sum(float(item.get("execution_ms", 0.0)) for item in stage_rows) / 1000
            queue = sum(float(item.get("queue_ms", 0.0)) for item in stage_rows) / 1000
            transfer = sum(float(item.get("transfer_ms", 0.0)) for item in stage_rows) / 1000
            serialisation = (
                sum(
                    float(item.get("serialisation_ms", 0.0))
                    + float(item.get("deserialisation_ms", 0.0))
                    + float(item.get("integrity_validation_ms", 0.0))
                    + float(item.get("stream_queue_ms", 0.0))
                    for item in stage_rows
                )
                / 1000
            )
            end_to_end = float(row.get("end_to_end_latency_seconds", 0.0))
            coordinator = max(0.0, end_to_end - execution - queue - transfer - serialisation)
            per_request_contributors.append(
                {
                    "stage_execution_seconds": execution,
                    "worker_queue_seconds": queue,
                    "direct_transfer_seconds": transfer,
                    "serialisation_and_validation_seconds": serialisation,
                    "coordinator_and_other_seconds": coordinator,
                }
            )
            transport_delta = row.get("transport_delta")
            if isinstance(transport_delta, dict):
                total_messages += int(transport_delta.get("data_messages_sent", 0))
                total_messages += int(transport_delta.get("data_messages_received", 0))
            total_request_seconds += end_to_end
        contributor_names = (
            "stage_execution_seconds",
            "worker_queue_seconds",
            "direct_transfer_seconds",
            "serialisation_and_validation_seconds",
            "coordinator_and_other_seconds",
        )
        contributors = [
            {
                "contributor": name,
                "median_seconds": _median([float(row[name]) for row in per_request_contributors]),
            }
            for name in contributor_names
        ]
        contributors.sort(
            key=lambda row: float(row["median_seconds"] or 0.0),
            reverse=True,
        )
        process_samples = [
            row
            for row in tables["worker_memory.csv"]
            if int(row["worker_count"]) == count
            and row.get("phase") == "warm"
            and row.get("monotonic_timestamp_ns") is not None
            and row.get("cpu_user_seconds") is not None
        ]
        by_process: dict[int, list[dict[str, Any]]] = {}
        for row in process_samples:
            by_process.setdefault(int(row["process_id"]), []).append(row)
        cpu_profiles: list[dict[str, Any]] = []
        for process_id, rows in sorted(by_process.items()):
            ordered = sorted(rows, key=lambda row: int(row["monotonic_timestamp_ns"]))
            first = ordered[0]
            last = ordered[-1]
            elapsed = (
                int(last["monotonic_timestamp_ns"]) - int(first["monotonic_timestamp_ns"])
            ) / 1_000_000_000
            cpu_delta = (
                float(last.get("cpu_user_seconds", 0.0))
                + float(last.get("cpu_system_seconds", 0.0))
                - float(first.get("cpu_user_seconds", 0.0))
                - float(first.get("cpu_system_seconds", 0.0))
            )
            context_delta = (
                int(last.get("voluntary_context_switches", 0) or 0)
                + int(last.get("involuntary_context_switches", 0) or 0)
                - int(first.get("voluntary_context_switches", 0) or 0)
                - int(first.get("involuntary_context_switches", 0) or 0)
            )
            cpu_profiles.append(
                {
                    "process_id": process_id,
                    "role": last.get("role"),
                    "cpu_percent_of_one_core": (cpu_delta / elapsed * 100 if elapsed > 0 else None),
                    "context_switches_per_second": (
                        context_delta / elapsed if elapsed > 0 else None
                    ),
                    "peak_thread_count": max(
                        (
                            int(row["thread_count"])
                            for row in ordered
                            if row.get("thread_count") is not None
                        ),
                        default=None,
                    ),
                    "peak_file_handle_count": max(
                        (
                            int(row["file_handle_count"])
                            for row in ordered
                            if row.get("file_handle_count") is not None
                        ),
                        default=None,
                    ),
                }
            )
        resources = [
            row
            for row in tables["resource_usage.csv"]
            if int(row["worker_count"]) == count and row.get("phase") == "warm"
        ]
        streams = [
            row
            for row in tables["stream_metrics.csv"]
            if int(row["worker_count"]) == count and row.get("phase") == "warm"
        ]
        profiles.append(
            {
                "worker_count": count,
                "measured_request_count": len(warm),
                "five_largest_end_to_end_latency_contributors": contributors[:5],
                "process_cpu_and_overhead": cpu_profiles,
                "median_gpu_utilisation_percent": _median(
                    [
                        float(row["gpu_utilisation_percent"])
                        for row in resources
                        if row.get("gpu_utilisation_percent") is not None
                    ]
                ),
                "peak_gpu_memory_bytes": _maximum(
                    [
                        float(row["gpu_memory_used_bytes"])
                        for row in resources
                        if row.get("gpu_memory_used_bytes") is not None
                    ]
                ),
                "peak_cuda_context_count": max(
                    (
                        int(row["cuda_context_count"])
                        for row in resources
                        if row.get("cuda_context_count") is not None
                    ),
                    default=None,
                ),
                "grpc_data_message_rate_per_second": (
                    total_messages / total_request_seconds if total_request_seconds > 0 else None
                ),
                "event_loop_lag_p95_ms": _median(
                    [
                        float(row["event_loop_lag_p95_ms"])
                        for row in streams
                        if row.get("event_loop_lag_p95_ms") is not None
                    ]
                ),
            }
        )
    return {
        "profile_requested": profile_requested,
        "capture_policy": (
            "Core profiling counters are captured for every measured warm session; "
            "--profile marks the run for representative-count analysis."
        ),
        "representative_worker_counts": representative_counts,
        "profiles": profiles,
    }


def _build_failure_diagnostics(
    *,
    session_rows: list[dict[str, Any]],
    lifecycle_events: list[dict[str, Any]],
    tables: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for session in session_rows:
        if session.get("passed"):
            continue
        worker_count = int(session["worker_count"])
        phase = str(session["phase"])
        repeat = int(session["repeat"])
        matching_events = [
            row
            for row in lifecycle_events
            if str(row.get("worker_id", "")).startswith(f"fanout-{worker_count:02d}-{phase}")
        ]
        last_by_worker: dict[str, dict[str, Any]] = {}
        for event in matching_events:
            worker_id = str(event.get("worker_id"))
            previous = last_by_worker.get(worker_id)
            if previous is None or int(event.get("monotonic_timestamp_ns", -1)) >= int(
                previous.get("monotonic_timestamp_ns", -1)
            ):
                last_by_worker[worker_id] = event
        resources = [
            row
            for row in tables["resource_usage.csv"]
            if int(row["worker_count"]) == worker_count
            and row.get("phase") == phase
            and int(row.get("repeat", -1)) == repeat
        ]
        worker_memory = [
            row
            for row in tables["worker_memory.csv"]
            if int(row["worker_count"]) == worker_count
            and row.get("phase") == phase
            and int(row.get("repeat", -1)) == repeat
        ]
        correctness = [
            row
            for row in tables["correctness.csv"]
            if int(row["worker_count"]) == worker_count
            and row.get("phase") == phase
            and int(row.get("repeat", -1)) == repeat
            and row.get("passed") is False
        ]
        latest_resource = (
            max(resources, key=lambda row: int(row.get("monotonic_timestamp_ns", -1)))
            if resources
            else None
        )
        failures.append(
            {
                "worker_count": worker_count,
                "phase": phase,
                "repeat": repeat,
                "failure_type": session.get("failure_type"),
                "failure_message": session.get("failure_message"),
                "oom": session.get("oom"),
                "request_timeout": session.get("request_timeout"),
                "worker_crash": session.get("worker_crash"),
                "cleanup": session.get("cleanup"),
                "last_lifecycle_event_by_worker": last_by_worker,
                "latest_resource_state": latest_resource,
                "peak_worker_memory_rows": worker_memory,
                "correctness_failures": correctness,
            }
        )
    return {
        "failure_count": len(failures),
        "failures": failures,
        "diagnostic_policy": (
            "Failures retain structured lifecycle tails, worker stdout/stderr paths, "
            "exit codes, resource state, memory evidence, and correctness mismatches."
        ),
    }


def _tensor_mapping_hash(manifest: Any) -> str:
    payload = json.dumps(
        manifest.tensor_to_stages,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _prepare_stage_layout(
    *,
    repository_root: Path,
    description: Any,
    worker_count: int,
) -> tuple[Path, Any, dict[str, Any]]:
    root = (
        repository_root / "artifacts" / "models" / "qwen3-0.6b" / f"stage-count-{worker_count}"
    ).resolve()
    manifest_path = root / "manifest.json"
    cache_path = root / "layout_cache.json"
    validation_started = time.perf_counter()
    if manifest_path.is_file() and cache_path.is_file():
        try:
            manifest = load_manifest(manifest_path)
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            compatible = (
                manifest.model_id == description.model_id
                and manifest.model_revision == description.model_revision
                and manifest.schema_version == str(cache.get("manifest_schema_version"))
                and len(manifest.stages) == worker_count
                and manifest.weight_dtype == cache.get("dtype")
                and _tensor_mapping_hash(manifest) == cache.get("tensor_mapping_hash")
                and int(cache.get("stage_count", -1)) == worker_count
                and manifest.shard_hashes == cache.get("stage_hashes")
                and cache.get("layout_builder_version") == _LAYOUT_BUILDER_VERSION
            )
            if not compatible:
                raise IntegrityError("cached stage layout metadata is incompatible")
            verify_manifest_shards(manifest, root)
            return (
                root,
                manifest,
                {
                    "worker_count": worker_count,
                    "stage_count": worker_count,
                    "layout_root": str(root),
                    "reused": True,
                    "manifest_schema_version": manifest.schema_version,
                    "tensor_mapping_hash": _tensor_mapping_hash(manifest),
                    "dtype": manifest.weight_dtype,
                    "model_revision": manifest.model_revision,
                    "layout_builder_version": _LAYOUT_BUILDER_VERSION,
                    "validation_seconds": time.perf_counter() - validation_started,
                    "one_time_shard_build_seconds": 0.0,
                    "stage_layer_ranges": [
                        [stage.layer_start, stage.layer_end] for stage in manifest.stages
                    ],
                    "stage_weight_bytes": [
                        stage.required_memory_bytes for stage in manifest.stages
                    ],
                    "stage_hashes": manifest.shard_hashes,
                },
            )
        except Exception as exc:
            invalid = root.with_name(
                root.name + "-invalid-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            )
            root.rename(invalid)
            invalid.joinpath("invalid_reason.txt").write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
    build_timings: dict[str, float] = {}
    maximum_stage_bytes = max(
        2 * sum(tensor.bytes for tensor in description.tensors),
        1,
    )
    manifest = shard_model(
        description,
        output=root,
        target_stage_bytes=math.ceil(maximum_stage_bytes / worker_count),
        maximum_stage_bytes=maximum_stage_bytes,
        stage_count=worker_count,
        build_timings=build_timings,
    )
    verify_manifest_shards(manifest, root)
    cache_payload = {
        "model_id": manifest.model_id,
        "model_revision": manifest.model_revision,
        "manifest_schema_version": manifest.schema_version,
        "stage_count": worker_count,
        "tensor_mapping_hash": _tensor_mapping_hash(manifest),
        "dtype": manifest.weight_dtype,
        "stage_hashes": manifest.shard_hashes,
        "layout_builder_version": _LAYOUT_BUILDER_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _write_json(cache_path, cache_payload)
    return (
        root,
        manifest,
        {
            "worker_count": worker_count,
            "stage_count": worker_count,
            "layout_root": str(root),
            "reused": False,
            "manifest_schema_version": manifest.schema_version,
            "tensor_mapping_hash": _tensor_mapping_hash(manifest),
            "dtype": manifest.weight_dtype,
            "model_revision": manifest.model_revision,
            "layout_builder_version": _LAYOUT_BUILDER_VERSION,
            "validation_seconds": time.perf_counter() - validation_started,
            **build_timings,
            "stage_layer_ranges": [
                [stage.layer_start, stage.layer_end] for stage in manifest.stages
            ],
            "stage_weight_bytes": [stage.required_memory_bytes for stage in manifest.stages],
            "stage_hashes": manifest.shard_hashes,
        },
    )


def _prompt_with_exact_tokens(
    tokenizer: Any,
    *,
    seed: str,
    token_count: int,
) -> tuple[str, list[int]]:
    text = seed
    token_ids: list[int] = []
    while len(token_ids) < token_count:
        text += " " + seed
        token_ids = [int(value) for value in tokenizer(text, return_tensors=None)["input_ids"]]
    token_ids = token_ids[:token_count]
    return tokenizer.decode(token_ids, skip_special_tokens=False), token_ids


class EvidenceAccumulator:
    def __init__(self, *, run_dir: Path, experiment_id: str, resume: bool) -> None:
        self.run_dir = run_dir
        self.experiment_id = experiment_id
        self.tables: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLE_FILES}
        self.session_rows: list[dict[str, Any]] = []
        self.count_rows: dict[int, dict[str, Any]] = {}
        self.lifecycle_events: list[dict[str, Any]] = []
        self.completed_counts: dict[str, dict[str, Any]] = {}
        state_path = run_dir / "experiment_state.json"
        if resume and state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(state.get("tables"), dict):
                for name in TABLE_FILES:
                    rows = state["tables"].get(name, [])
                    if isinstance(rows, list):
                        self.tables[name] = rows
            self.session_rows = list(state.get("session_rows", []))
            self.count_rows = {
                int(key): value for key, value in state.get("count_rows", {}).items()
            }
            self.completed_counts = dict(state.get("completed_counts", {}))
            lifecycle_path = run_dir / "lifecycle_events.jsonl"
            if lifecycle_path.is_file():
                for line in lifecycle_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        self.lifecycle_events.append(json.loads(line))

    def _base(self, row: dict[str, Any]) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id, **row}

    def quarantine_incomplete_counts(self, valid_counts: set[int]) -> set[int]:
        observed_counts = {
            int(row["worker_count"])
            for row in self.session_rows
            if row.get("worker_count") is not None
        } | set(self.count_rows)
        incomplete = observed_counts - valid_counts
        if not incomplete:
            return set()
        quarantined_tables = {
            name: [
                row
                for row in rows
                if row.get("worker_count") is not None and int(row["worker_count"]) in incomplete
            ]
            for name, rows in self.tables.items()
        }
        quarantined_sessions = [
            row for row in self.session_rows if int(row["worker_count"]) in incomplete
        ]
        prefixes = tuple(f"fanout-{count:02d}-" for count in sorted(incomplete))
        quarantined_lifecycle = [
            row
            for row in self.lifecycle_events
            if str(row.get("worker_id", "")).startswith(prefixes)
        ]
        _write_json(
            self.run_dir / "interrupted_evidence.json",
            {
                "quarantined_at": datetime.now(UTC).isoformat(),
                "incomplete_worker_counts": sorted(incomplete),
                "reason": (
                    "Incomplete points are preserved but excluded from resumed "
                    "measurements; they will be rerun from phase A."
                ),
                "session_rows": quarantined_sessions,
                "tables": quarantined_tables,
                "lifecycle_events": quarantined_lifecycle,
            },
        )
        self.session_rows = [
            row for row in self.session_rows if int(row["worker_count"]) not in incomplete
        ]
        for name, rows in self.tables.items():
            self.tables[name] = [
                row
                for row in rows
                if row.get("worker_count") is None or int(row["worker_count"]) not in incomplete
            ]
        self.lifecycle_events = [
            row
            for row in self.lifecycle_events
            if not str(row.get("worker_id", "")).startswith(prefixes)
        ]
        for count in incomplete:
            self.count_rows.pop(count, None)
            self.completed_counts.pop(str(count), None)
        return incomplete

    def quarantine_phase_attempts(self, phase: str) -> int:
        matching_sessions = [row for row in self.session_rows if str(row.get("phase")) == phase]
        matching_tables = {
            name: [row for row in rows if str(row.get("phase")) == phase]
            for name, rows in self.tables.items()
        }
        worker_marker = f"-{phase}-"
        matching_lifecycle = [
            row for row in self.lifecycle_events if worker_marker in str(row.get("worker_id", ""))
        ]
        if not matching_sessions and not any(matching_tables.values()):
            return 0
        attempt_path = self.run_dir / f"failed_{phase}_attempts.json"
        previous: list[dict[str, Any]] = []
        if attempt_path.is_file():
            payload = json.loads(attempt_path.read_text(encoding="utf-8"))
            if isinstance(payload.get("attempts"), list):
                previous = payload["attempts"]
        previous.append(
            {
                "quarantined_at": datetime.now(UTC).isoformat(),
                "reason": (
                    f"Prior {phase} evidence is preserved but excluded before a resume retry."
                ),
                "session_rows": matching_sessions,
                "tables": matching_tables,
                "lifecycle_events": matching_lifecycle,
            }
        )
        _write_json(attempt_path, {"phase": phase, "attempts": previous})
        self.session_rows = [row for row in self.session_rows if str(row.get("phase")) != phase]
        for name, rows in self.tables.items():
            self.tables[name] = [row for row in rows if str(row.get("phase")) != phase]
        self.lifecycle_events = [
            row
            for row in self.lifecycle_events
            if worker_marker not in str(row.get("worker_id", ""))
        ]
        return len(matching_sessions)

    def add_session(self, result: FanoutSessionResult) -> dict[str, Any]:
        base = {
            "worker_count": result.worker_count,
            "phase": result.phase,
            "repeat": result.repeat,
        }
        self.lifecycle_events.extend(result.lifecycle_events)
        self.tables["worker_lifecycle.csv"].extend(
            self._base(row) for row in result.worker_lifecycle_rows
        )
        self.tables["pipeline_readiness.csv"].append(
            self._base(
                {
                    **base,
                    "pipeline_ready_seconds": result.pipeline_ready_seconds,
                    "passed": result.passed,
                    "failure_type": result.failure_type,
                    "failure_message": result.failure_message,
                }
            )
        )
        self.tables["resource_usage.csv"].extend(self._base(row) for row in result.resource_rows)
        self.tables["worker_memory.csv"].extend(
            self._base(row) for row in result.worker_memory_rows
        )
        self.tables["gpu_process_memory.csv"].extend(
            self._base(row) for row in result.gpu_process_memory_rows
        )
        for health in result.health_rows:
            proof = health.get("proof", {})
            shards = proof.get("shards", {})
            current = shards.get("current_process_memory", {})
            for stage_payload in shards.get("stages", {}).values():
                load = stage_payload.get("load_record", {})
                module_state = stage_payload.get("module_state", {})
                cache_history = module_state.get("cache_history", [])
                peak_kv_cache_bytes = max(
                    (
                        int(cache.get("cache_bytes", 0))
                        for cache in cache_history
                        if isinstance(cache, dict)
                    ),
                    default=0,
                )
                self.tables["worker_memory.csv"].append(
                    self._base(
                        {
                            **base,
                            "sample_source": "worker-health-proof",
                            "worker_id": health.get("worker_id"),
                            "stage_id": load.get("stage_id"),
                            "process_id": load.get("process_id"),
                            "rss_bytes": current.get("host_rss_bytes"),
                            "torch_cuda_allocated_bytes": current.get("cuda_allocated_bytes"),
                            "torch_cuda_reserved_bytes": current.get("cuda_reserved_bytes"),
                            "torch_cuda_peak_allocated_bytes": current.get(
                                "cuda_peak_allocated_bytes"
                            ),
                            "torch_cuda_peak_reserved_bytes": current.get(
                                "cuda_peak_reserved_bytes"
                            ),
                            "stage_weight_bytes": load.get("total_loaded_weight_bytes"),
                            "kv_cache_bytes": module_state.get("cache_bytes"),
                            "peak_kv_cache_bytes": peak_kv_cache_bytes,
                        }
                    )
                )
                for boundary in stage_payload.get("module_state", {}).get("boundary_records", []):
                    self.tables["correctness.csv"].append(
                        self._base(
                            {
                                **base,
                                "evidence_type": "boundary",
                                **boundary,
                            }
                        )
                    )
        measured_worker_activation_bytes = 0
        for request in result.request_results:
            per_stage_rows = request.get("per_stage")
            if not isinstance(per_stage_rows, list):
                continue
            for stage_row in per_stage_rows:
                if not isinstance(stage_row, dict):
                    continue
                if int(stage_row.get("stage_id", -1)) < result.worker_count - 1:
                    measured_worker_activation_bytes += int(
                        stage_row.get("activation_bytes_sent", 0)
                    )
        self.tables["activation_traffic.csv"].append(
            self._base(
                {
                    **base,
                    "data_plane": result.transport_metrics.get("data_plane"),
                    "coordinator_activation_bytes": result.transport_metrics.get(
                        "coordinator_activation_bytes"
                    ),
                    "worker_to_worker_activation_bytes": measured_worker_activation_bytes,
                    "session_total_worker_to_worker_activation_bytes": (
                        result.transport_metrics.get("worker_to_worker_activation_bytes")
                    ),
                    "coordinator_input_activation_bytes": result.transport_metrics.get(
                        "coordinator_input_activation_bytes"
                    ),
                    "coordinator_final_result_bytes": result.transport_metrics.get(
                        "coordinator_final_result_bytes"
                    ),
                }
            )
        )
        self.tables["stream_metrics.csv"].append(
            self._base(
                {
                    **base,
                    "peer_streams_created": result.transport_metrics.get("peer_streams_created"),
                    "peer_channels_created": result.transport_metrics.get("peer_channels_created"),
                    "peer_active_pairs": result.transport_metrics.get("peer_active_pairs"),
                    "serialisation_time_ms": result.transport_metrics.get("serialisation_time_ms"),
                    "deserialisation_time_ms": result.transport_metrics.get(
                        "deserialisation_time_ms"
                    ),
                    "hop_transfer_time_ms": result.transport_metrics.get("hop_transfer_time_ms"),
                    "event_loop_lag_median_ms": result.transport_metrics.get(
                        "event_loop_lag_median_ms"
                    ),
                    "event_loop_lag_p95_ms": result.transport_metrics.get("event_loop_lag_p95_ms"),
                    "event_loop_lag_maximum_ms": result.transport_metrics.get(
                        "event_loop_lag_maximum_ms"
                    ),
                }
            )
        )
        for cache_row in result.transport_metrics.get("cache_validation", []):
            self.tables["correctness.csv"].append(
                self._base(
                    {
                        **base,
                        "evidence_type": "stage-local-kv-cache",
                        **cache_row,
                    }
                )
            )
        for request in result.request_results:
            destination = (
                "cold_inference.csv" if result.phase.startswith("cold") else "warm_inference.csv"
            )
            self.tables[destination].append(self._base(request))
            self.tables["correctness.csv"].append(
                self._base(
                    {
                        **base,
                        "evidence_type": "token-identity",
                        "request_id": request["request_id"],
                        "variant": request.get("variant"),
                        "concurrency": request.get("concurrency"),
                        "token_identity": request["token_identity"],
                        "first_mismatching_token": request.get("first_mismatching_token"),
                        "distributed_token_ids": request["distributed_token_ids"],
                        "reference_token_ids": request["reference_token_ids"],
                        "route_generation": request.get("route_generation"),
                        "passed": request["passed"],
                    }
                )
            )
        peak_gpu = max(
            (
                int(row["gpu_memory_used_bytes"])
                for row in result.resource_rows
                if row.get("gpu_memory_used_bytes") is not None
            ),
            default=None,
        )
        gpu_total = max(
            (
                int(row["gpu_memory_total_bytes"])
                for row in result.resource_rows
                if row.get("gpu_memory_total_bytes") is not None
            ),
            default=None,
        )
        peak_system_fraction = max(
            (
                float(row["system_memory_used_fraction"])
                for row in result.resource_rows
                if row.get("system_memory_used_fraction") is not None
            ),
            default=None,
        )
        peak_host = max(
            (
                int(row["aggregate_experiment_rss_bytes"])
                for row in result.resource_rows
                if row.get("aggregate_experiment_rss_bytes") is not None
            ),
            default=None,
        )
        exact = all(bool(request.get("token_identity")) for request in result.request_results)
        direct = (
            result.transport_metrics.get("data_plane") == "direct"
            and int(result.transport_metrics.get("coordinator_activation_bytes", -1)) == 0
            and (
                result.worker_count == 1
                or not result.request_results
                or int(result.transport_metrics.get("worker_to_worker_activation_bytes", 0)) > 0
            )
        )
        warm_single = [
            float(row["output_tokens_per_second"])
            for row in result.request_results
            if int(row.get("concurrency", 1)) == 1
            and row.get("output_tokens_per_second") is not None
            and result.phase == "warm"
        ]
        warm_concurrency = [
            float(row["aggregate_verified_tokens_per_second"])
            for row in result.request_results
            if int(row.get("concurrency", 1)) == 4
            and row.get("aggregate_verified_tokens_per_second") is not None
        ]
        warm_latency = [
            float(row["end_to_end_latency_seconds"])
            for row in result.request_results
            if int(row.get("concurrency", 1)) == 1
            and row.get("end_to_end_latency_seconds") is not None
            and result.phase == "warm"
        ]
        lifecycle_errors = (
            validate_lifecycle_events(
                result.lifecycle_events,
                require_complete_workers=True,
            )
            if result.passed
            else []
        )
        session_row = {
            **base,
            "passed": result.passed,
            "runnable_generation": result.runnable_generation,
            "exact_token_identity": exact,
            "direct_data_plane": direct,
            "clean_shutdown": bool(result.cleanup.get("clean_shutdown")),
            "worker_crash": bool(result.cleanup.get("unexpected_worker_crashes")),
            "oom": "out of memory" in (result.failure_message or "").lower(),
            "request_timeout": result.failure_type == "TimeoutError",
            "stale_cache": any(
                int(stage_payload.get("module_state", {}).get("cache_count", 0)) != 0
                for health in result.health_rows
                for stage_payload in health.get("proof", {})
                .get("shards", {})
                .get("stages", {})
                .values()
            )
            or result.transport_metrics.get("stage_local_kv_cache_valid") is not True,
            "peak_gpu_memory_bytes": peak_gpu,
            "peak_gpu_memory_fraction": (
                peak_gpu / gpu_total if peak_gpu is not None and gpu_total else None
            ),
            "peak_system_memory_fraction": peak_system_fraction,
            "peak_host_memory_bytes": peak_host,
            "warm_output_tokens_per_second": _median(warm_single),
            "concurrency_4_verified_tps": _median(warm_concurrency),
            "warm_end_to_end_seconds": _median(warm_latency),
            "failure_type": result.failure_type,
            "failure_message": result.failure_message,
            "lifecycle_validation_errors": lifecycle_errors,
            "cleanup": result.cleanup,
        }
        self.session_rows.append(session_row)
        return session_row

    def persist(self, *, state_extra: dict[str, Any] | None = None) -> None:
        write_all_tables(self.run_dir, self.tables)
        write_lifecycle_events(
            self.run_dir / "lifecycle_events.jsonl",
            self.lifecycle_events,
        )
        state = {
            "experiment_id": self.experiment_id,
            "tables": self.tables,
            "session_rows": self.session_rows,
            "count_rows": {str(key): value for key, value in self.count_rows.items()},
            "completed_counts": self.completed_counts,
            "updated_at": datetime.now(UTC).isoformat(),
            **(state_extra or {}),
        }
        _write_json(self.run_dir / "experiment_state.json", state)


def _count_result(
    *,
    count: int,
    accumulator: EvidenceAccumulator,
    config: FanoutExperimentConfig,
) -> dict[str, Any]:
    sessions = [row for row in accumulator.session_rows if int(row["worker_count"]) == count]
    main_sessions = [
        row
        for row in sessions
        if row["phase"]
        in {
            "cached_cold_load_only",
            "cold_no_stage_warmup",
            "cold_with_stage_warmup",
            "warm",
            "hot_standby",
        }
    ]
    warm_sessions = [row for row in main_sessions if row["phase"] == "warm"]
    runnable = count_is_runnable(main_sessions)
    stable = runnable and count_is_stable(
        main_sessions,
        repeats=config.sweep.repeats,
        max_gpu_memory_fraction=config.resource_limits.max_gpu_memory_fraction_for_stable,
        max_system_memory_fraction=config.resource_limits.max_system_memory_fraction_for_stable,
    )
    lifecycle = [
        row
        for row in accumulator.tables["worker_lifecycle.csv"]
        if int(row["worker_count"]) == count and row["phase"] == "cached_cold_load_only"
    ]
    pipeline = [
        row
        for row in accumulator.tables["pipeline_readiness.csv"]
        if int(row["worker_count"]) == count
        and row["phase"] == "cached_cold_load_only"
        and row.get("pipeline_ready_seconds") is not None
    ]
    worker_ready = [
        float(row["worker_ready_seconds"])
        for row in lifecycle
        if row.get("worker_ready_seconds") is not None
    ]
    cold = [
        row for row in accumulator.tables["cold_inference.csv"] if int(row["worker_count"]) == count
    ]
    warm = [
        row
        for row in accumulator.tables["warm_inference.csv"]
        if int(row["worker_count"]) == count and row["phase"] == "warm"
    ]
    warm_single = [row for row in warm if int(row.get("concurrency", 1)) == 1]
    warm_four = [row for row in warm if int(row.get("concurrency", 1)) == 4]
    failure = next(
        (
            row.get("failure_message") or row.get("failure_type")
            for row in main_sessions
            if not row.get("passed")
        ),
        None,
    )
    row = {
        "experiment_id": accumulator.experiment_id,
        "worker_count": count,
        "attempted": True,
        "runnable": runnable,
        "runnable_numeric": int(runnable),
        "stable": stable,
        "correctness_status": (
            "PASS"
            if runnable
            and all(
                bool(item.get("exact_token_identity"))
                for item in main_sessions
                if item.get("runnable_generation")
            )
            else "FAIL"
        ),
        "median_pipeline_ready_seconds": _median(
            [float(row["pipeline_ready_seconds"]) for row in pipeline]
        ),
        "median_worker_ready_seconds": _median(worker_ready),
        "maximum_worker_ready_seconds": _maximum(worker_ready),
        "median_cuda_initialisation_seconds": _median(
            [
                float(row["cuda_initialisation_seconds"])
                for row in lifecycle
                if row.get("cuda_initialisation_seconds") is not None
            ]
        ),
        "median_shard_read_seconds": _median(
            [
                float(row["shard_read_seconds"])
                for row in lifecycle
                if row.get("shard_read_seconds") is not None
            ]
        ),
        "median_weight_load_seconds": _median(
            [
                float(row["weight_load_seconds"])
                for row in lifecycle
                if row.get("weight_load_seconds") is not None
            ]
        ),
        "median_local_warmup_seconds": _median(
            [
                float(row["local_warmup_seconds"])
                for row in lifecycle
                if row.get("local_warmup_seconds") is not None
            ]
        ),
        "median_cold_ttft_no_stage_warmup_seconds": _median(
            [
                float(row["ttft_seconds"])
                for row in cold
                if row.get("variant") == "cold_no_stage_warmup"
                and row.get("ttft_seconds") is not None
            ]
        ),
        "median_cold_ttft_with_stage_warmup_seconds": _median(
            [
                float(row["ttft_seconds"])
                for row in cold
                if row.get("variant") == "cold_with_stage_warmup"
                and row.get("ttft_seconds") is not None
            ]
        ),
        "median_warm_ttft_seconds": _median(
            [
                float(row["ttft_seconds"])
                for row in warm_single
                if row.get("ttft_seconds") is not None
            ]
        ),
        "median_warm_end_to_end_seconds": _median(
            [
                float(row["end_to_end_latency_seconds"])
                for row in warm_single
                if row.get("end_to_end_latency_seconds") is not None
            ]
        ),
        "median_warm_output_tokens_per_second": _median(
            [
                float(row["output_tokens_per_second"])
                for row in warm_single
                if row.get("output_tokens_per_second") is not None
            ]
        ),
        "median_concurrency_4_verified_tps": _median(
            [
                float(row["aggregate_verified_tokens_per_second"])
                for row in warm_four
                if row.get("aggregate_verified_tokens_per_second") is not None
            ]
        ),
        "warm_throughput_coefficient_of_variation": coefficient_of_variation(
            [
                float(row["warm_output_tokens_per_second"])
                for row in warm_sessions
                if row.get("warm_output_tokens_per_second") is not None
            ]
        ),
        "peak_gpu_memory_bytes": _maximum(
            [
                float(row["peak_gpu_memory_bytes"])
                for row in warm_sessions
                if row.get("peak_gpu_memory_bytes") is not None
            ]
        ),
        "peak_gpu_memory_fraction": _maximum(
            [
                float(row["peak_gpu_memory_fraction"])
                for row in warm_sessions
                if row.get("peak_gpu_memory_fraction") is not None
            ]
        ),
        "peak_host_memory_bytes": _maximum(
            [
                float(row["peak_host_memory_bytes"])
                for row in warm_sessions
                if row.get("peak_host_memory_bytes") is not None
            ]
        ),
        "peak_system_memory_fraction": _maximum(
            [
                float(row["peak_system_memory_fraction"])
                for row in warm_sessions
                if row.get("peak_system_memory_fraction") is not None
            ]
        ),
        "failure_reason": failure,
    }
    accumulator.count_rows[count] = row
    accumulator.completed_counts[str(count)] = {
        "complete": True,
        "evidence_schema": "experiment-003-count-v1",
        "runnable": runnable,
        "stable": stable,
        "session_count": len(sessions),
        "session_evidence_sha256": session_evidence_digest(sessions),
    }
    accumulator.tables["worker_count_results.csv"] = [
        accumulator.count_rows[key] for key in sorted(accumulator.count_rows)
    ]
    return row


def run_worker_fanout_experiment(
    *,
    config_path: str | Path,
    model_id: str | None = None,
    revision: str | None = None,
    worker_counts: list[int] | None = None,
    repeats: int | None = None,
    max_worker_count: int | None = None,
    skip_acquisition_tests: bool = False,
    skip_rejoin_test: bool = False,
    resume: bool = False,
    output: str | Path = "artifacts/runs",
    profile: bool = False,
    smoke: bool = False,
    keep_workers: bool = False,
) -> Experiment003Run:
    if keep_workers:
        raise ValueError("KeepWorkers is incompatible with mandatory clean-shutdown evidence")
    repository_root = Path.cwd().resolve()
    requested_config_path = Path(config_path).expanduser().resolve()
    config = load_fanout_experiment_config(requested_config_path)
    if model_id is not None:
        config.model.model_id = model_id
    if revision is not None:
        config.model.revision = revision
    if config.model.revision != IMMUTABLE_QWEN3_REVISION:
        raise ValueError(f"resolved Experiment 003 revision must remain {IMMUTABLE_QWEN3_REVISION}")
    if worker_counts is not None:
        config.sweep.initial_worker_counts = worker_counts
    if repeats is not None:
        config.sweep.repeats = repeats
    if max_worker_count is not None:
        config.sweep.initial_worker_counts = [
            count for count in config.sweep.initial_worker_counts if count <= max_worker_count
        ]
        config.sweep.maximum_worker_count = max_worker_count
    if smoke:
        config.sweep.initial_worker_counts = [1, 2, 4]
        config.sweep.maximum_worker_count = 4
        config.sweep.repeats = 1
        config.workloads.cold.max_new_tokens = 4
        config.workloads.warm.max_new_tokens = 8
        config.hot_standby_idle_seconds = [0]
        skip_acquisition_tests = True
        skip_rejoin_test = True
    requested_payload = config.model_dump(mode="json")
    fingerprint = config_fingerprint(requested_payload)
    output_path = Path(output).expanduser().resolve()
    run_dir: Path
    if resume and (output_path / "experiment_state.json").is_file():
        run_dir = output_path
        direct_state = json.loads((run_dir / "experiment_state.json").read_text(encoding="utf-8"))
        if direct_state.get("config_fingerprint") != fingerprint:
            raise IntegrityError("resume state is incompatible with the requested configuration")
        experiment_id = str(direct_state.get("experiment_id", run_dir.name))
    elif resume and output_path.is_dir():
        possible_candidates = sorted(
            (
                path
                for path in output_path.glob("*-worker-fanout-*")
                if (path / "experiment_state.json").is_file()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        candidates = []
        for path in possible_candidates:
            state = json.loads((path / "experiment_state.json").read_text(encoding="utf-8"))
            if state.get("config_fingerprint") == fingerprint:
                candidates.append(path)
        if candidates:
            run_dir = candidates[0]
            experiment_id = str(
                json.loads((run_dir / "experiment_state.json").read_text(encoding="utf-8")).get(
                    "experiment_id", run_dir.name
                )
            )
        else:
            raise IntegrityError("no compatible incomplete worker-fanout run was found to resume")
    elif resume:
        raise IntegrityError(f"resume output path does not exist: {output_path}")
    if not resume:
        experiment_id = (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-worker-fanout-" + uuid4().hex[:8]
        )
        run_dir = output_path / experiment_id
        run_dir.mkdir(parents=True, exist_ok=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "charts").mkdir(exist_ok=True)
    if not resume or not (run_dir / "config.requested.yaml").is_file():
        shutil.copy2(requested_config_path, run_dir / "config.requested.yaml")
    origin_monotonic_ns = time.monotonic_ns()
    accumulator = EvidenceAccumulator(
        run_dir=run_dir,
        experiment_id=experiment_id,
        resume=resume,
    )
    previous_state: dict[str, Any] = {}
    if (run_dir / "experiment_state.json").is_file():
        previous_state = json.loads((run_dir / "experiment_state.json").read_text(encoding="utf-8"))
    completed_from_resume = valid_completed_counts(
        previous_state,
        expected_fingerprint=fingerprint,
    )
    if (
        resume
        and previous_state
        and not completed_from_resume
        and previous_state.get("config_fingerprint") not in {None, fingerprint}
    ):
        raise IntegrityError("resume state is incompatible with the requested configuration")
    if resume:
        accumulator.quarantine_incomplete_counts(completed_from_resume)
    print("[1/9] Inspecting environment", flush=True)
    if not resume or not (run_dir / "environment_before.json").is_file():
        environment_before = _environment_probe(
            run_dir / "environment_before.json",
            run_dir / "logs" / "environment_before.log",
        )
        _write_json(run_dir / "git.json", _git_evidence(repository_root))
    else:
        environment_before = json.loads(
            (run_dir / "environment_before.json").read_text(encoding="utf-8")
        )
    fatal_error: dict[str, Any] | None = None
    stage_layouts: dict[int, dict[str, Any]] = {}
    manifests: dict[int, Any] = {}
    shard_roots: dict[int, Path] = {}
    adaptive_rows: list[dict[str, Any]] = []
    reference: dict[str, Any] = {}
    resolved: Any = None
    description: Any = None
    architecture_config: dict[str, Any] = {}
    cold_prompt = ""
    cold_ids: list[int] = []
    warm_prompt = ""
    warm_ids: list[int] = []

    def persist() -> None:
        accumulator.persist(
            state_extra={
                "config_fingerprint": fingerprint,
                "smoke": smoke,
                "profile": profile,
                "stage_layouts": {str(key): value for key, value in stage_layouts.items()},
                "adaptive_search": adaptive_rows,
            }
        )
        _write_json(
            run_dir / "stage_layouts.json",
            [stage_layouts[key] for key in sorted(stage_layouts)],
        )
        _write_json(
            run_dir / "adaptive_search.json",
            adaptive_search_summary(
                initial_worker_counts=config.sweep.initial_worker_counts,
                adaptive_search_enabled=config.sweep.adaptive_search,
                attempts=adaptive_rows,
                maximum_worker_count=config.sweep.maximum_worker_count,
            ),
        )

    try:
        resolved = resolve_model(
            config.model.model_id,
            revision=config.model.revision,
            allow_download=True,
        )
        if resolved.revision != IMMUTABLE_QWEN3_REVISION:
            raise IntegrityError(
                f"model resolved to {resolved.revision}, expected {IMMUTABLE_QWEN3_REVISION}"
            )
        description = inspect_qwen3_model(resolved)
        inspection = model_inspection_payload(description)
        _write_json(run_dir / "model_inspection.json", inspection)
        architecture_config = json.loads(
            (resolved.path / "config.json").read_text(encoding="utf-8")
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            resolved.path,
            local_files_only=True,
        )
        cold_prompt, cold_ids = _prompt_with_exact_tokens(
            tokenizer,
            seed=(
                "Cold-start distributed inference must preserve exact tokens, direct "
                "activation transport, cache ownership, and immutable model identity."
            ),
            token_count=config.workloads.cold.input_tokens_approx,
        )
        warm_prompt, warm_ids = _prompt_with_exact_tokens(
            tokenizer,
            seed=(
                "A sustained worker fan-out benchmark records latency throughput memory "
                "streams queues and correctness without adding physical GPU compute."
            ),
            token_count=config.workloads.warm.input_tokens_approx,
        )
        print("[2/9] Preparing stage layouts", flush=True)
        initial_counts = initial_sweep_order(
            config.sweep.initial_worker_counts,
            config.sweep.maximum_worker_count,
        )
        for count in initial_counts:
            if count in completed_from_resume and count in accumulator.count_rows:
                continue
            shard_root, manifest, layout = _prepare_stage_layout(
                repository_root=repository_root,
                description=description,
                worker_count=count,
            )
            shard_roots[count] = shard_root
            manifests[count] = manifest
            stage_layouts[count] = layout
            print(
                f"Prepared {count} stages: "
                f"{'reused' if layout['reused'] else 'built'} "
                f"in {layout.get('one_time_shard_build_seconds', 0):.3f}s",
                flush=True,
            )
            persist()
        if not resume or not (run_dir / "reference.json").is_file():
            reference_requests = [
                {
                    "request_id": "reference-cold",
                    "name": "fixed-cold-128-token-prompt",
                    "prompt": cold_prompt,
                    "prompt_token_ids": cold_ids,
                    "max_new_tokens": config.workloads.cold.max_new_tokens,
                },
                {
                    "request_id": "reference-warm",
                    "name": "fixed-warm-128-token-prompt",
                    "prompt": warm_prompt,
                    "prompt_token_ids": warm_ids,
                    "max_new_tokens": config.workloads.warm.max_new_tokens,
                },
                {
                    "request_id": "reference-warmup",
                    "name": "fixed-warm-128-token-prompt-pipeline-warmup",
                    "prompt": warm_prompt,
                    "prompt_token_ids": warm_ids,
                    "max_new_tokens": config.warmup.full_pipeline_tokens,
                },
            ]
            reference = run_reference_suite_subprocess(
                model_id=config.model.model_id,
                model_revision=resolved.revision,
                model_path=resolved.path,
                requests=reference_requests,
                device=config.model.device,
                dtype_name=config.model.dtype,
                stage_layer_ends=list(range(1, 29)),
                output_dir=run_dir,
                log_path=run_dir / "logs" / "reference.log",
            )
        else:
            reference = json.loads((run_dir / "reference.json").read_text(encoding="utf-8"))
        reference_by_id = {str(row["request_id"]): row for row in reference["results"]}
        boundary_root = Path(reference["boundary_root"]).resolve()

        def spec(
            *,
            request_id: str,
            reference_id: str,
            prompt_ids: list[int],
            max_new_tokens: int,
            variant: str,
            concurrency: int = 1,
            group: str | None = None,
            idle_seconds: float | None = None,
        ) -> dict[str, Any]:
            return {
                "request_id": request_id,
                "reference_id": reference_id,
                "prompt_token_ids": prompt_ids,
                "max_new_tokens": max_new_tokens,
                "variant": variant,
                "concurrency": concurrency,
                "concurrency_group": group,
                "idle_seconds": idle_seconds,
            }

        async def session_for(
            *,
            count: int,
            phase: str,
            repeat_index: int,
            requests: list[dict[str, Any]],
            local_warmup: bool,
            pipeline_warmup: bool,
            boundary: bool = False,
            hot_idle: list[float] | None = None,
            shard_overrides: dict[int, Path] | None = None,
            acquisition_events: dict[int, dict[str, Any]] | None = None,
            rejoin_stage: int | None = None,
        ) -> FanoutSessionResult:
            if count not in manifests:
                shard_root, manifest, layout = _prepare_stage_layout(
                    repository_root=repository_root,
                    description=description,
                    worker_count=count,
                )
                shard_roots[count] = shard_root
                manifests[count] = manifest
                stage_layouts[count] = layout
                persist()
            warmup_request = spec(
                request_id=(f"warmup-c{count:02d}-{phase}-r{repeat_index:02d}"),
                reference_id="reference-warmup",
                prompt_ids=warm_ids,
                max_new_tokens=config.warmup.full_pipeline_tokens,
                variant="full_pipeline_warmup",
            )
            result = await run_fanout_session(
                config=config,
                experiment_id=experiment_id,
                origin_monotonic_ns=origin_monotonic_ns,
                worker_count=count,
                phase=phase,
                repeat=repeat_index,
                manifest=manifests[count],
                architecture_config=architecture_config,
                model_path=resolved.path,
                shard_root=shard_roots[count],
                session_root=(
                    run_dir
                    / "logs"
                    / f"count-{count:02d}"
                    / f"{phase}-r{repeat_index:02d}-{uuid4().hex[:6]}"
                ),
                requests=requests,
                reference_by_id=reference_by_id,
                stage_local_warmup=local_warmup,
                pipeline_warmup=pipeline_warmup,
                pipeline_warmup_request=warmup_request,
                boundary_root=boundary_root,
                boundary_enabled=boundary,
                hot_idle_seconds=hot_idle,
                shard_roots_by_stage=shard_overrides,
                acquisition_events_by_stage=acquisition_events,
                rejoin_stage_id=rejoin_stage,
                rejoin_after_tokens=config.rejoin.committed_tokens_before_failure,
            )
            accumulator.add_session(result)
            persist()
            peak_gpu = max(
                (
                    int(row["gpu_memory_used_bytes"])
                    for row in result.resource_rows
                    if row.get("gpu_memory_used_bytes") is not None
                ),
                default=0,
            )
            print(
                f"Testing {count} workers, {phase}, repeat {repeat_index}/"
                f"{config.sweep.repeats}: "
                f"pipeline_ready={result.pipeline_ready_seconds!s}s "
                f"peak_gpu={peak_gpu / 1024**3:.2f}GiB "
                f"Result: {'PASS' if result.passed else 'FAIL'}",
                flush=True,
            )
            return result

        def run_count(count: int) -> dict[str, Any]:
            if count in completed_from_resume and count in accumulator.count_rows:
                print(f"Resuming: count {count} already has valid complete evidence", flush=True)
                return accumulator.count_rows[count]
            print(f"Testing {count} workers", flush=True)
            load_failures = 0
            for repeat_index in range(1, config.sweep.repeats + 1):
                result = asyncio.run(
                    session_for(
                        count=count,
                        phase="cached_cold_load_only",
                        repeat_index=repeat_index,
                        requests=[],
                        local_warmup=True,
                        pipeline_warmup=False,
                    )
                )
                if not result.passed:
                    load_failures += 1
                if load_failures >= 2:
                    return _count_result(
                        count=count,
                        accumulator=accumulator,
                        config=config,
                    )
            for variant, local_warmup in (
                ("cold_no_stage_warmup", False),
                ("cold_with_stage_warmup", True),
            ):
                for repeat_index in range(1, config.sweep.repeats + 1):
                    request = spec(
                        request_id=(f"{variant}-c{count:02d}-r{repeat_index:02d}"),
                        reference_id="reference-cold",
                        prompt_ids=cold_ids,
                        max_new_tokens=config.workloads.cold.max_new_tokens,
                        variant=variant,
                    )
                    asyncio.run(
                        session_for(
                            count=count,
                            phase=variant,
                            repeat_index=repeat_index,
                            requests=[request],
                            local_warmup=local_warmup,
                            pipeline_warmup=False,
                        )
                    )
            for repeat_index in range(1, config.sweep.repeats + 1):
                requests = [
                    spec(
                        request_id=f"warm-c1-c{count:02d}-r{repeat_index:02d}",
                        reference_id="reference-warm",
                        prompt_ids=warm_ids,
                        max_new_tokens=config.workloads.warm.max_new_tokens,
                        variant="warm_sustained",
                        concurrency=1,
                    )
                ]
                requests.extend(
                    spec(
                        request_id=(f"warm-c4-{member}-c{count:02d}-r{repeat_index:02d}"),
                        reference_id="reference-warm",
                        prompt_ids=warm_ids,
                        max_new_tokens=config.workloads.warm.max_new_tokens,
                        variant="warm_sustained",
                        concurrency=4,
                        group=f"warm-c4-c{count:02d}-r{repeat_index:02d}",
                    )
                    for member in range(4)
                )
                asyncio.run(
                    session_for(
                        count=count,
                        phase="warm",
                        repeat_index=repeat_index,
                        requests=requests,
                        local_warmup=True,
                        pipeline_warmup=True,
                        boundary=(
                            repeat_index == 1
                            and count
                            in {
                                int(value)
                                for value in config.correctness.boundary_validation_counts
                                if isinstance(value, int)
                            }
                        ),
                    )
                )
            hot_requests = [
                spec(
                    request_id=f"hot-{int(idle)}s-c{count:02d}",
                    reference_id="reference-cold",
                    prompt_ids=cold_ids,
                    max_new_tokens=config.workloads.cold.max_new_tokens,
                    variant="hot_standby",
                    idle_seconds=idle,
                )
                for idle in config.hot_standby_idle_seconds
            ]
            asyncio.run(
                session_for(
                    count=count,
                    phase="hot_standby",
                    repeat_index=1,
                    requests=hot_requests,
                    local_warmup=True,
                    pipeline_warmup=True,
                    hot_idle=config.hot_standby_idle_seconds,
                )
            )
            return _count_result(
                count=count,
                accumulator=accumulator,
                config=config,
            )

        first_failure: int | None = None
        last_success = 0
        for index, count in enumerate(initial_counts, start=3):
            print(f"[{min(index, 8)}/9] Testing {count} workers", flush=True)
            row = run_count(count)
            adaptive_rows.append(
                {
                    "worker_count": count,
                    "search_phase": "coarse",
                    "runnable": row["runnable"],
                    "stable": row["stable"],
                    "failure_reason": row.get("failure_reason"),
                }
            )
            if row["runnable"]:
                last_success = max(last_success, count)
            else:
                first_failure = count
                break
        if first_failure is None and last_success < config.sweep.maximum_worker_count:
            count = config.sweep.maximum_worker_count
            row = run_count(count)
            adaptive_rows.append(
                {
                    "worker_count": count,
                    "search_phase": "maximum-probe",
                    "runnable": row["runnable"],
                    "stable": row["stable"],
                    "failure_reason": row.get("failure_reason"),
                }
            )
            if row["runnable"]:
                last_success = count
            else:
                first_failure = count
        if (
            config.sweep.adaptive_search
            and first_failure is not None
            and first_failure - last_success > 1
        ):
            while True:
                candidate = next_adaptive_count(last_success, first_failure)
                if candidate is None:
                    break
                row = run_count(candidate)
                adaptive_rows.append(
                    {
                        "worker_count": candidate,
                        "search_phase": "adaptive-binary",
                        "lower_success_before": last_success,
                        "upper_failure_before": first_failure,
                        "runnable": row["runnable"],
                        "stable": row["stable"],
                        "failure_reason": row.get("failure_reason"),
                    }
                )
                if row["runnable"]:
                    last_success = candidate
                else:
                    first_failure = candidate
        maximum_runnable = max(
            (count for count, row in accumulator.count_rows.items() if row.get("runnable")),
            default=0,
        )
        maximum_stable = max(
            (count for count, row in accumulator.count_rows.items() if row.get("stable")),
            default=0,
        )
        maximum_boundary_already_valid = any(
            int(row.get("worker_count", -1)) == maximum_runnable
            and row.get("phase") == "boundary_validation"
            and row.get("passed") is True
            and row.get("exact_token_identity") is True
            and row.get("direct_data_plane") is True
            for row in accumulator.session_rows
        )
        if (
            maximum_runnable
            and maximum_runnable
            not in {
                int(value)
                for value in config.correctness.boundary_validation_counts
                if isinstance(value, int)
            }
            and not maximum_boundary_already_valid
        ):
            boundary_request = spec(
                request_id=f"boundary-maximum-runnable-c{maximum_runnable:02d}",
                reference_id="reference-warm",
                prompt_ids=warm_ids,
                max_new_tokens=config.workloads.warm.max_new_tokens,
                variant="boundary_validation",
            )
            asyncio.run(
                session_for(
                    count=maximum_runnable,
                    phase="boundary_validation",
                    repeat_index=1,
                    requests=[boundary_request],
                    local_warmup=True,
                    pipeline_warmup=True,
                    boundary=True,
                )
            )
        _write_json(
            run_dir / "adaptive_search.json",
            adaptive_search_summary(
                initial_worker_counts=initial_counts,
                adaptive_search_enabled=config.sweep.adaptive_search,
                attempts=adaptive_rows,
                maximum_worker_count=config.sweep.maximum_worker_count,
            ),
        )

        acquisition_status = "SKIPPED" if skip_acquisition_tests else "FAIL"
        print("[7/9] Running node-state acquisition tests", flush=True)
        acquisition_work = run_dir / ".acquisition-work"
        if not skip_acquisition_tests and maximum_stable > 0:
            representative_counts: list[int] = []
            for value in config.node_states.unprovisioned.representative_stage_counts:
                count = maximum_stable if value == "maximum_stable" else int(value)
                if (
                    1 <= count <= config.sweep.maximum_worker_count
                    and count not in representative_counts
                ):
                    if count not in manifests:
                        shard_root, manifest, layout = _prepare_stage_layout(
                            repository_root=repository_root,
                            description=description,
                            worker_count=count,
                        )
                        shard_roots[count] = shard_root
                        manifests[count] = manifest
                        stage_layouts[count] = layout
                    representative_counts.append(count)
            acquisition_status = "PASS"
            case_index = 0
            expected_acquisition_keys: set[tuple[int, str, str]] = set()
            for count in representative_counts:
                manifest = manifests[count]
                role_stages = [
                    ("first", 0),
                    ("middle", count // 2),
                    ("final", count - 1),
                ]
                seen_stages: set[int] = set()
                for role, stage_id in role_stages:
                    if stage_id in seen_stages:
                        continue
                    seen_stages.add(stage_id)
                    stage = manifest.stages[stage_id]
                    for profile_name in config.node_states.unprovisioned.acquisition_profiles:
                        case_index += 1
                        case_key = (count, role, profile_name)
                        expected_acquisition_keys.add(case_key)
                        existing_rows = [
                            row
                            for row in accumulator.tables["acquisition_results.csv"]
                            if int(row.get("worker_count", -1)) == count
                            and row.get("stage_role") == role
                            and row.get("profile") == profile_name
                        ]
                        existing_valid = any(
                            row.get("session_passed") is True
                            and row.get("atomic_rename") is True
                            and row.get("expected_hash")
                            == manifest.shard_hashes[f"stage-{stage_id:03d}"]
                            and row.get("actual_hash") == row.get("expected_hash")
                            and int(row.get("shard_bytes", 0)) > 0
                            and float(row.get("total_acquisition_duration_seconds", -1)) >= 0
                            for row in existing_rows
                        )
                        if resume and existing_valid:
                            print(
                                f"Reusing valid acquisition evidence: {count} workers, "
                                f"{role}, {profile_name}",
                                flush=True,
                            )
                            continue
                        if existing_rows:
                            accumulator.tables["acquisition_results.csv"] = [
                                row
                                for row in accumulator.tables["acquisition_results.csv"]
                                if not (
                                    int(row.get("worker_count", -1)) == count
                                    and row.get("stage_role") == role
                                    and row.get("profile") == profile_name
                                )
                            ]
                        profile_settings = config.shard_acquisition_profiles[profile_name]
                        worker_root = acquisition_work / f"case-{case_index:03d}" / "worker-root"
                        destination = worker_root / f"stage-{stage_id:03d}"
                        assignment_ns = time.monotonic_ns()
                        acquisition_started_ns = time.monotonic_ns()
                        transfer = acquire_shard_directory(
                            source=shard_roots[count] / f"stage-{stage_id:03d}",
                            destination=destination,
                            expected_hash=manifest.shard_hashes[f"stage-{stage_id:03d}"],
                            profile=profile_name,
                            bandwidth_mbps=profile_settings.bandwidth_mbps,
                            latency_ms=profile_settings.latency_ms,
                        )
                        acquisition_completed_ns = time.monotonic_ns()
                        acquisition_event = {
                            stage_id: {
                                "assignment_created_ns": assignment_ns,
                                "acquisition_started_ns": acquisition_started_ns,
                                "acquisition_completed_ns": acquisition_completed_ns,
                                "details": {
                                    "node_state": "unprovisioned",
                                    "profile": profile_name,
                                    "measurement_class": ("emulated-shard-acquisition"),
                                    "bandwidth_mbps": profile_settings.bandwidth_mbps,
                                    "latency_ms": profile_settings.latency_ms,
                                },
                            }
                        }
                        result = asyncio.run(
                            session_for(
                                count=count,
                                phase=f"unprovisioned_{profile_name}_{role}",
                                repeat_index=1,
                                requests=[],
                                local_warmup=True,
                                pipeline_warmup=False,
                                shard_overrides={stage_id: worker_root},
                                acquisition_events=acquisition_event,
                            )
                        )
                        lifecycle = next(
                            (
                                row
                                for row in result.worker_lifecycle_rows
                                if int(row["stage_id"]) == stage_id
                            ),
                            {},
                        )
                        row = {
                            "experiment_id": experiment_id,
                            "worker_count": count,
                            "stage_count": count,
                            "stage_role": role,
                            "stage_id": stage_id,
                            "stage_layer_start": stage.layer_start,
                            "stage_layer_end": stage.layer_end,
                            **transfer.to_dict(),
                            "time_to_contribution_seconds": lifecycle.get(
                                "cold_time_to_contribution_seconds"
                            ),
                            "shard_read_seconds": lifecycle.get("shard_read_seconds"),
                            "verification_seconds_worker": lifecycle.get(
                                "shard_verification_seconds"
                            ),
                            "weight_load_seconds": lifecycle.get("weight_load_seconds"),
                            "warmup_seconds": lifecycle.get("local_warmup_seconds"),
                            "worker_ready_seconds": lifecycle.get("worker_ready_seconds"),
                            "session_passed": result.passed,
                            "file_cache_label": ("local-shard-read-with-uncontrolled-os-cache"),
                        }
                        accumulator.tables["acquisition_results.csv"].append(row)
                        if not result.passed:
                            acquisition_status = "FAIL"
                        if worker_root.parent.exists():
                            shutil.rmtree(worker_root.parent)
                        persist()
            completed_acquisition_keys = {
                (
                    int(row.get("worker_count", -1)),
                    str(row.get("stage_role")),
                    str(row.get("profile")),
                )
                for row in accumulator.tables["acquisition_results.csv"]
                if row.get("session_passed") is True
                and row.get("atomic_rename") is True
                and row.get("actual_hash") == row.get("expected_hash")
            }
            if not expected_acquisition_keys or not expected_acquisition_keys.issubset(
                completed_acquisition_keys
            ):
                acquisition_status = "FAIL"

        print("[8/9] Running rejoin and cache replay", flush=True)
        rejoin_status = "SKIPPED" if skip_rejoin_test else "FAIL"
        rejoin_result: dict[str, Any] = {
            "status": "SKIPPED",
            "reason": "disabled or no stable worker count",
        }
        if not skip_rejoin_test and maximum_stable > 0:
            if resume:
                accumulator.quarantine_phase_attempts("rejoin")
            target_stage = maximum_stable // 2
            request = spec(
                request_id=f"rejoin-c{maximum_stable:02d}",
                reference_id="reference-cold",
                prompt_ids=cold_ids,
                max_new_tokens=config.workloads.cold.max_new_tokens,
                variant="rejoin",
            )
            session = asyncio.run(
                session_for(
                    count=maximum_stable,
                    phase="rejoin",
                    repeat_index=1,
                    requests=[request],
                    local_warmup=True,
                    pipeline_warmup=True,
                    rejoin_stage=target_stage,
                )
            )
            rejoin_result = session.rejoin or {
                "status": "FAIL",
                "error": session.failure_message or "no rejoin metrics",
            }
            rejoin_status = str(rejoin_result.get("status", "FAIL"))
        _write_json(run_dir / "rejoin_result.json", rejoin_result)

        print("[9/9] Calculating economics and generating reports", flush=True)
        accumulator.tables["node_economics.csv"] = []
        tested_counts = sorted(accumulator.count_rows)
        for count in tested_counts:
            count_row = accumulator.count_rows[count]
            throughput = float(count_row.get("median_warm_output_tokens_per_second") or 0.0)
            cached_startup = count_row.get("median_worker_ready_seconds")
            hot_rows = [
                row
                for row in accumulator.tables["warm_inference.csv"]
                if int(row["worker_count"]) == count
                and row.get("phase") == "hot_standby"
                and row.get("hot_time_to_contribution_seconds") is not None
            ]
            hot_startup = _median(
                [float(row["hot_time_to_contribution_seconds"]) for row in hot_rows]
            )
            states: list[tuple[str, str | None, float]] = []
            if cached_startup is not None:
                states.append(("cached-cold", None, float(cached_startup)))
            if hot_startup is not None:
                states.append(("hot-standby", None, float(hot_startup)))
            acquisition_rows = [
                row
                for row in accumulator.tables["acquisition_results.csv"]
                if int(row["worker_count"]) == count
                and row.get("time_to_contribution_seconds") is not None
            ]
            for profile_name in sorted({str(row["profile"]) for row in acquisition_rows}):
                startup = _median(
                    [
                        float(row["time_to_contribution_seconds"])
                        for row in acquisition_rows
                        if row["profile"] == profile_name
                    ]
                )
                if startup is not None:
                    states.append(("unprovisioned", profile_name, startup))
            for node_state, state_profile, startup in states:
                break_even = {
                    f"minimum_lease_{int(target * 100)}_seconds": (
                        minimum_lease_duration(startup, target)
                    )
                    for target in config.economics.productive_fraction_targets
                }
                for lease in config.economics.availability_seconds:
                    accumulator.tables["node_economics.csv"].append(
                        {
                            "experiment_id": experiment_id,
                            "worker_count": count,
                            "node_state": node_state,
                            "profile": state_profile,
                            "lease_seconds": lease,
                            "startup_seconds": startup,
                            "verified_tokens_per_second": throughput,
                            "productive_fraction": productive_fraction(lease, startup),
                            "productive_tokens": productive_tokens(
                                lease,
                                startup,
                                throughput,
                            ),
                            **break_even,
                        }
                    )
        single_optimum, concurrency_optimum = performance_optima(accumulator.count_rows.values())
        profiling_evidence = _build_profiling_evidence(
            profile_requested=profile,
            maximum_stable=maximum_stable,
            tables=accumulator.tables,
        )
        _write_json(run_dir / "profiling.json", profiling_evidence)
        _write_json(
            run_dir / "failure_diagnostics.json",
            _build_failure_diagnostics(
                session_rows=accumulator.session_rows,
                lifecycle_events=accumulator.lifecycle_events,
                tables=accumulator.tables,
            ),
        )
        maximum_attempted = max(accumulator.count_rows, default=0)
        first_non_runnable = min(
            (
                count
                for count, row in accumulator.count_rows.items()
                if not row.get("runnable") and count > maximum_runnable
            ),
            default=None,
        )
        first_failure_reason = (
            accumulator.count_rows[first_non_runnable].get("failure_reason")
            if first_non_runnable is not None
            else None
        )
        lifecycle_status = (
            "PASS"
            if accumulator.session_rows
            and all(
                not row.get("lifecycle_validation_errors")
                for row in accumulator.session_rows
                if row.get("passed")
            )
            else "FAIL"
        )
        successful_generation_sessions = [
            row for row in accumulator.session_rows if row.get("runnable_generation")
        ]
        correctness_status = (
            "PASS"
            if successful_generation_sessions
            and all(row.get("exact_token_identity") for row in successful_generation_sessions)
            else "FAIL"
        )
        direct_status = (
            "PASS"
            if successful_generation_sessions
            and all(row.get("direct_data_plane") for row in successful_generation_sessions)
            else "FAIL"
        )
        resource_rows = accumulator.tables["resource_usage.csv"]
        resource_required_fields = (
            "gpu_memory_used_bytes",
            "gpu_memory_free_bytes",
            "gpu_memory_total_bytes",
            "aggregate_experiment_rss_bytes",
            "system_available_ram_bytes",
            "gpu_utilisation_percent",
            "memory_controller_utilisation_percent",
            "power_draw_watts",
            "temperature_c",
            "graphics_clock_mhz",
            "memory_clock_mhz",
            "system_commit_or_swap_used_bytes",
            "cuda_context_count",
        )
        gpu_process_rows = accumulator.tables["gpu_process_memory.csv"]
        resource_status = (
            "PASS"
            if resource_rows
            and all(
                any(row.get(field) is not None for row in resource_rows)
                for field in resource_required_fields
            )
            and accumulator.tables["worker_memory.csv"]
            and gpu_process_rows
            and any(
                row.get("gpu_process_memory_bytes") is not None
                or row.get("nvml_gpu_memory_bytes") is not None
                for row in gpu_process_rows
            )
            else "FAIL"
        )
        environment_after = _environment_probe(
            run_dir / "environment_after.json",
            run_dir / "logs" / "environment_after.log",
        )
        environment_status = (
            "PASS"
            if environment_before.get("cuda_available") is True
            and environment_after.get("cuda_available") is True
            and environment_before.get("pytorch_version")
            == environment_after.get("pytorch_version")
            and environment_before.get("transformers_version")
            == environment_after.get("transformers_version")
            else "FAIL"
        )
        cached_summary = (
            accumulator.count_rows.get(maximum_stable, {}).get("median_worker_ready_seconds")
            if maximum_stable
            else None
        )
        hot_summary = _median(
            [
                float(row["hot_time_to_contribution_seconds"])
                for row in accumulator.tables["warm_inference.csv"]
                if int(row["worker_count"]) == maximum_stable
                and row.get("phase") == "hot_standby"
                and row.get("hot_time_to_contribution_seconds") is not None
            ]
        )
        unprovisioned_summary: dict[str, float] = {}
        for profile_name in config.shard_acquisition_profiles:
            values = [
                float(row["time_to_contribution_seconds"])
                for row in accumulator.tables["acquisition_results.csv"]
                if int(row["worker_count"]) == maximum_stable
                and row["profile"] == profile_name
                and row.get("time_to_contribution_seconds") is not None
            ]
            if values:
                unprovisioned_summary[profile_name] = statistics.median(values)
        if first_non_runnable is not None:
            measured_limiting_factor = (
                f"the confirmed {first_non_runnable}-worker failure: "
                f"{first_failure_reason or 'see preserved failure diagnostics'}"
            )
        elif maximum_runnable == 28:
            measured_limiting_factor = (
                "the model architecture's 28 decoder-layer semantic ceiling; "
                "no higher non-zero-layer partition exists"
            )
        elif smoke:
            measured_limiting_factor = (
                "the smoke-run boundary; smoke evidence is not a worker-limit measurement"
            )
        else:
            measured_limiting_factor = "the highest completed measured worker count"
        resolved_config = config.model_dump(mode="json")
        resolved_config["model"]["resolved_local_snapshot_path"] = str(resolved.path)
        resolved_config["resolved"] = {
            "maximum_runnable_worker_count": maximum_runnable,
            "maximum_stable_worker_count": maximum_stable,
            "boundary_validation_counts": sorted(
                {
                    int(value) if isinstance(value, int) else maximum_runnable
                    for value in config.correctness.boundary_validation_counts
                }
            ),
            "representative_acquisition_stage_counts": sorted(
                {
                    int(value) if isinstance(value, int) else maximum_stable
                    for value in config.node_states.unprovisioned.representative_stage_counts
                    if (isinstance(value, int) or maximum_stable)
                }
            ),
            "stage_layout_roots": {str(key): str(value) for key, value in shard_roots.items()},
            "smoke": smoke,
        }
        (run_dir / "config.resolved.yaml").write_text(
            yaml.safe_dump(resolved_config, sort_keys=False),
            encoding="utf-8",
        )
        mandatory_statuses = {
            "environment_status": environment_status,
            "real_model_status": (
                "PASS"
                if reference.get("full_model_loaded") is True and maximum_runnable > 0
                else "FAIL"
            ),
            "correctness_status": correctness_status,
            "direct_data_plane_status": direct_status,
            "lifecycle_instrumentation_status": lifecycle_status,
            "resource_measurement_status": resource_status,
            "acquisition_experiment_status": acquisition_status,
            "rejoin_status": rejoin_status,
        }
        integrity = (
            "PASS"
            if all(status in {"PASS", "SKIPPED"} for status in mandatory_statuses.values())
            and maximum_runnable > 0
            else "FAIL"
        )
        summary: dict[str, Any] = {
            "experiment_integrity_status": integrity,
            **mandatory_statuses,
            "maximum_semantic_worker_count": 28,
            "maximum_attempted_worker_count": maximum_attempted,
            "maximum_runnable_worker_count": maximum_runnable,
            "maximum_stable_worker_count": maximum_stable,
            "single_request_latency_optimal_worker_count": single_optimum,
            "concurrency_4_throughput_optimal_worker_count": concurrency_optimum,
            "cached_cold_median_time_to_contribution_seconds": cached_summary,
            "hot_standby_median_time_to_contribution_seconds": hot_summary,
            "unprovisioned_time_to_contribution_by_profile": unprovisioned_summary,
            "first_non_runnable_worker_count": first_non_runnable,
            "first_non_runnable_failure_reason": first_failure_reason,
            "measured_limiting_factor": measured_limiting_factor,
            "execution_mode": "single-host-loopback-real-model-fanout",
            "model_id": config.model.model_id,
            "model_revision": resolved.revision,
            "smoke": smoke,
            "profile": profile,
            "file_cache_control": config.file_cache_control,
            "run_directory": str(run_dir),
            "report_path": str(run_dir / "report.html"),
            "completed_at": datetime.now(UTC).isoformat(),
        }
        summary["overall_status"] = (
            "PASS"
            if summary["experiment_integrity_status"] == "PASS"
            and all(
                value in {"PASS", "SKIPPED"}
                for key, value in mandatory_statuses.items()
                if key
                not in {
                    "acquisition_experiment_status",
                    "rejoin_status",
                }
                or value != "FAIL"
            )
            else "FAIL"
        )
        write_all_tables(run_dir, accumulator.tables)
        write_charts(run_dir, accumulator.tables)
        _write_json(run_dir / "summary.json", summary)
        report = render_report(
            run_dir=run_dir,
            summary=summary,
            count_rows=[accumulator.count_rows[key] for key in sorted(accumulator.count_rows)],
            acquisition_rows=accumulator.tables["acquisition_results.csv"],
            economics_rows=accumulator.tables["node_economics.csv"],
            rejoin=rejoin_result,
        )
        persist()
        write_artifact_manifest(run_dir)
        if acquisition_work.exists():
            shutil.rmtree(acquisition_work)
        return Experiment003Run(
            run_directory=run_dir,
            report_path=report,
            summary=summary,
        )
    except Exception as exc:
        fatal_error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        _write_json(run_dir / "fatal_error.json", fatal_error)
        with suppress_exception():
            _environment_probe(
                run_dir / "environment_after.json",
                run_dir / "logs" / "environment_after.log",
            )
        persist()
        summary = {
            "experiment_integrity_status": "FAIL",
            "environment_status": "FAIL",
            "real_model_status": "FAIL",
            "correctness_status": "FAIL",
            "direct_data_plane_status": "FAIL",
            "lifecycle_instrumentation_status": "FAIL",
            "resource_measurement_status": "FAIL",
            "acquisition_experiment_status": ("SKIPPED" if skip_acquisition_tests else "FAIL"),
            "rejoin_status": "SKIPPED" if skip_rejoin_test else "FAIL",
            "maximum_semantic_worker_count": 28,
            "maximum_attempted_worker_count": max(accumulator.count_rows, default=0),
            "maximum_runnable_worker_count": max(
                (count for count, row in accumulator.count_rows.items() if row.get("runnable")),
                default=0,
            ),
            "maximum_stable_worker_count": max(
                (count for count, row in accumulator.count_rows.items() if row.get("stable")),
                default=0,
            ),
            "single_request_latency_optimal_worker_count": 0,
            "concurrency_4_throughput_optimal_worker_count": 0,
            "cached_cold_median_time_to_contribution_seconds": None,
            "hot_standby_median_time_to_contribution_seconds": None,
            "unprovisioned_time_to_contribution_by_profile": {},
            "overall_status": "FAIL",
            "fatal_error": fatal_error,
            "run_directory": str(run_dir),
            "report_path": str(run_dir / "report.html"),
        }
        _write_json(run_dir / "summary.json", summary)
        report = render_report(
            run_dir=run_dir,
            summary=summary,
            count_rows=[accumulator.count_rows[key] for key in sorted(accumulator.count_rows)],
            acquisition_rows=accumulator.tables["acquisition_results.csv"],
            economics_rows=accumulator.tables["node_economics.csv"],
            rejoin={},
        )
        write_all_tables(run_dir, accumulator.tables)
        return Experiment003Run(
            run_directory=run_dir,
            report_path=report,
            summary=summary,
        )


class suppress_exception:
    """Tiny context manager used only while preserving fatal-run evidence."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> bool:
        return True
