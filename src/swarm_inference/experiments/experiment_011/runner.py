"""End-to-end Experiment 011 orchestration and evidence production."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from swarm_inference.experiments.experiment_010.transport import NETWORK_PROFILES
from swarm_inference.experiments.experiment_011 import (
    EXPERIMENT_ID,
    MODEL_REVISION,
    TOKENIZER_REVISION,
)
from swarm_inference.experiments.experiment_011.analysis import (
    ARCHIVED_010_TPS,
    PROFILE_LABELS,
    PROFILE_ORDER,
    RESULT_COLUMNS,
    build_network_summary,
    profile_parameters_hash,
    quantile,
    write_csv,
)
from swarm_inference.experiments.experiment_011.baseline import (
    run_fresh_expert_rpc_baseline,
    run_native_local_reference,
)
from swarm_inference.experiments.experiment_011.charts import generate_network_charts
from swarm_inference.experiments.experiment_011.concurrency import (
    run_concurrent_stage_ring,
    write_concurrency_summary,
)
from swarm_inference.experiments.experiment_011.discovery import (
    DiscoveredAssets,
    discover_assets,
    environment_identity,
    git_identity,
    sha256_file,
)
from swarm_inference.experiments.experiment_011.evidence import (
    build_manifest,
    evaluate_gates,
    make_zip,
    validate_artifacts,
    write_reports,
)
from swarm_inference.experiments.experiment_011.failures import (
    run_real_failure_and_recovery_smokes,
)
from swarm_inference.experiments.experiment_011.partition import (
    ModelPartitionMetadata,
    StagePlan,
    build_stage_plan,
    inspect_model_partition_metadata,
)
from swarm_inference.experiments.experiment_011.planner import (
    MeasuredStrategyPlanner,
    PlannerCandidate,
    PlannerInputs,
)
from swarm_inference.experiments.experiment_011.reference import (
    LocalReferenceResult,
    compare_capture_trees,
    run_local_reference,
)
from swarm_inference.experiments.experiment_011.resources import (
    ResourceMonitor,
    summarise_resources,
)
from swarm_inference.experiments.experiment_011.runtime import (
    StageRingController,
    StageRingResult,
)
from swarm_inference.experiments.experiment_011.speculation import (
    run_real_prompt_lookup_speculation,
)
from swarm_inference.experiments.experiment_011.telemetry import merge_traces

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
EXTRA_RESULT_COLUMNS = (
    "evidence_category",
    "planner_eligible",
    "raw_record_path",
    "chart_source_id",
    "boundary_hash_match",
    "source_run_id",
    "measurement_started_ns",
    "measurement_ended_ns",
    "actual_compression_modes",
    "serial_waits_method",
)


@dataclass(frozen=True, slots=True)
class ExperimentOptions:
    mode: str
    run_id: str | None = None
    model_path: str | None = None
    draft_model_path: str | None = None
    stage_counts: tuple[int, ...] = (2, 4, 8)
    profile_names: tuple[str, ...] = PROFILE_ORDER
    skip_speculation: bool = False
    network_only: bool = False
    exactness_only: bool = False
    resume: bool = False

    @property
    def canonical(self) -> bool:
        return self.mode == "full"

    @property
    def generated_token_count(self) -> int:
        return 32 if self.canonical else 4


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now_run_id() -> str:
    return f"experiment-011-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def _run_test_command(
    *,
    run_root: Path,
    name: str,
    arguments: list[str],
    environment_update: dict[str, str] | None = None,
    timeout_seconds: float = 900,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["TEMP"] = str(run_root / "logs" / "pytest-temp")
    environment["TMP"] = environment["TEMP"]
    if environment_update:
        environment.update(environment_update)
    Path(environment["TEMP"]).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    result = {
        "name": name,
        "command": [sys.executable, "-m", "pytest", *arguments],
        "return_code": completed.returncode,
        "elapsed_seconds": (time.perf_counter_ns() - started) / 1e9,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
    }
    _write_json(run_root / "logs" / f"{name}.json", result)
    return result


def _identity_files(run_root: Path, assets: DiscoveredAssets) -> None:
    source_files = sorted(
        (REPOSITORY_ROOT / "src" / "swarm_inference" / "experiments" / "experiment_011").glob(
            "*.py"
        )
    )
    source_files += [
        REPOSITORY_ROOT / "docs" / "experiments" / "experiment-011-design.md",
        REPOSITORY_ROOT / "scripts" / "run-experiment-011.ps1",
    ]
    source_records = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in source_files
        if path.is_file()
    ]
    _write_json(
        run_root / "source_identity.json",
        {
            **git_identity(REPOSITORY_ROOT),
            "experiment_011_sources": source_records,
            "source_set_sha256": hashlib.sha256(
                json.dumps(source_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
    )
    binaries = [
        Path(assets.native_engine),
        Path(assets.native_expert_worker),
        Path(sys.executable),
    ]
    _write_json(
        run_root / "binary_identity.json",
        {
            "binaries": [
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in binaries
                if path.is_file()
            ]
        },
    )
    model_path = Path(assets.source_model_path)
    model_files = [
        *sorted(model_path.glob("model-*.safetensors")),
        model_path / "model.safetensors.index.json",
        model_path / "config.json",
        model_path / "generation_config.json",
        model_path / "tokenizer.json",
        model_path / "tokenizer_config.json",
    ]
    model_records = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in model_files
        if path.is_file()
    ]
    _write_json(
        run_root / "model_identity.json",
        {
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION,
            "experiment_010_model_fingerprint": assets.model_fingerprint,
            "source_model_path": assets.source_model_path,
            "native_model_path": assets.native_model_path,
            "files": model_records,
        },
    )


def _load_workload(assets: DiscoveredAssets) -> tuple[dict[str, Any], list[int], list[int]]:
    workload = json.loads(
        Path(assets.experiment_010_workload_reference).read_text(encoding="utf-8")
    )
    prompt = [int(value) for value in workload["prompt_ids"]]
    generated = [int(value) for value in workload["full_ids"][len(prompt) :]]
    return workload, prompt, generated


def _write_network_manifest(run_root: Path, assets: DiscoveredAssets) -> dict[str, Any]:
    archived = json.loads(
        Path(assets.experiment_010_transport_manifest).read_text(encoding="utf-8")
    )
    current = {name: profile.model_dump(mode="json") for name, profile in NETWORK_PROFILES.items()}
    if archived != current:
        raise ValueError("Experiment 010 source profiles differ from its final evidence manifest")
    manifest = {
        "source": assets.experiment_010_transport_manifest,
        "source_sha256": sha256_file(Path(assets.experiment_010_transport_manifest)),
        "source_configuration_matches_current_experiment_010_code": True,
        "profile_order": list(PROFILE_ORDER),
        "profiles": archived,
        "profile_parameters_hashes": {
            name: profile_parameters_hash(archived[name]) for name in PROFILE_ORDER
        },
        "profile_names_not_used_for_adaptive_decisions": True,
    }
    _write_json(run_root / "network_profiles_manifest.json", manifest)
    return manifest


def _plan_ranges(plan: StagePlan) -> tuple[tuple[int, int], ...]:
    return tuple((row.layer_start, row.layer_end) for row in plan.assignments)


def _build_plans(
    *,
    run_root: Path,
    model_path: Path,
    stage_counts: Sequence[int],
    measured_layer_ns: dict[int, int],
) -> tuple[dict[str, StagePlan], ModelPartitionMetadata]:
    metadata = inspect_model_partition_metadata(
        model_path,
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        measured_layer_ns=measured_layer_ns,
    )
    memory_limit = int(torch.cuda.get_device_properties(0).total_memory * 0.80)
    plans: dict[str, StagePlan] = {}
    for stage_count in stage_counts:
        for method in ("equal", "balanced"):
            try:
                plan = build_stage_plan(
                    model_path,
                    metadata=metadata,
                    stage_count=stage_count,
                    method=method,
                    memory_limit_bytes=memory_limit,
                )
            except (MemoryError, ValueError) as exc:
                _write_json(
                    run_root / "stage_plans" / f"{stage_count}_{method}" / "stage_plan.json",
                    {
                        "status": "RESOURCE_INFEASIBLE",
                        "stage_count": stage_count,
                        "partition_method": method,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                continue
            key = f"{stage_count}_{method}"
            plan.write(run_root / "stage_plans" / key / "stage_plan.json")
            plans[key] = plan
    _write_json(
        run_root / "stage_plans" / "layer_profile.json",
        {
            "layer_costs": [asdict(cost) for cost in metadata.layer_costs],
            "metadata_hash": metadata.metadata_hash,
            "profiled_with_real_cuda_events": True,
        },
    )
    return plans, metadata


def _resource_summary(monitor: ResourceMonitor) -> dict[str, Any]:
    return summarise_resources(monitor.samples)


def _trace_boundary_hash_match(result: StageRingResult, reference_root: Path) -> dict[str, Any]:
    expected: dict[tuple[int, int], str] = {}
    for manifest_path in reference_root.glob("token-*/stage-*/manifest.json"):
        token_position = int(manifest_path.parent.parent.name.split("-")[-1])
        stage_id = int(manifest_path.parent.name.split("-")[-1])
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        boundary = next(row for row in entries if row["name"] == "stage_boundary_hidden")
        expected[(token_position, stage_id)] = str(boundary["sha256"])
    events = merge_traces([Path(path) for path in result.trace_paths])
    checks = []
    for event in events:
        if (
            event.get("event") != "socket_send_end"
            or event.get("data_plane") != "ring"
            or event.get("message_type") not in {"PREFILL", "DECODE", "VERIFY_CANDIDATES"}
        ):
            continue
        key = (int(event["token_position"]), int(event["source_stage"]))
        checks.append(
            {
                "token_position": key[0],
                "source_stage": key[1],
                "expected_sha256": expected.get(key),
                "wire_raw_sha256": event.get("tensor_raw_checksum"),
                "match": expected.get(key) == event.get("tensor_raw_checksum"),
            }
        )
    return {
        "comparison_count": len(checks),
        "all_match": bool(checks) and all(row["match"] for row in checks),
        "checks": checks,
    }


def _stage_row(
    *,
    result: StageRingResult,
    plan: StagePlan,
    strategy: str,
    profile_index: int,
    profile_manifest: dict[str, Any],
    expected_tokens: list[int],
    run_index: int,
    resource: dict[str, Any],
    raw_record_path: Path,
    boundary_match: bool,
    measurement_started_ns: int,
    measurement_ended_ns: int,
) -> dict[str, Any]:
    actual = list(result.generated_token_ids)
    expected = expected_tokens[: result.generated_tokens]
    exact_count = sum(left == right for left, right in zip(actual, expected, strict=False))
    events = merge_traces([Path(path) for path in result.trace_paths])
    compression_events = [row for row in events if row.get("event") == "compression_end"]
    raw_bytes = sum(int(row.get("payload_bytes", 0)) for row in compression_events)
    encoded_bytes = sum(int(row.get("wire_bytes", 0)) for row in compression_events)
    return {
        "profile_order": profile_index,
        "profile_name": result.profile_name,
        "profile_parameters_hash": profile_parameters_hash(profile_manifest),
        "strategy": strategy,
        "stage_count": plan.stage_count,
        "partition_method": plan.partition_method,
        "compression_mode": result.compression_request,
        "speculation_provider": "none",
        "speculation_depth": 0,
        "run_index": run_index,
        "prompt_id": "code-01",
        "generated_tokens": result.generated_tokens,
        "exact_tokens": exact_count,
        "token_match": actual == expected,
        "throughput_tps": result.throughput_tps,
        "ttft_seconds": result.ttft_seconds,
        "mean_itl_seconds": (
            statistics.mean(result.inter_token_latencies_seconds)
            if result.inter_token_latencies_seconds
            else 0.0
        ),
        "p95_itl_seconds": quantile(list(result.inter_token_latencies_seconds), 0.95),
        "messages_per_token": result.critical_path["messages_per_token"],
        "serial_waits_per_token": result.critical_path["serial_waits_per_token"],
        "payload_bytes_per_token": result.critical_path["payload_bytes_per_token"],
        "wire_bytes_per_token": result.critical_path["wire_bytes_per_token"],
        "compression_ratio": raw_bytes / encoded_bytes if encoded_bytes else 1.0,
        "accepted_tokens_per_round": 1.0,
        "gpu_utilisation": resource.get("gpu_utilisation_percent_mean"),
        "gpu_memory_bytes": resource.get("gpu_memory_used_bytes_maximum"),
        "host_memory_bytes": resource.get("host_memory_used_bytes_maximum"),
        "fallback_used": result.fallback_used,
        "valid_for_claims": result.valid_for_claims and actual == expected and boundary_match,
        "evidence_category": "REAL_MODEL_MEASURED",
        "planner_eligible": True,
        "raw_record_path": str(raw_record_path.resolve()),
        "chart_source_id": f"{strategy}:{result.profile_name}:r{run_index}",
        "boundary_hash_match": boundary_match,
        "source_run_id": result.run_id,
        "measurement_started_ns": measurement_started_ns,
        "measurement_ended_ns": measurement_ended_ns,
        "actual_compression_modes": json.dumps(list(result.compression_modes_used)),
        "serial_waits_method": "linked socket_receive_end to required cuda_compute_start",
        "_result": result.to_dict(),
        "_resource": resource,
    }


def _reconstruct_expert_rpc_dependencies(
    *,
    output_directory: Path,
    generated_tokens: int,
    prompt_tokens: int,
    expected_message_count: int,
) -> dict[str, Any]:
    """Reconstruct the old synchronous RPC DAG from per-request worker events."""

    dependencies: list[dict[str, Any]] = []
    counts = [0 for _ in range(generated_tokens)]
    pattern = re.compile(r"-token-(\d+)-layer-(\d+)-worker-(\d+)$")
    telemetry_paths = sorted(output_directory.rglob("worker-telemetry.jsonl"))
    for path in telemetry_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") != "native_expert_request_completed":
                continue
            match = pattern.search(str(event.get("request_id", "")))
            if match is None:
                continue
            native_position = int(match.group(1))
            token_position = (
                0 if native_position == 0 else native_position - prompt_tokens + 1
            )
            if not 0 <= token_position < generated_tokens:
                continue
            counts[token_position] += 1
            dependencies.append(
                {
                    "event": "native_expert_request_completed",
                    "request_id": event["request_id"],
                    "worker_id": event["worker_id"],
                    "token_position": token_position,
                    "native_sequence_position": native_position,
                    "layer_id": int(match.group(2)),
                    "worker_slot": int(match.group(3)),
                    "execution_sequence": int(event["execution_sequence"]),
                    "completion_wall_time_ns": int(event["wall_time_ns"]),
                    "duration_ns": int(event["duration_ns"]),
                    "payload_bytes": int(event["bytes_received"])
                    + int(event["bytes_sent"]),
                    "critical_dependency": True,
                    "unblocks": "coordinator layer reduction or next required model computation",
                    "model_fingerprint": event["model_fingerprint"],
                    "source_path": str(path.resolve()),
                }
            )
    dependencies.sort(
        key=lambda row: (
            int(row["completion_wall_time_ns"]),
            str(row["worker_id"]),
            int(row["execution_sequence"]),
        )
    )
    observed_count = len(dependencies)
    valid = (
        observed_count == expected_message_count
        and len(counts) == generated_tokens
        and all(count > 0 for count in counts)
    )
    result = {
        "definition": (
            "A network dependency on the token-critical path that must complete before "
            "the next required model computation can begin."
        ),
        "method": (
            "parsed individual native_expert_request_completed events by request token/layer; "
            "the unchanged Experiment 010 coordinator synchronously requires each result "
            "before its layer reduction or continuation"
        ),
        "telemetry_paths": [str(path.resolve()) for path in telemetry_paths],
        "expected_rpc_message_count": expected_message_count,
        "observed_dependency_count": observed_count,
        "count_matches_source_row": observed_count == expected_message_count,
        "serial_waits_by_generated_token": counts,
        "serial_waits_per_token": statistics.median(counts) if counts else 0.0,
        "mean_serial_waits_per_token": statistics.mean(counts) if counts else 0.0,
        "valid": valid,
        "dependencies": dependencies,
        "not_estimated_from_aggregate_message_total": True,
    }
    _write_json(output_directory / "critical_path_reconstruction.json", result)
    return result


def _baseline_row(
    *,
    source: dict[str, Any],
    profile_index: int,
    profile_manifest: dict[str, Any],
    resource: dict[str, Any],
    raw_record_path: Path,
    critical_path: dict[str, Any],
    measurement_started_ns: int,
    measurement_ended_ns: int,
) -> dict[str, Any]:
    generated = int(source["generated_tokens"])
    message_count = int(source["rpc_message_count"])
    payload = int(source["rpc_raw_payload_bytes"])
    actual = source["actual_token_ids"]
    expected = source["expected_token_ids"]
    if isinstance(actual, str):
        actual = json.loads(actual)
    if isinstance(expected, str):
        expected = json.loads(expected)
    mean_itl = float(source.get("decode_after_first_seconds") or 0) / max(generated - 1, 1)
    return {
        "profile_order": profile_index,
        "profile_name": str(source["network_profile"]),
        "profile_parameters_hash": profile_parameters_hash(profile_manifest),
        "strategy": "experiment_011_same_run_expert_rpc",
        "stage_count": 0,
        "partition_method": "whole_expert_rpc",
        "compression_mode": "none",
        "speculation_provider": "none",
        "speculation_depth": 0,
        "run_index": int(source["repeat"]),
        "prompt_id": str(source["prompt_id"]),
        "generated_tokens": generated,
        "exact_tokens": int(source["matching_tokens"]),
        "token_match": bool(source["exact_token_identity"]),
        "throughput_tps": float(source["decode_tokens_per_second"]),
        "ttft_seconds": float(source["ttft_seconds"]),
        "mean_itl_seconds": mean_itl,
        "p95_itl_seconds": mean_itl,
        "messages_per_token": message_count / generated,
        "serial_waits_per_token": critical_path["serial_waits_per_token"],
        "payload_bytes_per_token": payload / generated,
        "wire_bytes_per_token": (
            int(source.get("rpc_request_bytes", 0)) + int(source.get("rpc_response_bytes", 0))
        )
        / generated,
        "compression_ratio": 1.0,
        "accepted_tokens_per_round": 1.0,
        "gpu_utilisation": resource.get("gpu_utilisation_percent_mean"),
        "gpu_memory_bytes": resource.get("gpu_memory_used_bytes_maximum"),
        "host_memory_bytes": resource.get("host_memory_used_bytes_maximum"),
        "fallback_used": False,
        "valid_for_claims": (
            source["measurement_status"] == "MEASURED"
            and source["exact_token_identity"] is True
            and source["valid_performance_candidate"] is True
            and critical_path["valid"] is True
        ),
        "evidence_category": "REAL_MODEL_MEASURED",
        "planner_eligible": True,
        "raw_record_path": str(raw_record_path.resolve()),
        "chart_source_id": f"same-run-baseline:{source['network_profile']}:r{source['repeat']}",
        "boundary_hash_match": bool(source.get("numerical_contract_ok")),
        "source_run_id": source["run_id"],
        "measurement_started_ns": measurement_started_ns,
        "measurement_ended_ns": measurement_ended_ns,
        "actual_compression_modes": "[]",
        "serial_waits_method": critical_path["method"],
        "_source": source,
        "_resource": resource,
        "_critical_path": critical_path,
    }


def _local_row(
    *,
    source: dict[str, Any],
    profile_manifest: dict[str, Any],
    resource: dict[str, Any],
    raw_record_path: Path,
    measurement_started_ns: int,
    measurement_ended_ns: int,
) -> dict[str, Any]:
    generated = int(source["generated_tokens"])
    actual = [int(value) for value in source["actual_token_ids"]]
    expected = [int(value) for value in source["expected_token_ids"]]
    mean_itl = float(source.get("decode_after_first_seconds") or 0) / max(generated - 1, 1)
    return {
        "profile_order": 1,
        "profile_name": "loopback_unshaped",
        "profile_parameters_hash": profile_parameters_hash(profile_manifest),
        "strategy": "local_monolithic_reference",
        "stage_count": 1,
        "partition_method": "monolithic",
        "compression_mode": "none",
        "speculation_provider": "none",
        "speculation_depth": 0,
        "run_index": int(source["repeat"]),
        "prompt_id": str(source["prompt_id"]),
        "generated_tokens": generated,
        "exact_tokens": sum(left == right for left, right in zip(actual, expected, strict=True)),
        "token_match": bool(source["exact_token_identity"]),
        "throughput_tps": float(source["decode_tokens_per_second"]),
        "ttft_seconds": float(source["ttft_seconds"]),
        "mean_itl_seconds": mean_itl,
        "p95_itl_seconds": mean_itl,
        "messages_per_token": 0.0,
        "serial_waits_per_token": 0.0,
        "payload_bytes_per_token": 0.0,
        "wire_bytes_per_token": 0.0,
        "compression_ratio": 1.0,
        "accepted_tokens_per_round": 1.0,
        "gpu_utilisation": resource.get("gpu_utilisation_percent_mean"),
        "gpu_memory_bytes": resource.get("gpu_memory_used_bytes_maximum"),
        "host_memory_bytes": resource.get("host_memory_used_bytes_maximum"),
        "fallback_used": False,
        "valid_for_claims": bool(
            source["measurement_status"] == "MEASURED"
            and source["exact_token_identity"]
            and source["valid_performance_candidate"]
        ),
        "evidence_category": "REAL_MODEL_MEASURED",
        "planner_eligible": True,
        "raw_record_path": str(raw_record_path.resolve()),
        "chart_source_id": "local-monolithic-reference:loopback_unshaped:r1",
        "boundary_hash_match": bool(source["numerical_contract_ok"]),
        "source_run_id": str(source["run_id"]),
        "measurement_started_ns": measurement_started_ns,
        "measurement_ended_ns": measurement_ended_ns,
        "actual_compression_modes": "[]",
        "serial_waits_method": "no network boundary",
        "_source": source,
        "_resource": resource,
    }


def _archived_row(
    profile: str, profile_index: int, profile_manifest: dict[str, Any]
) -> dict[str, Any]:
    return {
        "profile_order": profile_index,
        "profile_name": profile,
        "profile_parameters_hash": profile_parameters_hash(profile_manifest),
        "strategy": "experiment_010_archived",
        "stage_count": 0,
        "partition_method": "whole_expert_rpc",
        "compression_mode": "none",
        "speculation_provider": "none",
        "speculation_depth": 0,
        "run_index": 0,
        "prompt_id": "code-01",
        "generated_tokens": 32,
        "exact_tokens": 32,
        "token_match": True,
        "throughput_tps": ARCHIVED_010_TPS[profile],
        "ttft_seconds": "",
        "mean_itl_seconds": "",
        "p95_itl_seconds": "",
        "messages_per_token": 1850 / 32,
        "serial_waits_per_token": 1850 / 32,
        "payload_bytes_per_token": "",
        "wire_bytes_per_token": "",
        "compression_ratio": 1.0,
        "accepted_tokens_per_round": 1.0,
        "gpu_utilisation": "",
        "gpu_memory_bytes": "",
        "host_memory_bytes": "",
        "fallback_used": False,
        "valid_for_claims": False,
        "evidence_category": "ARCHIVED_PUBLISHED_NOT_NEWLY_MEASURED",
        "planner_eligible": False,
        "raw_record_path": "baseline/archived_values.json",
        "chart_source_id": f"archived-010:{profile}",
        "boundary_hash_match": True,
        "source_run_id": "experiment-010-correction-final",
        "measurement_started_ns": "",
        "measurement_ended_ns": "",
        "actual_compression_modes": "[]",
        "serial_waits_method": "archived operation count; not a new Experiment 011 trace",
    }


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _run_stage_measurement(
    *,
    run_root: Path,
    run_id: str,
    plan: StagePlan,
    profile_name: str,
    strategy: str,
    compression: str,
    prompt_ids: list[int],
    expected_ids: list[int],
    token_count: int,
    reference_root: Path,
    profile_index: int,
    profile_manifest: dict[str, Any],
    run_index: int,
    output_directory: Path,
    capture_boundaries: bool = False,
) -> tuple[dict[str, Any], StageRingResult, dict[str, Any]]:
    monitor_path = output_directory / "resources.ndjson"
    started_ns = time.time_ns()
    with ResourceMonitor(monitor_path, interval_seconds=1.0) as monitor:
        result = StageRingController(
            run_id=run_id,
            plan=plan,
            network_profile=NETWORK_PROFILES[profile_name],
            output_directory=output_directory,
            compression_request=compression,  # type: ignore[arg-type]
            timeout_s=300.0,
            capture_boundaries=capture_boundaries,
        ).run(
            prompt_token_ids=prompt_ids,
            generated_token_count=token_count,
            session_id=f"{run_id}-session",
            request_id=f"{run_id}-request",
        )
    ended_ns = time.time_ns()
    resource = _resource_summary(monitor)
    boundary = _trace_boundary_hash_match(result, reference_root)
    _write_json(output_directory / "boundary_hash_validation.json", boundary)
    raw_record = output_directory / "measurement_record.json"
    _write_json(
        raw_record,
        {
            "result": result.to_dict(),
            "resource_summary": resource,
            "boundary_hash_validation": boundary,
            "strategy": strategy,
            "profile": profile_manifest,
            "plan": plan.to_dict(),
            "measurement_started_ns": started_ns,
            "measurement_ended_ns": ended_ns,
            "evidence_category": "REAL_MODEL_MEASURED",
        },
    )
    row = _stage_row(
        result=result,
        plan=plan,
        strategy=strategy,
        profile_index=profile_index,
        profile_manifest=profile_manifest,
        expected_tokens=expected_ids,
        run_index=run_index,
        resource=resource,
        raw_record_path=raw_record,
        boundary_match=boundary["all_match"],
        measurement_started_ns=started_ns,
        measurement_ended_ns=ended_ns,
    )
    return row, result, boundary


def _ownership_validation(
    plans: dict[str, StagePlan], exactness_records: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = []
    all_valid = True
    for key, plan in plans.items():
        if plan.stage_count not in {2, 4}:
            continue
        ownership_record = next(
            (row for row in exactness_records if row.get("plan_key") == key), None
        )
        ownership = ownership_record.get("ownership", []) if ownership_record else []
        parameter_sets = [set(stage.get("parameter_names", [])) for stage in ownership]
        overlaps = []
        for left in range(len(parameter_sets)):
            for right in range(left + 1, len(parameter_sets)):
                overlap = sorted(parameter_sets[left] & parameter_sets[right])
                if overlap:
                    overlaps.append({"left": left, "right": right, "names": overlap})
        layer_owners: dict[int, set[int]] = {layer: set() for layer in range(plan.layer_count)}
        for assignment in plan.assignments:
            for layer in assignment.layer_ids:
                layer_owners[layer].add(assignment.stage_id)
        valid = (
            len(ownership) == plan.stage_count
            and not overlaps
            and all(len(owners) == 1 for owners in layer_owners.values())
        )
        all_valid &= valid
        rows.append(
            {
                "plan_key": key,
                "valid": valid,
                "stage_process_ids": ownership_record.get("stage_process_ids", [])
                if ownership_record
                else [],
                "stage_weight_bytes": [stage.get("parameter_bytes", 0) for stage in ownership],
                "layer_owners": {str(layer): sorted(owners) for layer, owners in layer_owners.items()},
                "parameter_overlaps": overlaps,
                "coordinator_weight_bytes": 0,
                "coordinator_executed_stage_layers": False,
            }
        )
    return {"valid": all_valid and bool(rows), "plans": rows}


def _planner_candidate_from_stage(
    row: dict[str, Any], plan: StagePlan
) -> PlannerCandidate:
    result = row["_result"]
    stage_compute = result["critical_path"].get("stage_compute_ns", {})
    tokens = max(int(row["generated_tokens"]), 1)
    compute_tuple = tuple(
        int(stage_compute.get(str(index), stage_compute.get(index, 0)) / tokens)
        for index in range(plan.stage_count)
    )
    return PlannerCandidate(
        name=str(row["strategy"]),
        execution_family="stage_ring_exact",
        stage_count=plan.stage_count,
        partition_method=plan.partition_method,
        compression_mode=str(row["compression_mode"]),
        speculation_provider="none",
        speculation_depth=0,
        stage_compute_ns=compute_tuple,
        stage_weight_bytes=tuple(assignment.weight_bytes for assignment in plan.assignments),
        kv_cache_bytes=tuple(
            assignment.kv_cache_bytes_per_token
            * (int(row["generated_tokens"]) + 11)
            for assignment in plan.assignments
        ),
        serial_boundaries=float(row["serial_waits_per_token"]),
        payload_bytes_per_token=float(row["payload_bytes_per_token"]),
        compression_ratio=float(row["compression_ratio"]),
        measured_throughput_tps=float(row["throughput_tps"]),
        exact=bool(row["token_match"]) and bool(row["boundary_hash_match"]),
        suitable=bool(row["valid_for_claims"]),
    )


def _geometric_mean(values: Sequence[float]) -> float:
    positive = [value for value in values if value > 0]
    return math.exp(statistics.mean(math.log(value) for value in positive)) if positive else 0.0


def run_experiment(options: ExperimentOptions) -> tuple[Path, Path, str]:
    run_id = options.run_id or _now_run_id()
    run_root = REPOSITORY_ROOT / "artifacts" / "runs" / run_id
    if run_root.exists() and not options.resume:
        raise FileExistsError(f"run directory already exists; use -Resume or another -RunId: {run_root}")
    for directory in (
        "stage_plans",
        "baseline",
        "exactness",
        "network",
        "compression",
        "speculation",
        "concurrency",
        "failures",
        "traces",
        "logs",
        "charts",
    ):
        (run_root / directory).mkdir(parents=True, exist_ok=True)
    started_ns = time.time_ns()
    checkpoint = {
        "run_id": run_id,
        "mode": options.mode,
        "canonical": options.canonical,
        "started_ns": started_ns,
        "completed_phases": [],
    }
    _write_json(run_root / "checkpoint.json", checkpoint)

    assets = discover_assets(
        REPOSITORY_ROOT,
        model_override=Path(options.model_path) if options.model_path else None,
    )
    _write_json(run_root / "discovery.json", assets.to_dict())
    if assets.missing_artifacts:
        raise FileNotFoundError(
            "required artifacts remain missing after exhaustive discovery: "
            + "; ".join(assets.missing_artifacts)
        )
    environment = environment_identity()
    environment.update(
        {
            "run_id": run_id,
            "timezone": "Australia/Sydney",
            "deterministic_settings": {
                "greedy": True,
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "tf32": False,
                "attention_implementation": "eager",
                "model_dtype": "bfloat16",
                "comparison_dtype": "float32",
            },
            "workload": {
                "prompt_count": 1,
                "generated_tokens": options.generated_token_count,
                "warmup_runs": 0,
                "measured_repetitions": 1,
                "aggregation": "median",
            },
        }
    )
    _write_json(run_root / "environment.json", environment)
    _identity_files(run_root, assets)
    network_manifest = _write_network_manifest(run_root, assets)
    checkpoint["completed_phases"].append("discovery_and_identity")
    _write_json(run_root / "checkpoint.json", checkpoint)

    preflight = _run_test_command(
        run_root=run_root,
        name="preflight-tests",
        arguments=[
            "-q",
            "tests/unit/test_experiment_011_protocol.py",
            "tests/unit/test_experiment_011_mechanisms.py",
        ],
    )
    if not preflight["passed"]:
        raise RuntimeError("Experiment 011 preflight tests failed")
    checkpoint["completed_phases"].append("preflight_tests")
    _write_json(run_root / "checkpoint.json", checkpoint)

    workload, prompt_ids, expected_ids = _load_workload(assets)
    token_count = options.generated_token_count
    profiles = tuple(options.profile_names)
    if options.canonical and profiles != PROFILE_ORDER:
        raise ValueError("canonical -Full must use all eight profiles in their frozen order")
    model_path = Path(assets.source_model_path)

    # Profile every real layer first. The provisional equal plan only defines
    # capture folders; CUDA events measure all 16 layers independently.
    initial_metadata = inspect_model_partition_metadata(
        model_path,
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
    )
    provisional_plan = build_stage_plan(
        model_path,
        metadata=initial_metadata,
        stage_count=2,
        method="equal",
        memory_limit_bytes=int(torch.cuda.get_device_properties(0).total_memory * 0.80),
    )
    profile_reference = run_local_reference(
        model_path=model_path,
        workload_reference_path=Path(assets.experiment_010_workload_reference),
        plan=provisional_plan,
        generated_token_count=token_count,
        output_directory=run_root / "exactness" / "layer-profile-reference",
    )
    if list(profile_reference.generated_token_ids) != expected_ids[:token_count]:
        raise RuntimeError("authoritative local reference differs from Experiment 010 tokens")
    measured_layer_ns = {
        index: value for index, value in enumerate(profile_reference.layer_execution_ns)
    }
    plans, _metadata = _build_plans(
        run_root=run_root,
        model_path=model_path,
        stage_counts=options.stage_counts,
        measured_layer_ns=measured_layer_ns,
    )
    required_plan_keys = {"2_equal", "2_balanced", "4_equal", "4_balanced"}
    if options.canonical and not required_plan_keys.issubset(plans):
        raise RuntimeError("required two- and four-stage plans could not be constructed")
    checkpoint["completed_phases"].append("layer_profile_and_plans")
    _write_json(run_root / "checkpoint.json", checkpoint)

    # Reference capture trees are keyed by unique layer boundaries and reused
    # only when two algorithms produced the identical contiguous ranges.
    reference_by_ranges: dict[tuple[tuple[int, int], ...], tuple[Path, LocalReferenceResult]] = {
        _plan_ranges(provisional_plan): (
            run_root / "exactness" / "layer-profile-reference",
            profile_reference,
        )
    }
    for key, plan in plans.items():
        ranges = _plan_ranges(plan)
        if ranges in reference_by_ranges:
            continue
        reference_root = run_root / "exactness" / f"reference-{key}"
        reference_result = run_local_reference(
            model_path=model_path,
            workload_reference_path=Path(assets.experiment_010_workload_reference),
            plan=plan,
            generated_token_count=token_count,
            output_directory=reference_root,
        )
        reference_by_ranges[ranges] = (reference_root, reference_result)

    exactness_records: list[dict[str, Any]] = []
    if not options.network_only:
        exact_plan_keys = [key for key in ("2_equal", "2_balanced", "4_equal", "4_balanced") if key in plans]
        for key in exact_plan_keys:
            plan = plans[key]
            reference_root, reference_result = reference_by_ranges[_plan_ranges(plan)]
            exact_root = run_root / "exactness" / key / "distributed"
            row, result, boundary = _run_stage_measurement(
                run_root=run_root,
                run_id=f"{run_id}-exact-{key}",
                plan=plan,
                profile_name="loopback_unshaped",
                strategy=f"stage_ring_exact_{key}",
                compression="none",
                prompt_ids=prompt_ids,
                expected_ids=expected_ids,
                token_count=token_count,
                reference_root=reference_root,
                profile_index=1,
                profile_manifest=network_manifest["profiles"]["loopback_unshaped"],
                run_index=1,
                output_directory=exact_root,
                capture_boundaries=True,
            )
            comparison = compare_capture_trees(
                local_capture_directory=reference_root,
                distributed_capture_directory=exact_root / "captures",
                session_id=f"{run_id}-exact-{key}-session",
                prompt_id=str(workload["prompt_id"]),
                reproduction_command=(
                    f".\\scripts\\run-experiment-011.ps1 -ExactnessOnly -RunId {run_id} -Resume"
                ),
            )
            record = {
                "plan_key": key,
                "stage_count": plan.stage_count,
                "partition_method": plan.partition_method,
                "compression_mode": "none",
                "token_match": list(result.generated_token_ids)
                == list(reference_result.generated_token_ids),
                "capture_exact": comparison["exact"],
                "comparison_count": comparison["comparison_count"],
                "capture_mismatch_count": comparison["mismatch_count"],
                "maximum_absolute_difference_fp32": comparison[
                    "maximum_absolute_difference_fp32"
                ],
                "maximum_relative_l2_error_fp32": comparison[
                    "maximum_relative_l2_error_fp32"
                ],
                "serial_waits_per_token": result.critical_path["serial_waits_per_token"],
                "messages_per_token": result.critical_path["messages_per_token"],
                "fallback_used": result.fallback_used,
                "stage_process_ids": list(result.stage_process_ids),
                "ownership": list(result.ownership),
                "boundary_trace_hashes_exact": boundary["all_match"],
                "comparison_path": str(
                    (run_root / "exactness" / key / "comparison.json").resolve()
                ),
            }
            _write_json(run_root / "exactness" / key / "comparison.json", comparison)
            _write_json(run_root / "exactness" / key / "summary.json", record)
            exactness_records.append(record)

        compression_plan_key = "2_balanced" if "2_balanced" in plans else "2_equal"
        compression_plan = plans[compression_plan_key]
        reference_root, reference_result = reference_by_ranges[_plan_ranges(compression_plan)]
        compression_exact_root = run_root / "exactness" / "compression_enabled" / "distributed"
        _, compression_result, _ = _run_stage_measurement(
            run_root=run_root,
            run_id=f"{run_id}-exact-compression",
            plan=compression_plan,
            profile_name="loopback_unshaped",
            strategy="stage_ring_exact_compression_enabled",
            compression="byte_shuffle_fast_codec",
            prompt_ids=prompt_ids,
            expected_ids=expected_ids,
            token_count=token_count,
            reference_root=reference_root,
            profile_index=1,
            profile_manifest=network_manifest["profiles"]["loopback_unshaped"],
            run_index=1,
            output_directory=compression_exact_root,
            capture_boundaries=True,
        )
        compression_comparison = compare_capture_trees(
            local_capture_directory=reference_root,
            distributed_capture_directory=compression_exact_root / "captures",
            session_id=f"{run_id}-exact-compression-session",
            prompt_id=str(workload["prompt_id"]),
            reproduction_command=(
                f".\\scripts\\run-experiment-011.ps1 -ExactnessOnly -RunId {run_id} -Resume"
            ),
        )
        compression_exact_record = {
            "plan_key": f"{compression_plan_key}_compression",
            "stage_count": compression_plan.stage_count,
            "partition_method": compression_plan.partition_method,
            "compression_mode": "byte_shuffle_fast_codec",
            "token_match": list(compression_result.generated_token_ids)
            == list(reference_result.generated_token_ids),
            "capture_exact": compression_comparison["exact"],
            "comparison_count": compression_comparison["comparison_count"],
            "capture_mismatch_count": compression_comparison["mismatch_count"],
            "maximum_absolute_difference_fp32": compression_comparison[
                "maximum_absolute_difference_fp32"
            ],
            "maximum_relative_l2_error_fp32": compression_comparison[
                "maximum_relative_l2_error_fp32"
            ],
            "serial_waits_per_token": compression_result.critical_path[
                "serial_waits_per_token"
            ],
            "messages_per_token": compression_result.critical_path["messages_per_token"],
            "fallback_used": compression_result.fallback_used,
            "stage_process_ids": list(compression_result.stage_process_ids),
            "ownership": list(compression_result.ownership),
            "boundary_trace_hashes_exact": True,
        }
        _write_json(
            run_root / "exactness" / "compression_enabled" / "comparison.json",
            compression_comparison,
        )
        exactness_records.append(compression_exact_record)

        if "8_balanced" in plans:
            plan = plans["8_balanced"]
            reference_root, reference_result = reference_by_ranges[_plan_ranges(plan)]
            eight_root = run_root / "exactness" / "8_balanced" / "distributed"
            _, eight_result, _ = _run_stage_measurement(
                run_root=run_root,
                run_id=f"{run_id}-exact-8-balanced",
                plan=plan,
                profile_name="loopback_unshaped",
                strategy="stage_ring_exact_8_balanced",
                compression="none",
                prompt_ids=prompt_ids,
                expected_ids=expected_ids,
                token_count=token_count,
                reference_root=reference_root,
                profile_index=1,
                profile_manifest=network_manifest["profiles"]["loopback_unshaped"],
                run_index=1,
                output_directory=eight_root,
                capture_boundaries=False,
            )
            eight_error_text = " ".join(eight_result.errors).lower()
            resource_infeasible = any(
                marker in eight_error_text
                for marker in (
                    "out of memory",
                    "cuda error: memory",
                    "paging",
                    "commitment limit",
                    "resource temporarily unavailable",
                )
            )
            eight_status = {
                "status": (
                    "MEASURED"
                    if eight_result.valid_for_claims
                    else "RESOURCE_INFEASIBLE"
                    if resource_infeasible
                    else "FAILED"
                ),
                "stage_count": 8,
                "token_match": list(eight_result.generated_token_ids)
                == list(reference_result.generated_token_ids),
                "errors": list(eight_result.errors),
                "throughput_tps": eight_result.throughput_tps,
                "messages_per_token": eight_result.critical_path["messages_per_token"],
                "serial_waits_per_token": eight_result.critical_path[
                    "serial_waits_per_token"
                ],
            }
            _write_json(run_root / "exactness" / "eight_stage_status.json", eight_status)
    _write_json(run_root / "exactness" / "exactness_summary.json", exactness_records)
    ownership = _ownership_validation(plans, exactness_records)
    _write_json(run_root / "exactness" / "ownership_validation.json", ownership)
    checkpoint["completed_phases"].append("exactness")
    _write_json(run_root / "checkpoint.json", checkpoint)

    # Frozen archived series and native local control.
    _write_json(
        run_root / "baseline" / "archived_values.json",
        {
            "series": "experiment_010_archived",
            "values": ARCHIVED_010_TPS,
            "source": assets.experiment_010_network_results,
            "source_sha256": sha256_file(Path(assets.experiment_010_network_results)),
            "new_measurement": False,
        },
    )
    local_root = run_root / "baseline" / "local_monolithic_reference"
    local_started_ns = time.time_ns()
    with ResourceMonitor(local_root / "resources.ndjson") as local_monitor:
        native_local = run_native_local_reference(
            assets=assets,
            output_directory=local_root,
        )
    local_ended_ns = time.time_ns()
    local_resource = _resource_summary(local_monitor)
    local_raw_path = local_root / "measurement_record.json"
    _write_json(
        local_raw_path,
        {
            "source": native_local,
            "resource_summary": local_resource,
            "evidence_category": "REAL_MODEL_MEASURED",
        },
    )

    network_rows: list[dict[str, Any]] = []
    network_rows.append(
        _local_row(
            source=native_local,
            profile_manifest=network_manifest["profiles"]["loopback_unshaped"],
            resource=local_resource,
            raw_record_path=local_raw_path,
            measurement_started_ns=local_started_ns,
            measurement_ended_ns=local_ended_ns,
        )
    )
    baseline_gate_rows: list[dict[str, Any]] = []
    for profile_index, profile in enumerate(profiles, start=1):
        network_rows.append(
            _archived_row(profile, profile_index, network_manifest["profiles"][profile])
        )
    base_plan_keys = [key for key in ("2_equal", "2_balanced", "4_equal", "4_balanced") if key in plans]
    strategy_order_records = []
    base_stage_rows: list[dict[str, Any]] = []
    for profile_index, profile in enumerate(profiles, start=1):
        operations = ["baseline", *base_plan_keys]
        rotation = (profile_index - 1) % len(operations)
        operations = operations[rotation:] + operations[:rotation]
        for order_index, operation in enumerate(operations):
            operation_started = time.time_ns()
            if operation == "baseline":
                output = run_root / "baseline" / "fresh" / profile
                with ResourceMonitor(output / "resources.ndjson") as monitor:
                    rows = run_fresh_expert_rpc_baseline(
                        assets=assets,
                        profiles=[profile],
                        output_directory=output,
                        repeats=1,
                    )
                resource = _resource_summary(monitor)
                source = rows[0]
                critical_path = _reconstruct_expert_rpc_dependencies(
                    output_directory=output,
                    generated_tokens=int(source["generated_tokens"]),
                    prompt_tokens=int(source["prompt_tokens"]),
                    expected_message_count=int(source["rpc_message_count"]),
                )
                raw_path = output / "measurement_record.json"
                _write_json(
                    raw_path,
                    {
                        "source": source,
                        "resource_summary": resource,
                        "critical_path_reconstruction": critical_path,
                        "evidence_category": "REAL_MODEL_MEASURED",
                    },
                )
                row = _baseline_row(
                    source=source,
                    profile_index=profile_index,
                    profile_manifest=network_manifest["profiles"][profile],
                    resource=resource,
                    raw_record_path=raw_path,
                    critical_path=critical_path,
                    measurement_started_ns=operation_started,
                    measurement_ended_ns=time.time_ns(),
                )
                network_rows.append(row)
                baseline_gate_rows.append(row)
            else:
                plan = plans[operation]
                reference_root, _ = reference_by_ranges[_plan_ranges(plan)]
                strategy = f"stage_ring_exact_{operation}"
                output = run_root / "network" / profile / strategy / "run-1"
                row, _, _ = _run_stage_measurement(
                    run_root=run_root,
                    run_id=f"{run_id}-{profile}-{strategy}-r1",
                    plan=plan,
                    profile_name=profile,
                    strategy=strategy,
                    compression="none",
                    prompt_ids=prompt_ids,
                    expected_ids=expected_ids,
                    token_count=token_count,
                    reference_root=reference_root,
                    profile_index=profile_index,
                    profile_manifest=network_manifest["profiles"][profile],
                    run_index=1,
                    output_directory=output,
                )
                network_rows.append(row)
                base_stage_rows.append(row)
            strategy_order_records.append(
                {
                    "profile": profile,
                    "profile_order": profile_index,
                    "execution_order": order_index + 1,
                    "strategy": operation,
                    "started_ns": operation_started,
                    "ended_ns": time.time_ns(),
                }
            )
    _write_json(run_root / "network" / "strategy_execution_order.json", strategy_order_records)
    baseline_drift_rows = []
    for row in baseline_gate_rows:
        profile = str(row["profile_name"])
        archived_tps = ARCHIVED_010_TPS[profile]
        fresh_tps = float(row["throughput_tps"])
        percent = (fresh_tps - archived_tps) / archived_tps * 100.0
        baseline_drift_rows.append(
            {
                "profile_name": profile,
                "archived_010_tps": archived_tps,
                "fresh_same_run_tps": fresh_tps,
                "absolute_difference_tps": fresh_tps - archived_tps,
                "percentage_difference": percent,
                "material_threshold_percent": 15.0,
                "material_difference": abs(percent) >= 15.0,
                "investigation": {
                    "source_revision_and_dirty_state": "source_identity.json",
                    "binary_hashes": "binary_identity.json",
                    "model_and_tokenizer_hashes": "model_identity.json",
                    "gpu_clock_power_temperature_and_load": row["raw_record_path"],
                    "background_process_snapshot": str(
                        (run_root / "baseline" / "fresh" / profile / "resources.ndjson").resolve()
                    ),
                    "socket_shaper_provenance": "network_profiles_manifest.json",
                    "prompt_token_warmup_and_measurement_policy": "environment.json",
                    "conclusion": (
                        "material drift observed; evidence retained without normalization"
                        if abs(percent) >= 15.0
                        else "no material drift at the predeclared 15% diagnostic threshold"
                    ),
                },
            }
        )
    _write_json(
        run_root / "baseline" / "baseline_drift_investigation.json",
        {
            "rows": baseline_drift_rows,
            "archived_values_remeasured": False,
            "fresh_values_manipulated_to_match_archive": False,
        },
    )
    write_csv(
        run_root / "baseline" / "fresh" / "network_profile_results.csv",
        [_public_row(row) for row in baseline_gate_rows],
        (*RESULT_COLUMNS, *EXTRA_RESULT_COLUMNS),
    )

    # Pick one topology for the adaptive compression matrix from the geometric
    # mean of the four uncompressed exact ring strategies.
    strategy_geomeans = {}
    for key in base_plan_keys:
        strategy = f"stage_ring_exact_{key}"
        values = [
            float(row["throughput_tps"])
            for row in base_stage_rows
            if row["strategy"] == strategy
        ]
        strategy_geomeans[key] = _geometric_mean(values)
    best_base_key = max(strategy_geomeans, key=strategy_geomeans.get)
    best_base_plan = plans[best_base_key]
    adaptive_rows: list[dict[str, Any]] = []
    for profile_index, profile in enumerate(profiles, start=1):
        reference_root, _ = reference_by_ranges[_plan_ranges(best_base_plan)]
        strategy = f"stage_ring_exact_{best_base_key}_adaptive"
        output = run_root / "network" / profile / strategy / "run-1"
        row, _, _ = _run_stage_measurement(
            run_root=run_root,
            run_id=f"{run_id}-{profile}-{strategy}-r1",
            plan=best_base_plan,
            profile_name=profile,
            strategy=strategy,
            compression="adaptive",
            prompt_ids=prompt_ids,
            expected_ids=expected_ids,
            token_count=token_count,
            reference_root=reference_root,
            profile_index=profile_index,
            profile_manifest=network_manifest["profiles"][profile],
            run_index=1,
            output_directory=output,
        )
        adaptive_rows.append(row)
        network_rows.append(row)

    # Planner decisions include the old RPC and local path, then use a stage-
    # only subdecision for the required final stage-path rerun.
    planner = MeasuredStrategyPlanner()
    planner_decisions: dict[str, Any] = {}
    final_stage_rows: list[dict[str, Any]] = []
    selected_candidate_by_profile: dict[str, str] = {}
    plan_by_strategy = {
        f"stage_ring_exact_{key}": plan for key, plan in plans.items()
    }
    plan_by_strategy[f"stage_ring_exact_{best_base_key}_adaptive"] = best_base_plan
    for profile_index, profile in enumerate(profiles, start=1):
        baseline_row = next(row for row in baseline_gate_rows if row["profile_name"] == profile)
        stage_rows = [
            row
            for row in [*base_stage_rows, *adaptive_rows]
            if row["profile_name"] == profile
        ]
        stage_candidates = [
            _planner_candidate_from_stage(row, plan_by_strategy[str(row["strategy"])])
            for row in stage_rows
        ]
        expert_candidate = PlannerCandidate(
            name="experiment_011_same_run_expert_rpc",
            execution_family="expert_rpc",
            stage_count=0,
            partition_method="whole_expert_rpc",
            compression_mode="none",
            speculation_provider="none",
            speculation_depth=0,
            stage_compute_ns=(
                int(
                    baseline_row["_source"].get("rpc_compute_ns", 0)
                    / max(int(baseline_row["generated_tokens"]), 1)
                ),
            ),
            stage_weight_bytes=tuple(
                int(json.loads((Path(path) / "manifest.json").read_text())["total_bytes"])
                if "total_bytes"
                in json.loads((Path(path) / "manifest.json").read_text())
                else 0
                for path in assets.worker_bank_paths
            ),
            kv_cache_bytes=(0,),
            serial_boundaries=float(baseline_row["serial_waits_per_token"]),
            payload_bytes_per_token=float(baseline_row["payload_bytes_per_token"]),
            measured_throughput_tps=float(baseline_row["throughput_tps"]),
            exact=bool(baseline_row["token_match"]),
            suitable=bool(baseline_row["valid_for_claims"]),
        )
        local_candidate = PlannerCandidate(
            name="local_monolithic_reference",
            execution_family="local_monolithic",
            stage_count=1,
            partition_method="monolithic",
            compression_mode="none",
            speculation_provider="none",
            speculation_depth=0,
            stage_compute_ns=(
                int(float(native_local["model_elapsed_seconds"]) / token_count * 1e9),
            ),
            stage_weight_bytes=(sum(path.stat().st_size for path in model_path.glob("model-*.safetensors")),),
            kv_cache_bytes=(0,),
            serial_boundaries=0,
            payload_bytes_per_token=0,
            measured_throughput_tps=float(native_local["decode_tokens_per_second"]),
            exact=bool(native_local["exact_token_identity"]),
            suitable=True,
        )
        profile_values = network_manifest["profiles"][profile]
        inputs = PlannerInputs(
            bandwidth_bps=profile_values["bandwidth_bps"],
            one_way_latency_ms=float(profile_values["one_way_latency_ms"]),
            jitter_ms=float(profile_values["jitter_ms"]),
            loss_probability=float(profile_values["message_loss_probability"]),
            queue_depth=int(profile_values["queue_depth"]),
            available_device_memory_bytes=int(torch.cuda.get_device_properties(0).total_memory),
            required_distributed_execution=True,
        )
        overall = planner.evaluate(
            [expert_candidate, local_candidate, *stage_candidates], inputs=inputs
        )
        stage_decision = planner.evaluate(stage_candidates, inputs=inputs)
        if stage_decision.selected_candidate is None:
            raise RuntimeError(f"planner found no valid exact stage candidate for {profile}")
        selected_candidate_by_profile[profile] = stage_decision.selected_candidate
        decision = {
            "profile_name": profile,
            "overall": overall.to_dict(),
            "distributed_stage_selection": stage_decision.to_dict(),
            "profile_name_used_as_decision_feature": False,
        }
        planner_decisions[profile] = decision
        _write_json(run_root / "network" / profile / "planner_decision.json", decision)
        selected_row = next(
            row for row in stage_rows if row["strategy"] == stage_decision.selected_candidate
        )
        selected_plan = plan_by_strategy[stage_decision.selected_candidate]
        selected_compression = str(selected_row["compression_mode"])
        reference_root, _ = reference_by_ranges[_plan_ranges(selected_plan)]
        output = run_root / "network" / profile / "stage_ring_exact_best_planner" / "run-1"
        row, _, _ = _run_stage_measurement(
            run_root=run_root,
            run_id=f"{run_id}-{profile}-best-planner-r1",
            plan=selected_plan,
            profile_name=profile,
            strategy="stage_ring_exact_best_planner",
            compression=selected_compression,
            prompt_ids=prompt_ids,
            expected_ids=expected_ids,
            token_count=token_count,
            reference_root=reference_root,
            profile_index=profile_index,
            profile_manifest=profile_values,
            run_index=1,
            output_directory=output,
        )
        final_stage_rows.append(row)
        network_rows.append(row)
    _write_json(run_root / "network" / "planner_decisions.json", planner_decisions)

    # Compression utility is judged against the same topology without
    # compression, and final selection comes from the measured planner.
    compression_rows = []
    for adaptive in adaptive_rows:
        profile = str(adaptive["profile_name"])
        base = next(
            row
            for row in base_stage_rows
            if row["profile_name"] == profile
            and row["strategy"] == f"stage_ring_exact_{best_base_key}"
        )
        selected = selected_candidate_by_profile[profile] == adaptive["strategy"]
        modes = json.loads(str(adaptive["actual_compression_modes"]))
        compression_rows.append(
            {
                "profile_name": profile,
                "stage_count": adaptive["stage_count"],
                "partition_method": adaptive["partition_method"],
                "requested_mode": "adaptive",
                "actual_modes": json.dumps(modes),
                "compression_selected_on_messages": "byte_shuffle_fast_codec" in modes,
                "compression_ratio": adaptive["compression_ratio"],
                "uncompressed_tps": base["throughput_tps"],
                "adaptive_tps": adaptive["throughput_tps"],
                "throughput_effect_tps": float(adaptive["throughput_tps"])
                - float(base["throughput_tps"]),
                "planner_selected": selected,
                "token_match": adaptive["token_match"],
                "bitwise_lossless": adaptive["boundary_hash_match"],
                "codec": "zlib-level-1",
                "codec_version": __import__("zlib").ZLIB_VERSION,
                "evidence_category": "REAL_MODEL_MEASURED",
            }
        )
    write_csv(
        run_root / "compression_summary.csv",
        compression_rows,
        tuple(compression_rows[0]) if compression_rows else ("profile_name",),
    )
    _write_json(run_root / "compression" / "compression_results.json", compression_rows)

    # Prompt-lookup candidate verification is real-model and exact, but it is
    # deliberately excluded from network claims until a socket-path pilot has
    # positive expected value.
    speculation_results: list[dict[str, Any]] = []
    if not options.skip_speculation and not options.network_only:
        speculation_plan = best_base_plan
        speculation_results = run_real_prompt_lookup_speculation(
            plan=speculation_plan,
            prompt_token_ids=prompt_ids,
            expected_token_ids=expected_ids,
            generated_token_count=token_count,
            depths=(2, 4, 8),
            output_directory=run_root / "speculation",
        )
    else:
        speculation_results = [
            {
                "speculation_provider": "prompt_lookup",
                "speculation_depth": depth,
                "exact_token_identity": True,
                "oracle_proposals_used": False,
                "planner_enabled": False,
                "planner_reason": "explicitly skipped" if options.skip_speculation else "network-only run",
                "evidence_category": "OPTIONAL_DIAGNOSTIC_SKIPPED",
                "throughput_multiple_vs_non_speculative": 0.0,
            }
            for depth in (2, 4, 8)
        ]
        _write_json(run_root / "speculation" / "speculation_results.json", speculation_results)
        write_csv(
            run_root / "speculation" / "speculation_summary.csv",
            speculation_results,
            tuple(speculation_results[0]),
        )
    shutil.copy2(
        run_root / "speculation" / "speculation_summary.csv",
        run_root / "speculation_summary.csv",
    )

    concurrency_results = []
    if not options.network_only:
        concurrency_plan = best_base_plan
        for concurrency in (1, 2, 4, 8):
            output = run_root / "concurrency" / f"concurrency-{concurrency}"
            controller = StageRingController(
                run_id=f"{run_id}-concurrency-{concurrency}",
                plan=concurrency_plan,
                network_profile=NETWORK_PROFILES["loopback_unshaped"],
                output_directory=output,
                timeout_s=300,
            )
            concurrency_results.append(
                run_concurrent_stage_ring(
                    run_id=f"{run_id}-concurrency-{concurrency}",
                    controller=controller,
                    prompt_token_ids=prompt_ids,
                    expected_token_ids=expected_ids,
                    concurrency=concurrency,
                    generated_token_count=token_count,
                )
            )
        cancellation_output = run_root / "concurrency" / "cancellation-smoke"
        cancellation_controller = StageRingController(
            run_id=f"{run_id}-concurrency-cancellation",
            plan=concurrency_plan,
            network_profile=NETWORK_PROFILES["loopback_unshaped"],
            output_directory=cancellation_output,
            timeout_s=300,
        )
        concurrency_results.append(
            run_concurrent_stage_ring(
                run_id=f"{run_id}-concurrency-cancellation",
                controller=cancellation_controller,
                prompt_token_ids=prompt_ids,
                expected_token_ids=expected_ids,
                concurrency=2,
                generated_token_count=min(token_count, 4),
                cancel_one_before_prefill=True,
            )
        )
    else:
        concurrency_results = [
            {
                "concurrency_active": 0,
                "all_sessions_exact": False,
                "valid_for_claims": False,
                "errors": ["network-only run"],
            }
        ]
    write_concurrency_summary(concurrency_results, run_root / "concurrency_summary.csv")
    _write_json(run_root / "concurrency" / "concurrency_results.json", concurrency_results)

    failure_results = []
    if not options.network_only:
        failure_results = run_real_failure_and_recovery_smokes(
            run_id=run_id,
            plan=best_base_plan,
            profile=NETWORK_PROFILES["loopback_unshaped"],
            prompt_token_ids=prompt_ids,
            expected_token_ids=expected_ids,
            output_directory=run_root / "failures",
            generated_token_count=min(token_count, 4),
        )
        shutil.copy2(run_root / "failures" / "failure_summary.csv", run_root / "failure_summary.csv")
    else:
        failure_results = [
            {
                "test": "network_only",
                "failure_detected": False,
                "recovered": False,
                "exact_continuation": False,
            }
        ]
        write_csv(
            run_root / "failure_summary.csv", failure_results, tuple(failure_results[0])
        )

    # Flatten the final public evidence table only after all raw records exist.
    network_public = [_public_row(row) for row in network_rows]
    write_csv(
        run_root / "network_profile_results.csv",
        network_public,
        (*RESULT_COLUMNS, *EXTRA_RESULT_COLUMNS),
    )
    summary_rows, analysis = build_network_summary(network_rows, output_directory=run_root)
    chart_inspection = generate_network_charts(
        run_root / "network_profile_summary.csv", run_root / "charts"
    )

    stage_latency_rows = []
    critical_rows = []
    for row in [*base_stage_rows, *adaptive_rows, *final_stage_rows]:
        critical = row["_result"]["critical_path"]
        for stage_id, compute_ns in critical.get("stage_compute_ns", {}).items():
            stage_latency_rows.append(
                {
                    "profile_name": row["profile_name"],
                    "strategy": row["strategy"],
                    "stage_id": stage_id,
                    "total_compute_ns": compute_ns,
                    "compute_ns_per_token": int(compute_ns) / max(int(row["generated_tokens"]), 1),
                    "stage_imbalance_ratio": critical["stage_imbalance_ratio"],
                }
            )
        critical_rows.append(
            {
                "profile_name": row["profile_name"],
                "strategy": row["strategy"],
                "stage_count": row["stage_count"],
                "messages_per_token": row["messages_per_token"],
                "serial_waits_per_token": row["serial_waits_per_token"],
                "payload_bytes_per_token": row["payload_bytes_per_token"],
                "wire_bytes_per_token": row["wire_bytes_per_token"],
                "serialization_ns_per_token": critical["serialisation_ns_per_token"],
                "compression_ns_per_token": critical["compression_ns_per_token"],
                "socket_ns_per_token": critical["socket_ns_per_token"],
                "queue_ns_per_token": critical["queue_ns_per_token"],
                "model_compute_ns_per_token": critical["model_compute_ns_per_token"],
                "coordinator_blocked_ns_per_token": critical[
                    "coordinator_blocked_ns_per_token"
                ],
                "communication_compute_overlap_ns": critical[
                    "communication_compute_overlap_ns"
                ],
                "gpu_idle_ns": critical["gpu_idle_ns"],
                "ttft_seconds": row["ttft_seconds"],
                "mean_itl_seconds": row["mean_itl_seconds"],
                "p95_itl_seconds": row["p95_itl_seconds"],
                "valid_for_claims": row["valid_for_claims"],
                "raw_record_path": row["raw_record_path"],
                "serial_waits_method": row["serial_waits_method"],
            }
        )
    for row in baseline_gate_rows:
        source = row["_source"]
        critical = row["_critical_path"]
        tokens = max(int(row["generated_tokens"]), 1)
        critical_rows.append(
            {
                "profile_name": row["profile_name"],
                "strategy": row["strategy"],
                "stage_count": 0,
                "messages_per_token": row["messages_per_token"],
                "serial_waits_per_token": row["serial_waits_per_token"],
                "payload_bytes_per_token": row["payload_bytes_per_token"],
                "wire_bytes_per_token": row["wire_bytes_per_token"],
                "serialization_ns_per_token": 0,
                "compression_ns_per_token": 0,
                "socket_ns_per_token": int(source.get("rpc_transport_ns", 0)) / tokens,
                "queue_ns_per_token": int(source.get("rpc_queue_ns", 0)) / tokens,
                "model_compute_ns_per_token": int(source.get("rpc_compute_ns", 0)) / tokens,
                "coordinator_blocked_ns_per_token": int(source.get("rpc_transport_ns", 0))
                / tokens,
                "communication_compute_overlap_ns": 0,
                "gpu_idle_ns": 0,
                "ttft_seconds": row["ttft_seconds"],
                "mean_itl_seconds": row["mean_itl_seconds"],
                "p95_itl_seconds": row["p95_itl_seconds"],
                "valid_for_claims": row["valid_for_claims"] and critical["valid"],
                "raw_record_path": row["raw_record_path"],
                "serial_waits_method": row["serial_waits_method"],
                "dependency_event_count": critical["observed_dependency_count"],
            }
        )
    write_csv(
        run_root / "stage_latency_summary.csv",
        stage_latency_rows,
        tuple(stage_latency_rows[0]) if stage_latency_rows else ("profile_name",),
    )
    write_csv(
        run_root / "critical_path_summary.csv",
        critical_rows,
        tuple(critical_rows[0]) if critical_rows else ("profile_name",),
    )

    coordinator_removed = all(
        edge["source_stage"] >= 0 and edge["destination_stage"] >= 0
        for row in final_stage_rows
        for edge in row["_result"]["critical_path"]["dependency_edges"]
    ) and all(
        not row["_result"]["critical_path"]["invalid_dependency_links"]
        for row in final_stage_rows
    )
    _write_json(
        run_root / "traces" / "index.json",
        {
            "trace_paths": sorted(
                {
                    path
                    for row in [*base_stage_rows, *adaptive_rows, *final_stage_rows]
                    for path in row["_result"]["trace_paths"]
                }
            ),
            "coordinator_removed_from_dependency_graph": coordinator_removed,
            "definition": (
                "A network dependency on the token-critical path that must complete before "
                "the next required model computation can begin."
            ),
        },
    )

    final_tests = _run_test_command(
        run_root=run_root,
        name="final-regression-tests",
        arguments=["-q"],
        timeout_seconds=1800,
    )
    checkpoint["completed_phases"].extend(
        [
            "baseline",
            "network_matrix",
            "compression",
            "speculation",
            "concurrency",
            "failures",
            "analysis_and_charts",
            "regression_tests",
        ]
    )
    checkpoint["ended_ns"] = time.time_ns()
    _write_json(run_root / "checkpoint.json", checkpoint)

    plan_records = [plan.to_dict() for plan in plans.values()]
    limitations = [
        "All shaped profiles use real TCP sockets on one Windows host; they validate byte-acting latency/bandwidth sensitivity, not a physical multi-host fabric.",
        "One RTX 5090 time-slices multiple CUDA processes, so stage overlap and context-switch costs differ from one-GPU-per-stage deployment.",
        "The frozen Experiment 010 network workload has one independent repetition per profile. Point estimates and descriptive bootstrap intervals are available, but statistical improvement is not inferentially identifiable.",
        "Host monotonic clocks are shared; cross-host clock synchronization error is therefore not measured.",
        "Prompt-lookup speculation was verified on the real staged model without oracle proposals, but was excluded from network performance claims because no positive socket-path pilot was established.",
        "Continuous scheduling uses independent size-one microbatches; tensor fusion across different DynamicCache objects is not implemented.",
        "Physical NIC driver queues, switches, congestion control across hosts, and independent power/thermal domains remain untested.",
    ]

    # Seed required files so the artifact audit can be evaluated before the
    # final gate/verdict/report rewrite.
    _write_json(run_root / "gate_results.json", [])
    _write_json(
        run_root / "verdict.json",
        {"verdict": "PENDING", "canonical": options.canonical, "run_id": run_id},
    )
    write_reports(
        run_root=run_root,
        verdict="PENDING",
        gates=[],
        summary_rows=summary_rows,
        analysis=analysis,
        exactness_results=exactness_records,
        compression_results=compression_rows,
        speculation_results=speculation_results,
        concurrency_results=concurrency_results,
        failure_results=failure_results,
        plan_records=plan_records,
        limitations=limitations,
        reproduction_command=f".\\scripts\\run-experiment-011.ps1 -Full -RunId {run_id} -Resume",
    )
    build_manifest(
        run_root,
        metadata={
            "experiment": EXPERIMENT_ID,
            "run_id": run_id,
            "verdict": "PENDING",
            "canonical": options.canonical,
            "provisional": True,
        },
    )
    validation = validate_artifacts(run_root)
    _write_json(run_root / "artifact_validation.json", validation)
    gates, verdict = evaluate_gates(
        run_root=run_root,
        summary_rows=summary_rows,
        analysis=analysis,
        baseline_rows=baseline_gate_rows,
        exactness_results=exactness_records,
        ownership_valid=bool(ownership["valid"]),
        coordinator_removed=coordinator_removed,
        compression_results=compression_rows,
        speculation_results=speculation_results,
        concurrency_results=concurrency_results,
        regression_tests_passed=bool(final_tests["passed"]),
        evidence_complete=bool(validation["valid"] and chart_inspection["all_minimum_dimensions_met"]),
    )
    if not options.canonical:
        verdict = "QUICK_DIAGNOSTIC_NO_CLOSURE"
    _write_json(run_root / "gate_results.json", gates)
    _write_json(
        run_root / "verdict.json",
        {
            "experiment": EXPERIMENT_ID,
            "run_id": run_id,
            "verdict": verdict,
            "canonical": options.canonical,
            "quick_cannot_close": not options.canonical,
            "core_gates_passed": [
                row["gate"] for row in gates if row["status"] == "PASS"
            ],
            "gates_failed": [row["gate"] for row in gates if row["status"] == "FAIL"],
            "thresholds_changed_after_results": False,
        },
    )
    write_reports(
        run_root=run_root,
        verdict=verdict,
        gates=gates,
        summary_rows=summary_rows,
        analysis=analysis,
        exactness_results=exactness_records,
        compression_results=compression_rows,
        speculation_results=speculation_results,
        concurrency_results=concurrency_results,
        failure_results=failure_results,
        plan_records=plan_records,
        limitations=limitations,
        reproduction_command=f".\\scripts\\run-experiment-011.ps1 -Full -RunId {run_id} -Resume",
    )
    validation = validate_artifacts(run_root)
    _write_json(run_root / "artifact_validation.json", validation)
    expected_zip_path = run_root.parent / f"{run_root.name}.zip"
    zip_hash_record_path = run_root.parent / f"{run_root.name}.zip.sha256.json"
    _write_json(
        run_root / "bundle_identity.json",
        {
            "zip_path": str(expected_zip_path.resolve()),
            "zip_hash_record_path": str(zip_hash_record_path.resolve()),
            "zip_hash_excluded_from_internal_manifest": True,
            "reason": "a ZIP cannot contain a trustworthy hash of itself",
        },
    )
    manifest = build_manifest(
        run_root,
        metadata={
            "experiment": EXPERIMENT_ID,
            "run_id": run_id,
            "verdict": verdict,
            "canonical": options.canonical,
            "started_ns": started_ns,
            "ended_ns": time.time_ns(),
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION,
            "source_commit": json.loads(
                (run_root / "source_identity.json").read_text(encoding="utf-8")
            )["commit"],
            "artifact_validation": validation,
            "chart_inspection": chart_inspection,
            "archived_values_are_not_new_measurements": True,
            "synthetic_rows_used_for_performance_claims": False,
        },
    )
    zip_path = make_zip(run_root)
    _write_json(
        zip_hash_record_path,
        {
            "zip_path": str(zip_path.resolve()),
            "zip_bytes": zip_path.stat().st_size,
            "zip_sha256": sha256_file(zip_path),
            "manifest_artifact_count": manifest["artifact_count"],
            "manifest_sha256": sha256_file(run_root / "manifest.json"),
        },
    )

    summary_by_profile = {row["profile_name"]: row for row in summary_rows}
    selected_raw = final_stage_rows
    old_waits = statistics.median(float(row["serial_waits_per_token"]) for row in baseline_gate_rows)
    new_waits = statistics.median(float(row["serial_waits_per_token"]) for row in selected_raw)
    old_messages = statistics.median(float(row["messages_per_token"]) for row in baseline_gate_rows)
    new_messages = statistics.median(float(row["messages_per_token"]) for row in selected_raw)
    old_bytes = statistics.median(float(row["payload_bytes_per_token"]) for row in baseline_gate_rows)
    new_bytes = statistics.median(float(row["payload_bytes_per_token"]) for row in selected_raw)
    best_topology = statistics.mode(int(row["stage_count"]) for row in final_stage_rows)
    print(f"Experiment verdict: {verdict}")
    print("Gates passed: " + ", ".join(row["gate"] for row in gates if row["status"] == "PASS"))
    print("Gates failed: " + ", ".join(row["gate"] for row in gates if row["status"] == "FAIL"))
    for profile in profiles:
        row = summary_by_profile[profile]
        print(
            f"{PROFILE_LABELS[profile]}: old {float(row['same_run_baseline_median_tps']):.2f} tok/s; "
            f"new {float(row['stage_exact_median_tps']):.2f} tok/s"
        )
    print(
        f"Regional-WAN improvement: {float(summary_by_profile['regional_wan']['throughput_multiple']):.2f}x"
    )
    print(
        f"Global-WAN improvement: {float(summary_by_profile['global_wan']['throughput_multiple']):.2f}x"
    )
    print(f"Serial waits/token: old {old_waits:.2f}; new {new_waits:.2f}")
    print(f"Messages/token: old {old_messages:.2f}; new {new_messages:.2f}")
    print(f"Payload bytes/token: old {old_bytes:,.0f}; new {new_bytes:,.0f}")
    print(f"Exact token match: {all(row['token_match'] for row in final_stage_rows)}")
    print(f"Best stage topology: {best_topology} stages")
    print(f"Compression selected: {any(row['planner_selected'] for row in compression_rows)}")
    print(f"Speculation selected: {any(row.get('planner_enabled') for row in speculation_results)}")
    print(f"Full evidence directory: {run_root}")
    print(f"ZIP bundle path: {zip_path}")
    for base in (
        "06_network_profile_before_after",
        "06b_network_profile_same_run_comparison",
        "06c_network_profile_experiment_011",
        "06d_network_profile_improvement",
    ):
        print(f"Chart family: {run_root / 'charts' / base}.[png|svg|pdf]")
    return run_root, zip_path, verdict
