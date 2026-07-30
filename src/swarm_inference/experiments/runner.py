"""Experiment matrix execution and complete artifact writing."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import psutil
import yaml

from swarm_inference.config.models import ExperimentConfig
from swarm_inference.doctor import inspect_environment
from swarm_inference.experiments.charts import generate_charts
from swarm_inference.experiments.metrics import (
    evaluate_experiment,
    project_acceptance_status,
)
from swarm_inference.experiments.reporting import render_html_report
from swarm_inference.experiments.scaling import (
    capacity_normalised_efficiency,
    homogeneous_scaling_efficiency,
    marginal_throughput,
    predicted_ideal_throughput,
    throughput_gain,
)
from swarm_inference.model.manifest import hash_shard_directory
from swarm_inference.protocol.checksums import sha256_file
from swarm_inference.simulation.simulator import SimulationResult, Simulator


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    run_id: str
    run_dir: Path
    report_path: Path
    passed: bool
    summary: dict[str, Any]


def _git_metadata() -> dict[str, Any]:
    def run(*arguments: str) -> tuple[int, str]:
        try:
            result = subprocess.run(
                ["git", *arguments],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 1, str(exc)
        return result.returncode, (result.stdout or result.stderr).strip()

    code, commit = run("rev-parse", "HEAD")
    status_code, status = run("status", "--porcelain")
    return {
        "commit": commit if code == 0 else "unavailable",
        "dirty": bool(status) if status_code == 0 else None,
        "status": status,
    }


def _package_versions() -> dict[str, str]:
    names = [
        "swarm-inference-lab",
        "torch",
        "transformers",
        "safetensors",
        "grpcio",
        "protobuf",
        "numpy",
        "pandas",
        "matplotlib",
        "pydantic",
        "typer",
    ]
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def collect_environment(
    *,
    config: ExperimentConfig,
    start: datetime,
    end: datetime | None = None,
) -> dict[str, Any]:
    doctor = inspect_environment(required_ports=())
    lock_path = Path("uv.lock")
    return {
        "git": _git_metadata(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "operating_system": doctor.details["os"],
        "cpu": doctor.details["cpu"],
        "gpu": doctor.details["nvidia"],
        "driver_version": doctor.details["nvidia"].get("driver_version"),
        "pytorch_version": doctor.details["torch"].get("version"),
        "cuda_runtime_version": doctor.details["torch"].get("cuda_runtime"),
        "packages": _package_versions(),
        "package_lock_hash": sha256_file(lock_path) if lock_path.is_file() else None,
        "model_identifier": config.model_id,
        "model_revision": config.model_revision,
        "model_file_hashes": {},
        "random_seed": config.seed,
        "start_timestamp": start.isoformat(),
        "end_timestamp": end.isoformat() if end else None,
        "execution_mode": config.execution_mode.value,
        "hostname": platform.node(),
        "process_id": os.getpid(),
        "system_ram_bytes": psutil.virtual_memory().total,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_manifest(config: ExperimentConfig, result: SimulationResult) -> dict[str, Any]:
    model = config.model
    return {
        "schema_version": "1",
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "architecture": f"synthetic-{model.routing}",
        "layer_count": model.layer_count,
        "hidden_size": model.hidden_size,
        "weight_dtype": model.activation_dtype,
        "total_weight_bytes": model.layer_count * model.bytes_per_layer,
        "per_layer_weight_bytes": [model.bytes_per_layer] * model.layer_count,
        "estimated_cache_bytes_per_token_per_layer": model.cache_bytes_per_token_per_layer,
        "activation_bytes_per_stage_boundary": model.activation_bytes,
        "stages": [stage.model_dump(mode="json") for stage in result.stages],
        "shard_hashes": {str(stage.stage_id): "synthetic-deterministic" for stage in result.stages},
        "compatible_worker_backends": ["synthetic"],
        "evidence_kind": "emulated",
    }


def _run_key(result: SimulationResult) -> str:
    return f"n{result.node_count}-c{result.concurrent_requests}"


def run_experiment(config: ExperimentConfig) -> ExperimentRun:
    """Execute a configured matrix and write the complete standard artifact set."""

    start = datetime.now(UTC)
    run_id = f"{start.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = Path(config.output_root).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "charts").mkdir()

    node_counts = config.node_counts or [sum(profile.count for profile in config.nodes)]
    concurrency_counts = config.concurrent_request_counts or [config.workload.concurrent_requests]
    results: list[SimulationResult] = []
    if config.execution_mode.value == "simulation":
        for concurrency in concurrency_counts:
            for node_count in node_counts:
                results.append(
                    Simulator(
                        config,
                        node_count=node_count,
                        concurrent_requests=concurrency,
                    ).run()
                )
    else:
        raise NotImplementedError(
            "experiment runner currently supports simulation directly; "
            "single-host-loopback uses run_loopback_experiment"
        )
    end = datetime.now(UTC)
    environment = collect_environment(config=config, start=start, end=end)

    events: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    workers: list[dict[str, Any]] = []
    network: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for result in results:
        key = _run_key(result)
        summaries.append({"run_key": key, **result.summary})
        for event in result.events:
            row = {"run_key": key, **event.to_dict()}
            events.append(row)
            if event.event_type in {
                "worker_failed",
                "request_failed",
                "stage_recovery_started",
                "stage_recovery_completed",
                "corrupt_result",
                "worker_quarantined",
                "backpressure",
            }:
                failures.append(row)
        requests.extend({"run_key": key, **row} for row in result.requests)
        stages.extend({"run_key": key, **row} for row in result.stage_metrics)
        workers.extend({"run_key": key, **row} for row in result.workers)
        network.extend({"run_key": key, **row} for row in result.network_metrics)

    scaling_rows = _scaling_rows(results)
    experiment_criteria = evaluate_experiment(
        config=config, summaries=summaries, scaling_rows=scaling_rows
    )
    project_criteria = project_acceptance_status(
        config=config, experiment_criteria=experiment_criteria
    )
    primary = max(
        summaries,
        key=lambda row: (
            int(row["concurrent_request_count"]),
            int(row["node_count"]),
        ),
    )
    baseline = min(
        summaries,
        key=lambda row: (
            int(row["concurrent_request_count"]),
            int(row["node_count"]),
        ),
    )
    acceptance_payload = [item.to_dict() for item in experiment_criteria + project_criteria]
    summary = {
        "schema_version": "1",
        "run_id": run_id,
        "execution_mode": config.execution_mode.value,
        "values": "emulated" if config.execution_mode.value == "simulation" else "measured",
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "seed": config.seed,
        "start_timestamp": start.isoformat(),
        "end_timestamp": end.isoformat(),
        "primary_metric_definition": (
            "committed output tokens from successfully completed and verified requests "
            "divided by experiment elapsed time"
        ),
        "primary_result": primary,
        "baseline_result": baseline,
        "matrix_results": summaries,
        "acceptance_criteria": acceptance_payload,
        "status": (
            "PASS" if all(item["status"] == "PASS" for item in acceptance_payload) else "FAIL"
        ),
        "failed_acceptance_criteria": [
            item["name"] for item in acceptance_payload if item["status"] == "FAIL"
        ],
        "limitations": [
            "Simulation results are not physical distributed performance.",
            "Synthetic execution is not evidence of real-model kernel performance.",
            "Configured profiles marked assumed have not been measured on physical hardware.",
            "Aggregate throughput is not single-request generation speed.",
        ],
        "result_fingerprints": {
            _run_key(result): result.deterministic_fingerprint() for result in results
        },
    }

    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    _write_json(run_dir / "environment.json", environment)
    _write_json(run_dir / "model_manifest.json", _synthetic_manifest(config, results[0]))
    _write_json(
        run_dir / "worker_manifest.json",
        {
            "profiles": [node.model_dump(mode="json") for node in config.nodes],
            "workers": workers,
            "placement_decisions": [
                {
                    "run_key": _run_key(result),
                    **asdict(decision),
                }
                for result in results
                for decision in result.placement.decisions
            ],
        },
    )
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_jsonl(run_dir / "requests.jsonl", requests)
    _write_jsonl(run_dir / "stage_metrics.jsonl", stages)
    _write_jsonl(run_dir / "worker_metrics.jsonl", workers)
    _write_jsonl(run_dir / "network_metrics.jsonl", network)
    _write_csv(
        run_dir / "scaling.csv",
        scaling_rows,
        [
            "node_count",
            "concurrent_requests",
            "throughput",
            "baseline_throughput",
            "throughput_gain",
            "marginal_throughput",
            "homogeneous_scaling_efficiency",
            "predicted_ideal_throughput",
            "capacity_normalised_efficiency",
        ],
    )
    _write_csv(
        run_dir / "latency.csv",
        requests,
        [
            "run_key",
            "request_id",
            "time_to_first_token_s",
            "decode_tokens_s",
            "end_to_end_s",
            "queueing_s",
            "network_s",
            "stage_execution_s",
            "replay_s",
        ],
    )
    _write_csv(
        run_dir / "failures.csv",
        failures,
        [
            "run_key",
            "sequence",
            "simulated_time_s",
            "event_type",
            "request_id",
            "worker_id",
            "stage_id",
            "details",
        ],
    )
    _write_json(run_dir / "summary.json", summary)
    generate_charts(
        chart_dir=run_dir / "charts",
        scaling_rows=scaling_rows,
        requests=requests,
        stages=stages,
        workers=workers,
        network=network,
        failures=failures,
    )
    report_path = render_html_report(
        run_dir=run_dir,
        summary=summary,
        scaling_rows=scaling_rows,
        request_rows=requests,
    )
    write_artifact_manifest(run_dir)
    return ExperimentRun(
        run_id=run_id,
        run_dir=run_dir,
        report_path=report_path,
        passed=summary["status"] == "PASS",
        summary=summary,
    )


def _scaling_rows(results: list[SimulationResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[int, list[SimulationResult]] = {}
    for result in results:
        grouped.setdefault(result.concurrent_requests, []).append(result)
    for concurrency, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda result: result.node_count)
        baseline = ordered[0]
        baseline_tps = float(baseline.summary["aggregate_verified_output_tokens_s"])
        previous_tps = baseline_tps
        for index, result in enumerate(ordered):
            throughput = float(result.summary["aggregate_verified_output_tokens_s"])
            ideal = predicted_ideal_throughput(
                float(stage["aggregate_service_rate"]) for stage in result.stage_metrics
            )
            rows.append(
                {
                    "node_count": result.node_count,
                    "concurrent_requests": concurrency,
                    "throughput": throughput,
                    "baseline_throughput": baseline_tps,
                    "throughput_gain": throughput_gain(throughput, baseline_tps),
                    "marginal_throughput": (
                        0.0 if index == 0 else marginal_throughput(previous_tps, throughput)
                    ),
                    "homogeneous_scaling_efficiency": homogeneous_scaling_efficiency(
                        throughput=throughput,
                        baseline_throughput=baseline_tps,
                        node_count=result.node_count,
                        baseline_node_count=baseline.node_count,
                    ),
                    "predicted_ideal_throughput": ideal,
                    "capacity_normalised_efficiency": capacity_normalised_efficiency(
                        throughput, ideal
                    ),
                }
            )
            previous_tps = throughput
    return rows


def write_artifact_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "artifact_manifest.json"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != manifest_path)
    _write_json(
        manifest_path,
        {
            "algorithm": "sha256",
            "files": {path.relative_to(run_dir).as_posix(): sha256_file(path) for path in files},
        },
    )


_REAL_MODEL_REQUIRED_ARTIFACTS = (
    "config.requested.yaml",
    "config.resolved.yaml",
    "environment.json",
    "git.json",
    "model_inspection.json",
    "model_manifest.json",
    "shard_hashes.json",
    "reference.json",
    "distributed.json",
    "correctness.json",
    "worker_load_proofs.json",
    "coordinator_proof.json",
    "transport_metrics.json",
    "cache_metrics.json",
    "prompt_results.jsonl",
    "cache_replay.json",
    "events.jsonl",
    "summary.json",
    "report.html",
    "quality_gates.json",
    "log_validation.json",
    "artifact_manifest.json",
    "tensors/boundary-diagnostics.json",
)
_REAL_MODEL_REQUIRED_LOGS = (
    "logs/coordinator.log",
    "logs/reference.log",
    "logs/worker-000.log",
    "logs/worker-001.log",
    "logs/worker-002.log",
    "logs/worker-003.log",
)
_REAL_MODEL_REQUIRED_CHARTS = (
    "charts/stage_weight_bytes.png",
    "charts/worker_memory.png",
    "charts/prefill_latency.png",
    "charts/decode_latency.png",
    "charts/boundary_errors.png",
    "charts/activation_bytes.png",
    "charts/kv_cache_bytes.png",
)
_REAL_MODEL_STATUS_FIELDS = (
    "experiment_integrity_status",
    "environment_status",
    "model_revision_status",
    "sharding_status",
    "stage_isolation_status",
    "real_model_execution_status",
    "direct_data_plane_status",
    "kv_cache_status",
    "boundary_correctness_status",
    "token_identity_status",
    "cache_replay_status",
    "prompt_suite_status",
)


def _read_json_artifact(
    root: Path,
    relative: str,
    errors: list[str],
) -> Any:
    path = root / relative
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        errors.append(f"invalid {relative}: {exc}")
        return None


def _validate_artifact_manifest(root: Path, errors: list[str]) -> None:
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files")
        if manifest.get("algorithm") != "sha256" or not isinstance(files, dict):
            errors.append("invalid artifact_manifest.json: expected sha256 files mapping")
            return
        for relative, expected in files.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                errors.append(f"artifact manifest path escapes run directory: {relative}")
            elif not path.is_file():
                errors.append(f"manifest file missing: {relative}")
            elif sha256_file(path) != expected:
                errors.append(f"artifact checksum mismatch: {relative}")
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        errors.append(f"invalid artifact_manifest.json: {exc}")


def _validate_real_model_run(root: Path, summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        *_REAL_MODEL_REQUIRED_ARTIFACTS,
        *_REAL_MODEL_REQUIRED_LOGS,
        *_REAL_MODEL_REQUIRED_CHARTS,
    )
    for relative in required:
        path = root / relative
        if not path.is_file():
            errors.append(f"missing required real-model artifact: {relative}")
        elif path.stat().st_size <= 0:
            errors.append(f"empty required real-model artifact: {relative}")

    for relative in ("config.requested.yaml", "config.resolved.yaml"):
        path = root / relative
        if not path.is_file():
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                errors.append(f"invalid {relative}: expected YAML mapping")
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"invalid {relative}: {exc}")

    environment = _read_json_artifact(root, "environment.json", errors)
    manifest = _read_json_artifact(root, "model_manifest.json", errors)
    shard_hashes = _read_json_artifact(root, "shard_hashes.json", errors)
    reference = _read_json_artifact(root, "reference.json", errors)
    distributed = _read_json_artifact(root, "distributed.json", errors)
    worker_proofs = _read_json_artifact(root, "worker_load_proofs.json", errors)
    quality_gates = _read_json_artifact(root, "quality_gates.json", errors)
    _read_json_artifact(root, "correctness.json", errors)
    _read_json_artifact(root, "coordinator_proof.json", errors)
    _read_json_artifact(root, "transport_metrics.json", errors)
    _read_json_artifact(root, "cache_metrics.json", errors)
    _read_json_artifact(root, "cache_replay.json", errors)
    _read_json_artifact(root, "log_validation.json", errors)
    _read_json_artifact(root, "tensors/boundary-diagnostics.json", errors)

    for relative in ("prompt_results.jsonl", "events.jsonl"):
        path = root / relative
        if not path.is_file():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid {relative} line {line_number}: {exc}")
                break

    invalid_statuses = [
        field
        for field in (*_REAL_MODEL_STATUS_FIELDS, "overall_status")
        if summary.get(field) not in {"PASS", "FAIL"}
    ]
    if invalid_statuses:
        errors.append("missing or invalid real-model status fields: " + ", ".join(invalid_statuses))
    else:
        derived_overall = (
            "PASS"
            if all(summary.get(field) == "PASS" for field in _REAL_MODEL_STATUS_FIELDS)
            else "FAIL"
        )
        if summary.get("overall_status") != derived_overall:
            errors.append("overall_status is inconsistent with mandatory real-model statuses")

    shard_validation: dict[str, Any] = {}
    if isinstance(worker_proofs, list) and worker_proofs:
        from swarm_inference.worker.shard_manager import verify_load_record_checksum

        shard_paths: dict[int, Path] = {}
        for proof in worker_proofs:
            if not isinstance(proof, dict) or not isinstance(proof.get("stage_id"), int):
                errors.append("invalid worker load proof entry")
                continue
            stage_id = int(proof["stage_id"])
            if not verify_load_record_checksum(proof):
                errors.append(f"worker proof stage {stage_id} load-record checksum mismatch")
            shard_path = Path(str(proof.get("shard_path", ""))).expanduser().resolve()
            shard_paths[stage_id] = shard_path
            weights_path = shard_path / "weights.safetensors"
            expected_hash = str(proof.get("shard_hash", ""))
            if not weights_path.is_file():
                errors.append(f"worker proof stage {stage_id} shard is unavailable")
            elif hash_shard_directory(shard_path) != expected_hash:
                errors.append(f"worker proof stage {stage_id} shard checksum mismatch")
        if len(shard_paths) != 4:
            errors.append("worker load proofs do not identify exactly four stage shards")
        model_root = next(iter(shard_paths.values())).parent if shard_paths else None
        if model_root is not None:
            validation_path = model_root / "validation.json"
            try:
                shard_validation = json.loads(validation_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, TypeError) as exc:
                errors.append(f"invalid model shard validation.json: {exc}")
            model_manifest_path = model_root / "manifest.json"
            try:
                model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
                if isinstance(manifest, dict) and model_manifest != manifest:
                    errors.append("run model_manifest.json differs from loaded shard manifest")
            except (json.JSONDecodeError, OSError, TypeError) as exc:
                errors.append(f"invalid model shard manifest.json: {exc}")
            model_hashes_path = model_root / "hashes.json"
            try:
                model_hashes = json.loads(model_hashes_path.read_text(encoding="utf-8"))
                if isinstance(shard_hashes, dict) and model_hashes != shard_hashes:
                    errors.append("run shard_hashes.json differs from model hashes.json")
            except (json.JSONDecodeError, OSError, TypeError) as exc:
                errors.append(f"invalid model hashes.json: {exc}")
    else:
        errors.append("worker_load_proofs.json must contain four proofs")

    if all(
        isinstance(payload, dict)
        for payload in (environment, manifest, reference, distributed, quality_gates)
    ):
        from swarm_inference.experiments.real_status import (
            evaluate_experiment_002_status,
        )

        evaluated = evaluate_experiment_002_status(
            {
                "environment": environment,
                "manifest": manifest,
                "shard_validation": shard_validation,
                "reference": reference,
                "distributed": distributed,
                "quality_gates": quality_gates,
            }
        )
        for field, expected in evaluated.items():
            if summary.get(field) != expected:
                errors.append(
                    f"recorded {field}={summary.get(field)!r} "
                    f"does not match evaluated status {expected!r}"
                )

    report_path = root / "report.html"
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        conclusion = (
            "PASS: A real Qwen3 model was split across four process-isolated workers, "
            "real hidden states crossed worker boundaries, stage-local KV caches were "
            "used, and distributed greedy output matched the independent full-model "
            "reference exactly."
            if summary.get("overall_status") == "PASS"
            else "FAIL: The real distributed Qwen3 experiment did not satisfy all "
            "correctness and isolation criteria."
        )
        if conclusion not in report:
            errors.append("report.html is missing the status-appropriate conclusion")

    _validate_artifact_manifest(root, errors)
    return errors


def validate_run(run_dir: str | Path) -> list[str]:
    root = Path(run_dir).expanduser().resolve()
    summary_payload: dict[str, Any] = {}
    summary_path = root / "summary.json"
    if summary_path.is_file():
        try:
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded_summary, dict):
                summary_payload = loaded_summary
        except (json.JSONDecodeError, OSError, TypeError):
            summary_payload = {}
    if summary_payload.get("execution_mode") == "single-host-loopback-real-model":
        return _validate_real_model_run(root, summary_payload)

    required = [
        "config.resolved.yaml",
        "environment.json",
        "model_manifest.json",
        "worker_manifest.json",
        "events.jsonl",
        "requests.jsonl",
        "stage_metrics.jsonl",
        "worker_metrics.jsonl",
        "network_metrics.jsonl",
        "scaling.csv",
        "latency.csv",
        "failures.csv",
        "summary.json",
        "report.html",
        "artifact_manifest.json",
    ]
    errors = [
        f"missing required artifact: {name}" for name in required if not (root / name).is_file()
    ]
    if summary_payload.get("schema_version") == "2" and summary_payload.get("child_runs"):
        matrix_required = [
            "calibration.json",
            "environment_before.json",
            "environment_after.json",
            "dependency_diff.json",
            "matrix.csv",
            "replica_distribution.csv",
            "transport_breakdown.csv",
            "capacity_prediction.csv",
        ]
        errors.extend(
            f"missing required matrix artifact: {name}"
            for name in matrix_required
            if not (root / name).is_file()
        )
    for chart in [
        "aggregate_throughput.png",
        "per_request_throughput.png",
        "scaling_efficiency.png",
        "stage_utilisation.png",
        "queue_depth.png",
        "network_bytes.png",
        "failure_recovery.png",
    ]:
        if not (root / "charts" / chart).is_file():
            errors.append(f"missing required chart: charts/{chart}")
    if summary_payload.get("schema_version") == "2" and summary_payload.get("child_runs"):
        for chart in [
            "scaling_ratios.png",
            "replica_distribution.png",
            "latency_breakdown.png",
            "coordinator_vs_peer_bytes.png",
            "capacity_prediction_error.png",
            "stream_reuse.png",
        ]:
            if not (root / "charts" / chart).is_file():
                errors.append(f"missing required matrix chart: charts/{chart}")
    _validate_artifact_manifest(root, errors)
    return errors
