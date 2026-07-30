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
    _write_artifact_manifest(run_dir)
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


def _write_artifact_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "artifact_manifest.json"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != manifest_path)
    _write_json(
        manifest_path,
        {
            "algorithm": "sha256",
            "files": {path.relative_to(run_dir).as_posix(): sha256_file(path) for path in files},
        },
    )


def validate_run(run_dir: str | Path) -> list[str]:
    root = Path(run_dir).expanduser().resolve()
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
    summary_payload: dict[str, Any] = {}
    summary_path = root / "summary.json"
    if summary_path.is_file():
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, TypeError):
            summary_payload = {}
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
    manifest_path = root / "artifact_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for relative, expected in manifest.get("files", {}).items():
                path = root / relative
                if not path.is_file():
                    errors.append(f"manifest file missing: {relative}")
                elif sha256_file(path) != expected:
                    errors.append(f"artifact checksum mismatch: {relative}")
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            errors.append(f"invalid artifact_manifest.json: {exc}")
    return errors
