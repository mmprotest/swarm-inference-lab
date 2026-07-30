"""Measured single-host loopback replicated-stage scaling matrix."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import statistics
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from swarm_inference.config.models import ExecutionMode, ExperimentConfig
from swarm_inference.experiments.calibration import (
    available_cpu_ids,
    calibrate_synthetic_compute,
)
from swarm_inference.experiments.charts import (
    generate_charts,
    generate_experiment_001_charts,
)
from swarm_inference.experiments.loopback import run_loopback_experiment
from swarm_inference.experiments.reporting import render_html_report
from swarm_inference.experiments.runner import (
    ExperimentRun,
    _write_artifact_manifest,
    collect_environment,
    validate_run,
)
from swarm_inference.experiments.scaling import (
    capacity_normalised_efficiency,
    homogeneous_scaling_efficiency,
    marginal_throughput,
    throughput_gain,
)
from swarm_inference.experiments.status import evaluate_matrix_statuses

ProgressCallback = Callable[[str], None]
HISTORICAL_RUN = Path("artifacts/runs/20260730T032044Z-loopback-matrix-7cf6e3a7")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _package_inventory() -> dict[str, str]:
    inventory: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            inventory[name.lower()] = distribution.version
    return dict(sorted(inventory.items()))


def _environment_snapshot(
    config: ExperimentConfig,
    *,
    start: datetime,
    end: datetime | None = None,
) -> dict[str, Any]:
    return {
        **collect_environment(config=config, start=start, end=end),
        "installed_packages": _package_inventory(),
        "backend": config.backend,
        "data_plane": config.data_plane.value,
    }


def _dependency_diff(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    before_packages = dict(before.get("installed_packages", {}))
    after_packages = dict(after.get("installed_packages", {}))
    removed = {
        name: version for name, version in before_packages.items() if name not in after_packages
    }
    added = {
        name: version for name, version in after_packages.items() if name not in before_packages
    }
    changed = {
        name: {
            "before": before_packages[name],
            "after": after_packages[name],
        }
        for name in before_packages.keys() & after_packages.keys()
        if before_packages[name] != after_packages[name]
    }
    return {
        "removed": removed,
        "added": added,
        "changed": changed,
        "mutation_failure": bool(removed or changed),
        "policy": (
            "Packages present before the run must remain at the same version; "
            "additions are recorded but are not performed by the experiment runner."
        ),
    }


def _historical_baseline() -> dict[str, Any]:
    supplied = {
        "2": 330.18607160331686,
        "4": 330.57940148705944,
        "8": 322.4298300773147,
    }
    source_path = HISTORICAL_RUN / "summary.json"
    throughputs = dict(supplied)
    label = "supplied historical evidence"
    source = "values supplied with Experiment 001 because the original artifact was unavailable"
    if source_path.is_file():
        try:
            historical = json.loads(source_path.read_text(encoding="utf-8"))
            measured = {
                str(int(row["node_count"])): float(row["aggregate_verified_output_tokens_s"])
                for row in historical.get("matrix_results", [])
                if int(row.get("concurrent_request_count", -1)) == 64
                and int(row.get("node_count", -1)) in {2, 4, 8}
            }
            if set(measured) == {"2", "4", "8"}:
                throughputs = measured
                label = "measured historical artifact"
                source = str(HISTORICAL_RUN)
        except (OSError, TypeError, ValueError):
            pass
    throughput_2 = throughputs["2"]
    throughput_4 = throughputs["4"]
    throughput_8 = throughputs["8"]
    return {
        "source": source,
        "concurrency": 64,
        "throughput_by_workers": throughputs,
        "scaling_ratios": {
            "2_to_4": throughput_4 / throughput_2,
            "4_to_8": throughput_8 / throughput_4,
            "2_to_8": throughput_8 / throughput_2,
        },
        "label": label,
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _confidence_interval_95(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
    }.get(len(values) - 1, 1.96)
    mean = statistics.fmean(values)
    half_width = critical * statistics.stdev(values) / math.sqrt(len(values))
    return float(mean - half_width), float(mean + half_width)


def _aggregate_points(point_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in point_rows:
        grouped.setdefault(
            (int(row["node_count"]), int(row["concurrent_request_count"])),
            [],
        ).append(row)
    aggregate: list[dict[str, Any]] = []
    for (worker_count, concurrency), rows in sorted(grouped.items()):
        throughputs = [float(row["aggregate_verified_output_tokens_s"]) for row in rows]
        stdev = statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
        mean = statistics.fmean(throughputs)
        ci_low, ci_high = _confidence_interval_95(throughputs)
        aggregate.append(
            {
                "node_count": worker_count,
                "concurrent_request_count": concurrency,
                "data_plane_mode": rows[0]["data_plane_mode"],
                "backend": rows[0]["backend"],
                "simulated_duration_s": _median(
                    [float(row["measured_duration_s"]) for row in rows]
                ),
                "measured_duration_s": _median([float(row["measured_duration_s"]) for row in rows]),
                "aggregate_verified_output_tokens_s": _median(throughputs),
                "throughput_mean": float(mean),
                "throughput_min": min(throughputs),
                "throughput_max": max(throughputs),
                "throughput_stdev": float(stdev),
                "throughput_cv": float(stdev / mean) if mean else 0.0,
                "throughput_ci95_low": ci_low,
                "throughput_ci95_high": ci_high,
                "repeat_count": len(rows),
                "verified_output_tokens": int(
                    statistics.median([int(row["verified_output_tokens"]) for row in rows])
                ),
                "completed_verified_requests": int(
                    statistics.median([int(row["completed_verified_requests"]) for row in rows])
                ),
                "accepted_requests": int(
                    statistics.median([int(row["accepted_requests"]) for row in rows])
                ),
                "completion_fraction": min(float(row["completion_fraction"]) for row in rows),
                "committed_token_correctness": min(
                    float(
                        row.get(
                            "committed_token_correctness",
                            row["completion_fraction"],
                        )
                    )
                    for row in rows
                ),
                "mean_request_tokens_s": _median(
                    [float(row["mean_request_tokens_s"]) for row in rows]
                ),
                "mean_time_to_first_token_s": _median(
                    [float(row["mean_time_to_first_token_s"]) for row in rows]
                ),
                "mean_end_to_end_s": _median([float(row["mean_end_to_end_s"]) for row in rows]),
                "minimum_stage_utilisation": min(
                    float(row["minimum_stage_utilisation"]) for row in rows
                ),
                "mean_stage_utilisation": _median(
                    [float(row["mean_stage_utilisation"]) for row in rows]
                ),
                "network_bytes": int(
                    statistics.median([int(row["network_bytes"]) for row in rows])
                ),
                "capacity_imbalance": max(float(row["capacity_imbalance"]) for row in rows),
                "failed_requests": sum(int(row["failed_requests"]) for row in rows),
                "recovered_route_changes": sum(int(row["recovered_route_changes"]) for row in rows),
                "replay_bytes": sum(int(row["replay_bytes"]) for row in rows),
                "replay_duration_s": sum(float(row["replay_duration_s"]) for row in rows),
                "quarantined_workers": max(int(row["quarantined_workers"]) for row in rows),
                "idle_workers": max(int(row["idle_workers"]) for row in rows),
                "coordinator_control_bytes": int(
                    statistics.median([int(row["coordinator_control_bytes"]) for row in rows])
                ),
                "coordinator_activation_bytes": sum(
                    int(row["coordinator_activation_bytes"]) for row in rows
                ),
                "worker_to_worker_activation_bytes": int(
                    statistics.median(
                        [int(row["worker_to_worker_activation_bytes"]) for row in rows]
                    )
                ),
                "peer_channels_created": int(
                    statistics.median([int(row["peer_channels_created"]) for row in rows])
                ),
                "peer_streams_created": int(
                    statistics.median([int(row["peer_streams_created"]) for row in rows])
                ),
                "peer_stream_reconnects": int(
                    statistics.median([int(row["peer_stream_reconnects"]) for row in rows])
                ),
                "active_peer_pairs": int(
                    statistics.median([int(row["active_peer_pairs"]) for row in rows])
                ),
                "data_messages_sent": int(
                    statistics.median([int(row["data_messages_sent"]) for row in rows])
                ),
                "data_messages_received": int(
                    statistics.median([int(row["data_messages_received"]) for row in rows])
                ),
                "meaningful_replica_fraction": min(
                    float(row["meaningful_replica_fraction"]) for row in rows
                ),
                "replica_imbalance_ratio": max(
                    float(row["replica_imbalance_ratio"]) for row in rows
                ),
                "reservation_leaks": sum(int(row["reservation_leaks"]) for row in rows),
                "predicted_throughput": _median(
                    [float(row["predicted_throughput"]) for row in rows]
                ),
                "absolute_prediction_error": _median(
                    [float(row["absolute_prediction_error"]) for row in rows]
                ),
                "prediction_error_fraction": _median(
                    [float(row["prediction_error_fraction"]) for row in rows]
                ),
                "capacity_normalised_efficiency": _median(
                    [float(row["capacity_normalised_efficiency"]) for row in rows]
                ),
                "affinity_status": sorted({str(row["affinity_status"]) for row in rows}),
            }
        )
    return aggregate


def _scaling_rows(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[int, list[dict[str, Any]]] = {}
    for point in points:
        grouped.setdefault(
            int(point["concurrent_request_count"]),
            [],
        ).append(point)
    for concurrency, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(row["node_count"]))
        baseline = ordered[0]
        baseline_tps = float(baseline["aggregate_verified_output_tokens_s"])
        previous = baseline_tps
        for index, point in enumerate(ordered):
            throughput = float(point["aggregate_verified_output_tokens_s"])
            predicted = float(point["predicted_throughput"])
            output.append(
                {
                    "node_count": int(point["node_count"]),
                    "concurrent_requests": concurrency,
                    "throughput": throughput,
                    "throughput_min": float(point["throughput_min"]),
                    "throughput_max": float(point["throughput_max"]),
                    "throughput_mean": float(point["throughput_mean"]),
                    "throughput_stdev": float(point["throughput_stdev"]),
                    "throughput_cv": float(point["throughput_cv"]),
                    "throughput_ci95_low": point["throughput_ci95_low"],
                    "throughput_ci95_high": point["throughput_ci95_high"],
                    "repeat_count": int(point["repeat_count"]),
                    "baseline_throughput": baseline_tps,
                    "throughput_gain": throughput_gain(
                        throughput,
                        baseline_tps,
                    ),
                    "marginal_throughput": (
                        0.0 if index == 0 else marginal_throughput(previous, throughput)
                    ),
                    "homogeneous_scaling_efficiency": (
                        homogeneous_scaling_efficiency(
                            throughput=throughput,
                            baseline_throughput=baseline_tps,
                            node_count=int(point["node_count"]),
                            baseline_node_count=int(baseline["node_count"]),
                        )
                    ),
                    "predicted_ideal_throughput": predicted,
                    "predicted_throughput": predicted,
                    "absolute_prediction_error": float(point["absolute_prediction_error"]),
                    "prediction_error_fraction": float(point["prediction_error_fraction"]),
                    "capacity_normalised_efficiency": (
                        capacity_normalised_efficiency(throughput, predicted)
                    ),
                }
            )
            previous = throughput
    return output


def _annotated_rows(
    child_run: ExperimentRun,
    *,
    point_id: str,
    repeat: int,
    maximum_request_samples: int = 64,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    request_rows = _read_jsonl(child_run.run_dir / "requests.jsonl")
    if len(request_rows) > maximum_request_samples:
        stride = max(1, len(request_rows) // maximum_request_samples)
        request_rows = request_rows[::stride][:maximum_request_samples]
    common = {
        "point_id": point_id,
        "repeat": repeat,
        "child_run_id": child_run.run_id,
    }
    requests = [{**row, **common} for row in request_rows]
    stages = [{**row, **common} for row in _read_jsonl(child_run.run_dir / "stage_metrics.jsonl")]
    workers = [{**row, **common} for row in _read_jsonl(child_run.run_dir / "worker_metrics.jsonl")]
    networks = [
        {**row, **common} for row in _read_jsonl(child_run.run_dir / "network_metrics.jsonl")
    ]
    failures: list[dict[str, Any]] = []
    return requests, stages, workers, networks, failures


def _replica_distribution_rows(
    worker_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    totals: dict[tuple[str, int, int], int] = {}
    for row in worker_rows:
        stage_id = row.get("assigned_stage_id")
        if stage_id is None:
            continue
        key = (str(row["point_id"]), int(row["repeat"]), int(stage_id))
        totals[key] = totals.get(key, 0) + int(row.get("operations", 0))
    output: list[dict[str, Any]] = []
    for row in worker_rows:
        stage_id = row.get("assigned_stage_id")
        if stage_id is None:
            continue
        key = (str(row["point_id"]), int(row["repeat"]), int(stage_id))
        operations = int(row.get("operations", 0))
        total = totals.get(key, 0)
        share = operations / total if total else 0.0
        output.append(
            {
                "point_id": row["point_id"],
                "repeat": row["repeat"],
                "worker_id": row["worker_id"],
                "stage_id": stage_id,
                "operations": operations,
                "tokens_contributed": row.get("tokens_contributed", 0),
                "busy_time_s": row.get("busy_time_s", 0),
                "reserved_work": row.get("reserved_work", 0),
                "in_flight_work": row.get("in_flight_work", 0),
                "operation_share": share,
                "meaningfully_used": share >= 0.05,
                "bytes_sent": row.get("bytes_sent", 0),
                "time_starved_s": row.get("time_starved_s", 0),
                "time_overloaded_s": row.get("time_overloaded_s", 0),
            }
        )
    return output


def _transport_rows(point_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "point_id",
        "repeat",
        "node_count",
        "concurrent_request_count",
        "data_plane_mode",
        "coordinator_control_bytes",
        "coordinator_activation_bytes",
        "worker_to_worker_activation_bytes",
        "peer_channels_created",
        "peer_streams_created",
        "peer_stream_reconnects",
        "active_peer_pairs",
        "data_messages_sent",
        "data_messages_received",
        "serialisation_time_ms",
        "deserialisation_time_ms",
        "stream_queue_time_ms",
        "hop_transfer_time_ms",
        "stage_execution_time_ms",
        "admission_time_ms",
        "route_reservation_time_ms",
    ]
    return [{field: row.get(field) for field in fields} for row in point_rows]


def _capacity_rows(point_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "point_id",
        "repeat",
        "node_count",
        "concurrent_request_count",
        "predicted_throughput",
        "aggregate_verified_output_tokens_s",
        "absolute_prediction_error",
        "prediction_error_fraction",
        "capacity_normalised_efficiency",
    ]
    return [{field: row.get(field) for field in fields} for row in point_rows]


def _matrix_columns(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "point_id",
        "repeat",
        "child_run_id",
        "child_run_dir",
        "node_count",
        "concurrent_request_count",
        "measured_duration_s",
        "aggregate_verified_output_tokens_s",
        "completion_fraction",
        "committed_token_correctness",
        "meaningful_replica_fraction",
        "replica_imbalance_ratio",
        "coordinator_activation_bytes",
        "worker_to_worker_activation_bytes",
        "peer_streams_created",
        "data_messages_sent",
        "predicted_throughput",
        "prediction_error_fraction",
        "affinity_status",
        "data_plane_mode",
        "backend",
    ]
    available = {key for row in rows for key in row}
    return [column for column in preferred if column in available]


def _write_parent_artifacts(
    *,
    run_dir: Path,
    config: ExperimentConfig,
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    stages: list[dict[str, Any]],
    workers: list[dict[str, Any]],
    network: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    point_rows: list[dict[str, Any]],
    scaling_rows: list[dict[str, Any]],
    worker_counts: list[int],
    concurrency_counts: list[int],
    repeats: int,
    child_runs: list[dict[str, Any]],
) -> Path:
    aggregate_points = list(summary.get("matrix_results", []))
    replica_rows = _replica_distribution_rows(workers)
    transport_rows = _transport_rows(point_rows)
    capacity_rows = _capacity_rows(point_rows)
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    _write_json(
        run_dir / "environment.json",
        summary.get("environment_after", {}),
    )
    _write_json(
        run_dir / "model_manifest.json",
        {
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "architecture": "synthetic-dense-calibrated-cpu",
            "layer_count": config.model.layer_count,
            "hidden_size": config.model.hidden_size,
            "activation_bytes": config.model.activation_bytes,
            "total_weight_bytes": (config.model.layer_count * config.model.bytes_per_layer),
            "stage_count": config.model.stage_count,
            "cpu_work_units": config.model.cpu_work_units,
            "evidence_kind": "measured synthetic single-host-loopback matrix",
        },
    )
    _write_json(
        run_dir / "worker_manifest.json",
        {
            "worker_counts": worker_counts,
            "concurrency_counts": concurrency_counts,
            "repeats": repeats,
            "data_plane_mode": config.data_plane.value,
            "backend": config.backend,
            "children": child_runs,
        },
    )
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_jsonl(run_dir / "requests.jsonl", requests)
    _write_jsonl(run_dir / "stage_metrics.jsonl", stages)
    _write_jsonl(run_dir / "worker_metrics.jsonl", workers)
    _write_jsonl(run_dir / "network_metrics.jsonl", network)
    _write_csv(
        run_dir / "matrix.csv",
        point_rows,
        _matrix_columns(point_rows),
    )
    _write_csv(
        run_dir / "scaling.csv",
        scaling_rows,
        list(scaling_rows[0])
        if scaling_rows
        else [
            "node_count",
            "concurrent_requests",
            "throughput",
        ],
    )
    _write_csv(
        run_dir / "replica_distribution.csv",
        replica_rows,
        list(replica_rows[0])
        if replica_rows
        else [
            "point_id",
            "repeat",
            "worker_id",
            "stage_id",
            "operations",
        ],
    )
    _write_csv(
        run_dir / "transport_breakdown.csv",
        transport_rows,
        list(transport_rows[0])
        if transport_rows
        else [
            "point_id",
            "repeat",
            "data_plane_mode",
        ],
    )
    _write_csv(
        run_dir / "capacity_prediction.csv",
        capacity_rows,
        list(capacity_rows[0])
        if capacity_rows
        else [
            "point_id",
            "repeat",
            "predicted_throughput",
        ],
    )
    _write_csv(
        run_dir / "latency.csv",
        requests,
        [
            "point_id",
            "repeat",
            "request_id",
            "time_to_first_token_s",
            "decode_tokens_s",
            "end_to_end_s",
            "queueing_s",
            "network_s",
            "stage_execution_s",
            "admission_time_ms",
            "route_reservation_time_ms",
        ],
    )
    _write_csv(
        run_dir / "failures.csv",
        failures,
        [
            "point_id",
            "repeat",
            "event_type",
            "request_id",
            "worker_id",
            "stage_id",
            "detail",
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
    generate_experiment_001_charts(
        chart_dir=run_dir / "charts",
        scaling_rows=scaling_rows,
        aggregate_points=aggregate_points,
        replica_rows=replica_rows,
        transport_rows=transport_rows,
        capacity_rows=capacity_rows,
        stage_rows=stages,
    )
    return render_html_report(
        run_dir=run_dir,
        summary=summary,
        scaling_rows=scaling_rows,
        request_rows=requests,
    )


async def run_loopback_matrix(
    config: ExperimentConfig,
    *,
    repeats: int | None = None,
    duration_s: float | None = None,
    progress_callback: ProgressCallback | None = print,
) -> ExperimentRun:
    """Run, preserve, and truthfully evaluate the replicated-stage matrix."""

    if config.execution_mode != ExecutionMode.SINGLE_HOST_LOOPBACK:
        raise ValueError("loopback matrix requires execution_mode=single-host-loopback")
    resolved_repeats = (
        repeats
        if repeats is not None
        else config.matrix.repeats
        if config.matrix is not None
        else 3
    )
    if resolved_repeats < 1:
        raise ValueError("repeats must be at least one")
    worker_counts = sorted(set(int(value) for value in config.node_counts))
    concurrency_counts = sorted(set(int(value) for value in config.concurrent_request_counts))
    if len(worker_counts) < 2:
        raise ValueError("loopback matrix requires at least two worker counts")
    if not concurrency_counts:
        raise ValueError("loopback matrix requires concurrency levels")
    if min(worker_counts) < config.model.stage_count:
        raise ValueError("every point must cover every model stage")
    measured_duration_s = duration_s or config.steady_state_s
    if measured_duration_s <= 0:
        raise ValueError("measurement duration must be positive")

    resolved_config = config.model_copy(deep=True)
    start = datetime.now(UTC)
    run_id = f"{start.strftime('%Y%m%dT%H%M%SZ')}-loopback-matrix-{uuid4().hex[:8]}"
    run_dir = Path(resolved_config.output_root).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "charts").mkdir()
    point_root = run_dir / "points"
    point_root.mkdir()
    environment_before = _environment_snapshot(
        resolved_config,
        start=start,
    )
    _write_json(run_dir / "environment_before.json", environment_before)

    if resolved_config.synthetic_compute.mode == "calibrated_cpu":
        if progress_callback is not None:
            progress_callback("Calibrating synthetic stage kernel...")
        cpu_ids = available_cpu_ids()
        calibration = calibrate_synthetic_compute(
            resolved_config.synthetic_compute,
            cpu_id=cpu_ids[0] if cpu_ids else None,
        )
        if not calibration.acceptable:
            raise RuntimeError(
                "synthetic calibration median "
                f"{calibration.median_stage_ms:.3f} ms is outside "
                f"[{resolved_config.synthetic_compute.acceptable_min_ms}, "
                f"{resolved_config.synthetic_compute.acceptable_max_ms}] ms"
            )
        resolved_config.synthetic_compute.work_units = calibration.work_units
        resolved_config.model.cpu_work_units = calibration.work_units
        resolved_config.model.cpu_kernel_buffer_bytes = (
            resolved_config.synthetic_compute.activation_bytes
        )
        calibration_payload = {
            "mode": "calibrated_cpu",
            **calibration.to_dict(),
            "frozen_across_matrix": True,
        }
        if progress_callback is not None:
            progress_callback(
                "Calibration result: "
                f"{calibration.median_stage_ms:.1f} ms median, "
                f"work_units={calibration.work_units}"
            )
    else:
        resolved_config.model.cpu_work_units = resolved_config.synthetic_compute.work_units or 0
        calibration_payload = {
            "mode": resolved_config.synthetic_compute.mode,
            "work_units": resolved_config.model.cpu_work_units,
            "frozen_across_matrix": True,
            "acceptable": (resolved_config.synthetic_compute.mode == "transport_only"),
        }
    _write_json(run_dir / "calibration.json", calibration_payload)

    point_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    workers: list[dict[str, Any]] = []
    network: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    child_runs: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    profile_result: dict[str, Any] | None = None
    total_points = len(worker_counts) * len(concurrency_counts) * resolved_repeats
    point_number = 0

    try:
        for worker_count in worker_counts:
            for concurrency in concurrency_counts:
                for repeat in range(1, resolved_repeats + 1):
                    point_number += 1
                    point_id = f"n{worker_count}-c{concurrency}-r{repeat}"
                    if progress_callback is not None:
                        progress_callback(
                            f"Starting matrix point {point_number}/{total_points}: "
                            f"workers={worker_count} concurrency={concurrency} "
                            f"repeat={repeat}"
                        )
                    events.append(
                        {
                            "sequence": len(events),
                            "event_type": "matrix_point_started",
                            "point_id": point_id,
                            "worker_count": worker_count,
                            "concurrent_requests": concurrency,
                            "repeat": repeat,
                        }
                    )
                    point_config = resolved_config.model_copy(deep=True)
                    point_config.matrix = None
                    point_config.name = f"{resolved_config.name}-{point_id}"
                    point_config.node_counts = []
                    point_config.concurrent_request_counts = []
                    point_config.workload.concurrent_requests = concurrency
                    point_config.output_root = str(point_root)
                    point_config.profiling.enabled = bool(
                        resolved_config.profiling.enabled
                        and worker_count == max(worker_counts)
                        and concurrency == max(concurrency_counts)
                        and repeat == 1
                    )
                    child = await run_loopback_experiment(
                        point_config,
                        worker_count=worker_count,
                        sustained=True,
                        duration_s=measured_duration_s,
                        progress_callback=progress_callback,
                    )
                    errors = validate_run(child.run_dir)
                    child_profile = child.run_dir / "profile.json"
                    if child_profile.exists():
                        profile_result = {
                            "point_id": point_id,
                            "child_run_id": child.run_id,
                            **json.loads(child_profile.read_text(encoding="utf-8")),
                        }
                        _write_json(run_dir / "profile.json", profile_result)
                    validation_errors.extend(f"{point_id}: {error}" for error in errors)
                    primary = dict(child.summary["primary_result"])
                    primary["committed_token_correctness"] = (
                        1.0
                        if primary["completion_fraction"] == 1.0 and primary["failed_requests"] == 0
                        else 0.0
                    )
                    primary.update(
                        {
                            "point_id": point_id,
                            "repeat": repeat,
                            "child_run_id": child.run_id,
                            "child_run_dir": str(child.run_dir.relative_to(run_dir)),
                        }
                    )
                    point_rows.append(primary)
                    child_runs.append(
                        {
                            "point_id": point_id,
                            "worker_count": worker_count,
                            "concurrent_requests": concurrency,
                            "repeat": repeat,
                            "run_id": child.run_id,
                            "run_dir": str(child.run_dir.relative_to(run_dir)),
                            "report": str(child.report_path.relative_to(run_dir)),
                            "artifact_validation_errors": errors,
                            "data_plane_mode": resolved_config.data_plane.value,
                        }
                    )
                    (
                        sampled_requests,
                        child_stages,
                        child_workers,
                        child_network,
                        child_failures,
                    ) = _annotated_rows(
                        child,
                        point_id=point_id,
                        repeat=repeat,
                    )
                    requests.extend(sampled_requests)
                    stages.extend(child_stages)
                    workers.extend(child_workers)
                    network.extend(child_network)
                    failures.extend(child_failures)
                    events.append(
                        {
                            "sequence": len(events),
                            "event_type": "matrix_point_completed",
                            "point_id": point_id,
                            "worker_count": worker_count,
                            "concurrent_requests": concurrency,
                            "repeat": repeat,
                            "throughput": primary["aggregate_verified_output_tokens_s"],
                            "completion_fraction": primary["completion_fraction"],
                            "child_run_id": child.run_id,
                        }
                    )
                    if progress_callback is not None:
                        progress_callback(
                            f"Completed point {point_number}/{total_points}: "
                            f"{primary['aggregate_verified_output_tokens_s']:.1f} "
                            "verified tokens/s"
                        )
                        progress_callback(f"Child artifact: {child.run_dir}")
    except BaseException as exc:
        end = datetime.now(UTC)
        environment_after = _environment_snapshot(
            resolved_config,
            start=start,
            end=end,
        )
        dependency_diff = _dependency_diff(
            environment_before,
            environment_after,
        )
        _write_json(run_dir / "environment_after.json", environment_after)
        _write_json(run_dir / "dependency_diff.json", dependency_diff)
        partial_summary = {
            "schema_version": "2",
            "run_id": run_id,
            "execution_mode": resolved_config.execution_mode.value,
            "data_plane": resolved_config.data_plane.value,
            "backend": resolved_config.backend,
            "status": "FAIL",
            "experiment_integrity_status": "FAIL",
            "correctness_status": "FAIL",
            "direct_data_plane_status": "FAIL",
            "replica_utilisation_status": "FAIL",
            "capacity_prediction_status": "FAIL",
            "scaling_hypothesis_status": "FAIL",
            "overall_status": "FAIL",
            "partial": True,
            "failure": f"{type(exc).__name__}: {exc}",
            "start_timestamp": start.isoformat(),
            "end_timestamp": end.isoformat(),
            "primary_result": point_rows[-1] if point_rows else {},
            "baseline_result": point_rows[0] if point_rows else {},
            "matrix_results": _aggregate_points(point_rows),
            "raw_point_results": point_rows,
            "acceptance_criteria": [],
            "failed_acceptance_criteria": ["integrity:complete_matrix"],
            "child_runs": child_runs,
            "environment_after": environment_after,
            "profile": profile_result,
        }
        _write_json(run_dir / "summary.json", partial_summary)
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_csv(
            run_dir / "matrix.csv",
            point_rows,
            _matrix_columns(point_rows),
        )
        raise

    end = datetime.now(UTC)
    environment_after = _environment_snapshot(
        resolved_config,
        start=start,
        end=end,
    )
    dependency_diff = _dependency_diff(
        environment_before,
        environment_after,
    )
    _write_json(run_dir / "environment_after.json", environment_after)
    _write_json(run_dir / "dependency_diff.json", dependency_diff)
    aggregate_points = _aggregate_points(point_rows)
    scaling_rows = _scaling_rows(aggregate_points)
    statuses, criteria, status_evidence = evaluate_matrix_statuses(
        config=resolved_config,
        point_rows=point_rows,
        worker_counts=worker_counts,
        concurrency_counts=concurrency_counts,
        repeats=resolved_repeats,
        measured_duration_s=measured_duration_s,
        child_validation_errors=validation_errors,
        dependency_mutation_failure=bool(dependency_diff["mutation_failure"]),
    )
    primary_concurrency = max(concurrency_counts)
    primary_candidates = [
        row
        for row in aggregate_points
        if int(row["concurrent_request_count"]) == primary_concurrency
    ]
    primary = max(
        primary_candidates,
        key=lambda row: int(row["node_count"]),
    )
    baseline = min(
        primary_candidates,
        key=lambda row: int(row["node_count"]),
    )
    failed = [item["name"] for item in criteria if item["status"] == "FAIL"]
    summary = {
        "schema_version": "2",
        "run_id": run_id,
        "execution_mode": resolved_config.execution_mode.value,
        "data_plane": resolved_config.data_plane.value,
        "backend": resolved_config.backend,
        "values": (
            "Measured deterministic synthetic CPU execution across "
            "process-isolated workers on one host with loopback transport. "
            "This is single-host-loopback evidence, not physical distributed "
            "scaling."
        ),
        "model_id": resolved_config.model_id,
        "model_revision": resolved_config.model_revision,
        "seed": resolved_config.seed,
        "start_timestamp": start.isoformat(),
        "end_timestamp": end.isoformat(),
        "primary_metric_definition": (
            "median committed verified output tokens per measured second across three repeats"
        ),
        "primary_result": primary,
        "baseline_result": baseline,
        "historical_baseline": _historical_baseline(),
        "matrix_results": aggregate_points,
        "raw_point_results": point_rows,
        "status_evidence": status_evidence,
        "acceptance_criteria": criteria,
        **statuses,
        "status": statuses["overall_status"],
        "failed_acceptance_criteria": failed,
        "limitations": [
            (
                "All workers share one physical host, memory subsystem, "
                "operating system, and loopback network stack."
            ),
            ("Calibrated deterministic CPU work is not real-model kernel performance evidence."),
            (
                "Routes remain fixed for a request lifetime; admissions are "
                "balanced atomically and generation changes are reserved for "
                "failure recovery."
            ),
            "Aggregate throughput is not single-request generation speed.",
        ],
        "child_runs": child_runs,
        "calibration": calibration_payload,
        "dependency_diff": dependency_diff,
        "environment_after": environment_after,
        "profile": profile_result,
    }
    report = _write_parent_artifacts(
        run_dir=run_dir,
        config=resolved_config,
        summary=summary,
        events=events,
        requests=requests,
        stages=stages,
        workers=workers,
        network=network,
        failures=failures,
        point_rows=point_rows,
        scaling_rows=scaling_rows,
        worker_counts=worker_counts,
        concurrency_counts=concurrency_counts,
        repeats=resolved_repeats,
        child_runs=child_runs,
    )
    _write_artifact_manifest(run_dir)
    return ExperimentRun(
        run_id=run_id,
        run_dir=run_dir,
        report_path=report,
        passed=statuses["overall_status"] == "PASS",
        summary=summary,
    )
