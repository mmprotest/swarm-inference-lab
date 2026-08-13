"""Ordered physical shard replay and preregistered model validation."""

from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_020.replay import (
    _operator_wall,
    _span_wall,
    _tasks,
)
from swarm_inference.experiments.experiment_020.sharded_graph import E020ShardedK3Graph
from swarm_inference.experiments.experiment_020.simulation import (
    measured_service,
    validate_single_resource_rows,
)
from swarm_inference.experiments.experiment_020.transport import (
    Frame,
    MessageType,
    decode_frame,
    encode_frame,
    new_run_credential,
)

from .io import atomic_write_json, write_csv

HIDDEN = 7168
LAYERS = 93


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_l2(expected: np.ndarray, actual: np.ndarray) -> float:
    left = np.asarray(expected, dtype=np.float64).reshape(-1)
    right = np.asarray(actual, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-30))


def _timing_p50(value: Mapping[str, Any]) -> float:
    return float(value.get("p50_ms", value.get("median_ms", value.get("p50", 0.0))))


def _physical_inputs(repo: Path) -> dict[str, dict[str, Any]]:
    e019 = repo / "artifacts" / "experiment-019"
    e020 = repo / "artifacts" / "experiment-020"
    return {
        "attention": _read(e020 / "physical" / "attention-raw.json"),
        "legacy_expert": _read(e019 / "physical" / "expert-stripe-raw.json"),
        "grouped_calibration": _read(
            e020 / "physical" / "expert-grouped-robust-calibration.json"
        ),
        "grouped_heldout": _read(
            e020 / "physical" / "expert-grouped-robust-heldout.json"
        ),
        "other_calibration": _read(
            e020 / "physical" / "other-shards-robust-calibration.json"
        ),
        "other_heldout": _read(
            e020 / "physical" / "other-shards-robust-heldout.json"
        ),
        "protocol_calibration": _read(
            e020 / "runtime" / "protocol-heldout-raw.json"
        ),
        "protocol_heldout": _read(
            e020 / "runtime" / "protocol-validation-raw.json"
        ),
    }


def build_validation_service(repo: Path) -> Any:
    values = _physical_inputs(repo)
    return measured_service(
        values["attention"],
        values["legacy_expert"],
        values["grouped_calibration"],
        values["other_calibration"],
        values["protocol_calibration"],
        degree=8,
        rows=1,
    )


class _GpuSampler:
    def __init__(self, interval_seconds: float = 5.0) -> None:
        self.interval_seconds = interval_seconds
        self.rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(10.0, self.interval_seconds * 2))

    def _sample(self) -> None:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return
        fields = [value.strip() for value in completed.stdout.strip().splitlines()[0].split(",")]
        if len(fields) != 7:
            return
        self.rows.append(
            {
                "sample_unix_ns": time.time_ns(),
                "gpu_timestamp": fields[0],
                "gpu_name": fields[1],
                "utilization_percent": fields[2],
                "memory_used_mib": fields[3],
                "memory_total_mib": fields[4],
                "power_watts": fields[5],
                "temperature_c": fields[6],
            }
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                self._sample()
            self._stop.wait(self.interval_seconds)
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            self._sample()


def _protocol_roundtrip(
    *,
    credential: bytes,
    message_type: MessageType,
    worker_id: str,
    layer: int,
    payload: bytes,
) -> int:
    frame = Frame(
        message_type,
        "e021-ordered-replay",
        0,
        worker_id,
        f"layer-{layer:03d}",
        payload,
    )
    encoded = encode_frame(frame, credential)
    decoded = decode_frame(encoded, credential)
    if decoded != frame:
        raise RuntimeError("authenticated local frame roundtrip changed the task")
    return len(encoded)


def _worker_services(
    layer_records: Mapping[int, Mapping[str, Any]],
    layer_operations: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer, record in sorted(layer_records.items()):
        attention = record["attention"]
        for worker in attention["workers"]:
            rows.append(
                {
                    "layer": layer,
                    "worker_id": worker["worker_id"],
                    "operator": f"{attention['attention_type']}_attention_stripe",
                    "duration_ms": worker["wall"]["p50_ms"],
                    "runtime_weight_bytes": worker["runtime_weight_bytes"],
                    "evidence_class": "PHYSICAL_SHARD_EXECUTION",
                }
            )
        for operation in layer_operations[layer]:
            if "duration_ms" not in operation:
                continue
            rows.append(
                {
                    "layer": layer,
                    "worker_id": operation.get("worker_id"),
                    "operator": operation.get("operator"),
                    "duration_ms": operation["duration_ms"],
                    "runtime_weight_bytes": operation.get("runtime_weight_bytes", 0),
                    "evidence_class": "PHYSICAL_SHARD_EXECUTION",
                }
            )
    return rows


def _max_operation(
    operations: Sequence[Mapping[str, Any]],
    operator: str,
) -> float:
    values = [
        float(row["duration_ms"])
        for row in operations
        if row.get("operator") == operator and "duration_ms" in row
    ]
    if not values:
        raise RuntimeError(f"ordered replay emitted no {operator!r} operation")
    return max(values)


def _shared_prediction(other: Mapping[str, Any], *, degree: int = 8, rows: int = 1) -> float:
    result = next(
        row
        for row in other["results"]
        if int(row["degree"]) == degree and int(row["rows"]) == rows
    )
    return max(
        _timing_p50(worker["duration"])
        for worker in result["operators"]["shared_expert"]["workers"]
    )


def run_ordered_physical_replay(
    repo: Path,
    artifact_root: Path,
    *,
    checkpoint: Path = Path("F:/models/Kimi-K3"),
    cuda_library: Path | None = None,
    shard_library: Path | None = None,
    grouped_library: Path | None = None,
    oracle_root: Path | None = None,
) -> dict[str, Any]:
    """Execute layers 0..8 contiguously and compare the event model unchanged."""

    cuda_library = cuda_library or (
        repo / "artifacts" / "experiment-016" / "cuda" / "coli_cuda-sm120-h016-final.dll"
    )
    shard_library = shard_library or (
        repo
        / "artifacts"
        / "experiment-019"
        / "physical"
        / "exp019-kda-shard-sm120-v2.dll"
    )
    grouped_library = grouped_library or (
        repo
        / "artifacts"
        / "experiment-020"
        / "physical"
        / "e020-grouped-top16-sm120.dll"
    )
    oracle_root = oracle_root or (
        repo / "artifacts" / "experiment-014" / "oracle-full-93-idot0"
    )
    values = _physical_inputs(repo)
    service = measured_service(
        values["attention"],
        values["legacy_expert"],
        values["grouped_calibration"],
        values["other_calibration"],
        values["protocol_calibration"],
        degree=8,
        rows=1,
    )
    predicted_tasks = _tasks(service, values["protocol_calibration"])
    credential = new_run_credential()
    trace = np.memmap(
        oracle_root / "hidden-trace.f32",
        mode="r",
        dtype="<f4",
        shape=(3 * (LAYERS + 1), HIDDEN),
    )
    graph = E020ShardedK3Graph(
        checkpoint.resolve(),
        cuda_library.resolve(),
        shard_library.resolve(),
        grouped_library.resolve(),
        degree=8,
        depth_span=8,
    )
    sampler = _GpuSampler()
    sampler.start()
    layer_records: dict[int, dict[str, Any]] = {}
    layer_operations: dict[int, list[dict[str, Any]]] = {}
    layer_walls: dict[int, float] = {}
    cumulative_walls: dict[int, float] = {}
    protocol_bytes = 0
    hidden_errors: list[dict[str, Any]] = []
    try:
        hidden, _embedding = graph.embedding(163584)
        hidden = hidden[0]
        residuals: list[np.ndarray] = []
        # Layer zero establishes the exact recurrent/AttnRes boundary but is
        # outside the preregistered layer-1..8 mixed-span timer.
        hidden, residuals, _layer_zero = graph.execute_layer(0, hidden, residuals)
        hidden_errors.append(
            {
                "layer": 0,
                "relative_l2_error": _relative_l2(trace[0], hidden),
            }
        )
        span_started = time.perf_counter_ns()
        for layer in range(1, 9):
            operations_before = len(graph.worker_operations)
            started = time.perf_counter_ns()
            protocol_bytes += _protocol_roundtrip(
                credential=credential,
                message_type=MessageType.EXECUTE_SHARD,
                worker_id=f"depth-group-{layer // 8:03d}.controller-batch",
                layer=layer,
                payload=json.dumps(
                    {
                        "operator": "ordered_exact_shard_layer_batch",
                        "layer": layer,
                        "stripe_degree": 8,
                        "whole_layer_fallback": False,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
            hidden, residuals, record = graph.execute_layer(layer, hidden, residuals)
            protocol_bytes += _protocol_roundtrip(
                credential=credential,
                message_type=MessageType.SHARD_RESULT,
                worker_id=f"depth-group-{layer // 8:03d}.controller-batch",
                layer=layer,
                payload=np.ascontiguousarray(hidden, dtype=np.float32).tobytes(),
            )
            wall_ms = (time.perf_counter_ns() - started) / 1e6
            layer_walls[layer] = wall_ms
            cumulative_walls[layer] = (time.perf_counter_ns() - span_started) / 1e6
            layer_records[layer] = record
            layer_operations[layer] = [
                dict(value) for value in graph.worker_operations[operations_before:]
            ]
            hidden_errors.append(
                {
                    "layer": layer,
                    "relative_l2_error": _relative_l2(trace[layer], hidden),
                }
            )
    finally:
        sampler.stop()
        loader_audit = list(graph.loader.audit)
        graph.close()

    rows: list[dict[str, Any]] = []

    def add_row(
        workload: str,
        predicted_ms: float,
        actual_ms: float,
        compute_tasks: int,
        network_tasks: int,
        source: str,
    ) -> None:
        rows.append(
            {
                "workload": workload,
                "predicted_sharded_wall_ms": predicted_ms,
                "actual_sharded_wall_ms": actual_ms,
                "absolute_percentage_error": abs(predicted_ms - actual_ms) / actual_ms,
                "compute_task_count": compute_tasks,
                "network_task_count": network_tasks,
                "physical_actual_source": source,
                "normalization_applied": False,
                "post_hoc_multiplier": "",
                "evidence_class": "PHYSICAL_SHARD_EXECUTION",
            }
        )

    add_row(
        "grouped_expert_stripe",
        _operator_wall(predicted_tasks, 1, "expert_stripe_local_top16_accumulation"),
        _max_operation(layer_operations[1], "grouped_expert_stripe_bank_top16"),
        1,
        0,
        "native grouped top-16 call inside the contiguous layer-1 replay",
    )
    add_row(
        "KDA_stripe",
        _operator_wall(predicted_tasks, 1, "KDA_head_projection_stripe"),
        max(float(row["wall"]["p50_ms"]) for row in layer_records[1]["attention"]["workers"]),
        1,
        0,
        "native KDA stripe call inside the contiguous layer-1 replay",
    )
    add_row(
        "MLA_stripe",
        _operator_wall(predicted_tasks, 3, "Gated_MLA_head_projection_stripe"),
        max(float(row["wall"]["p50_ms"]) for row in layer_records[3]["attention"]["workers"]),
        1,
        0,
        "native MLA stripe call inside the contiguous layer-3 replay",
    )
    add_row(
        "projection_stripe",
        _operator_wall(predicted_tasks, 1, "latent_down_row_projection_stripe"),
        _max_operation(layer_operations[1], "latent_down_projection_stripe"),
        1,
        0,
        "native latent-down projection call inside the contiguous layer-1 replay",
    )
    add_row(
        "shared_expert_stripe",
        _shared_prediction(values["other_calibration"]),
        _max_operation(layer_operations[1], "shared_expert_stripe"),
        1,
        0,
        "native shared-expert stripe call inside the contiguous layer-1 replay",
    )
    definitions = [
        ("complete_sharded_KDA_layer", {1}, layer_walls[1]),
        ("complete_sharded_MLA_layer", {3}, layer_walls[3]),
        ("mixed_2_layer_span", set(range(1, 3)), cumulative_walls[2]),
        ("mixed_4_layer_span", set(range(1, 5)), cumulative_walls[4]),
        ("mixed_8_layer_span", set(range(1, 9)), cumulative_walls[8]),
    ]
    for workload, layers, actual in definitions:
        predicted, compute_count, network_count = _span_wall(predicted_tasks, layers)
        add_row(
            workload,
            predicted,
            actual,
            compute_count,
            network_count,
            (
                "one contiguous wall-clock interval around ordered real K3 shard "
                "execution plus authenticated local batch frames; no compute overlap"
            ),
        )
    error_validation = validate_single_resource_rows(rows)
    residency_match = False
    numerical_correctness = max(
        float(row["relative_l2_error"]) for row in hidden_errors
    ) <= 2e-5
    status = (
        "PASS"
        if error_validation["status"] == "PASS"
        and residency_match
        and numerical_correctness
        else "FAIL"
    )
    worker_services = _worker_services(layer_records, layer_operations)
    write_csv(artifact_root / "validation" / "ordered-shard-replay.csv", rows)
    write_csv(artifact_root / "validation" / "heldout-service.csv", worker_services)
    write_csv(artifact_root / "physical" / "worker-services.csv", worker_services)
    write_csv(artifact_root / "physical" / "gpu-samples.csv", sampler.rows)
    ordered = {
        "schema_version": "experiment-021-ordered-physical-replay-v1",
        "status": status,
        "model_validation": error_validation,
        "model_validation_error_gate_status": error_validation["status"],
        "same_compute_resource": True,
        "physical_compute_resources": 1,
        "compute_overlap": False,
        "operation_order_preserved": True,
        "physical_equivalent_workload_executed": True,
        "same_residency_policy_as_headline_model": residency_match,
        "residency_mismatch": (
            "The current native graph loads and uploads active shard material during "
            "the timed path; the headline event model assumes complete per-worker "
            "expert banks and other weights are already resident."
        ),
        "normalization_applied": False,
        "global_multiplier": None,
        "post_hoc_multiplier": None,
        "authenticated_protocol_roundtrip_bytes": protocol_bytes,
        "hidden_correctness": hidden_errors,
        "hidden_correctness_status": "PASS" if numerical_correctness else "FAIL",
        "maximum_hidden_relative_l2_error": max(
            float(row["relative_l2_error"]) for row in hidden_errors
        ),
        "rows": rows,
        "layer_walls_ms": layer_walls,
        "cumulative_span_walls_ms": cumulative_walls,
        "worker_operation_count": len(worker_services),
        "all_compute_operations_have_worker_id": all(
            bool(row.get("worker_id")) for row in worker_services
        ),
        "direct_read_request_count": len(loader_audit),
        "direct_read_bytes": sum(int(row["bytes_read"]) for row in loader_audit),
        "direct_read_audit_sha256": hashlib.sha256(
            json.dumps(loader_audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "decisive_failure": (
            None
            if status == "PASS"
            else "ordered physical path and headline event model do not represent the same resident-worker runtime"
        ),
    }
    atomic_write_json(artifact_root / "physical" / "ordered-workloads.json", ordered)
    return ordered


def validation_percentages(receipt: Mapping[str, Any]) -> dict[str, float]:
    model = receipt["model_validation"]
    return {
        "median_percent": 100 * float(model["median_error"]),
        "p90_percent": 100 * float(model["p90_error"]),
        "maximum_percent": 100 * float(model["maximum_error"]),
    }


__all__ = [
    "build_validation_service",
    "run_ordered_physical_replay",
    "validation_percentages",
]
