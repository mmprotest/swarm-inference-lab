"""Physical chunk and residual gates for the Experiment 022 completion pass."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _array_fingerprint
from swarm_inference.execution.kimi_k3_graph_runtime import KimiCudaGraphRunner
from swarm_inference.experiments.experiment_019.physical import ROUTED_EXPERTS
from swarm_inference.experiments.experiment_020.transport import (
    Frame,
    MessageType,
    decode_frame,
    encode_frame,
)
from swarm_inference.experiments.experiment_022.event_model import (
    DeterministicWorkerEventEngine,
    EventTask,
)
from swarm_inference.experiments.experiment_022.resident_replay import (
    replay_resident_layer,
)

HIDDEN = 7168
LAYERS = 93
TOKEN_IDS = (163584, 18699, 11)
CALIBRATION = {"KDA": 45, "Gated_MLA": 47}
HELD_OUT = {"KDA": 89, "Gated_MLA": 91}
CHUNKS = (1, 2, 4)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _file_receipt(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _trace_rows(trace: np.memmap[Any, Any], boundary: int, rows: int) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack([trace[(row % 3) * (LAYERS + 1) + boundary] for row in range(rows)]),
        dtype=np.float32,
    )


def _whole_layer_reference(
    checkpoint: Path,
    cuda_library: Path,
    oracle_root: Path,
    *,
    layer: int,
    rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute the identical isolated layer workload through the whole K3 path."""

    trace = np.memmap(
        oracle_root / "hidden-trace.f32",
        mode="r",
        dtype="<f4",
        shape=(3 * (LAYERS + 1), HIDDEN),
    )
    runner = KimiCudaGraphRunner(checkpoint, cuda_library)
    try:
        token_ids = [TOKEN_IDS[row % len(TOKEN_IDS)] for row in range(rows)]
        embeddings, embedding_record = runner.embed(token_ids)
        hidden = _trace_rows(trace, layer - 1, rows)
        residuals = np.zeros((rows, 8, HIDDEN), dtype=np.float32)
        snapshots = list(range(0, layer, 12))
        for slot, snapshot in enumerate(snapshots):
            residuals[:, slot] = (
                embeddings
                if snapshot == 0
                else _trace_rows(trace, snapshot - 1, rows)
            )
        output, next_count, record = runner.execute_layer(
            layer,
            hidden,
            residuals,
            len(snapshots),
            list(range(rows)),
            maximum_context=256,
        )
        routes = [
            [int(value) for value in row["selected_expert_ids"]]
            for row in record["routes"]
        ]
        return output, {
            "evidence_class": "PHYSICAL",
            "implementation": record["backend_identity"],
            "layer": layer,
            "rows": rows,
            "attention_type": record["attention_type"],
            "output_fingerprint": _array_fingerprint(output),
            "state_output": record["state_output"],
            "routes": routes,
            "next_block_count": next_count,
            "embedding_fingerprint": embedding_record["output_fingerprint"],
            "timing": record["timing"],
            "correctness_control_only": True,
            "excluded_from_shard_service_timing": True,
        }
    finally:
        runner.close()


class _GpuSampler:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        query = (
            "timestamp,name,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw,clocks.sm"
        )
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--query-gpu={query}",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                fields = [value.strip() for value in completed.stdout.strip().split(",")]
                if len(fields) == 8:
                    self.rows.append(
                        {
                            "sample_wall_time_ns": time.time_ns(),
                            "nvidia_timestamp": fields[0],
                            "gpu_name": fields[1],
                            "gpu_utilization_percent": fields[2],
                            "memory_used_mib": fields[3],
                            "memory_total_mib": fields[4],
                            "temperature_c": fields[5],
                            "power_w": fields[6],
                            "sm_clock_mhz": fields[7],
                        }
                    )
            except (OSError, subprocess.SubprocessError):
                self.rows.append(
                    {
                        "sample_wall_time_ns": time.time_ns(),
                        "error": "nvidia-smi sample unavailable",
                    }
                )
            self._stop.wait(0.5)


def _protocol_overhead(
    *, input_bytes: int, output_bytes: int, repeats: int = 41
) -> dict[str, float]:
    credential = hashlib.sha256(b"e022-completion-protocol-profile").digest()
    request = Frame(
        MessageType.EXECUTE_SHARD,
        "protocol-profile",
        0,
        "worker-profile",
        "state-profile",
        bytes(input_bytes),
    )
    response = Frame(
        MessageType.SHARD_RESULT,
        "protocol-profile",
        0,
        "worker-profile",
        "state-profile",
        bytes(output_bytes),
    )
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        decode_frame(encode_frame(request, credential), credential)
        decode_frame(encode_frame(response, credential), credential)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p90_ms": float(np.percentile(samples, 90)),
        "minimum_ms": min(samples),
    }


def _candidate_class(operator: str) -> str | None:
    if operator.startswith("KDA_attention_stripe"):
        return "KDA_SHARD"
    if operator.startswith("Gated_MLA_attention_stripe"):
        return "MLA_SHARD"
    if "expert_stripe_bank" in operator and "reduction" not in operator:
        return "EXPERT_STRIPE"
    if operator.startswith("shared_expert_stripe") and "reduction" not in operator:
        return "SHARED_EXPERT_SHARD"
    if (
        operator.startswith("latent_down_projection_stripe")
        or operator.startswith("latent_up_projection_stripe")
    ) and "reduction" not in operator:
        return "PROJECTION_SHARD"
    if operator.endswith("native_reduction"):
        return "REDUCTION_CONTRIBUTION"
    return None


def _service_rows(receipts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for receipt in receipts:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for records in receipt["operation_records"]:
            for row in records:
                candidate = _candidate_class(str(row.get("operator", "")))
                if candidate is not None:
                    grouped[(candidate, str(row.get("worker_id", "")))].append(row)
        rows = int(receipt["rows"])
        boundary_bytes = rows * HIDDEN * 4
        protocol = _protocol_overhead(
            input_bytes=boundary_bytes,
            output_bytes=boundary_bytes,
        )
        for (candidate, worker), records in grouped.items():
            wall = [float(row["duration_ms"]) for row in records]
            cuda = [float(row.get("cuda_ms", 0.0)) for row in records]
            launches = [int(row.get("physical_launches", 0)) for row in records]
            resident_weight_bytes = max(
                int(row.get("runtime_weight_bytes", 0)) for row in records
            )
            persistent_state_bytes = max(
                int(row.get("persistent_state_bytes", 0)) for row in records
            )
            output.append(
                {
                    "evidence_class": "PHYSICAL",
                    "layer": receipt["layer"],
                    "attention_type": receipt["attention_type"],
                    "candidate_type": candidate,
                    "worker_id": worker,
                    "partition_degree": receipt["degree"],
                    "chunk_rows": rows,
                    "direct_native_service_ms": statistics.median(wall),
                    "cuda_wall_ms": statistics.median(cuda),
                    "host_wall_ms": statistics.median(wall),
                    "host_minus_cuda_ms": max(
                        0.0, statistics.median(wall) - statistics.median(cuda)
                    ),
                    "physical_launches": int(statistics.median(launches)),
                    "resident_weight_bytes": resident_weight_bytes,
                    "persistent_state_bytes": persistent_state_bytes,
                    "resident_bytes": resident_weight_bytes
                    + persistent_state_bytes,
                    "worker_protocol_overhead_ms": protocol["p50_ms"],
                    "protocol_p90_ms": protocol["p90_ms"],
                    "checkpoint_reads_in_timed_region": 0,
                    "state_mutated": candidate in {"KDA_SHARD", "MLA_SHARD"},
                    "complete_dag_correctness": receipt["status"],
                    "relative_l2": receipt["correctness"]["relative_l2_error"],
                    "routes_exact": receipt["routes_exact_against_physical_reference"],
                    "state_exact": receipt["state_exact_against_physical_reference"],
                    "no_nan_inf": receipt["all_output_values_finite"],
                    "production_native_binding": True,
                    "service_source": "resident ordered K3 shard DAG operation record",
                }
            )
        output.append(
            {
                "evidence_class": "PHYSICAL",
                "layer": receipt["layer"],
                "attention_type": receipt["attention_type"],
                "candidate_type": "COMPLETE_SHARDED_LAYER_DAG",
                "worker_id": "single-gpu-ordered-replay",
                "partition_degree": receipt["degree"],
                "chunk_rows": rows,
                "direct_native_service_ms": receipt["wall"]["p50_ms"],
                "cuda_wall_ms": "",
                "host_wall_ms": receipt["wall"]["p50_ms"],
                "host_minus_cuda_ms": "",
                "physical_launches": int(
                    statistics.median(
                        int(row["launch_count"]) for row in receipt["instrumentation"]
                    )
                ),
                "resident_weight_bytes": "",
                "persistent_state_bytes": "",
                "resident_bytes": int(receipt["resident_total_bytes"])
                - int(receipt["resident_free_bytes_after_prepare"]),
                "worker_protocol_overhead_ms": protocol["p50_ms"],
                "protocol_p90_ms": protocol["p90_ms"],
                "checkpoint_reads_in_timed_region": receipt["timed_checkpoint_reads"],
                "state_mutated": True,
                "complete_dag_correctness": receipt["status"],
                "relative_l2": receipt["correctness"]["relative_l2_error"],
                "routes_exact": receipt["routes_exact_against_physical_reference"],
                "state_exact": receipt["state_exact_against_physical_reference"],
                "no_nan_inf": receipt["all_output_values_finite"],
                "production_native_binding": True,
                "service_source": "resident ordered complete sharded K3 layer DAG",
            }
        )
    return output


def _timer_harness_ms(event_count: int) -> float:
    samples: list[float] = []
    for _ in range(2001):
        started = time.perf_counter_ns()
        time.perf_counter_ns()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return statistics.median(samples) * max(1, event_count)


def _ledger_for_sample(
    receipt: dict[str, Any], index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    outer = float(receipt["wall_samples_ms"][index])
    all_records = receipt["operation_records"][index]
    experiment_records = [
        row
        for row in all_records
        if str(row.get("cost_classification", "")) == "EXPERIMENT_ONLY"
    ]
    records = [row for row in all_records if row not in experiment_records]
    instrument = receipt["instrumentation"][index]
    phase_wall = {
        str(name): float(value)
        for name, value in dict(instrument.get("phase_wall_ms", {})).items()
    }
    measured_phase_wall = sum(phase_wall.values())
    cuda_total = sum(float(row.get("cuda_ms", 0.0)) for row in records)
    reduction_records = [
        row
        for row in records
        if str(row.get("operator", "")).endswith("native_reduction")
    ]
    reduction_cuda = sum(float(row.get("cuda_ms", 0.0)) for row in reduction_records)
    reduction_wall = sum(float(row.get("duration_ms", 0.0)) for row in reduction_records)
    parallel_groups: dict[str, list[float]] = defaultdict(list)
    parallel_wall_groups: dict[str, list[float]] = defaultdict(list)
    operation_wall_by_phase: dict[str, float] = defaultdict(float)
    experiment_operation_wall_by_phase: dict[str, float] = defaultdict(float)
    for row in records:
        phase = str(row.get("phase", "UNATTRIBUTED"))
        operation_wall_by_phase[phase] += float(row.get("duration_ms", 0.0))
        if "stripe_index" in row and float(row.get("cuda_ms", 0.0)) > 0:
            group = f"{phase}:{row.get('operator')}"
            parallel_groups[group].append(float(row["cuda_ms"]))
            parallel_wall_groups[group].append(
                float(row.get("duration_ms", row.get("cuda_ms", 0.0)))
            )
    for row in experiment_records:
        experiment_operation_wall_by_phase[
            str(row.get("phase", "UNATTRIBUTED"))
        ] += float(row.get("duration_ms", 0.0))
    device_serialization = sum(
        max(0.0, sum(values) - max(values))
        for values in parallel_groups.values()
        if len(values) > 1
    )
    total_worker_serialization = sum(
        max(0.0, sum(values) - max(values))
        for values in parallel_wall_groups.values()
        if len(values) > 1
    )
    # The operation records and phase clocks are nested measurements.  First
    # remove the measured excess work from executing independent stripe workers
    # sequentially on one GPU.  The remaining operation intervals are the
    # service an independent worker or reducer would actually pay.
    operation_wall = sum(float(row.get("duration_ms", 0.0)) for row in records)
    reduction_cuda = min(reduction_cuda, cuda_total)
    device_serialization = min(device_serialization, max(0.0, cuda_total - reduction_cuda))
    total_worker_serialization = min(
        total_worker_serialization, max(0.0, operation_wall - reduction_wall)
    )
    kernel_compute = max(0.0, cuda_total - reduction_cuda - device_serialization)

    parallel_phase_names = {
        "attention_workers_and_collective",
        "dense_mlp_workers_and_collective",
        "latent_down_workers",
        "expert_workers_and_collective",
        "latent_up_workers_and_collective",
        "shared_workers_and_collective",
    }
    experiment_phase_names = {"receipt_assembly"}
    phase_gap: dict[str, float] = {
        name: max(
            0.0,
            duration
            - operation_wall_by_phase.get(name, 0.0)
            - experiment_operation_wall_by_phase.get(name, 0.0),
        )
        for name, duration in phase_wall.items()
    }
    phase_overlap: dict[str, float] = {
        name: max(
            0.0,
            operation_wall_by_phase.get(name, 0.0)
            + experiment_operation_wall_by_phase.get(name, 0.0)
            - duration,
        )
        for name, duration in phase_wall.items()
    }
    parallel_phase_gap = sum(
        value for name, value in phase_gap.items() if name in parallel_phase_names
    )
    experiment_operation_wall = sum(
        float(row.get("duration_ms", 0.0)) for row in experiment_records
    )
    experiment_device_copy_wall = sum(
        float(row.get("device_copy_ms", 0.0)) for row in experiment_records
    )
    receipt_harness = sum(
        value for name, value in phase_gap.items() if name in experiment_phase_names
    )
    coordinator_gap = sum(
        value
        for name, value in phase_gap.items()
        if name not in parallel_phase_names and name not in experiment_phase_names
    )

    # Persistent non-kernel time is the exclusive remainder of the directly
    # timed critical-worker operation intervals plus coordinator-only phases.
    critical_operation_wall = max(
        0.0, operation_wall - reduction_wall - total_worker_serialization
    )
    raw_device_copy = max(
        0.0,
        float(instrument.get("h2d_wall_ms", 0.0))
        + float(instrument.get("d2h_wall_ms", 0.0))
        - experiment_device_copy_wall,
    )
    parallel_device_copy = min(parallel_phase_gap, raw_device_copy)
    parallel_harness = max(0.0, parallel_phase_gap - parallel_device_copy)
    harness = (
        experiment_operation_wall + receipt_harness + parallel_harness
    )
    remaining = max(0.0, critical_operation_wall - kernel_compute) + coordinator_gap

    def take(raw: float) -> float:
        nonlocal remaining
        value = min(max(0.0, raw), remaining)
        remaining -= value
        return value

    # Only sum-minus-max directly proves work that disappears when independent
    # workers execute concurrently.  An otherwise unattributed gap around the
    # in-process sequential replay loop is experiment-harness time, except for
    # device copies independently observed by the CUDA instrumentation.  It is
    # neither invented worker service nor an invented emulation speedup.
    serialization = total_worker_serialization
    device_copy = parallel_device_copy + take(
        max(0.0, raw_device_copy - parallel_device_copy)
    )
    launch_gap = take(float(instrument.get("launch_submit_ms", 0.0)))
    sync_extra = take(
        max(0.0, float(instrument.get("cuda_sync_wall_ms", 0.0)) - cuda_total)
    )
    # This is exclusive CPU time inside directly timed, non-overlapping phase
    # boundaries after nested CUDA/copy/submit/sync intervals are removed.  It
    # is not the unexplained outer-wall remainder and is never assigned to a
    # synthetic barrier.
    worker_local_host = remaining
    boundary_mismatch = max(0.0, outer - measured_phase_wall)
    boundary_overrun = max(0.0, measured_phase_wall - outer)
    overlap_mismatch = sum(phase_overlap.values())
    # A nested operation interval that is materially longer than its enclosing
    # phase cannot be decomposed honestly without start/end interval data.  Do
    # not hide that inconsistency by charging it to a production cost class.
    # The whole outer interval becomes unexplained and the sample fails.
    timer_tolerance_ms = max(0.01, outer * 0.005)
    inconsistent_boundaries = (
        overlap_mismatch > timer_tolerance_ms
        or boundary_overrun > timer_tolerance_ms
    )
    unexplained = outer if inconsistent_boundaries else boundary_mismatch
    components = {
        "kernel_compute_ms": kernel_compute,
        "launch_gap_ms": launch_gap,
        "worker_local_host_ms": worker_local_host,
        "cuda_sync_ms": sync_extra,
        "device_copy_ms": device_copy,
        "local_reduction_ms": reduction_wall,
        "single_gpu_serialization_artifact_ms": serialization,
        "experimental_harness_ms": harness,
        "worker_protocol_ms": 0.0,
        "unexplained_ms": unexplained,
    }
    if inconsistent_boundaries:
        components = {name: 0.0 for name in components}
        components["unexplained_ms"] = outer
    accounted = sum(components.values())
    unexplained_fraction = unexplained / max(outer, 1e-30)
    reconciliation_fraction = abs(accounted - outer) / max(outer, 1e-30)
    row = {
        "evidence_class": "PHYSICAL boundary-instrumented decomposition",
        "layer": receipt["layer"],
        "attention_type": receipt["attention_type"],
        "chunk_rows": receipt["rows"],
        "sample_index": index,
        "total_outer_wall_ms": outer,
        **components,
        "accounted_ms": accounted,
        "unexplained_fraction": unexplained_fraction,
        "reconciliation_fraction": reconciliation_fraction,
        "status": (
            "PASS"
            if unexplained_fraction <= 0.10 and reconciliation_fraction <= 0.10
            else "FAIL"
        ),
    }
    raw = {
        "layer": receipt["layer"],
        "chunk_rows": receipt["rows"],
        "sample_index": index,
        "cuda_event_total_ms": cuda_total,
        "phase_wall_ms": phase_wall,
        "operation_wall_by_phase_ms": dict(operation_wall_by_phase),
        "experiment_operation_wall_by_phase_ms": dict(
            experiment_operation_wall_by_phase
        ),
        "experiment_operation_wall_ms": experiment_operation_wall,
        "experiment_device_copy_wall_ms": experiment_device_copy_wall,
        "parallel_device_copy_ms": parallel_device_copy,
        "parallel_python_harness_ms": parallel_harness,
        "experiment_operations": experiment_records,
        "exclusive_phase_gap_ms": phase_gap,
        "phase_operation_overlap_ms": phase_overlap,
        "measured_phase_wall_ms": measured_phase_wall,
        "outer_phase_boundary_missing_ms": boundary_mismatch,
        "nested_timer_overlap_ms": overlap_mismatch,
        "phase_boundary_overrun_ms": boundary_overrun,
        "timer_consistency_tolerance_ms": timer_tolerance_ms,
        "inconsistent_timer_boundaries": inconsistent_boundaries,
        "raw_instrumentation": instrument,
        "parallel_cuda_groups": parallel_groups,
        "parallel_wall_groups": parallel_wall_groups,
        "parallel_phase_gap_ms": parallel_phase_gap,
        "operation_count": len(all_records),
        "production_operation_count": len(records),
        "experiment_operation_count": len(experiment_records),
        "decomposition_method": (
            "non-overlapping directly timed phase walls; directly timed native operation "
            "intervals; sum-minus-max stripe serialization; measured coordinator, parallel "
            "and receipt phase gaps split into measured device copy and explicit resident-"
            "replay Python harness; outer/phase mismatch remains unexplained"
        ),
    }
    return row, raw


def _residual_artifacts(
    receipts: Sequence[dict[str, Any]], validation_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for receipt in receipts:
        for index in range(len(receipt["wall_samples_ms"])):
            row, trace = _ledger_for_sample(receipt, index)
            ledger.append(row)
            raw.append(trace)
    _write_csv(validation_root / "residual-ledger.csv", ledger)
    maximum_fraction = max(float(row["unexplained_fraction"]) for row in ledger)
    maximum_reconciliation = max(
        float(row["reconciliation_fraction"]) for row in ledger
    )
    classification = {
        "schema_version": "experiment-022-completion-residual-classification-v2",
        "status": (
            "PASS"
            if maximum_fraction <= 0.10 and maximum_reconciliation <= 0.10
            else "FAIL"
        ),
        "target_unexplained_fraction": 0.05,
        "hard_maximum_unexplained_fraction": 0.10,
        "maximum_unexplained_fraction": maximum_fraction,
        "maximum_reconciliation_fraction": maximum_reconciliation,
        "artificial_barrier_residual": False,
        "outside_outer_experiment_only": [
            "fixture_state_reset_ms_outside_outer"
        ],
        "components": [
            {"component": "kernel_compute_ms", "class": "PERSISTS_PER_WORKER"},
            {"component": "launch_gap_ms", "class": "PERSISTS_PER_WORKER"},
            {"component": "worker_local_host_ms", "class": "PERSISTS_PER_WORKER"},
            {"component": "cuda_sync_ms", "class": "PERSISTS_PER_WORKER"},
            {"component": "device_copy_ms", "class": "PERSISTS_PER_WORKER"},
            {"component": "local_reduction_ms", "class": "PERSISTS_PER_COLLECTIVE"},
            {
                "component": "single_gpu_serialization_artifact_ms",
                "class": "SINGLE_GPU_EMULATION_ARTIFACT",
                "distributed_model_charge": False,
            },
            {
                "component": "experimental_harness_ms",
                "class": "EXPERIMENT_ONLY",
                "distributed_model_charge": False,
            },
            {"component": "worker_protocol_ms", "class": "PERSISTS_PER_WORKER"},
            {"component": "network_transport", "class": "NETWORK_MODELLED_SEPARATELY"},
            {"component": "unexplained_ms", "class": "INVALID_IF_OVER_HARD_MAXIMUM"},
        ],
        "raw_boundary_measurements": raw,
    }
    _write_json(validation_root / "residual-classification.json", classification)
    return ledger, classification


def _validation(
    receipts: Sequence[dict[str, Any]], validation_root: Path
) -> dict[str, Any]:
    lookup = {
        (str(row["attention_type"]), int(row["layer"]), int(row["rows"])): row
        for row in receipts
    }
    rows: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    service_receipts: list[dict[str, Any]] = []

    def phase_gap(
        operation_rows: Sequence[dict[str, Any]], phase_wall: dict[str, Any]
    ) -> dict[str, float]:
        operation_wall: dict[str, float] = defaultdict(float)
        for operation_row in operation_rows:
            operation_wall[str(operation_row.get("phase", "UNATTRIBUTED"))] += float(
                operation_row.get("duration_ms", 0.0)
            )
        return {
            str(name): max(0.0, float(duration) - operation_wall.get(str(name), 0.0))
            for name, duration in phase_wall.items()
        }

    def calibrate(receipt: dict[str, Any]) -> dict[str, Any]:
        operation_samples: dict[tuple[str, str], list[float]] = defaultdict(list)
        operation_classification: dict[str, str] = {}
        phase_gap_samples: dict[str, list[float]] = defaultdict(list)
        for operation_rows, instrumentation in zip(
            receipt["operation_records"], receipt["instrumentation"], strict=True
        ):
            for operation_row in operation_rows:
                key = (
                    str(operation_row.get("phase", "UNATTRIBUTED")),
                    str(operation_row["operator"]),
                )
                operation_samples[key].append(float(operation_row["duration_ms"]))
                operation_classification[f"{key[0]}|{key[1]}"] = str(
                    operation_row.get("cost_classification", "PRODUCTION")
                )
            for name, duration in phase_gap(
                operation_rows, dict(instrumentation["phase_wall_ms"])
            ).items():
                phase_gap_samples[name].append(duration)
        return {
            "operation_ms": {
                f"{phase}|{operator}": statistics.median(values)
                for (phase, operator), values in sorted(operation_samples.items())
            },
            "operation_classification": operation_classification,
            "phase_gap_ms": {
                name: statistics.median(values)
                for name, values in sorted(phase_gap_samples.items())
            },
            "task_template": [
                {
                    "phase": str(row.get("phase", "UNATTRIBUTED")),
                    "operator": str(row["operator"]),
                    "cost_classification": str(
                        row.get("cost_classification", "PRODUCTION")
                    ),
                }
                for row in receipt["operation_records"][0]
            ],
            "phase_order": list(receipt["instrumentation"][0]["phase_wall_ms"]),
        }

    def replay(model: dict[str, Any]) -> tuple[float, int]:
        by_phase: dict[str, list[str]] = defaultdict(list)
        for task in model["task_template"]:
            by_phase[str(task["phase"])].append(str(task["operator"]))
        tasks: list[EventTask] = []
        dependency: tuple[str, ...] = ()
        index = 0
        for phase in model["phase_order"]:
            for operator in by_phase.get(str(phase), []):
                identifier = f"validation-{index:03d}.{phase}.{operator}"
                key = f"{phase}|{operator}"
                tasks.append(
                    EventTask(
                        task_id=identifier,
                        resource_id="compute:single-rtx5090",
                        dependency_ids=dependency,
                        duration_ms=float(model["operation_ms"][key]),
                        category=(
                            "experiment_only"
                            if str(task.get("cost_classification"))
                            == "EXPERIMENT_ONLY"
                            else "compute"
                        ),
                        node_id="single-rtx5090",
                        operation=operator,
                    )
                )
                dependency = (identifier,)
                index += 1
            gap = float(model["phase_gap_ms"].get(str(phase), 0.0))
            if gap > 0:
                identifier = f"validation-{index:03d}.{phase}.exclusive-host-gap"
                tasks.append(
                    EventTask(
                        task_id=identifier,
                        resource_id="compute:single-rtx5090",
                        dependency_ids=dependency,
                        duration_ms=gap,
                        category=(
                            "experiment_only"
                            if str(phase) == "receipt_assembly"
                            else "compute"
                        ),
                        node_id="single-rtx5090",
                        operation=f"{phase}_exclusive_host_gap",
                    )
                )
                dependency = (identifier,)
                index += 1
        run = DeterministicWorkerEventEngine().run(tasks)
        return run.makespan_ms, len(tasks)

    for attention_type in ("KDA", "Gated_MLA"):
        calibration_layer = CALIBRATION[attention_type]
        heldout_layer = HELD_OUT[attention_type]
        for chunk in CHUNKS:
            calibration = lookup[(attention_type, calibration_layer, chunk)]
            event_model = calibrate(calibration)
            predicted_calibration, task_count = replay(event_model)
            measured_calibration = float(calibration["wall"]["p50_ms"])
            calibration_error = (
                abs(predicted_calibration - measured_calibration)
                / max(measured_calibration, 1e-30)
                * 100.0
            )
            service_receipts.append(
                {
                    "attention_type": attention_type,
                    "calibration_layer": calibration_layer,
                    "chunk_rows": chunk,
                    "partition_degree": int(calibration["degree"]),
                    "event_model": event_model,
                    "normalization_or_global_correction": False,
                    "experiment_only_tasks_reconstruct_outer_wall_but_are_not_planner_service": True,
                    "production_excluded_operation_classes": ["EXPERIMENT_ONLY"],
                }
            )
            calibration_row = {
                "split": "calibration",
                "attention_type": attention_type,
                "layer": calibration_layer,
                "chunk_rows": chunk,
                "predicted_ms": predicted_calibration,
                "measured_ms": measured_calibration,
                "absolute_percent_error": calibration_error,
                "service_key": f"{attention_type}:chunk-{chunk}:degree-{calibration['degree']}",
                "single_resource_event_task_count": task_count,
                "normalization_or_correction_factor": False,
            }
            rows.append(calibration_row)
            heldout_receipt = lookup[(attention_type, heldout_layer, chunk)]
            # The algorithmic ordered task template is fixed by layer class and
            # degree.  Assert the held-out implementation exposes exactly that
            # template; no held-out timing value is used to create a service.
            heldout_template = [
                {
                    "phase": str(row.get("phase", "UNATTRIBUTED")),
                    "operator": str(row["operator"]),
                    "cost_classification": str(
                        row.get("cost_classification", "PRODUCTION")
                    ),
                }
                for row in heldout_receipt["operation_records"][0]
            ]
            if heldout_template != event_model["task_template"]:
                raise RuntimeError(
                    f"held-out ordered DAG differs for {attention_type} chunk {chunk}"
                )
            predicted_heldout, heldout_task_count = replay(event_model)
            observed = float(heldout_receipt["wall"]["p50_ms"])
            error = abs(predicted_heldout - observed) / max(observed, 1e-30) * 100.0
            heldout_row = {
                "split": "heldout",
                "attention_type": attention_type,
                "layer": heldout_layer,
                "chunk_rows": chunk,
                "predicted_ms": predicted_heldout,
                "measured_ms": observed,
                "absolute_percent_error": error,
                "service_key": f"{attention_type}:chunk-{chunk}:degree-{calibration['degree']}",
                "single_resource_event_task_count": heldout_task_count,
                "normalization_or_correction_factor": False,
            }
            rows.append(heldout_row)
            heldout.append(heldout_row)
    # The preregistered gate is applied to genuinely held-out physical layers;
    # calibration reconstruction is retained as a diagnostic and cannot dilute
    # held-out error.
    errors = [float(row["absolute_percent_error"]) for row in heldout]
    median = float(np.percentile(errors, 50))
    p90 = float(np.percentile(errors, 90))
    maximum = max(errors)
    status = "PASS" if median <= 5 and p90 <= 10 and maximum <= 15 else "FAIL"
    _write_csv(validation_root / "ordered-dag-validation.csv", rows)
    _write_csv(validation_root / "heldout-validation.csv", heldout)
    _write_json(validation_root / "repaired-event-services.json", service_receipts)
    result = {
        "schema_version": "experiment-022-completion-model-validation-v2",
        "status": status,
        "evidence_class": "PHYSICALLY GROUNDED MODEL validated against PHYSICAL replay",
        "calibration_layers": CALIBRATION,
        "heldout_layers": HELD_OUT,
        "chunks": list(CHUNKS),
        "degree": 8,
        "absolute_percent_error": {
            "median": median,
            "p90": p90,
            "maximum": maximum,
        },
        "thresholds": {"median": 5.0, "p90": 10.0, "maximum": 15.0},
        "normalization": False,
        "global_correction_factor": False,
        "service_features": [
            "attention_type",
            "chunk_rows",
            "partition_degree",
            "native_operation",
            "ordered_phase",
        ],
        "validation_method": (
            "calibration-layer native-operation and exclusive-phase services replayed "
            "as the actual ordered DAG on one concrete compute resource; explicitly "
            "classified experiment-only tasks are included only to reconcile measured "
            "outer wall and are excluded from the planner service catalog"
        ),
        "gate_population": "held-out physical layers only",
        "row_count": len(rows),
        "heldout_row_count": len(heldout),
    }
    _write_json(validation_root / "model-validation.json", result)
    return result


def run(
    *,
    checkpoint: Path,
    cuda_library: Path,
    shard_library: Path,
    grouped_library: Path,
    oracle_root: Path,
    completion_root: Path,
    degree: int = 8,
    warmup: int = 1,
    iterations: int = 3,
) -> dict[str, Any]:
    physical_root = completion_root / "physical"
    validation_root = completion_root / "validation"
    sampler = _GpuSampler()
    sampler.start()
    receipts: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    try:
        for attention_type, layers in (
            ("KDA", (CALIBRATION["KDA"], HELD_OUT["KDA"])),
            ("Gated_MLA", (CALIBRATION["Gated_MLA"], HELD_OUT["Gated_MLA"])),
        ):
            for layer in layers:
                for rows in CHUNKS:
                    reference_output, reference = _whole_layer_reference(
                        checkpoint,
                        cuda_library,
                        oracle_root,
                        layer=layer,
                        rows=rows,
                    )
                    reference["expected_attention_type"] = attention_type
                    references.append(reference)
                    receipt = replay_resident_layer(
                        checkpoint,
                        cuda_library,
                        shard_library,
                        grouped_library,
                        oracle_root,
                        layer=layer,
                        degree=degree,
                        rows=rows,
                        warmup=warmup,
                        iterations=iterations,
                        exact_reference_output=reference_output,
                        exact_reference_routes=reference["routes"],
                        exact_reference_state_fingerprint=reference["state_output"][
                            "fingerprint"
                        ],
                    )
                    receipt["reference"] = reference
                    receipts.append(receipt)
                    print(
                        f"[E022 completion] {attention_type} layer={layer} rows={rows} "
                        f"status={receipt['status']} p50={receipt['wall']['p50_ms']:.3f} ms "
                        f"rel_l2={receipt['correctness']['relative_l2_error']:.3e}",
                        flush=True,
                    )
    finally:
        sampler.stop()
        _write_csv(physical_root / "gpu-samples.csv", sampler.rows)

    kda = [row for row in receipts if row["attention_type"] == "KDA"]
    mla = [row for row in receipts if row["attention_type"] == "Gated_MLA"]
    _write_json(
        physical_root / "resident-kda-traces.json",
        {
            "schema_version": "experiment-022-completion-resident-kda-traces-v1",
            "receipts": kda,
        },
    )
    _write_json(
        physical_root / "resident-mla-traces.json",
        {
            "schema_version": "experiment-022-completion-resident-mla-traces-v1",
            "receipts": mla,
        },
    )
    services = _service_rows(receipts)
    for chunk in CHUNKS:
        _write_csv(
            physical_root / f"chunk-{chunk}-services.csv",
            [row for row in services if int(row["chunk_rows"]) == chunk],
        )
    ledger, classification = _residual_artifacts(receipts, validation_root)
    validation = _validation(receipts, validation_root)
    reconciliation = {
        "schema_version": "experiment-022-completion-accounting-reconciliation-v2",
        "status": (
            "PASS"
            if all(row["status"] == "PASS" for row in ledger)
            and max(float(row["reconciliation_fraction"]) for row in ledger)
            <= 0.10
            else "FAIL"
        ),
        "one_owner_per_cost": True,
        "network_is_not_charged_as_local_compute": True,
        "artificial_reduction_barriers": False,
        "maximum_reconciliation_fraction": max(
            float(row["reconciliation_fraction"]) for row in ledger
        ),
        "cost_owners": {
            "worker_compute": ["kernel_compute_ms"],
            "worker_software_overhead": [
                "launch_gap_ms",
                "worker_local_host_ms",
                "cuda_sync_ms",
                "device_copy_ms",
                "worker_protocol_ms",
            ],
            "collective_compute": ["local_reduction_ms"],
            "network_transport": ["event_model_network_edges_only"],
            "scheduler_control_plane": ["event_model_scheduler_edges_only"],
            "state_wait": ["event_model_state_dependencies_only"],
            "excluded_single_gpu_artifact": [
                "single_gpu_serialization_artifact_ms"
            ],
            "excluded_experiment_only": ["experimental_harness_ms"],
        },
    }
    _write_json(validation_root / "accounting-reconciliation.json", reconciliation)
    receipt_keys = {
        (str(row["attention_type"]), int(row["layer"]), int(row["rows"]))
        for row in receipts
    }
    expected_receipt_keys = {
        (attention_type, layer, chunk)
        for attention_type, layers in (
            ("KDA", (CALIBRATION["KDA"], HELD_OUT["KDA"])),
            (
                "Gated_MLA",
                (CALIBRATION["Gated_MLA"], HELD_OUT["Gated_MLA"]),
            ),
        )
        for layer in layers
        for chunk in CHUNKS
    }
    receipt_matrix_complete = (
        len(receipts) == len(expected_receipt_keys)
        and receipt_keys == expected_receipt_keys
    )
    all_correct = receipt_matrix_complete and all(
        row["status"] == "PASS"
        and row.get("arbitrary_route_full_expert_bank_resident") is True
        and int(row.get("resident_expert_bank_count", -1)) == degree
        and int(row.get("expected_resident_expert_bank_count", -1)) == degree
        and row.get("resident_experts_per_bank") == [ROUTED_EXPERTS] * degree
        for row in receipts
    )
    status = (
        "PASS"
        if all_correct
        and classification["status"] == "PASS"
        and validation["status"] == "PASS"
        and reconciliation["status"] == "PASS"
        else "FAIL"
    )
    gpu_names = sorted(
        {
            str(row["gpu_name"])
            for row in sampler.rows
            if row.get("gpu_name")
        }
    )
    if not gpu_names:
        status = "FAIL"
    input_receipt = {
        "schema_version": "experiment-022-completion-physical-inputs-v1",
        "status": "PASS" if gpu_names else "FAIL",
        "checkpoint": {
            "root": str(checkpoint.resolve()),
            "config": _file_receipt(checkpoint / "config.json"),
            "index": _file_receipt(checkpoint / "model.safetensors.index.json"),
        },
        "native_libraries": {
            "canonical_cuda": _file_receipt(cuda_library),
            "shard_and_quantizer": _file_receipt(shard_library),
            "grouped_expert": _file_receipt(grouped_library),
        },
        "immutable_oracle": {
            "hidden_trace": _file_receipt(oracle_root / "hidden-trace.f32"),
            "routes": _file_receipt(oracle_root / "routes.txt"),
            "logits": _file_receipt(oracle_root / "prefill-logits.f32"),
        },
        "gpu_names_observed": gpu_names,
    }
    _write_json(
        completion_root / "implementation" / "physical-inputs.json",
        input_receipt,
    )
    summary = {
        "schema_version": "experiment-022-completion-physical-gates-v2",
        "status": status,
        "receipt_count": len(receipts),
        "receipt_matrix_complete": receipt_matrix_complete,
        "receipt_keys": [
            {
                "attention_type": attention_type,
                "layer": layer,
                "chunk_rows": chunk,
            }
            for attention_type, layer, chunk in sorted(receipt_keys)
        ],
        "layers": sorted({int(row["layer"]) for row in receipts}),
        "chunks": sorted({int(row["rows"]) for row in receipts}),
        "sub_layer_chunk_2_physically_validated": all(
            row["status"] == "PASS" for row in receipts if row["rows"] == 2
        ),
        "sub_layer_chunk_4_physically_validated": all(
            row["status"] == "PASS" for row in receipts if row["rows"] == 4
        ),
        "arbitrary_route_full_expert_banks_resident": all(
            row.get("arbitrary_route_full_expert_bank_resident") is True
            for row in receipts
        ),
        "residual_classification_status": classification["status"],
        "model_validation_status": validation["status"],
        "accounting_reconciliation_status": reconciliation["status"],
        "physical_reference_count": len(references),
        "physical_input_receipt": "implementation/physical-inputs.json",
    }
    _write_json(validation_root / "physical-gates-summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--shard-library", type=Path, required=True)
    parser.add_argument("--grouped-library", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--completion-root", type=Path, required=True)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    result = run(
        checkpoint=args.checkpoint.resolve(),
        cuda_library=args.cuda_library.resolve(),
        shard_library=args.shard_library.resolve(),
        grouped_library=args.grouped_library.resolve(),
        oracle_root=args.oracle_root.resolve(),
        completion_root=args.completion_root.resolve(),
        degree=args.degree,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
