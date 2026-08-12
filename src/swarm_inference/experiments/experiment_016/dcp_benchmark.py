"""Exact CPU DCP component benchmark at Kimi K3 attention geometry."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from swarm_inference.execution.dcp import (
    combine_attention_partials,
    dcp_partial_payload_bytes,
    full_attention,
    shard_attention,
)
from swarm_inference.experiments.experiment_015.network import NetworkProfile

SCHEMA_VERSION = "experiment-016-dcp-component-v1"
CONTEXTS = (2048, 8192, 32768)
DEGREES = (1, 2, 4, 8)
SEED = 1601601


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _p50(values: list[float]) -> float:
    return float(np.percentile(values, 50))


def benchmark(output_path: Path, csv_path: Path) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "scope": (
            "real local CPU execution of the exact sufficient-statistic reducer at "
            "Kimi K3 head/value geometry; not a complete Kimi MLA or CUDA DCP path"
        ),
        "configuration": {
            "contexts": list(CONTEXTS),
            "degrees": list(DEGREES),
            "query_rows": 1,
            "heads": 64,
            "value_dimension": 128,
            "input_dtype": "float32",
            "reference_accumulation_dtype": "float64",
            "seed": SEED,
            "combine_iterations": 50,
        },
        "network_shaping": {
            "domain": "single-machine analytical shaping only",
            "rtt_ms": 0.25,
            "jitter_ms": 0.0,
            "bandwidth_gbps": 25.0,
            "packet_loss_percent": 0.0,
            "measured_loopback_base_ms_from_experiment_015": 0.6102,
        },
        "rows": [],
    }
    _atomic_json(output_path, receipt)
    process = psutil.Process()
    rows: list[dict[str, Any]] = []
    peak_rss = process.memory_info().rss
    try:
        for context in CONTEXTS:
            generator = np.random.default_rng(SEED + context)
            scores = generator.standard_normal((64, context), dtype=np.float32)
            values = generator.standard_normal((64, context, 128), dtype=np.float32)
            peak_rss = max(peak_rss, process.memory_info().rss)
            reference_started = time.perf_counter_ns()
            reference = full_attention(scores, values)
            reference_wall_ms = (time.perf_counter_ns() - reference_started) / 1e6
            peak_rss = max(peak_rss, process.memory_info().rss)
            for degree in DEGREES:
                score_shards = np.array_split(scores, degree, axis=-1)
                value_shards = np.array_split(values, degree, axis=-2)
                partials = []
                partial_wall_ms: list[float] = []
                for shard_index, (score_shard, value_shard) in enumerate(
                    zip(score_shards, value_shards, strict=True)
                ):
                    started = time.perf_counter_ns()
                    partials.append(
                        shard_attention(
                            score_shard,
                            value_shard,
                            shard_index=shard_index,
                        )
                    )
                    partial_wall_ms.append((time.perf_counter_ns() - started) / 1e6)
                    peak_rss = max(peak_rss, process.memory_info().rss)
                combine_wall_ms: list[float] = []
                actual: np.ndarray | None = None
                for _ in range(50):
                    started = time.perf_counter_ns()
                    actual = combine_attention_partials(partials)
                    combine_wall_ms.append((time.perf_counter_ns() - started) / 1e6)
                if actual is None:
                    raise RuntimeError("DCP combination emitted no output")
                difference = actual - reference
                relative_l2 = float(np.linalg.norm(difference) / np.linalg.norm(reference))
                maximum_absolute = float(np.max(np.abs(difference)))
                payload = dcp_partial_payload_bytes(query_rows=1)
                network = NetworkProfile(
                    "h016-internal-shaped",
                    rtt_ms=0.25,
                    bandwidth_gbps=25.0,
                )
                shaped_transport_ms = network.service_ms(payload) if degree > 1 else 0.0
                combine_p50 = _p50(combine_wall_ms)
                sequential_ms = sum(partial_wall_ms) + combine_p50
                ideal_parallel_ms = max(partial_wall_ms) + combine_p50
                shaped_parallel_ms = ideal_parallel_ms + shaped_transport_ms
                row = {
                    "context_tokens": context,
                    "degree": degree,
                    "reference_wall_ms": reference_wall_ms,
                    "partial_compute_sum_wall_ms": sum(partial_wall_ms),
                    "partial_compute_max_wall_ms": max(partial_wall_ms),
                    "combine_wall_p50_ms": combine_p50,
                    "local_sequential_end_to_end_wall_ms": sequential_ms,
                    "ideal_parallel_shards_plus_combine_ms": ideal_parallel_ms,
                    "shaped_transport_ms": shaped_transport_ms,
                    "shaped_parallel_component_ms": shaped_parallel_ms,
                    "ideal_parallel_component_speedup_vs_degree1_reference": (
                        reference_wall_ms / shaped_parallel_ms
                    ),
                    "partial_payload_bytes_per_worker": payload,
                    "synchronization_count_assumed": 2 if degree > 1 else 0,
                    "maximum_absolute_error": maximum_absolute,
                    "relative_l2_error": relative_l2,
                    "repeat_bit_exact": bool(
                        np.array_equal(actual, combine_attention_partials(partials))
                    ),
                    "finite": bool(np.isfinite(actual).all()),
                    "evidence_class": (
                        "MEASURED_LOCAL_CPU_COMPONENT"
                        if degree == 1
                        else "MEASURED_CPU_PLUS_SHAPED_PARALLELISM"
                    ),
                }
                rows.append(row)
                receipt["rows"] = rows
                _atomic_json(output_path, receipt)
                print(
                    f"[h016-dcp] context={context} degree={degree} "
                    f"relative_l2={relative_l2:.3e} shaped_ms={shaped_parallel_ms:.3f}",
                    flush=True,
                )
            del reference, scores, values
        receipt["peak_process_rss_bytes"] = peak_rss
        receipt["maximum_relative_l2_error"] = max(float(row["relative_l2_error"]) for row in rows)
        receipt["correctness_pass"] = all(
            float(row["relative_l2_error"]) <= 2e-12
            and bool(row["repeat_bit_exact"])
            and bool(row["finite"])
            for row in rows
        )
        receipt["complete_kimi_dcp_gate"] = "NOT_ESTABLISHED"
        receipt["cuda_dcp_kernel_available"] = False
        receipt["status"] = "PASS" if receipt["correctness_pass"] else "FAIL"
        _write_csv(csv_path, rows)
        _atomic_json(output_path, receipt)
        return receipt
    except BaseException as exc:
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(output_path, receipt)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    arguments = parser.parse_args()
    benchmark(arguments.output, arguments.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
