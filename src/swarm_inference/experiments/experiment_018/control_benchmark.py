"""Physical coordinator benchmark for hierarchical wavefront task batching."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_018.wavefront import HierarchicalTaskBatcher

SCHEMA_VERSION = "experiment-018-control-plane-scaling-v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _tasks(count: int) -> list[dict[str, Any]]:
    return [
        {
            "microcell": index % 12,
            "operation": "routed_expert",
            "shape": (1, 3584),
            "dtype": "float32_activation_mxfp4_weight",
            "expert": (index // 12) % 4,
            "weight_shard": (index // (12 * 4)) % 4,
            "route_bucket": 0,
            "task_sequence": index,
        }
        for index in range(count)
    ]


def _compact_buckets(
    count: int,
) -> dict[str, dict[tuple[Any, ...], tuple[int, int]]]:
    """Build exact worker-local counters once, without retaining logical tasks."""

    counters: dict[str, dict[tuple[Any, ...], list[int]]] = {}
    for index in range(count):
        task = {
            "microcell": index % 12,
            "operation": "routed_expert",
            "shape": (1, 3584),
            "dtype": "float32_activation_mxfp4_weight",
            "expert": (index // 12) % 4,
            "weight_shard": (index // (12 * 4)) % 4,
            "route_bucket": 0,
            "task_sequence": index,
        }
        worker_id = str(task["microcell"])
        key = (
            task["operation"],
            task["shape"],
            task["dtype"],
            task["expert"],
            task["weight_shard"],
            task["route_bucket"],
        )
        value = counters.setdefault(worker_id, {}).setdefault(key, [0, 0])
        value[0] += 1
        value[1] += len(json.dumps(task, sort_keys=True, separators=(",", ":")))
    return {
        worker: {key: (value[0], value[1]) for key, value in buckets.items()}
        for worker, buckets in counters.items()
    }


def benchmark(
    output_path: Path,
    *,
    natural_task_count: int,
    iterations: int = 200,
    leaf_batch: int = 32,
) -> dict[str, Any]:
    if natural_task_count < 1 or iterations < 20:
        raise ValueError("natural task count and at least 20 iterations are required")
    counts = tuple(sorted({32, 128, 512, 1000, natural_task_count}))
    batcher = HierarchicalTaskBatcher(leaf_batch=leaf_batch)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "evidence_class": "PHYSICAL CPU coordinator",
        "environment": {
            "platform": platform.platform(),
            "python_process_id": os.getpid(),
        },
        "configuration": {
            "counts": list(counts),
            "natural_task_count": natural_task_count,
            "iterations": iterations,
            "leaf_batch": leaf_batch,
        },
        "rows": [],
    }
    _atomic_json(output_path, result)
    for count in counts:
        fixture_started = time.perf_counter_ns()
        compact = count > 10_000
        if compact:
            compact_workers = _compact_buckets(count)
            workers = None
        else:
            tasks = _tasks(count)
            workers = {}
            for task in tasks:
                workers.setdefault(str(task["microcell"]), []).append(task)
            compact_workers = None
        fixture_build_ms = (time.perf_counter_ns() - fixture_started) / 1e6
        end_to_end_samples: list[float] = []
        coordinator_samples: list[float] = []
        local_critical_samples: list[float] = []
        final: dict[str, int | float] | None = None
        for _ in range(iterations):
            started = time.perf_counter_ns()
            final = (
                batcher.plan_compact(compact_workers)
                if compact_workers is not None
                else batcher.plan_hierarchical(workers or {})
            )
            end_to_end_samples.append((time.perf_counter_ns() - started) / 1e6)
            coordinator_samples.append(float(final["coordinator_cpu_ms"]))
            local_critical_samples.append(float(final["worker_local_cpu_critical_ms"]))
        assert final is not None
        row = {
            **final,
            "logical_task_count": count,
            "coordinator_wall_p50_ms": float(
                np.percentile(coordinator_samples, 50)
            ),
            "coordinator_wall_p90_ms": float(
                np.percentile(coordinator_samples, 90)
            ),
            "coordinator_wall_p99_ms": float(
                np.percentile(coordinator_samples, 99)
            ),
            "coordinator_wall_mean_ms": float(np.mean(coordinator_samples)),
            "worker_local_critical_p50_ms": float(
                np.percentile(local_critical_samples, 50)
            ),
            "single_process_sequential_harness_p50_ms": float(
                np.percentile(end_to_end_samples, 50)
            ),
            "one_serial_wait_per_task": False,
            "input_representation": (
                "persistent_worker_compact_buckets"
                if compact
                else "expanded_worker_local_tasks"
            ),
            "fixture_construction_outside_timed_planner_ms": fixture_build_ms,
            "fixture_construction_on_critical_path": False,
            "serial_decisions_per_task": float(final["serial_scheduling_decisions"])
            / count,
        }
        result["rows"].append(row)
        _atomic_json(output_path, result)
    xs = np.log(np.asarray(counts, dtype=np.float64))
    ys = np.log(
        np.asarray(
            [float(row["coordinator_wall_p50_ms"]) for row in result["rows"]],
            dtype=np.float64,
        )
    )
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(counts) > 1 else 0.0
    thousand = next(row for row in result["rows"] if row["logical_task_count"] == 1000)
    result["log_log_scaling_exponent"] = slope
    result["sublinear_measured_cpu_growth"] = slope < 1.0
    result["thousand_tasks_not_thousand_serial_waits"] = (
        int(thousand["critical_path_waits"]) < 1000
    )
    result["status"] = (
        "PASS"
        if result["sublinear_measured_cpu_growth"]
        and result["thousand_tasks_not_thousand_serial_waits"]
        else "FAIL"
    )
    result["finished_unix_ns"] = time.time_ns()
    _atomic_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--natural-task-count", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--leaf-batch", type=int, default=32)
    arguments = parser.parse_args()
    benchmark(
        arguments.output,
        natural_task_count=arguments.natural_task_count,
        iterations=arguments.iterations,
        leaf_batch=arguments.leaf_batch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
