"""Deterministic synthetic CPU-kernel calibration in an isolated process."""

from __future__ import annotations

import multiprocessing
import os
import statistics
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

import psutil

from swarm_inference.config.models import SyntheticComputeConfig
from swarm_inference.model.synthetic import deterministic_cpu_kernel


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    target_stage_ms: float
    median_stage_ms: float
    minimum_stage_ms: float
    maximum_stage_ms: float
    p95_stage_ms: float
    work_units: int
    activation_bytes: int
    warmup_iterations: int
    measurement_iterations: int
    cpu_id: int | None
    actual_affinity: list[int]
    process_id: int
    single_thread_environment: dict[str, str]
    acceptable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_stage_ms": self.target_stage_ms,
            "median_stage_ms": self.median_stage_ms,
            "minimum_stage_ms": self.minimum_stage_ms,
            "maximum_stage_ms": self.maximum_stage_ms,
            "p95_stage_ms": self.p95_stage_ms,
            "work_units": self.work_units,
            "activation_bytes": self.activation_bytes,
            "warmup_iterations": self.warmup_iterations,
            "measurement_iterations": self.measurement_iterations,
            "cpu_id": self.cpu_id,
            "actual_affinity": self.actual_affinity,
            "process_id": self.process_id,
            "single_thread_environment": self.single_thread_environment,
            "acceptable": self.acceptable,
        }


_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _measure(work_units: int, activation_bytes: int, iterations: int) -> list[float]:
    payload = bytes((index * 17 + 31) % 256 for index in range(activation_bytes))
    seed = b"experiment-001-calibration"
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        deterministic_cpu_kernel(
            payload,
            seed_material=seed,
            work_units=work_units,
            buffer_bytes=activation_bytes,
        )
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return samples


def _calibration_child(
    connection: Connection,
    config_payload: dict[str, Any],
    cpu_id: int | None,
) -> None:
    try:
        for name, value in _THREAD_ENVIRONMENT.items():
            os.environ[name] = value
        process = psutil.Process()
        if cpu_id is not None:
            try:
                process.cpu_affinity([cpu_id])
            except (AttributeError, OSError, ValueError, psutil.Error):
                cpu_id = None
        config = SyntheticComputeConfig.model_validate(config_payload)
        if config.work_units is not None:
            work_units = config.work_units
        else:
            probe_units = 32
            probe = _measure(probe_units, config.activation_bytes, 9)
            probe_median = max(statistics.median(probe), 0.001)
            work_units = max(1, round(probe_units * config.target_stage_ms / probe_median))
            for _ in range(4):
                tuning = _measure(work_units, config.activation_bytes, 21)
                tuning_median = max(statistics.median(tuning), 0.001)
                if config.acceptable_min_ms <= tuning_median <= config.acceptable_max_ms:
                    break
                work_units = max(
                    1,
                    round(work_units * config.target_stage_ms / tuning_median),
                )
        _measure(
            work_units,
            config.activation_bytes,
            config.calibration_warmup_iterations,
        )
        samples = _measure(
            work_units,
            config.activation_bytes,
            config.calibration_measurement_iterations,
        )
        ordered = sorted(samples)
        p95_index = min(len(ordered) - 1, round((len(ordered) - 1) * 0.95))
        median = float(statistics.median(samples))
        result = CalibrationResult(
            target_stage_ms=config.target_stage_ms,
            median_stage_ms=median,
            minimum_stage_ms=float(min(samples)),
            maximum_stage_ms=float(max(samples)),
            p95_stage_ms=float(ordered[p95_index]),
            work_units=work_units,
            activation_bytes=config.activation_bytes,
            warmup_iterations=config.calibration_warmup_iterations,
            measurement_iterations=config.calibration_measurement_iterations,
            cpu_id=cpu_id,
            actual_affinity=(
                list(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else []
            ),
            process_id=os.getpid(),
            single_thread_environment={
                name: os.environ.get(name, "") for name in _THREAD_ENVIRONMENT
            },
            acceptable=config.acceptable_min_ms <= median <= config.acceptable_max_ms,
        )
        connection.send({"ok": True, "result": result.to_dict()})
    except BaseException as exc:
        connection.send(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    finally:
        connection.close()


def calibrate_synthetic_compute(
    config: SyntheticComputeConfig,
    *,
    cpu_id: int | None,
    timeout_s: float = 120.0,
) -> CalibrationResult:
    """Calibrate once in a spawned process and return immutable evidence."""

    if config.mode != "calibrated_cpu":
        raise ValueError("calibration requires synthetic_compute.mode=calibrated_cpu")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_calibration_child,
        args=(child, config.model_dump(mode="python"), cpu_id),
        name="synthetic-calibration",
    )
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_s):
            process.terminate()
            process.join(5)
            raise TimeoutError(f"synthetic calibration exceeded {timeout_s:.1f} seconds")
        payload = parent.recv()
    finally:
        parent.close()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
    if not payload["ok"]:
        raise RuntimeError(
            f"synthetic calibration failed: {payload['error_type']}: {payload['error']}"
        )
    return CalibrationResult(**payload["result"])


def available_cpu_ids() -> list[int]:
    """Return CPUs this process is allowed to use, preserving OS numbering."""

    process = psutil.Process()
    try:
        affinity = list(process.cpu_affinity())
    except (AttributeError, OSError, ValueError, psutil.Error):
        affinity = list(range(psutil.cpu_count(logical=True) or 1))
    return sorted(set(int(cpu_id) for cpu_id in affinity))
