"""Low-frequency host/GPU telemetry for measured Experiment 011 runs."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import psutil


def _gpu_snapshot() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        values = [value.strip() for value in completed.stdout.strip().split(",")]
        if len(values) < 6:
            raise ValueError("nvidia-smi returned incomplete telemetry")
        return {
            "gpu_temperature_c": float(values[0]),
            "gpu_sm_clock_mhz": float(values[1]),
            "gpu_power_watts": float(values[2]),
            "gpu_utilisation_percent": float(values[3]),
            "gpu_memory_used_bytes": int(float(values[4]) * 1024 * 1024),
            "gpu_memory_total_bytes": int(float(values[5]) * 1024 * 1024),
        }
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"gpu_query_error": f"{type(exc).__name__}: {exc}"}


def resource_snapshot(*, include_processes: bool = False) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    snapshot: dict[str, Any] = {
        "timestamp_ns": time.time_ns(),
        "monotonic_ns": time.perf_counter_ns(),
        "cpu_utilisation_percent": psutil.cpu_percent(interval=None),
        "host_memory_used_bytes": memory.used,
        "host_memory_available_bytes": memory.available,
        "host_memory_total_bytes": memory.total,
        "process_count": len(psutil.pids()),
        **_gpu_snapshot(),
    }
    if include_processes:
        processes = []
        for process in psutil.process_iter(["pid", "name", "username", "memory_info"]):
            try:
                info = process.info
                memory_info = info.get("memory_info")
                processes.append(
                    {
                        "pid": int(info["pid"]),
                        "name": str(info.get("name") or ""),
                        "username": str(info.get("username") or ""),
                        "rss_bytes": int(memory_info.rss) if memory_info is not None else 0,
                    }
                )
            except (psutil.Error, OSError):
                continue
        snapshot["background_processes"] = sorted(
            processes, key=lambda row: (-int(row["rss_bytes"]), int(row["pid"]))
        )
    return snapshot


class ResourceMonitor:
    def __init__(self, output_path: Path, *, interval_seconds: float = 1.0) -> None:
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.samples.append(resource_snapshot())

    def __enter__(self) -> ResourceMonitor:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.samples.append({"phase": "start", **resource_snapshot(include_processes=True)})
        self._thread = threading.Thread(
            target=self._sample_loop, name="experiment-011-resource-monitor", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_seconds * 2, 2.0))
        self.samples.append({"phase": "end", **resource_snapshot(include_processes=True)})
        with self.output_path.open("w", encoding="utf-8") as handle:
            for sample in self.samples:
                handle.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")


def summarise_resources(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in samples if name in row]

    fields = (
        "gpu_temperature_c",
        "gpu_sm_clock_mhz",
        "gpu_power_watts",
        "gpu_utilisation_percent",
        "gpu_memory_used_bytes",
        "host_memory_used_bytes",
        "cpu_utilisation_percent",
        "process_count",
    )
    summary: dict[str, Any] = {"sample_count": len(samples)}
    for field in fields:
        observed = values(field)
        summary[f"{field}_mean"] = sum(observed) / len(observed) if observed else None
        summary[f"{field}_maximum"] = max(observed) if observed else None
    return summary
