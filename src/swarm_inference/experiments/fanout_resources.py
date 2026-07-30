"""System-wide and process-local resource sampling for Experiment 003."""

from __future__ import annotations

import csv
import ctypes
import io
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import psutil


def _mib_to_bytes(value: float | None) -> int | None:
    return int(value * 1024 * 1024) if value is not None else None


def _optional_float(value: str) -> float | None:
    normalised = value.strip()
    if not normalised or normalised.lower() in {"n/a", "na", "[n/a]", "not supported"}:
        return None
    try:
        return float(normalised)
    except ValueError:
        return None


def parse_nvml_process_memory(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in csv.reader(io.StringIO(text)):
        if not raw or not raw[0].strip():
            continue
        try:
            pid = int(raw[0].strip())
        except ValueError:
            continue
        memory_mib = _optional_float(raw[1]) if len(raw) > 1 else None
        rows.append(
            {
                "process_id": pid,
                "nvml_gpu_memory_bytes": (
                    int(memory_mib * 1024 * 1024) if memory_mib is not None else None
                ),
                "gpu_process_memory_source": "nvidia-smi-compute-apps",
            }
        )
    return rows


def aggregate_nvml_process_memory(rows: Iterable[dict[str, Any]]) -> int | None:
    values = [row.get("nvml_gpu_memory_bytes") for row in rows]
    if not values or any(value is None for value in values):
        return None
    return sum(int(value) for value in values if value is not None)


def parse_nvidia_gpu_query(text: str) -> dict[str, float | None]:
    row = next(csv.reader(io.StringIO(text)), [])
    names = (
        "gpu_memory_used_mib",
        "gpu_memory_free_mib",
        "gpu_memory_total_mib",
        "gpu_utilisation_percent",
        "memory_controller_utilisation_percent",
        "power_draw_watts",
        "temperature_c",
        "graphics_clock_mhz",
        "memory_clock_mhz",
        "pcie_tx_kib_s",
        "pcie_rx_kib_s",
    )
    return {
        name: _optional_float(row[index]) if index < len(row) else None
        for index, name in enumerate(names)
    }


def parse_windows_gpu_process_memory(
    text: str,
    *,
    process_ids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """Parse Windows PDH dedicated-usage counters, aggregating adapter instances by PID."""

    rows = [row for row in csv.reader(io.StringIO(text)) if row]
    header_index = next(
        (index for index, row in enumerate(rows) if row[0].startswith("(PDH-CSV")),
        None,
    )
    if header_index is None or header_index + 1 >= len(rows):
        return []
    headers = rows[header_index]
    values = rows[header_index + 1]
    targets = set(process_ids) if process_ids is not None else None
    by_pid: dict[int, int] = {}
    for header, raw_value in zip(headers[1:], values[1:], strict=False):
        match = re.search(r"GPU Process Memory\(pid_(\d+)_", header, flags=re.IGNORECASE)
        if match is None:
            continue
        process_id = int(match.group(1))
        if targets is not None and process_id not in targets:
            continue
        try:
            value = max(0, int(float(raw_value)))
        except ValueError:
            continue
        by_pid[process_id] = by_pid.get(process_id, 0) + value
    return [
        {
            "process_id": process_id,
            "nvml_gpu_memory_bytes": None,
            "gpu_process_memory_bytes": memory_bytes,
            "gpu_process_memory_source": "windows-pdh-dedicated-usage",
        }
        for process_id, memory_bytes in sorted(by_pid.items())
        if memory_bytes > 0
    ]


def query_windows_gpu_process_memory(
    process_ids: Iterable[int],
) -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    targets = list(dict.fromkeys(process_ids))
    if not targets:
        return []

    class CounterValueUnion(ctypes.Union):
        _fields_: ClassVar[list[tuple[str, Any]]] = [
            ("long_value", ctypes.c_long),
            ("double_value", ctypes.c_double),
            ("large_value", ctypes.c_longlong),
            ("ansi_string_value", ctypes.c_char_p),
            ("wide_string_value", ctypes.c_wchar_p),
        ]

    class CounterValue(ctypes.Structure):
        _anonymous_: ClassVar[tuple[str, ...]] = ("value",)
        _fields_: ClassVar[list[tuple[str, Any]]] = [
            ("status", ctypes.c_ulong),
            ("value", CounterValueUnion),
        ]

    class CounterValueItem(ctypes.Structure):
        _fields_: ClassVar[list[tuple[str, Any]]] = [
            ("name", ctypes.c_wchar_p),
            ("value", CounterValue),
        ]

    pdh: Any = ctypes.WinDLL("pdh.dll")
    query = ctypes.c_void_p()
    counter = ctypes.c_void_p()
    pdh.PdhOpenQueryW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    pdh.PdhAddEnglishCounterW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
    pdh.PdhGetFormattedCounterArrayW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_void_p,
    ]
    pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
    if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
        return []
    try:
        if (
            pdh.PdhAddEnglishCounterW(
                query,
                r"\GPU Process Memory(*)\Dedicated Usage",
                0,
                ctypes.byref(counter),
            )
            != 0
        ):
            return []
        if pdh.PdhCollectQueryData(query) != 0:
            return []
        buffer_size = ctypes.c_ulong(0)
        item_count = ctypes.c_ulong(0)
        status = pdh.PdhGetFormattedCounterArrayW(
            counter,
            0x00000400,  # PDH_FMT_LARGE
            ctypes.byref(buffer_size),
            ctypes.byref(item_count),
            None,
        )
        if ctypes.c_uint32(status).value != 0x800007D2:  # PDH_MORE_DATA
            return []
        buffer = ctypes.create_string_buffer(buffer_size.value)
        status = pdh.PdhGetFormattedCounterArrayW(
            counter,
            0x00000400,
            ctypes.byref(buffer_size),
            ctypes.byref(item_count),
            ctypes.cast(buffer, ctypes.c_void_p),
        )
        if status != 0:
            return []
        items = ctypes.cast(buffer, ctypes.POINTER(CounterValueItem))
        target_set = set(targets)
        by_pid: dict[int, int] = {}
        for index in range(item_count.value):
            item = items[index]
            match = re.match(r"pid_(\d+)_", item.name or "", flags=re.IGNORECASE)
            if match is None:
                continue
            process_id = int(match.group(1))
            if process_id not in target_set or item.value.status != 0:
                continue
            by_pid[process_id] = by_pid.get(process_id, 0) + max(0, int(item.value.large_value))
        return [
            {
                "process_id": process_id,
                "nvml_gpu_memory_bytes": None,
                "gpu_process_memory_bytes": memory_bytes,
                "gpu_process_memory_source": "windows-pdh-dedicated-usage",
            }
            for process_id, memory_bytes in sorted(by_pid.items())
            if memory_bytes > 0
        ]
    finally:
        pdh.PdhCloseQuery(query)


def _query_pcie_throughput() -> tuple[float | None, float | None]:
    result = subprocess.run(
        ["nvidia-smi", "dmon", "-c", "1", "-s", "t"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        return None, None
    data_lines = [
        line.split()
        for line in result.stdout.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not data_lines or len(data_lines[-1]) < 3:
        return None, None
    rx_mib_s = _optional_float(data_lines[-1][1])
    tx_mib_s = _optional_float(data_lines[-1][2])
    return (
        rx_mib_s * 1024 if rx_mib_s is not None else None,
        tx_mib_s * 1024 if tx_mib_s is not None else None,
    )


def query_nvidia_smi(
    *, process_ids: Iterable[int] | None = None
) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu,"
            "utilization.memory,power.draw,temperature.gpu,clocks.current.graphics,"
            "clocks.current.memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if gpu.returncode != 0:
        return {}, []
    gpu_values = parse_nvidia_gpu_query(gpu.stdout)
    rx_kib_s, tx_kib_s = _query_pcie_throughput()
    gpu_values["pcie_rx_kib_s"] = rx_kib_s
    gpu_values["pcie_tx_kib_s"] = tx_kib_s
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    process_rows = parse_nvml_process_memory(processes.stdout) if processes.returncode == 0 else []
    targets = set(process_ids) if process_ids is not None else None
    if targets is not None:
        process_rows = [row for row in process_rows if int(row["process_id"]) in targets]
    pdh_rows = query_windows_gpu_process_memory(targets or [])
    pdh_by_pid = {int(row["process_id"]): row for row in pdh_rows}
    merged: list[dict[str, Any]] = []
    for row in process_rows:
        process_id = int(row["process_id"])
        if row.get("nvml_gpu_memory_bytes") is not None:
            row["gpu_process_memory_bytes"] = row["nvml_gpu_memory_bytes"]
            merged.append(row)
        elif process_id in pdh_by_pid:
            merged.append(pdh_by_pid.pop(process_id))
    merged.extend(pdh_by_pid.values())
    return gpu_values, merged


def process_snapshot(process_id: int) -> dict[str, int | float | None]:
    try:
        process = psutil.Process(process_id)
        memory = process.memory_info()
        switches = process.num_ctx_switches()
        cpu_times = process.cpu_times()
        return {
            "process_id": process_id,
            "rss_bytes": int(memory.rss),
            "vms_bytes": int(memory.vms),
            "thread_count": int(process.num_threads()),
            "file_handle_count": (
                int(process.num_handles()) if hasattr(process, "num_handles") else None
            ),
            "voluntary_context_switches": int(switches.voluntary),
            "involuntary_context_switches": int(switches.involuntary),
            "cpu_user_seconds": float(cpu_times.user),
            "cpu_system_seconds": float(cpu_times.system),
        }
    except (psutil.Error, OSError):
        return {
            "process_id": process_id,
            "rss_bytes": None,
            "vms_bytes": None,
            "thread_count": None,
            "file_handle_count": None,
            "voluntary_context_switches": None,
            "involuntary_context_switches": None,
            "cpu_user_seconds": None,
            "cpu_system_seconds": None,
        }


class ResourceSampler:
    """Sample at one-second cadence while a measured session is active."""

    def __init__(
        self,
        *,
        phase: str,
        worker_count: int,
        repeat: int,
        coordinator_pid: int,
        worker_pids: Callable[[], list[int]],
        interval_seconds: float = 1.0,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("resource sample interval must be positive")
        self.phase = phase
        self.worker_count = worker_count
        self.repeat = repeat
        self.coordinator_pid = coordinator_pid
        self.worker_pids = worker_pids
        self.interval_seconds = interval_seconds
        self.resource_rows: list[dict[str, Any]] = []
        self.worker_rows: list[dict[str, Any]] = []
        self.gpu_process_rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"resource-sampler-{self.worker_count}-{self.phase}-{self.repeat}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 3))
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            sample_started = time.monotonic()
            timestamp = datetime.now(UTC).isoformat()
            monotonic_ns = time.monotonic_ns()
            pids = list(dict.fromkeys([self.coordinator_pid, *self.worker_pids()]))
            process_rows = [process_snapshot(pid) for pid in pids]
            rss_values = [row.get("rss_bytes") for row in process_rows]
            aggregate_rss = sum(int(value) for value in rss_values if value is not None)
            virtual = psutil.virtual_memory()
            swap = psutil.swap_memory()
            gpu, gpu_processes = query_nvidia_smi(process_ids=pids)
            base = {
                "phase": self.phase,
                "worker_count": self.worker_count,
                "repeat": self.repeat,
                "wall_clock_utc": timestamp,
                "monotonic_timestamp_ns": monotonic_ns,
            }
            self.resource_rows.append(
                {
                    **base,
                    "gpu_memory_used_bytes": _mib_to_bytes(gpu.get("gpu_memory_used_mib")),
                    "gpu_memory_free_bytes": _mib_to_bytes(gpu.get("gpu_memory_free_mib")),
                    "gpu_memory_total_bytes": _mib_to_bytes(gpu.get("gpu_memory_total_mib")),
                    "gpu_utilisation_percent": gpu.get("gpu_utilisation_percent"),
                    "memory_controller_utilisation_percent": gpu.get(
                        "memory_controller_utilisation_percent"
                    ),
                    "power_draw_watts": gpu.get("power_draw_watts"),
                    "temperature_c": gpu.get("temperature_c"),
                    "graphics_clock_mhz": gpu.get("graphics_clock_mhz"),
                    "memory_clock_mhz": gpu.get("memory_clock_mhz"),
                    "pcie_tx_kib_s": gpu.get("pcie_tx_kib_s"),
                    "pcie_rx_kib_s": gpu.get("pcie_rx_kib_s"),
                    "aggregate_experiment_rss_bytes": aggregate_rss,
                    "system_total_ram_bytes": int(virtual.total),
                    "system_available_ram_bytes": int(virtual.available),
                    "system_memory_used_fraction": float(virtual.percent / 100),
                    "system_commit_or_swap_used_bytes": int(swap.used),
                    "python_process_count": len(pids),
                    "cuda_context_count": len(gpu_processes),
                    "sampler_event_loop_lag_ms": max(
                        0.0,
                        (time.monotonic() - sample_started - self.interval_seconds) * 1000,
                    ),
                    "gpu_aggregate_memory_source": "nvidia-smi",
                }
            )
            for row in process_rows:
                role = "coordinator" if row["process_id"] == self.coordinator_pid else "worker"
                self.worker_rows.append({**base, "role": role, **row})
            for row in gpu_processes:
                self.gpu_process_rows.append({**base, **row})
            remaining = self.interval_seconds - (time.monotonic() - sample_started)
            self._stop.wait(max(0.0, remaining))


def environment_command_snapshot() -> dict[str, Any]:
    """Small non-import-mutating environment fingerprint used before and after a run."""

    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "process_id": os.getpid(),
        "python_executable": sys.executable,
        "working_directory": str(Path.cwd().resolve()),
        "virtual_memory": dict(psutil.virtual_memory()._asdict()),
        "swap_memory": dict(psutil.swap_memory()._asdict()),
    }
