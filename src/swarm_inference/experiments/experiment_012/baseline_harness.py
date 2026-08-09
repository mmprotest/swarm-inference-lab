"""Freeze Experiment 012 flat and local-scheduler-tree baselines.

The controlled workload is synthetic and exact.  Every worker is nevertheless
an independent persistent OS process with a real loopback TCP endpoint.  This
module exists to measure coordinator/protocol scaling, not model quality.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

from swarm_inference.microworker_protocol import (
    MAGIC,
    NETWORK_PROFILES,
    PROTOCOL_VERSION,
    LinkProfile,
    aggregate_digest,
    combine_aggregates,
    leaf_aggregate,
    make_leaf_request,
    receive_message,
    send_message,
    shaped_round_trip,
    worker_contribution,
)

REQUIRED_WORKER_COUNTS = (2, 8, 32, 128, 512, 1000)
BASELINE_MODES = ("flat_root_to_leaf", "local_scheduler_tree_root_to_leaf")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: dict[str, Any], lock: threading.Lock | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"

    def append() -> None:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)

    if lock is None:
        append()
    else:
        with lock:
            append()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _process_rss_bytes(process_id: int) -> int | None:
    if os.name != "nt":
        status = Path(f"/proc/{process_id}/status")
        if not status.exists():
            return None
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    handle = kernel32.OpenProcess(0x0400 | 0x0010, False, process_id)
    if not handle:
        return None
    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), ctypes.sizeof(counters)):
            return None
        return int(counters.PeakWorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)


def _system_memory() -> dict[str, int | None]:
    if os.name != "nt":
        return {"total_physical_bytes": None, "available_physical_bytes": None}

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(status)):
        return {"total_physical_bytes": None, "available_physical_bytes": None}
    return {
        "total_physical_bytes": int(status.ullTotalPhys),
        "available_physical_bytes": int(status.ullAvailPhys),
    }


@dataclass(slots=True)
class WorkerProcess:
    worker_id: str
    worker_index: int
    endpoint: str
    process_id: int
    launcher_process_id: int
    process: subprocess.Popen[bytes]
    log_handle: TextIO
    ready: dict[str, Any]
    runtime_profile: dict[str, Any] = field(default_factory=dict)
    peak_rss_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class LocalSchedulerNode:
    node_id: str
    worker: WorkerProcess | None = None
    children: tuple[LocalSchedulerNode, ...] = ()


@dataclass(frozen=True, slots=True)
class LocalSchedulerTopology:
    root_children: tuple[LocalSchedulerNode, ...]
    depth: int
    node_count: int


def build_local_scheduler_topology(
    workers: list[WorkerProcess], branch_factor: int
) -> LocalSchedulerTopology:
    if branch_factor < 2:
        raise ValueError("branch factor must be at least two")

    def build(items: list[WorkerProcess], prefix: str) -> tuple[LocalSchedulerNode, ...]:
        if len(items) <= branch_factor:
            return tuple(
                LocalSchedulerNode(node_id=f"{prefix}/{worker.worker_id}", worker=worker)
                for worker in items
            )
        group_size = math.ceil(len(items) / branch_factor)
        groups = [items[index : index + group_size] for index in range(0, len(items), group_size)]
        return tuple(
            LocalSchedulerNode(
                node_id=f"{prefix}/local-group-{index:04d}",
                children=build(group, f"{prefix}/local-group-{index:04d}"),
            )
            for index, group in enumerate(groups)
        )

    root_children = build(workers, "root")

    def depth(node: LocalSchedulerNode) -> int:
        return 1 if node.worker is not None else 1 + max(depth(child) for child in node.children)

    def count(node: LocalSchedulerNode) -> int:
        return 1 + sum(count(child) for child in node.children)

    return LocalSchedulerTopology(
        root_children=root_children,
        depth=max((depth(node) for node in root_children), default=0),
        node_count=sum(count(node) for node in root_children),
    )


class WorkerPool:
    def __init__(
        self,
        *,
        count: int,
        directory: Path,
        startup_deadline_s: float,
        protocol_script: Path,
        runtime_profiles: tuple[dict[str, Any], ...] | None = None,
        include_site_packages: bool = False,
    ) -> None:
        self.count = count
        self.directory = directory
        self.startup_deadline_s = startup_deadline_s
        self.protocol_script = protocol_script
        self.runtime_profiles = runtime_profiles or tuple({} for _ in range(count))
        self.include_site_packages = include_site_packages
        if len(self.runtime_profiles) != count:
            raise ValueError("runtime profile count must equal worker count")
        self.workers: list[WorkerProcess] = []
        self.startup_records: list[dict[str, Any]] = []

    def start(self) -> list[WorkerProcess]:
        self.directory.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()
        creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
        try:
            for worker_index in range(self.count):
                worker_id = f"worker-{worker_index:06d}"
                worker_directory = self.directory / worker_id
                worker_directory.mkdir(parents=True, exist_ok=True)
                ready_path = worker_directory / "ready.json"
                config_path = worker_directory / "config.json"
                log_path = worker_directory / "process.log"
                trace_path = worker_directory / "trace.jsonl"
                config = {
                    "worker_id": worker_id,
                    "worker_index": worker_index,
                    "host": "127.0.0.1",
                    "port": 0,
                    "ready_path": str(ready_path.resolve()),
                    "trace_path": str(trace_path.resolve()),
                    "log_path": str(log_path.resolve()),
                    "runtime_profile": dict(self.runtime_profiles[worker_index]),
                }
                _write_json(config_path, config)
                log_handle = log_path.open("w", encoding="utf-8", newline="\n")
                try:
                    if self.include_site_packages:
                        # Executing this file directly puts ``swarm_inference`` first on
                        # sys.path, where its logging.py shadows the standard library while
                        # importing torch. Module execution preserves normal import semantics.
                        command = [
                            sys.executable,
                            "-m",
                            "swarm_inference.microworker_protocol",
                            "serve",
                            "--config",
                            str(config_path),
                        ]
                    else:
                        command = [
                            sys.executable,
                            "-S",
                            str(self.protocol_script),
                            "serve",
                            "--config",
                            str(config_path),
                        ]
                    process = subprocess.Popen(
                        command,
                        cwd=str(self.protocol_script.parents[2]),
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        creationflags=creation_flags,
                    )
                except BaseException:
                    log_handle.close()
                    raise
                placeholder = WorkerProcess(
                    worker_id=worker_id,
                    worker_index=worker_index,
                    endpoint="",
                    process_id=process.pid,
                    launcher_process_id=process.pid,
                    process=process,
                    log_handle=log_handle,
                    ready={},
                    runtime_profile=dict(self.runtime_profiles[worker_index]),
                )
                self.workers.append(placeholder)
                self.startup_records.append(
                    {
                        "worker_id": worker_id,
                        "worker_index": worker_index,
                        "process_id": process.pid,
                        "spawned_unix_ns": time.time_ns(),
                        "runtime_profile": dict(self.runtime_profiles[worker_index]),
                    }
                )

            pending = {worker.worker_id: worker for worker in self.workers}
            while pending:
                if time.perf_counter() - started_at >= self.startup_deadline_s:
                    raise TimeoutError(
                        f"{len(pending)} of {self.count} workers missed the startup deadline"
                    )
                for worker_id, worker in list(pending.items()):
                    ready_path = self.directory / worker_id / "ready.json"
                    if ready_path.exists():
                        ready = json.loads(ready_path.read_text(encoding="utf-8"))
                        reported_process_id = int(ready["process_id"])
                        if reported_process_id <= 0:
                            raise RuntimeError(f"invalid ready process identity for {worker_id}")
                        worker.endpoint = str(ready["endpoint"])
                        worker.ready = ready
                        worker.process_id = reported_process_id
                        worker.peak_rss_bytes = _process_rss_bytes(worker.launcher_process_id)
                        pending.pop(worker_id)
                    elif worker.process.poll() is not None:
                        raise RuntimeError(
                            f"{worker_id} exited during startup with code {worker.process.returncode}"
                        )
                if pending:
                    time.sleep(0.02)
            return self.workers
        except BaseException as error:
            self.startup_records.append(
                {
                    "event": "pool_start_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "started_workers": len(self.workers),
                    "elapsed_s": time.perf_counter() - started_at,
                }
            )
            raise

    @staticmethod
    def _control(worker: WorkerProcess, kind: str) -> dict[str, Any]:
        host, port_text = worker.endpoint.rsplit(":", 1)
        connection = socket.create_connection((host, int(port_text)), timeout=2.0)
        try:
            connection.settimeout(2.0)
            send_message(
                connection,
                {"magic": MAGIC, "protocol_version": PROTOCOL_VERSION, "kind": kind},
            )
            response, _ = receive_message(connection)
            return response
        finally:
            connection.close()

    def sample_memory(self) -> int | None:
        snapshot = self.sample_memory_snapshot()
        return snapshot["maximum_worker_rss_bytes"] if snapshot is not None else None

    def sample_memory_snapshot(self) -> dict[str, int | float] | None:
        observed: list[int] = []
        for worker in self.workers:
            current = _process_rss_bytes(worker.launcher_process_id)
            if current is not None:
                worker.peak_rss_bytes = max(worker.peak_rss_bytes or 0, current)
            if worker.peak_rss_bytes is not None:
                observed.append(worker.peak_rss_bytes)
        if not observed:
            return None
        return {
            "observed_workers": len(observed),
            "total_worker_rss_bytes": sum(observed),
            "maximum_worker_rss_bytes": max(observed),
            "median_worker_rss_bytes": statistics.median(observed),
        }

    def stop(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "requested": len(self.workers),
            "graceful": 0,
            "terminated": 0,
            "killed": 0,
            "errors": [],
        }
        ready_workers = [worker for worker in self.workers if worker.endpoint]
        with ThreadPoolExecutor(max_workers=min(64, max(1, len(ready_workers)))) as executor:
            controls = {
                executor.submit(self._control, worker, "shutdown"): worker
                for worker in ready_workers
            }
            for future, worker in list(controls.items()):
                try:
                    future.result(timeout=5.0)
                except BaseException as error:
                    result["errors"].append(
                        {
                            "worker_id": worker.worker_id,
                            "phase": "shutdown_request",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
        for worker in self.workers:
            try:
                worker.process.wait(timeout=5.0)
                result["graceful"] += 1
            except subprocess.TimeoutExpired:
                worker.process.terminate()
                result["terminated"] += 1
                try:
                    worker.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    worker.process.kill()
                    result["killed"] += 1
                    worker.process.wait(timeout=3.0)
            finally:
                worker.log_handle.close()
        return result


@dataclass(slots=True)
class TrialCollector:
    expected: int
    condition: threading.Condition = field(default_factory=threading.Condition)
    results: list[tuple[WorkerProcess, dict[str, Any], dict[str, Any], str]] = field(
        default_factory=list
    )
    error: BaseException | None = None
    coordinator_waits: int = 0
    scheduler_dispatch_ns: int = 0

    def record(
        self,
        result: tuple[WorkerProcess, dict[str, Any], dict[str, Any], str],
    ) -> None:
        with self.condition:
            if self.error is None:
                self.results.append(result)
            self.condition.notify_all()

    def fail(self, error: BaseException) -> None:
        with self.condition:
            if self.error is None:
                self.error = error
            self.condition.notify_all()

    def wait(
        self, deadline_unix_ns: int
    ) -> list[tuple[WorkerProcess, dict[str, Any], dict[str, Any], str]]:
        with self.condition:
            while len(self.results) < self.expected and self.error is None:
                remaining_ns = deadline_unix_ns - time.time_ns()
                if remaining_ns <= 0:
                    self.error = TimeoutError("baseline collector deadline elapsed")
                    break
                self.coordinator_waits += 1
                self.condition.wait(timeout=remaining_ns / 1_000_000_000)
            if self.error is not None:
                raise self.error
            return list(self.results)


class BaselineRunner:
    def __init__(
        self,
        *,
        workers: list[WorkerProcess],
        output_directory: Path,
        branch_factor: int,
        maximum_root_concurrency: int,
        payload_bytes: int,
        operation_deadline_s: float,
        network_profile: LinkProfile,
        cycle_id: str = "BASELINE-012",
    ) -> None:
        self.workers = workers
        self.output_directory = output_directory
        self.branch_factor = branch_factor
        self.maximum_root_concurrency = maximum_root_concurrency
        self.payload_bytes = payload_bytes
        self.operation_deadline_s = operation_deadline_s
        self.network_profile = network_profile
        self.cycle_id = cycle_id
        self.trace_lock = threading.Lock()
        self.trace_path = output_directory / "traces" / "root.jsonl"
        self.leaf_path = output_directory / "raw" / "leaf-observations.jsonl"
        self.local_topology = build_local_scheduler_topology(workers, branch_factor)
        self.expected = combine_aggregates(
            [
                leaf_aggregate(worker.worker_id, worker_contribution(worker.worker_index))
                for worker in workers
            ]
        )

    def _request(
        self,
        *,
        worker: WorkerProcess,
        operation_id: str,
        generation: int,
        local_path: str,
    ) -> tuple[WorkerProcess, dict[str, Any], dict[str, Any], str]:
        request_id = f"{operation_id}:{worker.worker_id}"
        deadline_unix_ns = time.time_ns() + int(self.operation_deadline_s * 1_000_000_000)
        request = make_leaf_request(
            request_id=request_id,
            operation_id=operation_id,
            execution_generation=generation,
            parent_worker="stage-owner",
            worker_id=worker.worker_id,
            worker_index=worker.worker_index,
            deadline_unix_ns=deadline_unix_ns,
            route_lease_id="baseline-route-generation-1",
            ordering_key=worker.worker_id,
            payload_bytes=self.payload_bytes,
            trace_id=operation_id,
            span_id=f"root-to-{worker.worker_id}",
        )
        response, transport = shaped_round_trip(
            endpoint=worker.endpoint,
            message=request,
            profile=self.network_profile,
            timeout_s=self.operation_deadline_s,
            sender_id="stage-owner",
            receiver_id=worker.worker_id,
        )
        if response.get("request_id") != request_id:
            raise ValueError("baseline response request identity mismatch")
        event = {
            "event": "root_leaf_round_trip",
            "operation_id": operation_id,
            "request_id": request_id,
            "sender_worker_id": "stage-owner",
            "receiver_worker_id": worker.worker_id,
            "sender_process_id": os.getpid(),
            "receiver_process_id": worker.process_id,
            "local_scheduler_path": local_path,
            "network_depth": 1,
            "timestamp_unix_ns": time.time_ns(),
            **transport,
        }
        _append_jsonl(self.trace_path, event, self.trace_lock)
        return worker, dict(response["aggregate"]), transport, local_path

    def _flat(
        self, operation_id: str, generation: int
    ) -> tuple[list[tuple[WorkerProcess, dict[str, Any], dict[str, Any], str]], dict[str, int]]:
        results: list[tuple[WorkerProcess, dict[str, Any], dict[str, Any], str]] = []
        dispatch_started = time.perf_counter_ns()
        with ThreadPoolExecutor(max_workers=self.maximum_root_concurrency) as executor:
            futures: dict[Future[Any], WorkerProcess] = {
                executor.submit(
                    self._request,
                    worker=worker,
                    operation_id=operation_id,
                    generation=generation,
                    local_path="root",
                ): worker
                for worker in self.workers
            }
            dispatch_ns = time.perf_counter_ns() - dispatch_started
            for future in as_completed(futures):
                results.append(future.result())
        return results, {"scheduler_dispatch_ns": dispatch_ns, "root_coordinator_waits": 1}

    def _local_scheduler(
        self, operation_id: str, generation: int
    ) -> tuple[list[tuple[WorkerProcess, dict[str, Any], dict[str, Any], str]], dict[str, int]]:
        collector = TrialCollector(expected=len(self.workers))
        executor = ThreadPoolExecutor(max_workers=self.maximum_root_concurrency)

        def dispatch(node: LocalSchedulerNode, path: str) -> None:
            try:
                if node.worker is not None:
                    collector.record(
                        self._request(
                            worker=node.worker,
                            operation_id=operation_id,
                            generation=generation,
                            local_path=path,
                        )
                    )
                    return
                started = time.perf_counter_ns()
                for child in node.children:
                    executor.submit(dispatch, child, f"{path}/{child.node_id}")
                with collector.condition:
                    collector.scheduler_dispatch_ns += time.perf_counter_ns() - started
            except BaseException as error:
                collector.fail(error)

        dispatch_started = time.perf_counter_ns()
        for node in self.local_topology.root_children:
            executor.submit(dispatch, node, node.node_id)
        root_dispatch_ns = time.perf_counter_ns() - dispatch_started
        deadline = time.time_ns() + int(self.operation_deadline_s * 1_000_000_000)
        try:
            results = collector.wait(deadline)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        return results, {
            "scheduler_dispatch_ns": root_dispatch_ns + collector.scheduler_dispatch_ns,
            "root_coordinator_waits": collector.coordinator_waits,
        }

    def _tree_reduce(self, aggregates: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
        level = list(aggregates)
        rounds = 0
        while len(level) > 1:
            level = [
                combine_aggregates(level[index : index + self.branch_factor])
                for index in range(0, len(level), self.branch_factor)
            ]
            rounds += 1
        return level[0], rounds

    def run_trial(
        self,
        *,
        mode: str,
        trial_index: int,
        warmup: bool,
        generation: int,
    ) -> dict[str, Any]:
        if mode not in BASELINE_MODES:
            raise ValueError(f"unsupported baseline mode {mode!r}")
        operation_id = (
            f"baseline-{mode}-n{len(self.workers)}-"
            f"{'warmup' if warmup else 'trial'}-{trial_index}-{uuid4().hex[:8]}"
        )
        wall_started_ns = time.perf_counter_ns()
        cpu_started_ns = time.process_time_ns()
        status = "ok"
        error: dict[str, Any] | None = None
        results: list[tuple[WorkerProcess, dict[str, Any], dict[str, Any], str]] = []
        dispatch: dict[str, int] = {}
        reduction_started_ns = 0
        reduction_ns = 0
        reduction_depth = 0
        actual: dict[str, Any] | None = None
        try:
            if mode == "flat_root_to_leaf":
                results, dispatch = self._flat(operation_id, generation)
            else:
                results, dispatch = self._local_scheduler(operation_id, generation)
            reduction_started_ns = time.perf_counter_ns()
            actual, reduction_depth = self._tree_reduce([item[1] for item in results])
            reduction_ns = time.perf_counter_ns() - reduction_started_ns
            if actual != self.expected:
                status = "incorrect"
                error = {
                    "error_type": "CorrectnessError",
                    "error": "aggregate or contribution proof differs from flat reference",
                }
        except BaseException as exception:
            status = "failed"
            error = {"error_type": type(exception).__name__, "error": str(exception)}
        root_cpu_ns = time.process_time_ns() - cpu_started_ns
        latency_ns = time.perf_counter_ns() - wall_started_ns
        leaf_latencies_ms = [item[2]["elapsed_ns"] / 1_000_000 for item in results]
        request_bytes = sum(int(item[2]["request_bytes"]) for item in results)
        response_bytes = sum(int(item[2]["response_bytes"]) for item in results)
        connect_ns = sum(int(item[2]["connect_ns"]) for item in results)
        client_cpu_ns = sum(int(item[2]["client_cpu_ns"]) for item in results)
        logical_depth = 1 if mode == "flat_root_to_leaf" else self.local_topology.depth
        row: dict[str, Any] = {
            "schema_version": "1.0",
            "experiment_id": "012",
            "cycle_id": self.cycle_id,
            "mode": mode,
            "worker_count": len(self.workers),
            "branch_factor": self.branch_factor,
            "maximum_root_concurrency": self.maximum_root_concurrency,
            "network_profile": self.network_profile.name,
            "payload_bytes": self.payload_bytes,
            "trial_index": trial_index,
            "warmup": warmup,
            "operation_id": operation_id,
            "execution_generation": generation,
            "status": status,
            "correctness": status == "ok",
            "expected": self.expected,
            "actual": actual,
            "expected_digest": aggregate_digest(self.expected),
            "actual_digest": aggregate_digest(actual) if actual is not None else None,
            "root_rpc_count": len(results),
            "root_leaf_rpc_count": len(results),
            "root_messages_sent": len(results),
            "root_messages_received": len(results),
            "root_messages_total": 2 * len(results),
            "root_bytes_sent": request_bytes,
            "root_bytes_received": response_bytes,
            "root_bytes_total": request_bytes + response_bytes,
            "root_direct_degree": len({item[0].worker_id for item in results}),
            "root_serial_waits": len(results),
            "root_coordinator_waits": int(dispatch.get("root_coordinator_waits", 0)),
            "root_cpu_ns": root_cpu_ns,
            "root_client_cpu_ns": client_cpu_ns,
            "scheduler_dispatch_ns": int(dispatch.get("scheduler_dispatch_ns", 0)),
            "connection_count": len(results),
            "connection_ns": connect_ns,
            "total_messages": 2 * len(results),
            "total_bytes": request_bytes + response_bytes,
            "leaf_rpc_count": len(results),
            "hierarchy_depth": logical_depth,
            "observed_network_depth": 1,
            "fanout_depth": logical_depth,
            "reduction_depth": reduction_depth,
            "critical_path_sync_points": 1 + reduction_depth,
            "end_to_end_latency_ns": latency_ns,
            "end_to_end_latency_ms": latency_ns / 1_000_000,
            "throughput_ops_s": 1_000_000_000 / latency_ns if latency_ns else 0.0,
            "leaf_latency_p50_ms": _percentile(leaf_latencies_ms, 50),
            "leaf_latency_p95_ms": _percentile(leaf_latencies_ms, 95),
            "leaf_latency_p99_ms": _percentile(leaf_latencies_ms, 99),
            "leaf_latency_min_ms": min(leaf_latencies_ms, default=0.0),
            "leaf_latency_max_ms": max(leaf_latencies_ms, default=0.0),
            "reduction_ns": reduction_ns,
            "maximum_queue_depth": 1 if results else 0,
            "retries": 0,
            "failures": 0 if status == "ok" else 1,
            "stragglers": sum(
                latency > _percentile(leaf_latencies_ms, 50) * 2 for latency in leaf_latencies_ms
            ),
            "duplicated_work": 0,
            "unexpected_serialization": mode == "local_scheduler_tree_root_to_leaf",
            "error": error,
            "measured_unix_ns": time.time_ns(),
        }
        for worker, _aggregate, transport, local_path in results:
            _append_jsonl(
                self.leaf_path,
                {
                    "operation_id": operation_id,
                    "mode": mode,
                    "worker_count": len(self.workers),
                    "worker_id": worker.worker_id,
                    "worker_index": worker.worker_index,
                    "worker_process_id": worker.process_id,
                    "local_scheduler_path": local_path,
                    "warmup": warmup,
                    **transport,
                },
                self.trace_lock,
            )
        return row


def _topology_record(
    *, workers: list[WorkerProcess], topology: LocalSchedulerTopology, branch_factor: int
) -> dict[str, Any]:
    def node_value(node: LocalSchedulerNode) -> dict[str, Any]:
        return {
            "node_id": node.node_id,
            "node_kind": "worker_endpoint" if node.worker is not None else "root_local_task",
            "worker_id": node.worker.worker_id if node.worker is not None else None,
            "endpoint": node.worker.endpoint if node.worker is not None else None,
            "children": [node_value(child) for child in node.children],
        }

    return {
        "worker_count": len(workers),
        "branch_factor": branch_factor,
        "logical_depth": topology.depth,
        "local_scheduler_node_count": topology.node_count - len(workers),
        "network_depth": 1,
        "runtime_truth": "all leaf TCP edges originate at stage-owner",
        "root_children": [node_value(node) for node in topology.root_children],
    }


def _source_identity(repo_root: Path, files: list[Path]) -> dict[str, Any]:
    def git(*arguments: str) -> dict[str, Any]:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "command": ["git", *arguments],
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }

    identities = []
    for path in files:
        identities.append(
            {
                "path": str(path.relative_to(repo_root)).replace("\\", "/"),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "git_diff_stat": git("diff", "--stat"),
        "source_files": identities,
        "captured_unix_ns": time.time_ns(),
    }


def _summarize(rows: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for worker_count in sorted({int(row["worker_count"]) for row in rows}):
        for mode in BASELINE_MODES:
            selected = [
                row
                for row in rows
                if row["worker_count"] == worker_count
                and row["mode"] == mode
                and not row["warmup"]
                and row["status"] == "ok"
                and row["correctness"]
            ]
            if not selected:
                continue
            summaries.append(
                {
                    "worker_count": worker_count,
                    "mode": mode,
                    "successful_trials": len(selected),
                    "root_messages_median": statistics.median(
                        row["root_messages_total"] for row in selected
                    ),
                    "root_bytes_median": statistics.median(
                        row["root_bytes_total"] for row in selected
                    ),
                    "root_serial_waits_median": statistics.median(
                        row["root_serial_waits"] for row in selected
                    ),
                    "root_direct_degree_median": statistics.median(
                        row["root_direct_degree"] for row in selected
                    ),
                    "root_cpu_ms_median": statistics.median(
                        row["root_cpu_ns"] / 1_000_000 for row in selected
                    ),
                    "end_to_end_latency_p50_ms": _percentile(
                        [row["end_to_end_latency_ms"] for row in selected], 50
                    ),
                    "end_to_end_latency_p95_ms": _percentile(
                        [row["end_to_end_latency_ms"] for row in selected], 95
                    ),
                    "end_to_end_latency_p99_ms": _percentile(
                        [row["end_to_end_latency_ms"] for row in selected], 99
                    ),
                    "throughput_ops_s_median": statistics.median(
                        row["throughput_ops_s"] for row in selected
                    ),
                    "hierarchy_depth_median": statistics.median(
                        row["hierarchy_depth"] for row in selected
                    ),
                    "total_messages_median": statistics.median(
                        row["total_messages"] for row in selected
                    ),
                }
            )
    return {
        "schema_version": "1.0",
        "cycle_id": "BASELINE-012",
        "evidence_classification": "measured synthetic workload over independent loopback processes",
        "attempts": attempts,
        "summaries": summaries,
        "failed_trials": [row for row in rows if row["status"] != "ok"],
        "generated_unix_ns": time.time_ns(),
    }


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = list(summary["summaries"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_baselines(
    *,
    output_directory: Path,
    worker_counts: tuple[int, ...],
    branch_factor: int,
    maximum_root_concurrency: int,
    payload_bytes: int,
    warmup_trials: int,
    measured_trials: int,
    network_profile: LinkProfile,
    operation_deadline_s: float,
    startup_deadline_s: float,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    protocol_script = repo_root / "src" / "swarm_inference" / "microworker_protocol.py"
    baseline_file = Path(__file__).resolve()
    raw_directory = output_directory / "raw"
    traces_directory = output_directory / "traces"
    topology_directory = output_directory / "topologies"
    raw_directory.mkdir(parents=True, exist_ok=True)
    traces_directory.mkdir(parents=True, exist_ok=True)
    topology_directory.mkdir(parents=True, exist_ok=True)
    trials_path = raw_directory / "trials.jsonl"
    errors_path = raw_directory / "errors.jsonl"
    rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    source_identity = _source_identity(repo_root, [protocol_script, baseline_file])
    _write_json(output_directory / "source-identity.json", source_identity)
    snapshot_directory = raw_directory / "source-snapshot"
    snapshot_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(protocol_script, snapshot_directory / protocol_script.name)
    shutil.copy2(baseline_file, snapshot_directory / baseline_file.name)
    _write_json(
        output_directory / "environment.json",
        {
            "captured_unix_ns": time.time_ns(),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "hostname": socket.gethostname(),
            "physical_boundary": "one host; loopback endpoints only",
            "network_evidence": "simulated/shaped, not physical LAN or WAN",
            **_system_memory(),
        },
    )
    _write_json(
        output_directory / "network-profiles.json",
        {name: asdict(profile) for name, profile in NETWORK_PROFILES.items()},
    )

    for worker_count in worker_counts:
        scale_started_ns = time.perf_counter_ns()
        scale_directory = raw_directory / f"workers-{worker_count:04d}"
        pool = WorkerPool(
            count=worker_count,
            directory=scale_directory / "processes",
            startup_deadline_s=startup_deadline_s,
            protocol_script=protocol_script,
        )
        attempt: dict[str, Any] = {
            "worker_count": worker_count,
            "status": "starting",
            "attempted_unix_ns": time.time_ns(),
            "requested_processes": worker_count,
        }
        try:
            workers = pool.start()
            attempt["status"] = "started"
            attempt["started_processes"] = len(workers)
            attempt["startup_elapsed_ms"] = (time.perf_counter_ns() - scale_started_ns) / 1_000_000
            identities = [worker.ready for worker in workers]
            _write_json(scale_directory / "worker-identities.json", identities)
            runner = BaselineRunner(
                workers=workers,
                output_directory=output_directory,
                branch_factor=branch_factor,
                maximum_root_concurrency=maximum_root_concurrency,
                payload_bytes=payload_bytes,
                operation_deadline_s=operation_deadline_s,
                network_profile=network_profile,
            )
            _write_json(
                topology_directory / f"workers-{worker_count:04d}.json",
                _topology_record(
                    workers=workers,
                    topology=runner.local_topology,
                    branch_factor=branch_factor,
                ),
            )
            generation = 1
            for warmup_index in range(warmup_trials):
                for mode in BASELINE_MODES:
                    row = runner.run_trial(
                        mode=mode,
                        trial_index=warmup_index,
                        warmup=True,
                        generation=generation,
                    )
                    generation += 1
                    rows.append(row)
                    _append_jsonl(trials_path, row)
                    if row["status"] != "ok":
                        _append_jsonl(errors_path, row)
            for trial_index in range(measured_trials):
                for mode in BASELINE_MODES:
                    row = runner.run_trial(
                        mode=mode,
                        trial_index=trial_index,
                        warmup=False,
                        generation=generation,
                    )
                    generation += 1
                    rows.append(row)
                    _append_jsonl(trials_path, row)
                    if row["status"] != "ok":
                        _append_jsonl(errors_path, row)
            attempt["peak_worker_rss_bytes"] = pool.sample_memory()
            attempt["status"] = "completed"
        except BaseException as error:
            attempt.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "started_processes": len(pool.workers),
                    "startup_records": pool.startup_records,
                }
            )
            _append_jsonl(errors_path, {"event": "scale_failed", **attempt})
        finally:
            attempt["shutdown"] = pool.stop()
            attempt["total_scale_elapsed_ms"] = (
                time.perf_counter_ns() - scale_started_ns
            ) / 1_000_000
            attempts.append(attempt)
            _write_json(raw_directory / "scale-attempts.json", attempts)

    summary = _summarize(rows, attempts)
    _write_json(output_directory / "baseline-summary.json", summary)
    _write_summary_csv(output_directory / "baseline-summary.csv", summary)
    return summary


def _parse_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not counts or any(item <= 0 for item in counts):
        raise argparse.ArgumentTypeError("worker counts must be positive comma-separated integers")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze Experiment 012 baselines")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=_parse_counts, default=REQUIRED_WORKER_COUNTS)
    parser.add_argument("--branch-factor", type=int, default=8)
    parser.add_argument("--maximum-root-concurrency", type=int, default=32)
    parser.add_argument("--payload-bytes", type=int, default=256)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--measured-trials", type=int, default=5)
    parser.add_argument(
        "--network-profile", choices=sorted(NETWORK_PROFILES), default="same_host_shaped"
    )
    parser.add_argument("--operation-deadline-s", type=float, default=30.0)
    parser.add_argument("--startup-deadline-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    summary = run_baselines(
        output_directory=args.output.resolve(),
        worker_counts=args.counts,
        branch_factor=args.branch_factor,
        maximum_root_concurrency=args.maximum_root_concurrency,
        payload_bytes=args.payload_bytes,
        warmup_trials=args.warmup_trials,
        measured_trials=args.measured_trials,
        network_profile=NETWORK_PROFILES[args.network_profile],
        operation_deadline_s=args.operation_deadline_s,
        startup_deadline_s=args.startup_deadline_s,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    failed_attempts = [item for item in summary["attempts"] if item["status"] == "failed"]
    failed_trials = summary["failed_trials"]
    return 1 if failed_attempts or failed_trials else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_MODES",
    "REQUIRED_WORKER_COUNTS",
    "BaselineRunner",
    "LocalSchedulerNode",
    "LocalSchedulerTopology",
    "WorkerPool",
    "build_local_scheduler_topology",
    "run_baselines",
]
