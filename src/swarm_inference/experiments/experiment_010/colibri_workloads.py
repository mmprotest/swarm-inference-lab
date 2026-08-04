"""Mandatory real-model Colibri workload measurements for Experiment 010.

This module orchestrates the existing patched Colibri executable and native C
expert workers.  It is not a model implementation: every measured token is
sampled by Colibri's original fixed-replay path, and every distributed expert
result enters through the downstream hook inside ``moe()``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import time
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import psutil
from tokenizers import Tokenizer

from swarm_inference.experiments.experiment_010.colibri_native import (
    COLIBRI_MODEL_REVISION,
    NativeColibriExpertWorker,
    NativeColibriExpertWorkerManager,
    _generated_token_ids,
    _last_jsonl_record,
    _worker_process_accounting,
    write_colibri_expert_plan,
)
from swarm_inference.experiments.experiment_010.colibri_token_path import (
    _base_environment,
    build_local_reference_suite,
)
from swarm_inference.experiments.experiment_010.relay import ExpertRelayManager
from swarm_inference.experiments.experiment_010.transport import NETWORK_PROFILES

_SPEED = re.compile(r"Speed:\s*([0-9.]+)\s*tok/s\s*\(([0-9.]+)s for (\d+) tokens\)")
_TIMING = re.compile(
    r"TTFT:\s*([0-9.]+)s\s*\|\s*Prefill:\s*([0-9.]+)s\s*\|"
    r"\s*Decode after first token:\s*([0-9.]+)s"
)
_CACHE = re.compile(r"Expert cache hit rate:\s*([0-9.]+)%\s*\(hit=(\d+) miss=(\d+)\)")
_RSS = re.compile(r"PEAK RSS:\s*([0-9.]+)\s*GB")


@dataclass(frozen=True, slots=True)
class RealCandidate:
    name: str
    bank_paths: tuple[Path, ...] = ()
    mode: str = "rpc"
    response_mode: str = "per_expert_exact"
    data_plane: str = "direct_tcp"
    network_profile: str = "loopback_unshaped"
    shard_layout: str = "whole"
    exact_contract: bool = True
    coordinator_model: Path | None = None
    worker_memory_budget_bytes: int = 512 * 1024 * 1024

    @property
    def local(self) -> bool:
        return not self.bank_paths


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _last_jsonl(path: Path) -> dict[str, Any]:
    return _last_jsonl_record(path)


def parse_engine_metrics(stdout: str) -> dict[str, Any]:
    speed = _SPEED.search(stdout)
    timing = _TIMING.search(stdout)
    cache = _CACHE.search(stdout)
    rss = _RSS.search(stdout)
    return {
        "decode_tokens_per_second": float(speed.group(1)) if speed else None,
        "model_elapsed_seconds": float(speed.group(2)) if speed else None,
        "reported_generated_tokens": int(speed.group(3)) if speed else None,
        "ttft_seconds": float(timing.group(1)) if timing else None,
        "prefill_seconds": float(timing.group(2)) if timing else None,
        "decode_after_first_seconds": float(timing.group(3)) if timing else None,
        "cache_hit_rate_percent": float(cache.group(1)) if cache else None,
        "logical_cache_hits": int(cache.group(2)) if cache else None,
        "logical_cache_misses": int(cache.group(3)) if cache else None,
        "peak_rss_gb": float(rss.group(1)) if rss else None,
    }


def _process_sample(process: psutil.Process, *, role: str, identity: str) -> dict[str, Any] | None:
    try:
        full = process.memory_full_info()
        basic = process.memory_info()
        io = process.io_counters()
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "timestamp_ns": time.time_ns(),
            "process_role": role,
            "process_identity": identity,
            "pid": process.pid,
            "working_set_bytes": int(full.rss),
            "private_bytes": int(getattr(full, "private", getattr(full, "uss", 0))),
            "commit_size_bytes": int(getattr(full, "pagefile", full.vms)),
            "peak_working_set_bytes": int(getattr(full, "peak_wset", full.rss)),
            "page_fault_count": int(getattr(basic, "num_page_faults", 0)),
            "thread_count": process.num_threads(),
            "storage_read_bytes": int(io.read_bytes),
            "storage_write_bytes": int(io.write_bytes),
            "system_available_physical_bytes": int(virtual.available),
            "system_committed_bytes_proxy": int(virtual.total - virtual.available + swap.used),
            "system_commit_limit_bytes_proxy": int(virtual.total + swap.total),
            "pagefile_usage_bytes": int(swap.used),
        }
    except (OSError, psutil.Error):
        return None


def _write_samples(path: Path, samples: Iterable[dict[str, Any]], run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for sample in samples:
            handle.write(json.dumps({"run_id": run_id, **sample}, sort_keys=True) + "\n")


def _telemetry_metrics(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in events if row.get("event") == "expert_rpc_request_completed"]
    sent = [row for row in events if row.get("event") == "expert_rpc_bytes_sent"]
    received = [row for row in events if row.get("event") == "expert_rpc_bytes_received"]
    queues = [row for row in events if row.get("event") == "expert_rpc_queue_ns"]
    computes = [row for row in events if row.get("event") == "expert_rpc_compute_ns"]
    transports = [row for row in events if row.get("event") == "expert_rpc_transport_ns"]
    selected = {
        (int(row.get("layer_id", -1)), int(expert))
        for row in completed
        for expert in row.get("expert_ids", [])
    }
    return {
        "rpc_message_count": len(completed),
        "rpc_request_bytes": sum(int(row.get("byte_count", 0)) for row in sent),
        "rpc_response_bytes": sum(int(row.get("byte_count", 0)) for row in received),
        "rpc_raw_payload_bytes": sum(int(row.get("byte_count", 0)) for row in sent + received),
        "rpc_queue_ns": sum(int(row.get("duration_ns", 0)) for row in queues),
        "rpc_compute_ns": sum(int(row.get("duration_ns", 0)) for row in computes),
        "rpc_transport_ns": sum(int(row.get("duration_ns", 0)) for row in transports),
        "remote_results_consumed": sum(
            row.get("remote_result_consumed") is True for row in completed
        ),
        "forbidden_local_expert_loads": sum(
            row.get("event") == "forbidden_local_expert_load" for row in events
        ),
        "expert_union_size": len(selected),
        "worker_ids": sorted({str(row.get("worker_id")) for row in completed}),
    }


def _counter_delta(after: dict[str, Any], before: dict[str, Any], name: str) -> int | None:
    if name not in after:
        return None
    # A newly started native worker does not emit a synthetic all-zero sample.
    # Treat a counter absent from the pre-request snapshot as its documented
    # initial value instead of discarding the real post-request measurement.
    after_value = int(after[name])
    before_value = int(before.get(name, 0))
    # Resumed evidence directories retain prior worker telemetry.  Native
    # counters restart at zero with the new process; a monotonicity break is a
    # process-session boundary, not a negative cache or I/O measurement.
    if after_value < before_value:
        return after_value
    return after_value - before_value


class NativeLevelASession:
    """Persistent native workers for one measured candidate configuration."""

    def __init__(
        self,
        *,
        candidate: RealCandidate,
        worker_executable: Path,
        engine: Path,
        model_path: Path,
        root: Path,
        model_fingerprint: str,
        quantization_fingerprint: str = "native-colibri-int8-v1",
        worker_threads: int = 3,
        memory_budget_bytes: int | None = None,
        idot: bool = True,
    ) -> None:
        if candidate.local:
            raise ValueError("a native worker session requires a distributed candidate")
        if candidate.network_profile not in NETWORK_PROFILES:
            raise ValueError(f"unknown network profile {candidate.network_profile}")
        if (
            candidate.network_profile != "loopback_unshaped"
            and candidate.data_plane != "relayed_tcp"
        ):
            raise ValueError("network shaping must act through the real relay payload path")
        if candidate.worker_memory_budget_bytes < 64 * 1024 * 1024:
            raise ValueError("native worker memory budget must be at least 64 MiB")
        self.candidate = candidate
        self.engine = engine.resolve()
        self.model_path = (candidate.coordinator_model or model_path).resolve()
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_fingerprint = model_fingerprint
        self.quantization_fingerprint = quantization_fingerprint
        self.worker_threads = worker_threads
        self.memory_budget_bytes = (
            candidate.worker_memory_budget_bytes
            if memory_budget_bytes is None
            else memory_budget_bytes
        )
        self.idot = idot
        self.manager = NativeColibriExpertWorkerManager(self.root / "workers", worker_executable)
        self.relays = ExpertRelayManager(self.root / "relays")
        self.workers: list[NativeColibriExpertWorker] = []
        self.plan_path: Path | None = None

    def start(self) -> NativeLevelASession:
        if self.workers:
            return self
        for bank_path in self.candidate.bank_paths:
            manifest = json.loads((bank_path / "manifest.json").read_text(encoding="utf-8"))
            self.workers.append(
                self.manager.start(
                    worker_id=str(manifest["worker_id"]),
                    bank_path=bank_path,
                    model_id="colibri-olmoe",
                    model_revision=COLIBRI_MODEL_REVISION,
                    quantization_fingerprint=self.quantization_fingerprint,
                    model_fingerprint=self.model_fingerprint,
                    memory_budget_bytes=self.memory_budget_bytes,
                    thread_count=self.worker_threads,
                    idot=self.idot,
                )
            )
        plan_workers = self.workers
        if self.candidate.data_plane == "relayed_tcp":
            plan_workers = [
                replace(
                    worker,
                    endpoint=self.relays.start(
                        target_endpoint=worker.endpoint,
                        profile=NETWORK_PROFILES[self.candidate.network_profile],
                    ).endpoint,
                )
                for worker in self.workers
            ]
        self.plan_path = write_colibri_expert_plan(
            self.root / "plan.json",
            model_fingerprint=self.model_fingerprint,
            quantization_fingerprint=self.quantization_fingerprint,
            phase="decode",
            workers=plan_workers,
            local_experts=[],
        )
        return self

    def close(self) -> None:
        self.relays.close()
        self.manager.close()

    def __enter__(self) -> NativeLevelASession:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()

    def environment(self, run_root: Path, *, coordinator_threads: int) -> dict[str, str]:
        if not self.plan_path:
            raise RuntimeError("native session has not started")
        telemetry = run_root / "coordinator-telemetry.jsonl"
        route = run_root / "route.trace"
        for path in (telemetry, route):
            if path.is_file():
                path.unlink()
        environment = _base_environment(
            self.model_path, threads=coordinator_threads, idot=self.idot
        )
        environment.update(
            {
                "COLI_SWARM_EXPERT_MODE": self.candidate.mode,
                "COLI_SWARM_EXPERT_PLAN": str(self.plan_path),
                "COLI_SWARM_EXPERT_TIMEOUT_MS": "600000",
                "COLI_SWARM_EXPERT_FALLBACK": "fail",
                "COLI_SWARM_EXPERT_RESPONSE_MODE": self.candidate.response_mode,
                "COLI_SWARM_EXPERT_DATA_PLANE": self.candidate.data_plane,
                "COLI_SWARM_EXPERT_DETERMINISM": (
                    "exact" if self.candidate.exact_contract else "quality_bounded"
                ),
                "COLI_SWARM_EXPERT_TELEMETRY": str(telemetry),
                "COLI_SWARM_REQUEST_NAMESPACE": run_root.name,
                "ROUTE_TRACE": str(route),
            }
        )
        return environment

    def worker_counters(self) -> dict[str, dict[str, Any]]:
        return {
            worker.worker_id: _last_jsonl(
                worker.configuration_path.parent / "worker-telemetry.jsonl"
            )
            for worker in self.workers
        }


def _run_process(
    *,
    engine: Path,
    reference: Path,
    environment: dict[str, str],
    run_root: Path,
    run_id: str,
    workers: Sequence[NativeColibriExpertWorker],
    timeout_seconds: float,
) -> tuple[subprocess.CompletedProcess[str], int, list[dict[str, Any]]]:
    command = [str(engine), "16", "8", str(reference)]
    started = time.perf_counter_ns()
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    samples: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds
    coordinator = psutil.Process(process.pid)
    while process.poll() is None:
        sample = _process_sample(coordinator, role="coordinator", identity=run_id)
        if sample:
            samples.append(sample)
        for worker in workers:
            sample = _process_sample(
                psutil.Process(worker.process.pid),
                role="expert_worker",
                identity=worker.worker_id,
            )
            if sample:
                samples.append(sample)
        if time.monotonic() >= deadline:
            process.terminate()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
            if process.poll() is None:
                process.kill()
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(command, timeout_seconds, stdout, stderr)
        time.sleep(0.10)
    stdout, stderr = process.communicate()
    elapsed = time.perf_counter_ns() - started
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "stdout.log").write_text(stdout, encoding="utf-8")
    (run_root / "stderr.log").write_text(stderr, encoding="utf-8")
    _write_samples(run_root / "memory.ndjson", samples, run_id)
    return (
        subprocess.CompletedProcess(command, process.returncode, stdout, stderr),
        elapsed,
        samples,
    )


def measure_reference(
    *,
    run_id: str,
    workload: str,
    candidate: RealCandidate,
    engine: Path,
    model_path: Path,
    reference_path: Path,
    run_root: Path,
    prompt_id: str,
    repeat: int,
    coordinator_threads: int,
    timeout_seconds: float,
    model_fingerprint: str,
    session: NativeLevelASession | None = None,
) -> dict[str, Any]:
    result_path = run_root / "measurement.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("run_id") == run_id and existing.get("measurement_status") == "MEASURED":
            return existing
    run_root.mkdir(parents=True, exist_ok=True)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    prompt_count = len(reference["prompt_ids"])
    expected = [int(value) for value in reference["full_ids"][prompt_count:]]
    if candidate.local:
        environment = _base_environment(model_path.resolve(), threads=coordinator_threads)
        environment.update(
            {
                "COLI_SWARM_EXPERT_MODE": "local",
                "ROUTE_TRACE": str(run_root / "route.trace"),
                "COLI_SWARM_BRIDGE": "1",
                "COLI_SWARM_TELEMETRY": "trace",
                "COLI_SWARM_BRIDGE_PATH": str(run_root / "bridge.ndjson"),
                "COLI_REQUEST_ID": run_id,
                "COLI_MODEL_REVISION": COLIBRI_MODEL_REVISION,
            }
        )
        workers: Sequence[NativeColibriExpertWorker] = ()
        before: dict[str, dict[str, Any]] = {}
    else:
        if session is None:
            raise ValueError("distributed measurement requires a native session")
        environment = session.environment(run_root, coordinator_threads=coordinator_threads)
        workers = session.workers
        before = session.worker_counters()
    completed, wall_ns, samples = _run_process(
        engine=engine.resolve(),
        reference=reference_path.resolve(),
        environment=environment,
        run_root=run_root,
        run_id=run_id,
        workers=workers,
        timeout_seconds=timeout_seconds,
    )
    actual = _generated_token_ids(completed.stdout)
    parsed = parse_engine_metrics(completed.stdout)
    events = _read_jsonl(run_root / "coordinator-telemetry.jsonl")
    rpc = (
        _telemetry_metrics(events)
        if not candidate.local
        else {
            "rpc_message_count": 0,
            "rpc_request_bytes": 0,
            "rpc_response_bytes": 0,
            "rpc_raw_payload_bytes": 0,
            "rpc_queue_ns": 0,
            "rpc_compute_ns": 0,
            "rpc_transport_ns": 0,
            "remote_results_consumed": 0,
            "forbidden_local_expert_loads": 0,
            "expert_union_size": None,
            "worker_ids": [],
        }
    )
    cache_delta: dict[str, dict[str, int | None]] = {}
    accounting: list[dict[str, Any]] = []
    relay_metrics: list[dict[str, Any]] = []
    if session is not None:
        after = session.worker_counters()
        for worker_id, row in after.items():
            cache_delta[worker_id] = {
                name: _counter_delta(row, before.get(worker_id, {}), name)
                for name in (
                    "logical_cache_hits",
                    "logical_cache_misses",
                    "resident_cache_hits",
                    "nonresident_cache_hits",
                    "cache_hits_with_page_fault",
                    "cache_evictions",
                    "process_io_read_bytes",
                    "process_io_write_bytes",
                    "page_fault_count",
                )
            }
        accounting = [_worker_process_accounting(worker) for worker in session.workers]
        relay_metrics = session.relays.snapshots() if session.relays.relays else []
    exact = actual == expected
    numerical_contract_ok = exact if candidate.exact_contract else completed.returncode == 0
    measurement_status = (
        "MEASURED" if completed.returncode == 0 and len(actual) == len(expected) else "RUN_FAILED"
    )
    row = {
        "schema_version": "experiment-010-real-colibri-workload-v1",
        "run_id": run_id,
        "measurement_status": measurement_status,
        "evidence_category": "REAL_MODEL_MEASURED",
        "workload": workload,
        "configuration": candidate.name,
        "data_plane": "local" if candidate.local else candidate.data_plane,
        "network_profile": candidate.network_profile,
        "response_mode": "local" if candidate.local else candidate.response_mode,
        "shard_layout": candidate.shard_layout,
        "prompt_id": prompt_id,
        "repeat": repeat,
        "prompt_tokens": prompt_count,
        "generated_tokens": len(expected),
        "return_code": completed.returncode,
        "wall_elapsed_ns": wall_ns,
        "actual_token_ids": actual,
        "expected_token_ids": expected,
        "matching_tokens": sum(a == b for a, b in zip(actual, expected, strict=False)),
        "exact_token_identity": exact,
        "numerical_contract_ok": numerical_contract_ok,
        "coordinator_thread_count": coordinator_threads,
        "worker_count": len(workers),
        "model_path": str((candidate.coordinator_model or model_path).resolve()),
        "model_fingerprint": model_fingerprint,
        "colibri_binary_sha256": _sha256_file(engine.resolve()),
        "expert_worker_binary_sha256": (
            _sha256_file(session.manager.executable) if session is not None else None
        ),
        "reference_path": str(reference_path.resolve()),
        "memory_sample_count": len(samples),
        "memory_timeseries_path": str((run_root / "memory.ndjson").resolve()),
        "worker_counter_deltas": cache_delta,
        "worker_process_accounting": accounting,
        "relay_metrics": relay_metrics,
        **parsed,
        **rpc,
    }
    row["valid_performance_candidate"] = bool(
        completed.returncode == 0
        and exact
        and rpc.get("forbidden_local_expert_loads", 0) == 0
        and all(
            (metrics.get("nonresident_cache_hits") or 0)
            <= (metrics.get("resident_cache_hits") or 0)
            for metrics in cache_delta.values()
        )
    )
    _write_json(result_path, row)
    return row


def _flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            if isinstance(value, (dict, list))
            else value
        )
        for key, value in row.items()
    }


def write_measurement_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(_flatten_for_csv(row) for row in rows)


def percentile(values: Sequence[float], percentage: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    at = (len(ordered) - 1) * percentage
    lower = math.floor(at)
    upper = math.ceil(at)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (at - lower)


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["workload"]), str(row["configuration"]), str(row["network_profile"]))
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (workload, configuration, profile), members in sorted(groups.items()):
        throughput = [
            float(row["decode_tokens_per_second"])
            for row in members
            if row.get("decode_tokens_per_second") is not None
        ]
        ttft = [
            float(row["ttft_seconds"]) for row in members if row.get("ttft_seconds") is not None
        ]
        wall = [float(row["wall_elapsed_ns"]) / 1e9 for row in members]
        summaries.append(
            {
                "workload": workload,
                "configuration": configuration,
                "network_profile": profile,
                "measured_rows": len(members),
                "valid_rows": sum(bool(row.get("valid_performance_candidate")) for row in members),
                "exact_token_rows": sum(bool(row.get("exact_token_identity")) for row in members),
                "median_decode_tokens_per_second": statistics.median(throughput)
                if throughput
                else None,
                "median_ttft_seconds": statistics.median(ttft) if ttft else None,
                "p50_wall_seconds": percentile(wall, 0.50),
                "p95_wall_seconds": percentile(wall, 0.95),
                "rpc_raw_payload_bytes": sum(
                    int(row.get("rpc_raw_payload_bytes", 0)) for row in members
                ),
                "rpc_message_count": sum(int(row.get("rpc_message_count", 0)) for row in members),
            }
        )
    return summaries


def prepare_short_references(
    *,
    engine: Path,
    model_path: Path,
    output_directory: Path,
    prompt_count: int = 20,
    generated_tokens: int = 128,
    thread_count: int = 4,
) -> dict[str, Any]:
    return build_local_reference_suite(
        engine=engine,
        model_path=model_path,
        output_directory=output_directory,
        generated_tokens=generated_tokens,
        thread_count=thread_count,
        timeout_seconds=1800,
        prompt_limit=prompt_count,
        required_prompt_count=prompt_count,
    )


_PREFILL_SEEDS = (
    "Distributed systems need explicit ownership, deterministic reduction, and observable failure semantics.",
    "A careful performance study separates prefill, decode, queueing, transport, storage, and verification costs.",
    "Reliable scientific evidence preserves raw measurements, negative results, model identity, and executable provenance.",
    "Memory residency differs from logical cache membership when virtual memory can page inactive expert tensors.",
    "A useful planner rejects harmful workers and can prefer local execution when remote capacity does not improve latency.",
)


def _exact_length_prompt_ids(tokenizer: Tokenizer, *, seed_index: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("prefill context length must be positive")
    values: list[int] = []
    block = 0
    while len(values) < count:
        text = (
            f"{_PREFILL_SEEDS[seed_index % len(_PREFILL_SEEDS)]} "
            f"Evidence block {block}; retain ordering and check every boundary.\n"
        )
        values.extend(tokenizer.encode(text).ids)
        block += 1
    return values[:count]


def prefill_context_supported(*, context_length: int, advertised_context_limit: int) -> bool:
    """Return whether a requested official prefill context is executable.

    Eight thousand tokens is the explicitly tested downstream-workspace target.
    Beyond that point the exact model container must advertise the requested
    context; a larger temporary attention allocation is not a model capability.
    """

    if context_length < 1 or advertised_context_limit < 1:
        raise ValueError("context lengths must be positive")
    return context_length <= 8192 or context_length <= advertised_context_limit


def prepare_prefill_references(
    *,
    engine: Path,
    model_path: Path,
    output_directory: Path,
    context_lengths: Sequence[int],
    model_fingerprint: str,
    prompt_count: int = 5,
    generated_tokens: int = 64,
    thread_count: int = 16,
    timeout_seconds: float = 14_400,
) -> dict[str, Any]:
    """Capture exact long-context local Colibri oracles, checkpointing each prompt."""

    if prompt_count != 5:
        raise ValueError("the official long-prefill matrix requires exactly five prompts")
    if generated_tokens < 64:
        raise ValueError("the official long-prefill matrix requires at least 64 generated tokens")
    engine = engine.resolve()
    model = model_path.resolve()
    output = output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = model / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    config_path = model / "config.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    advertised_context_limit = int(model_config.get("max_position_embeddings", 0))
    if advertised_context_limit < 1:
        raise ValueError("model config does not declare max_position_embeddings")
    rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for context_length in context_lengths:
        # The correction's downstream attention-workspace patch is explicitly
        # validated for the mandatory 8K workload.  Longer contexts are only
        # eligible when the exact pinned model configuration advertises them;
        # allocating a larger workspace does not alter the model's RoPE/context
        # contract and must not be misreported as 32K support.
        if not prefill_context_supported(
            context_length=context_length,
            advertised_context_limit=advertised_context_limit,
        ):
            unsupported.append(
                {
                    "context_length": context_length,
                    "measurement_status": "UNSUPPORTED_BY_MODEL",
                    "measured": False,
                    "metrics": None,
                    "reason": (
                        f"requested context {context_length} exceeds the pinned model's "
                        f"advertised max_position_embeddings={advertised_context_limit}"
                    ),
                    "config_path": str(config_path.resolve()),
                    "config_sha256": _sha256_file(config_path),
                }
            )
            continue
        for seed_index in range(prompt_count):
            prompt_id = f"prefill-{context_length}-{seed_index + 1:02d}"
            run_root = output / f"context-{context_length}" / prompt_id
            reference_path = run_root / "reference.json"
            result_path = run_root / "local-result.json"
            if reference_path.is_file() and result_path.is_file():
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    existing.get("measurement_status") == "MEASURED"
                    and existing.get("exact_token_identity") is True
                    and existing.get("prompt_tokens") == context_length
                    and existing.get("generated_tokens") == generated_tokens
                ):
                    rows.append(existing)
                    continue
            run_root.mkdir(parents=True, exist_ok=True)
            prompt_ids = _exact_length_prompt_ids(
                tokenizer, seed_index=seed_index, count=context_length
            )
            capture_path = run_root / "capture-reference.json"
            _write_json(
                capture_path,
                {
                    "prompt_id": prompt_id,
                    "workload_group": "long_prefill",
                    "prompt_ids": prompt_ids,
                    "full_ids": [*prompt_ids, *([0] * generated_tokens)],
                },
            )
            environment = _base_environment(model, threads=thread_count)
            environment.update(
                {
                    "COLI_SWARM_EXPERT_MODE": "local",
                    "COLI_REQUEST_ID": prompt_id,
                    "COLI_MODEL_REVISION": COLIBRI_MODEL_REVISION,
                    "ROUTE_TRACE": str(run_root / "route.trace"),
                }
            )
            completed, wall_ns, samples = _run_process(
                engine=engine,
                reference=capture_path,
                environment=environment,
                run_root=run_root,
                run_id=prompt_id,
                workers=(),
                timeout_seconds=timeout_seconds,
            )
            token_ids = _generated_token_ids(completed.stdout)
            exact = completed.returncode == 0 and len(token_ids) == generated_tokens
            result = {
                "schema_version": "experiment-010-real-colibri-prefill-reference-v1",
                "measurement_status": "MEASURED" if exact else "RUN_FAILED",
                "evidence_category": "REAL_MODEL_MEASURED",
                "prompt_id": prompt_id,
                "prompt_tokens": context_length,
                "generated_tokens": generated_tokens,
                "return_code": completed.returncode,
                "exact_token_identity": exact,
                "actual_token_ids": token_ids,
                "wall_elapsed_ns": wall_ns,
                "memory_sample_count": len(samples),
                "memory_timeseries_path": str((run_root / "memory.ndjson").resolve()),
                "model_fingerprint": model_fingerprint,
                "colibri_binary_sha256": _sha256_file(engine),
                "tokenizer_sha256": _sha256_file(tokenizer_path),
                **parse_engine_metrics(completed.stdout),
            }
            _write_json(result_path, result)
            if not exact:
                raise RuntimeError(f"local long-prefill oracle capture failed for {prompt_id}")
            _write_json(
                reference_path,
                {
                    "prompt_id": prompt_id,
                    "workload_group": "long_prefill",
                    "prompt_ids": prompt_ids,
                    "full_ids": [*prompt_ids, *token_ids],
                },
            )
            rows.append(result)
            write_measurement_csv(output / "local-prefill-results.csv", rows)
    supported_context_lengths = [
        length
        for length in context_lengths
        if not any(row["context_length"] == length for row in unsupported)
    ]
    complete = len(rows) == len(supported_context_lengths) * prompt_count and all(
        row["measurement_status"] == "MEASURED" and row["exact_token_identity"] for row in rows
    )
    summary = {
        "schema_version": "experiment-010-real-colibri-prefill-reference-suite-v1",
        "requested_context_lengths": list(context_lengths),
        "supported_context_lengths": supported_context_lengths,
        "advertised_context_limit": advertised_context_limit,
        "prompt_count_per_context": prompt_count,
        "generated_tokens_per_prompt": generated_tokens,
        "measured_rows": len(rows),
        "unsupported_contexts": unsupported,
        "complete": complete,
        "rows": rows,
    }
    _write_json(output / "completion.json", summary)
    return summary


def run_candidate_repeats(
    *,
    candidate: RealCandidate,
    references: Sequence[Path],
    repeats: int,
    engine: Path,
    worker_executable: Path,
    model_path: Path,
    output_directory: Path,
    model_fingerprint: str,
    workload: str = "short_decode_performance",
    coordinator_threads: int = 4,
    worker_threads: int = 3,
    timeout_seconds: float = 1800,
) -> list[dict[str, Any]]:
    output = output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    session_context: NativeLevelASession | None = None
    try:
        if not candidate.local:
            session_context = NativeLevelASession(
                candidate=candidate,
                worker_executable=worker_executable,
                engine=engine,
                model_path=model_path,
                root=output / "session",
                model_fingerprint=model_fingerprint,
                worker_threads=worker_threads,
            ).start()
        for repeat_index in range(1, repeats + 1):
            for reference_path in references:
                reference = json.loads(reference_path.read_text(encoding="utf-8"))
                prompt_id = str(reference.get("prompt_id", reference_path.parent.name))
                run_id = f"{workload}-{candidate.name}-{prompt_id}-r{repeat_index}"
                safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id)
                rows.append(
                    measure_reference(
                        run_id=run_id,
                        workload=workload,
                        candidate=candidate,
                        engine=engine,
                        model_path=model_path,
                        reference_path=reference_path,
                        run_root=output / "runs" / safe,
                        prompt_id=prompt_id,
                        repeat=repeat_index,
                        coordinator_threads=coordinator_threads,
                        timeout_seconds=timeout_seconds,
                        model_fingerprint=model_fingerprint,
                        session=session_context,
                    )
                )
                write_measurement_csv(output / "measurements.csv", rows)
                write_measurement_csv(output / "summary.csv", summarize_rows(rows))
    finally:
        if session_context is not None:
            session_context.close()
    successful_rows = sum(
        row.get("measurement_status") == "MEASURED"
        and (not candidate.exact_contract or row.get("exact_token_identity") is True)
        for row in rows
    )
    _write_json(
        output / "completion.json",
        {
            "workload": workload,
            "configuration": candidate.name,
            "required_rows": len(references) * repeats,
            "measured_rows": len(rows),
            "successful_rows": successful_rows,
            "complete": successful_rows == len(references) * repeats,
            "candidate": {
                **asdict(candidate),
                "bank_paths": [str(path) for path in candidate.bank_paths],
                "coordinator_model": (
                    str(candidate.coordinator_model) if candidate.coordinator_model else None
                ),
            },
        },
    )
    return rows


def run_network_profile_matrix(
    *,
    candidate: RealCandidate,
    reference_paths: Sequence[Path],
    profiles: Sequence[str],
    repeats: int,
    engine: Path,
    worker_executable: Path,
    model_path: Path,
    output_directory: Path,
    model_fingerprint: str,
    coordinator_threads: int = 4,
    worker_threads: int = 3,
    timeout_seconds: float = 7200,
) -> list[dict[str, Any]]:
    if candidate.local:
        raise ValueError("network profile matrix requires native expert workers")
    if not reference_paths:
        raise ValueError("network profile matrix requires real Colibri references")
    output = output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for profile in profiles:
        if profile not in NETWORK_PROFILES:
            raise ValueError(f"unknown network profile {profile}")
        profiled = replace(
            candidate,
            name=f"{candidate.name}_{profile}",
            data_plane="relayed_tcp",
            network_profile=profile,
        )
        rows = run_candidate_repeats(
            candidate=profiled,
            references=reference_paths,
            repeats=repeats,
            engine=engine,
            worker_executable=worker_executable,
            model_path=model_path,
            output_directory=output / profile,
            model_fingerprint=model_fingerprint,
            workload="network_profile_real_token_path",
            coordinator_threads=coordinator_threads,
            worker_threads=worker_threads,
            timeout_seconds=timeout_seconds,
        )
        all_rows.extend(rows)
        write_measurement_csv(output / "network_profile_results.csv", all_rows)
    required = len(profiles) * len(reference_paths) * repeats
    complete = len(all_rows) == required and all(
        row["measurement_status"] == "MEASURED" and row["exact_token_identity"] for row in all_rows
    )
    _write_json(
        output / "completion.json",
        {
            "workload": "network_profile_real_token_path",
            "profiles": list(profiles),
            "required_rows": required,
            "measured_rows": len(all_rows),
            "complete": complete,
        },
    )
    return all_rows


def measure_concurrent_group(
    *,
    run_id: str,
    workload: str,
    candidate: RealCandidate,
    engine: Path,
    model_path: Path,
    references: Sequence[Path],
    roles: Sequence[str],
    run_root: Path,
    repeat: int,
    coordinator_threads: int,
    timeout_seconds: float,
    model_fingerprint: str,
    session: NativeLevelASession | None = None,
) -> dict[str, Any]:
    """Launch simultaneous Colibri coordinators against one shared worker set."""

    if len(references) != len(roles) or not references:
        raise ValueError("concurrent references and roles must be non-empty and aligned")
    result_path = run_root / "group-measurement.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("run_id") == run_id and existing.get("measurement_status") == "MEASURED":
            return existing
    run_root.mkdir(parents=True, exist_ok=True)
    processes: list[dict[str, Any]] = []
    before = session.worker_counters() if session else {}
    group_started = time.perf_counter_ns()
    for index, (reference_path, role) in enumerate(zip(references, roles, strict=True)):
        request_root = run_root / f"request-{index:02d}-{role}"
        request_root.mkdir(parents=True, exist_ok=True)
        if candidate.local:
            environment = _base_environment(model_path.resolve(), threads=coordinator_threads)
            environment.update(
                {
                    "COLI_SWARM_EXPERT_MODE": "local",
                    "ROUTE_TRACE": str(request_root / "route.trace"),
                    "COLI_REQUEST_ID": f"{run_id}-{index}",
                }
            )
        else:
            if session is None:
                raise ValueError("distributed concurrent group requires a session")
            environment = session.environment(request_root, coordinator_threads=coordinator_threads)
        command = [str(engine.resolve()), "16", "8", str(reference_path.resolve())]
        started = time.perf_counter_ns()
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(
            {
                "process": process,
                "psutil": psutil.Process(process.pid),
                "started_ns": started,
                "ended_ns": None,
                "reference": reference_path,
                "role": role,
                "root": request_root,
                "index": index,
            }
        )
    samples: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds
    while any(item["process"].poll() is None for item in processes):
        for item in processes:
            process = item["process"]
            if process.poll() is None:
                sample = _process_sample(
                    item["psutil"],
                    role="coordinator",
                    identity=f"{run_id}-{item['index']}",
                )
                if sample:
                    samples.append(sample)
            elif item["ended_ns"] is None:
                item["ended_ns"] = time.perf_counter_ns()
        if session:
            for worker in session.workers:
                sample = _process_sample(
                    psutil.Process(worker.process.pid),
                    role="expert_worker",
                    identity=worker.worker_id,
                )
                if sample:
                    samples.append(sample)
        if time.monotonic() >= deadline:
            for item in processes:
                if item["process"].poll() is None:
                    item["process"].terminate()
            time.sleep(1)
            for item in processes:
                if item["process"].poll() is None:
                    item["process"].kill()
            raise TimeoutError(f"concurrent Colibri group {run_id} timed out")
        time.sleep(0.10)
    group_ended = time.perf_counter_ns()
    _write_samples(run_root / "memory.ndjson", samples, run_id)
    request_rows: list[dict[str, Any]] = []
    expert_sets: list[set[tuple[int, int]]] = []
    for item in processes:
        process = item["process"]
        stdout, stderr = process.communicate()
        item["ended_ns"] = item["ended_ns"] or time.perf_counter_ns()
        request_root: Path = item["root"]
        (request_root / "stdout.log").write_text(stdout, encoding="utf-8")
        (request_root / "stderr.log").write_text(stderr, encoding="utf-8")
        reference = json.loads(item["reference"].read_text(encoding="utf-8"))
        prompt_count = len(reference["prompt_ids"])
        expected = [int(value) for value in reference["full_ids"][prompt_count:]]
        actual = _generated_token_ids(stdout)
        events = _read_jsonl(request_root / "coordinator-telemetry.jsonl")
        rpc = (
            _telemetry_metrics(events)
            if not candidate.local
            else {
                "rpc_message_count": 0,
                "rpc_request_bytes": 0,
                "rpc_response_bytes": 0,
                "rpc_raw_payload_bytes": 0,
                "rpc_queue_ns": 0,
                "rpc_compute_ns": 0,
                "rpc_transport_ns": 0,
                "remote_results_consumed": 0,
                "forbidden_local_expert_loads": 0,
                "expert_union_size": None,
                "worker_ids": [],
            }
        )
        selected = {
            (int(event.get("layer_id", -1)), int(expert))
            for event in events
            if event.get("event") == "expert_rpc_request_completed"
            for expert in event.get("expert_ids", [])
        }
        expert_sets.append(selected)
        request_rows.append(
            {
                "request_index": item["index"],
                "service_role": item["role"],
                "prompt_id": str(reference.get("prompt_id", item["reference"].parent.name)),
                "prompt_tokens": prompt_count,
                "generated_tokens": len(expected),
                "return_code": process.returncode,
                "wall_elapsed_ns": int(item["ended_ns"] - item["started_ns"]),
                "exact_token_identity": actual == expected,
                "matching_tokens": sum(a == b for a, b in zip(actual, expected, strict=False)),
                "expected_token_ids": expected,
                "actual_token_ids": actual,
                **parse_engine_metrics(stdout),
                **rpc,
            }
        )
        _write_json(request_root / "request-measurement.json", request_rows[-1])
    overlaps: list[float] = []
    for left in range(len(expert_sets)):
        for right in range(left + 1, len(expert_sets)):
            union = expert_sets[left] | expert_sets[right]
            if union:
                overlaps.append(len(expert_sets[left] & expert_sets[right]) / len(union))
    after = session.worker_counters() if session else {}
    worker_deltas = {
        worker_id: {
            name: _counter_delta(row, before.get(worker_id, {}), name)
            for name in (
                "logical_cache_hits",
                "logical_cache_misses",
                "resident_cache_hits",
                "nonresident_cache_hits",
                "cache_hits_with_page_fault",
                "cache_evictions",
                "process_io_read_bytes",
                "page_fault_count",
            )
        }
        for worker_id, row in after.items()
    }
    exact = all(row["exact_token_identity"] for row in request_rows)
    successful = all(row["return_code"] == 0 for row in request_rows)
    group_seconds = (group_ended - group_started) / 1e9
    verified_tokens = sum(
        row["generated_tokens"] if row["exact_token_identity"] else 0 for row in request_rows
    )
    latency = [row["wall_elapsed_ns"] / 1e9 for row in request_rows]
    interactive = [
        row["wall_elapsed_ns"] / 1e9 for row in request_rows if row["service_role"] == "interactive"
    ]
    background_tokens = sum(
        row["generated_tokens"]
        for row in request_rows
        if row["service_role"] == "background" and row["exact_token_identity"]
    )
    worker_count = len(session.workers) if session else 0
    rpc_compute_ns = sum(row["rpc_compute_ns"] for row in request_rows)
    rpc_queue_ns = sum(row["rpc_queue_ns"] for row in request_rows)
    worker_time_capacity_ns = (group_ended - group_started) * worker_count
    worker_compute_utilization = (
        min(1.0, rpc_compute_ns / worker_time_capacity_ns) if worker_time_capacity_ns else None
    )
    worker_queue_fraction = (
        rpc_queue_ns / (rpc_queue_ns + rpc_compute_ns) if rpc_queue_ns + rpc_compute_ns else None
    )
    result = {
        "schema_version": "experiment-010-real-colibri-concurrent-workload-v1",
        "run_id": run_id,
        "measurement_status": "MEASURED" if successful else "RUN_FAILED",
        "evidence_category": "REAL_MODEL_MEASURED",
        "workload": workload,
        "configuration": candidate.name,
        "model_fingerprint": model_fingerprint,
        "colibri_binary_sha256": _sha256_file(engine.resolve()),
        "expert_worker_binary_sha256": (
            _sha256_file(session.manager.executable) if session is not None else None
        ),
        "data_plane": "local" if candidate.local else candidate.data_plane,
        "network_profile": candidate.network_profile,
        "response_mode": "local" if candidate.local else candidate.response_mode,
        "shard_layout": candidate.shard_layout,
        "repeat": repeat,
        "concurrency": len(request_rows),
        "group_elapsed_ns": group_ended - group_started,
        "aggregate_verified_tokens_per_second": verified_tokens / group_seconds,
        "per_request_verified_tokens_per_second": [
            row["generated_tokens"] / (row["wall_elapsed_ns"] / 1e9)
            if row["exact_token_identity"]
            else 0.0
            for row in request_rows
        ],
        "p50_latency_seconds": percentile(latency, 0.50),
        "p95_latency_seconds": percentile(latency, 0.95),
        "interactive_p95_seconds": percentile(interactive, 0.95),
        "background_verified_tokens_per_second": background_tokens / group_seconds,
        "exact_group_token_identity": exact,
        "verified_tokens": verified_tokens,
        "expected_tokens": sum(row["generated_tokens"] for row in request_rows),
        "worker_count": worker_count,
        "worker_compute_utilization_fraction": worker_compute_utilization,
        "worker_queue_fraction": worker_queue_fraction,
        "worker_saturated": (
            worker_compute_utilization is not None and worker_compute_utilization >= 0.90
        ),
        "starvation_detected": background_tokens == 0
        and any(row["service_role"] == "background" for row in request_rows),
        "rpc_message_count": sum(row["rpc_message_count"] for row in request_rows),
        "rpc_raw_payload_bytes": sum(row["rpc_raw_payload_bytes"] for row in request_rows),
        "rpc_queue_ns": rpc_queue_ns,
        "rpc_compute_ns": rpc_compute_ns,
        "mean_expert_overlap_jaccard": statistics.mean(overlaps) if overlaps else None,
        "worker_counter_deltas": worker_deltas,
        "worker_process_accounting": (
            [_worker_process_accounting(worker) for worker in session.workers] if session else []
        ),
        "relay_metrics": session.relays.snapshots() if session and session.relays.relays else [],
        "requests": request_rows,
        "memory_timeseries_path": str((run_root / "memory.ndjson").resolve()),
    }
    _write_json(result_path, result)
    return result


def run_concurrent_matrix(
    *,
    candidate: RealCandidate,
    reference_paths: Sequence[Path],
    concurrency_levels: Sequence[int],
    repeats: int,
    engine: Path,
    worker_executable: Path,
    model_path: Path,
    output_directory: Path,
    model_fingerprint: str,
    coordinator_threads: int = 1,
    worker_threads: int = 4,
    timeout_seconds: float = 3600,
) -> list[dict[str, Any]]:
    if max(concurrency_levels, default=0) > len(reference_paths):
        raise ValueError("not enough references for requested concurrency")
    output = output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    session: NativeLevelASession | None = None
    rows: list[dict[str, Any]] = []
    try:
        if not candidate.local:
            session = NativeLevelASession(
                candidate=candidate,
                worker_executable=worker_executable,
                engine=engine,
                model_path=model_path,
                root=output / "session",
                model_fingerprint=model_fingerprint,
                worker_threads=worker_threads,
            ).start()
        for repeat_index in range(1, repeats + 1):
            for concurrency in concurrency_levels:
                run_id = f"concurrent-{candidate.name}-c{concurrency}-r{repeat_index}"
                rows.append(
                    measure_concurrent_group(
                        run_id=run_id,
                        workload="concurrent_decode",
                        candidate=candidate,
                        engine=engine,
                        model_path=model_path,
                        references=reference_paths[:concurrency],
                        roles=["interactive"] * concurrency,
                        run_root=output / "runs" / run_id,
                        repeat=repeat_index,
                        coordinator_threads=coordinator_threads,
                        timeout_seconds=timeout_seconds,
                        model_fingerprint=model_fingerprint,
                        session=session,
                    )
                )
                write_measurement_csv(output / "concurrent_decode.csv", rows)
    finally:
        if session:
            session.close()
    complete = len(rows) == len(concurrency_levels) * repeats and all(
        row["measurement_status"] == "MEASURED" and row["exact_group_token_identity"]
        for row in rows
    )
    _write_json(
        output / "completion.json",
        {
            "workload": "concurrent_decode",
            "configuration": candidate.name,
            "required_groups": len(concurrency_levels) * repeats,
            "measured_groups": len(rows),
            "complete": complete,
        },
    )
    return rows


def run_mixed_service_matrix(
    *,
    candidate: RealCandidate,
    interactive_reference: Path,
    background_references: Sequence[Path],
    background_counts: Sequence[int],
    repeats: int,
    engine: Path,
    worker_executable: Path,
    model_path: Path,
    output_directory: Path,
    model_fingerprint: str,
    coordinator_threads: int = 1,
    worker_threads: int = 4,
) -> list[dict[str, Any]]:
    if max(background_counts, default=0) > len(background_references):
        raise ValueError("not enough background references")
    output = output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    session: NativeLevelASession | None = None
    rows: list[dict[str, Any]] = []
    try:
        if not candidate.local:
            session = NativeLevelASession(
                candidate=candidate,
                worker_executable=worker_executable,
                engine=engine,
                model_path=model_path,
                root=output / "session",
                model_fingerprint=model_fingerprint,
                worker_threads=worker_threads,
            ).start()
        for repeat_index in range(1, repeats + 1):
            for background_count in background_counts:
                run_id = f"mixed-{candidate.name}-bg{background_count}-r{repeat_index}"
                rows.append(
                    measure_concurrent_group(
                        run_id=run_id,
                        workload="mixed_service",
                        candidate=candidate,
                        engine=engine,
                        model_path=model_path,
                        references=[
                            interactive_reference,
                            *background_references[:background_count],
                        ],
                        roles=["interactive", *(["background"] * background_count)],
                        run_root=output / "runs" / run_id,
                        repeat=repeat_index,
                        coordinator_threads=coordinator_threads,
                        timeout_seconds=3600,
                        model_fingerprint=model_fingerprint,
                        session=session,
                    )
                )
                write_measurement_csv(output / "mixed_service.csv", rows)
    finally:
        if session:
            session.close()
    complete = len(rows) == len(background_counts) * repeats and all(
        row["measurement_status"] == "MEASURED" and row["exact_group_token_identity"]
        for row in rows
    )
    _write_json(
        output / "completion.json",
        {
            "workload": "mixed_service",
            "configuration": candidate.name,
            "required_groups": len(background_counts) * repeats,
            "measured_groups": len(rows),
            "complete": complete,
        },
    )
    return rows


def _candidate_from_arguments(arguments: argparse.Namespace) -> RealCandidate:
    banks = tuple(path.resolve() for path in arguments.bank)
    return RealCandidate(
        name=arguments.configuration,
        bank_paths=banks,
        mode=arguments.mode,
        response_mode=arguments.response_mode,
        data_plane=arguments.data_plane,
        network_profile=arguments.network_profile,
        shard_layout=arguments.shard_layout,
        exact_contract=not arguments.quality_bounded,
        coordinator_model=arguments.coordinator_model.resolve()
        if arguments.coordinator_model
        else None,
        worker_memory_budget_bytes=arguments.worker_memory_budget_mb * 1024 * 1024,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-short-references", action="store_true")
    parser.add_argument("--prepare-prefill-references", action="store_true")
    parser.add_argument("--run-candidate", action="store_true")
    parser.add_argument("--run-concurrent", action="store_true")
    parser.add_argument("--run-mixed", action="store_true")
    parser.add_argument("--run-network-matrix", action="store_true")
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--prompt-count", type=int, default=20)
    parser.add_argument("--generated-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--configuration", default="local")
    parser.add_argument("--bank", type=Path, action="append", default=[])
    parser.add_argument("--mode", choices=("rpc", "hybrid", "planner"), default="rpc")
    parser.add_argument(
        "--response-mode",
        choices=("per_expert_exact", "per_worker_fast"),
        default="per_expert_exact",
    )
    parser.add_argument(
        "--data-plane",
        choices=("direct_tcp", "relayed_tcp", "shared_memory"),
        default="direct_tcp",
    )
    parser.add_argument(
        "--network-profile", choices=tuple(NETWORK_PROFILES), default="loopback_unshaped"
    )
    parser.add_argument("--shard-layout", default="whole")
    parser.add_argument("--quality-bounded", action="store_true")
    parser.add_argument("--coordinator-model", type=Path)
    parser.add_argument("--model-fingerprint")
    parser.add_argument("--coordinator-threads", type=int, default=4)
    parser.add_argument("--worker-threads", type=int, default=3)
    parser.add_argument("--worker-memory-budget-mb", type=int, default=512)
    parser.add_argument("--concurrency-level", type=int, action="append", default=[])
    parser.add_argument("--background-count", type=int, action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--context-length", type=int, action="append", default=[])
    parser.add_argument("--profile", choices=tuple(NETWORK_PROFILES), action="append", default=[])
    parser.add_argument("--workload", default="short_decode_performance")
    arguments = parser.parse_args()
    if arguments.prepare_short_references:
        summary = prepare_short_references(
            engine=arguments.engine,
            model_path=arguments.model,
            output_directory=arguments.output,
            prompt_count=arguments.prompt_count,
            generated_tokens=arguments.generated_tokens,
            thread_count=arguments.coordinator_threads,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    if arguments.prepare_prefill_references:
        if not arguments.model_fingerprint:
            parser.error("--prepare-prefill-references requires --model-fingerprint")
        summary = prepare_prefill_references(
            engine=arguments.engine,
            model_path=arguments.model,
            output_directory=arguments.output,
            context_lengths=arguments.context_length or [8192],
            model_fingerprint=arguments.model_fingerprint,
            prompt_count=arguments.prompt_count,
            generated_tokens=arguments.generated_tokens,
            thread_count=arguments.coordinator_threads,
            timeout_seconds=arguments.timeout_seconds,
        )
        print(
            json.dumps(
                {"measured_rows": summary["measured_rows"], "complete": summary["complete"]},
                sort_keys=True,
            )
        )
        return 0 if summary["complete"] else 2
    if arguments.run_network_matrix:
        if not arguments.reference_root or not arguments.worker or not arguments.model_fingerprint:
            parser.error("network matrix requires reference-root, worker, and model-fingerprint")
        references = sorted(arguments.reference_root.glob("*/reference.json"))[
            : arguments.prompt_count
        ]
        if len(references) != arguments.prompt_count:
            parser.error("reference root does not contain the requested prompt count")
        rows = run_network_profile_matrix(
            candidate=_candidate_from_arguments(arguments),
            reference_paths=references,
            profiles=arguments.profile or list(NETWORK_PROFILES),
            repeats=arguments.repeats,
            engine=arguments.engine,
            worker_executable=arguments.worker,
            model_path=arguments.model,
            output_directory=arguments.output,
            model_fingerprint=arguments.model_fingerprint,
            coordinator_threads=arguments.coordinator_threads,
            worker_threads=arguments.worker_threads,
            timeout_seconds=arguments.timeout_seconds,
        )
        complete = all(
            row.get("measurement_status") == "MEASURED" and row.get("exact_token_identity") is True
            for row in rows
        )
        print(json.dumps({"measured_rows": len(rows), "complete": complete}, sort_keys=True))
        return 0 if complete else 2
    if arguments.run_candidate:
        if not arguments.reference_root or not arguments.worker or not arguments.model_fingerprint:
            parser.error("--run-candidate requires reference-root, worker, and model-fingerprint")
        references = sorted(arguments.reference_root.glob("*/reference.json"))[
            : arguments.prompt_count
        ]
        if len(references) != arguments.prompt_count:
            parser.error("reference root does not contain the requested prompt count")
        rows = run_candidate_repeats(
            candidate=_candidate_from_arguments(arguments),
            references=references,
            repeats=arguments.repeats,
            engine=arguments.engine,
            worker_executable=arguments.worker,
            model_path=arguments.model,
            output_directory=arguments.output,
            model_fingerprint=arguments.model_fingerprint,
            workload=arguments.workload,
            coordinator_threads=arguments.coordinator_threads,
            worker_threads=arguments.worker_threads,
            timeout_seconds=arguments.timeout_seconds,
        )
        complete = all(
            row.get("measurement_status") == "MEASURED"
            and (
                not _candidate_from_arguments(arguments).exact_contract
                or row.get("exact_token_identity") is True
            )
            for row in rows
        )
        print(json.dumps({"measured_rows": len(rows), "complete": complete}, sort_keys=True))
        return 0 if complete else 2
    if arguments.run_concurrent or arguments.run_mixed:
        if not arguments.reference_root or not arguments.worker or not arguments.model_fingerprint:
            parser.error(
                "concurrent workloads require reference-root, worker, and model-fingerprint"
            )
        references = sorted(arguments.reference_root.glob("*/reference.json"))
        candidate = _candidate_from_arguments(arguments)
        if arguments.run_concurrent:
            concurrency_levels = arguments.concurrency_level or [2, 4, 8]
            rows = run_concurrent_matrix(
                candidate=candidate,
                reference_paths=references,
                concurrency_levels=concurrency_levels,
                repeats=arguments.repeats,
                engine=arguments.engine,
                worker_executable=arguments.worker,
                model_path=arguments.model,
                output_directory=arguments.output,
                model_fingerprint=arguments.model_fingerprint,
                coordinator_threads=arguments.coordinator_threads,
                worker_threads=arguments.worker_threads,
                timeout_seconds=arguments.timeout_seconds,
            )
        else:
            background_counts = arguments.background_count or [1, 4]
            required = 1 + max(background_counts)
            if len(references) < required:
                parser.error("reference root does not contain enough mixed-service prompts")
            rows = run_mixed_service_matrix(
                candidate=candidate,
                interactive_reference=references[0],
                background_references=references[1:required],
                background_counts=background_counts,
                repeats=arguments.repeats,
                engine=arguments.engine,
                worker_executable=arguments.worker,
                model_path=arguments.model,
                output_directory=arguments.output,
                model_fingerprint=arguments.model_fingerprint,
                coordinator_threads=arguments.coordinator_threads,
                worker_threads=arguments.worker_threads,
            )
        complete = all(
            row.get("measurement_status") == "MEASURED"
            and row.get("exact_group_token_identity") is True
            for row in rows
        )
        print(json.dumps({"measured_groups": len(rows), "complete": complete}, sort_keys=True))
        return 0 if complete else 2
    parser.error("select a reference-preparation or workload-run mode")


if __name__ == "__main__":
    raise SystemExit(main())
