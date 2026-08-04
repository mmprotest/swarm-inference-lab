"""Lifecycle support for the native Colibri Experiment 010 expert worker."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import psutil

from swarm_inference.experiments.experiment_010.colibri_expert_bank import verify_bank
from swarm_inference.experiments.experiment_010.relay import ExpertRelayManager
from swarm_inference.experiments.experiment_010.transport import NETWORK_PROFILES
from swarm_inference.worker.abi import WorkerProtocolVersion
from swarm_inference.worker.universal import UniversalWorkerClient

COLIBRI_MODEL_REVISION = "pinned-b085b48888a88d9a1c00b151a9979774b72cdbfd"


@dataclass(slots=True)
class NativeColibriExpertWorker:
    worker_id: str
    process: subprocess.Popen[bytes]
    endpoint: str
    control_endpoint: str
    bank_path: Path
    ownership_path: Path
    configuration_path: Path
    ready_path: Path
    stdout_path: Path
    stderr_path: Path
    identity: dict[str, Any]
    capabilities: dict[str, Any]
    negotiated_protocol: dict[str, Any]
    replica_of: str | None = None


class NativeColibriExpertWorkerManager:
    """Start exact C workers and verify their Universal Worker handshake."""

    def __init__(self, root: Path, executable: Path) -> None:
        self.root = root.expanduser().resolve()
        self.executable = executable.expanduser().resolve()
        if not self.executable.is_file():
            raise FileNotFoundError(self.executable)
        self.root.mkdir(parents=True, exist_ok=True)
        self.workers: dict[str, NativeColibriExpertWorker] = {}
        self.lifecycle_records: list[dict[str, Any]] = []

    @staticmethod
    def _endpoint(value: str) -> tuple[str, int]:
        host, separator, raw_port = value.rpartition(":")
        if not separator or not host:
            raise ValueError(f"invalid worker endpoint {value!r}")
        port = int(raw_port)
        if not 1 <= port <= 65535:
            raise ValueError(f"invalid worker endpoint {value!r}")
        return host, port

    def start(
        self,
        *,
        worker_id: str,
        bank_path: Path,
        model_id: str,
        model_revision: str,
        quantization_fingerprint: str,
        model_fingerprint: str,
        memory_budget_bytes: int,
        thread_count: int = 1,
        cpu_affinity: list[int] | None = None,
        fixed_delay_ms: int = 0,
        fault_schedule: dict[str, Any] | None = None,
        corruption_schedule: dict[str, Any] | None = None,
        replica_of: str | None = None,
        cuda_target: str | None = None,
        cuda_device: int = 0,
        idot: bool = True,
        timeout_seconds: float = 30.0,
    ) -> NativeColibriExpertWorker:
        if worker_id in self.workers:
            raise ValueError(f"worker {worker_id} is already running")
        bank = bank_path.expanduser().resolve()
        verified = verify_bank(bank)
        if verified["worker_id"] != worker_id and verified["worker_id"] != replica_of:
            raise ValueError(
                "worker ID differs from the exact bank manifest without an explicit replica_of"
            )
        if verified["source_model_fingerprint"] != model_fingerprint:
            raise ValueError("worker model fingerprint differs from the exact bank manifest")
        if memory_budget_bytes <= 0 or thread_count <= 0 or fixed_delay_ms < 0:
            raise ValueError("native worker budgets and delays are invalid")
        if cuda_device < 0:
            raise ValueError("CUDA device must be nonnegative")
        for label, schedule in (
            ("fault_schedule", fault_schedule),
            ("corruption_schedule", corruption_schedule),
        ):
            if schedule is not None and not isinstance(schedule, dict):
                raise ValueError(f"{label} must be a JSON object")
        if cuda_target is not None and not (
            cuda_target == "all" or re.fullmatch(r"\d+:\d+", cuda_target)
        ):
            raise ValueError("CUDA target must be 'all' or layer:expert")

        directory = (self.root / worker_id).resolve()
        if self.root not in directory.parents:
            raise ValueError("native worker directory escaped its manager root")
        directory.mkdir(parents=True, exist_ok=True)
        configuration_path = directory / "worker-config.json"
        ready_path = directory / "ready.json"
        stdout_path = directory / "stdout.log"
        stderr_path = directory / "stderr.log"
        telemetry_path = directory / "worker-telemetry.jsonl"
        if ready_path.is_file():
            ready_path.unlink()
        configuration = {
            "schema_version": "experiment-010-native-colibri-worker-v1",
            "worker_id": worker_id,
            "model_id": model_id,
            "model_revision": model_revision,
            "quantization_fingerprint": quantization_fingerprint,
            "model_fingerprint": model_fingerprint,
            "model_path": str(bank),
            "ownership_manifest": str(bank / "ownership.json"),
            "memory_budget_mb": (memory_budget_bytes + (1024 * 1024 - 1)) // (1024 * 1024),
            "host": "127.0.0.1",
            "port": 0,
            "control_port": 0,
            "fixed_delay_ms": fixed_delay_ms,
            "telemetry_path": str(telemetry_path),
            "fault_schedule": fault_schedule,
            "corruption_schedule": corruption_schedule,
        }
        configuration_path.write_text(
            json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": str(thread_count),
                "OMP_DYNAMIC": "FALSE",
                "IDOT": "1" if idot else "0",
            }
        )
        if cuda_target is not None:
            environment.update(
                {
                    "COLI_SWARM_EXPERT_CUDA_TARGET": cuda_target,
                    "COLI_SWARM_EXPERT_CUDA_DEVICE": str(cuda_device),
                }
            )
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        stdout = stdout_path.open("ab")
        stderr = stderr_path.open("ab")
        started_ns = time.time_ns()
        process = subprocess.Popen(
            [
                str(self.executable),
                "--serve",
                str(configuration_path),
                str(ready_path),
            ],
            cwd=self.root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
        )
        stdout.close()
        stderr.close()
        try:
            if cpu_affinity:
                psutil.Process(process.pid).cpu_affinity(cpu_affinity)
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    detail = stderr_path.read_text(encoding="utf-8", errors="replace")
                    raise RuntimeError(
                        f"native Colibri expert worker exited with {process.returncode}: {detail}"
                    )
                if ready_path.is_file():
                    ready = json.loads(ready_path.read_text(encoding="utf-8"))
                    endpoint = str(ready["endpoint"])
                    control_endpoint = str(ready["control_endpoint"])
                    host, port = self._endpoint(control_endpoint)

                    async def inspect(
                        control_host: str = host,
                        control_port: int = port,
                        data_endpoint: str = endpoint,
                    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
                        client = UniversalWorkerClient(
                            control_host, control_port, timeout_seconds=5.0
                        )
                        negotiated = await client.negotiate(
                            WorkerProtocolVersion(
                                major=1,
                                minor=1,
                                capabilities={
                                    "clean-shutdown",
                                    "direct-expert-data-plane",
                                    "heartbeat",
                                    "moe-expert",
                                },
                            )
                        )
                        identity = await client.identity()
                        capabilities = await client.capabilities()
                        heartbeat = await client.heartbeat()
                        if identity.worker_id != worker_id or heartbeat["worker_id"] != worker_id:
                            raise ValueError("native worker handshake identity mismatch")
                        if (
                            capabilities.backend_details.get("expert_data_endpoint")
                            != data_endpoint
                        ):
                            raise ValueError(
                                "native worker control plane advertised the wrong data plane"
                            )
                        return (
                            negotiated.model_dump(mode="json"),
                            identity.model_dump(mode="json"),
                            capabilities.model_dump(mode="json"),
                        )

                    negotiated, identity, capabilities = asyncio.run(inspect())
                    worker = NativeColibriExpertWorker(
                        worker_id=worker_id,
                        process=process,
                        endpoint=endpoint,
                        control_endpoint=control_endpoint,
                        bank_path=bank,
                        ownership_path=bank / "ownership.json",
                        configuration_path=configuration_path,
                        ready_path=ready_path,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        identity=identity,
                        capabilities=capabilities,
                        negotiated_protocol=negotiated,
                        replica_of=replica_of,
                    )
                    self.workers[worker_id] = worker
                    self.lifecycle_records.append(
                        {
                            "worker_id": worker_id,
                            "pid": process.pid,
                            "binary": str(self.executable),
                            "bank": str(bank),
                            "endpoint": endpoint,
                            "control_endpoint": control_endpoint,
                            "started_ns": started_ns,
                            "status": "RUNNING",
                            "cuda_target": cuda_target,
                            "cuda_device": cuda_device if cuda_target is not None else None,
                            "idot": idot,
                            "replica_of": replica_of,
                            "fault_schedule": fault_schedule,
                            "corruption_schedule": corruption_schedule,
                        }
                    )
                    return worker
                time.sleep(0.05)
            raise TimeoutError(f"native worker {worker_id} did not become ready")
        except BaseException:
            self._terminate(process)
            raise

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        with suppress(psutil.Error):
            psutil.Process(process.pid).terminate()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=3)
        if process.poll() is None:
            with suppress(psutil.Error):
                psutil.Process(process.pid).kill()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=3)

    def stop(self, worker_id: str) -> None:
        worker = self.workers.pop(worker_id, None)
        if worker is None:
            return
        graceful = False
        if worker.process.poll() is None:
            host, port = self._endpoint(worker.control_endpoint)
            with suppress(Exception):
                graceful = bool(
                    asyncio.run(UniversalWorkerClient(host, port, timeout_seconds=3.0).shutdown())
                )
                worker.process.wait(timeout=3)
        self._terminate(worker.process)
        for record in reversed(self.lifecycle_records):
            if record["worker_id"] == worker_id and record["status"] == "RUNNING":
                record.update(
                    {
                        "stopped_ns": time.time_ns(),
                        "exit_code": worker.process.poll(),
                        "graceful": graceful,
                        "status": "STOPPED",
                    }
                )
                break

    def close(self) -> None:
        for worker_id in list(self.workers):
            self.stop(worker_id)

    def __enter__(self) -> NativeColibriExpertWorkerManager:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def whole_expert_ownership_from_banks(bank_paths: list[Path]) -> list[dict[str, int]]:
    """Return a disjoint, stable local-ownership list from native whole banks."""

    identities: set[tuple[int, int]] = set()
    for raw_path in bank_paths:
        bank = raw_path.expanduser().resolve()
        ownership_path = bank / "ownership.json"
        if not ownership_path.is_file():
            raise FileNotFoundError(ownership_path)
        ownership = json.loads(ownership_path.read_text(encoding="utf-8"))
        if ownership.get("owned_microshards"):
            raise ValueError("local hybrid ownership must come from whole-expert banks")
        for row in ownership.get("owned_experts", []):
            identity = (int(row["layer_id"]), int(row["expert_id"]))
            if identity in identities:
                raise ValueError(f"duplicate local expert ownership: {identity}")
            identities.add(identity)
    if not identities:
        raise ValueError("local hybrid ownership cannot be empty")
    return [
        {"layer_id": layer_id, "expert_id": expert_id} for layer_id, expert_id in sorted(identities)
    ]


def write_colibri_expert_plan(
    path: Path,
    *,
    model_fingerprint: str,
    quantization_fingerprint: str,
    phase: str,
    workers: list[NativeColibriExpertWorker],
    local_experts: list[dict[str, int]],
    reduction_order: list[str] | None = None,
    fallback_workers: dict[str, NativeColibriExpertWorker] | None = None,
) -> Path:
    """Write the exact plan schema consumed inside patched Colibri."""

    plan_path = path.expanduser().resolve()
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    replicas = fallback_workers or {}
    primary_ids = {worker.worker_id for worker in workers}
    if set(replicas) - primary_ids:
        raise ValueError("fallback_workers names an unknown primary worker")
    replica_ids = [worker.worker_id for worker in replicas.values()]
    if len(replica_ids) != len(set(replica_ids)) or primary_ids.intersection(replica_ids):
        raise ValueError("replica worker IDs must be unique and distinct from primaries")
    all_workers = [*workers, *replicas.values()]
    worker_rows = []
    for worker in all_workers:
        ownership = json.loads(worker.ownership_path.read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "worker_id": worker.worker_id,
            "endpoint": worker.endpoint,
            "owned_experts": ownership["owned_experts"],
            "owned_microshards": ownership["owned_microshards"],
        }
        if worker.replica_of is not None:
            row["replica_of"] = worker.replica_of
        worker_rows.append(row)
    plan = {
        "schema_version": "1.0",
        "model_fingerprint": model_fingerprint,
        "quantization_fingerprint": quantization_fingerprint,
        "phase": phase,
        "workers": worker_rows,
        "local_experts": local_experts,
        "fallback_workers": {
            primary_id: worker.worker_id for primary_id, worker in replicas.items()
        },
        "reduction_order": reduction_order or [worker.worker_id for worker in workers],
    }
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan_path


def _generated_token_ids(stdout: str) -> list[int]:
    match = re.search(r"(?:^|\n)C engine\s*:\s*([0-9 ]+)", stdout)
    return [int(value) for value in match.group(1).split()] if match else []


def _last_jsonl_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    with path.open("rb") as handle:
        position = handle.seek(0, os.SEEK_END)
        buffer = b""
        while position > 0 and buffer.count(b"\n") < 2:
            size = min(8192, position)
            position -= size
            handle.seek(position)
            buffer = handle.read(size) + buffer
    lines = [line for line in buffer.splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else {}


def _worker_process_accounting(worker: NativeColibriExpertWorker) -> dict[str, Any]:
    manifest = json.loads((worker.bank_path / "manifest.json").read_text(encoding="utf-8"))
    telemetry = _last_jsonl_record(worker.configuration_path.parent / "worker-telemetry.jsonl")
    try:
        process = psutil.Process(worker.process.pid)
        memory = process.memory_full_info()
        io = process.io_counters()
        basic_memory = process.memory_info()
        process_values: dict[str, Any] = {
            "pid": process.pid,
            "working_set_bytes": int(memory.rss),
            "private_bytes": int(getattr(memory, "private", getattr(memory, "uss", 0))),
            "commit_size_bytes": int(getattr(memory, "pagefile", memory.vms)),
            "peak_working_set_bytes": max(
                int(getattr(memory, "peak_wset", memory.rss)),
                int(telemetry.get("peak_working_set_bytes", 0)),
            ),
            "page_fault_count": int(getattr(basic_memory, "num_page_faults", 0)),
            "thread_count": process.num_threads(),
            "cpu_affinity": process.cpu_affinity() if hasattr(process, "cpu_affinity") else [],
            "storage_read_bytes": int(io.read_bytes),
            "storage_write_bytes": int(io.write_bytes),
            "process_alive": worker.process.poll() is None,
        }
    except (psutil.Error, OSError):
        process_values = {
            "pid": worker.process.pid,
            "working_set_bytes": None,
            "private_bytes": None,
            "commit_size_bytes": None,
            "peak_working_set_bytes": telemetry.get("peak_working_set_bytes"),
            "page_fault_count": telemetry.get("page_fault_count"),
            "thread_count": 0,
            "cpu_affinity": [],
            "storage_read_bytes": telemetry.get("process_io_read_bytes"),
            "storage_write_bytes": telemetry.get("process_io_write_bytes"),
            "process_alive": False,
        }
    return {
        "worker_id": worker.worker_id,
        **process_values,
        "expert_bank_bytes": int(manifest["total_expert_bytes"]),
        "owned_expert_count": int(
            manifest.get("owned_expert_count", len(manifest.get("owned_experts", [])))
        ),
        "owned_microshard_count": int(
            manifest.get("owned_microshard_count", len(manifest.get("owned_microshards", [])))
        ),
        "resident_expert_bytes": int(telemetry.get("expert_working_set_bytes", 0)),
        "cache_bytes": int(telemetry.get("cache_bytes", 0)),
        "resident_cache_bytes": int(telemetry.get("resident_cache_bytes", 0)),
        "cache_capacity_bytes": int(telemetry.get("cache_capacity_bytes", 0)),
        "memory_budget_bytes": int(telemetry.get("memory_budget_bytes", 0)),
        "logical_cache_hits": int(telemetry.get("logical_cache_hits", 0)),
        "resident_cache_hits": int(telemetry.get("resident_cache_hits", 0)),
        "nonresident_cache_hits": int(telemetry.get("nonresident_cache_hits", 0)),
        "cache_hits_with_page_fault": int(telemetry.get("cache_hits_with_page_fault", 0)),
        "cache_evictions": int(telemetry.get("cache_evictions", 0)),
        "cuda_requested": bool(telemetry.get("cuda_requested", False)),
        "cuda_initialized": bool(telemetry.get("cuda_initialized", False)),
        "cuda_device": int(telemetry.get("cuda_device", -1)),
        "cuda_target_layer": int(telemetry.get("cuda_target_layer", -1)),
        "cuda_target_expert": int(telemetry.get("cuda_target_expert", -1)),
        "cuda_resident_tensor_count": int(telemetry.get("cuda_resident_tensor_count", 0)),
        "cuda_resident_tensor_bytes": int(telemetry.get("cuda_resident_tensor_bytes", 0)),
        "cuda_weight_upload_ns": int(telemetry.get("cuda_weight_upload_ns", 0)),
        "cuda_execution_count": int(telemetry.get("cuda_execution_count", 0)),
        "cuda_execution_wall_ns": int(telemetry.get("cuda_execution_wall_ns", 0)),
        "cuda_h2d_ns": int(telemetry.get("cuda_h2d_ns", 0)),
        "cuda_kernel_ns": int(telemetry.get("cuda_kernel_ns", 0)),
        "cuda_d2h_ns": int(telemetry.get("cuda_d2h_ns", 0)),
        "cuda_fallback_count": int(telemetry.get("cuda_fallback_count", 0)),
        "resident_query_supported": bool(telemetry.get("resident_query_supported", False)),
        "pagefile_read_bytes": telemetry.get("pagefile_read_bytes"),
        "pagefile_read_bytes_available": bool(
            telemetry.get("pagefile_read_bytes_available", False)
        ),
        "memory_compression_bytes": telemetry.get("memory_compression_bytes"),
        "memory_compression_available": bool(telemetry.get("memory_compression_available", False)),
        "memory_counter_limitation": telemetry.get("memory_counter_limitation"),
        "worker_telemetry": str(worker.configuration_path.parent / "worker-telemetry.jsonl"),
        "bank_manifest": str(worker.bank_path / "manifest.json"),
    }


def run_native_rpc_replay(
    *,
    executable: Path,
    engine: Path,
    model_path: Path,
    reference_path: Path,
    bank_paths: list[Path],
    output_directory: Path,
    model_fingerprint: str,
    quantization_fingerprint: str = "native-colibri-int8-v1",
    mode: str = "rpc",
    response_mode: str = "per_expert_exact",
    data_plane: str = "direct_tcp",
    network_profile: str = "loopback_unshaped",
    coordinator_thread_count: int = 2,
    worker_thread_count: int = 1,
    memory_budget_bytes: int = 256 * 1024 * 1024,
    cuda_targets: dict[str, str] | None = None,
    cuda_device: int = 0,
    idot: bool = True,
    timeout_seconds: float = 600.0,
    expert_timeout_ms: int = 30_000,
    fallback_mode: str = "fail",
    replicate_workers: bool = False,
    fault_schedules: dict[str, dict[str, Any]] | None = None,
    corruption_schedules: dict[str, dict[str, Any]] | None = None,
    verify_every: int = 0,
    challenge_every: int = 0,
    hedge_every: int = 0,
    hedge_delay_ms: int = 5,
    quarantine_threshold: int = 1,
) -> dict[str, Any]:
    """Run a fixed replay through real native workers and patched Colibri."""

    if mode not in {"rpc", "hybrid", "planner"}:
        raise ValueError("native replay mode must be rpc, hybrid, or planner")
    if response_mode not in {"per_expert_exact", "per_worker_fast"}:
        raise ValueError("unsupported native response mode")
    if data_plane not in {"direct_tcp", "relayed_tcp", "shared_memory"}:
        raise ValueError("unsupported native data plane")
    if network_profile not in NETWORK_PROFILES:
        raise ValueError("unsupported native network profile")
    if network_profile != "loopback_unshaped" and data_plane != "relayed_tcp":
        raise ValueError("real network shaping requires the relayed TCP data plane")
    if fallback_mode not in {"fail", "local", "alternate"}:
        raise ValueError("unsupported native fallback mode")
    if (
        expert_timeout_ms <= 0
        or verify_every < 0
        or challenge_every < 0
        or hedge_every < 0
        or hedge_delay_ms < 0
    ):
        raise ValueError("native timeout and verification intervals are invalid")
    if quarantine_threshold <= 0:
        raise ValueError("quarantine threshold must be positive")
    if fallback_mode == "alternate" and not replicate_workers:
        raise ValueError("alternate fallback requires explicit replica workers")
    if not bank_paths:
        raise ValueError("native replay requires at least one worker bank")
    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    telemetry = output / "coordinator-telemetry.jsonl"
    route_trace = output / "route.trace"
    for generated in (telemetry, route_trace):
        if generated.is_file():
            generated.unlink()
    manager = NativeColibriExpertWorkerManager(output / "workers", executable)
    relay_manager = ExpertRelayManager(output / "relays")
    completed: subprocess.CompletedProcess[str] | None = None
    workers: list[NativeColibriExpertWorker] = []
    replicas: dict[str, NativeColibriExpertWorker] = {}
    accounting: list[dict[str, Any]] = []
    relay_metrics: list[dict[str, Any]] = []
    try:
        for bank_path in bank_paths:
            bank = bank_path.expanduser().resolve()
            manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
            workers.append(
                manager.start(
                    worker_id=str(manifest["worker_id"]),
                    bank_path=bank,
                    model_id="colibri-olmoe",
                    model_revision=COLIBRI_MODEL_REVISION,
                    quantization_fingerprint=quantization_fingerprint,
                    model_fingerprint=model_fingerprint,
                    memory_budget_bytes=memory_budget_bytes,
                    thread_count=worker_thread_count,
                    cuda_target=(cuda_targets or {}).get(str(manifest["worker_id"])),
                    cuda_device=cuda_device,
                    idot=idot,
                    fault_schedule=(fault_schedules or {}).get(str(manifest["worker_id"])),
                    corruption_schedule=(corruption_schedules or {}).get(
                        str(manifest["worker_id"])
                    ),
                )
            )
        if replicate_workers:
            for primary, bank_path in zip(workers, bank_paths, strict=True):
                replica_id = f"{primary.worker_id}-alternate"
                replicas[primary.worker_id] = manager.start(
                    worker_id=replica_id,
                    bank_path=bank_path,
                    model_id="colibri-olmoe",
                    model_revision=COLIBRI_MODEL_REVISION,
                    quantization_fingerprint=quantization_fingerprint,
                    model_fingerprint=model_fingerprint,
                    memory_budget_bytes=memory_budget_bytes,
                    thread_count=worker_thread_count,
                    replica_of=primary.worker_id,
                    idot=idot,
                )
        plan_workers = workers
        plan_replicas = replicas
        if data_plane == "relayed_tcp":
            plan_workers = [
                replace(
                    worker,
                    endpoint=relay_manager.start(
                        target_endpoint=worker.endpoint,
                        profile=NETWORK_PROFILES[network_profile],
                    ).endpoint,
                )
                for worker in workers
            ]
            plan_replicas = {
                primary_id: replace(
                    worker,
                    endpoint=relay_manager.start(
                        target_endpoint=worker.endpoint,
                        profile=NETWORK_PROFILES[network_profile],
                    ).endpoint,
                )
                for primary_id, worker in replicas.items()
            }
        plan = write_colibri_expert_plan(
            output / "plan.json",
            model_fingerprint=model_fingerprint,
            quantization_fingerprint=quantization_fingerprint,
            phase="decode",
            workers=plan_workers,
            local_experts=[],
            fallback_workers=plan_replicas,
        )
        environment = os.environ.copy()
        environment.update(
            {
                "SNAP": str(model_path.expanduser().resolve()),
                "COLI_SWARM_EXPERT_MODE": mode,
                "COLI_SWARM_EXPERT_PLAN": str(plan),
                "COLI_SWARM_EXPERT_TIMEOUT_MS": str(expert_timeout_ms),
                "COLI_SWARM_EXPERT_FALLBACK": fallback_mode,
                "COLI_SWARM_EXPERT_VERIFY_EVERY": str(verify_every),
                "COLI_SWARM_EXPERT_CHALLENGE_EVERY": str(challenge_every),
                "COLI_SWARM_EXPERT_HEDGE_EVERY": str(hedge_every),
                "COLI_SWARM_EXPERT_HEDGE_DELAY_MS": str(hedge_delay_ms),
                "COLI_SWARM_EXPERT_QUARANTINE_THRESHOLD": str(quarantine_threshold),
                "COLI_SWARM_EXPERT_RESPONSE_MODE": response_mode,
                "COLI_SWARM_EXPERT_DATA_PLANE": data_plane,
                "COLI_SWARM_EXPERT_DETERMINISM": (
                    "exact" if response_mode == "per_expert_exact" else "quality_bounded"
                ),
                "COLI_SWARM_EXPERT_TELEMETRY": str(telemetry),
                "ROUTE_TRACE": str(route_trace),
                "PILOT": "0",
                "HOT": "0",
                "OMP_NUM_THREADS": str(coordinator_thread_count),
                "OMP_DYNAMIC": "FALSE",
                "IDOT": "1" if idot else "0",
            }
        )
        started_ns = time.perf_counter_ns()
        completed = subprocess.run(
            [
                str(engine.expanduser().resolve()),
                "16",
                "8",
                str(reference_path.expanduser().resolve()),
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed_ns = time.perf_counter_ns() - started_ns
        (output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        accounting = [
            _worker_process_accounting(worker) for worker in [*workers, *replicas.values()]
        ]
        relay_metrics = relay_manager.snapshots() if relay_manager.relays else []
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        prompt_count = len(reference["prompt_ids"])
        expected_tokens = [int(value) for value in reference["full_ids"][prompt_count:]]
        actual_tokens = _generated_token_ids(completed.stdout)
        events = (
            [
                json.loads(line)
                for line in telemetry.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if telemetry.is_file()
            else []
        )
        result = {
            "schema_version": "experiment-010-native-colibri-rpc-replay-v1",
            "mode": mode,
            "response_mode": response_mode,
            "fallback_mode": fallback_mode,
            "expert_timeout_ms": expert_timeout_ms,
            "verify_every": verify_every,
            "challenge_every": challenge_every,
            "hedge_every": hedge_every,
            "hedge_delay_ms": hedge_delay_ms,
            "quarantine_threshold": quarantine_threshold,
            "data_plane": data_plane,
            "network_profile": network_profile,
            "return_code": completed.returncode,
            "elapsed_ns": elapsed_ns,
            "expected_token_ids": expected_tokens,
            "actual_token_ids": actual_tokens,
            "matching_tokens": sum(
                expected == actual
                for expected, actual in zip(expected_tokens, actual_tokens, strict=False)
            ),
            "expected_tokens": len(expected_tokens),
            "exact_token_identity": actual_tokens == expected_tokens,
            "remote_completed_requests": sum(
                event.get("event") == "expert_rpc_request_completed" for event in events
            ),
            "remote_results_consumed": sum(
                event.get("event") == "expert_rpc_request_completed"
                and event.get("remote_result_consumed") is True
                for event in events
            ),
            "forbidden_local_loads": sum(
                event.get("event") == "forbidden_local_expert_load" for event in events
            ),
            "fallback_events": sum(event.get("event") == "expert_rpc_fallback" for event in events),
            "corruption_detections": sum(
                event.get("event") == "expert_rpc_corruption_detected" for event in events
            ),
            "worker_quarantines": sum(
                event.get("event") == "expert_rpc_worker_quarantined" for event in events
            ),
            "clean_duplicate_verifications": sum(
                event.get("event")
                in {
                    "expert_rpc_sampled_duplicate_passed",
                    "expert_rpc_hidden_challenge_passed",
                    "expert_rpc_hedged_duplicate_passed",
                }
                for event in events
            ),
            "model_fingerprint": model_fingerprint,
            "worker_count": len(workers),
            "replica_worker_count": len(replicas),
            "fault_schedules": fault_schedules or {},
            "corruption_schedules": corruption_schedules or {},
            "cuda_targets": cuda_targets or {},
            "cuda_device": cuda_device if cuda_targets else None,
            "idot": idot,
            "worker_process_accounting": accounting,
            "relay_metrics": relay_metrics,
            "plan_path": str(plan),
            "telemetry_path": str(telemetry),
            "route_trace_path": str(route_trace),
        }
    finally:
        relay_manager.close()
        manager.close()
    result["worker_lifecycle"] = manager.lifecycle_records
    result["relay_lifecycle"] = relay_manager.lifecycle_records
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def run_native_hybrid_smoke(
    *,
    executable: Path,
    engine: Path,
    model_path: Path,
    reference_path: Path,
    bank_path: Path,
    output_directory: Path,
    worker_id: str,
    model_fingerprint: str,
    quantization_fingerprint: str,
    thread_count: int = 2,
) -> dict[str, Any]:
    """Run one real Colibri generation with an exact native remote contribution."""

    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    telemetry = output / "coordinator-telemetry.jsonl"
    if telemetry.is_file():
        telemetry.unlink()
    manager = NativeColibriExpertWorkerManager(output / "workers", executable)
    try:
        worker = manager.start(
            worker_id=worker_id,
            bank_path=bank_path,
            model_id="colibri-olmoe",
            model_revision=COLIBRI_MODEL_REVISION,
            quantization_fingerprint=quantization_fingerprint,
            model_fingerprint=model_fingerprint,
            memory_budget_bytes=64 * 1024 * 1024,
            thread_count=thread_count,
        )
        plan = write_colibri_expert_plan(
            output / "plan.json",
            model_fingerprint=model_fingerprint,
            quantization_fingerprint=quantization_fingerprint,
            phase="decode",
            workers=[worker],
            local_experts=[],
        )
        environment = os.environ.copy()
        environment.update(
            {
                "SNAP": str(model_path.expanduser().resolve()),
                "COLI_SWARM_EXPERT_MODE": "hybrid",
                "COLI_SWARM_EXPERT_PLAN": str(plan),
                "COLI_SWARM_EXPERT_TIMEOUT_MS": "30000",
                "COLI_SWARM_EXPERT_FALLBACK": "fail",
                "COLI_SWARM_EXPERT_RESPONSE_MODE": "per_expert_exact",
                "COLI_SWARM_EXPERT_DETERMINISM": "exact",
                "COLI_SWARM_EXPERT_TELEMETRY": str(telemetry),
                "PILOT": "0",
                "HOT": "0",
                "OMP_NUM_THREADS": str(thread_count),
            }
        )
        completed = subprocess.run(
            [
                str(engine.expanduser().resolve()),
                "16",
                "8",
                str(reference_path.expanduser().resolve()),
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        (output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        token_match = re.search(r"Matching tokens:\s*(\d+)/(\d+)", completed.stdout)
        events = (
            [
                json.loads(line)
                for line in telemetry.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if telemetry.is_file()
            else []
        )
        result = {
            "schema_version": "experiment-010-native-hybrid-smoke-v1",
            "return_code": completed.returncode,
            "matching_tokens": int(token_match.group(1)) if token_match else 0,
            "expected_tokens": int(token_match.group(2)) if token_match else 0,
            "remote_completed_requests": sum(
                event.get("event") == "expert_rpc_request_completed" for event in events
            ),
            "remote_results_consumed": sum(
                event.get("event") == "expert_rpc_request_completed"
                and event.get("remote_result_consumed") is True
                for event in events
            ),
            "forbidden_local_loads": sum(
                event.get("event") == "forbidden_local_expert_load" for event in events
            ),
            "model_fingerprint": model_fingerprint,
            "worker_id": worker_id,
            "worker_endpoint": worker.endpoint,
            "plan_path": str(plan),
            "telemetry_path": str(telemetry),
        }
    finally:
        manager.close()
    result["worker_lifecycle"] = manager.lifecycle_records
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-smoke", action="store_true")
    parser.add_argument("--rpc-replay", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--banks", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-id")
    parser.add_argument("--model-fingerprint")
    parser.add_argument("--cuda-target", action="append", default=[])
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--disable-idot", action="store_true")
    parser.add_argument("--quantization-fingerprint", default="native-colibri-int8-v1")
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
        "--network-profile",
        choices=tuple(NETWORK_PROFILES),
        default="loopback_unshaped",
    )
    arguments = parser.parse_args()
    if arguments.rpc_replay:
        required = (
            arguments.worker,
            arguments.engine,
            arguments.model,
            arguments.reference,
            arguments.output,
            arguments.model_fingerprint,
        )
        if any(value is None for value in required) or not arguments.banks:
            parser.error(
                "--rpc-replay requires worker, engine, model, reference, output, fingerprint, and banks"
            )
        result = run_native_rpc_replay(
            executable=arguments.worker,
            engine=arguments.engine,
            model_path=arguments.model,
            reference_path=arguments.reference,
            bank_paths=arguments.banks,
            output_directory=arguments.output,
            model_fingerprint=arguments.model_fingerprint,
            quantization_fingerprint=arguments.quantization_fingerprint,
            mode=arguments.mode,
            response_mode=arguments.response_mode,
            data_plane=arguments.data_plane,
            network_profile=arguments.network_profile,
            cuda_targets=dict(item.split("=", 1) for item in arguments.cuda_target),
            cuda_device=arguments.cuda_device,
            idot=not arguments.disable_idot,
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if result["return_code"] == 0 and result["exact_token_identity"] else 1
    if not arguments.hybrid_smoke or any(
        value is None
        for value in (
            arguments.worker,
            arguments.engine,
            arguments.model,
            arguments.reference,
            arguments.bank,
            arguments.output,
            arguments.worker_id,
            arguments.model_fingerprint,
        )
    ):
        parser.error("--hybrid-smoke requires all path, worker, and fingerprint arguments")
    result = run_native_hybrid_smoke(
        executable=arguments.worker,
        engine=arguments.engine,
        model_path=arguments.model,
        reference_path=arguments.reference,
        bank_path=arguments.bank,
        output_directory=arguments.output,
        worker_id=arguments.worker_id,
        model_fingerprint=arguments.model_fingerprint,
        quantization_fingerprint=arguments.quantization_fingerprint,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["return_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
