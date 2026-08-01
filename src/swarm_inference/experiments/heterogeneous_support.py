"""Shared orchestration helpers for Experiment 007."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from swarm_inference.backends.artifacts import canonical_json_hash
from swarm_inference.worker.abi import WorkerCapabilities, WorkerProtocolVersion
from swarm_inference.worker.universal import UniversalWorkerClient


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def yaml_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    materialised = rows
    if not fields:
        fields = ["status", "reason"]
        materialised = [{"status": "UNAVAILABLE", "reason": "no observations"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in materialised:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True, default=str)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_git_state(repository_root: Path) -> dict[str, Any]:
    def read(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()

    status = read("status", "--porcelain=v1")
    return {
        "commit": read("rev-parse", "HEAD"),
        "branch": read("branch", "--show-current"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "captured_at_utc": datetime.now(UTC).isoformat(),
    }


def find_reference_run(
    repository_root: Path,
    *,
    kind: str,
    configured: str | None,
) -> Path:
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = repository_root / path
        candidates = [path.resolve()]
    else:
        candidates = []
        for root_name in ("runs", "runs-final"):
            root = repository_root / "artifacts" / root_name
            if root.is_dir():
                candidates.extend(
                    item for item in root.iterdir() if item.is_dir() and kind in item.name
                )
        candidates.sort(key=lambda item: item.name, reverse=True)
    for candidate in candidates:
        summary_path = candidate / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("overall_status") == "PASS":
            return candidate
    raise FileNotFoundError(f"no valid PASS {kind} reference artifact was found")


def reference_evidence(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    return {
        "run_directory": str(path.resolve()),
        "run_id": path.name,
        "summary_sha256": sha256(summary_path),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "selected_at_utc": datetime.now(UTC).isoformat(),
    }


def partition_hash(partition_root: Path) -> str:
    parts = {
        name: sha256(partition_root / name) for name in ("manifest.json", "parallel_plan.json")
    }
    return canonical_json_hash(parts)


def stage_shard_hashes(partition_root: Path) -> list[str]:
    hashes: list[str] = []
    for stage_id in range(4):
        manifest = json.loads(
            (
                partition_root
                / "stages"
                / f"stage-{stage_id:03d}"
                / "ranks"
                / "rank-000"
                / "shard_manifest.json"
            ).read_text(encoding="utf-8")
        )
        hashes.append(str(manifest["weight_file_hash"]))
    return hashes


def cpu_capabilities(*, backend_features: list[str] | None = None) -> WorkerCapabilities:
    import psutil

    memory = psutil.virtual_memory()
    logical = os.cpu_count() or 1
    return WorkerCapabilities(
        architecture=platform.machine(),
        operating_system=platform.platform(),
        cpu_model=platform.processor() or "Intel x86-64 host CPU",
        physical_cpu_cores=psutil.cpu_count(logical=False) or logical,
        logical_cpu_cores=logical,
        cpu_features=_cpu_features(),
        system_memory_bytes=int(memory.total),
        supported_weight_formats=["safetensors", "GGUF/Q8_0", "GGUF/Q4_K_M"],
        supported_activation_dtypes=["bfloat16", "float16", "float32", "int8"],
        supported_cache_dtypes=["bfloat16", "float16", "q8_0", "q4_k_m"],
        supported_collectives=[],
        maximum_weight_bytes=int(memory.available * 0.75),
        maximum_cache_bytes=int(memory.available * 0.20),
        maximum_batch_size=max(1, logical // 2),
        maximum_context_length=4096,
        measured_network_upload_bps=20_000_000_000,
        measured_network_download_bps=20_000_000_000,
        coordinator_latency_ms=0.1,
        backend_features=backend_features or [],
    )


def cuda_capabilities() -> WorkerCapabilities:
    import psutil
    import torch

    base = cpu_capabilities(backend_features=["continuous_batching", "paged_kv", "prefix_cache"])
    free, total = torch.cuda.mem_get_info(0)
    return base.model_copy(
        update={
            "accelerator_type": "cuda",
            "accelerator_model": torch.cuda.get_device_name(0),
            "accelerator_memory_bytes": int(total),
            "maximum_weight_bytes": int(free * 0.75),
            "maximum_cache_bytes": int(free * 0.20),
            "maximum_batch_size": 64,
            "system_memory_bytes": int(psutil.virtual_memory().total),
        }
    )


def _cpu_features() -> list[str]:
    features: list[str] = []
    try:
        import torch

        features.append(str(torch.backends.cpu.get_cpu_capability()).lower())
        if bool(torch.backends.mkldnn.enabled):
            features.append("mkldnn")
    except (AttributeError, ImportError):
        pass
    return sorted(set(features))


def exact_token_length(seed: list[int], length: int, *, offset: int = 0) -> list[int]:
    if not seed or length <= 0:
        raise ValueError("token fixture requires a non-empty seed and positive length")
    repeated = seed * ((length + offset) // len(seed) + 2)
    return repeated[offset : offset + length]


@dataclass(slots=True)
class UniversalWorkerProcess:
    stage_id: int
    device: str
    process: subprocess.Popen[str]
    client: UniversalWorkerClient
    ready: dict[str, Any]
    stdout_handle: Any
    stderr_handle: Any

    async def close(self) -> None:
        with suppress(OSError, RuntimeError, TimeoutError):
            await self.client.shutdown()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.stdout_handle.close()
        self.stderr_handle.close()


def start_universal_stage_worker(
    *,
    repository_root: Path,
    partition_root: Path,
    partition_manifest_hash: str,
    stage_id: int,
    device: str,
    python_executable: Path,
    run_directory: Path,
    run_id: str,
) -> UniversalWorkerProcess:
    process_tag = f"{run_id}-stage-{stage_id}-{device}"
    ready_file = run_directory / "logs" / f"{process_tag}.ready.json"
    stdout_path = run_directory / "logs" / f"{process_tag}.stdout.log"
    stderr_path = run_directory / "logs" / f"{process_tag}.stderr.log"
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    if ready_file.is_file():
        ready_file.unlink()
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    command = [
        str(python_executable),
        "-m",
        "swarm_inference.worker.universal_process",
        "--partition",
        str(partition_root),
        "--partition-hash",
        partition_manifest_hash,
        "--stage-id",
        str(stage_id),
        "--device",
        device,
        "--dtype",
        "bfloat16",
        "--worker-id",
        f"exp007-{run_id}-stage-{stage_id}-{device}",
        "--ready-file",
        str(ready_file),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository_root / "src")
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 2) // 2))
    process = subprocess.Popen(
        command,
        cwd=repository_root,
        env=environment,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if ready_file.is_file():
            ready = json.loads(ready_file.read_text(encoding="utf-8"))
            client = UniversalWorkerClient(
                str(ready["host"]), int(ready["port"]), timeout_seconds=600
            )
            asyncio.run(
                client.negotiate(WorkerProtocolVersion(major=1, minor=0, capabilities={"jobs"}))
            )
            return UniversalWorkerProcess(
                stage_id=stage_id,
                device=device,
                process=process,
                client=client,
                ready=ready,
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
            )
        code = process.poll()
        if code is not None:
            stdout_handle.close()
            stderr_handle.close()
            diagnostic = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"stage {stage_id} {device} worker exited {code}: {diagnostic}")
        time.sleep(0.25)
    process.terminate()
    stdout_handle.close()
    stderr_handle.close()
    raise TimeoutError(f"stage {stage_id} {device} worker startup timed out")


def environment_snapshot() -> dict[str, Any]:
    import psutil

    return {
        "classification": ["measured_cuda", "measured_x86_cpu"],
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version,
        "python_executable": sys.executable,
        "logical_cpu_cores": os.cpu_count(),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "system_memory_bytes": psutil.virtual_memory().total,
        "pid": os.getpid(),
    }
