"""Lifecycle and telemetry for isolated Experiment 007 backend services."""

from __future__ import annotations

import json
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_for_http(endpoint: str, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "service did not answer"
    paths = ("/health", "/v1/models", "/props")
    while time.monotonic() < deadline:
        for path in paths:
            try:
                with urllib.request.urlopen(f"{endpoint}{path}", timeout=2) as response:
                    if 200 <= int(response.status) < 500:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"backend service at {endpoint} was not ready: {last_error}")


@dataclass(slots=True)
class ManagedDockerService:
    name: str
    endpoint: str
    command: list[str]
    process: subprocess.Popen[str]
    stdout_handle: TextIO
    stderr_handle: TextIO
    started_at: float
    launch_seconds: float = 0.0

    def close(self, *, repository_root: Path) -> None:
        subprocess.run(
            ["docker", "stop", "--time", "15", self.name],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        self.stdout_handle.close()
        self.stderr_handle.close()


def _launch_container(
    command: list[str],
    *,
    name: str,
    endpoint: str,
    repository_root: Path,
    log_root: Path,
    timeout_seconds: float,
) -> ManagedDockerService:
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_handle = (log_root / f"{name}.stdout.log").open("w", encoding="utf-8")
    stderr_handle = (log_root / f"{name}.stderr.log").open("w", encoding="utf-8")
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=repository_root,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )
    service = ManagedDockerService(
        name=name,
        endpoint=endpoint,
        command=command,
        process=process,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
        started_at=time.time(),
    )
    try:
        wait_for_http(endpoint, timeout_seconds=timeout_seconds)
    except Exception:
        service.close(repository_root=repository_root)
        raise
    service.launch_seconds = time.perf_counter() - started
    (log_root / f"{name}.command.json").write_text(
        json.dumps(
            {
                "command": command,
                "endpoint": endpoint,
                "launch_seconds": service.launch_seconds,
                "pid": process.pid,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return service


def start_sglang_service(
    *,
    image: str,
    repository_root: Path,
    huggingface_cache: Path,
    model_snapshot_relative: str,
    run_id: str,
    log_root: Path,
    maximum_running_requests: int = 64,
) -> ManagedDockerService:
    port = free_local_port()
    name = f"swarm-exp007-sglang-{run_id}"
    endpoint = f"http://127.0.0.1:{port}"
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--gpus",
        "all",
        "--shm-size",
        "16g",
        "--ipc=host",
        "-p",
        f"127.0.0.1:{port}:30000",
        "-v",
        f"{huggingface_cache.resolve()}:/root/.cache/huggingface:ro",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        "--entrypoint",
        "python3",
        image,
        "-m",
        "sglang.launch_server",
        "--model-path",
        f"/root/.cache/huggingface/{model_snapshot_relative}",
        "--dtype",
        "bfloat16",
        "--host",
        "0.0.0.0",
        "--port",
        "30000",
        "--skip-tokenizer-init",
        "--mem-fraction-static",
        "0.70",
        "--max-running-requests",
        str(maximum_running_requests),
        "--enable-deterministic-inference",
    ]
    return _launch_container(
        command,
        name=name,
        endpoint=endpoint,
        repository_root=repository_root,
        log_root=log_root,
        timeout_seconds=900,
    )


def start_llamacpp_service(
    *,
    environment_root: Path,
    build_name: str,
    gguf_path: Path,
    repository_root: Path,
    run_id: str,
    format_name: str,
    log_root: Path,
    thread_count: int,
    parallel: int = 4,
) -> ManagedDockerService:
    port = free_local_port()
    name = f"swarm-exp007-llama-{format_name.lower().replace('_', '-')}-{run_id}"
    endpoint = f"http://127.0.0.1:{port}"
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "-p",
        f"127.0.0.1:{port}:8080",
        "-v",
        f"{environment_root.resolve()}:/work:ro",
        "-v",
        f"{gguf_path.parent.resolve()}:/models:ro",
        "-e",
        f"OMP_NUM_THREADS={thread_count}",
        "ubuntu:24.04",
        f"/work/{build_name}/bin/llama-server",
        "--model",
        f"/models/{gguf_path.name}",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--threads",
        str(thread_count),
        "--threads-batch",
        str(thread_count),
        "--ctx-size",
        "4096",
        "--parallel",
        str(parallel),
        "--metrics",
        "--no-webui",
    ]
    return _launch_container(
        command,
        name=name,
        endpoint=endpoint,
        repository_root=repository_root,
        log_root=log_root,
        timeout_seconds=600,
    )


@dataclass(slots=True)
class HostTelemetry:
    interval_seconds: float = 0.1
    samples: list[dict[str, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        import psutil

        psutil.cpu_percent(interval=None)

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                memory = psutil.virtual_memory()
                row = {
                    "monotonic_seconds": time.monotonic(),
                    "host_cpu_percent": float(psutil.cpu_percent(interval=None)),
                    "host_memory_used_bytes": float(memory.used),
                    "host_memory_available_bytes": float(memory.available),
                }
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    if result.returncode == 0:
                        values = [
                            float(item.strip()) for item in result.stdout.splitlines()[0].split(",")
                        ]
                        row.update(
                            {
                                "gpu_utilisation_percent": values[0],
                                "memory_controller_utilisation_percent": values[1],
                                "gpu_power_watts": values[2],
                                "gpu_memory_used_bytes": values[3] * 1024 * 1024,
                            }
                        )
                except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
                    pass
                self.samples.append(row)

        self._thread = threading.Thread(target=sample, name="experiment-007-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, float]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.samples
