"""Unattended E021 host-agent and explicit per-GPU worker entry point."""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _credential_bytes() -> bytes:
    value = os.environ.get("SWARM_RUN_CREDENTIAL_FILE")
    if value and Path(value).is_file():
        credential = Path(value).read_bytes()
    else:
        encoded = os.environ.get("SWARM_RUN_CREDENTIAL_B64", "")
        try:
            credential = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise SystemExit("valid runtime credential is required") from exc
    if len(credential) < 32:
        raise SystemExit("runtime credential is too short")
    return credential


def _worker(arguments: argparse.Namespace) -> int:
    _credential_bytes()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(arguments.gpu_index):
        raise SystemExit("worker GPU binding does not match its explicit GPU index")
    manifest_root = Path(
        os.environ.get("SWARM_WORKER_MANIFEST_ROOT", "/opt/swarm/manifests")
    )
    manifest = manifest_root / f"{arguments.worker_id}.json"
    if not arguments.logical_device and not manifest.is_file():
        raise SystemExit(f"worker manifest is required: {manifest}")
    print(
        json.dumps(
            {
                "status": "worker-ready",
                "worker_id": arguments.worker_id,
                "gpu_index": arguments.gpu_index,
                "visible_device": visible,
                "whole_layer_fallback": False,
                "whole_expert_fallback": False,
                "credential_present": True,
                "credential_value_logged": False,
            }
        ),
        flush=True,
    )
    if arguments.self_test:
        return 0
    stopping = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        time.sleep(0.25)
    return 0


def _worker_command(gpu_index: int, pod_id: str, *, logical: bool, self_test: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "swarm_inference.experiments.experiment_020.worker_main",
        "worker",
        "--gpu-index",
        str(gpu_index),
        "--worker-id",
        f"{pod_id}.worker-{gpu_index:02d}",
    ]
    if logical:
        command.append("--logical-device")
    if self_test:
        command.append("--self-test")
    return command


def _host_agent(arguments: argparse.Namespace) -> int:
    _credential_bytes()
    if arguments.workers_per_pod != 8:
        raise SystemExit("E021 primary pod requires eight explicit GPU workers")
    pod_id = os.environ.get("SWARM_POD_ID", "pod-000")
    processes: list[subprocess.Popen[str]] = []
    ready: list[dict[str, Any]] = []
    try:
        for gpu_index in range(arguments.workers_per_pod):
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            process = subprocess.Popen(
                _worker_command(
                    gpu_index,
                    pod_id,
                    logical=arguments.logical_devices,
                    self_test=arguments.self_test,
                ),
                env=environment,
                stdout=subprocess.PIPE if arguments.self_test else None,
                stderr=subprocess.PIPE if arguments.self_test else None,
                text=True,
            )
            processes.append(process)
        if arguments.self_test:
            for process in processes:
                stdout, stderr = process.communicate(timeout=30)
                if process.returncode != 0:
                    raise RuntimeError(f"worker failed: {stderr.strip()}")
                ready.append(json.loads(stdout.strip()))
            receipt = {
                "status": "PASS",
                "host_agent_compute_resource": False,
                "explicit_worker_processes": len(processes),
                "unique_worker_ids": len({row["worker_id"] for row in ready}),
                "unique_gpu_bindings": len({row["visible_device"] for row in ready}),
                "whole_layer_fallback": False,
                "whole_expert_fallback": False,
                "workers": ready,
            }
            print(json.dumps(receipt), flush=True)
            return 0
        print(
            json.dumps(
                {
                    "status": "host-agent-ready",
                    "workers": len(processes),
                    "host_agent_compute_resource": False,
                    "credential_present": True,
                    "credential_value_logged": False,
                }
            ),
            flush=True,
        )
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        raise RuntimeError("a bounded GPU worker exited unexpectedly")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health")
    host = subparsers.add_parser("host-agent")
    host.add_argument("--workers-per-pod", type=int, default=8)
    host.add_argument("--logical-devices", action="store_true")
    host.add_argument("--self-test", action="store_true")
    worker = subparsers.add_parser("worker")
    worker.add_argument("--gpu-index", type=int, required=True)
    worker.add_argument("--worker-id", required=True)
    worker.add_argument("--logical-device", action="store_true")
    worker.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "health":
        print(json.dumps({"status": "healthy", "compute_resource": False}))
        return 0
    if arguments.command == "worker":
        return _worker(arguments)
    return _host_agent(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
