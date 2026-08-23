"""Launch one isolated E025 worker process per physical GPU in a Vast instance."""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any

BASE_PORT = 42525
MAX_WORKERS_PER_INSTANCE = 8


def _decode_specs(encoded: str | None, *, default_port: int) -> list[dict[str, Any]]:
    if not encoded:
        worker_id = os.environ.get("E025_WORKER_ID")
        if not worker_id:
            raise ValueError("E025_WORKER_ID or E025_WORKER_SPECS_B64 is required")
        return [
            {
                "worker_id": worker_id,
                "gpu_slot": 0,
                "port": default_port,
                "maximum_context": int(os.environ.get("E025_MAXIMUM_CONTEXT", "64")),
                "preserve_expert_endpoints": True,
            }
        ]
    try:
        value = json.loads(base64.b64decode(encoded, validate=True))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError("E025_WORKER_SPECS_B64 is invalid") from exc
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_WORKERS_PER_INSTANCE:
        raise ValueError("E025 worker specs must contain one to eight workers")
    specs: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("E025 worker spec is not an object")
        spec = {
            "worker_id": str(row.get("worker_id", "")),
            "gpu_slot": int(row.get("gpu_slot", -1)),
            "port": int(row.get("port", -1)),
            "maximum_context": int(row.get("maximum_context", 64)),
            "expert_endpoints": row.get("expert_endpoints"),
        }
        if not spec["worker_id"] or not 0 <= spec["gpu_slot"] < MAX_WORKERS_PER_INSTANCE:
            raise ValueError("E025 worker spec has an invalid identity or GPU slot")
        if not BASE_PORT <= spec["port"] < BASE_PORT + MAX_WORKERS_PER_INSTANCE:
            raise ValueError("E025 worker spec has an invalid serving port")
        specs.append(spec)
    if len({row["worker_id"] for row in specs}) != len(specs):
        raise ValueError("E025 worker specs contain duplicate worker IDs")
    if len({row["gpu_slot"] for row in specs}) != len(specs):
        raise ValueError("E025 worker specs contain duplicate physical GPU slots")
    if len({row["port"] for row in specs}) != len(specs):
        raise ValueError("E025 worker specs contain duplicate serving ports")
    return specs


def _child_environment(spec: dict[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("E025_WORKER_SPECS_B64", None)
    environment["E025_WORKER_ID"] = str(spec["worker_id"])
    environment["E025_MAXIMUM_CONTEXT"] = str(spec["maximum_context"])
    environment["E025_GPU_SLOT"] = str(spec["gpu_slot"])
    environment["CUDA_VISIBLE_DEVICES"] = str(spec["gpu_slot"])
    endpoints = spec.get("expert_endpoints")
    if endpoints is not None:
        environment["E025_EXPERT_ENDPOINTS_B64"] = base64.b64encode(
            json.dumps(endpoints, sort_keys=True, separators=(",", ":")).encode()
        ).decode("ascii")
    elif not spec.get("preserve_expert_endpoints"):
        environment.pop("E025_EXPERT_ENDPOINTS_B64", None)
    return environment


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=BASE_PORT)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    specs = _decode_specs(
        os.environ.get("E025_WORKER_SPECS_B64"),
        default_port=arguments.port,
    )
    children: list[subprocess.Popen[bytes]] = []
    terminating = False

    def stop_children(signum: int, _frame: Any) -> None:
        nonlocal terminating
        terminating = True
        for child in children:
            if child.poll() is None:
                child.send_signal(signum)

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)
    for spec in specs:
        children.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "swarm_inference.experiments.experiment_025.bootstrap",
                    "--worker-id",
                    str(spec["worker_id"]),
                    "--port",
                    str(spec["port"]),
                ],
                env=_child_environment(spec),
            )
        )
    while True:
        for child in children:
            returncode = child.poll()
            if returncode is not None:
                for sibling in children:
                    if sibling.poll() is None:
                        sibling.terminate()
                for sibling in children:
                    if sibling.poll() is None:
                        try:
                            sibling.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            sibling.kill()
                return 0 if terminating and returncode in {0, -signal.SIGTERM} else int(
                    returncode or 1
                )
        time.sleep(0.25)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BASE_PORT", "MAX_WORKERS_PER_INSTANCE", "main"]
