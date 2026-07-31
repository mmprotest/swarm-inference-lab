"""Emit package, CUDA, and GPU facts from an isolated engine environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload: dict[str, Any] = {
        "engine": arguments.engine,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "transformers",
                "sglang",
                "vllm",
                "tensorrt-llm",
                "flashinfer-python",
                "triton",
                "triton-windows",
            )
        },
    }
    try:
        import torch

        payload.update(
            {
                "torch_version": torch.__version__,
                "cuda_runtime_version": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                "compute_capability": (
                    list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
                ),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "cuda_available": False,
                "torch_probe_error": f"{type(exc).__name__}: {exc}",
            }
        )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        payload["nvidia_smi"] = result.stdout.strip()
        payload["nvidia_smi_return_code"] = result.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        payload["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("cuda_available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
