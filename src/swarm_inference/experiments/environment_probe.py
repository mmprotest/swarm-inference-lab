"""Isolated environment probe used before CUDA workers are spawned."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil


def collect_environment() -> dict[str, Any]:
    import safetensors
    import torch
    import transformers

    cuda_visible = bool(torch.cuda.is_available())
    gpu: dict[str, Any] = {}
    if cuda_visible:
        free, total = torch.cuda.mem_get_info(0)
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "model": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_vram_bytes": int(total),
            "available_vram_bytes": int(free),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "multiprocessor_count": int(properties.multi_processor_count),
        }
    driver = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            driver = result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired, IndexError):
        driver = None
    memory = psutil.virtual_memory()
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "safetensors_version": safetensors.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "nvidia_driver_version": driver,
        "cuda_available": cuda_visible,
        "gpu": gpu,
        "system_ram_total_bytes": int(memory.total),
        "system_ram_available_bytes": int(memory.available),
        "probe_process_id": __import__("os").getpid(),
        "probe_process_exited_before_workers": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    output = Path(arguments.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(collect_environment(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
