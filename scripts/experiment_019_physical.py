"""Run one reproducible Experiment 019 physical evidence arm."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_019.attention import (
    benchmark_attention_stripes,
)
from swarm_inference.experiments.experiment_019.other_physical import (
    benchmark_other_shards,
)
from swarm_inference.experiments.experiment_019.physical import (
    benchmark_expert_stripes,
)
from swarm_inference.experiments.experiment_019.protocol import (
    benchmark_control_plane,
    benchmark_persistent_protocol,
)
from swarm_inference.experiments.experiment_019.sharded_graph import (
    validate_depth_spans,
    validate_representative_layers,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class _GpuSampler:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5,
                ).stdout.strip()
                if output:
                    parts = [item.strip() for item in output.splitlines()[0].split(",")]
                    self.samples.append(
                        {
                            "timestamp": parts[0],
                            "gpu_utilization_percent": float(parts[1]),
                            "memory_controller_utilization_percent": float(parts[2]),
                            "vram_used_mib": float(parts[3]),
                            "power_w": float(parts[4]),
                            "temperature_c": float(parts[5]),
                        }
                    )
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(0.5)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arm",
        choices=(
            "attention",
            "depth",
            "expert",
            "other",
            "protocol",
            "representative",
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("F:/models/Kimi-K3"))
    parser.add_argument(
        "--cuda-library",
        type=Path,
        default=Path("artifacts/experiment-016/cuda/coli_cuda-sm120-h016-final.dll"),
    )
    parser.add_argument(
        "--shard-library",
        type=Path,
        default=Path("artifacts/experiment-019/physical/exp019-kda-shard-sm120-v2.dll"),
    )
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=Path("artifacts/experiment-014/oracle-full-93-idot0"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    sampler = _GpuSampler()
    sampler.start()
    try:
        receipt = _execute(arguments)
    finally:
        sampler.stop()
    receipt["gpu_samples"] = sampler.samples
    _write(arguments.output, receipt)
    print(json.dumps({"arm": arguments.arm, "status": receipt["status"]}))
    return 0 if receipt["status"] == "PASS" else 1


def _execute(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.arm == "attention":
        calibration = benchmark_attention_stripes(
            arguments.checkpoint,
            arguments.cuda_library,
            arguments.oracle_root / "hidden-trace.f32",
            arguments.shard_library,
            kda_layer=45,
            mla_layer=47,
            warmup=3,
            iterations=15,
        )
        heldout = benchmark_attention_stripes(
            arguments.checkpoint,
            arguments.cuda_library,
            arguments.oracle_root / "hidden-trace.f32",
            arguments.shard_library,
            kda_layer=89,
            mla_layer=91,
            warmup=3,
            iterations=15,
        )
        receipt = {
            "schema_version": "experiment-019-attention-service-validation-v1",
            "status": (
                "PASS"
                if calibration["status"] == "PASS" and heldout["status"] == "PASS"
                else "FAIL"
            ),
            "calibration": calibration,
            "heldout": heldout,
        }
    elif arguments.arm == "depth":
        receipt = validate_depth_spans(
            arguments.checkpoint,
            arguments.cuda_library,
            arguments.shard_library,
            arguments.oracle_root / "hidden-trace.f32",
            arguments.oracle_root / "routes.txt",
        )
    elif arguments.arm == "expert":
        receipt = benchmark_expert_stripes(
            arguments.checkpoint,
            arguments.cuda_library,
            arguments.oracle_root / "hidden-trace.f32",
            arguments.oracle_root / "routes.txt",
            quantizer_library=arguments.shard_library,
            warmup=1,
            iterations=5,
        )
    elif arguments.arm == "other":
        receipt = benchmark_other_shards(
            arguments.checkpoint,
            arguments.cuda_library,
            arguments.shard_library,
            arguments.oracle_root / "hidden-trace.f32",
        )
    elif arguments.arm == "representative":
        receipt = validate_representative_layers(
            arguments.checkpoint,
            arguments.cuda_library,
            arguments.shard_library,
            arguments.oracle_root / "hidden-trace.f32",
            arguments.oracle_root / "routes.txt",
        )
    else:
        protocol = benchmark_persistent_protocol(
            (128, 14336, 28672, 57344, 114688), iterations=100, warmup=10
        )
        scaling = benchmark_control_plane((100, 250, 376, 500, 1000, 2000, 2976))
        receipt = {
            "schema_version": "experiment-019-control-plane-evidence-v1",
            "status": protocol["status"],
            "protocol": protocol,
            "scaling": scaling,
        }
    return receipt


if __name__ == "__main__":
    raise SystemExit(main())
