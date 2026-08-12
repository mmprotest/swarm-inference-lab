"""Memory packing and paid-GPU-equivalent accounting for logical workers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_015.evidence import atomic_json, read_json


@dataclass(frozen=True, slots=True)
class HardwareClass:
    """Configurable worker hardware; absent fields prohibit economic claims."""

    name: str
    vram_gib: float
    memory_bandwidth_gbps: float | None = None
    compute_relative: float | None = None
    hourly_price_usd: float | None = None

    def __post_init__(self) -> None:
        if self.vram_gib <= 0:
            raise ValueError("hardware VRAM must be positive")
        for value in (
            self.memory_bandwidth_gbps,
            self.compute_relative,
            self.hourly_price_usd,
        ):
            if value is not None and value <= 0:
                raise ValueError("specified hardware characteristics must be positive")


DEFAULT_MEMORY_CLASSES = tuple(
    HardwareClass(name=f"generic-{size}gb", vram_gib=float(size))
    for size in (8, 12, 16, 24, 32, 48)
)


def worker_slots(
    hardware: HardwareClass,
    resident_worker_bytes: int,
    *,
    fixed_runtime_bytes: int = 1_073_741_824,
    safety_fraction: float = 0.10,
) -> int:
    """Memory-only slots; compute concurrency is intentionally separate."""
    if resident_worker_bytes <= 0 or fixed_runtime_bytes < 0:
        raise ValueError("packing bytes must be positive/non-negative")
    if not 0 <= safety_fraction < 1:
        raise ValueError("packing safety fraction must be in [0, 1)")
    usable = int(hardware.vram_gib * (1024**3) * (1.0 - safety_fraction))
    return max(0, (usable - fixed_runtime_bytes) // resident_worker_bytes)


def build_packing_model(
    repository_root: Path,
    output_path: Path,
    *,
    hardware_classes: tuple[HardwareClass, ...] = DEFAULT_MEMORY_CLASSES,
) -> dict[str, Any]:
    """Build memory-only packing and prevent it being mistaken for compute capacity."""
    root = repository_root.expanduser().resolve()
    microwork = read_json(root / "artifacts" / "experiment-015" / "microwork" / "results.json")
    logical = [
        {
            "name": f"expert-{row['workers']}-way-shard",
            "logical_workers_per_layer": int(row["workers"]),
            "resident_worker_bytes": int(row["worker_tracked_bytes"]),
            "source_worker_evidence_class": row["evidence_class"],
        }
        for row in microwork["measured_scaling"]
    ]
    rows: list[dict[str, Any]] = []
    for worker in logical:
        for hardware in hardware_classes:
            slots = worker_slots(hardware, int(worker["resident_worker_bytes"]))
            logical_count = int(worker["logical_workers_per_layer"])
            memory_devices = math.ceil(logical_count / slots) if slots else None
            rows.append(
                {
                    **worker,
                    "evidence_class": "PROJECTED",
                    "hardware": asdict(hardware),
                    "memory_slots_per_device": slots,
                    "memory_devices_for_one_layer": memory_devices,
                    "concurrent_compute_devices_required": "NOT_ESTABLISHED",
                    "paid_gpu_equivalent": "NOT_ESTABLISHED",
                    "economic_result_admitted": False,
                }
            )

    smallest_measured = min(int(item["resident_worker_bytes"]) for item in logical)
    # The smallest shard is not automatically useful: the retained 16-way run
    # was only 71.6% of its one-GPU reference.  Eight GiB is the smallest listed
    # class with room for the measured four-worker ready delta plus headroom.
    receipt: dict[str, Any] = {
        "schema_version": "experiment-015-worker-packing-v1",
        "cycle_id": "H015-PACK-001",
        "status": "PASS",
        "evidence_class": "PROJECTED",
        "hardware_classes": [asdict(item) for item in hardware_classes],
        "packing_assumptions": {
            "fixed_runtime_bytes_per_device": 1_073_741_824,
            "safety_fraction": 0.10,
            "compute_and_bandwidth_constraints_are_not_inferred_from_vram": True,
            "unknown_prices_are_not_imputed": True,
        },
        "rows": rows,
        "smallest_measured_logical_worker_bytes": smallest_measured,
        "minimum_memory_class_that_can_host_measured_four_worker_ready_delta_gib": 8,
        "minimum_economically_useful_worker_memory_gib": "NOT_ESTABLISHED",
        "decision": "MEMORY_PACKING_ONLY; NO PAID_GPU_REDUCTION_ADMITTED",
    }
    atomic_json(output_path, receipt)
    return receipt


__all__ = ["DEFAULT_MEMORY_CLASSES", "HardwareClass", "build_packing_model", "worker_slots"]
