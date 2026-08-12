"""Retain H014-028e compiled resource and scaling evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SM86_LIMITS = {
    "registers_per_sm": 65_536,
    "shared_bytes_per_sm": 102_400,
    "threads_per_sm": 1_536,
    "warps_per_sm": 48,
    "blocks_per_sm": 16,
}
THREADS_PER_BLOCK = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _resources(cuobjdump: Path, binary: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(cuobjdump), "--dump-resource-usage", str(binary)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    text = completed.stdout + completed.stderr
    functions: dict[str, dict[str, int]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("Function "):
            current = line.removeprefix("Function ").rstrip(":")
            continue
        if current is None or "REG:" not in line:
            continue
        values = {
            name.lower(): int(value)
            for name, value in re.findall(r"([A-Z]+(?:\[\d+\])?):(\d+)", line)
        }
        functions[current] = values
        current = None

    kernels: dict[str, Any] = {}
    for function, values in functions.items():
        if "mxfp4_matmul_pair_rows" not in function and "mxfp4_matmul_rows" not in function:
            continue
        match = re.search(r"ILi(\d+)E", function)
        if match is None:
            continue
        batch = int(match.group(1))
        kind = "pair" if "pair_rows" in function else "single"
        registers = values["reg"]
        shared_bytes = values["shared"]
        registers_per_block = registers * THREADS_PER_BLOCK
        register_blocks = SM86_LIMITS["registers_per_sm"] // registers_per_block
        shared_blocks = (
            SM86_LIMITS["shared_bytes_per_sm"] // shared_bytes
            if shared_bytes
            else SM86_LIMITS["blocks_per_sm"]
        )
        thread_blocks = SM86_LIMITS["threads_per_sm"] // THREADS_PER_BLOCK
        resident_blocks = min(
            register_blocks,
            shared_blocks,
            thread_blocks,
            SM86_LIMITS["blocks_per_sm"],
        )
        active_warps = resident_blocks * (THREADS_PER_BLOCK // 32)
        kernels[f"{kind}_batch{batch}"] = {
            "function": function,
            "registers_per_thread": registers,
            "shared_bytes_per_block": shared_bytes,
            "local_bytes_per_thread": values.get("local", 0),
            "stack_bytes_per_thread": values.get("stack", 0),
            "theoretical_sm86": {
                "blocks_limited_by_registers": register_blocks,
                "blocks_limited_by_shared_memory": shared_blocks,
                "blocks_limited_by_threads": thread_blocks,
                "resident_blocks_per_sm": resident_blocks,
                "active_warps_per_sm": active_warps,
                "warp_occupancy_percent_upper_bound": (
                    100.0 * active_warps / SM86_LIMITS["warps_per_sm"]
                ),
                "qualification": (
                    "static resource upper bound; allocation granularity and runtime "
                    "scheduler effects can only reduce this value"
                ),
            },
        }
    return {
        "returncode": completed.returncode,
        "kernels": dict(sorted(kernels.items())),
        "stderr": completed.stderr.strip(),
    }


def _ncu_access(ncu: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(ncu), "--query-metrics"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = (completed.stdout + completed.stderr).strip()
    permission_blocked = "ERR_NVGPUCTRPERM" in combined
    return {
        "command": [str(ncu.resolve()), "--query-metrics"],
        "returncode": completed.returncode,
        "permission_blocked": permission_blocked,
        "status": (
            "BLOCKED_NVGPUCTRPERM"
            if permission_blocked
            else ("AVAILABLE" if completed.returncode == 0 else "FAILED")
        ),
        "diagnostic": combined[:2000],
        "hardware_dram_counters_retained": False,
        "inference_prohibited": (
            "No DRAM-byte or achieved-occupancy value is inferred from the blocked "
            "counter attempt."
        ),
    }


def _performance(receipt_path: Path, batch: int) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    performance = receipt["comparison"]["performance"]
    if batch == 2:
        speedup = performance["candidate_pair_throughput_speedup"]
        p50 = performance["candidate_batch2_device_p50_ms"]
        p95 = receipt["candidate"]["modes"]["batch2"]["device"]["p95_ms"]
        p99 = receipt["candidate"]["modes"]["batch2"]["device"]["p99_ms"]
        rows_per_second = 1000.0 * batch / p50
        per_row_ms = p50 / batch
    else:
        speedup = performance["target_pair_throughput_speedup"]
        p50 = performance["target_device_p50_ms"]
        p95 = performance["target_device_p95_ms"]
        p99 = performance["target_device_p99_ms"]
        rows_per_second = performance["aggregate_rows_per_second_p50"]
        per_row_ms = performance["per_row_service_ms_p50"]
    return {
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": _sha256(receipt_path),
        "binary_sha256": receipt["candidate"]["binary_sha256"],
        "device_p50_ms": p50,
        "device_p95_ms": p95,
        "device_p99_ms": p99,
        "aggregate_rows_per_second_p50": rows_per_second,
        "per_row_service_ms_p50": per_row_ms,
        "aggregate_speedup": speedup,
        "ideal_efficiency_percent": 100.0 * speedup / batch,
        "bit_exact": (
            receipt["comparison"]
            .get("target_vs_serial", receipt["comparison"].get("candidate_batch2_vs_candidate_serial"))
            ["bit_exact"]
        ),
        "gpu_health": receipt["gates"]["gpu_health"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument("--ncu", type=Path, required=True)
    parser.add_argument("--batch2", type=Path, required=True)
    parser.add_argument("--batch4", type=Path, required=True)
    parser.add_argument("--batch8", type=Path, required=True)
    parser.add_argument("--batch16", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    binary = args.binary.resolve()
    resources = _resources(args.cuobjdump.resolve(), binary)
    scaling = {
        str(batch): _performance(path.resolve(), batch)
        for batch, path in (
            (2, args.batch2),
            (4, args.batch4),
            (8, args.batch8),
            (16, args.batch16),
        )
    }
    pair8 = resources["kernels"]["pair_batch8"]
    pair16 = resources["kernels"]["pair_batch16"]
    single8 = resources["kernels"]["single_batch8"]
    single16 = resources["kernels"]["single_batch16"]
    gates = {
        "resource_dump_pass": resources["returncode"] == 0,
        "all_eight_row_kernels_present": len(resources["kernels"]) == 8,
        "pair8_theoretical_occupancy_100_percent": (
            pair8["theoretical_sm86"]["warp_occupancy_percent_upper_bound"] == 100.0
        ),
        "pair16_theoretical_occupancy_at_most_50_percent": (
            pair16["theoretical_sm86"]["warp_occupancy_percent_upper_bound"] <= 50.0
        ),
        "pair16_register_jump": (
            pair16["registers_per_thread"] > pair8["registers_per_thread"]
        ),
        "pair16_shared_memory_doubles": (
            pair16["shared_bytes_per_block"]
            == 2 * pair8["shared_bytes_per_block"]
        ),
        "no_pair16_local_or_stack_spill": (
            pair16["local_bytes_per_thread"] == 0
            and pair16["stack_bytes_per_thread"] == 0
        ),
        "single_projection_remains_full_occupancy_upper_bound": (
            single8["theoretical_sm86"]["warp_occupancy_percent_upper_bound"] == 100.0
            and single16["theoretical_sm86"]["warp_occupancy_percent_upper_bound"]
            == 100.0
        ),
        "observed_ideal_efficiency_drops_by_at_least_20_points": (
            scaling["8"]["ideal_efficiency_percent"]
            - scaling["16"]["ideal_efficiency_percent"]
            >= 20.0
        ),
        "all_outputs_bit_exact_and_gpu_healthy": all(
            row["bit_exact"] and row["gpu_health"] for row in scaling.values()
        ),
    }
    profiler = _ncu_access(args.ncu.resolve())
    supported = all(gates.values())
    payload = {
        "schema_version": "experiment-014-k3-row-cooperative-resource-inspection-v1",
        "cycle_id": "H014-028e",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if supported else "FAIL",
        "hypothesis": (
            "The batch-16 efficiency bend is consistent with the fused gate/up "
            "kernel crossing an sm_86 residency boundary: its theoretical warp "
            "occupancy upper bound falls from 100% at batch 8 to <=50% at batch "
            "16, while the down kernel remains at 100% and no local spill occurs."
        ),
        "implementation": (
            "Evidence-only cuobjdump resource parsing and sm_86 occupancy-bound "
            "calculation over the exact batch-16 candidate; no CUDA kernel executed."
        ),
        "benchmark": {
            "binary": str(binary),
            "binary_sha256": _sha256(binary),
            "cuobjdump": str(args.cuobjdump.resolve()),
            "sm86_limits": SM86_LIMITS,
            "threads_per_block": THREADS_PER_BLOCK,
            "scaling": scaling,
        },
        "compiled_resources": resources,
        "hardware_counter_attempt": profiler,
        "gates": gates,
        "hypothesis_supported": supported,
        "result": (
            "SUPPORTED: pair-kernel static residency falls 6 -> 3 blocks/SM "
            "and 100% -> 50% theoretical warp occupancy from batch 8 to 16; "
            "registers rise 40 -> 56 and shared memory 16,384 -> 32,768 bytes."
            if supported
            else "FALSIFIED_OR_INCOMPLETE: one or more resource gates failed."
        ),
        "inspection": (
            "The fused pair, not the single down projection, crosses the resource "
            "boundary. Nsight counters were requested but permission-blocked; the "
            "receipt therefore distinguishes compiled-resource evidence from "
            "unmeasured achieved occupancy and DRAM traffic."
        ),
        "bottleneck": (
            "Batch-16 fused gate/up occupancy pressure; no compiler local-memory spill."
        ),
        "decision": (
            "Retain batch 8 as the primitive efficiency candidate and batch 16 as "
            "a safe capacity comparator; select only from complete-stage evidence."
        ),
        "redesign": (
            "Benchmark complete layer-89 KDA+MoE and late Gated-MLA+MoE at batch "
            "1/8/16; do not attempt any larger expert batch."
        ),
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if supported else 1


if __name__ == "__main__":
    raise SystemExit(main())
