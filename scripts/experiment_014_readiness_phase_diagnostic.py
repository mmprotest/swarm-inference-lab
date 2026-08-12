"""Retain H014-038r pre-compute readiness lifecycle thread attribution."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import psutil

from swarm_inference.execution.kimi_k3_stage import (
    _benchmark_persistent_final_stage,
    _process_thread_metadata,
)


def _snapshot() -> dict[str, Any]:
    process_threads = psutil.Process().threads()
    return {
        "os_thread_ids": sorted(item.id for item in process_threads),
        "os_thread_metadata": _process_thread_metadata(process_threads),
        "python_native_thread_ids": sorted(
            int(item.native_id)
            for item in threading.enumerate()
            if item.native_id is not None
        ),
    }


def _gpu_health() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,compute_cap,memory.total,memory.free,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return {"status": "UNAVAILABLE", "stderr": completed.stderr.strip()}
    names = (
        "index",
        "name",
        "uuid",
        "compute_capability",
        "memory_total_mib",
        "memory_free_mib",
        "temperature_c",
    )
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(values) != len(names):
        return {"status": "UNPARSEABLE", "stdout": completed.stdout.strip()}
    return {"status": "MEASURED", **dict(zip(names, values, strict=True))}


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    phase_records: list[dict[str, Any]] = []

    async def observe(phase: str) -> None:
        before = _snapshot()
        await asyncio.sleep(0.5)
        after = _snapshot()
        added = set(after["os_thread_ids"]) - set(before["os_thread_ids"])
        phase_records.append(
            {
                "phase": phase,
                "observation_ms": 500,
                "before_os_thread_count": len(before["os_thread_ids"]),
                "after_os_thread_count": len(after["os_thread_ids"]),
                "added_os_thread_ids": sorted(added),
                "removed_os_thread_ids": sorted(
                    set(before["os_thread_ids"]) - set(after["os_thread_ids"])
                ),
                "added_python_native_thread_ids": sorted(
                    set(after["python_native_thread_ids"])
                    - set(before["python_native_thread_ids"])
                ),
                "added_thread_metadata": [
                    row
                    for row in after["os_thread_metadata"]
                    if row["thread_id"] in added
                ],
            }
        )

    health_before = _gpu_health()
    benchmark = await _benchmark_persistent_final_stage(
        args.checkpoint,
        args.cuda_library,
        args.oracle_trace,
        args.oracle_routes,
        args.oracle_logits,
        registered=True,
        identity_manifest=args.identity_manifest,
        cycle_id="H014-038r",
        readiness_phase_observer=observe,
    )
    health_after = _gpu_health()
    first_added = next(
        (row for row in phase_records if row["added_os_thread_ids"]),
        None,
    )
    trigger_phase = first_added["phase"] if first_added is not None else None
    trigger_is_non_python_singleton = bool(
        first_added is not None
        and len(first_added["added_os_thread_ids"]) == 1
        and not first_added["added_python_native_thread_ids"]
    )
    gates = {
        "correctness_pass": benchmark["correctness"]["pass"] is True,
        "compute_thread_consistent": benchmark["lifecycle"][
            "compute_thread_consistent"
        ]
        is True,
        "all_expected_phases_observed": [row["phase"] for row in phase_records]
        == [
            "load_prepare_complete",
            "route_installed",
            "warmup_session_opened",
            "warmup_message_built",
        ],
        "gpu_identity_stable": (
            health_before.get("status") == "MEASURED"
            and health_after.get("status") == "MEASURED"
            and health_before.get("uuid") == health_after.get("uuid")
        ),
    }
    return {
        "schema_version": "experiment-014-h014-038r-readiness-phase-thread-v1",
        "cycle_id": "H014-038r",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "hypothesis": (
            "The seventh assigned-stage PREPARE call asynchronously schedules the "
            "singleton, which appears in the first 500 ms after load/PREPARE."
        ),
        "hypothesis_reproduced": (
            trigger_phase == "load_prepare_complete"
            and trigger_is_non_python_singleton
        ),
        "first_added_thread_phase": trigger_phase,
        "first_added_thread_is_non_python_singleton": trigger_is_non_python_singleton,
        "phase_records": phase_records,
        "benchmark": benchmark,
        "gpu_health_before": health_before,
        "gpu_health_after": health_after,
        "acceptance_gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--oracle-trace", type=Path, required=True)
    parser.add_argument("--oracle-routes", type=Path, required=True)
    parser.add_argument("--oracle-logits", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(_run(args))
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(
        json.dumps(
            {
                "status": result["status"],
                "hypothesis_reproduced": result["hypothesis_reproduced"],
                "first_added_thread_phase": result["first_added_thread_phase"],
                "benchmark_status": result["benchmark"]["status"],
                "output": str(destination),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
