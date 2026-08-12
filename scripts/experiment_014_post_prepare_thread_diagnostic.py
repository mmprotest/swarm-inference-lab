"""Retain H014-038s post-PREPARE versus route-trigger attribution."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from experiment_014_readiness_phase_diagnostic import _gpu_health, _snapshot

from swarm_inference.execution.kimi_k3_stage import _benchmark_persistent_final_stage


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    phase_records: list[dict[str, Any]] = []

    async def observe(phase: str) -> None:
        sample_count = 30 if phase == "load_prepare_complete" else 10
        before = _snapshot()
        previous = before
        transitions: list[dict[str, Any]] = []
        for sample in range(1, sample_count + 1):
            await asyncio.sleep(0.05)
            current = _snapshot()
            if current["os_thread_ids"] != previous["os_thread_ids"]:
                added = set(current["os_thread_ids"]) - set(previous["os_thread_ids"])
                transitions.append(
                    {
                        "sample": sample,
                        "elapsed_ms": sample * 50,
                        "added_os_thread_ids": sorted(added),
                        "removed_os_thread_ids": sorted(
                            set(previous["os_thread_ids"])
                            - set(current["os_thread_ids"])
                        ),
                        "added_python_native_thread_ids": sorted(
                            set(current["python_native_thread_ids"])
                            - set(previous["python_native_thread_ids"])
                        ),
                        "added_thread_metadata": [
                            row
                            for row in current["os_thread_metadata"]
                            if row["thread_id"] in added
                        ],
                    }
                )
            previous = current
        phase_records.append(
            {
                "phase": phase,
                "observation_ms": sample_count * 50,
                "sample_interval_ms": 50,
                "before_os_thread_count": len(before["os_thread_ids"]),
                "after_os_thread_count": len(previous["os_thread_ids"]),
                "transitions": transitions,
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
        cycle_id="H014-038s",
        readiness_phase_observer=observe,
    )
    health_after = _gpu_health()
    transitions = [
        {"phase": row["phase"], **transition}
        for row in phase_records
        for transition in row["transitions"]
    ]
    first = transitions[0] if transitions else None
    first_is_non_python_singleton = bool(
        first is not None
        and len(first["added_os_thread_ids"]) == 1
        and not first["added_python_native_thread_ids"]
    )
    later_transitions = [
        row for row in transitions if row["phase"] != "load_prepare_complete"
    ]
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
        "schema_version": "experiment-014-h014-038s-post-prepare-thread-v1",
        "cycle_id": "H014-038s",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "hypothesis": (
            "The singleton is delayed work from the seventh PREPARE execution and "
            "appears before any route is installed."
        ),
        "hypothesis_reproduced": bool(
            first is not None
            and first["phase"] == "load_prepare_complete"
            and first["elapsed_ms"] > 500
            and first_is_non_python_singleton
            and not later_transitions
        ),
        "first_transition": first,
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
                "first_transition": result["first_transition"],
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
