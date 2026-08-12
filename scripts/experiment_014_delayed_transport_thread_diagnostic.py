"""Retain H014-038p delayed native-thread isolation for Kimi transport."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKER = r"""
import json
import sys
import threading
import time

import psutil
import torch

from swarm_inference.transport.stage_tensor import pack_tensor, unpack_tensor

mode = sys.argv[1]
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
process = psutil.Process()

def snapshot():
    return {
        "os": sorted(item.id for item in process.threads()),
        "python_native": sorted(
            int(item.native_id)
            for item in threading.enumerate()
            if item.native_id is not None
        ),
    }

before = snapshot()
bit_exact = True
raw_bytes = None
encoded_bytes = None
if mode == "transport":
    source = torch.zeros((1, 9, 7168), dtype=torch.float32)
    for _ in range(250):
        packed = pack_tensor(source, requested_mode="none")
        restored, _ = unpack_tensor(packed.payload, packed.attributes())
        bit_exact = bit_exact and torch.equal(source, restored)
    raw_bytes = packed.raw_bytes
    encoded_bytes = packed.encoded_bytes
immediate = snapshot()
transitions = []
previous = immediate
for sample in range(1, 21):
    time.sleep(0.05)
    current = snapshot()
    if current != previous:
        transitions.append({
            "sample": sample,
            "elapsed_ms": sample * 50,
            "os_added": sorted(set(current["os"]) - set(previous["os"])),
            "os_removed": sorted(set(previous["os"]) - set(current["os"])),
            "python_native_added": sorted(
                set(current["python_native"]) - set(previous["python_native"])
            ),
            "python_native_removed": sorted(
                set(previous["python_native"]) - set(current["python_native"])
            ),
        })
    previous = current
print(json.dumps({
    "mode": mode,
    "torch_intraop_threads": torch.get_num_threads(),
    "torch_interop_threads": torch.get_num_interop_threads(),
    "before": before,
    "immediate": immediate,
    "after": previous,
    "immediate_os_added": sorted(set(immediate["os"]) - set(before["os"])),
    "delayed_os_added": sorted(set(previous["os"]) - set(immediate["os"])),
    "delayed_python_native_added": sorted(
        set(previous["python_native"]) - set(immediate["python_native"])
    ),
    "transitions": transitions,
    "bit_exact": bit_exact,
    "round_trips": 250 if mode == "transport" else 0,
    "raw_bytes": raw_bytes,
    "encoded_bytes": encoded_bytes,
}, sort_keys=True))
"""


def _run(mode: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-c", WORKER, mode],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    idle = _run("idle")
    transport = _run("transport")
    reproduced = (
        len(transport["delayed_os_added"]) == 1
        and not transport["delayed_python_native_added"]
        and not idle["delayed_os_added"]
    )
    gates = {
        "cpu_pool_1_by_1": all(
            row["torch_intraop_threads"] == 1
            and row["torch_interop_threads"] == 1
            for row in (idle, transport)
        ),
        "canonical_transport_bit_exact": (
            transport["bit_exact"] is True
            and transport["raw_bytes"] == 258_048
            and transport["encoded_bytes"] == 258_048
        ),
        "idle_control_stable": not idle["delayed_os_added"],
    }
    result = {
        "schema_version": "experiment-014-h014-038p-delayed-transport-thread-v1",
        "cycle_id": "H014-038p",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "hypothesis": (
            "The 1/1 canonical pack/unpack path asynchronously creates exactly one "
            "persistent non-Python OS thread during the next one second."
        ),
        "hypothesis_reproduced": reproduced,
        "idle_control": idle,
        "canonical_transport": transport,
        "acceptance_gates": gates,
        "decision": (
            "TRIGGER_REPRODUCED"
            if reproduced
            else "HYPOTHESIS_FALSIFIED_INSTRUMENT_END_TO_END_PHASES"
        ),
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
