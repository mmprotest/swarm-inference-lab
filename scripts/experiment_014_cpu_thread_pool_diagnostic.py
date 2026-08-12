"""Retain the H014-038m PyTorch transport thread/latency diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WORKER = r"""
import json
import statistics
import sys
import time

import psutil
import torch

from swarm_inference.transport.stage_tensor import pack_tensor, unpack_tensor

mode = sys.argv[1]
process = psutil.Process()
thread_ids = lambda: {item.id for item in process.threads()}
before = thread_ids()
if mode == "single":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
configured = thread_ids()
source = torch.zeros((1, 9, 7168), dtype=torch.float32)
timings = []
bit_exact = True
first_started = time.perf_counter_ns()
packed = pack_tensor(source, requested_mode="none")
restored, _ = unpack_tensor(packed.payload, packed.attributes())
first_ns = time.perf_counter_ns() - first_started
bit_exact = bit_exact and torch.equal(source, restored)
after_first = thread_ids()
for _ in range(250):
    started = time.perf_counter_ns()
    packed = pack_tensor(source, requested_mode="none")
    restored, _ = unpack_tensor(packed.payload, packed.attributes())
    timings.append((time.perf_counter_ns() - started) / 1e6)
    bit_exact = bit_exact and torch.equal(source, restored)
after = thread_ids()
ordered = sorted(timings)
percentile = lambda fraction: ordered[round((len(ordered) - 1) * fraction)]
print(json.dumps({
    "mode": mode,
    "torch_intraop_threads": torch.get_num_threads(),
    "torch_interop_threads": torch.get_num_interop_threads(),
    "thread_count_before": len(before),
    "thread_count_after_configuration": len(configured),
    "thread_count_after_first_round_trip": len(after_first),
    "thread_count_after": len(after),
    "threads_added_by_configuration": sorted(configured - before),
    "threads_added_by_first_round_trip": sorted(after_first - configured),
    "threads_added_total": sorted(after - configured),
    "first_round_trip_ms": first_ns / 1e6,
    "steady_round_trips": len(timings),
    "steady_p50_ms": percentile(0.50),
    "steady_p95_ms": percentile(0.95),
    "steady_p99_ms": percentile(0.99),
    "steady_mean_ms": statistics.fmean(timings),
    "bit_exact": bit_exact,
    "raw_bytes": packed.raw_bytes,
    "encoded_bytes": packed.encoded_bytes,
}, sort_keys=True))
"""


def _run(mode: str) -> dict[str, object]:
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
    default = _run("default")
    single = _run("single")
    ratio = float(single["steady_p50_ms"]) / float(default["steady_p50_ms"])
    gates = {
        "canonical_size": (
            default["raw_bytes"] == single["raw_bytes"] == 258_048
            and default["encoded_bytes"] == single["encoded_bytes"] == 258_048
        ),
        "bit_exact": default["bit_exact"] is True and single["bit_exact"] is True,
        "default_pool_is_20_by_20": (
            default["torch_intraop_threads"] == 20
            and default["torch_interop_threads"] == 20
        ),
        "default_first_round_trip_created_native_threads": len(
            default["threads_added_by_first_round_trip"]
        )
        > 0,
        "single_pool_is_1_by_1": (
            single["torch_intraop_threads"] == 1
            and single["torch_interop_threads"] == 1
        ),
        "single_round_trips_created_no_threads": not single["threads_added_total"],
        "single_transport_p50_within_10_percent": ratio <= 1.10,
    }
    result = {
        "schema_version": "experiment-014-h014-038m-cpu-thread-pool-v1",
        "cycle_id": "H014-038m",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "hypothesis": (
            "A single-thread PyTorch CPU transport contract prevents lazy native "
            "pool growth without degrading canonical boundary p50 by more than 10%."
        ),
        "default": default,
        "single_thread": single,
        "single_to_default_p50_ratio": ratio,
        "acceptance_gates": gates,
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
