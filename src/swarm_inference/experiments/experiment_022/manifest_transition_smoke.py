"""Physical lifecycle smoke for consecutive E022 manifest layer types."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _numerical_metrics

from .io import atomic_write_json
from .manifest_correctness import HIDDEN, LAYERS, RELATIVE_L2_GATE, ManifestK3Runner


def run(
    *,
    layers: Sequence[int],
    manifest_path: Path,
    checkpoint: Path,
    cuda_library: Path,
    grouped_library: Path,
    shard_library: Path,
    oracle_root: Path,
    state_reference_root: Path,
) -> dict[str, Any]:
    requested = [int(layer) for layer in layers]
    if len(requested) < 2 or any(not 1 <= layer < LAYERS for layer in requested):
        raise ValueError("transition smoke requires at least two layers in 1..92")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(
        (state_reference_root / "metadata.json").read_text(encoding="utf-8")
    )
    reference_layers = {int(row["layer"]): row for row in metadata["layers"]}
    trace = np.memmap(
        oracle_root / "hidden-trace.f32",
        mode="r",
        dtype="<f4",
        shape=(3 * (LAYERS + 1), HIDDEN),
    )
    runner = ManifestK3Runner(
        checkpoint,
        cuda_library,
        grouped_library,
        manifest,
        shard_library=shard_library,
        state_reference_root=state_reference_root,
    )
    rows: list[dict[str, Any]] = []
    close_error: str | None = None
    started = time.perf_counter_ns()
    try:
        for layer in requested:
            previous = reference_layers[layer - 1]
            compact = np.load(
                state_reference_root / previous["attnres"]["path"],
                allow_pickle=False,
            )
            block_residuals = np.zeros((1, 8, HIDDEN), dtype=np.float32)
            block_residuals[:, : compact.shape[1]] = compact
            hidden = np.ascontiguousarray(
                trace[layer - 1 : layer], dtype=np.float32
            )
            expected = np.ascontiguousarray(trace[layer : layer + 1], dtype=np.float32)
            layer_started = time.perf_counter_ns()
            output, next_count, evidence = runner.execute_layer(
                layer,
                hidden,
                block_residuals,
                int(compact.shape[1]),
                [0],
                maximum_context=256,
            )
            # This is the exact query that failed when a nested graph shut down
            # the process-global native runtime after layer 71.
            memory_after = runner.runtime.mem_info()
            metrics = _numerical_metrics(expected, output)
            state = runner.state_reference_receipts[-1]
            assignment = runner.assignments[layer]
            rows.append(
                {
                    "layer": layer,
                    "candidate_id": assignment["candidate_id"],
                    "partition_type": assignment["partition_type"],
                    "output_metrics": metrics,
                    "state_status": state["status"],
                    "next_block_count": next_count,
                    "outer_cuda_mem_info_after_layer": memory_after,
                    "native_primitive": evidence["native_primitive"],
                    "wall_ms": (time.perf_counter_ns() - layer_started) / 1e6,
                }
            )
    finally:
        try:
            runner.close()
        except BaseException as exc:  # pragma: no cover - physical failure receipt
            close_error = f"{type(exc).__name__}: {exc}"
    timed_reads = sum(
        int(row["checkpoint_reads_in_timed_region"])
        for row in [
            *runner.expert_worker_receipts,
            *runner.full_mixed_worker_receipts,
        ]
    )
    passed = (
        len(rows) == len(requested)
        and all(
            float(row["output_metrics"]["relative_l2_error"])
            <= RELATIVE_L2_GATE
            and row["state_status"] == "PASS"
            for row in rows
        )
        and timed_reads == 0
        and close_error is None
    )
    return {
        "schema_version": "experiment-022-manifest-transition-smoke-v1",
        "status": "PASS" if passed else "FAIL",
        "evidence_class": "PHYSICAL RTX 5090 consecutive manifest layers",
        "inventory_id": manifest["inventory_id"],
        "planner_level": manifest["planner_level"],
        "layers": rows,
        "timed_checkpoint_reads": timed_reads,
        "outer_authenticated_roundtrips": len(runner.dispatch_receipts),
        "close_error": close_error,
        "wall_seconds": (time.perf_counter_ns() - started) / 1e9,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--grouped-library", type=Path, required=True)
    parser.add_argument("--shard-library", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--state-reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run(
        layers=args.layer,
        manifest_path=args.manifest.resolve(),
        checkpoint=args.checkpoint.resolve(),
        cuda_library=args.cuda_library.resolve(),
        grouped_library=args.grouped_library.resolve(),
        shard_library=args.shard_library.resolve(),
        oracle_root=args.oracle_root.resolve(),
        state_reference_root=args.state_reference_root.resolve(),
    )
    atomic_write_json(args.output.resolve(), receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "layers": [
                    [row["layer"], row["partition_type"]]
                    for row in receipt["layers"]
                ],
                "timed_checkpoint_reads": receipt["timed_checkpoint_reads"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run"]
