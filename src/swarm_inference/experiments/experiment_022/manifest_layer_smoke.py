"""Focused physical smoke test for one layer of an E022 placement manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _numerical_metrics

from .io import atomic_write_json
from .manifest_correctness import HIDDEN, LAYERS, RELATIVE_L2_GATE, ManifestK3Runner


def run(
    *,
    layer: int,
    manifest_path: Path,
    checkpoint: Path,
    cuda_library: Path,
    grouped_library: Path,
    shard_library: Path,
    oracle_root: Path,
    state_reference_root: Path,
) -> dict[str, Any]:
    if not 1 <= layer < LAYERS:
        raise ValueError("focused manifest smoke requires layer 1..92")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(
        (state_reference_root / "metadata.json").read_text(encoding="utf-8")
    )
    previous = next(
        row for row in metadata["layers"] if int(row["layer"]) == layer - 1
    )
    compact = np.load(
        state_reference_root / previous["attnres"]["path"], allow_pickle=False
    )
    block_residuals = np.zeros((1, 8, HIDDEN), dtype=np.float32)
    block_residuals[:, : compact.shape[1]] = compact
    trace = np.memmap(
        oracle_root / "hidden-trace.f32",
        mode="r",
        dtype="<f4",
        shape=(3 * (LAYERS + 1), HIDDEN),
    )
    hidden = np.ascontiguousarray(trace[layer - 1 : layer], dtype=np.float32)
    expected = np.ascontiguousarray(trace[layer : layer + 1], dtype=np.float32)
    runner = ManifestK3Runner(
        checkpoint,
        cuda_library,
        grouped_library,
        manifest,
        shard_library=shard_library,
        state_reference_root=state_reference_root,
    )
    closed = False
    try:
        output, next_count, evidence = runner.execute_layer(
            layer,
            hidden,
            block_residuals,
            int(compact.shape[1]),
            [0],
            maximum_context=256,
        )
        state = runner.state_reference_receipts[-1]
        metrics = _numerical_metrics(expected, output)
        runner.close()
        closed = True
        expert_rows = runner.expert_worker_receipts
        mixed_rows = runner.full_mixed_worker_receipts
        timed_reads = sum(
            int(row["checkpoint_reads_in_timed_region"])
            for row in [*expert_rows, *mixed_rows]
        )
        assignment = runner.assignments[layer]
        expected_inner = (
            int(assignment["degree"])
            if assignment["partition_type"] in {"EXPERT_SHARD", "WHOLE_EXPERT"}
            else 0
        )
        inner_audits = sum(
            len(row.get("dispatcher_audit", ())) for row in expert_rows
        )
        passed = (
            float(metrics["relative_l2_error"]) <= RELATIVE_L2_GATE
            and state["status"] == "PASS"
            and timed_reads == 0
            and len(runner.dispatch_receipts) == 1
            and inner_audits == expected_inner
            and all(row.get("whole_layer_fallback") is False for row in expert_rows)
            and all(row.get("whole_layer_fallback") is False for row in mixed_rows)
            and (
                runner.expert_worker_process_audit["close_status"] == "PASS"
                if expected_inner
                else True
            )
        )
        return {
            "schema_version": "experiment-022-manifest-layer-smoke-v1",
            "status": "PASS" if passed else "FAIL",
            "evidence_class": "PHYSICAL RTX 5090 single selected manifest layer",
            "inventory_id": manifest["inventory_id"],
            "planner_level": manifest["planner_level"],
            "layer": layer,
            "candidate_id": assignment["candidate_id"],
            "partition_type": assignment["partition_type"],
            "degree": assignment["degree"],
            "output_metrics": metrics,
            "state_status": state["status"],
            "attention_state_relative_l2": state[
                "attention_state_relative_l2"
            ],
            "attnres_relative_l2": state["attnres_metrics"][
                "relative_l2_error"
            ],
            "next_block_count": next_count,
            "timed_checkpoint_reads": timed_reads,
            "outer_authenticated_roundtrips": len(runner.dispatch_receipts),
            "inner_authenticated_roundtrips": inner_audits,
            "expert_worker_receipts": expert_rows,
            "full_mixed_worker_receipts": mixed_rows,
            "persistent_worker": runner.expert_worker_process_audit,
            "evidence": evidence,
        }
    finally:
        if not closed:
            runner.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
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
        layer=args.layer,
        manifest_path=args.manifest.resolve(),
        checkpoint=args.checkpoint.resolve(),
        cuda_library=args.cuda_library.resolve(),
        grouped_library=args.grouped_library.resolve(),
        shard_library=args.shard_library.resolve(),
        oracle_root=args.oracle_root.resolve(),
        state_reference_root=args.state_reference_root.resolve(),
    )
    atomic_write_json(args.output.resolve(), receipt)
    print(json.dumps({
        "status": receipt["status"],
        "layer": receipt["layer"],
        "partition_type": receipt["partition_type"],
        "output_relative_l2": receipt["output_metrics"]["relative_l2_error"],
        "timed_checkpoint_reads": receipt["timed_checkpoint_reads"],
    }, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run"]
