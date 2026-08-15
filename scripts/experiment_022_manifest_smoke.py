"""Developer-only manifest layer smoke; never enters E022 headline statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from swarm_inference.experiments.experiment_022.manifest_correctness import (
    ManifestK3Runner,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--grouped-library", type=Path, required=True)
    parser.add_argument("--oracle-trace", type=Path, required=True)
    parser.add_argument("--state-reference-root", type=Path)
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    runner = ManifestK3Runner(
        args.checkpoint,
        args.cuda_library,
        args.grouped_library,
        manifest,
        state_reference_root=args.state_reference_root,
    )
    try:
        trace = np.memmap(
            args.oracle_trace, mode="r", dtype="<f4", shape=(282, 7168)
        )
        embedding, _ = runner.embed([163584])
        hidden = np.ascontiguousarray(trace[args.layer - 1][None, :])
        residuals = np.zeros((1, 8, 7168), dtype=np.float32)
        snapshots = list(range(0, args.layer, 12))
        for slot, snapshot in enumerate(snapshots):
            residuals[0, slot] = (
                embedding[0] if snapshot == 0 else trace[snapshot - 1]
            )
        output, _count, evidence = runner.execute_layer(
            args.layer,
            hidden,
            residuals,
            len(snapshots),
            [0],
            maximum_context=256,
        )
        reference = np.ascontiguousarray(trace[args.layer][None, :])
        relative_l2 = float(
            np.linalg.norm(output.astype(np.float64) - reference.astype(np.float64))
            / np.linalg.norm(reference.astype(np.float64))
        )
        result = {
            "status": "PASS" if relative_l2 <= 2e-5 else "FAIL",
            "layer": args.layer,
            "partition": evidence["manifest_assignment"]["partition_type"],
            "degree": evidence["manifest_assignment"]["degree"],
            "relative_l2": relative_l2,
            "expert_worker_receipts": len(runner.expert_worker_receipts),
            "all_complete_banks": all(
                row["complete_expert_bank_resident"]
                for row in runner.expert_worker_receipts
            ),
            "timed_checkpoint_reads": sum(
                row["checkpoint_reads_in_timed_region"]
                for row in runner.expert_worker_receipts
            ),
            "state_reference": (
                runner.state_reference_receipts[-1]
                if runner.state_reference_receipts
                else None
            ),
        }
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 1
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
