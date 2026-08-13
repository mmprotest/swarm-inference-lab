"""Run E022 full 93-layer correctness through a production worker process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_inference.experiments.experiment_022.io import atomic_write_json
from swarm_inference.experiments.experiment_022.worker_correctness import (
    run_worker_process_correctness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("F:/models/Kimi-K3"))
    parser.add_argument("--cuda-library", type=Path, default=Path("artifacts/experiment-016/cuda/coli_cuda-sm120-h016-final.dll"))
    parser.add_argument("--shard-library", type=Path, default=Path("artifacts/experiment-019/physical/exp019-kda-shard-sm120-v2.dll"))
    parser.add_argument("--grouped-library", type=Path, default=Path("artifacts/experiment-020/physical/e020-grouped-top16-sm120.dll"))
    parser.add_argument("--oracle-root", type=Path, default=Path("artifacts/experiment-014/oracle-full-93-idot0"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    arguments = parser.parse_args()
    receipt = run_worker_process_correctness(
        arguments.checkpoint,
        arguments.cuda_library,
        arguments.shard_library,
        arguments.grouped_library,
        arguments.oracle_root,
        arguments.raw_output,
        timeout_seconds=arguments.timeout_seconds,
    )
    atomic_write_json(arguments.output, receipt)
    print(json.dumps({"status": receipt["status"], "wall_seconds": receipt["wall_seconds"]}))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
