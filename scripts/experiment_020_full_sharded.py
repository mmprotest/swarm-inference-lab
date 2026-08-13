"""Execute the complete frozen P=8/depth=8 E021 candidate on the local RTX 5090."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from swarm_inference.experiments.experiment_020.sharded_graph import certify_full_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("F:/models/Kimi-K3"))
    parser.add_argument(
        "--cuda-library",
        type=Path,
        default=Path("artifacts/experiment-016/cuda/coli_cuda-sm120-h016-final.dll"),
    )
    parser.add_argument(
        "--shard-library",
        type=Path,
        default=Path("artifacts/experiment-019/physical/exp019-kda-shard-sm120-v2.dll"),
    )
    parser.add_argument(
        "--grouped-library",
        type=Path,
        default=Path("artifacts/experiment-020/physical/e020-grouped-top16-sm120.dll"),
    )
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=Path("artifacts/experiment-014/oracle-full-93-idot0"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiment-020/correctness/full-93-sharded.json"),
    )
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    result = certify_full_candidate(
        arguments.checkpoint.resolve(),
        arguments.cuda_library.resolve(),
        arguments.shard_library.resolve(),
        arguments.grouped_library.resolve(),
        arguments.oracle_root.resolve(),
        progress=not arguments.quiet,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, arguments.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "layers": result["executed_layers"],
                "maximum_relative_l2_error": result["maximum_relative_l2_error"],
                "routes_exact": result["routes_exact"],
                "greedy_token_match": result["head"]["greedy_token_match"],
            }
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
