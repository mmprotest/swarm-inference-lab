from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_inference.experiments.experiment_015.dspark_benchmark import (
    benchmark_dspark_reference,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 3, 5, 7])
    args = parser.parse_args()
    receipt = benchmark_dspark_reference(
        args.target,
        args.draft,
        args.trace,
        args.output,
        block_sizes=tuple(args.blocks),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
