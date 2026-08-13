"""Run one Experiment 020 physical RTX 5090 evidence arm."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from swarm_inference.experiments.experiment_019.other_physical import (
    benchmark_other_shards,
)
from swarm_inference.experiments.experiment_020.expert_grouped import (
    benchmark_grouped_expert_stripe,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("expert-grouped", "other"))
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--degree", type=int, choices=(8, 16), default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=31)
    parser.add_argument("--repeats", type=int, default=15)
    arguments = parser.parse_args()
    if arguments.arm == "expert-grouped":
        receipt = benchmark_grouped_expert_stripe(
            arguments.checkpoint.resolve(),
            arguments.cuda_library.resolve(),
            arguments.shard_library.resolve(),
            arguments.grouped_library.resolve(),
            (arguments.oracle_root / "hidden-trace.f32").resolve(),
            (arguments.oracle_root / "routes.txt").resolve(),
            degree=arguments.degree,
            warmup=arguments.warmup,
            iterations=arguments.iterations,
        )
    else:
        receipt = benchmark_other_shards(
            arguments.checkpoint.resolve(),
            arguments.cuda_library.resolve(),
            arguments.shard_library.resolve(),
            (arguments.oracle_root / "hidden-trace.f32").resolve(),
            degrees=(arguments.degree,),
            rows_sweep=(1,),
            repeats=arguments.repeats,
        )
    _write(arguments.output, receipt)
    print(json.dumps({"arm": arguments.arm, "status": receipt["status"]}))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
