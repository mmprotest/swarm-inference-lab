"""Generate Experiment 015 artifacts without running any physical fleet work."""

from __future__ import annotations

import argparse
from pathlib import Path

from swarm_inference.experiments.experiment_015.contracts import EconomicsConfig
from swarm_inference.experiments.experiment_015.finalize import finalize_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--gpu-hourly-price-usd", type=float, default=0.15)
    parser.add_argument("--output-price-per-million-usd", type=float, default=15.0)
    parser.add_argument("--target-gpu-margin-fraction", type=float, default=0.50)
    arguments = parser.parse_args()
    summary = finalize_experiment(
        arguments.repository_root,
        economics=EconomicsConfig(
            gpu_hourly_price_usd=arguments.gpu_hourly_price_usd,
            output_price_per_million_usd=arguments.output_price_per_million_usd,
            target_gpu_margin_fraction=arguments.target_gpu_margin_fraction,
        ),
    )
    print(summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
