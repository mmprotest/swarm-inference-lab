"""Generate the complete Experiment 020 evidence package and report."""

from __future__ import annotations

import json
from pathlib import Path

from swarm_inference.experiments.experiment_020.finalize import (
    finalize_experiment_020,
)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    summary = finalize_experiment_020(repo)
    print(
        json.dumps(
            {
                "classification": summary["classification"],
                "any_gpu_rented": summary["any_gpu_rented"],
                "any_vast_resource_mutated": summary["any_vast_resource_mutated"],
                "predicted_block16_chunk1_tok_s": summary[
                    "predicted_block16_chunk1_tok_s"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["classification"] == "E021_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
