from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_inference.experiments.experiment_025.preflight import (
    build_zero_spend_preflight,
    create_run_layout,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the zero-spend E025 preflight")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, default=Path(r"F:\models\Kimi-K3"))
    parser.add_argument(
        "--source-placement",
        type=Path,
        default=Path(
            "artifacts/experiment-014/deployment/h014-037a-final-physical-placement.json"
        ),
    )
    parser.add_argument("--artifact-parent", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--run-id")
    parser.add_argument(
        "--image-context",
        type=Path,
        default=Path("deployment/e025_context"),
    )
    arguments = parser.parse_args()
    layout = create_run_layout(arguments.artifact_parent, arguments.run_id)
    receipt = build_zero_spend_preflight(
        repo=arguments.repo.resolve(),
        checkpoint=arguments.checkpoint.resolve(),
        source_placement=arguments.source_placement.resolve(),
        layout=layout,
        image_context=arguments.image_context.resolve(),
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "run_id": receipt["run_id"],
                "run_root": str(layout.root),
                "vast_mutations_performed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "PREPARED_NOT_RENTAL_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
