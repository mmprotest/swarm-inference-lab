from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_inference.experiments.experiment_025.runpod_inventory import (
    collect_read_only_inventory,
)
from swarm_inference.experiments.experiment_025.runpod_planning import RUN_ID


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture E025 RunPod inventory through read-only operations only"
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(f"artifacts/runs/experiment-025-{RUN_ID}"),
    )
    parser.add_argument("--runpodctl", type=Path)
    arguments = parser.parse_args()
    repo = Path.cwd().resolve()
    output = arguments.run_root.resolve() / "preflight" / "runpod"
    receipt = collect_read_only_inventory(
        repo=repo,
        output_directory=output,
        runpodctl_path=arguments.runpodctl,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
