from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_inference.experiments.experiment_025.runpod_planning import RUN_ID
from swarm_inference.experiments.experiment_025.runpod_security import (
    scan_runpod_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan E025 RunPod artifacts for secrets")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(f"artifacts/runs/experiment-025-{RUN_ID}/preflight/runpod"),
    )
    arguments = parser.parse_args()
    receipt = scan_runpod_artifacts(arguments.artifact_root)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
