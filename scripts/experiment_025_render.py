from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_inference.experiments.experiment_025.reporting import render_final_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Render immutable E025 public evidence")
    parser.add_argument("--run-root", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = render_final_artifacts(arguments.run_root)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
