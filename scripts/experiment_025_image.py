from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_inference.experiments.experiment_025.image import build_and_publish_image
from swarm_inference.experiments.experiment_025.io import read_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and publish the frozen E025 image")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--repository",
        default="ghcr.io/mmprotest/swarm-inference-lab",
    )
    parser.add_argument("--no-push", action="store_true")
    arguments = parser.parse_args()
    code = read_json(arguments.run_root / "preflight" / "code-freeze.json")
    run_id = arguments.run_root.name.removeprefix("experiment-025-")
    receipt = build_and_publish_image(
        repo=arguments.repo.resolve(),
        image_repository=arguments.repository,
        tag=f"e025-{run_id.lower()}",
        source_id=str(code["source_tree_sha256"]),
        output_path=arguments.run_root / "preflight" / "deployment-image.json",
        push=not arguments.no_push,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
