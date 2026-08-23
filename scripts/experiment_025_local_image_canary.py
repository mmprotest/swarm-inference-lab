from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_inference.experiments.experiment_025.image import validate_local_5090_image


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the immutable E025 Linux image on the local RTX 5090"
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(r"F:\models\Kimi-K3"))
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=Path("artifacts/experiment-014/oracle-full-93-idot0"),
    )
    arguments = parser.parse_args()
    run_root = arguments.run_root.resolve()
    receipt = validate_local_5090_image(
        repo=arguments.repo.resolve(),
        image_receipt_path=run_root / "preflight" / "deployment-image.json",
        checkpoint=arguments.checkpoint.resolve(),
        oracle_root=arguments.oracle_root.resolve(),
        output_path=run_root / "preflight" / "local-5090-image-canary.json",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
