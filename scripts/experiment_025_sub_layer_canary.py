from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_inference.experiments.experiment_025.stages import (
    run_sub_layer_canary_stage,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the paid E025 physical sub-layer canary")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(r"F:\models\Kimi-K3"))
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=Path("artifacts/experiment-014/oracle-full-93-idot0"),
    )
    parser.add_argument("--disk-gb", type=int, default=60)
    arguments = parser.parse_args()
    run_root = arguments.run_root.resolve()
    run_id = run_root.name.removeprefix("experiment-025-")
    receipt = run_sub_layer_canary_stage(
        run_id=run_id,
        stage_root=run_root / "rental" / "stage-2-sub-layer-canary",
        image_receipt_path=run_root / "preflight" / "deployment-image.json",
        image_index_path=Path("deployment/e025_context/index.json").resolve(),
        test_receipt_path=run_root / "preflight" / "tests.json",
        rehearsal_receipt_path=run_root / "preflight" / "local-rehearsal.json",
        local_image_canary_path=run_root
        / "preflight"
        / "local-5090-image-canary.json",
        backbone_canary_path=run_root / "correctness" / "backbone-canary.json",
        private_root=Path(".e025-private").resolve() / run_id,
        checkpoint=arguments.checkpoint.resolve(),
        oracle_trace=(arguments.oracle_root / "hidden-trace.f32").resolve(),
        oracle_routes=(arguments.oracle_root / "routes.txt").resolve(),
        physical_placement=run_root / "preflight" / "physical-placement.json",
        output_path=run_root / "correctness" / "sub-layer-canary.json",
        disk_gb=arguments.disk_gb,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
