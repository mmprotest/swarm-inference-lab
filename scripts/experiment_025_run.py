from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_inference.experiments.experiment_025.headline import run_headline_stage
from swarm_inference.experiments.experiment_025.io import atomic_write_json, read_json
from swarm_inference.experiments.experiment_025.preflight import write_full_fleet_go
from swarm_inference.experiments.experiment_025.stages import _failed_machine_exclusions
from swarm_inference.experiments.experiment_025.vast_lifecycle import snapshot_and_rank


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fully gated E025 physical headline fleet")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(r"F:\models\Kimi-K3"))
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=Path("artifacts/experiment-014/oracle-full-93-idot0"),
    )
    parser.add_argument("--prompt", default="Hi")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--disk-gb", type=int, default=60)
    arguments = parser.parse_args()
    run_root = arguments.run_root.resolve()
    run_id = run_root.name.removeprefix("experiment-025-")
    image_index = read_json(Path("deployment/e025_context/index.json").resolve())
    distribution = read_json(run_root / "preflight" / "checkpoint-distribution.json")
    distribution_workers = {
        str(row["worker_id"]): row for row in distribution["workers"]
    }
    requirements = [
        {
            "worker_id": row["worker_id"],
            "role": row["role"],
            "download_bytes_cold_cache": row["download_bytes_cold_cache"],
            "assigned_tensor_bytes": row["assigned_tensor_bytes"],
            "temporary_disk_bytes": distribution_workers[str(row["worker_id"])][
                "temporary_disk_bytes"
            ],
        }
        for row in image_index["workers"]
    ]
    prior_failure_receipts = [
        _failed_machine_exclusions(
            run_root / "rental" / "stage-1-backbone-canary",
            stage_prefix="stage-1-backbone-canary",
        ),
        _failed_machine_exclusions(
            run_root / "rental" / "stage-2-sub-layer-canary",
            stage_prefix="stage-2-sub-layer-canary",
        ),
    ]
    excluded_machine_ids = {
        int(machine_id)
        for receipt in prior_failure_receipts
        for machine_id in receipt["machine_ids"]
    }
    atomic_write_json(
        run_root / "preflight" / "headline-failed-machine-exclusions.json",
        {
            "schema_version": "experiment-025-headline-machine-exclusions-v1",
            "policy": (
                "exclude every machine that previously failed authenticated readiness "
                "or a frozen E025 physical canary from headline selections and alternates"
            ),
            "machine_ids": sorted(excluded_machine_ids),
            "stage_receipts": prior_failure_receipts,
        },
    )
    offer_path = run_root / "preflight" / "full-fleet-offer-snapshot.json"
    snapshot_and_rank(
        output_path=offer_path,
        worker_requirements=requirements,
        disk_gb=arguments.disk_gb,
        excluded_machine_ids=excluded_machine_ids,
    )
    go_path = run_root / "preflight" / "FULL_FLEET_GO.json"
    go = write_full_fleet_go(
        repo=Path.cwd().resolve(),
        run_id=run_id,
        image_receipt_path=run_root / "preflight" / "deployment-image.json",
        test_receipt_path=run_root / "preflight" / "tests.json",
        rehearsal_receipt_path=run_root / "preflight" / "local-rehearsal.json",
        backbone_canary_path=run_root / "correctness" / "backbone-canary.json",
        sub_layer_canary_path=run_root / "correctness" / "sub-layer-canary.json",
        local_image_canary_path=run_root
        / "preflight"
        / "local-5090-image-canary.json",
        offer_snapshot_path=offer_path,
        output_path=go_path,
    )
    if go["status"] != "GO":
        print(json.dumps(go, indent=2, sort_keys=True))
        return 2
    receipt = run_headline_stage(
        run_id=run_id,
        stage_root=run_root / "rental" / "stage-4-headline",
        full_fleet_go_path=go_path,
        image_receipt_path=run_root / "preflight" / "deployment-image.json",
        private_root=Path(".e025-private").resolve() / run_id,
        physical_placement_path=run_root / "preflight" / "physical-placement.json",
        checkpoint=arguments.checkpoint.resolve(),
        oracle_trace=(arguments.oracle_root / "hidden-trace.f32").resolve(),
        oracle_routes=(arguments.oracle_root / "routes.txt").resolve(),
        correctness_output_path=run_root / "correctness" / "physical-two-token.json",
        generation_output_path=run_root / "generation" / "headline-generation.json",
        summary_output_path=run_root / "final" / "summary.json",
        prompt=arguments.prompt,
        max_new_tokens=arguments.max_new_tokens,
        disk_gb=arguments.disk_gb,
    )
    atomic_write_json(run_root / "final" / "headline-stage.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
