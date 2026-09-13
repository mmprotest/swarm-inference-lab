"""Attach the existing Stage 1 controller to instance 48441410 without renting again."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from swarm_inference.experiments.experiment_025.canary_runtime import (
    run_physical_stage_fixture,
)
from swarm_inference.experiments.experiment_025.io import (
    atomic_write_json,
    read_json,
    utc_now,
)
from swarm_inference.experiments.experiment_025.provisioning import (
    LiveWorker,
    wait_for_worker,
)
from swarm_inference.experiments.experiment_025.secrets import (
    load_transport_material,
)
from swarm_inference.experiments.experiment_025.stages import (
    _cost_summary,
    _failure_classification,
    _image,
    _retrieve_logs,
)
from swarm_inference.experiments.experiment_025.vast_lifecycle import (
    AppendOnlyLifecycleLedger,
    Offer,
    VastClient,
    destroy_all_from_ledger,
)


RUN_ID = "20260819T013016Z"
INSTANCE_ID = 48441410
WORKER_ID = "e025-stage-001"
STAGE_ROOT = Path(__file__).resolve().parent
RUN_ROOT = STAGE_ROOT.parent.parent


def main() -> int:
    ledger_path = STAGE_ROOT / "instance-ledger.jsonl"
    ledger = AppendOnlyLifecycleLedger(ledger_path, RUN_ID)
    client = VastClient(executable="vastai", ledger=ledger)
    go = read_json(STAGE_ROOT / "CANARY_GO.json")
    offer = Offer(**go["fleet_plan"]["workers"][0]["selected_offer"])
    image_reference, image_digest = _image(RUN_ROOT / "preflight" / "deployment-image.json")
    material = load_transport_material(
        Path(".e025-private").resolve() / RUN_ID,
        RUN_ID,
    )
    watchdog = read_json(STAGE_ROOT / "progress3-watchdog-receipt.json")
    # Preserve two minutes inside the independent hard stop for evidence and cleanup.
    readiness_deadline = float(watchdog["deadline_epoch"]) - 120.0
    worker: LiveWorker | None = None
    result = None
    failure = None
    try:
        worker = wait_for_worker(
            client=client,
            ledger=ledger,
            run_id=RUN_ID,
            worker_id=WORKER_ID,
            role="BACKBONE_STAGE",
            layer=1,
            worker_index=None,
            offer=offer,
            instance_id=INSTANCE_ID,
            credential=Path(material["credential_path"]).read_bytes(),
            certificate=Path(material["certificate_path"]),
            image_digest=image_digest,
            deadline_epoch=readiness_deadline,
        )
        result = run_physical_stage_fixture(
            worker=worker,
            checkpoint=Path(r"F:\models\Kimi-K3").resolve(),
            oracle_trace=Path(
                "artifacts/experiment-014/oracle-full-93-idot0/hidden-trace.f32"
            ).resolve(),
            oracle_routes=Path(
                "artifacts/experiment-014/oracle-full-93-idot0/routes.txt"
            ).resolve(),
            credential_path=Path(material["credential_path"]),
            certificate=Path(material["certificate_path"]),
            output_path=STAGE_ROOT / "physical-canary.json",
            cycle_id="E025-BACKBONE-CANARY",
        )
    except BaseException as exc:
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "classification": _failure_classification(exc),
        }
    finally:
        if worker is not None:
            _retrieve_logs(client, [worker], STAGE_ROOT / "logs")
        cleanup = destroy_all_from_ledger(
            ledger_path=ledger_path,
            run_id=RUN_ID,
            reason="single-backbone-canary-progress-aware-complete-or-aborted",
            executable="vastai",
            attempts=6,
        )
        costs = _cost_summary(ledger_path, RUN_ID)
        atomic_write_json(STAGE_ROOT / "rental-cost-summary.json", costs)
        atomic_write_json(STAGE_ROOT / "progress3-cleanup-verification.json", cleanup)
        if cleanup.get("zero_live_e025_instances") is True:
            atomic_write_json(
                STAGE_ROOT / "PROGRESS3_WATCHDOG_STOP",
                {"timestamp": utc_now(), "reason": "cleanup proven"},
            )
    payload = {
        "schema_version": "experiment-025-backbone-canary-stage-v1",
        "generated_at_utc": utc_now(),
        "status": (
            "PASS"
            if result is not None
            and result.get("status") == "PASS"
            and cleanup["zero_live_e025_instances"]
            else "FAIL"
        ),
        "run_id": RUN_ID,
        "image_reference": image_reference,
        "image_digest": image_digest,
        "worker": worker.endpoint() if worker else None,
        "result": result,
        "failure": failure,
        "cleanup": cleanup,
        "rental_costs": costs,
        "continuation": {
            "existing_instance_id": INSTANCE_ID,
            "new_rental_created": False,
            "progress_aware_operator_handoff": True,
        },
    }
    atomic_write_json(RUN_ROOT / "correctness" / "backbone-canary.json", payload)
    atomic_write_json(STAGE_ROOT / "backbone-canary-result.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
