"""Refresh only live full-fleet offers and write the E025 GO receipt."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_025 import vast_lifecycle
from swarm_inference.experiments.experiment_025.io import atomic_write_json, read_json
from swarm_inference.experiments.experiment_025.preflight import write_full_fleet_go
from swarm_inference.experiments.experiment_025.stages import _failed_machine_exclusions
from swarm_inference.experiments.experiment_025.vast_lifecycle import snapshot_and_rank


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def headline_progress_aware_failed_machine_exclusions(run_root: Path) -> dict[str, Any]:
    """Retain progressing deadline-abort hosts; exclude only attributable duds."""

    sources: dict[int, list[dict[str, Any]]] = {}
    retained_progressing_deadline_aborts: set[int] = set()
    attempt_roots = sorted(
        path
        for path in (run_root / "rental").glob("stage-4-headline-attempt-*")
        if path.is_dir()
    )
    for attempt in attempt_roots:
        ledger = _read_jsonl(attempt / "instance-ledger.jsonl")
        created = {
            int(row["instance_id"]): row
            for row in ledger
            if row.get("event") == "CREATE_CONFIRMED"
            and row.get("instance_id") is not None
            and row.get("machine_id") is not None
        }
        for row in ledger:
            if row.get("event") != "CREATE_ID_LOST" or row.get("machine_id") is None:
                continue
            machine_id = int(row["machine_id"])
            sources.setdefault(machine_id, []).append(
                {
                    "stage_directory": attempt.name,
                    "instance_id": None,
                    "offer_id": row.get("offer_id"),
                    "bootstrap_stage": "INSTANCE_CREATE",
                    "reason": "provider create returned no recoverable instance ID",
                }
            )

        event_paths = sorted(attempt.glob("physical-run-events*.jsonl"))
        events = _read_jsonl(event_paths[0]) if event_paths else []
        trusted_progress_observer = attempt.name.startswith(
            "stage-4-headline-attempt-004-"
        ) or any(row.get("event_type") == "ACQUISITION_POLICY_APPLIED" for row in events)
        last_progress_by_instance: dict[int, dict[str, Any]] = {}
        for row in events:
            if row.get("instance_id") is None:
                continue
            if row.get("event_type") in {"MODEL_DOWNLOAD_PROGRESS", "BOOTSTRAP_PROGRESS"}:
                last_progress_by_instance[int(row["instance_id"])] = row
        for row in events:
            instance_id = row.get("instance_id")
            if instance_id is None:
                continue
            instance_id = int(instance_id)
            ledger_row = created.get(instance_id, {})
            machine_value = row.get("machine_id", ledger_row.get("machine_id"))
            if machine_value is None:
                continue
            machine_id = int(machine_value)
            is_attributable_stall = (
                trusted_progress_observer
                and row.get("event_type") == "TIMEOUT"
                and row.get("bootstrap_stage") is not None
                and float(row.get("no_progress_seconds", 0.0)) >= 240.0
            )
            is_fatal_bootstrap = (
                row.get("event_type") == "WORKER_UNHEALTHY"
                and row.get("reason") == "fatal_bootstrap_marker"
            )
            if is_attributable_stall or is_fatal_bootstrap:
                sources.setdefault(machine_id, []).append(
                    {
                        "stage_directory": attempt.name,
                        "instance_id": instance_id,
                        "offer_id": ledger_row.get("offer_id"),
                        "bootstrap_stage": row.get("bootstrap_stage"),
                        "no_progress_seconds": row.get("no_progress_seconds"),
                        "reason": (
                            "fatal bootstrap log marker"
                            if is_fatal_bootstrap
                            else "no observable bootstrap progress for at least 240 seconds"
                        ),
                    }
                )
            elif row.get("event_type") in {"TIMEOUT", "ERROR"}:
                last_progress = last_progress_by_instance.get(instance_id)
                if last_progress is not None and last_progress.get("bootstrap_stage") in {
                    "MODEL_DOWNLOAD",
                    "PACKAGE_READY_OR_ACTIVATING",
                    "GPU_LOAD",
                    "WORKER_LISTENING",
                }:
                    retained_progressing_deadline_aborts.add(machine_id)

    normalized_sources = [
        {"machine_id": machine_id, "observations": observations}
        for machine_id, observations in sorted(sources.items())
    ]
    return {
        "schema_version": "experiment-025-headline-progress-aware-exclusions-v1",
        "stage_prefix": "stage-4-headline-attempt",
        "policy": (
            "temporarily exclude only provider create-ID loss, a trusted observer's "
            "240-second no-progress stall, or a fatal bootstrap marker; retain hosts "
            "aborted solely by the fleet deadline or shared abort"
        ),
        "machine_ids": [row["machine_id"] for row in normalized_sources],
        "sources": normalized_sources,
        "retained_progressing_deadline_abort_machine_ids": sorted(
            retained_progressing_deadline_aborts - set(sources)
        ),
    }


def main() -> int:
    run_root = Path("artifacts/runs/experiment-025-20260819T013016Z").resolve()
    run_id = "20260819T013016Z"
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
    stage2_helpers = runpy.run_path(
        str(run_root / "preflight" / "run-stage2-progress-aware.py")
    )
    stage2_failed_machine_exclusions = stage2_helpers[
        "progress_aware_failed_machine_exclusions"
    ]
    prior_failures = [
        _failed_machine_exclusions(
            run_root / "rental" / "stage-1-backbone-canary",
            stage_prefix="stage-1-backbone-canary",
        ),
        stage2_failed_machine_exclusions(
            run_root / "rental" / "stage-2-sub-layer-canary",
            stage_prefix="stage-2-sub-layer-canary",
        ),
        headline_progress_aware_failed_machine_exclusions(run_root),
    ]
    excluded = {
        int(machine_id)
        for receipt in prior_failures
        for machine_id in receipt["machine_ids"]
    }
    atomic_write_json(
        run_root / "preflight" / "headline-failed-machine-exclusions.json",
        {
            "schema_version": "experiment-025-headline-machine-exclusions-v1",
            "policy": (
                "exclude only machines with an attributable create-ID loss, progress-aware "
                "bootstrap stall, authenticated readiness failure, or frozen physical-canary "
                "failure; retain healthy siblings and deadline-aborted progressing hosts"
            ),
            "machine_ids": sorted(excluded),
            "stage_receipts": prior_failures,
        },
    )
    offer_path = run_root / "preflight" / "full-fleet-offer-snapshot.json"
    original_rank = vast_lifecycle.rank_grouped_offers_for_workers

    def rank_with_resilient_alternates(*args: object, **kwargs: object) -> dict[str, object]:
        kwargs["alternates_per_group"] = 16
        kwargs["reserved_alternate_machines"] = 20
        return original_rank(*args, **kwargs)

    vast_lifecycle.rank_grouped_offers_for_workers = rank_with_resilient_alternates
    snapshot = snapshot_and_rank(
        output_path=offer_path,
        worker_requirements=requirements,
        disk_gb=60,
        excluded_machine_ids=excluded,
    )
    alternate_machine_ids = {
        int(value["offer"]["machine_id"])
        for group in snapshot["fleet_plan"]["instance_groups"]
        for value in group.get("alternates", [])
    }
    snapshot["fleet_plan"]["alternate_resilience"] = {
        "reason": "headline attempt 001 exposed overlapping three-offer fallback lists",
        "alternates_requested_per_group": 16,
        "reserved_alternate_machines_requested": 20,
        "distinct_alternate_machine_count": len(alternate_machine_ids),
        "minimum_alternates_per_group": min(
            len(group.get("alternates", []))
            for group in snapshot["fleet_plan"]["instance_groups"]
        ),
    }
    atomic_write_json(offer_path, snapshot)
    go = write_full_fleet_go(
        repo=Path.cwd().resolve(),
        run_id=run_id,
        image_receipt_path=run_root / "preflight" / "deployment-image.json",
        test_receipt_path=run_root / "preflight" / "tests.json",
        rehearsal_receipt_path=run_root / "preflight" / "local-rehearsal.json",
        backbone_canary_path=run_root / "correctness" / "backbone-canary.json",
        sub_layer_canary_path=run_root / "correctness" / "sub-layer-canary.json",
        local_image_canary_path=run_root / "preflight" / "local-5090-image-canary.json",
        offer_snapshot_path=offer_path,
        output_path=run_root / "preflight" / "FULL_FLEET_GO.json",
    )
    fleet = go.get("fleet_plan", {})
    print(
        json.dumps(
            {
                "status": go.get("status"),
                "worker_count": fleet.get("worker_count"),
                "unique_machine_count": fleet.get("unique_selected_machine_count"),
                "total_active_rental_rate_usd_per_hour": fleet.get(
                    "total_active_rental_rate_usd_per_hour"
                ),
                "checks": go.get("checks"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if go.get("status") == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
