"""Fresh read-only Vast and budget gate for E025 Headline Attempt 006.

This module intentionally exposes no create or destroy path.  It reuses the frozen
97-role model placement, refreshes only marketplace/account state, and writes an
explicit GO/NO_GO receipt.  A failed offline credibility gate always forbids paid
instance creation, even when current marketplace capacity and account budget pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_025.io import (
    atomic_write_json,
    canonical_sha256,
    read_json,
    sha256_file,
    utc_now,
)
from swarm_inference.experiments.experiment_025.vast_lifecycle import (
    VastClient,
    snapshot_and_rank,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = (
    REPO_ROOT / "artifacts" / "runs" / "experiment-025-20260819T013016Z"
)
RUN_ID = "20260819T013016Z"


def _frozen_requirements(frozen_go: dict[str, Any]) -> list[dict[str, Any]]:
    workers = list(frozen_go["fleet_plan"]["workers"])
    requirements = [
        {
            "worker_id": str(row["worker_id"]),
            "role": str(row["role"]),
            "download_bytes_cold_cache": int(row["download_bytes_cold_cache"]),
            "assigned_tensor_bytes": int(row["assigned_tensor_bytes"]),
            "temporary_disk_bytes": int(row["temporary_disk_bytes"]),
        }
        for row in workers
    ]
    role_counts = Counter(row["role"] for row in requirements)
    if len(requirements) != 97 or role_counts != {
        "BACKBONE_STAGE": 92,
        "SUB_LAYER_PARENT": 1,
        "SUB_LAYER_WORKER": 4,
    }:
        raise RuntimeError("frozen Attempt 005 placement is not the required 97 roles")
    return requirements


def _history_by_machine(history: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["machine_id"]): dict(row)
        for row in history["machines"]
        if int(row["machine_id"]) >= 0
    }


def _attributable_instances(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    label_prefix = f"e025-{RUN_ID}-".lower()
    retained: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get("label", ""))
        if not label.lower().startswith(label_prefix):
            continue
        retained.append(
            {
                "instance_id": row.get("id", row.get("instance_id")),
                "label": label,
                "machine_id": row.get("machine_id"),
                "provider_state": row.get("actual_status", row.get("status")),
            }
        )
    return retained


def _with_artifact_hash(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["artifact_sha256"] = canonical_sha256(payload)
    return result


def run(run_root: Path, *, vast_executable: str = "vastai") -> dict[str, Any]:
    run_root = run_root.resolve()
    preflight = run_root / "preflight"
    policy_path = preflight / "attempt-006-acquisition-policy.json"
    simulation_path = preflight / "attempt-006-acquisition-simulation.json"
    history_path = preflight / "attempt-006-machine-reliability.json"
    recovery_path = preflight / "attempt-006-recovery-tests.json"
    boundary_path = preflight / "attempt-006-source-boundary.json"
    frozen_go_path = preflight / "FULL_FLEET_GO-attempt-005.json"
    policy = read_json(policy_path)
    simulation = read_json(simulation_path)
    history = read_json(history_path)
    recovery = read_json(recovery_path)
    boundary = read_json(boundary_path)
    frozen_go = read_json(frozen_go_path)
    requirements = _frozen_requirements(frozen_go)

    image_digest = str(policy["worker_image"]["immutable_digest"])
    deployment = read_json(preflight / "deployment-image.json")
    stage_1 = read_json(run_root / "correctness" / "backbone-canary.json")
    stage_2 = read_json(run_root / "correctness" / "sub-layer-canary.json")
    client = VastClient(executable=vast_executable)

    live_before: list[dict[str, Any]] = []
    live_after: list[dict[str, Any]] = []
    snapshot: dict[str, Any] | None = None
    query_error: str | None = None
    snapshot_path = preflight / "attempt-006-live-offer-snapshot.json"
    try:
        live_before = client.show_instances()
        snapshot = snapshot_and_rank(
            output_path=snapshot_path,
            worker_requirements=requirements,
            disk_gb=60,
            excluded_machine_ids={
                int(machine_id)
                for machine_id in history["hard_exclusion_machine_ids"]
            },
            acquisition_history=_history_by_machine(history),
            scoring_policy=dict(policy["offer_score"]),
            maximum_candidate_ready_seconds=(
                float(policy["acquisition_window_seconds"])
                - float(policy["ready_health_policy"]["stability_barrier_seconds"])
            ),
            stage_ttl_seconds=float(policy["hard_stage_ttl_seconds"]),
            executable=vast_executable,
        )
        live_after = client.show_instances()
    except Exception as exc:
        query_error = f"{type(exc).__name__}: {exc}"[-2000:]
        try:
            live_after = client.show_instances()
        except Exception as final_exc:
            query_error += f"; final zero-live query failed: {type(final_exc).__name__}: {final_exc}"

    attributable_before = _attributable_instances(live_before)
    attributable_after = _attributable_instances(live_after)
    if snapshot is None:
        failed_snapshot = _with_artifact_hash(
            {
                "schema_version": "experiment-025-attempt-006-live-offer-snapshot-v1",
                "generated_at_utc": utc_now(),
                "status": "NO_GO",
                "query_mode": "READ_ONLY_ON_DEMAND",
                "provider_mutations": [],
                "marketplace_blocker": query_error,
            }
        )
        atomic_write_json(snapshot_path, failed_snapshot)

    fleet = dict(snapshot.get("fleet_plan", {})) if snapshot else {}
    budget = dict(snapshot.get("redacted_budget", {})) if snapshot else {}
    projected_acquisition_cost = None
    projected_hard_stage_cost = None
    fresh_budget_required = None
    if snapshot is not None and fleet:
        hourly = float(
            fleet["total_effective_rate_including_requested_storage_usd_per_hour"]
        )
        ingress = float(fleet["total_expected_ingress_cost_usd"])
        projected_acquisition_cost = (
            hourly * float(policy["acquisition_window_seconds"]) / 3600.0
            + ingress
        )
        projected_hard_stage_cost = (
            hourly * float(policy["hard_stage_ttl_seconds"]) / 3600.0 + ingress
        )
        fresh_budget_required = projected_hard_stage_cost * float(
            policy["cost_guard"]["fresh_provider_budget_safety_multiplier"]
        )

    maximum_ready_seconds = fleet.get("maximum_expected_ready_seconds")
    maximum_candidate_ready_seconds = (
        float(policy["acquisition_window_seconds"])
        - float(policy["ready_health_policy"]["stability_barrier_seconds"])
    )
    # snapshot_and_rank deliberately changes fleet_plan.status to NO_GO when its
    # combined provider-budget check fails.  Capacity and budget are separate E025
    # gates, so determine marketplace feasibility from the fully assembled exact-
    # constraint plan rather than from that combined status bit.
    marketplace_ok = bool(
        snapshot is not None
        and int(fleet.get("worker_count", 0)) == 97
        and int(fleet.get("unique_selected_machine_count", 0))
        == int(fleet.get("instance_group_count", -1))
        and len(set(fleet.get("sub_layer_machine_ids", []))) == 4
        and maximum_ready_seconds is not None
        and float(maximum_ready_seconds) <= maximum_candidate_ready_seconds
    )
    budget_ok = bool(
        fresh_budget_required is not None
        and projected_acquisition_cost is not None
        and float(budget.get("conservative_available_usd", -1.0))
        >= fresh_budget_required
        and projected_acquisition_cost
        <= float(policy["cost_guard"]["maximum_total_acquisition_cost_usd"])
    )
    worker_boundary_ok = bool(
        boundary["all_worker_executed_files_unchanged"]
        and deployment.get("status") == "PASS"
        and deployment.get("immutable_digest") == image_digest
        and stage_1.get("status") == "PASS"
        and stage_1.get("image_digest") == image_digest
        and stage_2.get("status") == "PASS"
        and stage_2.get("image_digest") == image_digest
    )
    tests_ok = recovery.get("status") == "PASS"
    offline_credible = bool(
        simulation.get("status") == "PASS"
        and simulation.get("selected_policy") is not None
        and simulation.get("decision")
        == "PROCEED_TO_FRESH_READ_ONLY_PREFLIGHT"
    )
    gates = [
        (1, "Layer 89 parent serialization removed", tests_ok),
        (2, "READY roles remain under supervision through freeze", tests_ok),
        (3, "dead READY roles are replaceable", tests_ok),
        (4, "bounded live alternate refresh implemented and tested", tests_ok),
        (5, "reliability-aware scoring active in fresh ranking", snapshot is not None),
        (
            6,
            "bounded hedging tested if selected",
            tests_ok
            and (
                not bool(policy["hedge_policy"]["enabled"])
                or bool(
                    policy["hedge_policy"][
                        "controller_capability_implemented_and_tested"
                    ]
                )
            ),
        ),
        (7, "readiness requires 97 simultaneous healthy current roles", tests_ok),
        (8, "Layer 89 parent/fragment generations must match", tests_ok),
        (9, "focused recovery tests pass", tests_ok),
        (10, "worker image and physical canary validity intact", worker_boundary_ok),
        (11, "fresh marketplace has compatible capacity", marketplace_ok),
        (12, "fresh budget and acquisition cost guards pass", budget_ok),
        (
            13,
            "watchdog and cleanup paths armed",
            tests_ok
            and bool(policy["watchdog_cleanup_policy"]["independent_watchdog_required"])
            and not attributable_before
            and not attributable_after,
        ),
        (14, "offline analysis finds a credible acquisition path", offline_credible),
    ]
    gate_rows = [
        {"gate": number, "description": description, "passed": bool(passed)}
        for number, description, passed in gates
    ]
    go = all(row["passed"] for row in gate_rows)
    payload = _with_artifact_hash(
        {
            "schema_version": "experiment-025-attempt-006-live-preflight-v1",
            "generated_at_utc": utc_now(),
            "status": "GO" if go else "NO_GO",
            "run_id": RUN_ID,
            "attempt": 6,
            "query_mode": "READ_ONLY_ON_DEMAND",
            "provider_mutations": [],
            "paid_instances_created": 0,
            "worker_image_digest": image_digest,
            "controller_source_sha256": boundary["controller_source_sha256"],
            "acquisition_policy_file_sha256": sha256_file(policy_path),
            "acquisition_policy_canonical_sha256": canonical_sha256(policy),
            "offline_simulation_file_sha256": sha256_file(simulation_path),
            "recovery_tests_file_sha256": sha256_file(recovery_path),
            "frozen_model_placement_source": str(frozen_go_path),
            "frozen_required_role_count": len(requirements),
            "fresh_marketplace": {
                "query_succeeded": snapshot is not None,
                "marketplace_blocker": query_error,
                "raw_offer_count": snapshot.get("raw_offer_count") if snapshot else None,
                "fleet_plan_status": fleet.get("status"),
                "combined_snapshot_status": (
                    snapshot.get("status") if snapshot else None
                ),
                "capacity_assessed_separately_from_budget": True,
                "capacity_gate_passed": marketplace_ok,
                "selected_instance_group_count": fleet.get("instance_group_count"),
                "selected_worker_count": fleet.get("worker_count"),
                "distinct_selected_machine_count": fleet.get(
                    "unique_selected_machine_count"
                ),
                "distinct_fragment_machine_count": len(
                    set(fleet.get("sub_layer_machine_ids", []))
                ),
                "maximum_expected_ready_seconds": fleet.get(
                    "maximum_expected_ready_seconds"
                ),
                "offer_snapshot": str(snapshot_path),
            },
            "fresh_budget": {
                "conservative_available_usd": budget.get(
                    "conservative_available_usd"
                ),
                "projected_acquisition_cost_usd": projected_acquisition_cost,
                "projected_hard_stage_plus_ingress_usd": projected_hard_stage_cost,
                "required_with_safety_multiplier_usd": fresh_budget_required,
                "acquisition_cost_guard_usd": policy["cost_guard"][
                    "maximum_total_acquisition_cost_usd"
                ],
                "identity_fields_persisted": False,
                "api_key_persisted": False,
            },
            "zero_live_e025": {
                "before_query": len(attributable_before) == 0,
                "after_query": len(attributable_after) == 0,
                "attributable_before": attributable_before,
                "attributable_after": attributable_after,
            },
            "go_gates": gate_rows,
            "failed_gate_numbers": [
                row["gate"] for row in gate_rows if not row["passed"]
            ],
            "offline_decision": simulation["decision"],
            "attempt_006_launch_authorized": go,
            "decision": (
                "LAUNCH_HEADLINE_ATTEMPT_006"
                if go
                else "NO_GO_CREATE_NO_PAID_INSTANCES"
            ),
        }
    )
    receipt_path = preflight / "attempt-006-live-preflight.json"
    terminal_path = preflight / (
        "ATTEMPT_006_GO.json" if go else "ATTEMPT_006_NO_GO.json"
    )
    atomic_write_json(receipt_path, payload)
    atomic_write_json(terminal_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--vast-executable", default="vastai")
    args = parser.parse_args()
    payload = run(args.run_root, vast_executable=args.vast_executable)
    print(
        {
            "status": payload["status"],
            "failed_gate_numbers": payload["failed_gate_numbers"],
            "decision": payload["decision"],
            "paid_instances_created": payload["paid_instances_created"],
        }
    )
    return 0 if payload["status"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
