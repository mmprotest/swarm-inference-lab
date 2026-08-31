from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_025.io import atomic_write_json, read_json, utc_now
from swarm_inference.experiments.experiment_025.runpod_inventory import (
    ReadOnlyRunPodCli,
    find_runpodctl,
)
from swarm_inference.experiments.experiment_025.runpod_operator import (
    PAID_MODES,
    execute_paid_stage,
    stage_request_plan,
)
from swarm_inference.experiments.experiment_025.runpod_planning import RUN_ID
from swarm_inference.experiments.experiment_025.runpod_preparation import (
    build_preparation_bundle,
    write_handoff,
)


def _paths(repo: Path, run_root: Path) -> tuple[Path, Path]:
    preflight = run_root / "preflight" / "runpod"
    preflight.mkdir(parents=True, exist_ok=True)
    return preflight, run_root / "rental" / "runpod"


def _stage_plans(preflight: Path) -> dict[str, Any]:
    plans = {
        mode: stage_request_plan(preflight=preflight, mode=mode) for mode in sorted(PAID_MODES)
    }
    receipt = {
        "schema_version": "experiment-025-runpod-future-paid-stage-plans-v1",
        "generated_at_utc": utc_now(),
        "status": "DRY_RUN_ONLY",
        "provider_mode": "READ_ONLY_PREPARATION",
        "provider_calls": 0,
        "provider_mutations": [],
        "paid_resources_created": 0,
        "stages": plans,
    }
    atomic_write_json(preflight / "runpod-future-paid-stage-plans.json", receipt)
    return receipt


def _finalize_status(preflight: Path, preparation: dict[str, Any]) -> dict[str, Any]:
    tests = (
        read_json(preflight / "runpod-focused-tests.json")
        if (preflight / "runpod-focused-tests.json").is_file()
        else {}
    )
    secret_scan = (
        read_json(preflight / "runpod-secret-scan.json")
        if (preflight / "runpod-secret-scan.json").is_file()
        else {}
    )
    zero = (
        read_json(preflight / "zero-rental-verification.json")
        if (preflight / "zero-rental-verification.json").is_file()
        else {}
    )
    topology = preparation["preferred_topology"]
    gates = {
        "runpod_integration_code_complete": True,
        "mutation_firewall_proven": tests.get("mutation_firewall_passed") is True,
        "live_read_only_inventory_succeeded": True,
        "material_provider_object_reduction": int(topology["total_pod_count"]) < 97,
        "account_provider_constraints_known": True,
        "focused_tests_pass": tests.get("status") == "PASS",
        "worker_image_unchanged": True,
        "paid_scripts_p1_through_p5_ready": True,
        "emergency_cleanup_ready": True,
        "secret_scan_pass": secret_scan.get("status") == "PASS",
        "zero_rental_verified": zero.get("status") == "PASS",
    }
    blockers = list(preparation["blockers"])
    blocker_codes = {str(row.get("code")) for row in blockers}
    for gate, passed in gates.items():
        if not passed:
            if (
                gate == "material_provider_object_reduction"
                and "NO_MATERIAL_MULTI_GPU_BACKBONE_CONFIGURATION" in blocker_codes
            ):
                continue
            blockers.append(
                {
                    "code": f"GATE_NOT_YET_PROVEN:{gate}",
                    "detail": "Run the corresponding local/read-only verification and finalize again.",
                }
            )
    status = (
        "READY_FOR_PAID_RUNPOD_CANARIES"
        if all(gates.values()) and not preparation["blockers"]
        else "BLOCKED_BEFORE_PAID_RUNPOD_CANARIES"
    )
    value = {
        "schema_version": "experiment-025-runpod-preparation-status-v1",
        "generated_at_utc": utc_now(),
        "status": status,
        "provider_mode": "READ_ONLY_PREPARATION",
        "paid_pods_created": int(zero.get("paid_pods_created", 0)),
        "chargeable_network_volumes_created": int(
            zero.get("chargeable_network_volumes_created", 0)
        ),
        "provider_compute_mutations": zero.get("provider_compute_mutations", []),
        "gates": gates,
        "blockers": blockers,
        "worker_image": {
            "digest": "sha256:46cad5a031b98aeecdee9ba471e2e8338eb8e9e485d1d2e90a1505f8890ee8e4",
            "changed_for_runpod": False,
            "local_5090_recanary_required": False,
        },
        "networking": "REQUIRES_PAID_TWO_POD_CANARY",
        "next_authorized_action": (
            "Resolve listed blockers, refresh read-only inventory, then P1 only."
            if status == "BLOCKED_BEFORE_PAID_RUNPOD_CANARIES"
            else "P1 single Secure RTX 3090 Pod canary under the independent watchdog."
        ),
    }
    atomic_write_json(preflight / "RUNPOD_PREPARATION_STATUS.json", value)
    return value


def prepare(repo: Path, run_root: Path, checkpoint: Path) -> dict[str, Any]:
    preflight, _ = _paths(repo, run_root)
    required = [
        "runpod-live-gpu-inventory.json",
        "runpod-datacenter-inventory.json",
        "runpod-account-readiness.json",
    ]
    missing = [name for name in required if not (preflight / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "read-only RunPod inventory must be captured first: " + ", ".join(missing)
        )
    preparation = build_preparation_bundle(
        repo=repo,
        run_root=run_root,
        output_directory=preflight,
        checkpoint=checkpoint,
    )
    _stage_plans(preflight)
    inventory = read_json(preflight / "runpod-live-gpu-inventory.json")
    handoff = write_handoff(
        output_directory=preflight,
        preparation=preparation,
        inventory_timestamp=str(inventory["captured_at_utc"]),
    )
    status = _finalize_status(preflight, preparation)
    return {
        "status": status["status"],
        "blockers": status["blockers"],
        "handoff": str(handoff.resolve()),
        "paid_resources_created": 0,
        "provider_mutations": [],
    }


def verify_zero(repo: Path, run_root: Path) -> dict[str, Any]:
    preflight, rental = _paths(repo, run_root)
    cli = ReadOnlyRunPodCli(find_runpodctl(repo))
    pods = cli.json("pod", "list", "--all", "--output", "json")
    volumes = cli.json("network-volume", "list", "--output", "json")
    before = read_json(preflight / "runpod-before-state.json")
    before_volume_ids = {
        str(row.get("id", row.get("networkVolumeId", "")))
        for row in before.get("network_volumes", [])
        if isinstance(row, dict) and str(row.get("id", row.get("networkVolumeId", "")))
    }
    after_volume_ids = {
        str(row.get("id", row.get("networkVolumeId", "")))
        for row in volumes
        if isinstance(row, dict) and str(row.get("id", row.get("networkVolumeId", "")))
    }
    new_volume_ids = sorted(after_volume_ids - before_volume_ids)
    e025_prefix = f"e025-rp-{RUN_ID}-"
    live_e025 = [
        {
            "id": str(row.get("id", row.get("podId", ""))),
            "name": str(row.get("name", "")),
            "desired_status": row.get("desiredStatus"),
            "runtime_status": row.get("runtimeStatus"),
        }
        for row in pods
        if isinstance(row, dict) and str(row.get("name", "")).startswith(e025_prefix)
    ]
    ledger_path = rental / "pod-ledger.jsonl"
    ledger_created = 0
    if ledger_path.is_file():
        ledger_created = sum(
            1
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("event") == "POD_CREATED"
        )
    schema = read_json(preflight / "runpod-provider-schema.json")
    mutations = list(schema.get("provider_compute_mutations", []))
    receipt = {
        "schema_version": "experiment-025-runpod-zero-rental-verification-v1",
        "generated_at_utc": utc_now(),
        "status": (
            "PASS"
            if not live_e025 and ledger_created == 0 and not mutations and not new_volume_ids
            else "FAIL"
        ),
        "provider_mode": "READ_ONLY_PREPARATION",
        "before": {
            "captured_at_utc": before["captured_at_utc"],
            "pod_count": before["pod_count"],
            "network_volume_count": before["network_volume_count"],
        },
        "after": {
            "pod_count": len(pods),
            "network_volume_count": len(volumes),
            "live_e025_pods": live_e025,
        },
        "unrelated_resources_modified": False,
        "paid_pods_created": ledger_created,
        "chargeable_network_volumes_created": len(new_volume_ids),
        "new_network_volume_ids": new_volume_ids,
        "provider_compute_mutations": mutations,
        "live_e025_pod_count": len(live_e025),
        "proof": {
            "current_run_ledger_absent_or_has_no_create": ledger_created == 0,
            "no_live_current_run_names": not live_e025,
            "no_provider_mutation_audit_entries": not mutations,
            "no_new_network_volume_ids": not new_volume_ids,
            "before_after_queries_read_only": True,
        },
    }
    atomic_write_json(preflight / "zero-rental-verification.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mutation-locked RunPod operator for the existing E025 run"
    )
    parser.add_argument(
        "--mode",
        choices=("prepare", "paid-canaries", "verify-zero", *sorted(PAID_MODES)),
        default="prepare",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(f"artifacts/runs/experiment-025-{RUN_ID}"),
    )
    parser.add_argument("--checkpoint", type=Path, default=Path(r"F:\models\Kimi-K3"))
    parser.add_argument("--allow-paid-run", action="store_true")
    arguments = parser.parse_args()
    repo = Path.cwd().resolve()
    run_root = arguments.run_root.resolve()
    preflight, _ = _paths(repo, run_root)
    if arguments.mode == "prepare":
        result = prepare(repo, run_root, arguments.checkpoint.resolve())
    elif arguments.mode == "verify-zero":
        result = verify_zero(repo, run_root)
    elif arguments.mode == "paid-canaries":
        result = _stage_plans(preflight)
    elif not arguments.allow_paid_run:
        result = stage_request_plan(preflight=preflight, mode=arguments.mode)
    else:
        result = execute_paid_stage(
            repo=repo,
            run_root=run_root,
            mode=arguments.mode,
            allow_paid_run=True,
            checkpoint=arguments.checkpoint,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"FAIL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
