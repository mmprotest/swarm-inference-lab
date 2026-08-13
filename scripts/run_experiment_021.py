"""Canonical Experiment 021 preflight; Experiment 020 hard-blocks --apply."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_020.cost import estimate_e021_cost  # noqa: E402
from swarm_inference.experiments.experiment_020.vast import (  # noqa: E402
    VastSafetyError,
    assert_rental_armed,
    atomic_write_json,
)
from swarm_inference.experiments.experiment_020.vast_cli import (  # noqa: E402
    build_plan,
    inventory,
    render_plan,
)

ARTIFACT = ROOT / "artifacts" / "experiment-020"


def _image_receipt() -> dict[str, object]:
    completed = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "swarm-inference-lab:e021-sm86-e020",
            "--format",
            "{{json .}}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        return {"present": False, "digest": None, "published": False}
    value = json.loads(completed.stdout)
    digests = value.get("RepoDigests") or []
    return {
        "present": True,
        "local_id": value.get("Id"),
        "digest": digests[0] if digests else None,
        "size_bytes": value.get("Size"),
        "os": value.get("Os"),
        "architecture": value.get("Architecture"),
        "published": False,
    }


def _validate_static_inputs() -> dict[str, object]:
    required = {
        "placement": ARTIFACT / "placement" / "final-placement.json",
        "memory_audit": ARTIFACT / "placement" / "memory-audit.json",
        "manifest_hash_audit": ARTIFACT
        / "placement"
        / "manifest-hash-audit.json",
        "pod_bundles": ARTIFACT / "deployment" / "pod-bundles.json",
        "sm86_build": ARTIFACT / "deployment" / "sm86-build.json",
        "linux_build": ARTIFACT / "deployment" / "linux-build.json",
        "runbook": ROOT / "docs" / "experiments" / "EXPERIMENT_021_RUNBOOK.md",
        "model_index": Path("F:/models/Kimi-K3/model.safetensors.index.json"),
    }
    linux = (
        json.loads(required["linux_build"].read_text(encoding="utf-8"))
        if required["linux_build"].is_file()
        else {}
    )
    sm86 = (
        json.loads(required["sm86_build"].read_text(encoding="utf-8"))
        if required["sm86_build"].is_file()
        else {}
    )
    manifest_hashes = (
        json.loads(required["manifest_hash_audit"].read_text(encoding="utf-8"))
        if required["manifest_hash_audit"].is_file()
        else {}
    )
    return {
        "required_files": {name: path.is_file() for name, path in required.items()},
        "all_present": all(path.is_file() for path in required.values()),
        "linux_deployment_status": linux.get("status"),
        "linux_pre_rental_blockers": linux.get("pre_rental_blockers", []),
        "sm86_status": sm86.get("status"),
        "manifest_hash_status": manifest_hashes.get("status"),
        "deployment_ready": (
            linux.get("status") == "PASS"
            and sm86.get("status") == "PASS"
            and manifest_hashes.get("status") == "PASS"
            and manifest_hashes.get("placeholder_hashes_remaining") == 0
        ),
    }


def preflight(run_id: str) -> int:
    vast_root = ARTIFACT / "vast"
    cli, snapshot, runner = inventory(vast_root)
    plan, feasibility = build_plan(
        vast_root,
        snapshot,
        worker_manifest=ARTIFACT / "placement" / "final-placement.json",
    )
    commands = render_plan(vast_root, plan, runner, run_id=run_id)
    bundles = json.loads(
        (ARTIFACT / "deployment" / "pod-bundles.json").read_text(encoding="utf-8")
    )
    cost = estimate_e021_cost(snapshot, bundles)
    atomic_write_json(vast_root / "cost-estimate.json", cost)
    image = _image_receipt()
    static = _validate_static_inputs()
    bootstrap = vast_root / "rendered-bootstrap.txt"
    bootstrap.write_text(
        "# Render only; no command was executed on a Vast host.\n"
        "EXECUTED=false\n"
        "# Vast args mode preserves the image ENTRYPOINT; Docker-in-Docker is not used.\n"
        "# The current render invokes worker_main directly; bootstrap/cache integration is an E020 blocker.\n"
        "python -m swarm_inference.experiments.experiment_020.worker_main "
        "controller-host-agent  # BLOCKED until the production role is implemented\n",
        encoding="utf-8",
    )
    teardown = vast_root / "rendered-teardown.txt"
    teardown.write_text(
        "# Ledger-driven future teardown; no instance ID is known or executed in E020.\n"
        "EXECUTED=false\n"
        "python scripts/run_experiment_021.py --destroy-ledger "
        "artifacts/experiment-021/<run_id>/rental-ledger.json --apply "
        "--experiment-id experiment-021 --max-budget-usd <APPROVED_BUDGET>\n",
        encoding="utf-8",
    )
    image_ready = bool(image.get("digest")) and bool(image.get("published"))
    go = (
        cli["authentication_status"] == "AUTHENTICATED"
        and bool(static["all_present"])
        and bool(static["deployment_ready"])
        and feasibility["current_availability"] == "YES"
        and len(commands) == 12
        and image_ready
    )
    receipt = {
        "schema_version": "experiment-020-e021-preflight-v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "mode": "READ_ONLY",
        "status": "GO" if go else "NO_GO",
        "vast_authenticated": cli["authentication_status"] == "AUTHENTICATED",
        "ssh_key_present": cli["ssh_keys"]["ssh_key_present"],
        "marketplace_availability": feasibility["current_availability"],
        "rendered_launch_command_count": len(commands),
        "rendered_launch_commands_executed": 0,
        "image": image,
        "static_validation": static,
        "cost": cost,
        "no_rental": True,
        "no_vast_mutation": True,
        "go_conditions": {
            "fresh_complete_12_pod_plan": feasibility["current_availability"] == "YES",
            "approved_pinned_registry_image": image_ready,
            "production_deployment_ready": bool(static["deployment_ready"]),
            "explicit_future_budget_approval": False,
        },
    }
    atomic_write_json(vast_root / "e021-preflight.json", receipt)
    print(json.dumps({"status": receipt["status"], "availability": feasibility["current_availability"], "expected_cost_usd": cost["expected_cost"]["total_usd"]}))
    return 0 if go else 1


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--destroy-ledger", type=Path)
    parser.add_argument("--max-budget-usd", type=float)
    parser.add_argument("--experiment-id", default="experiment-021")
    parser.add_argument("--approved-plan", type=Path)
    parser.add_argument("--run-id", default="DRYRUN")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.preflight:
        return preflight(args.run_id)
    plan = (
        json.loads(args.approved_plan.read_text(encoding="utf-8"))
        if args.approved_plan
        else None
    )
    try:
        assert_rental_armed(
            experiment_id=args.experiment_id,
            apply=args.apply or args.destroy_ledger is not None,
            approved_plan=plan,
            max_budget_usd=args.max_budget_usd,
        )
    except VastSafetyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    raise RuntimeError("unreachable while EXPERIMENT_020_READ_ONLY is true")


if __name__ == "__main__":
    raise SystemExit(main())
