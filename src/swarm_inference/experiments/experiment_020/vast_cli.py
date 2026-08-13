"""Canonical read-only Vast fleet interface for E020 and E021 preflight."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .fleet import (
    FleetPolicy,
    NormalizedOffer,
    market_summary,
    plan_fleet,
    render_launch_commands,
)
from .vast import (
    E020_RENTAL_FORBIDDEN,
    VastCommandRunner,
    VastSafetyError,
    assert_rental_armed,
    atomic_write_json,
    redact_account_payload,
)

GPU_CLASSES = ("RTX 3090", "RTX 3090 Ti", "RTX A5000", "RTX A6000", "RTX 4090")


def _json_output(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        # Some Vast CLI builds emit a non-JSON informational line before --raw.
        for index, character in enumerate(stripped):
            if character in "[{":
                try:
                    return json.loads(stripped[index:])
                except json.JSONDecodeError:
                    continue
        raise


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("offers", "instances", "ssh_keys", "keys", "results", "data"):
            if isinstance(value.get(key), list):
                return [dict(row) for row in value[key] if isinstance(row, dict)]
        return [dict(value)] if value else []
    return []


def _package_version() -> str | None:
    for distribution in ("vastai", "vast-cli"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def _search_arguments(gpu_class: str, *, limit: int = 250) -> list[str]:
    # The installed CLI accepts one positional query string. gpu_ram is GB in
    # the search language (the returned raw field is normalized separately).
    name = gpu_class.replace(" ", "_")
    query = f"gpu_name={name} gpu_ram>=20 rentable=True"
    return [
        "search",
        "offers",
        query,
        "--limit",
        str(limit),
        "--raw",
    ]


def inventory(
    output_root: Path,
    *,
    executable: str | Path | None = None,
    limit: int = 250,
) -> tuple[dict[str, Any], dict[str, Any], VastCommandRunner]:
    resolved = str(executable or shutil.which("vastai") or "")
    if not resolved:
        raise FileNotFoundError("vastai executable was not found on PATH")
    runner = VastCommandRunner(resolved)
    output_root.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, Any]] = []
    try:
        help_result = runner.run(("--help",), timeout=30)
        version_result = runner.run(("--version",), timeout=30)
        user_result = runner.run(("show", "user", "--raw"), timeout=30)
        keys_result = runner.run(("show", "ssh-keys", "--raw"), timeout=30)
        instances_result = runner.run(("show", "instances", "--raw"), timeout=30)
        user_payload = _json_output(user_result.stdout) if user_result.returncode == 0 else None
        key_payload = _json_output(keys_result.stdout) if keys_result.returncode == 0 else None
        instance_payload = (
            _json_output(instances_result.stdout) if instances_result.returncode == 0 else None
        )
        offers: list[NormalizedOffer] = []
        searches = []
        for gpu_class in GPU_CLASSES:
            result = runner.run(_search_arguments(gpu_class, limit=limit), timeout=120)
            raw_rows = _rows(_json_output(result.stdout)) if result.returncode == 0 else []
            normalized = [NormalizedOffer.from_raw(row) for row in raw_rows]
            normalized = [row for row in normalized if row.offer_id >= 0]
            offers.extend(normalized)
            searches.append(
                {
                    "gpu_class": gpu_class,
                    "returncode": result.returncode,
                    "success": result.returncode == 0,
                    "offer_count": len(normalized),
                }
            )
            if result.returncode != 0:
                errors.append(
                    {
                        "operation": f"search {gpu_class}",
                        "returncode": result.returncode,
                        "stderr": result.stderr[-500:],
                    }
                )
        timestamp = datetime.now(UTC).isoformat()
        cli_validation = {
            "schema_version": "experiment-020-vast-cli-validation-v1",
            "timestamp": timestamp,
            "cli_executable_path": str(Path(resolved).resolve()),
            "cli_version": (
                version_result.stdout.strip()
                if version_result.returncode == 0 and version_result.stdout.strip()
                else _package_version()
            ),
            "help_success": help_result.returncode == 0,
            "authentication_status": "AUTHENTICATED" if user_result.returncode == 0 and user_payload else "FAILED",
            "user": redact_account_payload("user", user_payload),
            "ssh_keys": redact_account_payload("ssh_keys", _rows(key_payload)),
            "instances": redact_account_payload("instances", _rows(instance_payload)),
            "search_success": any(row["success"] for row in searches),
            "searches": searches,
            "api_key_persisted": False,
            "email_persisted": False,
            "ssh_key_material_persisted": False,
            "unrelated_instances_mutated": False,
            "errors": errors,
        }
        unique_offers = {row.offer_id: row for row in offers}
        snapshot = {
            "schema_version": "experiment-020-vast-offer-snapshot-v1",
            "snapshot_at": timestamp,
            "ephemeral_offer_ids": True,
            "headline_plan_mixes_gpu_classes": False,
            "offers": [asdict(row) for row in sorted(unique_offers.values(), key=lambda item: item.offer_id)],
            "searches": searches,
        }
        atomic_write_json(output_root / "cli-validation.json", cli_validation)
        atomic_write_json(output_root / "offer-snapshot.json", snapshot)
        return cli_validation, snapshot, runner
    finally:
        runner.write_audit(output_root / "safety-audit.json")


def build_plan(
    output_root: Path,
    snapshot: dict[str, Any],
    *,
    policy: FleetPolicy | None = None,
    worker_manifest: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = policy or FleetPolicy()
    manifest_validation: dict[str, Any] | None = None
    if worker_manifest is not None:
        manifest = json.loads(worker_manifest.read_text(encoding="utf-8"))
        manifest_validation = {
            "path": str(worker_manifest.resolve()),
            "worker_count": int(manifest["worker_count"]),
            "pod_count": int(manifest["pod_count"]),
            "workers_per_pod": int(manifest["workers_per_pod"]),
            "matches_policy": (
                int(manifest["worker_count"]) == policy.total_gpus
                and int(manifest["pod_count"]) == policy.required_pods
                and int(manifest["workers_per_pod"]) == policy.exact_gpus_per_pod
            ),
        }
        if not manifest_validation["matches_policy"]:
            raise ValueError("worker manifest topology does not match fleet policy")
    offers = [NormalizedOffer(**row) for row in snapshot.get("offers", [])]
    plan = plan_fleet(offers, policy)
    summaries = [market_summary(offers, policy, gpu_class) for gpu_class in GPU_CLASSES]
    selected = next(row for row in summaries if row["gpu_class"] == policy.preferred_gpu_class)
    feasibility = {
        "schema_version": "experiment-020-vast-fleet-feasibility-v1",
        "snapshot_at": snapshot.get("snapshot_at"),
        "snapshot_not_future_guarantee": True,
        "required_pods": policy.required_pods,
        "gpus_per_pod": policy.exact_gpus_per_pod,
        "total_gpus": policy.total_gpus,
        "primary_gpu_class": policy.preferred_gpu_class,
        "matching_offers_currently_available": selected["matching_offers"],
        "matching_p8_hosts": selected["matching_p8_hosts"],
        "current_availability": plan["availability"],
        "estimated_hourly_fleet_price": plan["estimated_hourly_fleet_price"],
        "currently_selected_hourly_price": plan["currently_selected_hourly_price"],
        "estimated_hourly_fleet_price_basis": plan[
            "estimated_hourly_fleet_price_basis"
        ],
        "classes": summaries,
        "worker_manifest_validation": manifest_validation,
        "deployment_constraint": (
            None
            if plan["currently_feasible"]
            else "E021 must not launch until a fresh snapshot contains 12 homogeneous complete P8 hosts; unrelated single-GPU hosts are WAN and are not substitutes"
        ),
    }
    policy_receipt = asdict(policy)
    policy_receipt.update(
        {
            "image_cuda_runtime": "13.0.1",
            "driver_requirement_source": "https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-driver",
            "driver_gate_reason": "CUDA 13.0 GA requires Linux driver >=580.65.06; fail closed before rental",
        }
    )
    atomic_write_json(output_root / "fleet-policy.json", policy_receipt)
    atomic_write_json(output_root / "dry-run-plan.json", plan)
    atomic_write_json(output_root / "fleet-feasibility.json", feasibility)
    return plan, feasibility


def render_plan(
    output_root: Path,
    plan: dict[str, Any],
    runner: VastCommandRunner,
    *,
    run_id: str = "DRYRUN",
) -> list[str]:
    help_result = runner.run(("create", "instance", "--help"), timeout=30)
    commands = render_launch_commands(
        plan,
        image="ghcr.io/swarm-inference-lab/e021-sm86:PINNED_DIGEST",
        disk_gb=200,
        run_id=run_id,
    )
    path = output_root / "rendered-launch-commands.txt"
    lines = [
        "# Experiment 020 render only",
        "EXECUTED=false",
        f"CREATE_HELP_DISCOVERED={str(help_result.returncode == 0).lower()}",
        "# Offer IDs are ephemeral; regenerate immediately before E021.",
        "# Secret placeholders are resolved only in memory and are never artifact values.",
        "# CONTROLLER_ROLE_VALIDATED=false (E020 readiness blocker)",
        *commands,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    runner.write_audit(output_root / "safety-audit.json")
    return commands


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="swarm-vast", description=__doc__)
    parser.add_argument(
        "command", choices=("inventory", "plan", "render-launch", "preflight", "launch")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/experiment-020/vast")
    )
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--experiment-id", default="experiment-020")
    parser.add_argument("--max-budget-usd", type=float)
    parser.add_argument(
        "--worker-manifest",
        type=Path,
        default=Path("artifacts/experiment-020/placement/final-placement.json"),
    )
    parser.add_argument("--gpu-class", default="RTX 3090")
    parser.add_argument("--pods", type=int, default=12)
    parser.add_argument("--workers-per-pod", type=int, default=8)
    parser.add_argument("--min-reliability", type=float, default=0.95)
    parser.add_argument("--min-disk-gb", type=float, default=200.0)
    parser.add_argument("--min-internet-down-mbps", type=float, default=500.0)
    parser.add_argument("--min-internet-up-mbps", type=float, default=200.0)
    parser.add_argument("--max-price-per-pod-hour", type=float, default=8.0)
    parser.add_argument("--allow-unverified", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _arguments(argv)
    try:
        if arguments.command == "launch":
            plan = (
                json.loads(arguments.snapshot.read_text(encoding="utf-8"))
                if arguments.snapshot
                else None
            )
            assert_rental_armed(
                experiment_id=arguments.experiment_id,
                apply=arguments.apply,
                approved_plan=plan,
                max_budget_usd=arguments.max_budget_usd,
            )
            raise AssertionError("E020 must never reach a Vast mutation boundary")
        cli, snapshot, runner = inventory(arguments.output_root)
        if arguments.command == "inventory":
            print(json.dumps({"status": "PASS", "offers": len(snapshot["offers"])}))
            return 0
        if arguments.snapshot:
            snapshot = json.loads(arguments.snapshot.read_text(encoding="utf-8"))
        policy = FleetPolicy(
            preferred_gpu_class=arguments.gpu_class,
            required_pods=arguments.pods,
            exact_gpus_per_pod=arguments.workers_per_pod,
            minimum_reliability=arguments.min_reliability,
            minimum_disk_gb=arguments.min_disk_gb,
            minimum_internet_down_mbps=arguments.min_internet_down_mbps,
            minimum_internet_up_mbps=arguments.min_internet_up_mbps,
            maximum_price_per_pod_hour=arguments.max_price_per_pod_hour,
            verified_only=not arguments.allow_unverified,
        )
        plan, feasibility = build_plan(
            arguments.output_root,
            snapshot,
            policy=policy,
            worker_manifest=(
                arguments.worker_manifest if arguments.worker_manifest.exists() else None
            ),
        )
        commands: list[str] = []
        if arguments.command in {"render-launch", "preflight"}:
            commands = render_plan(arguments.output_root, plan, runner)
        print(
            json.dumps(
                {
                    "status": "PASS" if cli["authentication_status"] == "AUTHENTICATED" else "FAIL",
                    "availability": feasibility["current_availability"],
                    "rendered_commands": len(commands),
                    "executed_commands": 0,
                }
            )
        )
        return 0 if cli["authentication_status"] == "AUTHENTICATED" else 1
    except VastSafetyError as exc:
        print(str(exc), file=sys.stderr)
        return 2 if str(exc) == E020_RENTAL_FORBIDDEN else 1


if __name__ == "__main__":
    raise SystemExit(main())
