"""Hard read-only Vast inventory and independent-machine fleet planning."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_019.placement import PlacementResult
from swarm_inference.experiments.experiment_020.fleet import NormalizedOffer, _version_tuple
from swarm_inference.experiments.experiment_020.vast import (
    VastCommandRunner,
    VastSafetyError,
    redact_account_payload,
)

from .io import atomic_write_json

MINIMUM_CUDA = "13.0"
MINIMUM_DRIVER = "580.65.06"


def _json_output(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        for index, character in enumerate(stripped):
            if character not in "[{":
                continue
            try:
                return json.loads(stripped[index:])
            except json.JSONDecodeError:
                continue
        raise


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("offers", "instances", "results", "data"):
            if isinstance(value.get(key), list):
                return [dict(row) for row in value[key] if isinstance(row, dict)]
        return [dict(value)] if value else []
    return []


def _search_arguments(*, maximum_gpu_ram_gib: int, limit: int) -> tuple[str, ...]:
    query = (
        f"rentable=True num_gpus=1 gpu_ram>=1 gpu_ram<={maximum_gpu_ram_gib}"
    )
    return ("search", "offers", query, "--limit", str(limit), "--raw")


def _kernel_compatible(offer: NormalizedOffer) -> bool:
    return (
        _version_tuple(offer.cuda_max) >= _version_tuple(MINIMUM_CUDA)
        and _version_tuple(offer.driver_version) >= _version_tuple(MINIMUM_DRIVER)
    )


def _offer_row(raw: dict[str, Any]) -> dict[str, Any]:
    offer = NormalizedOffer.from_raw(raw)
    compute_capability = raw.get("compute_cap", raw.get("compute_capability"))
    try:
        compute_capability_code = int(compute_capability)
    except (TypeError, ValueError):
        compute_capability_code = -1
    cuda_driver_compatible = _kernel_compatible(offer)
    native_sm86_compatible = compute_capability_code == 860
    return {
        **asdict(offer),
        "compute_capability": compute_capability,
        "rentable": bool(raw.get("rentable", True)),
        "cuda_driver_compatible": cuda_driver_compatible,
        "native_sm86_kernel_compatible": native_sm86_compatible,
        "native_kernel_and_cuda_driver_compatible": (
            cuda_driver_compatible and native_sm86_compatible
        ),
        "raw_host_bandwidth_metadata_is_not_inter_worker_rtt": True,
        "raw_selected_fields": {
            key: raw.get(key)
            for key in (
                "gpu_arch",
                "gpu_total_ram",
                "cpu_name",
                "cpu_ram",
                "duration",
                "storage_total_cost",
                "static_ip",
            )
            if key in raw
        },
    }


def _future_command(index: int, offer_id: int | None, cap: int) -> str:
    offer = str(offer_id) if offer_id is not None else f"<REFRESH_REQUIRED_OFFER_ID_{index:04d}>"
    return (
        f"vastai create instance {offer} "
        "--image ghcr.io/swarm-inference-lab/e021-independent@sha256:<PINNED_DIGEST_REQUIRED> "
        "--disk 40 --cancel-unavail "
        f"--label swarm-e021-future-machine-{index:04d} "
        "--env \"-e SWARM_RUN_CREDENTIAL_B64=<RUNTIME_SECRET_NOT_ARTIFACT> "
        f"-e SWARM_WORKER_ID=machine-{index:04d}.worker "
        f"-e SWARM_MEMORY_CAP_GIB={cap}\" "
        "--args worker --gpu-index 0"
    )


def run_read_only_market_inventory(
    artifact_root: Path,
    placements: dict[int, PlacementResult],
    *,
    limit: int = 500,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    vast_root = artifact_root / "vast"
    vast_root.mkdir(parents=True, exist_ok=True)
    executable = shutil.which("vastai")
    timestamp = datetime.now(UTC).isoformat()
    errors: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    searches: list[dict[str, Any]] = []
    user_receipt: dict[str, Any] = {"authenticated": False}
    runner: VastCommandRunner | None = None
    forbidden_guard_pass = False
    if executable:
        runner = VastCommandRunner(executable)
        for arguments in (("--version",), ("show", "user", "--raw")):
            try:
                completed = runner.run(arguments, timeout=30)
                if arguments[:2] == ("show", "user") and completed.returncode == 0:
                    user_receipt = redact_account_payload(
                        "user", _json_output(completed.stdout)
                    )
                if completed.returncode != 0:
                    errors.append(
                        {
                            "operation": " ".join(arguments),
                            "returncode": completed.returncode,
                            "stderr": completed.stderr[-500:],
                        }
                    )
            except Exception as exc:
                errors.append(
                    {
                        "operation": " ".join(arguments),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        # A single <=8 GiB query covers every declared E021 memory tier.  The
        # results are partitioned by cap after normalization.
        arguments = _search_arguments(maximum_gpu_ram_gib=8, limit=limit)
        try:
            completed = runner.run(arguments, timeout=180)
            raw_rows = _rows(_json_output(completed.stdout)) if completed.returncode == 0 else []
            normalized = [_offer_row(row) for row in raw_rows]
            searches.append(
                {
                    "query": arguments[2],
                    "returncode": completed.returncode,
                    "raw_offer_count": len(raw_rows),
                    "normalized_offer_count": len(normalized),
                }
            )
            if completed.returncode != 0:
                errors.append(
                    {
                        "operation": "search offers <=8 GiB single GPU",
                        "returncode": completed.returncode,
                        "stderr": completed.stderr[-1000:],
                    }
                )
        except Exception as exc:
            errors.append(
                {
                    "operation": "search offers <=8 GiB single GPU",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        try:
            runner.run(("create", "instance", "0"), timeout=1)
        except VastSafetyError:
            forbidden_guard_pass = True
        runner.write_audit(vast_root / "safety-audit.raw.json")
    else:
        errors.append({"operation": "locate vastai", "error": "executable not found"})

    unique: dict[int, dict[str, Any]] = {}
    for row in normalized:
        offer_id = int(row["offer_id"])
        if offer_id >= 0:
            unique[offer_id] = row
    offers = sorted(unique.values(), key=lambda row: int(row["offer_id"]))
    snapshot = {
        "schema_version": "experiment-021-single-machine-offer-snapshot-v1",
        "snapshot_at": timestamp,
        "cli_executable": str(Path(executable).resolve()) if executable else None,
        "authentication": user_receipt,
        "searches": searches,
        "query_scope": "independent offers with exactly one GPU and <=8 GiB VRAM",
        "offer_count": len(offers),
        "offers": offers,
        "offer_ids_are_ephemeral": True,
        "market_bandwidth_metadata_used_for_network_performance": False,
        "network_performance_evidence_class": "SHAPED_NETWORK",
        "errors": errors,
    }
    atomic_write_json(vast_root / "single-machine-offer-snapshot.json", snapshot)

    tiers = []
    chosen_for_8g: list[dict[str, Any]] = []
    for cap in sorted(placements, reverse=True):
        placement = placements[cap]
        minimum_mib = math_ceil_mib(placement.max_worker_peak_bytes)
        matching = [
            row
            for row in offers
            if int(row["num_gpus"]) == 1
            and minimum_mib <= int(row["gpu_ram_mib"]) <= cap * 1024
        ]
        compatible = [
            row
            for row in matching
            if row["native_kernel_and_cuda_driver_compatible"]
        ]
        best_by_machine: dict[int, dict[str, Any]] = {}
        for row in compatible:
            machine_id = int(row["machine_id"])
            prior = best_by_machine.get(machine_id)
            if prior is None or float(row["total_price_per_hour"]) < float(
                prior["total_price_per_hour"]
            ):
                best_by_machine[machine_id] = row
        candidates = sorted(
            best_by_machine.values(),
            key=lambda row: (
                -float(row["reliability"]),
                float(row["total_price_per_hour"]),
                int(row["offer_id"]),
            ),
        )
        required = len(placement.workers)
        chosen = candidates[:required]
        if cap == 8:
            chosen_for_8g = chosen
        feasible = len(chosen) == required
        hourly = (
            sum(float(row["total_price_per_hour"]) for row in chosen)
            if feasible
            else None
        )
        classes: dict[str, int] = {}
        for row in candidates:
            classes[str(row["gpu_name"])] = classes.get(str(row["gpu_name"]), 0) + 1
        tiers.append(
            {
                "worker_memory_cap_gib": cap,
                "minimum_required_usable_vram_mib": minimum_mib,
                "required_independent_machines": required,
                "raw_memory_matching_offers": len(matching),
                "kernel_compatible_offers": len(compatible),
                "kernel_requirement": (
                    "current pinned source image builds sm_86 native libraries and "
                    "requires CUDA >=13.0 / driver >=580.65.06"
                ),
                "unique_kernel_compatible_machines": len(candidates),
                "currently_feasible": feasible,
                "availability": "YES" if feasible else "PARTIAL" if candidates else "NO",
                "observed_eligible_gpu_classes": classes,
                "selected_offer_ids": [row["offer_id"] for row in chosen],
                "selected_machine_ids_unique": len({row["machine_id"] for row in chosen})
                == len(chosen),
                "compute_workers_per_machine": 1,
                "same_host_collective": False,
                "market_estimated_hourly_fleet_price_usd": hourly,
                "market_estimate_is_actual_cost_per_token": False,
            }
        )
    feasibility = {
        "schema_version": "experiment-021-fragmented-fleet-feasibility-v1",
        "snapshot_at": timestamp,
        "status": "PASS" if offers else "NO_LIVE_OFFERS_OR_QUERY_FAILED",
        "one_selected_compute_worker_per_machine": True,
        "multi_gpu_hosts_eligible": False,
        "same_host_collectives_assumed": False,
        "homogeneous_primary_fleet_preferred": True,
        "fallback_policy": (
            "refresh offers; prefer one homogeneous class; allow a mixed class only "
            "after per-class native kernel qualification and explicit straggler modeling"
        ),
        "tiers": tiers,
        "network_metadata_limitation": (
            "offer upload/download fields are acquisition/planning metadata and are "
            "not inter-worker RTT or throughput measurements"
        ),
    }
    atomic_write_json(vast_root / "fragmented-fleet-feasibility.json", feasibility)

    plan_path = vast_root / "rendered-future-plan.txt"
    required_8g = len(placements[8].workers)
    chosen_ids = [int(row["offer_id"]) for row in chosen_for_8g]
    commands = [
        _future_command(
            index,
            chosen_ids[index] if index < len(chosen_ids) else None,
            8,
        )
        for index in range(required_8g)
    ]
    plan_path.write_text(
        "\n".join(
            [
                "# EXPERIMENT 021 FUTURE PLAN — RENDERED ONLY",
                "EXECUTED=false",
                "RENTAL_APPROVED=false",
                "VAST_MUTATIONS=0",
                "# One command means one independent single-GPU machine / one worker.",
                "# Offer IDs are ephemeral and must be refreshed after explicit approval.",
                "# The image digest placeholder must be replaced by a published immutable E021 image.",
                *commands,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    raw_audit = {}
    raw_path = vast_root / "safety-audit.raw.json"
    if raw_path.is_file():
        raw_audit = json.loads(raw_path.read_text(encoding="utf-8"))
    audit = {
        "schema_version": "experiment-021-vast-safety-audit-v1",
        "experiment": "experiment-021",
        "mode": "READ_ONLY_HARD_LOCK",
        "zero_rental_policy": True,
        "gpu_rentals": 0,
        "vast_resource_mutations": 0,
        "create_start_stop_destroy_invocations": 0,
        "forbidden_mutation_guard_self_test": "PASS" if forbidden_guard_pass else "NOT_RUN",
        "commands": raw_audit.get("commands", []),
        "executed_read_only_command_count": sum(
            row.get("classification") == "READ_ONLY" and row.get("subprocess_invoked")
            for row in raw_audit.get("commands", [])
        ),
        "forbidden_commands_blocked_before_subprocess": sum(
            row.get("classification") == "FORBIDDEN" and not row.get("subprocess_invoked")
            for row in raw_audit.get("commands", [])
        ),
        "rendered_future_commands_executed": False,
    }
    atomic_write_json(vast_root / "safety-audit.json", audit)
    return snapshot, feasibility, audit


def math_ceil_mib(byte_count: int) -> int:
    return (int(byte_count) + 1024**2 - 1) // 1024**2


__all__ = ["run_read_only_market_inventory"]
