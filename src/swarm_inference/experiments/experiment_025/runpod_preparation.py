"""Assemble the complete zero-rental E025 RunPod preparation artifact bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swarm_inference.model.kimi_tokenizer import KIMI_TOKENIZER_ASSETS

from .controller import tokenize_prompt
from .io import atomic_write_json, canonical_sha256, sha256_file, utc_now
from .runpod_lifecycle import simulate_full_control_flow
from .runpod_planning import (
    BACKBONE_GPU_ID,
    FRAGMENT_GPU_IDS,
    GLOBAL_NETWORK_MEGABITS_PER_SECOND,
    IMAGE_DIGEST,
    IMAGE_REFERENCE,
    RUN_ID,
    assert_manifest_safety,
    build_role_manifests,
    headline_budget,
    load_frozen_roles,
    model_distribution_plan,
    pack_roles,
    paid_canary_budget,
    storage_plan,
    topology_metrics,
)

HEADLINE_PROMPT = (
    'Repeat exactly: "I am the swarm. 2.8 trillion parameters. Consumer GPUs. '
    'No datacenter GPUs required."'
)
HEADLINE_TARGET = (
    "I am the swarm. 2.8 trillion parameters. Consumer GPUs. No datacenter GPUs required."
)


def _gpu_rows(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["gpu_type_id"]): row for row in inventory["gpu_types"]}


def _tier(row: dict[str, Any], name: str) -> dict[str, Any]:
    return row.get("graphql", {}).get("tiers", {}).get(name, {})


def _schedulable_counts(row: dict[str, Any], tier: str) -> list[int]:
    return [
        int(value)
        for value in _tier(row, tier).get(
            "schedulable_gpu_counts_inferred_from_non_null_lowest_price", []
        )
    ]


def _count_price(row: dict[str, Any], tier: str, count: int) -> float | None:
    values = _tier(row, tier).get("count_queries", [])
    match = next((value for value in values if int(value["gpu_count"]) == count), None)
    if match is None or match.get("uninterruptablePrice") is None:
        return None
    # The GraphQL value is total price for the requested GPU count.
    return float(match["uninterruptablePrice"]) / count


def _live_datacenters(row: dict[str, Any]) -> list[str]:
    return [
        str(value["dataCenterId"])
        for value in row.get("datacenter_availability", [])
        if str(value.get("stockStatus", "")).lower() not in {"", "none"}
    ]


def _fragment_price_bounds(rows: dict[str, dict[str, Any]]) -> tuple[float, float]:
    prices = [
        float(rows[gpu_id]["community_price_per_gpu_hour"])
        for gpu_id in FRAGMENT_GPU_IDS
        if gpu_id in rows and rows[gpu_id].get("community_price_per_gpu_hour") is not None
    ]
    if not prices:
        raise ValueError("RunPod inventory contains no price for a validated fragment GPU")
    return min(prices), max(prices)


def _token_budget(checkpoint: Path) -> dict[str, Any]:
    tokenizer, prompt_ids, tokenizer_identity = tokenize_prompt(checkpoint, HEADLINE_PROMPT)
    encoded = tokenizer(HEADLINE_TARGET, add_special_tokens=False, return_tensors=None)["input_ids"]
    target_ids = [int(value) for value in encoded]
    roundtrip = tokenizer.decode(target_ids, skip_special_tokens=True)
    if roundtrip != HEADLINE_TARGET:
        raise ValueError("authoritative Kimi K3 tokenizer did not round-trip headline text")
    safe_max_new_tokens = len(target_ids)
    return {
        "schema_version": "experiment-025-runpod-headline-token-budget-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "prompt": HEADLINE_PROMPT,
        "exact_target_text": HEADLINE_TARGET,
        "prompt_token_ids": prompt_ids,
        "prompt_token_count": len(prompt_ids),
        "target_token_ids": target_ids,
        "target_token_count": len(target_ids),
        "target_roundtrip": roundtrip,
        "minimum_safe_max_new_tokens": safe_max_new_tokens,
        "generation_max_new_tokens": len(target_ids),
        "decoding": {
            "mode": "GREEDY_EXACT_REPETITION",
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "stop_after_target_tokens": len(target_ids),
            "semantic_gate": "decoded text must equal exact_target_text",
        },
        "tokenizer_identity": tokenizer_identity,
        "tokenizer_checkpoint": str(checkpoint.resolve()),
        "tokenizer_revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        "tokenizer_assets": {
            name: sha256_file(checkpoint / name) for name in KIMI_TOKENIZER_ASSETS
        },
    }


def build_preparation_bundle(
    *,
    repo: Path,
    run_root: Path,
    output_directory: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    gpu_inventory = __import__("json").loads(
        (output_directory / "runpod-live-gpu-inventory.json").read_text(encoding="utf-8")
    )
    dc_inventory = __import__("json").loads(
        (output_directory / "runpod-datacenter-inventory.json").read_text(encoding="utf-8")
    )
    account_readiness = __import__("json").loads(
        (output_directory / "runpod-account-readiness.json").read_text(encoding="utf-8")
    )
    rows = _gpu_rows(gpu_inventory)
    backbone_row = rows[BACKBONE_GPU_ID]
    supported_counts = _schedulable_counts(backbone_row, "SECURE")
    if not supported_counts:
        maximum_supported = 1
    else:
        maximum_supported = max(count for count in supported_counts if 1 <= count <= 8)
    backbone_price = _count_price(backbone_row, "SECURE", 1) or float(
        backbone_row["secure_price_per_gpu_hour"]
    )
    fragment_low_price, fragment_high_price = _fragment_price_bounds(rows)
    backbone_datacenters = _live_datacenters(backbone_row)
    fragment_datacenters = sorted(
        {
            datacenter
            for gpu_id in FRAGMENT_GPU_IDS
            if gpu_id in rows
            for datacenter in _live_datacenters(rows[gpu_id])
        }
    )
    roles = load_frozen_roles(repo, run_root)

    candidate_specs = [
        ("MAXIMUM_LIVE_GPU_PER_POD", maximum_supported),
        ("FOUR_GPU_BACKBONE", 4),
        ("TWO_GPU_BACKBONE", 2),
        ("ONE_GPU_NEGATIVE_CONTROL", 1),
    ]
    candidates: list[dict[str, Any]] = []
    for name, count in candidate_specs:
        supported = count in supported_counts
        pods = pack_roles(
            roles,
            gpu_count_per_backbone_pod=count,
            isolate_parent=True,
        )
        metrics = topology_metrics(
            pods,
            backbone_price_per_gpu_hour=backbone_price,
            fragment_price_bounds_per_gpu_hour=(
                fragment_low_price,
                fragment_high_price,
            ),
            supported=supported,
            support_reason=(
                "count-specific GraphQL lowestPrice returned live stock and price"
                if supported
                else "count-specific GraphQL lowestPrice returned no schedulable stock"
            ),
            stock_confidence=(
                "LOW_COUNT_CONFIGURATION_VISIBLE"
                if supported and str(backbone_row.get("cli_stock_status")).lower() == "low"
                else "UNSUPPORTED_CURRENT_SNAPSHOT"
            ),
        )
        pod_storage = [_source_storage_summary(pod) for pod in pods]
        candidates.append(
            {
                "candidate_name": name,
                "backbone_gpu_count_per_multi_gpu_pod": count,
                "parent_isolated": True,
                **metrics,
                "expected_per_pod_model_bytes": {
                    "minimum": min(row["source_weight_bytes"] for row in pod_storage),
                    "maximum": max(row["source_weight_bytes"] for row in pod_storage),
                },
                "expected_per_pod_peak_local_disk_bytes": {
                    "minimum": min(row["peak_bytes"] for row in pod_storage),
                    "maximum": max(row["peak_bytes"] for row in pod_storage),
                },
                "estimated_startup_download_burden_bytes": sum(
                    row["download_bytes"] for row in pod_storage
                ),
                "datacenter_concentration": backbone_datacenters,
            }
        )
    topology_candidates = {
        "schema_version": "experiment-025-runpod-topology-candidates-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS_INVENTORY_CONDITIONED",
        "evidence_class": "PROJECTION_USING_LIVE_READ_ONLY_INVENTORY",
        "supported_backbone_gpu_counts": supported_counts,
        "unsupported_candidates_are_not_deployable": True,
        "candidates": candidates,
    }
    preferred_pods = pack_roles(
        roles,
        gpu_count_per_backbone_pod=maximum_supported,
        isolate_parent=True,
    )
    packed_parent_pods = pack_roles(
        roles,
        gpu_count_per_backbone_pod=maximum_supported,
        isolate_parent=False,
    )
    preferred_metrics = topology_metrics(
        preferred_pods,
        backbone_price_per_gpu_hour=backbone_price,
        fragment_price_bounds_per_gpu_hour=(fragment_low_price, fragment_high_price),
        supported=maximum_supported in supported_counts,
        support_reason="maximum count-specific live Secure RTX 3090 configuration",
        stock_confidence="LOW_STOCK_REQUIRES_P5_SIMULTANEOUS_ALLOCATION_REHEARSAL",
    )
    parent_isolated_count = len([pod for pod in preferred_pods if pod["pod_class"] != "FRAGMENT"])
    parent_packed_count = len([pod for pod in packed_parent_pods if pod["pod_class"] != "FRAGMENT"])
    preferred_topology = {
        "schema_version": "experiment-025-runpod-preferred-topology-v1",
        "generated_at_utc": utc_now(),
        "status": (
            "PREFERRED_PROVISIONAL_LIVE_LOW_STOCK"
            if maximum_supported > 1
            else "SUPPORTED_ONE_GPU_NEGATIVE_CONTROL_ONLY"
        ),
        "evidence_class": "PROJECTION_USING_LIVE_READ_ONLY_INVENTORY",
        "gpu_type": BACKBONE_GPU_ID,
        "cloud_tier": "SECURE",
        "provider_stock_status": backbone_row.get("cli_stock_status"),
        "backbone_price_per_gpu_hour": backbone_price,
        "backbone_gpu_count_per_multi_gpu_pod": maximum_supported,
        "available_gpu_counts_field": _tier(backbone_row, "SECURE").get("availableGpuCounts_field"),
        "count_support_method": "COUNT_SPECIFIC_LOWEST_PRICE_QUERY",
        "layer89_parent": {
            "decision": "ISOLATED_ONE_GPU_SECURE_BACKBONE_POD",
            "reason": (
                "Fragment endpoint generation changes can recycle only the parent instead "
                "of discarding healthy adjacent stages."
            ),
            "isolated_backbone_pod_count": parent_isolated_count,
            "packed_backbone_pod_count": parent_packed_count,
            "incremental_provider_objects": parent_isolated_count - parent_packed_count,
            "incremental_gpu_hour_cost_usd": 0,
        },
        "fragment_pods": {
            "count": 4,
            "gpu_candidates": list(FRAGMENT_GPU_IDS),
            "cloud_tier": "COMMUNITY_RECORDED_AS_FALLBACK_NOT_AUTO_SELECTED",
            "must_have_pairwise_distinct_machine_ids": True,
            "pod_ids_do_not_prove_independence": True,
            "current_cli_located_stock": bool(fragment_datacenters),
        },
        **preferred_metrics,
        "provider_object_reduction_vs_97_vast_objects": 97
        - int(preferred_metrics["total_pod_count"]),
        "pods": preferred_pods,
        "frozen_placement_changed": False,
        "physical_gpu_roles": 97,
    }
    manifests = build_role_manifests(
        preferred_pods,
        backbone_datacenters=backbone_datacenters,
        fragment_datacenters=fragment_datacenters,
        # Public TCP is the minimum-change controller path. Private RunPod DNS
        # remains a configured P2 candidate but is not frozen before measurement.
        endpoint_mode="RUNPOD_PUBLIC_TCP",
    )
    manifest_safety = assert_manifest_safety(manifests)
    if manifest_safety["status"] != "PASS":
        raise RuntimeError(f"RunPod manifest safety failed: {manifest_safety}")
    manifest_artifact = {
        "schema_version": "experiment-025-runpod-pod-role-manifests-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS_PLANNED_RUNTIME_BINDINGS_REMAIN",
        "evidence_class": "LOCAL_DETERMINISTIC_PLAN",
        "worker_image_changed": False,
        "worker_image_digest": IMAGE_DIGEST,
        "manifest_safety": manifest_safety,
        "manifests": manifests,
    }
    storage = storage_plan(manifests)
    distribution = model_distribution_plan(manifests)
    datacenter_plan = {
        "schema_version": "experiment-025-runpod-datacenter-plan-v1",
        "generated_at_utc": utc_now(),
        "status": "PROVISIONAL_REQUIRES_PAID_ALLOCATION_REHEARSAL",
        "evidence_class": "LIVE_INVENTORY_PROJECTION",
        "preferred_backbone_datacenter": backbone_datacenters[0] if backbone_datacenters else None,
        "alternate_backbone_datacenters": backbone_datacenters[1:],
        "backbone_capacity": {
            "gpu_type": BACKBONE_GPU_ID,
            "supported_gpu_counts": supported_counts,
            "stock_status": backbone_row.get("cli_stock_status"),
            "datacenters": backbone_datacenters,
        },
        "fragment_capacity": {
            "gpu_types": list(FRAGMENT_GPU_IDS),
            "located_datacenters": fragment_datacenters,
            "status": (
                "CLI_LOCATED_STOCK_PRESENT"
                if fragment_datacenters
                else "GRAPHQL_LOW_PRICE_VISIBLE_BUT_CLI_HAS_NO_LOCATED_AVAILABLE_STOCK"
            ),
            "latency_risk": (
                "UNKNOWN_UNTIL_P2/P4" if fragment_datacenters else "BLOCKED_BY_LIVE_LOCATION"
            ),
        },
        "strategy": [
            "enough multi-GPU RTX 3090 capacity",
            "same Secure datacenter",
            "fewer Pods",
            "measured P2 network quality",
            "live price",
        ],
        "location_claim_policy": "RUNPOD_REPORTED_DATACENTER_HOST_LOCATION_METADATA_ONLY",
        "raw_datacenter_snapshot_sha256": canonical_sha256(dc_inventory["datacenters"]),
    }
    network_plan = {
        "schema_version": "experiment-025-runpod-network-plan-v1",
        "generated_at_utc": utc_now(),
        "status": "REQUIRES_PAID_TWO_POD_CANARY",
        "evidence_class": "DESIGN_ONLY_NOT_PHYSICALLY_VALIDATED",
        "modes": {
            "RUNPOD_GLOBAL_PRIVATE": {
                "endpoint": "<POD_ID>.runpod.internal:<internal_worker_port>",
                "documented_interpod_mbps": GLOBAL_NETWORK_MEGABITS_PER_SECOND,
                "controller_on_windows_reachable": False,
                "worker_to_worker_candidate": True,
            },
            "RUNPOD_PUBLIC_TCP": {
                "endpoint": "<publicIp>:<portMappings[internal_worker_port]>",
                "controller_on_windows_reachable": True,
                "secure_cloud_documentation": "Secure Cloud always has a public IP; mappings still require P2 proof.",
            },
            "POD_LOCAL": {
                "endpoint": "127.0.0.1:<internal_worker_port>",
                "use": "adjacent logical stages inside the same Pod where protocol routing permits",
            },
        },
        "controller_options": [
            {
                "option": "PUBLIC_CONTROL_PUBLIC_DATA",
                "preference": 1,
                "reason": "minimum-change local controller if public TCP passes P2",
            },
            {
                "option": "HYBRID_PUBLIC_CONTROL_PRIVATE_WORKER_DATA",
                "preference": 2,
                "reason": "only if P2 proves both identities and current protocol separation is safe",
            },
            {
                "option": "CONTROLLER_COLOCATED_ON_BACKBONE_POD",
                "preference": 3,
                "reason": "fallback only; controller performs no model compute",
            },
        ],
        "selected_mode": None,
        "selection_gate": "P2_TWO_POD_NETWORK_CANARY",
        "tls_auth_required_all_modes": True,
    }
    timeout_plan = {
        "status": "PROVISIONAL_PENDING_P1_P3_P5",
        "progress_events": [
            "POD_ALLOCATED",
            "CONTAINER_RUNNING",
            "MODEL_DOWNLOAD_BYTES",
            "MODEL_DOWNLOAD_COMPLETE",
            "SNAPSHOT_ACTIVATED",
            "GPU_LOAD",
            "WORKER_LISTENING",
            "WORKER_READY",
        ],
        "no_progress_timeout_seconds": 900,
        "hard_pod_bootstrap_timeout_seconds": 7200,
        "full_acquisition_hard_limit_seconds": 7200,
        "inference_reserve_seconds": 1800,
        "cleanup_reserve_seconds": 900,
        "healthy_progress_extends_no_progress_deadline": True,
        "values_are_assumptions_not_measurements": True,
    }
    network_volume_decision = {
        "selected": False,
        "decision": "DIRECT_SELECTIVE_CHECKPOINT_DOWNLOAD_FOR_FIRST_RUNPOD_ATTEMPT",
        "network_volumes_created": 0,
        "direct_download": {
            "pros": [
                "parallel per-Pod acquisition",
                "no provider data-transfer fee under current documentation",
                "no datacenter lock",
                "preserves physically validated E025 acquisition semantics",
            ]
        },
        "network_volume": {
            "potential_pros": ["single pre-stage", "persistent replacement data"],
            "risks": [
                "datacenter restriction",
                "shared I/O bottleneck",
                "storage cost",
                "new component before public proof",
                "many-Pod contention",
            ],
            "future_hook": "create payload supports optional networkVolumeId without changing provider records",
        },
    }
    account = account_readiness["account"]
    paid_budget = paid_canary_budget(
        backbone_price=backbone_price,
        fragment_low_price=fragment_low_price,
        fragment_high_price=fragment_high_price,
        preferred_backbone_gpu_count_per_pod=maximum_supported,
        fragment_stock_located=bool(fragment_datacenters),
        headline_storage_gb=int(storage["total_requested_container_disk_gb"]),
    )
    headline_cost = headline_budget(
        backbone_price=backbone_price,
        fragment_low_price=fragment_low_price,
        fragment_high_price=fragment_high_price,
        storage_gb=int(storage["total_requested_container_disk_gb"]),
        account_balance=(
            float(account["client_balance_usd"])
            if account.get("client_balance_usd") is not None
            else None
        ),
        account_hourly_limit=(
            float(account["spend_limit_per_hour_usd"])
            if account.get("spend_limit_per_hour_usd") is not None
            else None
        ),
    )
    token_budget = _token_budget(checkpoint)
    simulation = simulate_full_control_flow(manifests)
    blockers: list[dict[str, Any]] = []
    account_balance = (
        float(account["client_balance_usd"])
        if account.get("client_balance_usd") is not None
        else None
    )
    account_hourly_limit = (
        float(account["spend_limit_per_hour_usd"])
        if account.get("spend_limit_per_hour_usd") is not None
        else None
    )
    canary_credit_required = float(paid_budget["budget_with_safety_usd"]["high"])
    if account_balance is None:
        blockers.append(
            {
                "code": "ACCOUNT_CREDIT_UNAVAILABLE",
                "detail": "The read-only account probe did not expose current RunPod credit.",
                "resolution": "Confirm credit manually before P1.",
            }
        )
    elif account_balance <= 0:
        blockers.append(
            {
                "code": "ACCOUNT_CREDIT_ZERO",
                "detail": "RunPod requires funded credit before any on-demand Pod can be created.",
                "resolution": f"Fund at least ${paid_budget['minimum_recommended_account_credit_usd']} before P1.",
            }
        )
    elif account_balance < canary_credit_required:
        blockers.append(
            {
                "code": "ACCOUNT_CREDIT_BELOW_CANARY_SAFETY_BUDGET",
                "detail": (
                    f"Current credit ${account_balance:.2f} is below the P1-P5 safety "
                    f"projection ${canary_credit_required:.2f}."
                ),
                "resolution": (
                    f"Fund at least ${paid_budget['minimum_recommended_account_credit_usd']} "
                    "before P1."
                ),
            }
        )
    if account_hourly_limit is None:
        blockers.append(
            {
                "code": "ACCOUNT_HOURLY_LIMIT_UNAVAILABLE",
                "detail": "The read-only account probe did not expose the hourly spending limit.",
                "resolution": "Confirm the hourly limit manually before P1.",
            }
        )
    elif not headline_cost["account"]["hourly_limit_fits"]:
        blockers.append(
            {
                "code": "ACCOUNT_HOURLY_LIMIT_INSUFFICIENT",
                "detail": (
                    f"Current ${account_hourly_limit:.2f}/hour limit is below the projected "
                    f"${headline_cost['headline_hourly_usd']['high']:.2f}/hour fleet rate."
                ),
                "resolution": "Raise the RunPod hourly spending limit before P1.",
            }
        )
    if not fragment_datacenters:
        blockers.append(
            {
                "code": "NO_CLI_LOCATED_VALIDATED_AMPERE_FRAGMENT_STOCK",
                "detail": (
                    "GraphQL count-specific lowestPrice says Low for validated small Ampere "
                    "SKUs, but authenticated CLI inventory exposes no available datacenter."
                ),
                "resolution": "Re-run inventory until four single-GPU candidates have located stock, then use P4/P5 machineId gates.",
            }
        )
    if maximum_supported <= 1:
        blockers.append(
            {
                "code": "NO_MATERIAL_MULTI_GPU_BACKBONE_CONFIGURATION",
                "detail": "Current inventory does not reduce provider objects materially.",
            }
        )
    status_classification = (
        "BLOCKED_BEFORE_PAID_RUNPOD_CANARIES" if blockers else "READY_FOR_PAID_RUNPOD_CANARIES"
    )
    preferred_topology["preparation_blockers"] = blockers
    preferred_topology["preparation_status"] = status_classification
    manifest_artifact["provider_payload_notes"] = {
        "runtime_secret_values_persisted": False,
        "runtime_secret_injection": "EPHEMERAL_IN_MEMORY_AT_PAID_CREATE",
        "machine_id": (
            "Controller joins authoritative REST machineId to RUNPOD_POD_ID; "
            "E025_MACHINE_ID is never fabricated from Pod ID."
        ),
        "provider_side_ttl": (
            "runpodctl exposes --terminate-after, but REST v1 create does not. "
            "The chosen REST lifecycle therefore requires the independent watchdog."
        ),
    }
    artifact_values = {
        "runpod-topology-candidates.json": topology_candidates,
        "runpod-preferred-topology.json": preferred_topology,
        "runpod-pod-role-manifests.json": manifest_artifact,
        "runpod-pod-storage-plan.json": storage,
        "runpod-model-distribution-plan.json": distribution,
        "runpod-datacenter-plan.json": datacenter_plan,
        "runpod-network-plan.json": {
            **network_plan,
            "timeouts": timeout_plan,
            "network_volume_decision": network_volume_decision,
        },
        "runpod-paid-canary-budget.json": paid_budget,
        "runpod-headline-budget.json": headline_cost,
        "headline-token-budget.json": token_budget,
        "runpod-dry-run-results.json": simulation,
    }
    for name, value in artifact_values.items():
        atomic_write_json(output_directory / name, value)
    return {
        "status": status_classification,
        "blockers": blockers,
        "preferred_topology": preferred_topology,
        "manifests": manifest_artifact,
        "storage": storage,
        "distribution": distribution,
        "datacenter_plan": datacenter_plan,
        "network_plan": network_plan,
        "paid_budget": paid_budget,
        "headline_budget": headline_cost,
        "token_budget": token_budget,
        "simulation": simulation,
        "artifacts": sorted(artifact_values),
    }


def _source_storage_summary(pod: dict[str, Any]) -> dict[str, int]:
    return {
        "source_weight_bytes": sum(int(role["source_weight_bytes"]) for role in pod["roles"]),
        "download_bytes": sum(int(role["download_bytes_cold_cache"]) for role in pod["roles"]),
        "peak_bytes": sum(int(role["temporary_disk_bytes"]) for role in pod["roles"]),
    }


def write_handoff(
    *,
    output_directory: Path,
    preparation: dict[str, Any],
    inventory_timestamp: str,
) -> Path:
    topology = preparation["preferred_topology"]
    paid = preparation["paid_budget"]
    headline = preparation["headline_budget"]
    datacenter = preparation["datacenter_plan"]
    status = preparation["status"]
    blocker_lines = (
        "\n".join(f"- `{row['code']}` -- {row['detail']}" for row in preparation["blockers"])
        or "- None."
    )
    gpu_count_per_pod = int(topology["backbone_gpu_count_per_multi_gpu_pod"])
    packing_description = (
        "one GPU per backbone Pod (current negative-control-only stock)"
        if gpu_count_per_pod == 1
        else f"{gpu_count_per_pod} GPUs per multi-GPU backbone Pod"
    )
    text = f"""# E025 RunPod paid-run handoff

Preparation status: **`{status}`**

## Frozen facts

- Model: `moonshotai/Kimi-K3` at `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.
- Checkpoint fingerprint: `25162130a11904bac1220a7d654a3f7dfd616ea5f4035d488e40ac74ddea8f94`.
- Worker image: `{IMAGE_REFERENCE}` (unchanged).
- Physical placement SHA-256: `015c7bc6ce4f2724123df2315607e7c261ad4aba2dd1b9def5266bfa2ac334f5`.
- Frozen topology: 93 backbone GPU roles plus four Layer 89 fragment roles; 497,052 tensors and 1,559,965,606,912 source bytes, with zero duplicate/unassigned ownership.
- Complete Layer 89 runtime peak: 20,556,349,440 bytes (19.14459228515625 GiB). Its four real expert fragments already fit and ran on independent 8-12 GB consumer Ampere machines.
- Existing E025 Stage 1/Stage 2 physical evidence remains authoritative; this preparation did not rerun it.

## RunPod plan

- Backbone: `{BACKBONE_GPU_ID}`, Secure Cloud, {packing_description}.
- Provider objects: {topology["total_pod_count"]} Pods for 97 GPUs ({topology["backbone_pod_count"]} backbone/parent plus four single-GPU fragments), a reduction of {topology["provider_object_reduction_vs_97_vast_objects"]} objects from the Vast layout.
- Physical-host claim: four fragment `machineId` values must be pairwise distinct. Total distinct `machineId` values are measured after allocation; Pod IDs never prove host identity.
- Layer 89 parent: isolated on one Secure RTX 3090 Pod so endpoint-generation changes recycle only that Pod.
- Preferred backbone datacenter: `{datacenter["preferred_backbone_datacenter"]}`; alternates: `{datacenter["alternate_backbone_datacenters"]}`.
- Networking: `RUNPOD_PUBLIC_TCP` and `RUNPOD_GLOBAL_PRIVATE` are implemented as endpoint modes; selection remains `REQUIRES_PAID_TWO_POD_CANARY`.
- Distribution: direct selective checkpoint download with unchanged worker-scoped caches; no network volume.
- Container disk: exact per-Pod values are in `runpod-pod-storage-plan.json`; total requested is {preparation["storage"]["total_requested_container_disk_gb"]} GB.

## Live inventory snapshot

- Captured: `{inventory_timestamp}`.
- Secure RTX 3090 stock: {topology["provider_stock_status"]}; count-specific GraphQL scheduling visible for {gpu_count_per_pod}-GPU Pods, at ${topology["backbone_price_per_gpu_hour"]:.2f}/GPU-hour.
- Validated small Ampere fragment inventory: GraphQL Low-price records exist, but the authenticated CLI snapshot has no located available datacenter.
- Account credit: ${headline["account"]["current_credit_usd"]}; hourly spend limit: ${headline["account"]["hourly_limit_usd"]}.

## Current blockers

{blocker_lines}

## Paid gates remaining

1. P1 single Pod.
2. P2 network.
3. P3 multi-GPU.
4. P4 sub-layer.
5. P5 full acquisition rehearsal.
6. Headline run.

## Cost

- P1-P5 projected base: ${paid["base_total_usd"]["low"]:.2f}-${paid["base_total_usd"]["high"]:.2f}; safety budget: ${paid["budget_with_safety_usd"]["low"]:.2f}-${paid["budget_with_safety_usd"]["high"]:.2f}.
- Headline projected hourly: ${headline["headline_hourly_usd"]["low"]:.2f}-${headline["headline_hourly_usd"]["high"]:.2f}.
- Headline {headline["hard_window_hours"]:.2f}-hour hard window with safety: ${headline["hard_budget_with_safety_usd"]["low"]:.2f}-${headline["hard_budget_with_safety_usd"]["high"]:.2f}.
- Recommended headline account credit: ${headline["minimum_recommended_account_credit_usd"]}. Current balance is ${headline["account"]["current_credit_usd"]}; the hourly limit is ${headline["account"]["hourly_limit_usd"]}.

## Exact future commands

From the repository root in PowerShell:

```powershell
# Refresh inventory and regenerate plans; guaranteed read-only.
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod_inventory.py
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode prepare

# Review all P1-P5 requests without creating anything.
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode paid-canaries

# After funding, stock refresh, and explicit authorization, run one gate at a time.
$runpodSecret = Read-Host 'RunPod API key for this PowerShell process' -AsSecureString
$env:RUNPOD_API_KEY = [System.Net.NetworkCredential]::new('', $runpodSecret).Password
$env:E025_RUNPOD_ALLOW_RENTAL = 'YES'
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode p1 --allow-paid-run
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode p2 --allow-paid-run
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode p3 --allow-paid-run
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode p4 --allow-paid-run
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode p5 --allow-paid-run

# Only after P1-P5 receipts pass:
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode headline --allow-paid-run

# Independent emergency permanent deletion for this run only:
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod_cleanup.py --run-id {RUN_ID} --ledger artifacts/runs/experiment-025-{RUN_ID}/rental/runpod/pod-ledger.jsonl --allow-paid-run

# Read-only final verification:
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode verify-zero

# Remove ephemeral authorization and API-key material from this shell.
Remove-Item Env:E025_RUNPOD_ALLOW_RENTAL -ErrorAction SilentlyContinue
Remove-Item Env:RUNPOD_API_KEY -ErrorAction SilentlyContinue
```

Do not place the API key in a command, artifact, source file, or Git commit. Use the existing `runpodctl` configuration; the Python paid path accepts `RUNPOD_API_KEY` only from the process environment.
"""
    path = output_directory / "HANDOFF_FOR_PAID_RUN.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


__all__ = [
    "HEADLINE_PROMPT",
    "HEADLINE_TARGET",
    "build_preparation_bundle",
    "write_handoff",
]
