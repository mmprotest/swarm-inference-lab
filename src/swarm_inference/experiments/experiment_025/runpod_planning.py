"""Deterministic RunPod packing, manifest, storage, and budget planning for E025."""

from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .io import canonical_sha256, read_json, utc_now
from .providers.base import CloudTier
from .providers.runpod import SECRET_ENVIRONMENT_KEYS, redact_provider_payload
from .supervisor import BASE_PORT, MAX_WORKERS_PER_INSTANCE

RUN_ID = "20260819T013016Z"
RUN_ROOT_NAME = f"experiment-025-{RUN_ID}"
BACKBONE_GPU_ID = "NVIDIA GeForce RTX 3090"
BACKBONE_GPU_DISPLAY = "RTX 3090"
FRAGMENT_GPU_IDS = (
    "NVIDIA GeForce RTX 3070",
    "NVIDIA GeForce RTX 3080",
    "NVIDIA GeForce RTX 3080 Ti",
)
IMAGE_DIGEST = "sha256:46cad5a031b98aeecdee9ba471e2e8338eb8e9e485d1d2e90a1505f8890ee8e4"
IMAGE_TAG = "ghcr.io/mmprotest/swarm-inference-lab:e025-20260819t013016z"
IMAGE_REFERENCE = f"ghcr.io/mmprotest/swarm-inference-lab@{IMAGE_DIGEST}"
IMAGE_SIZE_BYTES = 4_545_330_397
CONTAINER_FIXED_MARGIN_BYTES = 10 * 1024**3
CONTAINER_SAFETY_MULTIPLIER = 1.25
CONTAINER_STORAGE_USD_PER_GB_MONTH = 0.10
GLOBAL_NETWORK_MEGABITS_PER_SECOND = 100


@dataclass(frozen=True, slots=True)
class FrozenRole:
    worker_id: str
    role: str
    layer: int
    worker_index: int | None
    assignment_sha256: str
    source_weight_bytes: int
    download_bytes_cold_cache: int
    assigned_tensor_bytes: int
    temporary_disk_bytes: int
    source_shards: tuple[dict[str, Any], ...]
    maximum_context: int = 64


def _worker_layer(worker_id: str) -> int:
    stage = re.fullmatch(r"e025-stage-(\d{3})(?:-parent)?", worker_id)
    if stage:
        return int(stage.group(1))
    fragment = re.fullmatch(r"e025-layer-(\d{3})-sub-\d{2}", worker_id)
    if fragment:
        return int(fragment.group(1))
    raise ValueError(f"unrecognized frozen E025 worker ID: {worker_id}")


def _worker_index(worker_id: str, role: str) -> int | None:
    if role != "SUB_LAYER_WORKER":
        return None
    return int(worker_id.rsplit("-", maxsplit=1)[1])


def load_frozen_roles(repo: Path, run_root: Path) -> list[FrozenRole]:
    """Load the 97 immutable logical roles without parsing the 253 MB placement."""

    image_index = read_json(repo / "deployment" / "e025_context" / "index.json")
    distribution = read_json(run_root / "preflight" / "checkpoint-distribution.json")
    distribution_by_worker = {str(row["worker_id"]): row for row in distribution["workers"]}
    roles: list[FrozenRole] = []
    for index_row in image_index["workers"]:
        worker_id = str(index_row["worker_id"])
        role = str(index_row["role"])
        distribution_row = distribution_by_worker[worker_id]
        template_path = (
            repo / "deployment" / "e025_context" / "workers" / worker_id / "worker-template.json"
        )
        template = read_json(template_path)
        roles.append(
            FrozenRole(
                worker_id=worker_id,
                role=role,
                layer=_worker_layer(worker_id),
                worker_index=_worker_index(worker_id, role),
                assignment_sha256=str(template["assignment_sha256"]),
                source_weight_bytes=int(template["source_weight_bytes"]),
                download_bytes_cold_cache=int(distribution_row["download_bytes_cold_cache"]),
                assigned_tensor_bytes=int(distribution_row["assigned_tensor_bytes"]),
                temporary_disk_bytes=int(distribution_row["temporary_disk_bytes"]),
                source_shards=tuple(
                    {
                        "name": str(shard["name"]),
                        "bytes": int(shard["file_bytes"]),
                        "sha256": str(shard["sha256"]),
                    }
                    for shard in distribution_row["source_shards"]
                ),
                maximum_context=int(template.get("maximum_context", 64)),
            )
        )
    if len(roles) != 97:
        raise ValueError(f"frozen E025 role count differs from 97: {len(roles)}")
    if len({role.worker_id for role in roles}) != 97:
        raise ValueError("frozen E025 worker IDs are not unique")
    backbone = [role for role in roles if role.role != "SUB_LAYER_WORKER"]
    fragments = [role for role in roles if role.role == "SUB_LAYER_WORKER"]
    if len(backbone) != 93 or sorted(role.layer for role in backbone) != list(range(93)):
        raise ValueError("frozen E025 backbone does not cover exactly layers 0 through 92")
    if len(fragments) != 4 or {role.layer for role in fragments} != {89}:
        raise ValueError("frozen E025 fragment set differs from four Layer 89 roles")
    return roles


def encode_worker_specs(specs: list[dict[str, Any]]) -> str:
    if not 1 <= len(specs) <= MAX_WORKERS_PER_INSTANCE:
        raise ValueError("one to eight E025 worker specs are required")
    payload = json.dumps(specs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


def decode_worker_specs(encoded: str) -> list[dict[str, Any]]:
    value = json.loads(base64.b64decode(encoded, validate=True))
    if not isinstance(value, list):
        raise ValueError("E025 worker specs payload is not a list")
    return value


def _chunks(rows: list[FrozenRole], size: int) -> list[list[FrozenRole]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _pod_name_for_roles(run_id: str, roles: list[FrozenRole]) -> str:
    if len(roles) == 1 and roles[0].role == "SUB_LAYER_PARENT":
        return f"e025-rp-{run_id}-layer89-parent"
    if len(roles) == 1 and roles[0].role == "SUB_LAYER_WORKER":
        return f"e025-rp-{run_id}-layer89-frag-{roles[0].worker_index}"
    return f"e025-rp-{run_id}-bb-{roles[0].layer:03d}-{roles[-1].layer:03d}"


def pack_roles(
    roles: list[FrozenRole],
    *,
    gpu_count_per_backbone_pod: int,
    isolate_parent: bool = True,
    run_id: str = RUN_ID,
) -> list[dict[str, Any]]:
    if not 1 <= gpu_count_per_backbone_pod <= MAX_WORKERS_PER_INSTANCE:
        raise ValueError("RunPod backbone packing must use one to eight GPUs per Pod")
    backbone = sorted(
        (role for role in roles if role.role != "SUB_LAYER_WORKER"),
        key=lambda role: role.layer,
    )
    fragments = sorted(
        (role for role in roles if role.role == "SUB_LAYER_WORKER"),
        key=lambda role: int(role.worker_index or 0),
    )
    parent = next(role for role in backbone if role.role == "SUB_LAYER_PARENT")
    groups: list[list[FrozenRole]]
    if isolate_parent:
        before = [role for role in backbone if role.layer < parent.layer]
        after = [role for role in backbone if role.layer > parent.layer]
        groups = [
            *_chunks(before, gpu_count_per_backbone_pod),
            [parent],
            *_chunks(after, gpu_count_per_backbone_pod),
        ]
    else:
        groups = _chunks(backbone, gpu_count_per_backbone_pod)
    groups.extend([[fragment] for fragment in fragments])
    pods: list[dict[str, Any]] = []
    for roles_in_pod in groups:
        name = _pod_name_for_roles(run_id, roles_in_pod)
        role_class = (
            "FRAGMENT"
            if roles_in_pod[0].role == "SUB_LAYER_WORKER"
            else "LAYER89_PARENT"
            if len(roles_in_pod) == 1 and roles_in_pod[0].role == "SUB_LAYER_PARENT"
            else "BACKBONE"
        )
        pods.append(
            {
                "logical_pod_id": name,
                "pod_name": name,
                "pod_class": role_class,
                "roles": [asdict(role) for role in roles_in_pod],
                "requested_gpu_count": len(roles_in_pod),
            }
        )
    return pods


def topology_metrics(
    pods: list[dict[str, Any]],
    *,
    backbone_price_per_gpu_hour: float | None,
    fragment_price_bounds_per_gpu_hour: tuple[float | None, float | None],
    supported: bool,
    support_reason: str,
    stock_confidence: str,
) -> dict[str, Any]:
    backbone_pods = [pod for pod in pods if pod["pod_class"] != "FRAGMENT"]
    fragment_pods = [pod for pod in pods if pod["pod_class"] == "FRAGMENT"]
    parent_pods = [pod for pod in pods if pod["pod_class"] == "LAYER89_PARENT"]
    layer_to_pod = {
        int(role["layer"]): str(pod["logical_pod_id"])
        for pod in backbone_pods
        for role in pod["roles"]
    }
    intra = sum(layer_to_pod[layer] == layer_to_pod[layer + 1] for layer in range(92))
    cross = 92 - intra
    backbone_hourly = (
        93 * backbone_price_per_gpu_hour if backbone_price_per_gpu_hour is not None else None
    )
    fragment_low, fragment_high = fragment_price_bounds_per_gpu_hour
    low = (
        backbone_hourly + 4 * fragment_low
        if backbone_hourly is not None and fragment_low is not None
        else None
    )
    high = (
        backbone_hourly + 4 * fragment_high
        if backbone_hourly is not None and fragment_high is not None
        else None
    )
    return {
        "supported": supported,
        "support_reason": support_reason,
        "backbone_gpu_count": 93,
        "backbone_pod_count": len(backbone_pods),
        "fragment_pod_count": len(fragment_pods),
        "parent_pod_count": len(parent_pods),
        "total_pod_count": len(pods),
        "total_gpu_count": 97,
        "required_unique_physical_host_count_minimum": 4,
        "pod_count_if_each_provider_object_lands_on_a_distinct_host": len(pods),
        "physical_host_count_note": (
            "RunPod may place distinct Pods on the same machine; only the four fragment "
            "machineId values are required to be pairwise distinct. Total distinct hosts "
            "is measured after allocation, never inferred from Pod IDs."
        ),
        "cross_pod_backbone_boundaries": cross,
        "intra_pod_backbone_boundaries": intra,
        "hourly_compute_estimate_usd": {"low": low, "high": high},
        "required_internal_ports": list(range(BASE_PORT, BASE_PORT + 8)),
        "stock_confidence": stock_confidence,
    }


def _worker_spec(role: dict[str, Any], slot: int, expert_endpoints: Any = None) -> dict[str, Any]:
    return {
        "worker_id": str(role["worker_id"]),
        "gpu_slot": slot,
        "port": BASE_PORT + slot,
        "maximum_context": int(role.get("maximum_context", 64)),
        "expert_endpoints": expert_endpoints,
    }


def _source_storage(pod: dict[str, Any]) -> dict[str, Any]:
    roles = pod["roles"]
    shard_union: dict[str, dict[str, Any]] = {}
    for role in roles:
        for shard in role["source_shards"]:
            shard_union[str(shard["name"])] = shard
    worker_scoped_download = sum(int(role["download_bytes_cold_cache"]) for role in roles)
    union_download = sum(int(row["bytes"]) for row in shard_union.values())
    final_packages = sum(int(role["assigned_tensor_bytes"]) for role in roles)
    acquisition_peak = sum(int(role["temporary_disk_bytes"]) for role in roles)
    pre_margin = acquisition_peak + IMAGE_SIZE_BYTES + CONTAINER_FIXED_MARGIN_BYTES
    requested_disk_gb = math.ceil(pre_margin * CONTAINER_SAFETY_MULTIPLIER / 1_000_000_000)
    return {
        "source_weight_ownership_bytes": sum(int(role["source_weight_bytes"]) for role in roles),
        "source_shard_references": sum(len(role["source_shards"]) for role in roles),
        "unique_source_shard_count": len(shard_union),
        "unique_source_shards": sorted(shard_union),
        "download_bytes_current_worker_scoped_cache": worker_scoped_download,
        "download_bytes_hypothetical_safe_shared_cache": union_download,
        "overlap_bytes_current_implementation": worker_scoped_download - union_download,
        "final_package_bytes": final_packages,
        "worker_acquisition_peak_bytes": acquisition_peak,
        "image_bytes": IMAGE_SIZE_BYTES,
        "fixed_logs_manifests_filesystem_margin_bytes": CONTAINER_FIXED_MARGIN_BYTES,
        "pre_safety_peak_bytes": pre_margin,
        "safety_multiplier": CONTAINER_SAFETY_MULTIPLIER,
        "requested_container_disk_gb": requested_disk_gb,
        "network_volume_bytes": 0,
    }


def _duration_seconds(download_bytes: int, megabits_per_second: float) -> float:
    return download_bytes * 8 / (megabits_per_second * 1_000_000)


def manifest_for_pod(
    pod: dict[str, Any],
    *,
    backbone_datacenters: list[str],
    fragment_datacenters: list[str],
    parent_endpoint_templates: list[dict[str, Any]],
    global_networking: bool,
) -> dict[str, Any]:
    is_fragment = pod["pod_class"] == "FRAGMENT"
    gpu_id = "INVENTORY_SELECTED_VALIDATED_AMPERE_FRAGMENT_GPU" if is_fragment else BACKBONE_GPU_ID
    tier = CloudTier.COMMUNITY if is_fragment else CloudTier.SECURE
    datacenters = fragment_datacenters if is_fragment else backbone_datacenters
    specs: list[dict[str, Any]] = []
    for slot, role in enumerate(pod["roles"]):
        endpoints = parent_endpoint_templates if role["role"] == "SUB_LAYER_PARENT" else None
        specs.append(_worker_spec(role, slot, endpoints))
    encoded = encode_worker_specs(specs)
    storage = _source_storage(pod)
    non_secret_environment = {
        "E025_RUN_ID": RUN_ID,
        "E025_IMAGE_DIGEST": IMAGE_DIGEST,
        "E025_WORKER_SPECS_B64": encoded,
    }
    secret_environment = {
        name: {"secret_reference": f"EPHEMERAL_RUNTIME:{name}"}
        for name in sorted(
            {
                "E025_RUN_CREDENTIAL_B64",
                "E025_TLS_CERT_B64",
                "E025_TLS_KEY_B64",
            }
        )
    }
    ports = [int(spec["port"]) for spec in specs]
    create_payload = {
        "name": pod["pod_name"],
        "cloudType": tier.value,
        "computeType": "GPU",
        "gpuTypeIds": [gpu_id],
        "gpuTypePriority": "custom",
        "gpuCount": len(specs),
        "dataCenterIds": datacenters,
        "dataCenterPriority": "custom",
        "imageName": IMAGE_TAG,
        "containerDiskInGb": storage["requested_container_disk_gb"],
        "volumeInGb": 0,
        "volumeMountPath": "/workspace",
        "ports": [f"{port}/tcp" for port in ports],
        "supportPublicIp": True,
        "globalNetworking": global_networking and tier is CloudTier.SECURE,
        "interruptible": False,
        "locked": False,
        "allowedCudaVersions": ["13.0"],
        "minDownloadMbps": 100,
        "minUploadMbps": 100,
        "minDiskBandwidthMBps": 200,
        "minRAMPerGPU": 16,
        "minVCPUPerGPU": 4,
        "env": {**non_secret_environment, **secret_environment},
    }
    return {
        **pod,
        "requested_gpu_type": gpu_id,
        "cloud_tier": tier.value,
        "preferred_datacenters": datacenters,
        "gpu_slots": [int(spec["gpu_slot"]) for spec in specs],
        "internal_serving_ports": ports,
        "worker_specs": specs,
        "E025_WORKER_SPECS_B64": encoded,
        "worker_specs_roundtrip_sha256": canonical_sha256(decode_worker_specs(encoded)),
        "fragment_endpoint_generation": (0 if pod["pod_class"] == "LAYER89_PARENT" else None),
        "parent_endpoint_binding": (
            "RUNTIME_REBIND_REQUIRED_AFTER_FRAGMENT_POD_IDS_OR_PUBLIC_MAPPINGS"
            if pod["pod_class"] == "LAYER89_PARENT"
            else None
        ),
        "storage": storage,
        "environment": {
            "non_secret": non_secret_environment,
            "secrets": secret_environment,
            "runpod_provided_metadata_available": [
                "RUNPOD_POD_ID",
                "RUNPOD_DC_ID",
                "RUNPOD_POD_HOSTNAME",
                "RUNPOD_GPU_COUNT",
                "RUNPOD_CPU_COUNT",
                "RUNPOD_PUBLIC_IP",
                "CUDA_VERSION",
            ],
            "identity_normalization": (
                "The controller joins authoritative REST Pod ID and machineId to each "
                "worker/GPU slot. The unchanged worker image may report unknown for its "
                "legacy Vast identity fields; no RunPod machine identity is fabricated."
            ),
            "runpod_api_key_logged": False,
            "environment_variable_count": len(non_secret_environment) + len(secret_environment),
            "provider_limit": 50,
        },
        "image": {
            "request_reference": IMAGE_TAG,
            "expected_digest": IMAGE_DIGEST,
            "immutable_digest_reference": IMAGE_REFERENCE,
            "reason_for_tag": (
                "REST v1 documents imageName as an image tag; this unique frozen tag "
                "already resolves to the validated digest. Worker READY still verifies digest."
            ),
        },
        "create_payload_redacted": redact_provider_payload(create_payload),
        "network_volume": {
            "selected": False,
            "networkVolumeId": None,
            "method": "DIRECT_SELECTIVE_CHECKPOINT_DOWNLOAD",
        },
    }


def build_role_manifests(
    pods: list[dict[str, Any]],
    *,
    backbone_datacenters: list[str],
    fragment_datacenters: list[str],
    endpoint_mode: str = "RUNPOD_GLOBAL_PRIVATE",
) -> list[dict[str, Any]]:
    endpoint_templates = [
        {
            "worker_id": f"e025-layer-089-sub-{index:02d}",
            "worker_index": index,
            "host": (
                f"<RUNPOD_POD_ID:e025-rp-{RUN_ID}-layer89-frag-{index}>.runpod.internal"
                if endpoint_mode == "RUNPOD_GLOBAL_PRIVATE"
                else f"<RUNPOD_PUBLIC_IP:e025-rp-{RUN_ID}-layer89-frag-{index}>"
            ),
            "port": (
                BASE_PORT
                if endpoint_mode == "RUNPOD_GLOBAL_PRIVATE"
                else f"<RUNPOD_PUBLIC_TCP_PORT:{BASE_PORT}>"
            ),
            "timeout_seconds": 180.0,
            "fragment_endpoint_generation": 0,
        }
        for index in range(4)
    ]
    return [
        manifest_for_pod(
            pod,
            backbone_datacenters=backbone_datacenters,
            fragment_datacenters=fragment_datacenters,
            parent_endpoint_templates=endpoint_templates,
            global_networking=endpoint_mode == "RUNPOD_GLOBAL_PRIVATE",
        )
        for pod in pods
    ]


def model_distribution_plan(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        storage = manifest["storage"]
        download = int(storage["download_bytes_current_worker_scoped_cache"])
        rows.append(
            {
                "logical_pod_id": manifest["logical_pod_id"],
                "pod_class": manifest["pod_class"],
                "worker_ids": [str(role["worker_id"]) for role in manifest["roles"]],
                **storage,
                "download_duration_seconds": {
                    "provider_minimum_100_mbps": _duration_seconds(download, 100),
                    "conservative_250_mbps": _duration_seconds(download, 250),
                    "planning_current_500_mbps_unverified": _duration_seconds(download, 500),
                },
                "gpu_load_duration": {
                    "status": "PROVISIONAL_TRANSFER_FROM_PRIOR_E025_EVIDENCE",
                    "seconds": 180,
                    "must_calibrate_in": "P1_AND_P3",
                },
            }
        )
    return {
        "schema_version": "experiment-025-runpod-model-distribution-v1",
        "generated_at_utc": utc_now(),
        "evidence_class": "PROJECTION_FROM_FROZEN_ASSIGNMENT",
        "status": "PASS_PLANNED_NOT_PHYSICALLY_RUNPOD_VALIDATED",
        "acquisition_method": "DIRECT_SELECTIVE_CHECKPOINT_DOWNLOAD",
        "shared_cache_analysis": {
            "current_behavior": (
                "Each supervisor child uses /var/lib/swarm/e025/<worker_id>/cache. "
                "Concurrent processes never share partial/object/package paths."
            ),
            "process_safety": "SAFE_BY_WORKER_PATH_ISOLATION",
            "overlapping_source_shards": (
                "Downloaded once per worker under the unchanged image; duplication is "
                "charged in every per-Pod peak and download total."
            ),
            "worker_image_change_required": False,
        },
        "internet_throughput_note": (
            "RunPod REST inventory exposes minDownloadMbps filters but not a measured "
            "current internet rate. All three durations are scenarios until P1/P3."
        ),
        "pods": rows,
        "totals": {
            "source_weight_ownership_bytes": sum(
                int(row["source_weight_ownership_bytes"]) for row in rows
            ),
            "download_bytes_current_worker_scoped_cache": sum(
                int(row["download_bytes_current_worker_scoped_cache"]) for row in rows
            ),
            "final_package_bytes": sum(int(row["final_package_bytes"]) for row in rows),
            "requested_container_disk_gb": sum(
                int(row["requested_container_disk_gb"]) for row in rows
            ),
        },
    }


def storage_plan(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "logical_pod_id": manifest["logical_pod_id"],
            "pod_class": manifest["pod_class"],
            **manifest["storage"],
        }
        for manifest in manifests
    ]
    return {
        "schema_version": "experiment-025-runpod-pod-storage-v1",
        "generated_at_utc": utc_now(),
        "evidence_class": "PROJECTION_FROM_PHYSICAL_ASSIGNMENT_ARTIFACTS",
        "status": "PASS_PLANNED",
        "container_disk_rate_usd_per_gb_month": CONTAINER_STORAGE_USD_PER_GB_MONTH,
        "network_volumes_created": 0,
        "volume_disk_requested_gb": 0,
        "pods": rows,
        "total_requested_container_disk_gb": sum(
            int(row["requested_container_disk_gb"]) for row in rows
        ),
    }


def hourly_storage_cost(total_disk_gb: int) -> float:
    return total_disk_gb * CONTAINER_STORAGE_USD_PER_GB_MONTH / (30 * 24)


def paid_canary_budget(
    *,
    backbone_price: float,
    fragment_low_price: float,
    fragment_high_price: float,
    preferred_backbone_gpu_count_per_pod: int,
    fragment_stock_located: bool,
    headline_storage_gb: int,
) -> dict[str, Any]:
    p3_gpu_count = max(2, preferred_backbone_gpu_count_per_pod)
    stages = [
        ("P1_SINGLE_SECURE_3090", 1, 0, 0.75, 100),
        ("P2_TWO_SECURE_POD_NETWORK", 2, 0, 0.50, 80),
        (
            "P3_MULTI_GPU_SECURE_3090",
            p3_gpu_count,
            0,
            1.00,
            180,
        ),
        ("P4_LAYER89_PROVIDER_CANARY", 1, 4, 1.00, 260),
        ("P5_FULL_ACQUISITION_REHEARSAL_NO_K3", 93, 4, 0.25, headline_storage_gb),
    ]
    rows: list[dict[str, Any]] = []
    for name, backbone_gpus, fragment_gpus, hours, storage_gb in stages:
        compute_low = hours * (backbone_gpus * backbone_price + fragment_gpus * fragment_low_price)
        compute_high = hours * (
            backbone_gpus * backbone_price + fragment_gpus * fragment_high_price
        )
        storage = hours * hourly_storage_cost(storage_gb)
        if name == "P3_MULTI_GPU_SECURE_3090":
            current_stock_supported = preferred_backbone_gpu_count_per_pod >= 2
            stock_basis = (
                "current preferred live multi-GPU count"
                if current_stock_supported
                else "minimum viable two-GPU projection; current multi-GPU stock is absent"
            )
        elif name in {
            "P4_LAYER89_PROVIDER_CANARY",
            "P5_FULL_ACQUISITION_REHEARSAL_NO_K3",
        }:
            current_stock_supported = fragment_stock_located
            stock_basis = (
                "located validated Ampere fragment stock"
                if current_stock_supported
                else "price-bounded projection; located validated Ampere fragment stock is absent"
            )
        else:
            current_stock_supported = True
            stock_basis = "current Secure RTX 3090 single-GPU price"
        rows.append(
            {
                "stage": name,
                "backbone_gpus": backbone_gpus,
                "fragment_gpus": fragment_gpus,
                "expected_duration_hours": hours,
                "compute_cost_usd": {"low": compute_low, "high": compute_high},
                "container_storage_cost_usd": storage,
                "data_transfer_cost_usd": 0,
                "current_stock_supported": current_stock_supported,
                "stock_basis": stock_basis,
                "base_cost_usd": {
                    "low": compute_low + storage,
                    "high": compute_high + storage,
                },
            }
        )
    base_low = sum(float(row["base_cost_usd"]["low"]) for row in rows)
    base_high = sum(float(row["base_cost_usd"]["high"]) for row in rows)
    safety_multiplier = 1.5
    return {
        "schema_version": "experiment-025-runpod-paid-canary-budget-v1",
        "generated_at_utc": utc_now(),
        "evidence_class": "PROJECTION_USING_LIVE_INVENTORY_PRICES",
        "status": "ESTIMATE_NOT_AUTHORIZATION_NOT_INVOICE",
        "data_transfer_assumption": (
            "RunPod Pods pricing documentation says no ingress or egress fees; "
            "recheck immediately before paid execution."
        ),
        "unsupported_stage_costs_are_projections": [
            row["stage"] for row in rows if not row["current_stock_supported"]
        ],
        "stages": rows,
        "base_total_usd": {"low": base_low, "high": base_high},
        "safety_multiplier": safety_multiplier,
        "budget_with_safety_usd": {
            "low": base_low * safety_multiplier,
            "high": base_high * safety_multiplier,
        },
        "minimum_recommended_account_credit_usd": math.ceil(
            max(60.0, base_high * safety_multiplier)
        ),
    }


def headline_budget(
    *,
    backbone_price: float,
    fragment_low_price: float,
    fragment_high_price: float,
    storage_gb: int,
    account_balance: float | None,
    account_hourly_limit: float | None,
) -> dict[str, Any]:
    compute_low = 93 * backbone_price + 4 * fragment_low_price
    compute_high = 93 * backbone_price + 4 * fragment_high_price
    storage_hourly = hourly_storage_cost(storage_gb)
    hourly_low = compute_low + storage_hourly
    hourly_high = compute_high + storage_hourly
    windows = {
        "model_acquisition_hours_provisional": 1.5,
        "inference_hours": 0.5,
        "cleanup_reserve_hours": 0.25,
    }
    hard_window = sum(windows.values())
    safety_multiplier = 1.25
    recommended = math.ceil(max(hourly_high, hourly_high * hard_window * safety_multiplier))
    return {
        "schema_version": "experiment-025-runpod-headline-budget-v1",
        "generated_at_utc": utc_now(),
        "evidence_class": "PROJECTION_USING_LIVE_INVENTORY_PRICES",
        "status": "ESTIMATE_NOT_AUTHORIZATION_NOT_INVOICE",
        "backbone_gpu_count": 93,
        "fragment_gpu_count": 4,
        "compute_hourly_usd": {"low": compute_low, "high": compute_high},
        "container_storage_requested_gb": storage_gb,
        "container_storage_hourly_usd": storage_hourly,
        "headline_hourly_usd": {"low": hourly_low, "high": hourly_high},
        "windows": windows,
        "hard_window_hours": hard_window,
        "hard_window_cost_usd": {
            "low": hourly_low * hard_window,
            "high": hourly_high * hard_window,
        },
        "safety_multiplier": safety_multiplier,
        "hard_budget_with_safety_usd": {
            "low": hourly_low * hard_window * safety_multiplier,
            "high": hourly_high * hard_window * safety_multiplier,
        },
        "minimum_recommended_account_credit_usd": recommended,
        "account": {
            "current_credit_usd": account_balance,
            "hourly_limit_usd": account_hourly_limit,
            "hourly_limit_fits": account_hourly_limit is not None
            and hourly_high <= account_hourly_limit,
            "credit_fits": account_balance is not None and account_balance >= recommended,
        },
        "data_transfer_cost_usd": 0,
        "data_transfer_assumption": (
            "Current RunPod Pods documentation states no ingress/egress fees."
        ),
    }


def assert_manifest_safety(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    worker_ids = [str(role["worker_id"]) for manifest in manifests for role in manifest["roles"]]
    failures: list[str] = []
    if len(worker_ids) != 97 or len(set(worker_ids)) != 97:
        failures.append("manifest does not cover 97 unique roles")
    fragment_manifests = [row for row in manifests if row["pod_class"] == "FRAGMENT"]
    if len(fragment_manifests) != 4 or any(
        int(row["requested_gpu_count"]) != 1 for row in fragment_manifests
    ):
        failures.append("Layer 89 fragments are not four single-GPU Pods")
    for manifest in manifests:
        specs = decode_worker_specs(str(manifest["E025_WORKER_SPECS_B64"]))
        slots = [int(row["gpu_slot"]) for row in specs]
        ports = [int(row["port"]) for row in specs]
        if slots != list(range(len(specs))):
            failures.append(f"non-contiguous GPU slots: {manifest['logical_pod_id']}")
        if ports != list(range(BASE_PORT, BASE_PORT + len(specs))):
            failures.append(f"non-contiguous ports: {manifest['logical_pod_id']}")
        if len(specs) > MAX_WORKERS_PER_INSTANCE:
            failures.append(f"supervisor limit exceeded: {manifest['logical_pod_id']}")
        if int(manifest["environment"]["environment_variable_count"]) > 50:
            failures.append(f"RunPod environment limit exceeded: {manifest['logical_pod_id']}")
        payload = manifest["create_payload_redacted"]
        encoded_payload = json.dumps(payload, sort_keys=True)
        if any(name in encoded_payload for name in SECRET_ENVIRONMENT_KEYS):
            # Secret variable names are expected, but their values must be descriptors.
            env = payload["env"]
            for name in SECRET_ENVIRONMENT_KEYS & set(env):
                if not isinstance(env[name], dict):
                    failures.append(f"secret value persisted for {name}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "logical_role_count": len(worker_ids),
        "unique_logical_role_count": len(set(worker_ids)),
        "fragment_pod_count": len(fragment_manifests),
    }


__all__ = [
    "BACKBONE_GPU_ID",
    "FRAGMENT_GPU_IDS",
    "GLOBAL_NETWORK_MEGABITS_PER_SECOND",
    "IMAGE_DIGEST",
    "IMAGE_REFERENCE",
    "IMAGE_TAG",
    "RUN_ID",
    "RUN_ROOT_NAME",
    "FrozenRole",
    "assert_manifest_safety",
    "build_role_manifests",
    "decode_worker_specs",
    "encode_worker_specs",
    "headline_budget",
    "load_frozen_roles",
    "model_distribution_plan",
    "pack_roles",
    "paid_canary_budget",
    "storage_plan",
    "topology_metrics",
]
