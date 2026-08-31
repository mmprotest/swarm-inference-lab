"""Read-only RunPod CLI and GraphQL inventory capture for E025."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .io import atomic_write_json, utc_now
from .providers.base import ProviderMutationPolicy
from .providers.runpod import RunPodProvider, UrllibRunPodTransport
from .runpod_planning import BACKBONE_GPU_ID, FRAGMENT_GPU_IDS

INTERESTING_GPU_IDS = (BACKBONE_GPU_ID, "NVIDIA GeForce RTX 3090 Ti", *FRAGMENT_GPU_IDS)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_runpodctl(repo: Path, explicit: Path | None = None) -> Path:
    candidates = [explicit] if explicit is not None else []
    resolved_from_path = shutil.which("runpodctl") or shutil.which("runpodctl.exe")
    if resolved_from_path:
        candidates.append(Path(resolved_from_path))
    candidates.append(repo / "runpodctl.exe")
    for candidate in candidates:
        if candidate is not None and candidate.expanduser().is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError(
        "runpodctl is absent; install the official Windows AMD64 binary and configure "
        "it with `runpodctl doctor` without placing the API key in this repository"
    )


class ReadOnlyRunPodCli:
    """Narrow CLI facade that cannot invoke a provider mutation command."""

    ALLOWED_PREFIXES = (
        ("version",),
        ("--help",),
        ("gpu", "list"),
        ("datacenter", "list"),
        ("pod", "list"),
        ("pod", "get"),
        ("user",),
        ("network-volume", "list"),
    )

    def __init__(self, executable: Path) -> None:
        self.executable = executable.resolve()
        self.invocations: list[dict[str, Any]] = []

    def run(self, *arguments: str, timeout_seconds: float = 60.0) -> str:
        if not any(tuple(arguments[: len(prefix)]) == prefix for prefix in self.ALLOWED_PREFIXES):
            raise PermissionError(
                f"read-only RunPod CLI facade rejected semantic operation: {arguments}"
            )
        process = subprocess.run(
            [str(self.executable), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        self.invocations.append(
            {
                "arguments": list(arguments),
                "returncode": process.returncode,
                "stderr": process.stderr[-2000:],
            }
        )
        if process.returncode != 0:
            raise RuntimeError(f"runpodctl {' '.join(arguments)} failed: {process.stderr[-1000:]}")
        return process.stdout

    def json(self, *arguments: str) -> Any:
        value = json.loads(self.run(*arguments))
        return value


def build_gpu_matrix_query() -> str:
    blocks: list[str] = []
    for gpu_index, gpu_id in enumerate(INTERESTING_GPU_IDS):
        price_fields: list[str] = []
        for count in range(1, 9):
            price_fields.append(
                f"s{count}: lowestPrice(input: {{ gpuCount: {count}, secureCloud: true }}) "
                "{ stockStatus uninterruptablePrice availableGpuCounts }"
            )
            price_fields.append(
                f"c{count}: lowestPrice(input: {{ gpuCount: {count}, secureCloud: false }}) "
                "{ stockStatus uninterruptablePrice availableGpuCounts }"
            )
        blocks.append(
            f'g{gpu_index}: gpuTypes(input: {{ id: "{gpu_id}" }}) '
            f"{{ id displayName memoryInGb {' '.join(price_fields)} }}"
        )
    return "query E025ReadOnlyGpuMatrix { " + " ".join(blocks) + " }"


def normalize_graphql_matrix(response: dict[str, Any]) -> list[dict[str, Any]]:
    if response.get("errors"):
        raise RuntimeError(f"RunPod GraphQL inventory returned errors: {response['errors']}")
    data = response.get("data")
    if not isinstance(data, dict):
        raise ValueError("RunPod GraphQL inventory has no data object")
    rows: list[dict[str, Any]] = []
    for alias in sorted(data, key=lambda value: int(value[1:])):
        values = data[alias]
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"RunPod GraphQL GPU alias {alias} is not a singleton")
        value = values[0]
        tiers: dict[str, Any] = {}
        for prefix, tier in (("s", "SECURE"), ("c", "COMMUNITY")):
            counts: list[dict[str, Any]] = []
            schedulable: list[int] = []
            for count in range(1, 9):
                price = value.get(f"{prefix}{count}")
                normalized = {
                    "gpu_count": count,
                    "stockStatus": price.get("stockStatus") if isinstance(price, dict) else None,
                    "uninterruptablePrice": (
                        price.get("uninterruptablePrice") if isinstance(price, dict) else None
                    ),
                    "availableGpuCounts": (
                        price.get("availableGpuCounts") if isinstance(price, dict) else None
                    ),
                }
                if (
                    normalized["stockStatus"] not in {None, "None", "none", ""}
                    and normalized["uninterruptablePrice"] is not None
                ):
                    schedulable.append(count)
                counts.append(normalized)
            tiers[tier] = {
                "count_queries": counts,
                "schedulable_gpu_counts_inferred_from_non_null_lowest_price": schedulable,
                "availableGpuCounts_field": next(
                    (
                        row["availableGpuCounts"]
                        for row in counts
                        if row["availableGpuCounts"] is not None
                    ),
                    None,
                ),
            }
        rows.append(
            {
                "gpu_type_id": str(value["id"]),
                "display_name": str(value.get("displayName", value["id"])),
                "memory_in_gb": value.get("memoryInGb"),
                "tiers": tiers,
            }
        )
    return rows


def _hash_identity(value: Any) -> dict[str, Any]:
    encoded = str(value).encode("utf-8")
    return {
        "present": bool(encoded),
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def sanitize_account(account: dict[str, Any]) -> dict[str, Any]:
    return {
        "client_balance_usd": account.get("clientBalance"),
        "current_spend_per_hour_usd": account.get("currentSpendPerHr"),
        "spend_limit_per_hour_usd": account.get("spendLimit"),
        "account_id": _hash_identity(account.get("id", "")),
        "email": _hash_identity(account.get("email", "")),
        "notifications": {
            key: value for key, value in account.items() if str(key).startswith("notify")
        },
    }


def sanitize_existing_resources(rows: Any, *, resource_type: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"RunPod {resource_type} list is not an array")
    sanitized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        resource_id = row.get("id", row.get("podId", ""))
        sanitized.append(
            {
                "id": _hash_identity(resource_id),
                "name": _hash_identity(row.get("name", "")),
                "desired_status": row.get("desiredStatus"),
                "runtime_status": row.get("runtimeStatus"),
                "data_center_id": row.get("dataCenterId"),
                "gpu_count": row.get(
                    "gpuCount",
                    row.get("gpu", {}).get("count") if isinstance(row.get("gpu"), dict) else None,
                ),
                "created_at": row.get("createdAt"),
            }
        )
    return sanitized


def _normalize_cli_gpu(rows: Any, graphql_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError("runpodctl GPU inventory is not an array")
    matrix = {row["gpu_type_id"]: row for row in graphql_rows}
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("gpuId") not in INTERESTING_GPU_IDS:
            continue
        gpu_id = str(row["gpuId"])
        normalized.append(
            {
                "gpu_type_id": gpu_id,
                "display_name": row.get("displayName"),
                "memory_in_gb": row.get("memoryInGb"),
                "cli_available": row.get("available"),
                "secure_cloud_exposed": row.get("secureCloud"),
                "community_cloud_exposed": row.get("communityCloud"),
                "secure_price_per_gpu_hour": row.get("securePricePerHr"),
                "community_price_per_gpu_hour": row.get("communityPricePerHr"),
                "cli_stock_status": row.get("stockStatus"),
                "datacenter_availability": row.get("dataCenterAvailability", []),
                "graphql": matrix.get(gpu_id),
                "cross_check": {
                    "minor_presentation_differences_allowed": True,
                    "cli_graphql_availability_conflict": (
                        row.get("available") is False
                        and any(
                            matrix.get(gpu_id, {})
                            .get("tiers", {})
                            .get(tier, {})
                            .get("schedulable_gpu_counts_inferred_from_non_null_lowest_price", [])
                            for tier in ("SECURE", "COMMUNITY")
                        )
                    ),
                },
            }
        )
    return sorted(normalized, key=lambda row: str(row["gpu_type_id"]))


def collect_read_only_inventory(
    *,
    repo: Path,
    output_directory: Path,
    runpodctl_path: Path | None = None,
) -> dict[str, Any]:
    """Capture authenticated CLI plus public GraphQL inventory without mutations."""

    output_directory.mkdir(parents=True, exist_ok=True)
    binary = find_runpodctl(repo, runpodctl_path)
    cli = ReadOnlyRunPodCli(binary)
    captured_at = utc_now()
    version_output = cli.run("version").strip()
    help_output = cli.run("--help")
    gpu_rows = cli.json("gpu", "list", "--include-unavailable", "--output", "json")
    datacenters = cli.json("datacenter", "list", "--output", "json")
    pods = cli.json("pod", "list", "--all", "--output", "json")
    volumes = cli.json("network-volume", "list", "--output", "json")
    account_raw = cli.json("user", "--output", "json")

    graphql_audit: list[dict[str, Any]] = []
    provider = RunPodProvider(
        transport=UrllibRunPodTransport(audit=graphql_audit.append),
        policy=ProviderMutationPolicy(),
    )
    graphql_response = provider.graphql_query(
        {"query": build_gpu_matrix_query(), "operationName": "E025ReadOnlyGpuMatrix"}
    )
    graphql_rows = normalize_graphql_matrix(graphql_response)
    normalized_gpu = _normalize_cli_gpu(gpu_rows, graphql_rows)
    account = sanitize_account(account_raw)
    existing_pods = sanitize_existing_resources(pods, resource_type="Pod")
    existing_volumes = sanitize_existing_resources(volumes, resource_type="network volume")
    account_readiness = {
        "schema_version": "experiment-025-runpod-account-readiness-v1",
        "captured_at_utc": captured_at,
        "status": "READ_ONLY_CAPTURE_COMPLETE",
        "authentication": "RUNPODCTL_EXISTING_CONFIGURATION",
        "account": account,
        "existing_pods": existing_pods,
        "existing_pod_count": len(existing_pods),
        "existing_network_volumes": existing_volumes,
        "existing_network_volume_count": len(existing_volumes),
        "manual_checks_remaining": [
            "Add sufficient account credit before P1; current balance is zero."
            if float(account.get("client_balance_usd") or 0) <= 0
            else "Reconfirm account balance immediately before P1.",
            "Reconfirm spend limit and current spend immediately before each paid stage.",
        ],
    }
    version_receipt = {
        "schema_version": "experiment-025-runpodctl-version-v1",
        "captured_at_utc": captured_at,
        "status": "PASS",
        "version_output": version_output,
        "binary_path": str(binary),
        "binary_bytes": binary.stat().st_size,
        "binary_sha256": _sha256_file(binary),
        "repository_local_existing_binary": binary.parent == repo.resolve(),
        "reinstalled": False,
        "help_sha256": hashlib.sha256(help_output.encode("utf-8")).hexdigest(),
        "help_output": help_output,
    }
    gpu_receipt = {
        "schema_version": "experiment-025-runpod-live-gpu-inventory-v1",
        "captured_at_utc": captured_at,
        "status": "READ_ONLY_CAPTURE_COMPLETE",
        "cli_command": "runpodctl gpu list --include-unavailable --output json",
        "graphql_operation": "query E025ReadOnlyGpuMatrix",
        "graphql_authenticated": False,
        "graphql_public_read_only_succeeded": True,
        "availableGpuCounts_note": (
            "The live GraphQL field returned null for relevant SKUs. Schedulable counts "
            "are conservatively inferred only where a count-specific lowestPrice query "
            "returned both stockStatus and uninterruptablePrice."
        ),
        "gpu_types": normalized_gpu,
        "raw_sanitized": {
            "cli_relevant_rows": [
                row
                for row in gpu_rows
                if isinstance(row, dict) and row.get("gpuId") in INTERESTING_GPU_IDS
            ],
            "graphql_relevant_rows": graphql_rows,
        },
    }
    datacenter_receipt = {
        "schema_version": "experiment-025-runpod-datacenter-inventory-v1",
        "captured_at_utc": captured_at,
        "status": "READ_ONLY_CAPTURE_COMPLETE",
        "cli_command": "runpodctl datacenter list --output json",
        "datacenters": datacenters,
    }
    provider_schema = {
        "schema_version": "experiment-025-runpod-provider-schema-v1",
        "captured_at_utc": captured_at,
        "status": "PASS",
        "provider_mode": ProviderMutationPolicy().receipt(),
        "surfaces": {
            "operator_cli": {"version": version_output, "read_only_facade": True},
            "rest_v1": {
                "base_url": "https://rest.runpod.io/v1",
                "lifecycle": ["POST /pods", "GET /pods", "GET /pods/{id}", "DELETE /pods/{id}"],
                "create_not_called": True,
                "terminate_after_field_observed": False,
            },
            "graphql": {
                "url": "https://api.runpod.io/graphql",
                "queries_only": True,
                "mutations_called": False,
                "query_fields": [
                    "gpuTypes",
                    "lowestPrice",
                    "stockStatus",
                    "uninterruptablePrice",
                    "availableGpuCounts",
                ],
            },
        },
        "official_documentation": {
            "rest_create": "https://docs.runpod.io/api-reference/pods/POST/pods",
            "rest_delete": "https://docs.runpod.io/api-reference/pods/DELETE/pods/podId",
            "graphql_gpu_inventory": "https://docs.runpod.io/sdks/graphql/manage-pods",
            "global_networking": "https://docs.runpod.io/pods/networking",
            "pricing": "https://docs.runpod.io/pods/pricing",
            "environment": "https://docs.runpod.io/pods/templates/environment-variables",
        },
        "graphql_audit": graphql_audit,
        "cli_invocations": cli.invocations,
        "provider_compute_mutations": [],
    }
    environment = {
        "schema_version": "experiment-025-runpod-preparation-environment-v1",
        "captured_at_utc": captured_at,
        "status": "PASS",
        "provider_mode": "READ_ONLY_PREPARATION",
        "python": os.sys.version,
        "platform": os.name,
        "repository": str(repo.resolve()),
        "artifact_directory": str(output_directory.resolve()),
        "paid_resources_created": 0,
    }
    artifacts = {
        "environment.json": environment,
        "runpodctl-version.json": version_receipt,
        "runpod-live-gpu-inventory.json": gpu_receipt,
        "runpod-datacenter-inventory.json": datacenter_receipt,
        "runpod-account-readiness.json": account_readiness,
        "runpod-provider-schema.json": provider_schema,
    }
    baseline_path = output_directory / "runpod-before-state.json"
    if not baseline_path.exists():
        atomic_write_json(
            baseline_path,
            {
                "schema_version": "experiment-025-runpod-before-state-v1",
                "captured_at_utc": captured_at,
                "provider_mode": "READ_ONLY_PREPARATION",
                "pod_count": len(existing_pods),
                "network_volume_count": len(existing_volumes),
                "pods": existing_pods,
                "network_volumes": existing_volumes,
                "e025_runpod_pod_count": 0,
                "provider_compute_mutations": [],
            },
        )
    for name, value in artifacts.items():
        atomic_write_json(output_directory / name, value)
    return {
        "captured_at_utc": captured_at,
        "artifacts": sorted(artifacts),
        "gpu_inventory": gpu_receipt,
        "datacenters": datacenter_receipt,
        "account_readiness": account_readiness,
        "provider_schema": provider_schema,
    }


__all__ = [
    "INTERESTING_GPU_IDS",
    "ReadOnlyRunPodCli",
    "build_gpu_matrix_query",
    "collect_read_only_inventory",
    "find_runpodctl",
    "normalize_graphql_matrix",
    "sanitize_account",
]
