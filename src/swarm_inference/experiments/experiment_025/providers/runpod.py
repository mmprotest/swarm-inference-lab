"""RunPod REST/GraphQL provider adapter for E025.

No network request is made at import time. Every provider method first passes
through the semantic mutation policy; this keeps preparation and simulator runs
incapable of allocating resources even if a caller accidentally reaches a paid
code path. Sensitive request fields are only exposed through redacted metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any, Protocol, cast

from .base import (
    EndpointMode,
    EndpointRecord,
    PhysicalAllocationRecord,
    ProviderMutationPolicy,
    ProviderOperation,
)

REST_BASE_URL = "https://rest.runpod.io/v1"
GRAPHQL_URL = "https://api.runpod.io/graphql"
SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "registry_auth",
    "secret",
    "tls_key",
    "token",
)
SECRET_ENVIRONMENT_KEYS = frozenset(
    {
        "E025_RUN_CREDENTIAL_B64",
        "E025_TLS_CERT_B64",
        "E025_TLS_KEY_B64",
        "HF_TOKEN",
        "RUNPOD_API_KEY",
    }
)


class RunPodTransport(Protocol):
    def request(
        self,
        *,
        operation: ProviderOperation,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any: ...


def _is_secret_key(key: object) -> bool:
    lowered = str(key).lower()
    return str(key).upper() in SECRET_ENVIRONMENT_KEYS or any(
        marker in lowered for marker in SECRET_KEY_MARKERS
    )


def secret_descriptor(value: object) -> dict[str, Any]:
    encoded = str(value).encode("utf-8")
    return {
        "redacted": True,
        "present": bool(encoded),
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def redact_provider_payload(value: Any, *, parent_key: str = "") -> Any:
    """Return a persistable provider payload with secrets replaced by descriptors."""

    if parent_key and _is_secret_key(parent_key):
        return secret_descriptor(value)
    if isinstance(value, Mapping):
        return {
            str(key): redact_provider_payload(item, parent_key=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_provider_payload(item, parent_key=parent_key) for item in value]
    if isinstance(value, tuple):
        return [redact_provider_payload(item, parent_key=parent_key) for item in value]
    return value


def graphql_operation(document: str) -> ProviderOperation:
    """Classify a GraphQL document without relying on the HTTP POST verb."""

    without_comments = re.sub(r"(?m)^\s*#.*$", "", document.lstrip("\ufeff"))
    operation = re.search(r"(?im)^\s*(query|mutation|subscription)\b", without_comments)
    if operation and operation.group(1).lower() == "mutation":
        return ProviderOperation.GRAPHQL_MUTATION
    if operation and operation.group(1).lower() == "subscription":
        # Subscriptions are not part of E025 preparation and should not silently
        # pass as an inventory query.
        return ProviderOperation.GRAPHQL_MUTATION
    if not operation and not without_comments.lstrip().startswith("{"):
        raise ValueError("GraphQL document has no recognized operation")
    return ProviderOperation.GRAPHQL_QUERY


class UrllibRunPodTransport:
    """Small standard-library REST/GraphQL transport with sanitized auditing."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        rest_base_url: str = REST_BASE_URL,
        graphql_url: str = GRAPHQL_URL,
        timeout_seconds: float = 30.0,
        audit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.api_key = api_key
        self.rest_base_url = rest_base_url.rstrip("/")
        self.graphql_url = graphql_url
        self.timeout_seconds = timeout_seconds
        self.audit = audit

    def request(
        self,
        *,
        operation: ProviderOperation,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        url = (
            self.graphql_url
            if operation
            in {
                ProviderOperation.GRAPHQL_QUERY,
                ProviderOperation.GRAPHQL_MUTATION,
            }
            else f"{self.rest_base_url}{path}"
        )
        headers = {
            "Accept": "application/json",
            "User-Agent": "swarm-inference-lab-e025/1.0 (read-only-inventory)",
        }
        body: bytes | None = None
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        audit_row = {
            "operation": operation.value,
            "method": method,
            "url": url,
            "authenticated": bool(self.api_key),
            "payload": redact_provider_payload(payload) if payload is not None else None,
        }
        if self.audit is not None:
            self.audit(audit_row)
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                encoded = response.read()
                if not encoded:
                    return None
                return json.loads(encoded)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(
                f"RunPod {operation.value} failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"RunPod {operation.value} request failed: {exc}") from exc


class RunPodProvider:
    def __init__(
        self,
        *,
        transport: RunPodTransport,
        policy: ProviderMutationPolicy | None = None,
    ) -> None:
        self.transport = transport
        self.policy = policy or ProviderMutationPolicy()

    def _request(
        self,
        operation: ProviderOperation,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        self.policy.require(operation)
        return self.transport.request(
            operation=operation,
            method=method,
            path=path,
            payload=payload,
        )

    def list_pods(self) -> Any:
        return self._request(ProviderOperation.LIST_PODS, "GET", "/pods")

    def get_pod(self, pod_id: str) -> Any:
        return self._request(ProviderOperation.GET_POD, "GET", f"/pods/{pod_id}")

    def create_pod(self, payload: dict[str, Any]) -> Any:
        return self._request(ProviderOperation.CREATE_POD, "POST", "/pods", payload)

    def start_pod(self, pod_id: str) -> Any:
        return self._request(ProviderOperation.START_POD, "POST", f"/pods/{pod_id}/start")

    def restart_pod(self, pod_id: str) -> Any:
        return self._request(ProviderOperation.RESTART_POD, "POST", f"/pods/{pod_id}/restart")

    def reset_pod(self, pod_id: str) -> Any:
        return self._request(ProviderOperation.RESET_POD, "POST", f"/pods/{pod_id}/reset")

    def update_pod(self, pod_id: str, payload: dict[str, Any]) -> Any:
        return self._request(ProviderOperation.UPDATE_POD, "PATCH", f"/pods/{pod_id}", payload)

    def stop_pod(self, pod_id: str) -> Any:
        return self._request(ProviderOperation.STOP_POD, "POST", f"/pods/{pod_id}/stop")

    def delete_pod(self, pod_id: str) -> Any:
        return self._request(ProviderOperation.DELETE_POD, "DELETE", f"/pods/{pod_id}")

    def create_network_volume(self, payload: dict[str, Any]) -> Any:
        return self._request(
            ProviderOperation.CREATE_NETWORK_VOLUME,
            "POST",
            "/networkvolumes",
            payload,
        )

    def list_network_volumes(self) -> Any:
        return self._request(ProviderOperation.LIST_VOLUMES, "GET", "/networkvolumes")

    def delete_network_volume(self, volume_id: str) -> Any:
        return self._request(
            ProviderOperation.DELETE_NETWORK_VOLUME,
            "DELETE",
            f"/networkvolumes/{volume_id}",
        )

    def graphql_query(self, payload: dict[str, Any]) -> Any:
        operation = graphql_operation(str(payload.get("query", "")))
        return self._request(operation, "POST", "/graphql", payload)


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def normalize_port_mappings(value: Any) -> dict[int, int]:
    result: dict[int, int] = {}
    if not isinstance(value, Mapping):
        return result
    for raw_internal, raw_external in value.items():
        match = re.match(r"^(\d+)", str(raw_internal))
        if not match:
            continue
        internal = int(match.group(1))
        candidates = raw_external if isinstance(raw_external, list) else [raw_external]
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                external = _first(candidate, "HostPort", "hostPort", "host_port")
            else:
                external = candidate
            try:
                parsed = int(cast(Any, external))
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                result[internal] = parsed
                break
    return result


def normalize_pod(value: Mapping[str, Any]) -> PhysicalAllocationRecord:
    machine = value.get("machine") if isinstance(value.get("machine"), Mapping) else {}
    gpu = value.get("gpu") if isinstance(value.get("gpu"), Mapping) else {}
    datacenter = value.get("dataCenter") if isinstance(value.get("dataCenter"), Mapping) else {}
    gpu_count = _first(cast(Mapping[str, Any], gpu), "count", "gpuCount")
    if gpu_count is None:
        gpu_count = _first(value, "gpuCount", "gpu_count")
    gpu_memory = _first(cast(Mapping[str, Any], gpu), "memoryInGb", "memory_in_gb")
    price = _first(value, "costPerHr", "costPerHour", "cost_per_hour")
    if price is None:
        cloud_type = str(_first(value, "cloudType", "cloud_type") or "").upper()
        price_field = "securePrice" if cloud_type == "SECURE" else "communityPrice"
        per_gpu = _first(cast(Mapping[str, Any], gpu), price_field)
        if per_gpu is not None and gpu_count is not None:
            price = float(per_gpu) * int(gpu_count)
    raw_machine_id = _first(value, "machineId", "machine_id")
    if raw_machine_id is None:
        raw_machine_id = _first(cast(Mapping[str, Any], machine), "id", "machineId")
    datacenter_id = _first(value, "dataCenterId", "data_center_id")
    if datacenter_id is None:
        datacenter_id = _first(cast(Mapping[str, Any], datacenter), "id")
    public_ip = _first(value, "publicIp", "public_ip", "publicIpAddress")
    pod_id = str(_first(value, "id", "podId", "pod_id") or "")
    record = PhysicalAllocationRecord(
        provider="runpod",
        pod_id=pod_id,
        machine_id=str(raw_machine_id) if raw_machine_id is not None else None,
        pod_name=str(_first(value, "name", "podName") or ""),
        desired_status=(
            str(_first(value, "desiredStatus", "desired_status"))
            if _first(value, "desiredStatus", "desired_status") is not None
            else None
        ),
        current_status=(
            str(_first(value, "runtimeStatus", "status", "currentStatus"))
            if _first(value, "runtimeStatus", "status", "currentStatus") is not None
            else None
        ),
        gpu_model=(str(_first(cast(Mapping[str, Any], gpu), "id", "displayName")) if gpu else None),
        gpu_count=int(gpu_count) if gpu_count is not None else None,
        gpu_vram_bytes=(int(float(gpu_memory) * 1024**3) if gpu_memory is not None else None),
        cost_per_hour=float(price) if price is not None else None,
        datacenter_id=str(datacenter_id) if datacenter_id is not None else None,
        location={
            key: datacenter[key]
            for key in ("name", "location", "country", "region", "city")
            if key in datacenter
        },
        public_ip=str(public_ip) if public_ip else None,
        port_mappings=normalize_port_mappings(
            _first(value, "portMappings", "port_mappings", "ports")
        ),
        private_identity=f"{pod_id}.runpod.internal" if pod_id else None,
        image_identity=(
            str(_first(value, "image", "imageName"))
            if _first(value, "image", "imageName") is not None
            else None
        ),
        creation_timestamp=(
            str(_first(value, "createdAt", "created_at"))
            if _first(value, "createdAt", "created_at") is not None
            else None
        ),
        termination_timestamp=(
            str(_first(value, "terminatedAt", "terminated_at"))
            if _first(value, "terminatedAt", "terminated_at") is not None
            else None
        ),
        raw_metadata=cast(dict[str, Any], redact_provider_payload(dict(value))),
    )
    return record


def allocation_dict(record: PhysicalAllocationRecord) -> dict[str, Any]:
    return asdict(record)


def resolve_endpoint(
    *,
    allocation: PhysicalAllocationRecord,
    worker_id: str,
    internal_port: int,
    mode: EndpointMode,
    fragment_endpoint_generation: int | None = None,
) -> EndpointRecord:
    if mode is EndpointMode.POD_LOCAL:
        host, port = "127.0.0.1", internal_port
    elif mode is EndpointMode.RUNPOD_GLOBAL_PRIVATE:
        if not allocation.pod_id:
            raise ValueError("RunPod private endpoint requires a Pod ID")
        host, port = f"{allocation.pod_id}.runpod.internal", internal_port
    elif mode is EndpointMode.RUNPOD_PUBLIC_TCP:
        if not allocation.public_ip:
            raise ValueError("RunPod public endpoint requires publicIp")
        if internal_port not in allocation.port_mappings:
            raise ValueError(
                f"RunPod public endpoint has no mapping for internal port {internal_port}"
            )
        host, port = allocation.public_ip, allocation.port_mappings[internal_port]
    else:  # pragma: no cover - exhaustive StrEnum guard
        raise ValueError(f"unsupported endpoint mode: {mode}")
    return EndpointRecord(
        mode=mode,
        pod_id=allocation.pod_id,
        worker_id=worker_id,
        internal_port=internal_port,
        host=host,
        port=port,
        fragment_endpoint_generation=fragment_endpoint_generation,
    )


__all__ = [
    "GRAPHQL_URL",
    "REST_BASE_URL",
    "SECRET_ENVIRONMENT_KEYS",
    "RunPodProvider",
    "RunPodTransport",
    "UrllibRunPodTransport",
    "allocation_dict",
    "graphql_operation",
    "normalize_pod",
    "normalize_port_mappings",
    "redact_provider_payload",
    "resolve_endpoint",
    "secret_descriptor",
]
