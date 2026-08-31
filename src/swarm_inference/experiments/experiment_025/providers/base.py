"""Provider-neutral records and the E025 paid-resource mutation firewall."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ProviderMode(StrEnum):
    """E025 provider safety mode.

    READ_ONLY_PREPARATION is deliberately the default.  PAID_RUN is only
    constructible through :meth:`ProviderMutationPolicy.from_intent` when both
    operator acknowledgements are present.
    """

    READ_ONLY_PREPARATION = "READ_ONLY_PREPARATION"
    PAID_RUN = "PAID_RUN"


class ProviderMutationForbidden(RuntimeError):
    """Raised before a provider operation could allocate or alter resources."""


class ProviderOperation(StrEnum):
    LIST_CAPACITY = "LIST_CAPACITY"
    LIST_DATACENTERS = "LIST_DATACENTERS"
    LIST_PODS = "LIST_PODS"
    GET_POD = "GET_POD"
    GET_ACCOUNT = "GET_ACCOUNT"
    LIST_VOLUMES = "LIST_VOLUMES"
    GRAPHQL_QUERY = "GRAPHQL_QUERY"
    CREATE_POD = "CREATE_POD"
    START_POD = "START_POD"
    RESTART_POD = "RESTART_POD"
    RESET_POD = "RESET_POD"
    UPDATE_POD = "UPDATE_POD"
    STOP_POD = "STOP_POD"
    DELETE_POD = "DELETE_POD"
    CREATE_NETWORK_VOLUME = "CREATE_NETWORK_VOLUME"
    DELETE_NETWORK_VOLUME = "DELETE_NETWORK_VOLUME"
    GRAPHQL_MUTATION = "GRAPHQL_MUTATION"


READ_ONLY_OPERATIONS = frozenset(
    {
        ProviderOperation.LIST_CAPACITY,
        ProviderOperation.LIST_DATACENTERS,
        ProviderOperation.LIST_PODS,
        ProviderOperation.GET_POD,
        ProviderOperation.GET_ACCOUNT,
        ProviderOperation.LIST_VOLUMES,
        ProviderOperation.GRAPHQL_QUERY,
    }
)


@dataclass(frozen=True)
class ProviderMutationPolicy:
    """Semantic provider-operation gate independent of HTTP verb.

    RunPod GraphQL queries normally use HTTP POST, so callers must identify the
    semantic operation rather than treating every POST as a mutation.
    """

    mode: ProviderMode = ProviderMode.READ_ONLY_PREPARATION
    cli_acknowledged: bool = False
    environment_acknowledged: bool = False

    @classmethod
    def from_intent(
        cls,
        *,
        allow_paid_run: bool,
        environment: dict[str, str] | None = None,
    ) -> ProviderMutationPolicy:
        values = os.environ if environment is None else environment
        environment_acknowledged = values.get("E025_RUNPOD_ALLOW_RENTAL") == "YES"
        enabled = allow_paid_run and environment_acknowledged
        return cls(
            mode=(ProviderMode.PAID_RUN if enabled else ProviderMode.READ_ONLY_PREPARATION),
            cli_acknowledged=allow_paid_run,
            environment_acknowledged=environment_acknowledged,
        )

    def require(self, operation: ProviderOperation | str) -> None:
        semantic_operation = ProviderOperation(operation)
        if semantic_operation in READ_ONLY_OPERATIONS:
            return
        if self.mode is not ProviderMode.PAID_RUN:
            raise ProviderMutationForbidden(
                f"RunPod operation {semantic_operation.value} is forbidden in "
                f"{self.mode.value}; both --allow-paid-run and "
                "E025_RUNPOD_ALLOW_RENTAL=YES are required"
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "cli_acknowledged": self.cli_acknowledged,
            "environment_acknowledged": self.environment_acknowledged,
            "paid_mutations_enabled": self.mode is ProviderMode.PAID_RUN,
        }


class CloudTier(StrEnum):
    SECURE = "SECURE"
    COMMUNITY = "COMMUNITY"


class EndpointMode(StrEnum):
    POD_LOCAL = "POD_LOCAL"
    RUNPOD_GLOBAL_PRIVATE = "RUNPOD_GLOBAL_PRIVATE"
    RUNPOD_PUBLIC_TCP = "RUNPOD_PUBLIC_TCP"


@dataclass(frozen=True)
class CapacityRecord:
    provider: str
    gpu_type: str
    gpu_type_id: str
    gpu_count_configurations: tuple[int, ...]
    cloud_tier: CloudTier
    datacenter_id: str | None
    stock_status: str | None
    current_price_per_gpu_hour: float | None
    availability_confidence: str
    network_capability: dict[str, Any] = field(default_factory=dict)
    storage_capability: dict[str, Any] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicalAllocationRecord:
    provider: str
    pod_id: str
    machine_id: str | None
    pod_name: str
    desired_status: str | None
    current_status: str | None
    gpu_model: str | None
    gpu_count: int | None
    gpu_vram_bytes: int | None
    cost_per_hour: float | None
    datacenter_id: str | None
    location: dict[str, Any] = field(default_factory=dict)
    public_ip: str | None = None
    port_mappings: dict[int, int] = field(default_factory=dict)
    private_identity: str | None = None
    image_identity: str | None = None
    creation_timestamp: str | None = None
    termination_timestamp: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EndpointRecord:
    mode: EndpointMode
    pod_id: str
    worker_id: str
    internal_port: int
    host: str
    port: int
    fragment_endpoint_generation: int | None = None

    @property
    def authority(self) -> str:
        return f"{self.host}:{self.port}"


__all__ = [
    "READ_ONLY_OPERATIONS",
    "CapacityRecord",
    "CloudTier",
    "EndpointMode",
    "EndpointRecord",
    "PhysicalAllocationRecord",
    "ProviderMode",
    "ProviderMutationForbidden",
    "ProviderMutationPolicy",
    "ProviderOperation",
]
