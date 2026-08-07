"""Immutable dependency and bridge protocol constants for Colibri."""

from __future__ import annotations

COLIBRI_REPOSITORY = "JustVugg/colibri"
COLIBRI_REPOSITORY_URL = "https://github.com/JustVugg/colibri.git"
COLIBRI_RELEASE = "v1.4.0"
COLIBRI_COMMIT = "b085b48888a88d9a1c00b151a9979774b72cdbfd"
COLIBRI_LICENSE = "Apache-2.0"
COLIBRI_BRIDGE_VERSION = "1.0"
COLIBRI_EVENT_SCHEMA_VERSION = "1.0"

BRIDGE_EVENT_TYPES = frozenset(
    {
        "engine_started",
        "engine_ready",
        "engine_stopped",
        "capability_report",
        "model_inventory",
        "tensor_inventory",
        "tier_inventory",
        "placement_plan",
        "request_started",
        "prefill_started",
        "prefill_completed",
        "decode_token_started",
        "decode_token_completed",
        "request_completed",
        "route_summary",
        "route_event",
        "expert_cache_hit",
        "expert_cache_miss",
        "expert_prefetch_started",
        "expert_prefetch_completed",
        "expert_loaded",
        "expert_promoted",
        "expert_demoted",
        "expert_evicted",
        "storage_read",
        "host_to_device_transfer",
        "device_to_host_transfer",
        "cpu_compute",
        "gpu_compute",
        "resource_snapshot",
        "warning",
        "error",
    }
)
