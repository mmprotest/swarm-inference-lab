"""Translate Colibri's version-2 resource plan without duplicating its planner."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from swarm_inference.backends.colibri.schemas import (
    ColibriCapabilityReport,
    ExpertInventoryEntry,
    SwarmColibriPlan,
    TensorInventoryEntry,
)

_SAFE_SETTING_MAP = {
    "OMP_NUM_THREADS": ("thread_policy", "omp_num_threads"),
    "COLI_NUMA": ("thread_policy", "numa_policy"),
    "PIPE": ("pipeline_policy", "io_pipeline"),
    "DIRECT": ("storage_policy", "direct_io"),
    "COLI_CUDA_PIPE": ("pipeline_policy", "cuda_resident_pipeline"),
    "COLI_CUDA_ASYNC": ("pipeline_policy", "cuda_async"),
    "LOADERS": ("storage_policy", "loader_threads"),
    "RAM_GB": ("routed_expert_tiers", "ram_budget_gb"),
    "CUDA_EXPERT_GB": ("routed_expert_tiers", "vram_expert_budget_gb"),
    "PIN_GB": ("routed_expert_tiers", "hot_expert_budget_gb"),
    "PILOT_REAL": ("prefetch_policy", "enabled"),
    "PILOT": ("prefetch_policy", "lookahead_depth"),
    "PREFETCH": ("prefetch_policy", "lookahead_depth"),
    "WIDE": ("prefetch_policy", "candidate_width"),
    "SMOOTH": ("prefetch_policy", "routing_momentum"),
    "CONF_LIMIT": ("prefetch_policy", "confidence_limit"),
    "PILOT_EVICT_GUARD": ("prefetch_policy", "eviction_guard"),
    "HOT": ("routed_expert_tiers", "hot_experts_per_layer"),
    "EXPERT_DROP": ("storage_policy", "drop_pages_after_expert_read"),
    "CHUNK": ("pipeline_policy", "chunked_prefill_size"),
}
_FORBIDDEN_SETTINGS = {
    "TOPP",
    "K3_TOPP",
    "TEMP",
    "TEMPERATURE",
    "EBITS",
    "DBITS",
    "QUANTIZATION",
    "TOP_K",
}


def _tune_values(native: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    tune = native.get("tune", {})
    if not isinstance(tune, dict):
        raise ValueError("Colibri plan tune section must be an object")
    for key, entry in tune.items():
        if key.startswith("_"):
            continue
        if not isinstance(entry, dict) or "value" not in entry:
            raise ValueError(f"invalid Colibri tune entry {key}")
        result[key] = str(entry["value"])
    return result


class ColibriPlanTranslator:
    """Import native placement decisions and enforce capacity/semantic gates."""

    def translate(
        self,
        native: dict[str, Any],
        *,
        hardware_fingerprint: str,
        tensors: list[TensorInventoryEntry] | None = None,
        experts: list[ExpertInventoryEntry] | None = None,
        capabilities: ColibriCapabilityReport | None = None,
    ) -> SwarmColibriPlan:
        if native.get("version") != 2:
            raise ValueError(f"unsupported Colibri resource plan version {native.get('version')!r}")
        policy = native.get("policy")
        if not isinstance(policy, dict) or not policy.get("quality_preserving", False):
            raise ValueError("Experiment 009 accepts only Colibri quality-preserving policies")
        model = native.get("model")
        tiers = native.get("tiers")
        cpu = native.get("cpu")
        if not isinstance(model, dict) or not isinstance(tiers, dict) or not isinstance(cpu, dict):
            raise ValueError("Colibri plan is missing model, tiers, or CPU sections")
        disk = tiers.get("disk")
        ram = tiers.get("ram")
        vram = tiers.get("vram")
        if not isinstance(disk, dict) or not isinstance(ram, dict) or not isinstance(vram, dict):
            raise ValueError("Colibri plan must describe disk, RAM, and VRAM tiers")

        expert_bytes = int(model.get("expert_bytes", 0))
        dense_bytes = int(model.get("dense_bytes", 0))
        hot_bytes = int(vram.get("hot_expert_bytes", 0))
        warm_bytes = int(ram.get("warm_expert_bytes", 0))
        cold_bytes = int(disk.get("cold_expert_bytes", 0))
        ram_budget = int(ram.get("budget_bytes", 0))
        vram_budget = int(vram.get("budget_bytes", 0))
        runtime_bytes = int(ram.get("runtime_bytes", 0))
        planner_cache_bytes = int(ram.get("expert_cache_bytes", 0))
        planner_cache_slots = int(ram.get("cache_slots_per_layer", 0))
        if (
            min(
                expert_bytes,
                dense_bytes,
                hot_bytes,
                warm_bytes,
                cold_bytes,
                ram_budget,
                vram_budget,
            )
            < 0
        ):
            raise ValueError("Colibri plan contains negative byte accounting")
        if min(planner_cache_bytes, planner_cache_slots) < 0:
            raise ValueError("Colibri plan contains negative cache capacity")
        if planner_cache_bytes > ram_budget:
            raise ValueError("Colibri expert cache capacity exceeds its RAM budget")
        if hot_bytes + warm_bytes + cold_bytes != expert_bytes:
            raise ValueError(
                "Colibri routed expert tier bytes do not reconcile to model expert bytes"
            )
        if hot_bytes > vram_budget:
            raise ValueError("Colibri plan places more hot experts than its VRAM budget")
        if dense_bytes + runtime_bytes + warm_bytes > ram_budget:
            raise ValueError("Colibri plan exceeds its RAM budget")
        if experts is not None:
            inventoried_expert_bytes = sum(expert.total_bytes for expert in experts)
            if inventoried_expert_bytes != expert_bytes:
                raise ValueError(
                    "Colibri plan expert bytes do not reconcile with the imported expert inventory: "
                    f"{expert_bytes} != {inventoried_expert_bytes}"
                )
            if int(model.get("expert_count", len(experts))) != len(experts):
                raise ValueError("Colibri plan expert count does not reconcile with inventory")
            per_layer: dict[int, int] = {}
            for expert in experts:
                per_layer[expert.layer_id] = per_layer.get(expert.layer_id, 0) + 1
            physical_slots_per_layer = max(per_layer.values(), default=0)
        else:
            physical_slots_per_layer = int(model.get("expert_count", 0))
        if tensors is not None:
            tensor_bytes = sum(tensor.byte_size for tensor in tensors)
            planned_tensor_bytes = dense_bytes + expert_bytes
            if tensor_bytes != planned_tensor_bytes:
                raise ValueError(
                    "Colibri plan dense+expert bytes do not reconcile with tensor inventory: "
                    f"{planned_tensor_bytes} != {tensor_bytes}"
                )

        tune = _tune_values(native)
        unsupported = ["tensor_microshard_execution", "dynamic_residency_reconfiguration"]
        if capabilities is not None:
            if hot_bytes and not (
                capabilities.supports_cuda
                or capabilities.supports_vulkan
                or capabilities.supports_metal
            ):
                raise ValueError("Colibri plan assigns experts to a non-executable VRAM tier")
            if not capabilities.supports_cuda:
                unsupported.append("cuda_execution")
            if not capabilities.supports_vulkan:
                unsupported.append("vulkan_execution")
            if not capabilities.supports_multi_gpu:
                unsupported.append("multi_gpu_execution")
        prefetch_enabled = tune.get("PILOT_REAL", tune.get("PILOT", "0")) not in {"0", "off"}
        # Colibri's planner describes the amount of RAM that could be devoted
        # to cache slots.  On large-memory hosts that theoretical slot count
        # can exceed the model's physical experts per layer.  Preserve the
        # planner capacity for provenance, but expose only an executable,
        # inventory-bounded cache to the swarm planner.
        cache_slots = min(planner_cache_slots, physical_slots_per_layer)
        cache_bytes = min(planner_cache_bytes, expert_bytes)
        return SwarmColibriPlan(
            model={
                **model,
                "policy": policy.get("name"),
                "projected_hit_rate": native.get("projected_hit_rate"),
            },
            hardware_fingerprint=hardware_fingerprint,
            dense_residency={
                "tier": "ram",
                "bytes": dense_bytes,
                "permanent": True,
                "execution": "colibri_managed",
            },
            shared_expert_residency={
                "tier": "ram",
                "bytes": int(model.get("shared_expert_bytes", 0)),
                "permanent": True,
                "execution": "colibri_managed",
            },
            routed_expert_tiers={
                "vram": {
                    "bytes": hot_bytes,
                    "expert_capacity": int(vram.get("expert_capacity", 0)),
                    "role": "learned_hot_expert_storage",
                },
                "ram": {
                    "bytes": warm_bytes,
                    "cache_bytes": cache_bytes,
                    "cache_slots_per_layer": cache_slots,
                    "planner_cache_capacity_bytes": planner_cache_bytes,
                    "planner_cache_capacity_slots_per_layer": planner_cache_slots,
                    "capacity_reconciled_to_inventory": (
                        cache_bytes != planner_cache_bytes or cache_slots != planner_cache_slots
                    ),
                    "role": "hot_plus_lru_expert_cache",
                },
                "nvme": {"bytes": cold_bytes, "role": "immutable_cold_storage"},
            },
            ram_budget_bytes=ram_budget,
            vram_budget_bytes=vram_budget,
            storage_policy={
                "model_bytes": int(disk.get("model_bytes", 0)),
                "available_bytes": int(disk.get("available_bytes", 0)),
                "direct_io": tune.get("DIRECT"),
                "loader_threads": tune.get("LOADERS"),
                "measured_probe_gbs": native.get("ssd_probe_gbs"),
                "probe_state": native.get("ssd_probe_state"),
            },
            prefetch_policy={
                "enabled": prefetch_enabled,
                "lookahead_depth": tune.get("PREFETCH"),
                "source": "colibri_native_policy",
            },
            pipeline_policy={
                "io_pipeline": tune.get("PIPE"),
                "cuda_resident_pipeline": tune.get("COLI_CUDA_PIPE"),
                "cuda_async": tune.get("COLI_CUDA_ASYNC"),
            },
            thread_policy={
                "physical_cores": int(cpu.get("physical_cores", 0)),
                "sockets": int(cpu.get("sockets", 1)),
                "policy": cpu.get("thread_policy", "physical-cores"),
                "omp_num_threads": tune.get("OMP_NUM_THREADS"),
                "numa": tune.get("COLI_NUMA"),
            },
            expected_bottleneck=str(native.get("expected_bottleneck", "unknown")),
            source="colibri_plan",
            unsupported_features=unsupported,
        )

    def bounded_adjustment(
        self,
        plan: SwarmColibriPlan,
        settings: dict[str, Any],
        *,
        supported_settings: set[str],
    ) -> SwarmColibriPlan:
        forbidden = _FORBIDDEN_SETTINGS.intersection(settings)
        if forbidden:
            raise ValueError(f"quality-affecting settings are forbidden: {sorted(forbidden)}")
        unknown = set(settings).difference(_SAFE_SETTING_MAP)
        if unknown:
            raise ValueError(f"settings are outside the bounded Colibri search: {sorted(unknown)}")
        unsupported = set(settings).difference(supported_settings)
        if unsupported:
            raise ValueError(f"backend does not execute requested settings: {sorted(unsupported)}")
        payload = plan.model_dump(mode="python")
        for setting, value in settings.items():
            section, field = _SAFE_SETTING_MAP[setting]
            section_payload = payload[section]
            if not isinstance(section_payload, dict):
                raise ValueError(f"invalid plan section {section}")
            section_payload[field] = value
        payload["source"] = "swarm_bounded_adjustment"
        payload["model"] = deepcopy(payload["model"])
        payload["model"]["bounded_settings"] = settings
        return SwarmColibriPlan.model_validate(payload)

    @staticmethod
    def environment(plan: SwarmColibriPlan) -> dict[str, str]:
        """Translate only explicitly represented, semantics-neutral settings."""

        env: dict[str, str] = {
            "RAM_GB": f"{plan.ram_budget_bytes / (1024**3):.3f}",
        }
        if plan.vram_budget_bytes:
            env["CUDA_EXPERT_GB"] = f"{plan.vram_budget_bytes / (1024**3):.3f}"
        reverse = {value: key for key, value in _SAFE_SETTING_MAP.items()}
        dumped = plan.model_dump(mode="python")
        for (section, field), setting in reverse.items():
            value = dumped.get(section, {}).get(field)
            if value is not None:
                env[setting] = str(int(value) if isinstance(value, bool) else value)
        return env
