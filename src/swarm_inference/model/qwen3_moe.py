"""Composable native-stage adapter for Transformers Qwen3 sparse MoE weights.

Attention, KV state, routing and dense endpoint operations remain in contiguous
native stages. Routed experts may remain local or be delegated through the
canonical direct expert data plane without relaying activations via the
coordinator.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

from swarm_inference.model.adapter import (
    AdapterSupportReport,
    AdapterSupportStatus,
    partition_metadata_from_description,
    validate_dense_stage_assignment,
)
from swarm_inference.model.architecture import normalize_model_architecture
from swarm_inference.model.descriptor import ResolvedModelDescriptor
from swarm_inference.model.partition import ModelPartitionMetadata, StageAssignment
from swarm_inference.model.qwen3 import Qwen3Adapter


class Qwen3MoeAdapter(Qwen3Adapter):
    """Qwen3 MoE safetensors adapter with exact routed-expert delegation."""

    adapter_id = "qwen3_moe"
    adapter_version = "1"
    supported_model_types: ClassVar[frozenset[str]] = frozenset({"qwen3_moe"})
    supported_architectures: ClassVar[frozenset[str]] = frozenset({"Qwen3MoeForCausalLM"})
    _native_raw_architectures: ClassVar[frozenset[str]] = frozenset(
        {"qwen3_moe", "qwen3moe", "Qwen3MoeForCausalLM"}
    )
    _expert_tensor = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")

    def probe_model(self, model: ResolvedModelDescriptor) -> AdapterSupportReport:
        if model.format != "safetensors":
            return AdapterSupportReport(
                self.adapter_id,
                AdapterSupportStatus.UNSUPPORTED_FORMAT,
                "native Qwen3 MoE execution requires a safetensors checkpoint; "
                "GGUF belongs to an engine that opens GGUF natively",
            )
        if normalize_model_architecture(model.architecture) != "qwen3_moe":
            return AdapterSupportReport(
                self.adapter_id,
                AdapterSupportStatus.UNSUPPORTED_ARCHITECTURE,
                f"architecture {model.architecture!r} is not Qwen3 sparse MoE",
            )
        if (
            model.architecture_raw is not None
            and model.architecture_raw not in self._native_raw_architectures
        ):
            return AdapterSupportReport(
                self.adapter_id,
                AdapterSupportStatus.UNSUPPORTED_ARCHITECTURE,
                "the Qwen3 MoE family is recognized, but native-stage currently "
                "supports the Transformers qwen3_moe representation only; "
                f"resolved representation={model.architecture_raw!r}",
            )
        return AdapterSupportReport(
            self.adapter_id,
            AdapterSupportStatus.SUPPORTED,
            "Qwen3 MoE safetensors supports native contiguous outer stages and "
            "direct local, whole-remote, or microsharded routed experts",
        )

    def validate_stage_assignment(
        self,
        model_path: Any,
        *,
        assignment: StageAssignment,
        stage_count: int,
        model_revision: str,
        tokenizer_revision: str,
        remote_experts: set[tuple[int, int]] | None = None,
    ) -> ModelPartitionMetadata:
        description = self.describe(
            Path(model_path),
            model_id=str(model_path),
            model_revision=model_revision,
        )
        remote = set(remote_experts or set())
        outside = sorted(
            item for item in remote if not assignment.layer_start <= item[0] < assignment.layer_end
        )
        if outside:
            raise ValueError(f"delegated expert lies outside its native stage: {outside[:3]}")
        expert_count = int(description.config.get("num_experts") or 0)
        if any(expert < 0 or expert >= expert_count for _, expert in remote):
            raise ValueError("delegated Qwen3 expert index is outside the configured expert bank")
        delegated_bytes = sum(
            tensor.bytes
            for tensor in description.tensors
            if (match := self._expert_tensor.match(tensor.name))
            and (int(match.group(1)), int(match.group(2))) in remote
        )
        metadata = partition_metadata_from_description(
            description,
            tokenizer_revision=tokenizer_revision,
        )
        validate_dense_stage_assignment(
            metadata,
            assignment=replace(
                assignment,
                weight_bytes=assignment.weight_bytes + delegated_bytes,
            ),
            stage_count=stage_count,
        )
        return metadata

    def exclude_delegated_tensor_names(
        self,
        tensor_names: tuple[str, ...],
        remote_experts: set[tuple[int, int]],
    ) -> set[str]:
        return {
            name
            for name in tensor_names
            if (match := self._expert_tensor.match(name))
            and (int(match.group(1)), int(match.group(2))) in remote_experts
        }

    def create_stage_executor(self, *args: Any, **kwargs: Any) -> Any:
        request = kwargs.pop("request", None)
        resolved_model_path = kwargs.pop("resolved_model_path", None)
        fast_path_profile_store = kwargs.pop("fast_path_profile_store", None)
        expert_tls = kwargs.pop("expert_tls", None)
        if request is None:
            return super().create_stage_executor(*args, **kwargs)
        from swarm_inference.backends.colibri.torch_backend import ColibriMoeBackend
        from swarm_inference.execution.microshard import MicroshardRange
        from swarm_inference.execution.moe import (
            HybridMoeBackend,
            LocalMoeBackend,
            MicroshardRemoteBackend,
            MicroshardTarget,
            WholeExpertRemoteBackend,
            WholeExpertTarget,
        )
        from swarm_inference.execution.qwen3_stage import Qwen3StageExecutor
        from swarm_inference.protocol.product import ProductStageExpertPlan
        from swarm_inference.transport.expert import ExpertTransportClient

        expert_plan = (
            ProductStageExpertPlan.model_validate(request.expert_plan)
            if request.expert_plan is not None
            else None
        )
        delegated_placements = (
            [item for item in expert_plan.placements if item.strategy != "local"]
            if expert_plan is not None
            else []
        )
        external_placements = [
            item
            for item in delegated_placements
            if item.strategy in {"whole-remote", "microshard-remote"}
        ]
        colibri_experts = {
            (item.layer_id, item.expert_id)
            for item in delegated_placements
            if item.strategy == "colibri"
        }
        delegated_experts = {
            (item.layer_id, item.expert_id)
            for item in delegated_placements
            if not item.local_fallback_permitted
        }
        backend_factory = None
        if delegated_placements:
            assert expert_plan is not None
            clients: dict[str, ExpertTransportClient] = {}
            whole_targets: dict[tuple[int, int], WholeExpertTarget] = {}
            micro_targets: dict[tuple[int, int], list[MicroshardTarget]] = {}
            placement: dict[tuple[int, int], str] = {}
            for item in expert_plan.placements:
                key = (item.layer_id, item.expert_id)
                placement[key] = item.strategy
                for worker_id, endpoint in item.worker_endpoints.items():
                    if not endpoint:
                        raise ValueError("remote expert placement has no direct endpoint")
                    clients.setdefault(
                        worker_id,
                        ExpertTransportClient(endpoint, tls=expert_tls),
                    )
                if item.strategy == "whole-remote":
                    if len(item.worker_ids) != 1:
                        raise ValueError("whole-expert placement requires exactly one owner")
                    worker_id = item.worker_ids[0]
                    whole_targets[key] = WholeExpertTarget(
                        worker_id=worker_id,
                        client=clients[worker_id],
                        expert_hash=item.expert_hashes.get(worker_id, ""),
                    )
                elif item.strategy == "microshard-remote":
                    micro_targets[key] = [
                        MicroshardTarget(
                            ownership=MicroshardRange(
                                worker_id=str(shard["worker_id"]),
                                layer_id=item.layer_id,
                                expert_id=item.expert_id,
                                hidden_start=int(shard["hidden_start"]),
                                hidden_end=int(shard["hidden_end"]),
                                logical_intermediate_dimension=int(
                                    shard["logical_intermediate_dimension"]
                                ),
                                content_hash=str(shard["content_hash"]),
                                quantization_group_size=(
                                    int(shard["quantization_group_size"])
                                    if shard.get("quantization_group_size") is not None
                                    else None
                                ),
                            ),
                            client=clients[str(shard["worker_id"])],
                        )
                        for shard in item.microshards
                    ]

            def make_backend(local_modules: dict[tuple[int, int], Any]) -> Any:
                local = LocalMoeBackend(local_modules)
                colibri = (
                    ColibriMoeBackend(
                        model_path=Path(resolved_model_path),
                        model_id=request.model_id,
                        model_revision=request.model_revision,
                        selected_experts=colibri_experts,
                        device=request.device,
                        cache_budget_bytes=expert_plan.cache_budget_bytes,
                    )
                    if colibri_experts and resolved_model_path is not None
                    else None
                )
                if colibri_experts and colibri is None:
                    raise FileNotFoundError(
                        "Colibri expert execution requires the immutable stage artifact"
                    )
                whole = (
                    WholeExpertRemoteBackend(
                        targets=whole_targets,
                        model_id=request.model_id,
                        model_revision=request.model_revision,
                        model_fingerprint=request.expert_model_fingerprint or "",
                        quantization_fingerprint=request.expert_quantization_fingerprint or "",
                        topology_id=request.topology_id,
                        route_generation=request.route_generation,
                    )
                    if whole_targets
                    else None
                )
                micro = (
                    MicroshardRemoteBackend(
                        targets=micro_targets,
                        model_id=request.model_id,
                        model_revision=request.model_revision,
                        model_fingerprint=request.expert_model_fingerprint or "",
                        quantization_fingerprint=request.expert_quantization_fingerprint or "",
                        topology_id=request.topology_id,
                        route_generation=request.route_generation,
                    )
                    if micro_targets
                    else None
                )
                return HybridMoeBackend(
                    local=local,
                    colibri=colibri,
                    whole_remote=whole,
                    microshard_remote=micro,
                    placement=placement,
                    fallback_placements={
                        (item.layer_id, item.expert_id)
                        for item in external_placements
                        if item.local_fallback_permitted
                    },
                    require_remote=expert_plan.require_remote_experts,
                )

            backend_factory = make_backend
        return Qwen3StageExecutor.from_load_request(
            request,
            resolved_model_path,
            fast_path_profile_store=fast_path_profile_store,
            moe_backend_factory=backend_factory,
            remote_experts=delegated_experts,
        )

    def fast_paths(self) -> tuple[Any, ...]:
        # Dense Qwen3 CUDA kernels are not interchangeable with sparse expert
        # kernels.  Eager Transformers execution is the verified native path.
        return ()


__all__ = ["Qwen3MoeAdapter"]
