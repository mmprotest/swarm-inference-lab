"""Native adapter for the OLMoE checkpoint family."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, ClassVar

from swarm_inference.config.models import ModelManifest, StageDefinition
from swarm_inference.exceptions import IntegrityError, UnsupportedArchitectureError
from swarm_inference.model.adapter import (
    AdapterSupportReport,
    AdapterSupportStatus,
    ComponentKind,
    ComponentRef,
    ModelDescription,
    TensorInfo,
)
from swarm_inference.model.descriptor import ResolvedModelDescriptor
from swarm_inference.model.partition import ModelPartitionMetadata, StageAssignment

_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


class OlmoeAdapter:
    adapter_id = "olmoe"
    adapter_version = "2"
    supported_model_types: ClassVar[frozenset[str]] = frozenset({"olmoe"})
    supported_architectures: ClassVar[frozenset[str]] = frozenset(
        {"OlmoeForCausalLM", "OLMoEForCausalLM"}
    )

    def supports(self, config: Any) -> bool:
        if hasattr(config, "to_dict"):
            config = config.to_dict()
        if not isinstance(config, dict):
            return False
        model_type = str(config.get("model_type", "")).lower()
        architectures = {str(item) for item in config.get("architectures") or []}
        return model_type in self.supported_model_types and (
            not architectures or bool(architectures & self.supported_architectures)
        )

    def probe_model(self, model: ResolvedModelDescriptor) -> AdapterSupportReport:
        if model.format != "safetensors":
            return AdapterSupportReport(
                self.adapter_id,
                AdapterSupportStatus.UNSUPPORTED_FORMAT,
                "native adapter requires a safetensors checkpoint",
            )
        architecture = (model.architecture or "").lower()
        if "olmoe" not in architecture:
            return AdapterSupportReport(
                self.adapter_id,
                AdapterSupportStatus.UNSUPPORTED_ARCHITECTURE,
                f"architecture {model.architecture!r} is not supported",
            )
        return AdapterSupportReport(
            self.adapter_id,
            AdapterSupportStatus.SUPPORTED,
            "immutable safetensors architecture is supported",
        )

    def map_tensor_to_component(self, tensor_name: str) -> ComponentRef:
        if tensor_name.startswith("model.embed_tokens."):
            return ComponentRef(ComponentKind.EMBEDDING)
        if tensor_name.startswith("model.layers."):
            fields = tensor_name.split(".")
            if len(fields) < 4:
                raise UnsupportedArchitectureError(f"invalid decoder tensor name {tensor_name}")
            return ComponentRef(ComponentKind.DECODER_LAYER, int(fields[2]))
        if tensor_name.startswith("model.norm."):
            return ComponentRef(ComponentKind.FINAL_NORM)
        if tensor_name.startswith("lm_head."):
            return ComponentRef(ComponentKind.OUTPUT_HEAD)
        raise UnsupportedArchitectureError(f"unmapped checkpoint tensor {tensor_name}")

    def describe(
        self,
        model_path: Path,
        *,
        model_id: str,
        model_revision: str,
    ) -> ModelDescription:
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise IntegrityError(f"model config is missing: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not self.supports(config):
            raise UnsupportedArchitectureError("checkpoint configuration is unsupported")
        index_path = model_path / "model.safetensors.index.json"
        raw_index = json.loads(index_path.read_text(encoding="utf-8"))
        mapping = raw_index.get("weight_map") if isinstance(raw_index, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise IntegrityError("safetensors index has no weight map")
        from safetensors import safe_open

        tensors: list[TensorInfo] = []
        by_file: dict[str, list[str]] = {}
        for name, source in mapping.items():
            by_file.setdefault(str(source), []).append(str(name))
        for source, names in sorted(by_file.items()):
            with safe_open(model_path / source, framework="pt", device="cpu") as handle:
                for name in sorted(names):
                    tensor = handle.get_slice(name)
                    shape = tuple(int(item) for item in tensor.get_shape())
                    dtype = str(tensor.get_dtype()).upper()
                    try:
                        width = _DTYPE_BYTES[dtype]
                    except KeyError as exc:
                        raise UnsupportedArchitectureError(
                            f"unsupported tensor dtype {dtype}"
                        ) from exc
                    tensors.append(
                        TensorInfo(
                            name=name,
                            source_file=source,
                            dtype=dtype.lower(),
                            shape=shape,
                            bytes=math.prod(shape) * width,
                            component=self.map_tensor_to_component(name),
                        )
                    )
        return ModelDescription(
            model_id=model_id,
            model_revision=model_revision,
            model_path=model_path,
            config=config,
            tensors=tensors,
            source_file_hashes={name: _sha256(model_path / name) for name in sorted(by_file)},
            config_file_hashes={"config.json": _sha256(config_path)},
            tokenizer_file_hashes={
                path.name: _sha256(path)
                for path in sorted(model_path.glob("tokenizer*"))
                if path.is_file()
            },
        )

    def inspect(self, model: ResolvedModelDescriptor) -> ModelDescription:
        if not model.local_paths:
            raise FileNotFoundError("native inspection requires acquired local files")
        roots = {str(Path(item).resolve().parent) for item in model.local_paths}
        if len(roots) != 1:
            raise IntegrityError("native checkpoint files do not share one directory")
        return self.describe(
            Path(next(iter(roots))),
            model_id=model.model_id,
            model_revision=model.revision,
        )

    def create_stage_module(
        self,
        config: Any,
        stage: StageDefinition,
        device: Any,
        dtype: Any,
    ) -> Any:
        raise TypeError("stage construction requires an acquired checkpoint path")

    def load_stage_weights(
        self,
        module: Any,
        shard_path: Path,
        *,
        manifest: ModelManifest,
    ) -> list[str]:
        raise TypeError("the stateful stage executor loads its owned artifact atomically")

    def build_stage_artifact(self, *args: Any, **kwargs: Any) -> Any:
        from swarm_inference.cluster.artifacts import build_native_stage_artifact

        return build_native_stage_artifact(self, *args, **kwargs)

    def create_stage_executor(self, *args: Any, **kwargs: Any) -> Any:
        request = kwargs.pop("request", None)
        resolved_model_path = kwargs.pop("resolved_model_path", None)
        kwargs.pop("fast_path_profile_store", None)
        if request is not None:
            return self._create_product_stage(request, resolved_model_path)
        from swarm_inference.execution.olmoe_stage import ContiguousOlmoeStage

        return ContiguousOlmoeStage(*args, **kwargs)

    def inspect_partition_metadata(
        self,
        model_path: Path,
        *,
        model_revision: str,
        tokenizer_revision: str,
    ) -> ModelPartitionMetadata:
        from swarm_inference.model.olmoe import inspect_olmoe_partition_metadata

        return inspect_olmoe_partition_metadata(
            model_path,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
        )

    def validate_stage_assignment(
        self,
        model_path: Path,
        *,
        assignment: StageAssignment,
        stage_count: int,
        model_revision: str,
        tokenizer_revision: str,
        remote_experts: set[tuple[int, int]] | None = None,
    ) -> ModelPartitionMetadata:
        from swarm_inference.model.olmoe import validate_olmoe_stage_assignment

        return validate_olmoe_stage_assignment(
            model_path,
            assignment=assignment,
            stage_count=stage_count,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            remote_experts=remote_experts,
        )

    @staticmethod
    def _create_product_stage(request: Any, resolved_model_path: Path | None) -> Any:
        if resolved_model_path is None:
            raise FileNotFoundError("native stage loading requires a resolved local checkpoint")
        import torch

        from swarm_inference.execution.olmoe_stage import ContiguousOlmoeStage
        from swarm_inference.protocol.product import ProductStageExpertPlan

        dtype_name = str(request.dtype).lower()
        dtype = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "f16": torch.float16,
            "float32": torch.float32,
            "f32": torch.float32,
        }[dtype_name]
        expert_plan = (
            ProductStageExpertPlan.model_validate(request.expert_plan)
            if request.expert_plan is not None
            else None
        )
        remote_placements = (
            [item for item in expert_plan.placements if item.strategy != "local"]
            if expert_plan is not None
            else []
        )
        remote_experts = {
            (item.layer_id, item.expert_id)
            for item in remote_placements
            if not item.local_fallback_permitted
        }
        backend_factory = None
        if remote_placements:
            assert expert_plan is not None
            from swarm_inference.execution.microshard import MicroshardRange
            from swarm_inference.execution.moe import (
                HybridMoeBackend,
                LocalMoeBackend,
                MicroshardRemoteBackend,
                MicroshardTarget,
                WholeExpertRemoteBackend,
                WholeExpertTarget,
            )
            from swarm_inference.transport.expert import ExpertTransportClient

            clients: dict[str, ExpertTransportClient] = {}
            whole_targets: dict[tuple[int, int], WholeExpertTarget] = {}
            micro_targets: dict[tuple[int, int], list[MicroshardTarget]] = {}
            placement: dict[tuple[int, int], str] = {}
            for item in expert_plan.placements:
                key = (item.layer_id, item.expert_id)
                placement[key] = item.strategy
                for worker_id, endpoint in item.worker_endpoints.items():
                    if not endpoint:
                        raise ValueError("remote expert placement has no reachable endpoint")
                    clients.setdefault(worker_id, ExpertTransportClient(endpoint))
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
                    targets = []
                    for shard in item.microshards:
                        worker_id = str(shard["worker_id"])
                        targets.append(
                            MicroshardTarget(
                                ownership=MicroshardRange(
                                    worker_id=worker_id,
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
                                client=clients[worker_id],
                            )
                        )
                    micro_targets[key] = targets

            def make_backend(local_modules: dict[tuple[int, int], torch.nn.Module]) -> Any:
                local = LocalMoeBackend(local_modules)
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
                    whole_remote=whole,
                    microshard_remote=micro,
                    placement=placement,
                    fallback_placements={
                        (item.layer_id, item.expert_id)
                        for item in remote_placements
                        if item.local_fallback_permitted
                    },
                    require_remote=expert_plan.require_remote_experts,
                )

            backend_factory = make_backend
        return ContiguousOlmoeStage(
            model_path=resolved_model_path,
            assignment=request.assignment,
            stage_count=request.stage_count,
            device=request.device,
            dtype=dtype,
            moe_backend_factory=backend_factory,
            remote_experts=remote_experts,
        )

    def reference_executor(self, model: ResolvedModelDescriptor, **kwargs: Any) -> Any:
        from transformers import AutoModelForCausalLM

        source = Path(model.local_paths[0]).parent if model.local_paths else model.model_id
        return AutoModelForCausalLM.from_pretrained(
            source,
            revision=None if model.local_paths else model.revision,
            local_files_only=bool(model.local_paths),
            **kwargs,
        )

    def fast_paths(self) -> tuple[Any, ...]:
        return ()


__all__ = ["OlmoeAdapter"]
