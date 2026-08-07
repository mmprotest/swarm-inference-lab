"""Native contiguous-stage adapter for Transformers Qwen3 sparse MoE weights.

Every decoder layer, including all of that layer's experts and router, stays on
the worker that owns the layer.  The WAN boundary is therefore still the
Experiment 011 activation boundary; this adapter does not introduce expert RPC.
"""

from __future__ import annotations

from typing import Any, ClassVar

from swarm_inference.model.adapter import AdapterSupportReport, AdapterSupportStatus
from swarm_inference.model.architecture import normalize_model_architecture
from swarm_inference.model.descriptor import ResolvedModelDescriptor
from swarm_inference.model.partition import ModelPartitionMetadata, StageAssignment
from swarm_inference.model.qwen3 import Qwen3Adapter


class Qwen3MoeAdapter(Qwen3Adapter):
    """Qwen3 MoE safetensors adapter using whole-layer expert ownership."""

    adapter_id = "qwen3_moe"
    adapter_version = "1"
    supported_model_types: ClassVar[frozenset[str]] = frozenset({"qwen3_moe"})
    supported_architectures: ClassVar[frozenset[str]] = frozenset({"Qwen3MoeForCausalLM"})
    _native_raw_architectures: ClassVar[frozenset[str]] = frozenset(
        {"qwen3_moe", "qwen3moe", "Qwen3MoeForCausalLM"}
    )

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
            "Qwen3 MoE safetensors is supported as contiguous stages with all "
            "experts local to their owning decoder layer",
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
        if remote_experts:
            raise ValueError(
                "Qwen3 MoE native stages currently require whole-expert ownership "
                "inside each contiguous layer; fine-grained remote expert RPC is not "
                "admitted across stage boundaries"
            )
        return super().validate_stage_assignment(
            model_path,
            assignment=assignment,
            stage_count=stage_count,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            remote_experts=None,
        )

    def fast_paths(self) -> tuple[Any, ...]:
        # Dense Qwen3 CUDA kernels are not interchangeable with sparse expert
        # kernels.  Eager Transformers execution is the verified native path.
        return ()


__all__ = ["Qwen3MoeAdapter"]
