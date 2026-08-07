from __future__ import annotations

from typing import Any

import pytest

from swarm_inference.model.architecture import ModelArchitectureProfile
from swarm_inference.model.descriptor import ModelFileDescriptor, ResolvedModelDescriptor
from swarm_inference.model.discovery import discover_models
from swarm_inference.model.resolver import ModelResolution


class _Inspector:
    def inspect(self, source: str, **kwargs: Any) -> ModelResolution:
        del kwargs
        if source == "broken/model":
            raise RuntimeError("fixture metadata unavailable")
        profile = ModelArchitectureProfile(
            architecture_id="qwen3_5_moe",
            adapter_id="qwen3-5-moe",
            dense_or_moe="moe",
            layer_count=4,
            hidden_size=32,
            attention_type="hybrid",
            expert_count=8,
            experts_per_token=2,
            shared_expert_count=0,
            expert_intermediate_size=48,
            router_type="softmax-top-k",
            tensor_layout="fixture",
            checkpoint_format="safetensors",
            capabilities=frozenset({"causal-lm", "routed-experts"}),
        )
        descriptor = ResolvedModelDescriptor(
            model_id=source,
            revision="a" * 40,
            content_fingerprint="sha256:" + "b" * 64,
            source_type="huggingface",
            format="safetensors",
            architecture=profile.architecture_id,
            architecture_raw="Qwen3_5MoeForConditionalGeneration",
            architecture_source="config.architectures",
            files=(ModelFileDescriptor(relative_path="model.safetensors", size_bytes=1),),
            weight_bytes=1,
            architecture_profile=profile,
        )
        return ModelResolution(descriptor)


def test_bounded_discovery_reports_profiles_and_failures_without_name_dispatch() -> None:
    report = discover_models(
        ("renamed/fine-tune", "renamed/fine-tune", "broken/model"),
        inspector=_Inspector(),
    )

    assert report.inspected_count == 2
    assert report.records[0].requested_reference == "renamed/fine-tune"
    assert report.records[0].architecture_id == "qwen3_5_moe"
    assert report.records[0].status == "PROFILED"
    assert report.records[1].status == "INSPECTION_FAILED"
    assert "fixture metadata unavailable" in str(report.records[1].inspection_error)


def test_discovery_limit_is_enforced_before_network_or_disk_inspection() -> None:
    with pytest.raises(ValueError, match="bounded limit"):
        discover_models(("one", "two"), inspector=_Inspector(), maximum_models=1)
