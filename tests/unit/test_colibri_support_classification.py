from __future__ import annotations

from swarm_inference.backends.colibri.adapters import Glm52ColibriAdapter
from swarm_inference.backends.colibri.architecture import (
    ColibriCompatibilityStatus,
    ColibriRuntimeCapabilities,
)
from swarm_inference.model.descriptor import ResolvedModelDescriptor


def _model(*, quantization: str = "int4-g64") -> ResolvedModelDescriptor:
    return ResolvedModelDescriptor(
        model_id="fixture/glm",
        revision="a" * 40,
        content_fingerprint="sha256:" + "1" * 64,
        source_type="local",
        format="safetensors",
        architecture="glm_moe_dsa",
        files=(),
        quantization=quantization,
        weight_bytes=1024,
        tokenizer_identity="sha256:" + "2" * 64,
        configuration={
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": 1,
            "hidden_size": 8,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 4,
            "n_group": 1,
        },
    )


def _runtime(*, installed: bool = True) -> ColibriRuntimeCapabilities:
    return ColibriRuntimeCapabilities(
        installed=installed,
        runtime_version="b085b488" if installed else None,
        binary_hashes={"colibri": "1" * 64} if installed else {},
        adapters=("glm-5.2",) if installed else (),
        formats=("safetensors",) if installed else (),
        quantizations=("int4-g64",) if installed else (),
        device_types=("cpu",) if installed else (),
    )


def test_unavailable_runtime_is_not_reported_as_an_unsupported_model() -> None:
    result = Glm52ColibriAdapter().supports(_model(), _runtime(installed=False))

    assert not result.supported
    assert not result.runtime_supported
    assert result.classification is ColibriCompatibilityStatus.RUNTIME_UNAVAILABLE
    assert "not installed" in " ".join(result.reasons)


def test_available_runtime_preserves_model_compatibility_classification() -> None:
    result = Glm52ColibriAdapter().supports(_model(quantization="float64"), _runtime())

    assert not result.supported
    assert result.runtime_supported
    assert result.classification is ColibriCompatibilityStatus.UNSUPPORTED_BY_PINNED_COLIBRI
    assert not result.quantization_supported
