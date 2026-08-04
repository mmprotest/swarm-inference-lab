from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from swarm_inference.config.models import TensorSpec
from swarm_inference.execution.interfaces import StageExecutor, WeightOwnership
from swarm_inference.execution.olmoe_stage import (
    ContiguousOlmoeStage,
    StageSessionState,
)
from swarm_inference.model.partition import (
    LayerCost,
    ModelPartitionMetadata,
    StageAssignment,
    build_stage_plan,
    stage_assignment_from_definition,
)
from swarm_inference.transport.stage_tensor import pack_tensor, unpack_tensor


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (torch.bfloat16, [1.0, -2.0, 3.5, 4.0, 5.0, 6.0]),
        (torch.float16, [1.0, -2.0, 3.5, 4.0, 5.0, 6.0]),
        (torch.float32, [1.0, -2.0, 3.5, 4.0, 5.0, 6.0]),
        (torch.int64, [1, -2, 3, 4, 5, 6]),
        (torch.int32, [1, -2, 3, 4, 5, 6]),
        (torch.uint8, [1, 2, 3, 4, 5, 6]),
    ],
)
def test_stage_tensor_round_trip_preserves_supported_dtype_and_shape(
    dtype: torch.dtype, values: list[float]
) -> None:
    source = torch.tensor(values, dtype=dtype).reshape(2, 3).transpose(0, 1)
    assert not source.is_contiguous()
    packed = pack_tensor(source, requested_mode="none")
    restored, decode_ns = unpack_tensor(packed.payload, packed.attributes())
    assert restored.dtype == dtype
    assert restored.shape == source.shape
    assert torch.equal(restored, source)
    assert packed.encode_ns >= 0
    assert decode_ns >= 0


def test_stage_tensor_rejects_unsupported_dtype_and_malformed_metadata() -> None:
    with pytest.raises(ValueError, match="unsupported tensor dtype"):
        pack_tensor(torch.tensor([True]), requested_mode="none")

    packed = pack_tensor(torch.arange(6, dtype=torch.float32), requested_mode="none")
    metadata = packed.attributes()
    with pytest.raises(ValueError, match="dimensions must be integers"):
        unpack_tensor(packed.payload, {**metadata, "shape": ["6"]})
    with pytest.raises(ValueError, match="encoded byte length"):
        unpack_tensor(
            packed.payload,
            {**metadata, "encoded_bytes": packed.encoded_bytes + 1},
        )
    with pytest.raises(ValueError, match="checksum"):
        unpack_tensor(packed.payload, {**metadata, "raw_checksum": "0" * 64})
    with pytest.raises(ValueError, match="byte length"):
        unpack_tensor(packed.payload, {**metadata, "raw_bytes": packed.raw_bytes + 4})


def _cost(layer_id: int, weight_bytes: int = 100) -> LayerCost:
    return LayerCost(
        layer_id=layer_id,
        execution_ns=1_000,
        weight_bytes=weight_bytes,
        kv_bytes_per_token=8,
        peak_temporary_bytes=16,
        activation_bytes=8,
        measured=True,
    )


def test_stage_plan_enforces_endpoint_memory_and_converts_stage_definition(
    tmp_path: Any,
) -> None:
    metadata = ModelPartitionMetadata(
        layer_costs=(_cost(0), _cost(1)),
        embedding_weight_bytes=50,
        final_weight_bytes=50,
        dtype_bytes=2,
        hidden_size=4,
        model_revision="revision",
        tokenizer_revision="tokenizer",
        metadata_hash="hash",
    )
    with pytest.raises(MemoryError):
        build_stage_plan(
            tmp_path,
            metadata=metadata,
            stage_count=2,
            method="balanced",
            memory_limit_bytes=120,
        )

    assignment = StageAssignment(
        stage_id=0,
        layer_start=0,
        layer_end=2,
        layer_ids=(0, 1),
        weight_bytes=200,
        estimated_compute_ns=2_000_000,
        measured_compute_ns=2_000_000,
        kv_cache_bytes_per_token=16,
        peak_temporary_bytes=32,
        activation_bytes=8,
        device="cpu",
        owns_embeddings=True,
        owns_final_norm=True,
        owns_output_projection=True,
    )
    spec = TensorSpec(dtype="bfloat16", shape=[1, "sequence", 4])
    definition = assignment.to_stage_definition(
        input_spec=spec,
        output_spec=spec,
        tensor_names=("weight",),
    )
    assert definition.owns_output_head
    assert definition.cache_spec.bytes_per_token == 16
    restored = stage_assignment_from_definition(
        definition,
        device="cpu",
        measured_compute_ns=assignment.measured_compute_ns,
        peak_temporary_bytes=assignment.peak_temporary_bytes,
        activation_bytes=assignment.activation_bytes,
    )
    assert replace(restored, measured_compute_ns=assignment.measured_compute_ns) == assignment


def _bare_stage() -> ContiguousOlmoeStage:
    stage = ContiguousOlmoeStage.__new__(ContiguousOlmoeStage)
    torch.nn.Module.__init__(stage)
    stage.sessions = {}
    stage._closed = False
    stage._ownership = WeightOwnership(
        stage_id=1,
        layer_start=2,
        layer_end=4,
        parameter_names=("model.layers.2.weight", "model.layers.3.weight"),
        parameter_bytes=32,
        parameter_count=8,
        owns_embeddings=False,
        owns_final_norm=True,
        owns_output_projection=True,
        ownership_hash="hash",
    )
    return stage


def _cache(elements: int) -> Any:
    layer = SimpleNamespace(
        keys=torch.zeros(elements, dtype=torch.float32),
        values=torch.zeros(elements, dtype=torch.float32),
    )
    return SimpleNamespace(layers=[layer])


def test_olmoe_stage_ownership_and_session_kv_release_are_isolated() -> None:
    stage = _bare_stage()
    assert isinstance(stage, StageExecutor)
    assert stage.ownership.layer_start == 2
    stage.sessions["first"] = StageSessionState(cache=cast(Any, _cache(2)))
    stage.sessions["second"] = StageSessionState(cache=cast(Any, _cache(3)))
    assert stage.kv_cache_bytes("first") == 16
    assert stage.kv_cache_bytes("second") == 24
    assert stage.close_session("first") == 16
    assert "first" not in stage.sessions
    assert stage.kv_cache_bytes("second") == 24
    assert stage.cancel_session("second") == 24
    assert not stage.sessions


def test_olmoe_stage_close_releases_every_session() -> None:
    stage = _bare_stage()
    stage.sessions["first"] = StageSessionState(cache=cast(Any, _cache(1)))
    stage.sessions["second"] = StageSessionState(cache=cast(Any, _cache(1)))
    stage.close()
    assert not stage.sessions
    with pytest.raises(RuntimeError, match="closed"):
        stage.open_session("new")
