from __future__ import annotations

from swarm_inference.execution import olmoe_stage
from swarm_inference.experiments.experiment_011 import (
    compression as legacy_compression,
)
from swarm_inference.experiments.experiment_011 import model as legacy_model
from swarm_inference.experiments.experiment_011 import partition as legacy_partition
from swarm_inference.experiments.experiment_011 import protocol as legacy_protocol
from swarm_inference.experiments.experiment_011 import telemetry as legacy_telemetry
from swarm_inference.experiments.experiment_011 import (
    tensor_transport as legacy_tensor_transport,
)
from swarm_inference.model import olmoe, partition
from swarm_inference.protocol import stage_ring
from swarm_inference.runtime import telemetry
from swarm_inference.transport import compression, stage_tensor


def test_experiment_011_compatibility_paths_are_canonical_objects() -> None:
    assert legacy_protocol.StageMessage is stage_ring.StageMessage
    assert legacy_protocol.encode_message is stage_ring.encode_message
    assert legacy_tensor_transport.PackedTensor is stage_tensor.PackedTensor
    assert legacy_tensor_transport.pack_tensor is stage_tensor.pack_tensor
    assert legacy_compression.CompressionResult is compression.CompressionResult
    assert legacy_compression.compress_lossless is compression.compress_lossless
    assert legacy_partition.StagePlan is partition.StagePlan
    assert (
        legacy_partition.inspect_model_partition_metadata is olmoe.inspect_olmoe_partition_metadata
    )
    assert legacy_model.ContiguousOlmoeStage is olmoe_stage.ContiguousOlmoeStage
    assert legacy_telemetry.TraceWriter is telemetry.TraceWriter
