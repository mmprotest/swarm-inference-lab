from __future__ import annotations

import numpy as np

from swarm_inference.config.models import OperationKind
from swarm_inference.protocol.messages import (
    ActivationMetadata,
    ActivationRequest,
    parse_message,
    serialize_message,
)


def test_protobuf_envelope_preserves_arbitrary_binary() -> None:
    payload = bytes(range(256)) * 4
    request = ActivationRequest(
        metadata=ActivationMetadata(
            request_id="r",
            tensor_id="t",
            stage_id=0,
            operation=OperationKind.PREFILL,
            token_position=0,
            sequence_length=4,
            cache_generation=0,
            model_id="synthetic",
            model_revision="v1",
        ),
        tensor_payload=payload,
    )
    decoded = parse_message(serialize_message(request), ActivationRequest)
    assert decoded.tensor_payload == payload
    assert decoded.metadata.operation == OperationKind.PREFILL
    assert np.frombuffer(decoded.tensor_payload, dtype=np.uint8).sum() > 0
