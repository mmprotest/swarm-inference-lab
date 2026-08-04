"""Deprecated compatibility exports for the canonical stage-ring protocol.

New code must import :mod:`swarm_inference.protocol.stage_ring` directly.
"""

from swarm_inference.protocol.stage_ring import (
    CONTROL_OPERATIONS,
    HEADER,
    MAGIC,
    MAX_METADATA_BYTES,
    MAX_PAYLOAD_BYTES,
    PROTOCOL_VERSION,
    STAGE_RING_PROTOCOL_VERSION,
    BufferPool,
    EncodedFrame,
    MessageSequenceValidator,
    Operation,
    SequenceAllocator,
    SessionValidator,
    SocketLike,
    StageMessage,
    TelemetryCallback,
    decode_message,
    encode_message,
    message_wire_identity,
    recv_message,
    send_encoded_frame,
    send_message,
)

__all__ = [
    "CONTROL_OPERATIONS",
    "HEADER",
    "MAGIC",
    "MAX_METADATA_BYTES",
    "MAX_PAYLOAD_BYTES",
    "PROTOCOL_VERSION",
    "STAGE_RING_PROTOCOL_VERSION",
    "BufferPool",
    "EncodedFrame",
    "MessageSequenceValidator",
    "Operation",
    "SequenceAllocator",
    "SessionValidator",
    "SocketLike",
    "StageMessage",
    "TelemetryCallback",
    "decode_message",
    "encode_message",
    "message_wire_identity",
    "recv_message",
    "send_encoded_frame",
    "send_message",
]
