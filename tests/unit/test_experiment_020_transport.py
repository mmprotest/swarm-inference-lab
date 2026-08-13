from __future__ import annotations

import pytest

from swarm_inference.experiments.experiment_020.control_plane import (
    SwarmHostAgent,
    logical_topology,
)
from swarm_inference.experiments.experiment_020.transport import (
    Frame,
    MessageType,
    ProtocolError,
    decode_frame,
    encode_frame,
)


def test_binary_frame_roundtrip_and_identity_fields() -> None:
    key = b"k" * 32
    expected = Frame(
        MessageType.EXECUTE_SHARD,
        "request-7",
        4,
        "pod-001.worker-03",
        "state-9",
        b"payload",
    )
    assert decode_frame(encode_frame(expected, key), key) == expected


def test_checksum_or_authentication_corruption_is_rejected() -> None:
    key = b"k" * 32
    encoded = bytearray(
        encode_frame(Frame(MessageType.HEALTH, "r", 0, "w", "s", b"ok"), key)
    )
    encoded[-1] ^= 1
    with pytest.raises(ProtocolError):
        decode_frame(bytes(encoded), key)


def test_wrong_per_run_credential_is_rejected() -> None:
    encoded = encode_frame(
        Frame(MessageType.HEALTH, "r", 0, "w", "s"), b"a" * 32
    )
    with pytest.raises(ProtocolError, match="authentication"):
        decode_frame(encoded, b"b" * 32)


def test_host_agent_manages_eight_explicit_workers_but_is_not_compute() -> None:
    workers = logical_topology(96)[:8]
    host = SwarmHostAgent("pod-000", workers)
    host.start()
    assert not host.is_compute_resource
    assert len(host.running) == 8
    assert all(host.health().values())
    host.stop()
    assert not host.running
