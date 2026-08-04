from __future__ import annotations

import socket
import threading
from dataclasses import replace

import pytest

from swarm_inference.experiments.experiment_011.protocol import (
    HEADER,
    MAGIC,
    BufferPool,
    MessageSequenceValidator,
    Operation,
    SessionValidator,
    StageMessage,
    decode_message,
    encode_message,
    recv_message,
    send_message,
)


def _message(*, operation: Operation = Operation.DECODE, sequence: int = 0) -> StageMessage:
    return StageMessage(
        operation=operation,
        model_revision="revision",
        tokenizer_revision="tokenizer",
        topology_id="topology",
        stage_id=1,
        layer_start=4,
        layer_end=8,
        session_id="session",
        request_id="request",
        sequence_number=sequence,
        token_position=2,
        source_stage=0,
        destination_stage=1,
        tensor_shape=(1, 1, 4),
        tensor_dtype="float32",
        compression_mode="none",
        payload=b"0123456789abcdef",
        attributes={"cache_position_start": 9},
    )


@pytest.mark.parametrize("operation", list(Operation))
def test_binary_protocol_round_trip_all_operations(operation: Operation) -> None:
    message = _message(operation=operation)
    encoded = encode_message(message)
    assert encoded.frame.startswith(MAGIC)
    assert encoded.wire_bytes == HEADER.size + encoded.metadata_bytes + encoded.payload_bytes
    assert decode_message(encoded.frame) == message


class _PartialSendSocket:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.buffer = bytearray()

    def send(self, data: bytes | memoryview) -> int:
        count = min(len(data), self.maximum)
        self.buffer.extend(bytes(data[:count]))
        return count


class _PartialReceiveSocket:
    def __init__(self, payload: bytes, maximum: int) -> None:
        self.payload = memoryview(payload)
        self.maximum = maximum
        self.offset = 0

    def recv_into(self, target: memoryview, nbytes: int = 0) -> int:
        if self.offset >= len(self.payload):
            return 0
        count = min(nbytes, self.maximum, len(self.payload) - self.offset)
        target[:count] = self.payload[self.offset : self.offset + count]
        self.offset += count
        return count


def test_partial_socket_writes_and_reads() -> None:
    sender = _PartialSendSocket(maximum=3)
    message = _message()
    send_message(sender, message)
    receiver = _PartialReceiveSocket(bytes(sender.buffer), maximum=2)
    assert recv_message(receiver) == message


def test_checksum_validation_rejects_corruption() -> None:
    frame = bytearray(encode_message(_message()).frame)
    frame[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        decode_message(frame)


def test_sequence_validation_rejects_duplicate_stale_and_gap() -> None:
    validator = MessageSequenceValidator()
    validator.validate(_message(sequence=4))
    with pytest.raises(ValueError, match="duplicate"):
        validator.validate(_message(sequence=4))
    with pytest.raises(ValueError, match="stale"):
        validator.validate(_message(sequence=3))
    with pytest.raises(ValueError, match="out-of-order"):
        validator.validate(_message(sequence=6))


def test_session_and_model_validation() -> None:
    validator = SessionValidator(model_revision="revision", topology_id="topology")
    validator.open("session")
    validator.validate(_message())
    wrong = replace(_message(), model_revision="wrong")
    with pytest.raises(ValueError, match="wrong model"):
        validator.validate(wrong)
    validator.close("session")
    with pytest.raises(ValueError, match="closed session"):
        validator.validate(_message())


def test_buffer_pool_reuses_bounded_buffers() -> None:
    pool = BufferPool(capacity=1, initial_size=128)
    first = pool.acquire(64)
    pool.release(first)
    second = pool.acquire(256)
    assert first is second
    assert len(second) >= 256
    pool.release(second)
    assert pool.allocations == 1
    assert pool.reuses == 2


def test_persistent_connection_reused_for_multiple_frames() -> None:
    left, right = socket.socketpair()

    def sender() -> None:
        send_message(left, _message(sequence=0))
        send_message(left, _message(sequence=1))
        left.close()

    thread = threading.Thread(target=sender)
    thread.start()
    pool = BufferPool(capacity=1, initial_size=128)
    first = recv_message(right, pool=pool)
    second = recv_message(right, pool=pool)
    right.close()
    thread.join(timeout=2)
    assert (first.sequence_number, second.sequence_number) == (0, 1)
    assert pool.allocations == 1
