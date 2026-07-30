from __future__ import annotations

import numpy as np
import pytest

from swarm_inference.exceptions import IntegrityError
from swarm_inference.protocol.checksums import sha256_bytes
from swarm_inference.protocol.tensor_codec import (
    ActivationTensor,
    decode_tensor,
    encode_tensor,
    reassemble_chunks,
    split_chunks,
)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int64])
def test_tensor_round_trip(dtype) -> None:
    source = np.arange(24).reshape(2, 3, 4).astype(dtype)
    encoded = encode_tensor(
        ActivationTensor(
            tensor_id="tensor-1",
            request_id="request-1",
            stage_id=2,
            token_position=7,
            sequence_length=3,
            array=source,
        )
    )
    decoded = decode_tensor(encoded)
    assert decoded.tensor_id == "tensor-1"
    assert decoded.request_id == "request-1"
    assert decoded.stage_id == 2
    assert decoded.token_position == 7
    assert np.array_equal(decoded.array, source)


def test_bad_tensor_checksum_is_rejected() -> None:
    encoded = bytearray(
        encode_tensor(
            ActivationTensor(
                tensor_id="x",
                request_id="r",
                stage_id=0,
                token_position=0,
                sequence_length=1,
                array=np.ones((1, 1, 4), dtype=np.float32),
            )
        )
    )
    encoded[-1] ^= 1
    with pytest.raises(IntegrityError, match="checksum"):
        decode_tensor(bytes(encoded))


def test_chunking_round_trip_and_checksum() -> None:
    payload = bytes(range(256)) * 100
    chunks = split_chunks(payload, tensor_id="large", max_chunk_bytes=1024)
    assert len(chunks) > 1
    assert reassemble_chunks(list(reversed(chunks))) == payload
    assert sha256_bytes(payload) == sha256_bytes(reassemble_chunks(chunks))


def test_missing_chunk_is_rejected() -> None:
    chunks = split_chunks(b"x" * 5000, tensor_id="large", max_chunk_bytes=1000)
    with pytest.raises(IntegrityError, match="missing"):
        reassemble_chunks(chunks[:-1])
