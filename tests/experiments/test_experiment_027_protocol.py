from __future__ import annotations

import numpy as np

from swarm_inference.experiments.experiment_027.protocol import (
    Flags,
    RESPONSE,
    ResponseKind,
    StageResponse,
    WIRE_MAGIC,
    final_output,
    hidden_output,
)


def _response(*, kind: ResponseKind, payload: bytes, flags: Flags = Flags(0)) -> StageResponse:
    return StageResponse(
        request_id=1,
        kind=kind,
        flags=flags,
        n_tokens=2,
        n_embd=3,
        n_vocab=5,
        top_k=2,
        stage_start=0,
        stage_end=1,
        payload=payload,
        compute_ns=1,
        total_ns=2,
        deserialize_ns=0,
        serialize_ns=0,
        round_trip_ns=3,
        client_serialize_ns=0,
    )


def test_wire_layout_matches_native_contract() -> None:
    assert WIRE_MAGIC == int.from_bytes(b"E027", "little")
    assert RESPONSE.size == 88


def test_hidden_payload_round_trip() -> None:
    hidden = np.arange(6, dtype="<f4").reshape(2, 3)
    parsed = hidden_output(_response(kind=ResponseKind.HIDDEN, payload=hidden.tobytes()))
    np.testing.assert_array_equal(parsed, hidden)


def test_final_payload_round_trip() -> None:
    ids = np.array([[4, 3], [2, 1]], dtype="<i4")
    scores = np.array([[2.0, 1.0], [3.0, 0.5]], dtype="<f4")
    nextn = np.arange(6, dtype="<f4").reshape(2, 3)
    full = np.arange(10, dtype="<f4").reshape(2, 5)
    payload = ids.tobytes() + scores.tobytes() + nextn.tobytes() + full.tobytes()
    parsed = final_output(
        _response(
            kind=ResponseKind.FINAL,
            payload=payload,
            flags=Flags.RETURN_NEXTN | Flags.RETURN_FULL_LOGITS,
        )
    )
    np.testing.assert_array_equal(parsed.top_ids, ids)
    np.testing.assert_array_equal(parsed.top_logits, scores)
    np.testing.assert_array_equal(parsed.nextn, nextn)
    np.testing.assert_array_equal(parsed.full_logits, full)
