from __future__ import annotations

import base64
import time

import pytest

from swarm_inference.microworker_protocol import (
    NETWORK_PROFILES,
    aggregate_digest,
    combine_argmax_aggregates,
    make_delegated_request,
    validate_operation_envelope,
)


def _candidate(worker_id: str, token_id: int, score: float, start: int) -> dict[str, object]:
    return {
        "mode": "vocabulary_argmax",
        "score": score,
        "score_float32_hex": "3f800000",
        "token_id": token_id,
        "winner_worker_id": worker_id,
        "contribution_count": 1,
        "local_logit_count": 10,
        "shards": [
            {
                "worker_id": worker_id,
                "token_start": start,
                "token_end": start + 10,
            }
        ],
    }


def test_vocabulary_argmax_is_order_independent_and_breaks_ties_by_token_id() -> None:
    low_token = _candidate("worker-b", 11, 2.0, 10)
    high_token = _candidate("worker-a", 25, 2.0, 20)

    forward = combine_argmax_aggregates([low_token, high_token])
    reverse = combine_argmax_aggregates([high_token, low_token])

    assert forward == reverse
    assert forward["token_id"] == 11
    assert [item["token_start"] for item in forward["shards"]] == [10, 20]
    assert aggregate_digest(forward) == aggregate_digest(reverse)


def test_vocabulary_argmax_rejects_duplicate_worker_shards() -> None:
    candidate = _candidate("worker-a", 2, 1.0, 0)

    with pytest.raises(ValueError, match="duplicate worker shard"):
        combine_argmax_aggregates([candidate, candidate])


def test_real_model_envelope_preserves_actual_hidden_payload() -> None:
    payload = bytes(range(32))
    request = make_delegated_request(
        request_id="request-1",
        operation_id="operation-1",
        execution_generation=1,
        parent_worker="stage-owner",
        worker_id="worker-000000",
        worker_index=0,
        assigned_child_workers=[],
        partition_start=0,
        partition_end=1,
        partition_worker_indices=[0],
        subtree_worker_count=1,
        deadline_unix_ns=time.time_ns() + 10_000_000_000,
        route_lease_id="lease-1",
        ordering_key="worker-000000",
        payload_bytes=len(payload),
        payload_b64=base64.b64encode(payload).decode("ascii"),
        trace_id="trace-1",
        parent_span_id="root",
        span_id="span-1",
        network_profile=NETWORK_PROFILES["same_host_shaped"],
        dispatch_policy="parallel",
        connection_policy="persistent",
        aggregation={
            "mode": "vocabulary_argmax",
            "ordering": "maximum_score_then_lowest_token_id",
        },
        workload={
            "kind": "real_model_lm_head",
            "model_id": "model",
            "model_revision": "revision",
            "tensor_name": "lm_head.weight",
            "hidden_size": 8,
            "vocabulary_size": 80,
            "hidden_dtype": "float32-le",
        },
    )

    validate_operation_envelope(request, allow_children=True)
    assert base64.b64decode(request["payload_b64"], validate=True) == payload
