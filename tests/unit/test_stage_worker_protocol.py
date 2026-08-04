from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from swarm_inference.model.partition import StageAssignment
from swarm_inference.protocol.messages import parse_message, serialize_message
from swarm_inference.protocol.stage_worker import (
    CancelStageSessionRequest,
    CloseStageSessionRequest,
    DrainWorkerRequest,
    GetStageCapabilitiesRequest,
    GetStageStatusRequest,
    InstallStageRouteRequest,
    LoadStageRequest,
    OpenStageSessionRequest,
    RemoveStageRouteRequest,
    StageRouteEndpoint,
    UnloadStageRequest,
)


def _assignment() -> StageAssignment:
    return StageAssignment(
        stage_id=0,
        layer_start=0,
        layer_end=1,
        layer_ids=(0,),
        weight_bytes=1,
        estimated_compute_ns=1,
        measured_compute_ns=None,
        kv_cache_bytes_per_token=1,
        peak_temporary_bytes=0,
        activation_bytes=1,
        device="cpu",
        owns_embeddings=True,
        owns_final_norm=True,
        owns_output_projection=True,
    )


def test_all_persistent_stage_control_requests_round_trip_through_any() -> None:
    common = {
        "worker_id": "worker",
        "request_id": "request",
        "model_id": "model",
        "model_revision": "revision",
        "tokenizer_revision": "tokenizer",
        "topology_id": "topology",
        "route_generation": 1,
        "device": "cpu",
        "dtype": "float32",
    }
    session = {**common, "stage_id": 0, "session_id": "session"}
    messages = [
        GetStageCapabilitiesRequest(worker_id="worker", request_id="capabilities"),
        LoadStageRequest(**common, stage_count=1, assignment=_assignment()),
        UnloadStageRequest(**common, stage_count=1, assignment=_assignment()),
        InstallStageRouteRequest(
            **common,
            assignment=_assignment(),
            previous_stage=None,
            next_stage=None,
            stage_count=1,
            lease_expiry_unix_ns=time.time_ns() + 1_000_000_000,
        ),
        RemoveStageRouteRequest(**common, stage_id=0),
        OpenStageSessionRequest(**session),
        CloseStageSessionRequest(**session),
        CancelStageSessionRequest(**session),
        GetStageStatusRequest(worker_id="worker", request_id="status"),
        DrainWorkerRequest(worker_id="worker", request_id="drain"),
    ]
    for message in messages:
        assert parse_message(serialize_message(message), type(message)) == message


def test_route_peer_can_carry_exact_adjacent_assignment() -> None:
    endpoint = StageRouteEndpoint(
        worker_id="worker-next",
        stage_id=0,
        data_endpoint="worker-next.test:50053",
        assignment=_assignment(),
    )
    assert parse_message(serialize_message(endpoint), StageRouteEndpoint) == endpoint


def test_stage_control_identities_reject_empty_strings() -> None:
    with pytest.raises(ValidationError, match="cannot be empty"):
        GetStageCapabilitiesRequest(worker_id=" ", request_id="request")
