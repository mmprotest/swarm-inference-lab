"""Opt-in, architecture-neutral real-model acceptance evidence gates.

The heavyweight run is performed by the canonical Swarm CLI on the hardware
under test.  These tests retain and verify its immutable evidence without
making any model family a product acceptance dependency.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest


def _consume_executed_evidence(gate: str) -> dict[str, Any]:
    configured = os.environ.get("SWARM_REAL_MODEL_ACCEPTANCE_EVIDENCE")
    if not configured:
        pytest.skip("SWARM_REAL_MODEL_ACCEPTANCE_EVIDENCE is not configured")
    source = Path(configured).expanduser().resolve() / f"{gate}.json"
    if not source.is_file():
        pytest.skip(f"executed real-model evidence is unavailable: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload.get("document_type") == "swarm-real-model-gate-evidence"
    assert payload.get("gate") == gate
    assert payload.get("executed") is True
    assert payload.get("status") == "PASS"
    assert payload.get("generated_token_ids") == payload.get("expected_token_ids")
    assert payload.get("generated_token_ids")
    assert payload.get("model_id") and payload.get("model_revision")
    profile = payload.get("architecture_profile")
    assert isinstance(profile, dict) and profile.get("architecture_id")
    assert payload.get("engine_id")

    destination_root = os.environ.get("SWARM_ACCEPTANCE_GATE_EVIDENCE")
    if destination_root:
        destination = Path(destination_root).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / source.name)
    return payload


def test_real_model_baseline_evidence() -> None:
    payload = _consume_executed_evidence("real-model-baseline")
    assert not payload.get("recovery_events")


def test_real_model_restart_and_replay_evidence() -> None:
    payload = _consume_executed_evidence("real-model-restart-and-replay")
    assert payload.get("recovery_events")


def test_real_model_whole_expert_evidence() -> None:
    payload = _consume_executed_evidence("real-model-whole-expert")
    assert payload.get("fallback_count") == 0
    assert int(payload.get("remote_expert_calls", 0)) > 0


def test_real_model_native_microshard_evidence() -> None:
    payload = _consume_executed_evidence("real-model-native-microshard")
    assert payload.get("fallback_count") == 0
    assert int(payload.get("remote_microshard_calls", 0)) > 0
