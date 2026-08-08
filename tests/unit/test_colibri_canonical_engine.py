from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from swarm_inference.backends.colibri.backend import ColibriBackend
from swarm_inference.engines.colibri import LocalColibriLifecycle
from swarm_inference.engines.interfaces import (
    ExecutionPlan,
    InferenceEvent,
    InferenceRequest,
    PhasePlan,
)
from swarm_inference.engines.local_capabilities import (
    discover_configured_general_engine_capabilities,
)
from swarm_inference.worker.abi import TokenPayload, WorkerJobResult, WorkerJobStatus
from swarm_inference.worker.engine_factory import _colibri_factory


class _Tokenizer:
    def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
        assert text == "hello"
        return {"input_ids": [11, 12]}

    def decode(self, token_ids: list[int], **_: Any) -> str:
        assert token_ids == [21, 22]
        return "answer"


def test_colibri_glm_prompt_uses_the_immutable_local_tokenizer(tmp_path: Path) -> None:
    model = tmp_path / "model"
    engine = tmp_path / "engine"
    model.mkdir()
    engine.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "glm_moe_dsa"}),
        encoding="utf-8",
    )
    backend = ColibriBackend(
        engine_directory=engine,
        model_path=model,
        model_id="org/model",
        model_revision="a" * 40,
        model_family="glm-5.2",
        ram_safety_reserve_bytes=0,
    )
    backend._tokenizer = _Tokenizer()

    payload = backend.prompt_payload("hello", tokenizer_hash="sha256:" + "1" * 64)

    assert payload.token_ids == [11, 12]
    assert payload.tokenizer_hash == "sha256:" + "1" * 64
    assert backend.decode_tokens([21, 22]) == "answer"


def _plan() -> ExecutionPlan:
    roles = {"worker-a": "critical_path_stage"}
    return ExecutionPlan(
        plan_id="colibri-plan",
        engine_id="colibri",
        model_fingerprint="sha256:" + "1" * 64,
        execution_identity="sha256:" + "2" * 64,
        objective="speed",
        topology="colibri-complete-model",
        worker_roles=roles,
        prefill_plan=PhasePlan(phase="prefill", worker_roles=roles),
        decode_plan=PhasePlan(phase="decode", worker_roles=roles),
        predicted_ttft_ms=1,
        predicted_decode_tokens_s=1,
        predicted_aggregate_tokens_s=1,
        score=1,
        engine_parameters={
            "model_id": "org/model",
            "model_revision": "a" * 40,
            "tokenizer_identity": "sha256:" + "3" * 64,
        },
    )


class _Capabilities:
    def model_dump(self, **_: Any) -> dict[str, str]:
        return {"backend": "colibri"}


class _Backend:
    def capabilities(self) -> _Capabilities:
        return _Capabilities()

    def prompt_payload(self, prompt: str, *, tokenizer_hash: str | None) -> TokenPayload:
        assert prompt == "hello"
        assert tokenizer_hash == "sha256:" + "3" * 64
        return TokenPayload(token_ids=[11, 12], tokenizer_hash=tokenizer_hash)

    async def execute(self, job: Any) -> WorkerJobResult:
        assert job.input_payload.token_ids == [11, 12]
        return WorkerJobResult(
            job_id=job.job_id,
            request_id=job.request_id,
            status=WorkerJobStatus.ACCEPTED,
            output_payload=TokenPayload(token_ids=[21, 22]),
            metrics={"route_trace": "verified"},
        )

    def decode_tokens(self, token_ids: list[int]) -> str:
        assert token_ids == [21, 22]
        return "answer"

    async def shutdown(self) -> None:
        return None


@pytest.mark.asyncio
async def test_colibri_lifecycle_accepts_a_normal_text_prompt() -> None:
    lifecycle = LocalColibriLifecycle(lambda _plan: _Backend())  # type: ignore[arg-type]
    deployment = await lifecycle.prepare(_plan())
    stream: AsyncIterator[InferenceEvent] = lifecycle.submit(
        deployment,
        InferenceRequest(request_id="request-a", prompt="hello", max_new_tokens=2),
    )

    events = [event async for event in stream]

    assert [event.event_type for event in events] == [
        "started",
        "token",
        "token",
        "completed",
    ]
    assert events[-1].text == "answer"
    assert events[-1].telemetry == {"route_trace": "verified"}
    await lifecycle.unload(deployment)


def _write_routing_runtime(tmp_path: Path) -> tuple[Path, Path, str]:
    engine = tmp_path / "engine"
    model = tmp_path / "model"
    profiles = tmp_path / "profiles"
    engine.mkdir()
    model.mkdir()
    profiles.mkdir()
    binary = engine / "colibri.exe"
    binary.write_bytes(b"pinned-colibri-binary")
    config = model / "config.json"
    config.write_text(json.dumps({"model_type": "glm_moe_dsa"}), encoding="utf-8")
    bitmap = profiles / "glm-hot.bin"
    bitmap.write_bytes(b"\x01\x00\x01\x00")
    fingerprint = "sha256:" + "4" * 64
    manifest = tmp_path / "runtime-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "runtime_revision": "colibri-pinned",
                "binary_hashes": {"colibri": hashlib.sha256(binary.read_bytes()).hexdigest()},
                "model_families": ["glm-5.2"],
                "formats": ["safetensors"],
                "engine_directory": "engine",
                "routing_profiles": [
                    {
                        "profile_id": "glm-hot-v1",
                        "adapter_id": "glm-5.2",
                        "model_fingerprint": fingerprint,
                        "hot_pin_path": "profiles/glm-hot.bin",
                        "hot_pin_sha256": hashlib.sha256(bitmap.read_bytes()).hexdigest(),
                        "settings": {
                            "PILOT": "1",
                            "WIDE": "2",
                            "PILOT_EVICT_GUARD": "1",
                        },
                        "exactness_passed": True,
                        "measured_utility": 0.0504,
                        "evidence_fingerprint": "experiment-009-reverse-confirmed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, config, fingerprint


def test_verified_routing_profile_is_advertised_and_changes_backend_environment(
    tmp_path: Path,
) -> None:
    manifest, config, fingerprint = _write_routing_runtime(tmp_path)
    capabilities = discover_configured_general_engine_capabilities(colibri_manifest=manifest)
    capability = next(item for item in capabilities if item.engine_id == "colibri")
    assert capability.enabled
    assert capability.fast_paths == ("routing-aware-placement",)
    assert [item.profile_id for item in capability.execution_profiles] == ["glm-hot-v1"]

    plan = _plan().model_copy(
        update={
            "model_fingerprint": fingerprint,
            "optional_mechanisms": {"routing_aware_placement": True},
            "engine_parameters": {
                "model_id": "org/model",
                "model_revision": "a" * 40,
                "model_family": "glm-5.2",
                "model_paths": [str(config)],
                "routing_profile_id": "glm-hot-v1",
            },
        }
    )
    backend = _colibri_factory(manifest)(plan)

    assert backend.environment == {
        "PILOT": "1",
        "WIDE": "2",
        "PILOT_EVICT_GUARD": "1",
        "COLI_HOT_PIN_PATH": str((tmp_path / "profiles" / "glm-hot.bin").resolve()),
    }
    assert backend.execution_profile_id == "glm-hot-v1"

    baseline = _colibri_factory(manifest)(
        plan.model_copy(
            update={
                "optional_mechanisms": {"routing_aware_placement": False},
                "engine_parameters": {
                    key: value
                    for key, value in plan.engine_parameters.items()
                    if key != "routing_profile_id"
                },
            }
        )
    )
    assert baseline.environment == {}
    assert baseline.execution_profile_id is None


def test_colibri_routing_profile_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest, _config, _fingerprint = _write_routing_runtime(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["routing_profiles"][0]["hot_pin_sha256"] = "0" * 64
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    capability = next(
        item
        for item in discover_configured_general_engine_capabilities(colibri_manifest=manifest)
        if item.engine_id == "colibri"
    )
    assert not capability.enabled
    assert "hash" in capability.detail
