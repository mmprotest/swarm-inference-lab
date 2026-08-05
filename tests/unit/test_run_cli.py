from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

from typer.testing import CliRunner

from swarm_inference.cli import app
from swarm_inference.protocol.messages import StreamEventType, SubmitStreamEvent

MODEL_REVISION = "a" * 40
TOKENIZER_REVISION = "sha256:" + "b" * 64


class _Report:
    def model_dump_json(self, *, indent: int | None = None) -> str:
        return json.dumps({"confidence": "measured"}, indent=indent)


class _Summary:
    run_id = "run-test"
    status = "completed"
    topology_id = "topology-test"
    output_token_ids: ClassVar[list[int]] = [7]
    plan = SimpleNamespace(topology_id="topology-test", report=_Report())

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": self.status,
            "topology_id": self.topology_id,
            "output_token_ids": self.output_token_ids,
            "decoded_text": "token",
            "plan": {"topology_id": "topology-test", "report": {"confidence": "measured"}},
        }


class _FakeOrchestrator:
    captured: ClassVar[dict[str, Any]] = {}

    def __init__(self, **kwargs: Any) -> None:
        self.progress_sink = kwargs["progress_sink"]
        self.stream_sink = kwargs["stream_sink"]

    async def run(self, **kwargs: Any) -> _Summary:
        type(self).captured = kwargs
        self.stream_sink(
            SubmitStreamEvent(
                event_type=StreamEventType.TOKEN_GENERATED,
                request_id="request-test",
                sequence_number=1,
                monotonic_timestamp_ns=time.monotonic_ns(),
                token_position=0,
                token_id=7,
                decoded_text_fragment="token",
                model_revision=MODEL_REVISION,
            )
        )
        return _Summary()


def _arguments(tmp_path: Path, *, prompt: str) -> list[str]:
    return [
        "run",
        "allenai/OLMoE-test",
        "--revision",
        MODEL_REVISION,
        "--tokenizer-revision",
        TOKENIZER_REVISION,
        "--prompt",
        prompt,
        "--state-root",
        str(tmp_path),
    ]


def test_run_json_is_final_only_and_never_echoes_prompt(monkeypatch, tmp_path: Path) -> None:
    secret_prompt = "sensitive prompt that must not be serialized"
    monkeypatch.setattr(
        "swarm_inference.commands.run.ClusterOrchestrator",
        _FakeOrchestrator,
    )
    result = CliRunner().invoke(app, [*_arguments(tmp_path, prompt=secret_prompt), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["output_token_ids"] == [7]
    assert "report" not in payload["plan"]
    assert secret_prompt not in result.output
    assert _FakeOrchestrator.captured["prompt"] == secret_prompt


def test_run_ndjson_has_bounded_public_stream_fields(monkeypatch, tmp_path: Path) -> None:
    secret_prompt = "another sensitive prompt"
    monkeypatch.setattr(
        "swarm_inference.commands.run.ClusterOrchestrator",
        _FakeOrchestrator,
    )
    result = CliRunner().invoke(app, [*_arguments(tmp_path, prompt=secret_prompt), "--ndjson"])

    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    assert lines[0]["event_type"] == "TOKEN_GENERATED"
    assert lines[-1]["run_id"] == "run-test"
    assert secret_prompt not in result.output


def test_run_requires_immutable_model_revision_before_orchestration(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "run",
            "allenai/OLMoE-test",
            "--revision",
            "main",
            "--tokenizer-revision",
            TOKENIZER_REVISION,
            "--prompt",
            "safe",
            "--state-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "immutable" in result.output
