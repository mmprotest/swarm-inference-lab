from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from swarm_inference.cli import app
from swarm_inference.config.product import load_product_config


def test_product_cli_help_exposes_complete_command_path() -> None:
    runner = CliRunner()
    root = runner.invoke(app, ["--help"])
    model = runner.invoke(app, ["model", "--help"])
    submit = runner.invoke(app, ["submit", "--help"])

    assert root.exit_code == 0, root.output
    assert model.exit_code == 0, model.output
    assert submit.exit_code == 0, submit.output
    for command in (
        "coordinator",
        "worker",
        "model",
        "submit",
        "status",
        "workers",
        "topology",
        "sessions",
        "cancel",
    ):
        assert command in root.output
    for command in ("inspect", "plan", "deploy", "unload"):
        assert command in model.output
    for option in ("--model-id", "--model-revision", "--stream", "--json", "--ndjson"):
        assert option in submit.output


def test_product_model_plan_validates_partition_before_network_access() -> None:
    result = CliRunner().invoke(
        app,
        [
            "model",
            "plan",
            "--coordinator",
            "127.0.0.1:1",
            "--model-id",
            "test/olmoe",
            "--revision",
            "exact-commit",
            "--partition",
            "microshards",
        ],
    )

    assert result.exit_code == 1
    assert "auto, equal, or balanced" in result.output


def test_product_configuration_is_non_experiment_and_local_only_by_default() -> None:
    path = Path("configs/product/olmoe-stage-ring.yaml")
    configuration = load_product_config(path)

    assert configuration.kind == "product-stage-ring"
    assert configuration.default_adapter_id == "olmoe"
    assert configuration.local_only_by_default


def test_product_status_commands_emit_machine_readable_json(monkeypatch) -> None:
    class Response:
        accepted = True

        def __init__(self, command: str) -> None:
            self.command = command

        def model_dump_json(self, *, indent: int | None = None) -> str:
            return json.dumps({"command": self.command, "ok": True}, indent=indent)

    class FakeCoordinatorClient:
        def __init__(self, endpoint: str) -> None:
            assert endpoint == "127.0.0.1:50051"

        async def status(self) -> Response:
            return Response("status")

        async def workers(self, _request: Any) -> Response:
            return Response("workers")

        async def topology_status(self, _request: Any) -> Response:
            return Response("topology")

        async def sessions(self, _request: Any) -> Response:
            return Response("sessions")

        async def cancel_request(self, request_id: str) -> Response:
            assert request_id == "request-1"
            return Response("cancel")

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        "swarm_inference.coordinator.service.CoordinatorClient",
        FakeCoordinatorClient,
    )
    runner = CliRunner()
    invocations = {
        "status": ["status", "--json"],
        "workers": ["workers", "--json"],
        "topology": ["topology", "--json"],
        "sessions": ["sessions", "--json"],
        "cancel": [
            "cancel",
            "--coordinator",
            "127.0.0.1:50051",
            "--request-id",
            "request-1",
            "--json",
        ],
    }
    for command, arguments in invocations.items():
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {"command": command, "ok": True}
