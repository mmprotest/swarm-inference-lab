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
    plan = runner.invoke(app, ["model", "plan", "--help"], terminal_width=200)
    submit = runner.invoke(app, ["submit", "--help"])
    worker = runner.invoke(app, ["worker", "--help"], terminal_width=200)

    assert root.exit_code == 0, root.output
    assert model.exit_code == 0, model.output
    assert plan.exit_code == 0, plan.output
    assert submit.exit_code == 0, submit.output
    assert worker.exit_code == 0, worker.output
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
    for option in ("--expert-policy", "--require-remo"):
        assert option in plan.output
    for option in (
        "--roles",
        "--expert-manif",
        "--expert-data-",
        "--expert-resid",
        "--expert-cache",
    ):
        assert option in worker.output


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


def test_product_model_plan_forwards_forced_expert_policy(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class JsonValue:
        plan_id = "forced-plan"
        report: Any

        def __init__(self) -> None:
            self.report = self

        def model_dump_json(self, *, indent: int | None = None) -> str:
            return json.dumps({"forced": True}, indent=indent)

    class FakeCoordinatorClient:
        def __init__(self, endpoint: str) -> None:
            assert endpoint == "127.0.0.1:50051"

        async def plan_model(self, request: Any) -> Any:
            captured["request"] = request
            return type("Response", (), {"plan": JsonValue()})()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        "swarm_inference.coordinator.service.CoordinatorClient",
        FakeCoordinatorClient,
    )
    output = tmp_path / "forced-plan.json"
    result = CliRunner().invoke(
        app,
        [
            "model",
            "plan",
            "--coordinator",
            "127.0.0.1:50051",
            "--model-id",
            "test/olmoe",
            "--revision",
            "exact-commit",
            "--expert-policy",
            "microshard-remote",
            "--require-remote-experts",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert request.expert_policy == "microshard-remote"
    assert request.require_remote_experts is True
    assert request.allow_expert_local_fallback is False
    assert output.is_file()


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


def test_product_status_prints_remote_expert_capacity_and_usage(monkeypatch) -> None:
    class FakeCoordinatorClient:
        def __init__(self, _endpoint: str) -> None:
            pass

        async def status(self) -> Any:
            return type(
                "Status",
                (),
                {
                    "coordinator_identity": "coordinator",
                    "coordinator_public_key_fingerprint": "sha256:key",
                    "uptime_s": 1.0,
                    "healthy_worker_count": 3,
                    "registered_worker_count": 3,
                    "active_topology_id": "topology",
                    "route_generation": 2,
                    "active_session_count": 0,
                    "generated_tokens": 4,
                    "throughput_tokens_s": 2.0,
                    "recovery_count": 0,
                    "recovering_requests": 0,
                    "expert_worker_count": 2,
                    "owned_experts": 1,
                    "owned_microshards": 2,
                    "expert_cache_resident_bytes": 4096,
                    "expert_cache_hits": 3,
                    "expert_cache_misses": 1,
                    "remote_expert_calls": 5,
                    "remote_microshard_calls": 7,
                    "expert_reduction_modes": ["fixed_order_fp32"],
                    "expert_fallbacks": 0,
                    "expert_bytes_transferred": 8192,
                    "expert_critical_path_ns": 1234,
                    "last_error": None,
                },
            )()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        "swarm_inference.coordinator.service.CoordinatorClient",
        FakeCoordinatorClient,
    )
    result = CliRunner().invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    for value in (
        "expert_workers=2",
        "owned_experts=1",
        "owned_microshards=2",
        "cache_hits=3",
        "cache_misses=1",
        "remote_expert_calls=5",
        "remote_microshard_calls=7",
        "reduction_modes=fixed_order_fp32",
        "fallbacks=0",
        "expert_bytes_transferred=8192",
        "expert_critical_path_ns=1234",
    ):
        assert value in result.output
