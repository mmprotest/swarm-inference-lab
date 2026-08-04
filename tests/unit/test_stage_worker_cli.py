from __future__ import annotations

from typing import Any

from typer.testing import CliRunner

from swarm_inference.cli import app


def test_worker_cli_passes_independent_stage_runtime_endpoints(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_run_worker(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("swarm_inference.worker.service.run_worker", fake_run_worker)
    result = CliRunner().invoke(
        app,
        [
            "worker",
            "--coordinator",
            "127.0.0.1:50051",
            "--backend",
            "torch-cpu",
            "--memory-limit-gb",
            "1",
            "--listen",
            "127.0.0.1:50052",
            "--advertise",
            "worker.test:50052",
            "--stage-runtime",
            "--data-listen",
            "0.0.0.0:50053",
            "--data-advertise",
            "worker.test:50053",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--max-stage-sessions",
            "17",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["advertised_endpoint"] == "worker.test:50052"
    assert captured["data_listen_endpoint"] == "0.0.0.0:50053"
    assert captured["data_advertised_endpoint"] == "worker.test:50053"
    assert captured["stage_runtime_enabled"] is True
    assert captured["device"] == "cpu"
    assert captured["max_stage_sessions"] == 17


def test_worker_cli_rejects_wildcard_data_advertisement(monkeypatch) -> None:
    async def unexpected_run_worker(**_kwargs: Any) -> None:
        raise AssertionError("invalid CLI reached worker startup")

    monkeypatch.setattr("swarm_inference.worker.service.run_worker", unexpected_run_worker)
    result = CliRunner().invoke(
        app,
        [
            "worker",
            "--coordinator",
            "127.0.0.1:50051",
            "--backend",
            "torch-cpu",
            "--memory-limit-gb",
            "1",
            "--listen",
            "127.0.0.1:50052",
            "--advertise",
            "worker.test:50052",
            "--stage-runtime",
            "--data-listen",
            "0.0.0.0:50053",
            "--data-advertise",
            "0.0.0.0:50053",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 1
    assert "data-advertise" in result.output
    assert "wildcard" in result.output
