from __future__ import annotations

from pathlib import Path

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
    for command in ("coordinator", "worker", "model", "submit", "topology", "workers"):
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
