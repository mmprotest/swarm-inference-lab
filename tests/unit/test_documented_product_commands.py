from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from swarm_inference.acceptance.productization import _parser
from swarm_inference.cli import app
from swarm_inference.config.product import load_product_config


def _help(*arguments: str) -> str:
    result = CliRunner().invoke(app, [*arguments, "--help"], terminal_width=320)
    assert result.exit_code == 0, result.output
    return result.output


def test_documented_product_command_tree_and_options_are_real() -> None:
    root = _help()
    identity = _help("identity")
    create = _help("identity", "create")
    trust = _help("identity", "trust")
    coordinator = _help("coordinator")
    worker = _help("worker")
    model = _help("model")
    inspect = _help("model", "inspect")
    plan = _help("model", "plan")
    deploy = _help("model", "deploy")
    unload = _help("model", "unload")
    submit = _help("submit")

    for command in (
        "identity",
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
        assert command in root
    for command in ("create", "show", "fingerprint", "trust", "untrust", "list-trusted"):
        assert command in identity
    for command in ("inspect", "plan", "deploy", "unload"):
        assert command in model

    for option in ("--path", "--kind", "--force", "--json"):
        assert option in create
    for option in ("--coordinator-state", "--fingerprint", "--identity", "--label"):
        assert option in trust
    for option in ("--config", "--state", "--listen", "--advertise"):
        assert option in coordinator
    for option in (
        "--coordinator",
        "--worker-id",
        "--identity",
        "--listen",
        "--advertise",
        "--stage-runtime",
        "--data-listen",
        "--data-adverti",
        "--model-snapsh",
        "--trusted-coor",
    ):
        assert option in worker
    for help_text in (inspect, plan):
        for option in ("--coordinator", "--model-id", "--revision", "--tokenizer-re"):
            assert option in help_text
    for option in ("--stage-count", "--partition", "--require-dist", "--output"):
        assert option in plan
    assert "--plan" in deploy
    assert "--topology-id" in unload
    for option in (
        "--coordinator",
        "--model-id",
        "--model-revision",
        "--prompt",
        "--max-new-tokens",
        "--stream",
        "--ndjson",
    ):
        assert option in submit


def test_recommended_product_configuration_is_secure_and_bootstrappable() -> None:
    config = load_product_config(Path("configs/product/olmoe-stage-ring.yaml"))
    assert config.require_trusted_workers is True
    assert config.trust_store_path == Path(".swarm/coordinator/trusted-workers.json")
    assert config.trusted_worker_fingerprints == []
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "identity create" in readme
    assert "identity trust" in readme
    assert "require_trusted_workers: true" in readme


def test_primary_documentation_states_current_boundaries_without_overclaiming() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    architecture = Path("docs/architecture.md").read_text(encoding="utf-8")
    security = Path("docs/security-boundary.md").read_text(encoding="utf-8")
    recovery = Path("docs/recovery.md").read_text(encoding="utf-8")
    physical = Path("docs/physical-two-machine-acceptance.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.lower().split())
    normalized_physical = " ".join(physical.split())

    assert "Direct stage-ring data plane" in readme
    assert "absent from steady-state hidden-state forwarding" in normalized_readme
    assert "session interleaving" in normalized_readme
    assert "not continuous tensor batching" in normalized_readme
    assert "restart-and-replay" in readme
    assert "not seamless failover" in normalized_readme
    assert "Experiment 011 is the latest completed experiment" in readme
    assert "Physical multi-machine product performance has not been proven" in readme
    assert "coordinator is not on the steady-state hidden-state forwarding path" in architecture
    assert "Neither mechanism supplies payload confidentiality" in security
    assert "There is no KV checkpoint transfer" in recovery
    assert "can never be used as physical evidence" in normalized_physical


def test_acceptance_script_documented_options_exist() -> None:
    parser = _parser()
    help_text = parser.format_help()
    assert "machine-identity" in help_text
    assert "run" in help_text
    run_parser = next(action for action in parser._actions if action.dest == "command").choices[
        "run"
    ]
    run_help = run_parser.format_help()
    for option in ("--output", "--real-model", "--physical-config"):
        assert option in run_help
