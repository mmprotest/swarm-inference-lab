from __future__ import annotations

from pathlib import Path

from typer.main import get_command
from typer.testing import CliRunner

from swarm_inference.acceptance.productization import _parser
from swarm_inference.cli import app
from swarm_inference.config.product import load_product_config


def _help(*arguments: str) -> str:
    result = CliRunner().invoke(app, [*arguments, "--help"], terminal_width=320)
    assert result.exit_code == 0, result.output
    return result.output


def _options(*arguments: str) -> set[str]:
    command = get_command(app)
    for argument in arguments:
        command = command.commands[argument]
    return {option for parameter in command.params for option in getattr(parameter, "opts", ())}


def test_documented_product_command_tree_and_options_are_real() -> None:
    root = _help()
    cluster = _help("cluster")
    node = _help("node")
    _help("run")

    for command in (
        "cluster",
        "node",
        "run",
        "coordinator",
        "worker",
        "identity",
        "model",
        "submit",
        "status",
        "workers",
        "topology",
        "sessions",
        "cancel",
    ):
        assert command in root
    for command in ("create", "pair", "status", "nodes", "revoke", "start", "stop", "delete"):
        assert command in cluster
    for command in (
        "join",
        "status",
        "configure",
        "doctor",
        "leave",
        "install-service",
        "uninstall-service",
        "update",
    ):
        assert command in node
    for option in (
        "--revision",
        "--tokenizer-revision",
        "--prompt",
        "--mode",
        "--engine",
        "--quant",
        "--dry-run",
        "--explain-plan",
        "--require-distributed",
        "--require-node",
        "--exclude-node",
        "--json",
        "--ndjson",
        "--yes",
    ):
        assert option in _options("run")


def test_low_level_product_commands_remain_available() -> None:
    identity = _help("identity")
    model = _help("model")
    _help("coordinator")
    _help("worker")
    _help("submit")

    for command in ("create", "show", "fingerprint", "trust", "untrust", "list-trusted"):
        assert command in identity
    for command in ("inspect", "plan", "deploy", "unload"):
        assert command in model
    for option in ("--config", "--state", "--listen", "--advertise"):
        assert option in _options("coordinator")
    for option in (
        "--coordinator",
        "--worker-id",
        "--identity",
        "--listen",
        "--advertise",
        "--stage-runtime",
        "--data-listen",
        "--data-advertise",
        "--model-snapshot",
        "--trusted-coordinator-fingerprint",
    ):
        assert option in _options("worker")
    for option in (
        "--coordinator",
        "--model-id",
        "--model-revision",
        "--prompt",
        "--max-new-tokens",
        "--stream",
        "--ndjson",
    ):
        assert option in _options("submit")


def test_normal_quick_start_has_no_manual_provisioning() -> None:
    config = load_product_config(Path("configs/product/universal-stage-ring.yaml"))
    assert config.require_trusted_workers is True
    assert config.trust_store_path == Path(".swarm/coordinator/trusted-workers.json")
    assert config.trusted_worker_fingerprints == []

    readme = Path("README.md").read_text(encoding="utf-8")
    quick_start = readme.split("## Cluster quick start", maxsplit=1)[1].split(
        "## Product architecture", maxsplit=1
    )[0]
    for command in ("swarm cluster create", "swarm node join", "swarm cluster status", "swarm run"):
        assert command in quick_start
    for manual_command in (
        "swarm identity",
        "swarm coordinator",
        "swarm worker",
        "swarm model inspect",
        "swarm model plan",
        "swarm model deploy",
        "swarm submit",
    ):
        assert manual_command not in quick_start


def test_primary_documentation_states_current_boundaries_without_overclaiming() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    architecture = Path("docs/architecture.md").read_text(encoding="utf-8")
    security = Path("docs/security-boundary.md").read_text(encoding="utf-8")
    recovery = Path("docs/recovery.md").read_text(encoding="utf-8")
    physical = Path("docs/physical-two-machine-acceptance.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.lower().split())
    normalized_physical = " ".join(physical.lower().split())
    normalized_security = " ".join(security.lower().split())

    assert "global wan" in normalized_readme
    assert "wan control and data channels use tls 1.3" in normalized_readme
    assert "plaintext remote peers" in normalized_readme
    assert "coordinator does not relay hidden states" in normalized_readme
    assert "restart-and-replay" in normalized_readme
    assert "physical multi-machine performance validation remains ongoing" in normalized_readme
    assert "qwen3 moe" in normalized_readme
    assert "llama.cpp rpc" in normalized_readme
    assert "coordinator is not on the steady-state hidden-state forwarding path" in architecture
    assert "canonical non-loopback control and data transports use tls 1.3" in normalized_security
    assert "verification of computation returned by a malicious worker" in normalized_security
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
    for option in (
        "--output",
        "--real-model",
        "--physical-config",
        "--linux-x86-physical-config",
        "--macos-arm64-physical-config",
        "--linux-arm64-physical-config",
    ):
        assert option in run_help
