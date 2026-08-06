from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from swarm_inference.platforms.base import (
    CommandResult,
    CommandSpec,
    FirewallRuleSpec,
    ServiceDefinition,
)
from swarm_inference.platforms.linux import LinuxPlatformAdapter
from swarm_inference.platforms.macos import MacOSPlatformAdapter
from swarm_inference.platforms.windows import WindowsPlatformAdapter


def _runner(command: CommandSpec, environment) -> CommandResult:
    del environment
    return CommandResult(command=command, exit_code=0, stdout="running active")


def _service(tmp_path: Path) -> ServiceDefinition:
    return ServiceDefinition(
        cluster_id="cluster1",
        node_id="node-12345678",
        executable=tmp_path / "swarm",
        arguments=["node", "agent", "--state", str(tmp_path / "state")],
        restart_limit=5,
        restart_delay_seconds=7,
    )


def test_windows_task_commands_are_user_scoped_and_restart_bounded(tmp_path: Path) -> None:
    adapter = WindowsPlatformAdapter(
        environment={"LOCALAPPDATA": str(tmp_path / "local")},
        home_directory=tmp_path,
        command_runner=_runner,
        administrator_probe=lambda: False,
    )
    command = adapter.service_install_commands(_service(tmp_path))[0]
    rendered = " ".join(command.arguments)
    assert "New-ScheduledTaskSettingsSet" in rendered
    assert "-RestartCount 5" in rendered
    assert "-RestartInterval" in rendered
    assert "-RunLevel Limited" in rendered

    firewall = adapter.configure_firewall(
        FirewallRuleSpec(
            cluster_id="cluster1",
            node_id="node-12345678",
            control_ports=[50051],
            data_ports=[50052],
            private_subnets=["192.168.0.0/16"],
        )
    )
    assert firewall.blocked
    assert firewall.private_only
    assert "-Profile Private" in firewall.remediation_command
    assert "192.168.0.0/16" in firewall.remediation_command


def test_linux_systemd_user_unit_has_restart_and_shutdown_bounds(tmp_path: Path) -> None:
    adapter = LinuxPlatformAdapter(
        environment={"PATH": "/usr/bin"},
        home_directory=tmp_path,
        command_runner=_runner,
    )
    commands = adapter.service_install_commands(_service(tmp_path))
    assert [item.arguments[:2] for item in commands] == [
        ["--user", "daemon-reload"],
        ["--user", "enable"],
    ]
    unit = next((tmp_path / ".config" / "systemd" / "user").glob("*.service"))
    content = unit.read_text(encoding="utf-8")
    assert "Restart=on-failure" in content
    assert "RestartSec=7" in content
    assert "StartLimitBurst=5" in content
    assert "TimeoutStopSec=15" in content


def test_macos_launch_agent_has_keepalive_and_throttle(tmp_path: Path) -> None:
    adapter = MacOSPlatformAdapter(
        environment={"PATH": "/usr/bin", "UID": "501"},
        home_directory=tmp_path,
        command_runner=_runner,
    )
    commands = adapter.service_install_commands(_service(tmp_path))
    assert commands[0].arguments[0] == "bootstrap"
    plist_path = next((tmp_path / "Library" / "LaunchAgents").glob("*.plist"))
    with plist_path.open("rb") as handle:
        payload = plistlib.load(handle)
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert payload["ThrottleInterval"] == 7
    assert payload["ProcessType"] == "Background"


def test_safe_service_environment_rejects_secret_persistence(tmp_path: Path) -> None:
    adapter = LinuxPlatformAdapter(home_directory=tmp_path, command_runner=_runner)
    with pytest.raises(ValueError, match="sensitive variable"):
        adapter.safe_environment({"PAIRING_SECRET": "do-not-persist"})
