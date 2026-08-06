from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from swarm_inference.platforms.base import (
    CommandResult,
    CommandSpec,
    FirewallRuleSpec,
    owned_firewall_resource_name,
)
from swarm_inference.platforms.linux import LinuxPlatformAdapter
from swarm_inference.platforms.macos import MacOSPlatformAdapter
from swarm_inference.platforms.windows import WindowsPlatformAdapter


def _spec(cluster: str = "cluster-a", *, control: int = 51001) -> FirewallRuleSpec:
    return FirewallRuleSpec(
        cluster_id=cluster,
        node_id="node-12345678",
        control_ports=[control],
        data_ports=[51002],
        private_subnets=["192.168.0.0/16"],
    )


def test_owned_resource_names_are_bounded_safe_deterministic_and_collision_resistant() -> None:
    labels = [f"SwarmInference-cluster-{index}-node-12345678" for index in range(512)]
    for platform_name, pattern, maximum in (
        ("linux", r"swarm_[0-9a-f]{20}", 32),
        ("macos", r"swarm-inference/[0-9a-f]{20}", 64),
        ("windows", r"SwarmInference-[0-9a-f]{20}", 64),
    ):
        names = [
            owned_firewall_resource_name(label, platform_name=platform_name)  # type: ignore[arg-type]
            for label in labels
        ]
        assert len(names) == len(set(names))
        assert all(len(name) <= maximum and re.fullmatch(pattern, name) for name in names)
        assert names[0] == owned_firewall_resource_name(
            labels[0],
            platform_name=platform_name,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("platform_name", ["linux", "macos", "windows"])
def test_two_clusters_have_isolated_removal_targets(platform_name: str, tmp_path: Path) -> None:
    first, second = _spec("cluster-a"), _spec("cluster-b")
    if platform_name == "linux":
        adapter = LinuxPlatformAdapter(home_directory=tmp_path)
    elif platform_name == "macos":
        adapter = MacOSPlatformAdapter(home_directory=tmp_path)
    else:
        adapter = WindowsPlatformAdapter(
            home_directory=tmp_path,
            administrator_probe=lambda: False,
        )
    first_name = first.resource_name(platform_name)  # type: ignore[arg-type]
    second_name = second.resource_name(platform_name)  # type: ignore[arg-type]
    removal = adapter.remove_firewall(first)
    assert first_name != second_name
    assert first_name in (removal.remediation_command or "")
    assert second_name not in (removal.remediation_command or "")


def test_linux_owned_status_reconciliation_and_broader_rule_reporting(tmp_path: Path) -> None:
    first, other = _spec(), _spec("cluster-b")
    calls: list[CommandSpec] = []

    def runner(command: CommandSpec, _environment: Mapping[str, str]) -> CommandResult:
        calls.append(command)
        if command.arguments[:4] == ["list", "table", "inet", first.resource_name("linux")]:
            stdout = (
                f"table inet {first.resource_name('linux')} {{ "
                "ip saddr 192.168.0.0/16 tcp dport 51001 accept; "
                "ip saddr 192.168.0.0/16 tcp dport 51002 accept; }"
            )
        else:
            stdout = "tcp dport 22 accept # unrelated broad operator rule"
        return CommandResult(command=command, exit_code=0, stdout=stdout)

    adapter = LinuxPlatformAdapter(home_directory=tmp_path, command_runner=runner)
    status = adapter.firewall_status(first)
    assert status.configured and status.private_only
    assert status.broader_existing_rules == ["tcp dport 22 accept # unrelated broad operator rule"]
    assert calls[0].arguments == ["list", "table", "inet", first.resource_name("linux")]
    assert other.resource_name("linux") not in " ".join(calls[0].argv)

    initial = adapter.configure_firewall(first).remediation_command
    repeated = adapter.configure_firewall(first).remediation_command
    changed = adapter.configure_firewall(_spec(control=52001)).remediation_command
    assert initial == repeated
    assert changed != initial
    assert first.resource_name("linux") in (changed or "")
    assert "192.168.0.0/16" in (changed or "")
    assert "0.0.0.0/0" not in (changed or "")


def test_macos_owned_status_reconciliation_and_broader_rule_reporting(tmp_path: Path) -> None:
    first, other = _spec(), _spec("cluster-b")
    calls: list[CommandSpec] = []

    def runner(command: CommandSpec, _environment: Mapping[str, str]) -> CommandResult:
        calls.append(command)
        if command.arguments[:2] == ["-a", first.resource_name("macos")]:
            stdout = (
                "pass in proto tcp from 192.168.0.0/16 to any port 51001\n"
                "pass in proto tcp from 192.168.0.0/16 to any port 51002\n"
            )
        else:
            stdout = "pass in proto tcp from any to any port 22\n"
        return CommandResult(command=command, exit_code=0, stdout=stdout)

    adapter = MacOSPlatformAdapter(home_directory=tmp_path, command_runner=runner)
    status = adapter.firewall_status(first)
    assert status.configured and status.private_only
    assert status.broader_existing_rules == ["pass in proto tcp from any to any port 22"]
    assert calls[0].arguments == ["-a", first.resource_name("macos"), "-sr"]
    assert other.resource_name("macos") not in " ".join(calls[0].argv)

    initial = adapter.configure_firewall(first).remediation_command
    repeated = adapter.configure_firewall(first).remediation_command
    changed = adapter.configure_firewall(_spec(control=52001)).remediation_command
    assert initial == repeated
    assert changed != initial
    assert first.resource_name("macos") in (changed or "")
    assert "192.168.0.0/16" in (changed or "")
    assert " from any " not in (changed or "")


def test_windows_reconciles_only_owned_rules_and_reports_unrelated_broad_rules(
    tmp_path: Path,
) -> None:
    first, other = _spec(), _spec("cluster-b")
    calls: list[CommandSpec] = []

    def runner(command: CommandSpec, _environment: Mapping[str, str]) -> CommandResult:
        calls.append(command)
        if command.description.startswith("inspect"):
            stdout = "OWNED=1\nBROAD=Operator-Public-SSH\n"
        else:
            stdout = ""
        return CommandResult(command=command, exit_code=0, stdout=stdout)

    adapter = WindowsPlatformAdapter(
        home_directory=tmp_path,
        command_runner=runner,
        administrator_probe=lambda: True,
    )
    configured = adapter.configure_firewall(first)
    assert configured.configured and configured.private_only
    script = calls[-1].arguments[-1]
    assert first.resource_name("windows") in script
    assert other.resource_name("windows") not in script
    assert "Remove-NetFirewallRule" in script
    assert script.index("Remove-NetFirewallRule") < script.index("New-NetFirewallRule")
    assert "-Profile Private" in script
    assert "192.168.0.0/16" in script

    status = adapter.firewall_status(first)
    assert status.configured and status.private_only
    assert status.broader_existing_rules == ["Operator-Public-SSH"]
    inspection = calls[-1].arguments[-1]
    assert first.resource_name("windows") in inspection
    assert other.resource_name("windows") not in inspection

    repeated = adapter._firewall_script(first)
    changed = adapter._firewall_script(_spec(control=52001))
    assert repeated == adapter._firewall_script(first)
    assert changed != repeated
    removal = adapter.remove_firewall(first)
    removal_script = calls[-1].arguments[-1]
    assert removal.configured is False
    assert first.resource_name("windows") in removal_script
    assert "Operator-Public-SSH" not in removal_script


@pytest.mark.parametrize(
    "mutation",
    [
        {"cluster_id": "cluster-a; Remove-Item C:/", "private_subnets": ["192.168.0.0/16"]},
        {"cluster_id": "cluster-a", "private_subnets": ["192.168.0.0/16; nft flush ruleset"]},
        {"cluster_id": "cluster-a", "private_subnets": ["0.0.0.0/0"]},
    ],
)
def test_firewall_spec_rejects_label_subnet_and_scope_injection(
    mutation: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "cluster_id": "cluster-a",
        "node_id": "node-12345678",
        "control_ports": [51001],
        "data_ports": [51002],
        "private_subnets": ["192.168.0.0/16"],
    }
    values.update(mutation)
    with pytest.raises(ValueError):
        FirewallRuleSpec.model_validate(values)
