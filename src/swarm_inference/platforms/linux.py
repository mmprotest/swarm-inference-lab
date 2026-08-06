"""Linux user service, firewall, hardware, and path adapter."""

from __future__ import annotations

import platform
import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

from swarm_inference.config.models import Backend
from swarm_inference.platforms.base import (
    BasePlatformAdapter,
    CommandRunner,
    CommandSpec,
    FirewallRuleSpec,
    FirewallStatus,
    PlatformIdentity,
    ServiceDefinition,
    ServiceStatus,
    default_command_runner,
    platform_implementation,
)


class LinuxPlatformAdapter(BasePlatformAdapter):
    service_mode = "systemd-user"

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        home_directory: Path | None = None,
        command_runner: CommandRunner = default_command_runner,
    ) -> None:
        super().__init__(
            environment=environment,
            home_directory=home_directory,
            command_runner=command_runner,
        )

    def identity(self) -> PlatformIdentity:
        architecture = platform.machine() or "unknown"
        implementation_status, implementation_reason = platform_implementation(
            "linux", architecture
        )
        return PlatformIdentity(
            system="linux",
            release=platform.release(),
            architecture=architecture,
            implementation_status=implementation_status,
            implementation_reason=implementation_reason,
        )

    def state_directory(self) -> Path:
        base = self.environment.get("XDG_STATE_HOME")
        return (Path(base) if base else self.home_directory / ".local" / "state") / (
            "swarm-inference"
        )

    def cache_directory(self) -> Path:
        base = self.environment.get("XDG_CACHE_HOME")
        return (Path(base) if base else self.home_directory / ".cache") / "swarm-inference"

    def log_directory(self) -> Path:
        return self.state_directory() / "logs"

    def _backend_candidates(self) -> Sequence[tuple[Backend, str]]:
        return (
            (Backend.TORCH_CUDA, "cuda"),
            (Backend.TORCH_CPU, "cpu"),
        )

    def _unit_path(self, definition: ServiceDefinition) -> Path:
        return (
            self.home_directory
            / ".config"
            / "systemd"
            / "user"
            / (f"{definition.service_name}.service")
        )

    def _write_unit(self, definition: ServiceDefinition) -> Path:
        unit_path = self._unit_path(definition)
        unit_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [str(definition.executable), *definition.arguments]
        environment_lines = [
            f"Environment={shlex.quote(f'{key}={value}')}"
            for key, value in sorted(self.safe_environment(definition.environment).items())
        ]
        working = definition.working_directory or definition.executable.parent
        content = "\n".join(
            [
                "[Unit]",
                "Description=Swarm Inference node agent",
                "After=network-online.target",
                "Wants=network-online.target",
                "",
                "[Service]",
                "Type=simple",
                f"ExecStart={shlex.join(argv)}",
                f"WorkingDirectory={shlex.quote(str(working))}",
                *environment_lines,
                "Restart=on-failure",
                f"RestartSec={definition.restart_delay_seconds}",
                f"StartLimitBurst={definition.restart_limit}",
                "StartLimitIntervalSec=300",
                "TimeoutStopSec=15",
                "KillMode=mixed",
                "NoNewPrivileges=true",
                "",
                "[Install]",
                "WantedBy=default.target",
                "",
            ]
        )
        unit_path.write_text(content, encoding="utf-8", newline="\n")
        return unit_path

    def service_install_commands(self, definition: ServiceDefinition) -> list[CommandSpec]:
        self._write_unit(definition)
        return [
            CommandSpec(
                executable="systemctl",
                arguments=["--user", "daemon-reload"],
                description="reload systemd user units",
            ),
            CommandSpec(
                executable="systemctl",
                arguments=["--user", "enable", "--now", f"{definition.service_name}.service"],
                description="enable and start systemd user agent",
            ),
        ]

    def service_uninstall_commands(self, definition: ServiceDefinition) -> list[CommandSpec]:
        return [
            CommandSpec(
                executable="systemctl",
                arguments=[
                    "--user",
                    "disable",
                    "--now",
                    f"{definition.service_name}.service",
                ],
                description="disable and stop systemd user agent",
            ),
            CommandSpec(
                executable="systemctl",
                arguments=["--user", "daemon-reload"],
                description="reload systemd user units",
            ),
        ]

    def uninstall_service(self, definition: ServiceDefinition) -> ServiceStatus:
        status = super().uninstall_service(definition)
        path = self._unit_path(definition)
        if status.installed is False and path.exists():
            path.unlink()
        return status

    def service_start_commands(self, definition: ServiceDefinition) -> list[CommandSpec]:
        return [
            CommandSpec(
                executable="systemctl",
                arguments=["--user", "start", f"{definition.service_name}.service"],
                description="start systemd user agent",
            )
        ]

    def service_stop_commands(self, definition: ServiceDefinition) -> list[CommandSpec]:
        return [
            CommandSpec(
                executable="systemctl",
                arguments=["--user", "stop", f"{definition.service_name}.service"],
                description="stop systemd user agent",
            )
        ]

    def service_status_command(self, definition: ServiceDefinition) -> CommandSpec:
        return CommandSpec(
            executable="systemctl",
            arguments=[
                "--user",
                "status",
                f"{definition.service_name}.service",
                "--no-pager",
            ],
            description="query systemd user agent",
        )

    def service_log_location(self, definition: ServiceDefinition) -> Path:
        return Path(f"journalctl --user-unit {definition.service_name}.service")

    def _firewall_remediation(self, specification: FirewallRuleSpec) -> str:
        ports = sorted({*specification.control_ports, *specification.data_ports})
        table = specification.resource_name("linux")
        commands = [
            shlex.join(["sudo", "nft", "delete", "table", "inet", table]),
            shlex.join(["sudo", "nft", "add", "table", "inet", table]),
            (
                shlex.join(["sudo", "nft", "add", "chain", "inet", table, "input"])
                + " "
                + shlex.quote("{ type filter hook input priority 0; policy accept; }")
            ),
        ]
        for subnet in specification.private_subnets:
            for port in ports:
                commands.append(
                    shlex.join(
                        [
                            "sudo",
                            "nft",
                            "add",
                            "rule",
                            "inet",
                            table,
                            "input",
                            "ip",
                            "saddr",
                            subnet,
                            "tcp",
                            "dport",
                            str(port),
                            "counter",
                            "accept",
                            "comment",
                            table,
                        ]
                    )
                )
        return "; ".join(commands)

    def configure_firewall(self, specification: FirewallRuleSpec) -> FirewallStatus:
        resource = specification.resource_name("linux")
        return FirewallStatus(
            owner_label=specification.owner_label,
            resource_name=resource,
            configured=False,
            private_only=True,
            blocked=True,
            detail="Linux user services cannot alter nftables without administrator approval",
            remediation_command=self._firewall_remediation(specification),
        )

    def firewall_status(self, specification: FirewallRuleSpec) -> FirewallStatus:
        resource = specification.resource_name("linux")
        result = self.command_runner(
            CommandSpec(
                executable="nft",
                arguments=["list", "table", "inet", resource],
                description="inspect owned nftables table",
            ),
            self.safe_environment(),
        )
        ports = sorted({*specification.control_ports, *specification.data_ports})
        configured = (
            result.succeeded
            and resource in result.stdout
            and all(
                re.search(rf"(?<![0-9.]){re.escape(subnet)}(?![0-9.])", result.stdout)
                for subnet in specification.private_subnets
            )
            and all(re.search(rf"(?<!\d){port}(?!\d)", result.stdout) for port in ports)
        )
        broad_result = self.command_runner(
            CommandSpec(
                executable="nft",
                arguments=["list", "ruleset"],
                description="report broader unrelated nftables allow rules",
            ),
            self.safe_environment(),
        )
        broader = [
            line.strip()
            for line in broad_result.stdout.splitlines()
            if "accept" in line
            and ("0.0.0.0/0" in line or ("tcp dport" in line and "ip saddr" not in line))
        ]
        return FirewallStatus(
            owner_label=specification.owner_label,
            resource_name=resource,
            configured=configured,
            private_only=configured,
            broader_existing_rules=broader,
            blocked=False,
            detail="owned nftables rule found" if configured else "owned nftables rule not found",
        )

    def remove_firewall(self, specification: FirewallRuleSpec) -> FirewallStatus:
        resource = specification.resource_name("linux")
        return FirewallStatus(
            owner_label=specification.owner_label,
            resource_name=resource,
            configured=True,
            private_only=True,
            blocked=True,
            detail="remove only the owned nftables table with administrator approval",
            remediation_command=shlex.join(["sudo", "nft", "delete", "table", "inet", resource]),
        )


__all__ = ["LinuxPlatformAdapter"]
