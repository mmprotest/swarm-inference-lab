"""Windows 11 user service, firewall, hardware, and path adapter."""

from __future__ import annotations

import ctypes
import platform
import subprocess
from collections.abc import Callable, Mapping, Sequence
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
    default_command_runner,
)


def _is_windows_administrator() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (AttributeError, OSError):
        return False


def _powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class WindowsPlatformAdapter(BasePlatformAdapter):
    service_mode = "windows-task"

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        home_directory: Path | None = None,
        command_runner: CommandRunner = default_command_runner,
        administrator_probe: Callable[[], bool] = _is_windows_administrator,
    ) -> None:
        super().__init__(
            environment=environment,
            home_directory=home_directory,
            command_runner=command_runner,
        )
        self.administrator_probe = administrator_probe

    def identity(self) -> PlatformIdentity:
        architecture = platform.machine() or "unknown"
        supported = architecture.lower() in {"amd64", "x86_64"}
        return PlatformIdentity(
            system="windows",
            release=platform.release(),
            architecture=architecture,
            support_status="validated" if supported else "unsupported",
            support_reason=(
                "Windows x86-64 CPU product path"
                if supported
                else f"Windows architecture {architecture} is not supported"
            ),
        )

    def _local_app_data(self) -> Path:
        value = self.environment.get("LOCALAPPDATA")
        return Path(value) if value else self.home_directory / "AppData" / "Local"

    def state_directory(self) -> Path:
        return self._local_app_data() / "SwarmInference"

    def cache_directory(self) -> Path:
        return self._local_app_data() / "SwarmInference" / "cache"

    def log_directory(self) -> Path:
        return self._local_app_data() / "SwarmInference" / "logs"

    def _backend_candidates(self) -> Sequence[tuple[Backend, str]]:
        return (
            (Backend.TORCH_CUDA, "cuda"),
            (Backend.TORCH_CPU, "cpu"),
        )

    def _task_path(self) -> str:
        return "\\SwarmInference\\"

    def _task_name(self, definition: ServiceDefinition) -> str:
        return definition.service_name

    def service_install_commands(self, definition: ServiceDefinition) -> list[CommandSpec]:
        arguments = subprocess.list2cmdline(definition.arguments)
        working = definition.working_directory or definition.executable.parent
        script = "; ".join(
            (
                f"$action=New-ScheduledTaskAction -Execute {_powershell_literal(str(definition.executable))} -Argument {_powershell_literal(arguments)} -WorkingDirectory {_powershell_literal(str(working))}",
                "$trigger=New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME",
                "$settings=New-ScheduledTaskSettingsSet "
                f"-RestartCount {definition.restart_limit} "
                f"-RestartInterval (New-TimeSpan -Seconds {definition.restart_delay_seconds}) "
                "-StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero)",
                f"Register-ScheduledTask -TaskPath {_powershell_literal(self._task_path())} "
                f"-TaskName {_powershell_literal(self._task_name(definition))} "
                "-Action $action -Trigger $trigger -Settings $settings "
                "-RunLevel Limited -User $env:USERNAME -Force | Out-Null",
                f"Start-ScheduledTask -TaskPath {_powershell_literal(self._task_path())} "
                f"-TaskName {_powershell_literal(self._task_name(definition))}",
            )
        )
        return [
            CommandSpec(
                executable="powershell.exe",
                arguments=["-NoProfile", "-NonInteractive", "-Command", script],
                description="install and start current-user Task Scheduler agent",
            )
        ]

    def service_uninstall_commands(self, definition: ServiceDefinition) -> list[CommandSpec]:
        full_name = self._task_path() + self._task_name(definition)
        return [
            CommandSpec(
                executable="schtasks.exe",
                arguments=["/Delete", "/TN", full_name, "/F"],
                description="remove owned current-user scheduled task",
            )
        ]

    def service_start_commands(self, definition: ServiceDefinition) -> list[CommandSpec]:
        return [
            CommandSpec(
                executable="schtasks.exe",
                arguments=[
                    "/Run",
                    "/TN",
                    self._task_path() + self._task_name(definition),
                ],
                description="start current-user scheduled task",
            )
        ]

    def service_stop_commands(self, definition: ServiceDefinition) -> list[CommandSpec]:
        return [
            CommandSpec(
                executable="schtasks.exe",
                arguments=[
                    "/End",
                    "/TN",
                    self._task_path() + self._task_name(definition),
                ],
                description="stop current-user scheduled task",
            )
        ]

    def service_status_command(self, definition: ServiceDefinition) -> CommandSpec:
        return CommandSpec(
            executable="schtasks.exe",
            arguments=[
                "/Query",
                "/TN",
                self._task_path() + self._task_name(definition),
                "/V",
                "/FO",
                "LIST",
            ],
            description="query current-user scheduled task",
        )

    def _firewall_script(self, specification: FirewallRuleSpec) -> str:
        label = specification.owner_label
        subnets = ",".join(specification.private_subnets or ["LocalSubnet"])
        statements: list[str] = []
        for kind, ports in (
            ("control", specification.control_ports),
            ("data", specification.data_ports),
        ):
            if not ports:
                continue
            name = f"{label}-{kind}"
            port_text = ",".join(str(port) for port in sorted(set(ports)))
            statements.append(
                f"if (-not (Get-NetFirewallRule -DisplayName {_powershell_literal(name)} "
                "-ErrorAction SilentlyContinue)) { "
                f"New-NetFirewallRule -DisplayName {_powershell_literal(name)} "
                "-Direction Inbound -Action Allow -Protocol TCP "
                f"-LocalPort {_powershell_literal(port_text)} -Profile Private "
                f"-RemoteAddress {_powershell_literal(subnets)} | Out-Null }}"
            )
        return "; ".join(statements)

    def _firewall_remediation(self, specification: FirewallRuleSpec) -> str:
        script = self._firewall_script(specification)
        return f'powershell.exe -NoProfile -Command "{script.replace(chr(34), chr(96) + chr(34))}"'

    def configure_firewall(self, specification: FirewallRuleSpec) -> FirewallStatus:
        if not self.administrator_probe():
            return FirewallStatus(
                owner_label=specification.owner_label,
                configured=False,
                private_only=True,
                blocked=True,
                detail="Windows firewall configuration requires an elevated PowerShell",
                remediation_command=self._firewall_remediation(specification),
            )
        command = CommandSpec(
            executable="powershell.exe",
            arguments=[
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                self._firewall_script(specification),
            ],
            description="apply owned private-profile LocalSubnet firewall rules",
        )
        result = self.command_runner(command, self.safe_environment())
        return FirewallStatus(
            owner_label=specification.owner_label,
            configured=result.succeeded,
            private_only=True,
            blocked=not result.succeeded,
            detail=(
                "owned private-subnet firewall rules are configured"
                if result.succeeded
                else result.stderr.strip() or "firewall rule creation failed"
            ),
            remediation_command=(
                None if result.succeeded else self._firewall_remediation(specification)
            ),
        )

    def firewall_status(self, specification: FirewallRuleSpec) -> FirewallStatus:
        label = specification.owner_label
        script = (
            f"$owned=Get-NetFirewallRule -DisplayName {_powershell_literal(label + '-*')} "
            "-ErrorAction SilentlyContinue; "
            "if ($owned) { 'OWNED=1' } else { 'OWNED=0' }; "
            "$broad=Get-NetFirewallRule -Enabled True -Direction Inbound -Action Allow "
            "-ErrorAction SilentlyContinue | Where-Object {$_.Profile -match 'Any|Public'}; "
            "if ($broad) { $broad | ForEach-Object { 'BROAD=' + $_.DisplayName } }"
        )
        result = self.command_runner(
            CommandSpec(
                executable="powershell.exe",
                arguments=["-NoProfile", "-NonInteractive", "-Command", script],
                description="inspect owned and broader Windows firewall rules",
            ),
            self.safe_environment(),
        )
        broader = [
            line.removeprefix("BROAD=").strip()
            for line in result.stdout.splitlines()
            if line.startswith("BROAD=")
        ]
        configured = result.succeeded and "OWNED=1" in result.stdout
        return FirewallStatus(
            owner_label=label,
            configured=configured,
            private_only=not broader,
            broader_existing_rules=broader,
            blocked=False,
            detail=(
                "owned firewall rules found" if configured else "owned firewall rules not found"
            ),
        )

    def remove_firewall(self, specification: FirewallRuleSpec) -> FirewallStatus:
        if not self.administrator_probe():
            label = specification.owner_label
            command = (
                'powershell.exe -NoProfile -Command "Get-NetFirewallRule '
                f"-DisplayName '{label}-*' | Remove-NetFirewallRule\""
            )
            return FirewallStatus(
                owner_label=label,
                configured=True,
                private_only=True,
                blocked=True,
                detail="removing owned firewall rules requires an elevated PowerShell",
                remediation_command=command,
            )
        script = (
            f"Get-NetFirewallRule -DisplayName {_powershell_literal(specification.owner_label + '-*')} "
            "-ErrorAction SilentlyContinue | Remove-NetFirewallRule"
        )
        result = self.command_runner(
            CommandSpec(
                executable="powershell.exe",
                arguments=["-NoProfile", "-NonInteractive", "-Command", script],
                description="remove only owned Windows firewall rules",
            ),
            self.safe_environment(),
        )
        return FirewallStatus(
            owner_label=specification.owner_label,
            configured=not result.succeeded,
            private_only=True,
            blocked=not result.succeeded,
            detail=(
                "owned firewall rules removed"
                if result.succeeded
                else result.stderr.strip() or "firewall rule removal failed"
            ),
        )


__all__ = ["WindowsPlatformAdapter"]
