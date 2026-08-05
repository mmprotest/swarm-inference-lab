"""macOS LaunchAgent, firewall, MPS, and path adapter."""

from __future__ import annotations

import os
import platform
import plistlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

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
)


class MacOSPlatformAdapter(BasePlatformAdapter):
    service_mode = "launch-agent"

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
        arm = architecture.lower() in {"arm64", "aarch64"}
        return PlatformIdentity(
            system="macos",
            release=platform.release(),
            architecture=architecture,
            support_status="implemented-unvalidated" if arm else "unsupported",
            support_reason=(
                "Apple Silicon MPS path is implemented and awaits physical acceptance"
                if arm
                else "the product MPS milestone supports Apple Silicon only"
            ),
        )

    def state_directory(self) -> Path:
        return self.home_directory / "Library" / "Application Support" / "SwarmInference"

    def cache_directory(self) -> Path:
        return self.home_directory / "Library" / "Caches" / "SwarmInference"

    def log_directory(self) -> Path:
        return self.home_directory / "Library" / "Logs" / "SwarmInference"

    def _backend_candidates(self) -> Sequence[tuple[Backend, str]]:
        return (
            (Backend.TORCH_MPS, "mps"),
            (Backend.TORCH_CPU, "cpu"),
        )

    def _label(self, definition: ServiceDefinition) -> str:
        return f"org.swarm-inference.{definition.cluster_id}.{definition.node_id}"

    def _plist_path(self, definition: ServiceDefinition) -> Path:
        return self.home_directory / "Library" / "LaunchAgents" / f"{self._label(definition)}.plist"

    def _write_plist(self, definition: ServiceDefinition) -> Path:
        path = self._plist_path(definition)
        path.parent.mkdir(parents=True, exist_ok=True)
        working = definition.working_directory or definition.executable.parent
        payload = {
            "Label": self._label(definition),
            "ProgramArguments": [str(definition.executable), *definition.arguments],
            "WorkingDirectory": str(working),
            "EnvironmentVariables": self.safe_environment(definition.environment),
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "ThrottleInterval": definition.restart_delay_seconds,
            "ProcessType": "Background",
            "StandardOutPath": str(self.service_log_location(definition)),
            "StandardErrorPath": str(self.service_log_location(definition)),
        }
        with path.open("wb") as handle:
            plistlib.dump(payload, handle, sort_keys=True)
        return path

    def _domain(self) -> str:
        fallback_uid = int(self.environment.get("UID", "0"))
        uid_getter = cast(Callable[[], int], getattr(os, "getuid", lambda: fallback_uid))
        return f"gui/{uid_getter()}"

    def service_install_commands(self, definition: ServiceDefinition) -> list[CommandSpec]:
        path = self._write_plist(definition)
        return [
            CommandSpec(
                executable="launchctl",
                arguments=["bootstrap", self._domain(), str(path)],
                description="install and start user LaunchAgent",
            )
        ]

    def service_uninstall_commands(self, definition: ServiceDefinition) -> list[CommandSpec]:
        return [
            CommandSpec(
                executable="launchctl",
                arguments=["bootout", self._domain(), str(self._plist_path(definition))],
                description="stop and remove user LaunchAgent",
            )
        ]

    def uninstall_service(self, definition: ServiceDefinition) -> ServiceStatus:
        status = super().uninstall_service(definition)
        path = self._plist_path(definition)
        if status.installed is False and path.exists():
            path.unlink()
        return status

    def service_start_commands(self, definition: ServiceDefinition) -> list[CommandSpec]:
        return [
            CommandSpec(
                executable="launchctl",
                arguments=["kickstart", "-k", f"{self._domain()}/{self._label(definition)}"],
                description="start user LaunchAgent",
            )
        ]

    def service_stop_commands(self, definition: ServiceDefinition) -> list[CommandSpec]:
        return [
            CommandSpec(
                executable="launchctl",
                arguments=["kill", "SIGTERM", f"{self._domain()}/{self._label(definition)}"],
                description="stop user LaunchAgent",
            )
        ]

    def service_status_command(self, definition: ServiceDefinition) -> CommandSpec:
        return CommandSpec(
            executable="launchctl",
            arguments=["print", f"{self._domain()}/{self._label(definition)}"],
            description="query user LaunchAgent",
        )

    def configure_firewall(self, specification: FirewallRuleSpec) -> FirewallStatus:
        ports = sorted({*specification.control_ports, *specification.data_ports})
        subnets = specification.private_subnets or ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]
        rules = " ".join(
            f"pass in proto tcp from {subnet} to any port {{{','.join(map(str, ports))}}}"
            for subnet in subnets
        )
        return FirewallStatus(
            owner_label=specification.owner_label,
            configured=False,
            private_only=True,
            blocked=True,
            detail="macOS packet-filter changes require explicit administrator approval",
            remediation_command=f"echo {rules!r} | sudo pfctl -a swarm-inference -f -",
        )

    def firewall_status(self, specification: FirewallRuleSpec) -> FirewallStatus:
        result = self.command_runner(
            CommandSpec(
                executable="pfctl",
                arguments=["-a", "swarm-inference", "-sr"],
                description="inspect owned macOS packet-filter anchor",
            ),
            self.safe_environment(),
        )
        return FirewallStatus(
            owner_label=specification.owner_label,
            configured=result.succeeded and "pass in" in result.stdout,
            private_only=result.succeeded,
            blocked=False,
            detail="owned packet-filter anchor inspected",
        )

    def remove_firewall(self, specification: FirewallRuleSpec) -> FirewallStatus:
        return FirewallStatus(
            owner_label=specification.owner_label,
            configured=True,
            private_only=True,
            blocked=True,
            detail="flush only the owned packet-filter anchor with administrator approval",
            remediation_command="sudo pfctl -a swarm-inference -F rules",
        )


__all__ = ["MacOSPlatformAdapter"]
