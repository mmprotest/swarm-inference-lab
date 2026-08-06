"""Injectable host boundary for service, firewall, hardware, and network operations."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import socket
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol

import psutil
from pydantic import Field, NonNegativeInt, PositiveInt, field_validator, model_validator

from swarm_inference.config.models import Backend, StrictModel

ImplementationStatus = Literal["implemented", "unsupported"]
ValidationStatus = Literal["validated", "failed", "not-run"]
_SERVICE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_PRIVATE_IPV4_NETWORKS = tuple(
    ipaddress.IPv4Network(value) for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
PLATFORM_IMPLEMENTATION_CONTRACT: dict[str, frozenset[str]] = {
    "windows": frozenset({"amd64", "x86_64"}),
    "linux": frozenset({"amd64", "x86_64", "aarch64", "arm64"}),
    "macos": frozenset({"aarch64", "arm64"}),
}


def platform_implementation(
    system: Literal["windows", "linux", "macos"],
    architecture: str,
) -> tuple[ImplementationStatus, str]:
    normalized = architecture.lower()
    if normalized in PLATFORM_IMPLEMENTATION_CONTRACT[system]:
        return (
            "implemented",
            f"{system} {architecture} product path is implemented; validation requires retained evidence",
        )
    return "unsupported", f"{system} architecture {architecture} is not implemented"


def owned_firewall_resource_name(
    owner_label: str,
    *,
    platform_name: Literal["windows", "linux", "macos"],
) -> str:
    """Return a bounded, injection-safe, collision-resistant owned resource name."""

    digest = hashlib.sha256(owner_label.encode("utf-8")).hexdigest()[:20]
    if platform_name == "windows":
        return f"SwarmInference-{digest}"
    if platform_name == "linux":
        return f"swarm_{digest}"
    return f"swarm-inference/{digest}"


class PlatformIdentity(StrictModel):
    system: Literal["windows", "linux", "macos"]
    release: str
    architecture: str
    implementation_status: ImplementationStatus
    implementation_reason: str

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_support_fields(cls, value: object) -> object:
        if isinstance(value, dict) and "support_status" in value:
            migrated = dict(value)
            legacy = str(migrated.pop("support_status"))
            reason = str(migrated.pop("support_reason", "legacy platform support state"))
            migrated["implementation_status"] = (
                "unsupported" if legacy == "unsupported" else "implemented"
            )
            migrated["implementation_reason"] = (
                f"{reason}; legacy status did not provide validation evidence"
            )
            return migrated
        return value


class BackendProbeResult(StrictModel):
    backend: Backend
    device: str
    detected: bool
    operational: bool
    reason: str
    device_name: str | None = None
    total_memory_bytes: NonNegativeInt = 0
    available_memory_bytes: NonNegativeInt = 0
    supported_dtypes: list[str] = Field(default_factory=list)
    probe_version: str = "torch-tensor-v1"


class InterfaceAddress(StrictModel):
    interface: str
    address: str
    prefix_length: NonNegativeInt | None = None
    is_private: bool
    is_loopback: bool
    is_up: bool
    mtu: PositiveInt | None = None


class ServiceDefinition(StrictModel):
    cluster_id: str
    node_id: str
    executable: Path
    arguments: list[str]
    environment: dict[str, str] = Field(default_factory=dict)
    working_directory: Path | None = None
    restart_limit: PositiveInt = 5
    restart_delay_seconds: PositiveInt = 5

    @field_validator("cluster_id", "node_id")
    @classmethod
    def validate_labels(cls, value: str) -> str:
        if not _SERVICE_LABEL.fullmatch(value):
            raise ValueError(
                "service labels may contain only letters, digits, dot, dash, underscore"
            )
        return value

    @property
    def service_name(self) -> str:
        return f"swarm-inference-{self.cluster_id}-{self.node_id}"


class CommandSpec(StrictModel):
    executable: str
    arguments: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    description: str

    @property
    def argv(self) -> list[str]:
        return [self.executable, *self.arguments]


class CommandResult(StrictModel):
    command: CommandSpec
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class ServiceStatus(StrictModel):
    service_name: str
    mode: Literal[
        "windows-task",
        "systemd-user",
        "launch-agent",
        "foreground",
        "unavailable",
    ]
    installed: bool
    running: bool
    detail: str
    log_path: Path


class FirewallRuleSpec(StrictModel):
    cluster_id: str
    node_id: str
    control_ports: list[PositiveInt]
    data_ports: list[PositiveInt]
    private_subnets: list[str]

    @field_validator("cluster_id", "node_id")
    @classmethod
    def validate_labels(cls, value: str) -> str:
        if not _SERVICE_LABEL.fullmatch(value):
            raise ValueError(
                "firewall labels may contain only letters, digits, dot, dash, underscore"
            )
        return value

    @property
    def owner_label(self) -> str:
        return f"SwarmInference-{self.cluster_id}-{self.node_id}"

    @field_validator("control_ports", "data_ports")
    @classmethod
    def validate_ports(cls, value: list[int]) -> list[int]:
        if len(value) > 32:
            raise ValueError("firewall port lists are bounded to 32 entries")
        if any(port > 65535 for port in value):
            raise ValueError("firewall ports must be in the range 1..65535")
        return sorted(set(value))

    @field_validator("private_subnets")
    @classmethod
    def validate_private_subnets(cls, value: list[str]) -> list[str]:
        if len(value) > 16:
            raise ValueError("firewall subnet lists are bounded to 16 entries")
        networks: list[str] = []
        for item in value:
            try:
                network = ipaddress.ip_network(item, strict=True)
            except ValueError as exc:
                raise ValueError(f"invalid private subnet {item!r}") from exc
            if not isinstance(network, ipaddress.IPv4Network) or not any(
                network.subnet_of(private) for private in _PRIVATE_IPV4_NETWORKS
            ):
                raise ValueError(f"firewall subnet is outside RFC1918 private space: {item}")
            networks.append(str(network))
        if not networks:
            raise ValueError("at least one RFC1918 private subnet is required")
        return sorted(set(networks))

    def resource_name(self, platform_name: Literal["windows", "linux", "macos"]) -> str:
        return owned_firewall_resource_name(self.owner_label, platform_name=platform_name)


class FirewallStatus(StrictModel):
    owner_label: str
    resource_name: str
    configured: bool
    private_only: bool
    broader_existing_rules: list[str] = Field(default_factory=list)
    blocked: bool = False
    detail: str
    remediation_command: str | None = None


class PlatformDiagnostic(StrictModel):
    name: str
    status: Literal["pass", "warning", "fail", "not-run"]
    detail: str
    remediation: str | None = None


CommandRunner = Callable[[CommandSpec, Mapping[str, str]], CommandResult]


class PlatformAdapter(Protocol):
    @property
    def service_mode(
        self,
    ) -> Literal["windows-task", "systemd-user", "launch-agent", "foreground", "unavailable"]: ...

    def identity(self) -> PlatformIdentity: ...

    def state_directory(self) -> Path: ...

    def cache_directory(self) -> Path: ...

    def log_directory(self) -> Path: ...

    def accelerator_probes(self) -> list[BackendProbeResult]: ...

    def interface_addresses(self) -> list[InterfaceAddress]: ...

    def routed_source_address(self, destination_endpoint: str) -> str: ...

    def service_install_commands(self, definition: ServiceDefinition) -> list[CommandSpec]: ...

    def service_uninstall_commands(self, definition: ServiceDefinition) -> list[CommandSpec]: ...

    def service_start_commands(self, definition: ServiceDefinition) -> list[CommandSpec]: ...

    def service_stop_commands(self, definition: ServiceDefinition) -> list[CommandSpec]: ...

    def service_status_command(self, definition: ServiceDefinition) -> CommandSpec | None: ...

    def install_service(self, definition: ServiceDefinition) -> ServiceStatus: ...

    def uninstall_service(self, definition: ServiceDefinition) -> ServiceStatus: ...

    def start_service(self, definition: ServiceDefinition) -> ServiceStatus: ...

    def stop_service(self, definition: ServiceDefinition) -> ServiceStatus: ...

    def service_status(self, definition: ServiceDefinition) -> ServiceStatus: ...

    def configure_firewall(self, specification: FirewallRuleSpec) -> FirewallStatus: ...

    def firewall_status(self, specification: FirewallRuleSpec) -> FirewallStatus: ...

    def remove_firewall(self, specification: FirewallRuleSpec) -> FirewallStatus: ...

    def diagnostics(self) -> list[PlatformDiagnostic]: ...

    def service_log_location(self, definition: ServiceDefinition) -> Path: ...

    def safe_environment(self, extra: Mapping[str, str] | None = None) -> dict[str, str]: ...


def default_command_runner(
    command: CommandSpec,
    environment: Mapping[str, str],
) -> CommandResult:
    try:
        completed = subprocess.run(
            command.argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=command.timeout_seconds,
            env=dict(environment),
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command=command,
            exit_code=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            timed_out=True,
        )
    except OSError as exc:
        return CommandResult(command=command, exit_code=127, stderr=str(exc))
    return CommandResult(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


class BasePlatformAdapter(ABC):
    """Portable implementation shared by the three explicitly supported hosts."""

    service_mode: Literal[
        "windows-task", "systemd-user", "launch-agent", "foreground", "unavailable"
    ]

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        home_directory: Path | None = None,
        command_runner: CommandRunner = default_command_runner,
    ) -> None:
        self.environment = dict(environment if environment is not None else os.environ)
        self.home_directory = (home_directory or Path.home()).expanduser().resolve()
        self.command_runner = command_runner

    @abstractmethod
    def identity(self) -> PlatformIdentity: ...

    @abstractmethod
    def state_directory(self) -> Path: ...

    @abstractmethod
    def cache_directory(self) -> Path: ...

    @abstractmethod
    def log_directory(self) -> Path: ...

    @abstractmethod
    def _backend_candidates(self) -> Sequence[tuple[Backend, str]]: ...

    @abstractmethod
    def service_install_commands(self, definition: ServiceDefinition) -> list[CommandSpec]: ...

    @abstractmethod
    def service_uninstall_commands(self, definition: ServiceDefinition) -> list[CommandSpec]: ...

    @abstractmethod
    def service_start_commands(self, definition: ServiceDefinition) -> list[CommandSpec]: ...

    @abstractmethod
    def service_stop_commands(self, definition: ServiceDefinition) -> list[CommandSpec]: ...

    @abstractmethod
    def service_status_command(self, definition: ServiceDefinition) -> CommandSpec | None: ...

    @abstractmethod
    def configure_firewall(self, specification: FirewallRuleSpec) -> FirewallStatus: ...

    @abstractmethod
    def firewall_status(self, specification: FirewallRuleSpec) -> FirewallStatus: ...

    @abstractmethod
    def remove_firewall(self, specification: FirewallRuleSpec) -> FirewallStatus: ...

    def _torch_probe(self, backend: Backend, device_name: str) -> BackendProbeResult:
        memory = psutil.virtual_memory()
        default = BackendProbeResult(
            backend=backend,
            device=device_name,
            detected=False,
            operational=False,
            reason="PyTorch is not importable",
            total_memory_bytes=memory.total if backend == Backend.TORCH_CPU else 0,
            available_memory_bytes=memory.available if backend == Backend.TORCH_CPU else 0,
        )
        try:
            import torch
        except (ImportError, OSError) as exc:
            return default.model_copy(update={"reason": f"PyTorch import failed: {exc}"})
        detected = backend == Backend.TORCH_CPU
        if backend == Backend.TORCH_CUDA:
            detected = bool(torch.cuda.is_available())
        elif backend == Backend.TORCH_MPS:
            detected = bool(
                getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            )
        if not detected:
            return default.model_copy(
                update={"reason": f"{backend.value} is not visible to PyTorch"}
            )
        try:
            device = torch.device(device_name)
            supported: list[str] = []
            for label, dtype in (
                ("float32", torch.float32),
                ("float16", torch.float16),
                ("bfloat16", torch.bfloat16),
            ):
                try:
                    values = torch.ones((4, 4), device=device, dtype=dtype)
                    result = values @ values
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                    if result.float().sum().item() == 64.0:
                        supported.append(label)
                except (RuntimeError, OSError, TypeError):
                    continue
            if not supported:
                raise RuntimeError("no dtype completed a correct tensor operation")
            total = memory.total
            available = memory.available
            device_label = "CPU"
            if backend == Backend.TORCH_CUDA:
                index = device.index if device.index is not None else torch.cuda.current_device()
                free, total_cuda = torch.cuda.mem_get_info(index)
                total = int(total_cuda)
                available = int(free)
                device_label = str(torch.cuda.get_device_name(index))
            elif backend == Backend.TORCH_MPS:
                device_label = "Apple Metal Performance Shaders"
            return BackendProbeResult(
                backend=backend,
                device=device_name,
                detected=True,
                operational=True,
                reason="correct tensor probe passed",
                device_name=device_label,
                total_memory_bytes=total,
                available_memory_bytes=available,
                supported_dtypes=supported,
            )
        except (RuntimeError, OSError, TypeError) as exc:
            return default.model_copy(
                update={
                    "detected": True,
                    "reason": f"operational tensor probe failed: {exc}",
                }
            )

    def accelerator_probes(self) -> list[BackendProbeResult]:
        return [
            self._torch_probe(backend, device) for backend, device in self._backend_candidates()
        ]

    def interface_addresses(self) -> list[InterfaceAddress]:
        statistics = psutil.net_if_stats()
        values: list[InterfaceAddress] = []
        for interface, addresses in psutil.net_if_addrs().items():
            status = statistics.get(interface)
            for address in addresses:
                if address.family not in {socket.AF_INET, socket.AF_INET6}:
                    continue
                raw = address.address.split("%", 1)[0]
                try:
                    parsed = ipaddress.ip_address(raw)
                except ValueError:
                    continue
                prefix_length = None
                if address.netmask:
                    try:
                        prefix_length = ipaddress.ip_network(
                            f"{raw}/{address.netmask}", strict=False
                        ).prefixlen
                    except ValueError:
                        prefix_length = None
                values.append(
                    InterfaceAddress(
                        interface=interface,
                        address=raw,
                        prefix_length=prefix_length,
                        is_private=parsed.is_private,
                        is_loopback=parsed.is_loopback,
                        is_up=bool(status and status.isup),
                        mtu=status.mtu if status and status.mtu > 0 else None,
                    )
                )
        return sorted(values, key=lambda item: (item.interface, item.address))

    def routed_source_address(self, destination_endpoint: str) -> str:
        from swarm_inference.host import split_endpoint

        host, port = split_endpoint(destination_endpoint)
        if port == 0:
            raise ValueError("routed-source discovery requires a non-zero destination port")
        candidates = socket.getaddrinfo(host, port, type=socket.SOCK_DGRAM)
        error: OSError | None = None
        for family, socket_type, protocol, _, address in candidates:
            try:
                with socket.socket(family, socket_type, protocol) as probe:
                    probe.settimeout(3.0)
                    probe.connect(address)
                    source = str(probe.getsockname()[0]).split("%", 1)[0]
                    return source
            except OSError as exc:
                error = exc
        raise OSError(f"could not discover a routed source address: {error}")

    def safe_environment(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        allowed = {
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "TEMP",
            "TMP",
            "HOME",
            "USERPROFILE",
            "LOCALAPPDATA",
            "APPDATA",
            "XDG_STATE_HOME",
            "XDG_CACHE_HOME",
            "LANG",
            "LC_ALL",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "CUDA_VISIBLE_DEVICES",
        }
        result = {key: value for key, value in self.environment.items() if key.upper() in allowed}
        for key, value in (extra or {}).items():
            upper = key.upper()
            if any(term in upper for term in ("SECRET", "PRIVATE_KEY", "PAIRING", "PROOF")):
                raise ValueError(f"sensitive variable {key!r} cannot be persisted in a service")
            result[key] = value
        return result

    def _run_commands(self, commands: Sequence[CommandSpec]) -> tuple[bool, str]:
        details: list[str] = []
        for command in commands:
            result = self.command_runner(command, self.safe_environment())
            if not result.succeeded:
                detail = result.stderr.strip() or result.stdout.strip() or "command failed"
                return False, f"{command.description}: {detail}"
            details.append(command.description)
        return True, "; ".join(details)

    def service_log_location(self, definition: ServiceDefinition) -> Path:
        return self.log_directory() / f"{definition.service_name}.log"

    def _service_status(
        self,
        definition: ServiceDefinition,
        *,
        installed: bool,
        running: bool,
        detail: str,
    ) -> ServiceStatus:
        return ServiceStatus(
            service_name=definition.service_name,
            mode=self.service_mode,
            installed=installed,
            running=running,
            detail=detail,
            log_path=self.service_log_location(definition),
        )

    def install_service(self, definition: ServiceDefinition) -> ServiceStatus:
        commands = self.service_install_commands(definition)
        if not commands:
            return self._service_status(
                definition,
                installed=False,
                running=False,
                detail="user service management is unavailable; use --foreground",
            )
        succeeded, detail = self._run_commands(commands)
        return self._service_status(
            definition,
            installed=succeeded,
            running=succeeded,
            detail=detail,
        )

    def uninstall_service(self, definition: ServiceDefinition) -> ServiceStatus:
        succeeded, detail = self._run_commands(self.service_uninstall_commands(definition))
        return self._service_status(
            definition,
            installed=not succeeded,
            running=False,
            detail=detail,
        )

    def start_service(self, definition: ServiceDefinition) -> ServiceStatus:
        succeeded, detail = self._run_commands(self.service_start_commands(definition))
        return self._service_status(
            definition,
            installed=True,
            running=succeeded,
            detail=detail,
        )

    def stop_service(self, definition: ServiceDefinition) -> ServiceStatus:
        succeeded, detail = self._run_commands(self.service_stop_commands(definition))
        return self._service_status(
            definition,
            installed=True,
            running=not succeeded,
            detail=detail,
        )

    def service_status(self, definition: ServiceDefinition) -> ServiceStatus:
        command = self.service_status_command(definition)
        if command is None:
            return self._service_status(
                definition,
                installed=False,
                running=False,
                detail="user service manager is unavailable",
            )
        result = self.command_runner(command, self.safe_environment())
        output = f"{result.stdout}\n{result.stderr}".lower()
        installed = result.exit_code == 0
        running = installed and any(
            marker in output for marker in ("running", "active", "state: ready")
        )
        return self._service_status(
            definition,
            installed=installed,
            running=running,
            detail=(result.stdout.strip() or result.stderr.strip() or "status queried"),
        )

    def diagnostics(self) -> list[PlatformDiagnostic]:
        identity = self.identity()
        return [
            PlatformDiagnostic(
                name="platform-implementation",
                status="pass" if identity.implementation_status == "implemented" else "fail",
                detail=identity.implementation_reason,
            )
        ]


__all__ = [
    "PLATFORM_IMPLEMENTATION_CONTRACT",
    "BackendProbeResult",
    "BasePlatformAdapter",
    "CommandResult",
    "CommandRunner",
    "CommandSpec",
    "FirewallRuleSpec",
    "FirewallStatus",
    "ImplementationStatus",
    "InterfaceAddress",
    "PlatformAdapter",
    "PlatformDiagnostic",
    "PlatformIdentity",
    "ServiceDefinition",
    "ServiceStatus",
    "ValidationStatus",
    "default_command_runner",
    "owned_firewall_resource_name",
    "platform_implementation",
]
