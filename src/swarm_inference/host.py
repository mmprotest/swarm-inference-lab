"""Cross-platform host, endpoint, and child-process helpers."""

from __future__ import annotations

import ipaddress
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import psutil

from swarm_inference.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class HostRuntime:
    """Normalised facts used by launchers and environment diagnostics."""

    system: str
    release: str
    machine: str
    is_wsl: bool

    @property
    def is_windows(self) -> bool:
        return self.system == "Windows"

    @property
    def is_linux(self) -> bool:
        return self.system == "Linux"

    @property
    def is_macos(self) -> bool:
        return self.system == "Darwin"


def detect_host_runtime() -> HostRuntime:
    system = platform.system()
    release = platform.release()
    if system == "Windows":
        windows_version = sys.getwindowsversion()
        product = "11" if windows_version.build >= 22000 else platform.release()
        release = f"{product} build {windows_version.build}"
    proc_version = ""
    proc_path = Path("/proc/version")
    if proc_path.is_file():
        proc_version = proc_path.read_text(encoding="utf-8", errors="replace")
    is_wsl = "microsoft" in f"{release} {proc_version}".lower() or bool(
        os.environ.get("WSL_DISTRO_NAME")
    )
    return HostRuntime(
        system=system,
        release=release,
        machine=platform.machine(),
        is_wsl=is_wsl,
    )


def split_endpoint(endpoint: str) -> tuple[str, int]:
    """Parse ``host:port`` and bracketed IPv6 endpoints without OS assumptions."""

    value = endpoint.strip()
    if not value:
        raise ConfigurationError("endpoint must not be empty")
    if value.startswith("["):
        closing = value.find("]")
        if closing < 0 or closing + 1 >= len(value) or value[closing + 1] != ":":
            raise ConfigurationError(
                f"invalid bracketed IPv6 endpoint {endpoint!r}; expected [address]:port"
            )
        host = value[1:closing]
        port_text = value[closing + 2 :]
    else:
        try:
            host, port_text = value.rsplit(":", 1)
        except ValueError as exc:
            raise ConfigurationError(f"invalid endpoint {endpoint!r}; expected host:port") from exc
        if ":" in host:
            raise ConfigurationError(
                f"IPv6 endpoint {endpoint!r} must use bracket notation [address]:port"
            )
    if not host:
        raise ConfigurationError(f"endpoint {endpoint!r} has no host")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ConfigurationError(f"endpoint {endpoint!r} has an invalid port") from exc
    if not 0 <= port <= 65535:
        raise ConfigurationError(f"endpoint {endpoint!r} port must be in [0, 65535]")
    return host, port


def format_endpoint(host: str, port: int) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def is_wildcard_host(host: str) -> bool:
    normalised = host.strip("[]").lower()
    return normalised in {"0.0.0.0", "::", "*"}


def is_loopback_host(host: str) -> bool:
    normalised = host.strip("[]").lower()
    if normalised == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalised).is_loopback
    except ValueError:
        return False


def local_ip_addresses() -> set[str]:
    """Return normalised IP addresses assigned to local interfaces."""

    addresses = {"127.0.0.1", "::1"}
    for interface in psutil.net_if_addrs().values():
        for address in interface:
            if address.family not in {socket.AF_INET, socket.AF_INET6}:
                continue
            # IPv6 scope identifiers are interface-local decorations.
            addresses.add(address.address.split("%", 1)[0].lower())
    return addresses


def endpoint_is_local(endpoint: str) -> bool:
    host, port = split_endpoint(endpoint)
    if is_wildcard_host(host) or is_loopback_host(host):
        return True
    try:
        resolved = {
            str(item[4][0]).split("%", 1)[0].lower()
            for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        }
    except OSError:
        return False
    return bool(resolved & local_ip_addresses())


def qualifies_as_remote_physical_worker(
    *,
    worker_hostname: str,
    endpoint: str | None,
    coordinator_hostname: str | None = None,
) -> bool:
    """Require independent host identity and a non-local advertised endpoint."""

    local_hostname = (coordinator_hostname or socket.gethostname()).strip().lower()
    remote_hostname = worker_hostname.strip().lower()
    return bool(
        endpoint
        and remote_hostname
        and remote_hostname != local_hostname
        and not endpoint_is_local(endpoint)
    )


def discover_source_address(coordinator_endpoint: str) -> str:
    """Return the local address the OS routes toward the coordinator.

    A UDP ``connect`` chooses an interface without transmitting application
    data. This works on Windows, Linux, and macOS and avoids guessing an
    interface name.
    """

    coordinator_host, coordinator_port = split_endpoint(coordinator_endpoint)
    if is_wildcard_host(coordinator_host):
        raise ConfigurationError("coordinator endpoint cannot use a wildcard host")
    if is_loopback_host(coordinator_host):
        return "::1" if ":" in coordinator_host else "127.0.0.1"
    try:
        addresses = socket.getaddrinfo(
            coordinator_host,
            coordinator_port,
            type=socket.SOCK_DGRAM,
        )
    except OSError as exc:
        raise ConfigurationError(
            f"could not resolve coordinator host {coordinator_host!r}: {exc}"
        ) from exc
    errors: list[str] = []
    for family, sock_type, protocol, _, address in addresses:
        with socket.socket(family, sock_type, protocol) as probe:
            try:
                probe.connect(address)
                local = str(probe.getsockname()[0])
            except OSError as exc:
                errors.append(str(exc))
                continue
        if local and not is_wildcard_host(local):
            return local
    detail = "; ".join(errors) if errors else "no routable address"
    raise ConfigurationError(
        f"could not discover a local route to {coordinator_endpoint}: {detail}"
    )


def resolve_advertised_endpoint(
    *,
    listen_endpoint: str,
    coordinator_endpoint: str,
    explicit_endpoint: str | None,
    option_name: str = "worker --advertise",
) -> str:
    """Resolve a coordinator-reachable worker endpoint.

    Binding to ``0.0.0.0`` or ``::`` is valid, but advertising either address
    is not. When no explicit advertisement is supplied, the routed local
    interface is discovered from the coordinator endpoint.
    """

    listen_host, listen_port = split_endpoint(listen_endpoint)
    if listen_port == 0:
        raise ConfigurationError(
            "physical worker listen endpoints must use an explicit non-zero port"
        )
    if explicit_endpoint is not None:
        advertised_host, advertised_port = split_endpoint(explicit_endpoint)
        if is_wildcard_host(advertised_host):
            raise ConfigurationError(
                f"{option_name} must be coordinator-reachable, not a wildcard address"
            )
        if advertised_port == 0:
            raise ConfigurationError(f"{option_name} must use a non-zero port")
        return format_endpoint(advertised_host, advertised_port)
    advertised_host = (
        discover_source_address(coordinator_endpoint)
        if is_wildcard_host(listen_host)
        else listen_host
    )
    if is_wildcard_host(advertised_host):
        raise ConfigurationError("could not derive a coordinator-reachable worker endpoint")
    return format_endpoint(advertised_host, listen_port)


def resolve_data_plane_advertised_endpoint(
    *,
    listen_endpoint: str,
    coordinator_endpoint: str,
    explicit_endpoint: str | None,
) -> str:
    """Resolve a separately advertised, coordinator-reachable data endpoint."""

    return resolve_advertised_endpoint(
        listen_endpoint=listen_endpoint,
        coordinator_endpoint=coordinator_endpoint,
        explicit_endpoint=explicit_endpoint,
        option_name="worker --data-advertise",
    )


def stop_process(
    process: subprocess.Popen[str],
    *,
    terminate_timeout_s: float = 10.0,
    kill_timeout_s: float = 5.0,
) -> None:
    """Stop a child process using APIs available on every supported OS."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=terminate_timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=kill_timeout_s)
