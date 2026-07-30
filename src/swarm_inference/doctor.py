"""Host compatibility inspection for synthetic, CPU, CUDA, and MPS workers."""

from __future__ import annotations

import json
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import psutil

from swarm_inference.host import detect_host_runtime


class DoctorBackend(StrEnum):
    AUTO = "auto"
    SYNTHETIC = "synthetic"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass(slots=True)
class DoctorCheck:
    name: str
    status: str
    detail: str
    required_for_cuda: bool = False


@dataclass(slots=True)
class DoctorReport:
    compatible_cuda: bool
    compatible_cpu: bool
    checks: list[DoctorCheck] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: str | None = None
    selected_backend: str = DoctorBackend.AUTO.value
    compatible_backends: dict[str, bool] = field(default_factory=dict)

    @property
    def selected_backend_compatible(self) -> bool:
        return self.compatible_backends.get(self.selected_backend, False)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_backend_compatible"] = self.selected_backend_compatible
        return payload


@dataclass(frozen=True, slots=True)
class TorchProbe:
    cpu: bool
    cuda: bool
    mps: bool
    details: dict[str, Any]
    cpu_message: str
    cuda_message: str
    mps_message: str


def _nvidia_smi() -> tuple[bool, dict[str, Any], str]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False, {}, "nvidia-smi is not on PATH"
    query = [
        executable,
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            query,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, {}, f"nvidia-smi failed: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return False, {}, f"nvidia-smi exited {result.returncode}: {detail}"
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            return False, {}, f"unexpected nvidia-smi response: {line}"
        rows.append(
            {
                "gpu_model": parts[0],
                "total_vram_mib": _safe_float(parts[1]),
                "free_vram_mib": _safe_float(parts[2]),
                "driver_version": parts[3],
                "summary": line,
            }
        )
    if not rows:
        return False, {}, "nvidia-smi returned no GPU rows"
    primary = dict(rows[0])
    primary["devices"] = rows
    return True, primary, "; ".join(str(row["summary"]) for row in rows)


def _safe_float(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def _torch_probe() -> TorchProbe:
    try:
        import torch
    except (ImportError, OSError) as exc:
        message = f"PyTorch import failed: {exc}"
        return TorchProbe(
            cpu=False,
            cuda=False,
            mps=False,
            details={"installed": False, "import_error": str(exc)},
            cpu_message=message,
            cuda_message=message,
            mps_message=message,
        )

    details: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible": False,
        "mps_visible": False,
    }
    try:
        cpu_result = float((torch.ones(4, dtype=torch.float32) * 2).sum().item())
        cpu_ok = cpu_result == 8.0
        cpu_message = f"PyTorch {torch.__version__} CPU operation passed"
    except (RuntimeError, OSError, AssertionError) as exc:
        cpu_ok = False
        cpu_message = f"PyTorch CPU operation failed: {exc}"

    cuda_ok = False
    cuda_message = "PyTorch is installed but CUDA is not visible"
    try:
        details["cuda_visible"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            details["compute_capability"] = f"{capability[0]}.{capability[1]}"
            details["gpu_model"] = torch.cuda.get_device_name(0)
            supported_arches = (
                torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
            )
            details["supported_arches"] = supported_arches
            arch = f"sm_{capability[0]}{capability[1]}"
            build_supports_gpu = not supported_arches or arch in supported_arches
            details["build_supports_gpu"] = build_supports_gpu
            if build_supports_gpu:
                probe = torch.ones(1, device="cuda")
                torch.cuda.synchronize()
                cuda_ok = bool(probe.item() == 1)
                cuda_message = f"CUDA operation passed on {details['gpu_model']}"
            else:
                cuda_message = f"installed PyTorch build lacks {arch}; supported={supported_arches}"
    except (RuntimeError, OSError, AssertionError) as exc:
        cuda_message = f"CUDA device query or operation failed: {exc}"

    mps_ok = False
    mps_message = "PyTorch MPS backend is not available"
    try:
        mps_backend = getattr(torch.backends, "mps", None)
        details["mps_visible"] = bool(mps_backend and mps_backend.is_available())
        if details["mps_visible"]:
            probe = torch.ones(1, device="mps")
            mps_ok = bool(probe.cpu().item() == 1)
            mps_message = "PyTorch MPS operation passed"
    except (RuntimeError, OSError, AssertionError) as exc:
        mps_message = f"PyTorch MPS operation failed: {exc}"

    return TorchProbe(
        cpu=cpu_ok,
        cuda=cuda_ok,
        mps=mps_ok,
        details=details,
        cpu_message=cpu_message,
        cuda_message=cuda_message,
        mps_message=mps_message,
    )


def _port_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _network_interfaces() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    stats = psutil.net_if_stats()
    for name, addresses in psutil.net_if_addrs().items():
        interface_stats = stats.get(name)
        result.append(
            {
                "name": name,
                "up": interface_stats.isup if interface_stats is not None else None,
                "speed_mbps": interface_stats.speed if interface_stats is not None else None,
                "addresses": [
                    address.address
                    for address in addresses
                    if address.family in {socket.AF_INET, socket.AF_INET6}
                ],
            }
        )
    return result


def _select_backend(
    requested: DoctorBackend,
    *,
    nvidia_visible: bool,
    system: str,
    machine: str,
) -> DoctorBackend:
    if requested != DoctorBackend.AUTO:
        return requested
    if nvidia_visible:
        return DoctorBackend.CUDA
    if system == "Darwin" and machine.lower() in {"arm64", "aarch64"}:
        return DoctorBackend.MPS
    return DoctorBackend.CPU


def _recommendation(selected: DoctorBackend, *, system: str) -> str:
    backend = selected.value
    if system == "Windows":
        return (
            "powershell.exe -NoProfile -ExecutionPolicy Bypass "
            f"-File .\\scripts\\bootstrap.ps1 -Backend {backend}"
        )
    return f"bash ./scripts/bootstrap.sh --backend {backend}"


def inspect_environment(
    *,
    required_ports: tuple[int, ...] = (50051, 50052, 50053, 50054, 50055),
    target_backend: DoctorBackend | str = DoctorBackend.AUTO,
    bind_host: str = "0.0.0.0",
) -> DoctorReport:
    requested = DoctorBackend(target_backend)
    host = detect_host_runtime()
    checks: list[DoctorCheck] = []
    checks.append(
        DoctorCheck(
            "host-runtime",
            "pass",
            (
                f"{host.system} {host.release} ({host.machine}); "
                f"WSL={'yes' if host.is_wsl else 'no'}"
            ),
        )
    )

    python_ok = sys.version_info[:2] == (3, 11)
    checks.append(
        DoctorCheck(
            "python",
            "pass" if python_ok else "fail",
            platform.python_version(),
            required_for_cuda=True,
        )
    )

    smi_ok, smi_details, smi_message = _nvidia_smi()
    selected = _select_backend(
        requested,
        nvidia_visible=smi_ok,
        system=host.system,
        machine=host.machine,
    )
    checks.append(
        DoctorCheck(
            "nvidia-smi",
            "pass" if smi_ok else ("fail" if selected == DoctorBackend.CUDA else "info"),
            smi_message,
            required_for_cuda=True,
        )
    )

    torch = _torch_probe()
    checks.extend(
        [
            DoctorCheck(
                "pytorch-cpu",
                "pass" if torch.cpu else ("fail" if selected == DoctorBackend.CPU else "info"),
                torch.cpu_message,
            ),
            DoctorCheck(
                "pytorch-cuda",
                "pass" if torch.cuda else ("fail" if selected == DoctorBackend.CUDA else "info"),
                torch.cuda_message,
                required_for_cuda=True,
            ),
            DoctorCheck(
                "pytorch-mps",
                "pass" if torch.mps else ("fail" if selected == DoctorBackend.MPS else "info"),
                torch.mps_message,
            ),
        ]
    )

    memory = psutil.virtual_memory()
    disk_target = Path.cwd().anchor or str(Path.cwd())
    disk = psutil.disk_usage(disk_target)
    ports = {str(port): _port_available(bind_host, port) for port in required_ports}
    ports_ok = all(ports.values())
    checks.append(
        DoctorCheck(
            "ports",
            "pass" if ports_ok else "fail",
            ", ".join(f"{port}={'free' if free else 'busy'}" for port, free in ports.items())
            or "no ports requested",
            required_for_cuda=False,
        )
    )

    compatible = {
        DoctorBackend.SYNTHETIC.value: python_ok and ports_ok,
        DoctorBackend.CPU.value: python_ok and torch.cpu and ports_ok,
        DoctorBackend.CUDA.value: python_ok and smi_ok and torch.cuda and ports_ok,
        DoctorBackend.MPS.value: python_ok and torch.mps and ports_ok,
    }
    selected_ok = compatible[selected.value]
    return DoctorReport(
        compatible_cuda=compatible[DoctorBackend.CUDA.value],
        compatible_cpu=compatible[DoctorBackend.CPU.value],
        selected_backend=selected.value,
        compatible_backends=compatible,
        checks=checks,
        details={
            "wsl": host.is_wsl,
            "selected_backend": selected.value,
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable,
            },
            "os": {
                "system": host.system,
                "release": host.release,
                "version": platform.version(),
                "machine": host.machine,
            },
            "cpu": {
                "model": platform.processor() or "unknown",
                "logical_count": psutil.cpu_count(logical=True),
                "physical_count": psutil.cpu_count(logical=False),
            },
            "memory": {
                "total_bytes": memory.total,
                "available_bytes": memory.available,
            },
            "disk": {
                "path": disk_target,
                "total_bytes": disk.total,
                "free_bytes": disk.free,
            },
            "nvidia": smi_details,
            "torch": torch.details,
            "network_interfaces": _network_interfaces(),
            "port_bind_host": bind_host,
            "required_ports": ports,
        },
        recommendation=None if selected_ok else _recommendation(selected, system=host.system),
    )


def render_doctor_report(report: DoctorReport, *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(report.to_dict(), indent=2, sort_keys=True)
    lines = [
        "swarm-inference-lab environment doctor",
        f"Selected backend: {report.selected_backend}",
        f"Selected backend status: {'PASS' if report.selected_backend_compatible else 'FAIL'}",
        f"Synthetic backend: {'PASS' if report.compatible_backends.get('synthetic') else 'FAIL'}",
        f"CPU backend: {'PASS' if report.compatible_cpu else 'FAIL'}",
        f"CUDA backend: {'PASS' if report.compatible_cuda else 'FAIL'}",
        f"MPS backend: {'PASS' if report.compatible_backends.get('mps') else 'FAIL'}",
    ]
    for check in report.checks:
        lines.append(f"[{check.status.upper():4}] {check.name}: {check.detail}")
    memory = report.details["memory"]
    disk = report.details["disk"]
    lines.extend(
        [
            f"System RAM: {memory['available_bytes'] / 2**30:.1f} GiB free / "
            f"{memory['total_bytes'] / 2**30:.1f} GiB total",
            f"Disk: {disk['free_bytes'] / 2**30:.1f} GiB free / "
            f"{disk['total_bytes'] / 2**30:.1f} GiB total",
        ]
    )
    if report.recommendation:
        lines.append(f"Recommendation: {report.recommendation}")
    return "\n".join(lines)
