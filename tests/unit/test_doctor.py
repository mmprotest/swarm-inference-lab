from __future__ import annotations

from swarm_inference import doctor
from swarm_inference.doctor import DoctorBackend, TorchProbe
from swarm_inference.host import HostRuntime


def test_declared_python_compatibility_range() -> None:
    assert doctor.python_version_supported(3, 11)
    assert doctor.python_version_supported(3, 12)
    assert doctor.python_version_supported(3, 13)
    assert not doctor.python_version_supported(3, 10)
    assert not doctor.python_version_supported(3, 14)


def test_windows_cuda_recommendation_is_native_not_wsl(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor,
        "detect_host_runtime",
        lambda: HostRuntime(
            system="Windows",
            release="11",
            machine="AMD64",
            is_wsl=False,
        ),
    )
    monkeypatch.setattr(
        doctor,
        "_nvidia_smi",
        lambda: (True, {"gpu_model": "test GPU"}, "test GPU"),
    )
    monkeypatch.setattr(
        doctor,
        "_torch_probe",
        lambda: TorchProbe(
            cpu=True,
            cuda=False,
            mps=False,
            details={"installed": True},
            cpu_message="passed",
            cuda_message="not visible",
            mps_message="not available",
        ),
    )
    report = doctor.inspect_environment(required_ports=(), target_backend=DoctorBackend.AUTO)
    assert report.selected_backend == "cuda"
    assert not report.selected_backend_compatible
    assert report.recommendation is not None
    assert "bootstrap.ps1 -Backend cuda" in report.recommendation
    assert "WSL" not in report.recommendation


def test_cpu_target_does_not_require_nvidia(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "_nvidia_smi", lambda: (False, {}, "not installed"))
    monkeypatch.setattr(
        doctor,
        "_torch_probe",
        lambda: TorchProbe(
            cpu=True,
            cuda=False,
            mps=False,
            details={"installed": True},
            cpu_message="passed",
            cuda_message="not visible",
            mps_message="not available",
        ),
    )
    report = doctor.inspect_environment(required_ports=(), target_backend=DoctorBackend.CPU)
    assert report.selected_backend_compatible
    nvidia = next(check for check in report.checks if check.name == "nvidia-smi")
    assert nvidia.status == "info"
