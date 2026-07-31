"""Isolated external-engine environment management for Experiment 004."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExternalEngineEnvironment:
    engine: str
    kind: str
    root: str
    requested_version: str
    status: str
    python_executable: str | None = None
    image: str | None = None
    installation_seconds: float = 0.0
    install_command: list[str] = field(default_factory=list)
    launch_command: list[str] = field(default_factory=list)
    lock_path: str | None = None
    installation_log: str | None = None
    failure_log: str | None = None
    gpu_compatible: bool = False
    diagnostic: str | None = None
    probe: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def _run_logged(
    command: list[str],
    *,
    log_path: Path,
    cwd: Path,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    started = time.time()
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "command": command,
                "cwd": str(cwd),
                "started_unix_seconds": started,
                "finished_unix_seconds": time.time(),
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def provision_huggingface_environment(
    *,
    repository_root: Path,
    environment_root: Path,
    transformers_version: str,
    torch_version: str,
) -> ExternalEngineEnvironment:
    root = environment_root / "huggingface"
    logs = root / "logs"
    python = root / "Scripts" / "python.exe"
    record = ExternalEngineEnvironment(
        engine="huggingface",
        kind="virtualenv",
        root=str(root.resolve()),
        requested_version=transformers_version,
        status="PENDING",
        python_executable=str(python.resolve()),
    )
    root.mkdir(parents=True, exist_ok=True)
    uv = shutil.which("uv")
    started = time.perf_counter()
    try:
        if not python.is_file():
            create_command = (
                [uv, "venv", str(root), "--python", sys.executable]
                if uv is not None
                else [sys.executable, "-m", "venv", str(root)]
            )
            create = _run_logged(
                create_command,
                log_path=logs / "create.json",
                cwd=repository_root,
                timeout=300,
            )
            if create.returncode != 0:
                raise RuntimeError("isolated Hugging Face venv creation failed")
        packages = [
            f"torch=={torch_version}",
            f"transformers=={transformers_version}",
            "safetensors",
            "huggingface-hub",
            "psutil",
            "numpy",
            "triton-windows==3.7.1.post27",
        ]
        install_command = (
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                *packages,
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu130",
            ]
            if uv is not None
            else [
                str(python),
                "-m",
                "pip",
                "install",
                *packages,
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu130",
            ]
        )
        record.install_command = install_command
        install_log = logs / "install.json"
        record.installation_log = str(install_log.resolve())
        install = _run_logged(
            install_command,
            log_path=install_log,
            cwd=repository_root,
            timeout=1800,
        )
        if install.returncode != 0:
            raise RuntimeError(
                f"isolated Hugging Face package install failed ({install.returncode})"
            )
        lock_path = root / "requirements.lock.txt"
        freeze_command = (
            [uv, "pip", "freeze", "--python", str(python)]
            if uv is not None
            else [str(python), "-m", "pip", "freeze"]
        )
        freeze = subprocess.run(
            freeze_command,
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        lock_path.write_text(freeze.stdout, encoding="utf-8")
        record.lock_path = str(lock_path.resolve())
        record.launch_command = [
            str(python),
            str(repository_root / "scripts" / "external_engine_benchmark.py"),
            "--job",
            "<job.json>",
            "--output",
            "<result.json>",
        ]
        probe_path = root / "environment.json"
        probe = _run_logged(
            [
                str(python),
                str(repository_root / "scripts" / "external_environment_probe.py"),
                "--engine",
                "huggingface",
                "--output",
                str(probe_path),
            ],
            log_path=logs / "probe.json",
            cwd=repository_root,
            timeout=120,
        )
        if probe_path.is_file():
            record.probe = json.loads(probe_path.read_text(encoding="utf-8"))
        record.gpu_compatible = probe.returncode == 0 and bool(record.probe.get("cuda_available"))
        record.status = "PASS" if record.gpu_compatible else "FAIL"
        if not record.gpu_compatible:
            record.diagnostic = "isolated Hugging Face CUDA probe failed"
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        record.status = "FAIL"
        record.diagnostic = f"{type(exc).__name__}: {exc}"
        failure = logs / "failure.json"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(
            json.dumps(record.payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        record.failure_log = str(failure.resolve())
    record.installation_seconds = time.perf_counter() - started
    return record


def inspect_linux_engine_prerequisites(
    *,
    repository_root: Path,
    environment_root: Path,
    engine: str,
    version: str,
    image: str,
) -> ExternalEngineEnvironment:
    root = environment_root / engine
    root.mkdir(parents=True, exist_ok=True)
    record = ExternalEngineEnvironment(
        engine=engine,
        kind="wsl2-or-container",
        root=str(root.resolve()),
        requested_version=version,
        image=image,
        status="FAIL",
    )
    observations: dict[str, Any] = {}
    commands = {
        "wsl": ["wsl.exe", "--list", "--verbose"],
        "docker_version": ["docker", "version"],
        "docker_gpu": [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "nvidia/cuda:13.0.0-base-ubuntu24.04",
            "nvidia-smi",
        ],
    }
    for name, command in commands.items():
        try:
            result = _run_logged(
                command,
                log_path=root / "logs" / f"{name}.json",
                cwd=repository_root,
                timeout=180,
            )
            observations[name] = {
                "command": command,
                "return_code": result.returncode,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            observations[name] = {
                "command": command,
                "error": f"{type(exc).__name__}: {exc}",
            }
    package_name = {
        "sglang": "sglang",
        "vllm": "vllm",
        "tensorrt_llm": "tensorrt_llm",
    }.get(engine, engine)
    pull_command = ["docker", "pull", image]
    record.install_command = pull_command
    record.installation_log = str((root / "logs" / "pull.json").resolve())
    script_name = {
        "sglang": "sglang_engine_benchmark.py",
        "vllm": "vllm_engine_benchmark.py",
        "tensorrt_llm": "tensorrt_llm_engine_benchmark.py",
    }.get(engine, "<engine benchmark script>")
    record.launch_command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--shm-size",
        "16g",
        "--ipc=host",
        "-v",
        "<repository>:/workspace",
        "-v",
        "<huggingface-cache>:/root/.cache/huggingface",
        *([] if engine == "tensorrt_llm" else ["--entrypoint", "python3"]),
        image,
        *(["python3"] if engine == "tensorrt_llm" else []),
        f"/workspace/scripts/{script_name}",
        "--job",
        "<job.json>",
        "--output",
        "<result.json>",
    ]
    started = time.perf_counter()
    try:
        pull = _run_logged(
            pull_command,
            log_path=root / "logs" / "pull.json",
            cwd=repository_root,
            timeout=7200,
        )
        observations["image_pull"] = {
            "command": pull_command,
            "return_code": pull.returncode,
            "stdout": pull.stdout[-4000:],
            "stderr": pull.stderr[-4000:],
        }
        if pull.returncode != 0:
            raise RuntimeError(f"container image pull failed with exit code {pull.returncode}")
        image_inspect = _run_logged(
            [
                "docker",
                "image",
                "inspect",
                image,
                "--format",
                "{{json .}}",
            ],
            log_path=root / "logs" / "image-inspect.json",
            cwd=repository_root,
            timeout=120,
        )
        observations["image_identity"] = (
            json.loads(image_inspect.stdout)
            if image_inspect.returncode == 0 and image_inspect.stdout.strip()
            else {
                "return_code": image_inspect.returncode,
                "stderr": image_inspect.stderr[-4000:],
            }
        )
        probe_code = (
            "import importlib.metadata as m,json,platform,sys,torch;"
            f"package={package_name!r};"
            "print(json.dumps({'python_version':platform.python_version(),"
            "'python_executable':sys.executable,'torch_version':torch.__version__,"
            "'cuda_version':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
            "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
            "'compute_capability':list(torch.cuda.get_device_capability(0)) "
            "if torch.cuda.is_available() else None,'engine_package':package,"
            "'engine_version':m.version(package)}))"
        )
        probe_command = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            *([] if engine == "tensorrt_llm" else ["--entrypoint", "python3"]),
            image,
            *(["python3"] if engine == "tensorrt_llm" else []),
            "-c",
            probe_code,
        ]
        engine_probe = _run_logged(
            probe_command,
            log_path=root / "logs" / "engine-probe.json",
            cwd=repository_root,
            timeout=600,
        )
        if engine_probe.returncode == 0 and engine_probe.stdout.strip():
            probe_line = engine_probe.stdout.strip().splitlines()[-1]
            observations["engine_probe"] = json.loads(probe_line)
        else:
            observations["engine_probe"] = {
                "return_code": engine_probe.returncode,
                "stdout": engine_probe.stdout[-4000:],
                "stderr": engine_probe.stderr[-4000:],
            }
        freeze_command = [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "python3",
            image,
            "-m",
            "pip",
            "freeze",
        ]
        freeze = _run_logged(
            freeze_command,
            log_path=root / "logs" / "package-lock.json",
            cwd=repository_root,
            timeout=600,
        )
        lock_path = root / "requirements.lock.txt"
        lock_path.write_text(freeze.stdout, encoding="utf-8")
        record.lock_path = str(lock_path.resolve())
        gpu_ok = (
            observations.get("docker_gpu", {}).get("return_code") == 0
            and engine_probe.returncode == 0
            and bool(observations.get("engine_probe", {}).get("cuda_available"))
        )
        record.gpu_compatible = gpu_ok
        record.status = "PASS" if gpu_ok else "FAIL"
        record.diagnostic = (
            None
            if gpu_ok
            else f"{engine} container package/CUDA probe failed; see engine-probe.json"
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        record.status = "FAIL"
        record.gpu_compatible = False
        record.diagnostic = f"{type(exc).__name__}: {exc}"
    record.installation_seconds = time.perf_counter() - started
    probe_path = root / "environment.json"
    probe_path.write_text(
        json.dumps(observations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    record.probe = observations
    record.failure_log = None if record.status == "PASS" else str(probe_path.resolve())
    return record
