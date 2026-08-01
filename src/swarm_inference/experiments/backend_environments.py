"""Isolated backend provisioning and immutable build evidence for Experiment 007."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swarm_inference.protocol.checksums import sha256_file


@dataclass(frozen=True, slots=True)
class LoggedCommand:
    command: list[str]
    return_code: int
    elapsed_seconds: float
    stdout_path: str
    stderr_path: str


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_root: Path,
    name: str,
    timeout_seconds: float = 3600,
    environment: dict[str, str] | None = None,
) -> LoggedCommand:
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{name}.stdout.log"
    stderr_path = log_root / f"{name}.stderr.log"
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        env=environment,
    )
    elapsed = time.perf_counter() - started
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    evidence = LoggedCommand(
        command=command,
        return_code=result.returncode,
        elapsed_seconds=elapsed,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )
    _json_write(
        log_root / f"{name}.command.json",
        {
            "command": command,
            "cwd": str(cwd),
            "return_code": result.returncode,
            "elapsed_seconds": elapsed,
        },
    )
    return evidence


def _docker_mount(path: Path) -> str:
    return str(path.expanduser().resolve())


def validate_sglang_environment(
    root: Path,
    *,
    image: str,
    expected_version: str,
    repository_root: Path,
    pull_when_missing: bool,
) -> dict[str, Any]:
    environment_root = root / "sglang"
    logs = environment_root / "logs"
    environment_root.mkdir(parents=True, exist_ok=True)
    inspect = run_logged(
        ["docker", "image", "inspect", image],
        cwd=repository_root,
        log_root=logs,
        name="image-inspect",
        timeout_seconds=120,
    )
    pull: LoggedCommand | None = None
    if inspect.return_code != 0 and pull_when_missing:
        pull = run_logged(
            ["docker", "pull", image],
            cwd=repository_root,
            log_root=logs,
            name="image-pull",
            timeout_seconds=3600,
        )
        inspect = run_logged(
            ["docker", "image", "inspect", image],
            cwd=repository_root,
            log_root=logs,
            name="image-inspect-after-pull",
            timeout_seconds=120,
        )
    probe = run_logged(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--entrypoint",
            "python3",
            image,
            "-c",
            (
                "import json,platform,sglang,sys,torch; print(json.dumps({"
                "'sglang':getattr(sglang,'__version__',None),"
                "'torch':torch.__version__,'cuda':torch.version.cuda,"
                "'python':sys.version,'compiler':platform.python_compiler(),"
                "'gpu':torch.cuda.get_device_name(0),"
                "'capability':torch.cuda.get_device_capability(0)}))"
            ),
        ],
        cwd=repository_root,
        log_root=logs,
        name="runtime-probe",
        timeout_seconds=300,
    )
    probe_payload: dict[str, Any] = {}
    if probe.return_code == 0:
        lines = Path(probe.stdout_path).read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                probe_payload = parsed
                break
    image_payload: list[dict[str, Any]] = []
    if inspect.return_code == 0:
        raw = Path(inspect.stdout_path).read_text(encoding="utf-8")
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            image_payload = parsed
    image_id = str(image_payload[0].get("Id", "")) if image_payload else ""
    image_config = image_payload[0].get("Config", {}) if image_payload else {}
    labels = image_config.get("Labels", {}) if isinstance(image_config, dict) else {}
    labels = labels if isinstance(labels, dict) else {}
    source_commit = str(
        labels.get("ai.sglang.build.commit")
        or labels.get("org.opencontainers.image.revision")
        or image_id
    )
    evidence = {
        "backend_id": "sglang",
        "root": str(environment_root),
        "kind": "isolated-docker-image",
        "source_commit": source_commit,
        "package_or_build_version": probe_payload.get("sglang"),
        "compiler": probe_payload.get("compiler"),
        "python_version": probe_payload.get("python"),
        "cuda_version": probe_payload.get("cuda"),
        "build_command": ["docker", "pull", image],
        "launch_command": [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            image,
        ],
        "environment_variables": {},
        "installation_log": pull.stdout_path if pull is not None else inspect.stdout_path,
        "binary_hash": image_id.removeprefix("sha256:"),
        "image_id": image_id,
        "image_labels": labels,
        "image": image,
        "probe": probe_payload,
        "status": (
            "PASS"
            if inspect.return_code == 0
            and probe.return_code == 0
            and probe_payload.get("sglang") == expected_version
            else "FAIL"
        ),
    }
    _json_write(environment_root / "environment.json", evidence)
    return evidence


def resolve_git_commit(repository: str, requested_commit: str | None, cwd: Path) -> str:
    reference = requested_commit or "HEAD"
    result = subprocess.run(
        ["git", "ls-remote", repository, reference],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"cannot resolve llama.cpp revision {reference}: {result.stderr}")
    commit = result.stdout.split()[0]
    if len(commit) != 40:
        raise RuntimeError("llama.cpp remote did not return a full commit hash")
    return commit


def provision_llamacpp_environment(
    root: Path,
    *,
    repository: str,
    commit: str,
    repository_root: Path,
) -> dict[str, Any]:
    environment_root = root / "llamacpp"
    logs = environment_root / "logs"
    source = environment_root / f"source-{commit[:12]}"
    # Run the CPU backend in a minimal, network-isolated runtime container.
    # Disabling OpenMP avoids a runtime libgomp dependency; a distinct build
    # directory preserves any older build rather than overwriting it.
    build = environment_root / f"build-x86_64-noomp-{commit[:12]}"
    environment_root.mkdir(parents=True, exist_ok=True)
    commands: list[LoggedCommand] = []
    if not source.is_dir():
        commands.append(
            run_logged(
                ["git", "clone", "--filter=blob:none", repository, str(source)],
                cwd=repository_root,
                log_root=logs,
                name="clone",
                timeout_seconds=1800,
            )
        )
        if commands[-1].return_code != 0:
            raise RuntimeError("llama.cpp clone failed")
        commands.append(
            run_logged(
                ["git", "checkout", "--detach", commit],
                cwd=source,
                log_root=logs,
                name="checkout",
                timeout_seconds=300,
            )
        )
        if commands[-1].return_code != 0:
            raise RuntimeError("llama.cpp pinned checkout failed")
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != commit or dirty:
        raise RuntimeError("llama.cpp source is not a clean pinned checkout")
    server = build / "bin" / "llama-server"
    quantize = build / "bin" / "llama-quantize"
    build_command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{_docker_mount(environment_root)}:/work",
        "-w",
        f"/work/{source.name}",
        "ubuntu:24.04",
        "bash",
        "-lc",
        (
            "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "cmake ninja-build g++ git && "
            f"cmake -S . -B /work/{build.name} -G Ninja "
            "-DGGML_NATIVE=ON -DGGML_CUDA=OFF -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF "
            "-DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON && "
            f"cmake --build /work/{build.name} --config Release --target "
            "llama-server llama-quantize -j2"
        ),
    ]
    if not server.is_file() or not quantize.is_file():
        commands.append(
            run_logged(
                build_command,
                cwd=repository_root,
                log_root=logs,
                name="build-x86_64",
                timeout_seconds=7200,
            )
        )
        if commands[-1].return_code != 0:
            raise RuntimeError("llama.cpp isolated x86-64 build failed")
    version = run_logged(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{_docker_mount(environment_root)}:/work:ro",
            "ubuntu:24.04",
            f"/work/{build.name}/bin/llama-server",
            "--version",
        ],
        cwd=repository_root,
        log_root=logs,
        name="version",
        timeout_seconds=120,
    )
    version_text = (
        Path(version.stdout_path).read_text(encoding="utf-8").strip()
        or Path(version.stderr_path).read_text(encoding="utf-8").strip()
    )
    evidence = {
        "backend_id": "llamacpp",
        "root": str(environment_root),
        "kind": "isolated-source-build",
        "source_repository": repository,
        "source_commit": commit,
        "package_or_build_version": version_text,
        "compiler": "Ubuntu 24.04 g++",
        "python_version": None,
        "cuda_version": None,
        "build_command": build_command,
        "launch_command": [
            "docker",
            "run",
            "--rm",
            "--network",
            "none-by-default; publish localhost explicitly",
            "ubuntu:24.04",
            str(server),
        ],
        "environment_variables": {
            "OMP_NUM_THREADS": "measured at launch",
            "GGML_CUDA": "OFF",
        },
        "installation_log": commands[-1].stdout_path if commands else version.stdout_path,
        "binary_hash": sha256_file(server),
        "quantize_binary_hash": sha256_file(quantize),
        "server_path": str(server),
        "quantize_path": str(quantize),
        "source_path": str(source),
        "status": "PASS" if version.return_code == 0 else "FAIL",
    }
    _json_write(environment_root / "environment.json", evidence)
    return evidence


def validate_torch_cpu_environment(
    root: Path,
    *,
    repository_root: Path,
    python_executable: Path,
) -> dict[str, Any]:
    environment_root = root / "torch-cpu"
    logs = environment_root / "logs"
    environment_root.mkdir(parents=True, exist_ok=True)
    probe = run_logged(
        [
            str(python_executable),
            "-c",
            (
                "import json,platform,sys,torch,transformers; "
                "print(json.dumps({'python':sys.version,'platform':platform.platform(),"
                "'torch':torch.__version__,'transformers':transformers.__version__,"
                "'cuda_available':torch.cuda.is_available(),"
                "'mkldnn':torch.backends.mkldnn.is_available()}))"
            ),
        ],
        cwd=repository_root,
        log_root=logs,
        name="runtime-probe",
        timeout_seconds=300,
    )
    payload: dict[str, Any] = {}
    if probe.return_code == 0:
        payload = json.loads(Path(probe.stdout_path).read_text(encoding="utf-8").splitlines()[-1])
    executable_hash = sha256_file(python_executable)
    evidence = {
        "backend_id": "torch-cpu",
        "root": str(environment_root),
        "kind": "isolated-python-environment",
        "source_commit": None,
        "package_or_build_version": payload.get("torch"),
        "compiler": platform.python_compiler(),
        "python_version": payload.get("python", sys.version),
        "cuda_version": None,
        "build_command": ["python", "-m", "venv", str(environment_root / "venv")],
        "launch_command": [str(python_executable), "-m", "swarm_inference.worker"],
        "environment_variables": {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(repository_root / "src"),
        },
        "installation_log": probe.stdout_path,
        "binary_hash": executable_hash,
        "python_executable": str(python_executable),
        "probe": payload,
        "status": "PASS" if probe.return_code == 0 else "FAIL",
    }
    _json_write(environment_root / "environment.json", evidence)
    return evidence


def provision_torch_cpu_environment(
    root: Path,
    *,
    repository_root: Path,
    recreate: bool = False,
) -> dict[str, Any]:
    """Create an additive CPU-only rank environment without touching the project venv."""

    environment_root = root / "torch-cpu"
    logs = environment_root / "logs"
    venv = environment_root / "venv"
    python_executable = venv / "Scripts" / "python.exe"
    environment_root.mkdir(parents=True, exist_ok=True)
    if recreate and venv.exists():
        raise RuntimeError(
            "automatic destructive recreation is disabled; remove the isolated torch-cpu "
            "environment explicitly before requesting recreation"
        )
    if not python_executable.is_file():
        created = run_logged(
            [sys.executable, "-m", "venv", str(venv)],
            cwd=repository_root,
            log_root=logs,
            name="create-venv",
            timeout_seconds=600,
        )
        if created.return_code != 0:
            raise RuntimeError("isolated torch-cpu virtual environment creation failed")
    probe = subprocess.run(
        [str(python_executable), "-c", "import torch,pydantic,transformers,safetensors"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    install_command = [
        str(python_executable),
        "-m",
        "pip",
        "install",
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
        "--extra-index-url",
        "https://pypi.org/simple",
        "torch>=2.11",
        "numpy>=1.26",
        "pydantic>=2.7",
        "safetensors>=0.4",
        "transformers>=4.51,<5",
        "psutil>=5.9",
    ]
    if probe.returncode != 0:
        installed = run_logged(
            install_command,
            cwd=repository_root,
            log_root=logs,
            name="install",
            timeout_seconds=7200,
        )
        if installed.return_code != 0:
            raise RuntimeError("isolated torch-cpu dependency installation failed")
    evidence = validate_torch_cpu_environment(
        root,
        repository_root=repository_root,
        python_executable=python_executable,
    )
    evidence["build_command"] = [
        [sys.executable, "-m", "venv", str(venv)],
        install_command,
    ]
    evidence["isolated_from_project_environment"] = True
    evidence["cpu_only_required"] = True
    evidence["status"] = (
        "PASS"
        if evidence.get("status") == "PASS"
        and not bool(evidence.get("probe", {}).get("cuda_available"))
        else "FAIL"
    )
    _json_write(environment_root / "environment.json", evidence)
    return evidence


def conversion_sidecar(
    *,
    gguf_path: Path,
    source_model_id: str,
    source_revision: str,
    conversion_command: list[str],
    conversion_version: str,
    quantisation: str,
    tokenizer_identity: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "source_model_id": source_model_id,
        "source_revision": source_revision,
        "gguf_path": str(gguf_path.resolve()),
        "conversion_command": subprocess.list2cmdline(conversion_command),
        "conversion_version": conversion_version,
        "quantisation": quantisation,
        "gguf_sha256": sha256_file(gguf_path),
        "tokenizer_hash": tokenizer_identity["tokenizer_hash"],
        "vocabulary_hash": tokenizer_identity["vocabulary_hash"],
        "special_tokens_hash": tokenizer_identity["special_tokens_hash"],
        "tokenizer_identity": tokenizer_identity,
    }
    _json_write(gguf_path.with_suffix(gguf_path.suffix + ".conversion.json"), payload)
    return payload


def build_gguf_artifacts(
    *,
    repository_root: Path,
    environment_evidence: dict[str, Any],
    model_snapshot: Path,
    output_root: Path,
    model_id: str,
    revision: str,
    tokenizer_identity: dict[str, Any],
    sglang_image: str,
) -> dict[str, dict[str, Any]]:
    """Convert once to BF16 GGUF, then create pinned Q8_0 and Q4_K_M artifacts."""

    source = Path(str(environment_evidence["source_path"])).resolve()
    environment_root = Path(str(environment_evidence["root"])).resolve()
    build = Path(str(environment_evidence["server_path"])).resolve().parents[1]
    output_root.mkdir(parents=True, exist_ok=True)
    logs = environment_root / "logs"
    base = output_root / f"qwen3-0.6b-{revision[:12]}-bf16.gguf"
    # Normal Hugging Face snapshots use relative symlinks into ../../blobs.
    # Mount the complete model cache root so those links remain valid in Docker.
    if model_snapshot.parent.name == "snapshots":
        model_mount = model_snapshot.parent.parent
        model_path_in_container = f"/hf-model/snapshots/{model_snapshot.name}"
    else:
        model_mount = model_snapshot
        model_path_in_container = "/model"
    conversion_command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{source}:/llama:ro",
        "-v",
        f"{model_mount.resolve()}:/hf-model:ro"
        if model_path_in_container != "/model"
        else f"{model_mount.resolve()}:/model:ro",
        "-v",
        f"{output_root.resolve()}:/out",
        "--entrypoint",
        "python3",
        sglang_image,
        "/llama/convert_hf_to_gguf.py",
        model_path_in_container,
        "--outfile",
        f"/out/{base.name}",
        "--outtype",
        "bf16",
    ]
    if not base.is_file():
        conversion = run_logged(
            conversion_command,
            cwd=repository_root,
            log_root=logs,
            name="convert-qwen3-0.6b-bf16",
            timeout_seconds=7200,
        )
        if conversion.return_code != 0 or not base.is_file():
            raise RuntimeError("Qwen3-0.6B BF16 GGUF conversion failed")
    artifacts: dict[str, dict[str, Any]] = {}
    for format_name in ("Q8_0", "Q4_K_M"):
        destination = output_root / f"qwen3-0.6b-{revision[:12]}-{format_name.lower()}.gguf"
        quantize_command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{environment_root}:/work:ro",
            "-v",
            f"{output_root.resolve()}:/models",
            "ubuntu:24.04",
            f"/work/{build.name}/bin/llama-quantize",
            f"/models/{base.name}",
            f"/models/{destination.name}",
            format_name,
        ]
        if not destination.is_file():
            quantized = run_logged(
                quantize_command,
                cwd=repository_root,
                log_root=logs,
                name=f"quantize-{format_name.lower()}",
                timeout_seconds=3600,
            )
            if quantized.return_code != 0 or not destination.is_file():
                raise RuntimeError(f"Qwen3-0.6B {format_name} quantisation failed")
        sidecar = conversion_sidecar(
            gguf_path=destination,
            source_model_id=model_id,
            source_revision=revision,
            conversion_command=[*conversion_command, "&&", *quantize_command],
            conversion_version=str(environment_evidence["source_commit"]),
            quantisation=format_name,
            tokenizer_identity=tokenizer_identity,
        )
        sidecar["source_bf16_gguf_sha256"] = sha256_file(base)
        sidecar["source_bf16_gguf_path"] = str(base)
        _json_write(destination.with_suffix(destination.suffix + ".conversion.json"), sidecar)
        artifacts[format_name] = sidecar
    return artifacts


def stable_environment_hash(evidence: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def remove_stale_process_file(path: Path) -> None:
    """Remove only a zero-content stale PID marker; never clean arbitrary environments."""

    if path.is_file() and path.stat().st_size == 0:
        path.unlink()


def executable_available(name: str) -> bool:
    return shutil.which(name) is not None


def isolated_environment_variables(repository_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository_root / "src")
    return environment
