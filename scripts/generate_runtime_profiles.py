"""Export exact hashed Windows CPU and CUDA dependency profiles from uv.lock."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

from release_common import (
    ROOT,
    ReleaseError,
    atomic_write_text,
    load_toolchain,
    run_captured,
    sha256_file,
)

PROFILE_NAMES = {
    "cpu": "windows-x64-cpu.requirements.lock",
    "cuda": "windows-x64-cuda.requirements.lock",
}
PYTORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda": "https://download.pytorch.org/whl/cu130",
}
DEVELOPMENT_PACKAGES = {
    "coverage",
    "hypothesis",
    "mypy",
    "mypy-extensions",
    "pytest",
    "pytest-asyncio",
    "pytest-cov",
    "ruff",
    "types-protobuf",
    "types-psutil",
    "types-pyyaml",
}


def _normalise_export(raw: str, backend: str) -> str:
    lines = raw.replace("\r\n", "\n").splitlines()
    body = [
        line
        for line in lines
        if not line.startswith("--index-url ")
        and not line.startswith("--extra-index-url ")
        and not line.startswith("#")
    ]
    while body and not body[0].strip():
        body.pop(0)
    text = (
        "\n".join(
            [
                "# Generated deterministically from uv.lock by uv 0.12.0.",
                f"# Target: Windows x86-64, CPython 3.11, backend={backend}.",
                "--require-hashes",
                # uv gives --extra-index-url higher priority than --index-url.
                # PyPI must therefore be the extra index; the exact +cpu/+cu130
                # torch versions fall through to their dedicated lower-priority index.
                f"--index-url {PYTORCH_INDEXES[backend]}",
                "--extra-index-url https://pypi.org/simple",
                "",
                *body,
            ]
        ).rstrip()
        + "\n"
    )
    lowered = text.lower()
    forbidden = ("-e ", "--editable", "file://", "../", "\\src\\", "/src/")
    if any(token in lowered for token in forbidden):
        raise ReleaseError(f"{backend} runtime profile contains a checkout or editable source")
    requirement_names = {
        match.group(1).lower().replace("_", "-")
        for match in re.finditer(r"(?m)^([A-Za-z0-9_.-]+)==", text)
    }
    leaked = sorted(requirement_names & DEVELOPMENT_PACKAGES)
    if leaked:
        raise ReleaseError(f"{backend} runtime profile contains development packages: {leaked}")
    blocks = re.split(r"(?m)(?=^[A-Za-z0-9_.-]+==)", text)
    unhashed = [
        block.split("==", 1)[0]
        for block in blocks
        if "==" in block and "--hash=sha256:" not in block
    ]
    if unhashed:
        raise ReleaseError(f"{backend} runtime profile has unhashed requirements: {unhashed}")
    if "swarm-inference-lab==" in lowered:
        raise ReleaseError("runtime profiles must not install the application wheel")
    if backend == "cpu":
        if "+cpu" not in lowered or "/whl/cpu" not in lowered or "/whl/cu130" in lowered:
            raise ReleaseError("CPU profile does not exclusively select the PyTorch CPU index")
        if "triton-windows==" in lowered:
            raise ReleaseError("CPU profile unexpectedly contains triton-windows")
    else:
        if "+cu130" not in lowered or "/whl/cu130" not in lowered:
            raise ReleaseError("CUDA profile does not select the pinned cu130 PyTorch index")
        if "triton-windows==" not in lowered:
            raise ReleaseError("CUDA profile is missing the pinned Windows Triton dependency")
    return text


def generate_profiles(*, uv: Path, output_directory: Path) -> dict[str, dict[str, object]]:
    toolchain = load_toolchain()
    uv_metadata = toolchain["uv"]
    if not isinstance(uv_metadata, dict):
        raise ReleaseError("toolchain uv metadata is malformed")
    expected_hash = uv_metadata["executable_sha256"]
    if sha256_file(uv) != expected_hash:
        raise ReleaseError("uv executable does not match the repository-controlled SHA-256 pin")
    version = run_captured([str(uv), "--version"], timeout_seconds=30).stdout.strip()
    if version.split()[1] != uv_metadata["version"]:
        raise ReleaseError(f"unexpected uv version output: {version}")
    output_directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "UV_CACHE_DIR": str((ROOT / ".uv-cache").resolve()),
            "UV_NO_SYSTEM_CONFIG": "1",
            "UV_PYTHON_DOWNLOADS": "never",
        }
    )
    summary: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(
        prefix="swarm-runtime-profiles-", dir=ROOT / ".tmp"
    ) as raw_root:
        temporary_root = Path(raw_root)
        for backend, filename in PROFILE_NAMES.items():
            raw_path = temporary_root / filename
            command = [
                str(uv),
                "export",
                "--locked",
                "--no-dev",
                "--extra",
                backend,
                "--no-emit-project",
                "--no-emit-local",
                "--no-annotate",
                "--no-header",
                "--emit-index-url",
                "--format",
                "requirements.txt",
                "--output-file",
                str(raw_path),
            ]
            run_captured(command, timeout_seconds=120, environment=environment)
            normalised = _normalise_export(raw_path.read_text(encoding="utf-8"), backend)
            destination = output_directory / filename
            atomic_write_text(destination, normalised)
            summary[backend] = {
                "filename": filename,
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uv", type=Path, required=True, help="Pinned, verified uv.exe path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "release/generated/payload",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        summary = generate_profiles(
            uv=arguments.uv.resolve(), output_directory=arguments.output_dir.resolve()
        )
    except (OSError, ReleaseError) as exc:
        print(json.dumps({"status": "FAIL", "detail": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps({"status": "PASS", "profiles": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
