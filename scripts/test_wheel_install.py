"""Install a built wheel into a fresh venv and prove checkout-independent import."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def _run(
    arguments: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            cwd=cwd,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "")[-4000:]
        stdout = (exc.stdout or "")[-2000:]
        raise RuntimeError(
            f"wheel-smoke command failed with exit {exc.returncode}: "
            f"{arguments[0]} {arguments[1]}; stdout={stdout!r}; stderr={stderr!r}"
        ) from exc


def _venv_python(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _progress(stage: str) -> None:
    print(f"wheel-smoke-stage={stage}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--python", default="3.11")
    parser.add_argument("--extra", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument(
        "--dependency-site-packages",
        type=Path,
        help=(
            "Use an already validated dependency environment while still installing "
            "the product wheel into a fresh venv. Intended for repeated test suites."
        ),
    )
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    wheel = arguments.wheel
    if wheel is None:
        values = sorted((repository / "dist").glob("*.whl"))
        if len(values) != 1:
            raise SystemExit(f"expected exactly one wheel under {repository / 'dist'}")
        wheel = values[0]
    wheel = wheel.expanduser().resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise SystemExit(f"built wheel is unavailable: {wheel}")
    started = time.monotonic()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["UV_HTTP_TIMEOUT"] = "120"
    scratch_root = (
        arguments.scratch_root.expanduser().resolve()
        if arguments.scratch_root is not None
        else Path(tempfile.gettempdir()).resolve()
    )
    scratch_root.mkdir(parents=True, exist_ok=True)
    if scratch_root.is_relative_to(repository / "src"):
        raise SystemExit("wheel smoke scratch root must not be inside the checkout source tree")
    # Python's recursive TemporaryDirectory cleanup can spend many minutes on
    # Windows venv junctions. A single fixed slot recreated by uv --clear is
    # bounded in storage and fresh for every invocation.
    environment_root = scratch_root / "swarm-wheel-validation"
    with nullcontext(str(environment_root)) as raw:
        root = Path(raw).resolve()
        environment_directory = root / "environment"
        _progress("creating-environment")
        _run(
            [
                arguments.uv,
                "venv",
                "--clear",
                str(environment_directory),
                "--python",
                arguments.python,
            ],
            cwd=scratch_root,
            environment=environment,
            timeout=900,
        )
        dependency_mode = "isolated-resolve"
        if arguments.dependency_site_packages is not None:
            dependency_site = arguments.dependency_site_packages.expanduser().resolve()
            if not dependency_site.is_dir():
                raise SystemExit(f"dependency site-packages is unavailable: {dependency_site}")
            site_result = _run(
                [
                    str(_venv_python(environment_directory)),
                    "-I",
                    "-c",
                    "import site; print(site.getsitepackages()[0])",
                ],
                cwd=root,
                environment=environment,
                timeout=60,
            )
            venv_site = Path(site_result.stdout.strip()).resolve()
            venv_site.mkdir(parents=True, exist_ok=True)
            (venv_site / "swarm-validated-dependencies.pth").write_text(
                f"{dependency_site}\n",
                encoding="utf-8",
            )
            dependency_mode = "prevalidated-site-packages"
        selected = f"{wheel}[{arguments.extra}]"
        install = [
            arguments.uv,
            "pip",
            "install",
            "--python",
            str(_venv_python(environment_directory)),
        ]
        if arguments.extra == "cpu":
            install.extend(("--torch-backend", "cpu"))
        elif arguments.extra == "cuda":
            install.extend(("--torch-backend", "cu130"))
        else:
            install.extend(("--torch-backend", "auto"))
        if arguments.dependency_site_packages is not None:
            install.append("--no-deps")
        install.append(selected)
        _progress("installing-wheel")
        installed = _run(
            install,
            cwd=root,
            environment=environment,
            timeout=1800,
        )
        probe_code = (
            "import json,pathlib,sys,swarm_inference; "
            "import swarm_inference.commands.cluster,swarm_inference.commands.node; "
            "paths=[str(pathlib.Path(getattr(m,'__file__','')).resolve()) for m in "
            "sys.modules.values() if getattr(m,'__file__',None)]; "
            "print(json.dumps({'module':str(pathlib.Path(swarm_inference.__file__).resolve()),"
            "'version':swarm_inference.__version__,"
            "'third_party_colibri_imports':[p for p in paths if '/third_party/colibri/' in "
            "p.replace('\\\\','/').lower()],"
            "'experiment_imports':sorted(n for n in sys.modules if "
            "n.startswith('swarm_inference.experiments'))},sort_keys=True))"
        )
        _progress("verifying-import")
        imported = _run(
            [str(_venv_python(environment_directory)), "-I", "-c", probe_code],
            cwd=root,
            environment=environment,
            timeout=60,
        )
        import_evidence = json.loads(imported.stdout)
        module_path = Path(str(import_evidence["module"])).resolve()
        if module_path.is_relative_to(repository / "src"):
            raise RuntimeError("wheel smoke imported source code from the checkout")
        if not module_path.is_relative_to(environment_directory):
            raise RuntimeError("wheel smoke did not import from the fresh validation venv")
        if import_evidence["third_party_colibri_imports"]:
            raise RuntimeError("wheel runtime imported the source-only Colibri checkout")
        if import_evidence["experiment_imports"]:
            raise RuntimeError("cluster product packages imported swarm_inference.experiments")
        with zipfile.ZipFile(wheel) as archive:
            wheel_colibri_sources = [
                name
                for name in archive.namelist()
                if name.lower().startswith("third_party/colibri/")
            ]
        if wheel_colibri_sources:
            raise RuntimeError("wheel contains source-only third_party/colibri files")
        _progress("running-doctor")
        help_result = _run(
            [
                str(_venv_python(environment_directory)),
                "-I",
                "-m",
                "swarm_inference.cli",
                "node",
                "doctor",
                "--json",
                "--state-root",
                str(root / "state"),
            ],
            cwd=root,
            environment=environment,
            timeout=180,
        )
        doctor = json.loads(help_result.stdout)
        result: dict[str, Any] = {
            "schema_version": 2,
            "status": "PASS",
            "wheel": str(wheel),
            "wheel_size_bytes": wheel.stat().st_size,
            "python": arguments.python,
            "extra": arguments.extra,
            "dependency_mode": dependency_mode,
            "module": str(module_path),
            "environment_root": str(environment_root),
            "checkout_imported": False,
            "third_party_colibri_imported": False,
            "experiments_imported": False,
            "wheel_contains_third_party_colibri": False,
            "doctor": doctor,
            "install_stdout_tail": installed.stdout[-2000:],
            "elapsed_seconds": time.monotonic() - started,
        }
    _progress("writing-evidence")
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, output)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
