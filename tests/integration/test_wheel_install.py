from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_built_wheel_installs_without_importing_checkout(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    uv = os.environ.get("UV") or shutil.which("uv")
    assert uv is not None, "the supported acceptance workflow requires the uv executable"
    distribution = tmp_path / "dist"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["UV_CACHE_DIR"] = str(repository / ".uv-cache")
    subprocess.run(
        [uv, "build", "--offline", "--wheel", "--out-dir", str(distribution)],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    wheels = list(distribution.glob("*.whl"))
    assert len(wheels) == 1
    output = tmp_path / "wheel-install.json"
    scratch_root = Path(
        os.environ.get("RUNNER_TEMP") or repository / ".tmp" / "wheel-smoke"
    ).resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    dependency_site = Path(sys.prefix) / ("Lib/site-packages" if os.name == "nt" else "lib")
    if os.name != "nt":
        dependency_site = next((Path(sys.prefix) / "lib").glob("python*/site-packages"))
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "test_wheel_install.py"),
            "--uv",
            uv,
            "--wheel",
            str(wheels[0]),
            "--python",
            sys.executable,
            "--extra",
            "cpu",
            "--output",
            str(output),
            "--scratch-root",
            str(scratch_root),
            "--dependency-site-packages",
            str(dependency_site),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS", result.stderr
    assert evidence["checkout_imported"] is False
    module = Path(evidence["module"])
    assert not module.is_relative_to(repository / "src")
    assert module.is_relative_to(Path(evidence["environment_root"]) / "environment")
