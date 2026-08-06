from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


def _msys_path(path: Path) -> str:
    resolved = path.resolve()
    value = resolved.as_posix()
    if len(value) >= 3 and value[1:3] == ":/":
        return f"/{value[0].lower()}{value[2:]}"
    return value


@pytest.mark.skipif(os.name != "nt", reason="PowerShell installer is the Windows path")
def test_windows_installer_defers_service_until_cluster_membership(tmp_path: Path) -> None:
    uv_log = tmp_path / "uv-calls.log"
    fake_uv = tmp_path / "uv.cmd"
    tool_bin = tmp_path / "tool-bin"
    tool_bin.mkdir()
    source = tmp_path / "doctor.cs"
    source.write_text(
        "using System; public class Doctor { public static int Main(string[] args) { "
        'Console.WriteLine("{\\"status\\":\\"pass\\",'
        '\\"backend_selection\\":{\\"selected_backend\\":\\"torch-cpu\\"}}"); '
        "return 0; } }\n",
        encoding="utf-8",
    )
    compiler = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
    if not compiler.is_file():
        pytest.skip("C# compiler is unavailable for the bounded doctor fixture")
    compiled = subprocess.run(
        [str(compiler), "/nologo", f"/out:{tool_bin / 'swarm.exe'}", str(source)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compiled.returncode == 0, compiled.stderr or compiled.stdout
    fake_uv.write_text(
        "@echo off\r\n"
        f'>>"{uv_log}" echo %*\r\n'
        'if "%~1"=="tool" if "%~2"=="dir" (\r\n'
        f"  echo {tool_bin}\r\n"
        ")\r\n"
        "exit /b 0\r\n",
        encoding="utf-8",
    )
    wheel = tmp_path / "swarm_inference_lab-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"installer contract fixture")
    environment = dict(os.environ)
    environment.update(
        {
            "LOCALAPPDATA": str(tmp_path / "local"),
            "TEMP": str(tmp_path),
            "TMP": str(tmp_path),
        }
    )
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(Path("scripts/install.ps1").resolve()),
            "-SourceWheel",
            str(wheel),
            "-UvPath",
            str(fake_uv),
            "-InstallService",
            "-Json",
        ],
        cwd=Path.cwd(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS"
    assert payload["service"] == "deferred-until-cluster-create-or-join"
    assert payload["install_service_preference"] == "requested-deferred"
    calls = uv_log.read_text(encoding="utf-8")
    assert "tool install" in calls
    assert "node install-service" not in calls


@pytest.mark.skipif(os.name != "nt", reason="Git sh fixture is used on Windows CI")
def test_unix_installer_defers_service_without_invoking_unpaired_command(tmp_path: Path) -> None:
    shell = Path(r"C:\Program Files\Git\usr\bin\sh.exe")
    if not shell.is_file():
        pytest.skip("Git sh is unavailable")
    fake_bin = tmp_path / "fake-bin"
    tool_bin = tmp_path / "tool-bin"
    fake_bin.mkdir()
    tool_bin.mkdir()
    calls = tmp_path / "uv-calls.log"
    python = _msys_path(Path(".venv/Scripts/python.exe"))
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> '{_msys_path(calls)}'\n"
        'if [ "${1:-}" = tool ] && [ "${2:-}" = dir ]; then\n'
        f"  printf '%s\\n' '{_msys_path(tool_bin)}'\n"
        "  exit 0\n"
        "fi\n"
        'if [ "${1:-}" = run ]; then\n'
        "  shift 3\n"
        '  [ "${1:-}" = python ] && shift\n'
        f"  exec '{python}' \"$@\"\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
        newline="\n",
    )
    uname = fake_bin / "uname"
    uname.write_text(
        "#!/bin/sh\n"
        'if [ "${1:-}" = -s ]; then echo Linux; '
        'elif [ "${1:-}" = -m ]; then echo x86_64; else echo Linux; fi\n',
        encoding="utf-8",
        newline="\n",
    )
    swarm = tool_bin / "swarm"
    swarm.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' "
        '\'{"status":"pass","backend_selection":{"selected_backend":"torch-cpu"}}\'\n',
        encoding="utf-8",
        newline="\n",
    )
    for executable in (uv, uname, swarm):
        executable.chmod(0o700)
    wheel = tmp_path / "swarm_inference_lab-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"installer contract fixture")
    environment = dict(os.environ)
    environment.update(
        {
            "PATH": f"{_msys_path(fake_bin)}:/usr/bin:/bin",
            "TMPDIR": _msys_path(tmp_path),
        }
    )
    completed = subprocess.run(
        [
            str(shell),
            _msys_path(Path("scripts/install.sh")),
            "--source-wheel",
            _msys_path(wheel),
            "--uv-path",
            _msys_path(uv),
            "--install-service",
            "--json",
        ],
        cwd=Path.cwd(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS"
    assert payload["service"] == "deferred-until-cluster-create-or-join"
    assert payload["install_service_preference"] == "requested-deferred"
    observed = calls.read_text(encoding="utf-8")
    assert "tool install" in observed
    assert "node install-service" not in observed
