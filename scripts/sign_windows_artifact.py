"""Authenticode signing helpers for release CI; never logs credentials."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from release_common import ROOT, ReleaseError


def _redact_sensitive(text: str) -> str:
    redacted = text
    for name in (
        "WINDOWS_SIGNING_PFX_BASE64",
        "WINDOWS_SIGNING_PASSWORD",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    ):
        value = os.environ.get(name)
        if value:
            redacted = redacted.replace(value, "<redacted>")
    return redacted


def _run_sensitive(argv: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReleaseError("Authenticode operation exceeded its bounded timeout") from exc
    if result.returncode != 0:
        diagnostic = _redact_sensitive((result.stderr or result.stdout).strip())[-3000:]
        raise ReleaseError(f"Authenticode operation failed: {diagnostic}")
    return result


def _powershell() -> str:
    candidate = shutil.which("powershell.exe") or shutil.which("powershell")
    if candidate is None:
        raise ReleaseError("Windows PowerShell is unavailable for signature verification")
    return candidate


def _signtool(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.resolve()
    else:
        discovered = shutil.which("signtool.exe") or shutil.which("signtool")
        if discovered is None:
            kits = Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
            matches = sorted(
                (kits / "Windows Kits" / "10" / "bin").glob("10.0.26100.*/x64/signtool.exe")
            )
            matches = [path for path in matches if path.is_file()]
            discovered = str(matches[-1]) if matches else None
        if discovered is None:
            raise ReleaseError("signtool.exe is unavailable on the Windows signing runner")
        candidate = Path(discovered).resolve()
    if not candidate.is_file():
        raise ReleaseError(f"signtool.exe is missing: {candidate}")
    return candidate


def authenticode_info(path: Path) -> dict[str, Any]:
    literal_path = str(path.resolve()).replace("'", "''")
    command = (
        f"$signature=Get-AuthenticodeSignature -LiteralPath '{literal_path}' "
        "-ErrorAction SilentlyContinue; "
        "$certificate=if($signature){$signature.SignerCertificate}else{$null}; "
        "if(-not $certificate){try{$certificate="
        "[System.Security.Cryptography.X509Certificates.X509Certificate2]::new("
        "[System.Security.Cryptography.X509Certificates.X509Certificate]::"
        f"CreateFromSignedFile('{literal_path}'))"
        "}catch{$certificate=$null}}; "
        "$status=if($signature){[string]$signature.Status}"
        "elseif($certificate){'CertificateOnly'}else{''}; "
        "[ordered]@{status=$status; "
        "subject=if($certificate){$certificate.Subject}else{$null}; "
        "thumbprint=if($certificate){$certificate.Thumbprint}else{$null}; "
        "timestamp_subject=if($signature.TimeStamperCertificate)"
        "{$signature.TimeStamperCertificate.Subject}else{$null}} | ConvertTo-Json -Compress"
    )
    result = _run_sensitive(
        [_powershell(), "-NoProfile", "-NonInteractive", "-Command", command],
        timeout_seconds=60,
    )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ReleaseError("Authenticode verification returned malformed diagnostics") from exc
    if not isinstance(value, dict):
        raise ReleaseError("Authenticode verification returned no signature object")
    return value


def signing_environment_present() -> bool:
    names = (
        "WINDOWS_SIGNING_PFX_BASE64",
        "WINDOWS_SIGNING_PASSWORD",
        "WINDOWS_SIGNING_TIMESTAMP_URL",
    )
    present = [bool(os.environ.get(name)) for name in names]
    if any(present) and not all(present):
        raise ReleaseError(f"signing configuration is incomplete; set all of {', '.join(names)}")
    return all(present)


def sign_file(path: Path, *, signtool: Path | None = None) -> dict[str, Any]:
    if not signing_environment_present():
        raise ReleaseError("Authenticode signing secrets are unavailable")
    timestamp_url = os.environ["WINDOWS_SIGNING_TIMESTAMP_URL"]
    if not timestamp_url.startswith("https://"):
        raise ReleaseError("WINDOWS_SIGNING_TIMESTAMP_URL must use HTTPS")
    try:
        pfx = base64.b64decode(os.environ["WINDOWS_SIGNING_PFX_BASE64"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ReleaseError("WINDOWS_SIGNING_PFX_BASE64 is not valid base64") from exc
    if not pfx:
        raise ReleaseError("decoded signing certificate is empty")
    (ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="swarm-signing-", dir=ROOT / ".tmp") as raw:
        pfx_path = Path(raw) / "publisher.pfx"
        pfx_path.write_bytes(pfx)
        try:
            _run_sensitive(
                [
                    str(_signtool(signtool)),
                    "sign",
                    "/fd",
                    "sha256",
                    "/td",
                    "sha256",
                    "/tr",
                    timestamp_url,
                    "/f",
                    str(pfx_path),
                    "/p",
                    os.environ["WINDOWS_SIGNING_PASSWORD"],
                    str(path.resolve()),
                ],
                timeout_seconds=180,
            )
        finally:
            pfx_path.unlink(missing_ok=True)
    information = authenticode_info(path)
    if information.get("status") != "Valid" or not information.get("subject"):
        raise ReleaseError("signed file failed Authenticode verification")
    if not information.get("timestamp_subject"):
        raise ReleaseError("signed file has no verified timestamp certificate")
    return information


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--signtool", type=Path)
    arguments = parser.parse_args(argv)
    try:
        information = sign_file(arguments.path.resolve(), signtool=arguments.signtool)
    except (OSError, ReleaseError) as exc:
        print(json.dumps({"status": "FAIL", "detail": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps({"status": "PASS", **information}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
