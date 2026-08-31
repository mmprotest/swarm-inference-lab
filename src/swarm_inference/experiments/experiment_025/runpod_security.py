"""Secret-leak validation for retained E025 RunPod preparation artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .io import atomic_write_json, sha256_file, utc_now
from .providers.runpod import SECRET_ENVIRONMENT_KEYS

PRIVATE_KEY_PATTERN = re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----")
RUNPOD_KEY_PATTERN = re.compile(r"\brpa_[A-Za-z0-9_-]{20,}\b")
BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*\b")
ALLOWED_SECRET_STRINGS = frozenset(
    {
        "<EPHEMERAL>",
        "<REDACTED>",
        "EPHEMERAL_IN_MEMORY_AT_PAID_CREATE",
    }
)


def _secret_key(key: str) -> bool:
    upper = key.upper()
    return (
        upper in SECRET_ENVIRONMENT_KEYS
        or any(
            marker in upper for marker in ("API_KEY", "CREDENTIAL_B64", "TLS_KEY_B64", "PASSWORD")
        )
        or upper in {"TOKEN", "AUTH_TOKEN", "ACCESS_TOKEN", "REFRESH_TOKEN"}
        or upper.endswith(("_AUTH_TOKEN", "_ACCESS_TOKEN", "_REFRESH_TOKEN"))
    )


def _inspect_json(value: Any, *, path: str, findings: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            location = f"{path}.{key}"
            if (
                _secret_key(str(key))
                and isinstance(item, str)
                and item not in ALLOWED_SECRET_STRINGS
                and not item.startswith(("EPHEMERAL_RUNTIME:", "{{ RUNPOD_SECRET_"))
            ):
                findings.append(
                    {
                        "kind": "RAW_SECRET_VALUE",
                        "location": location,
                        "value_bytes": len(item.encode("utf-8")),
                    }
                )
            _inspect_json(item, path=location, findings=findings)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _inspect_json(item, path=f"{path}[{index}]", findings=findings)


def scan_runpod_artifacts(output_directory: Path) -> dict[str, Any]:
    root = output_directory.resolve()
    findings: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "runpod-secret-scan.json":
            continue
        if path.suffix.lower() not in {".json", ".md", ".jsonl", ".ps1"}:
            continue
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for kind, pattern in (
            ("PRIVATE_KEY_PEM", PRIVATE_KEY_PATTERN),
            ("RUNPOD_API_KEY_PATTERN", RUNPOD_KEY_PATTERN),
            ("BEARER_TOKEN", BEARER_PATTERN),
        ):
            if pattern.search(text):
                findings.append({"kind": kind, "location": relative})
        if path.suffix.lower() == ".json":
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                findings.append(
                    {
                        "kind": "INVALID_JSON",
                        "location": relative,
                        "detail": str(exc),
                    }
                )
            else:
                _inspect_json(value, path=relative, findings=findings)
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    receipt = {
        "schema_version": "experiment-025-runpod-secret-scan-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if not findings else "FAIL",
        "artifact_root": str(root),
        "files_scanned": len(rows),
        "findings": findings,
        "checks": {
            "private_key_pem_absent": not any(row["kind"] == "PRIVATE_KEY_PEM" for row in findings),
            "runpod_api_key_absent": not any(
                row["kind"] == "RUNPOD_API_KEY_PATTERN" for row in findings
            ),
            "bearer_tokens_absent": not any(row["kind"] == "BEARER_TOKEN" for row in findings),
            "secret_environment_values_redacted": not any(
                row["kind"] == "RAW_SECRET_VALUE" for row in findings
            ),
        },
        "files": rows,
    }
    atomic_write_json(root / "runpod-secret-scan.json", receipt)
    return receipt


__all__ = ["scan_runpod_artifacts"]
