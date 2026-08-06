"""Detection of the native Windows application installation boundary."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def native_install_record() -> tuple[Path, dict[str, Any]] | None:
    """Return a validated native install root and record for this interpreter."""

    if not sys.platform.startswith("win"):
        return None
    executable = Path(sys.executable).resolve()
    candidates: list[Path] = []
    if (
        executable.parent.name.casefold() == "scripts"
        and executable.parent.parent.name.casefold() == "runtime"
    ):
        candidates.append(executable.parent.parent.parent)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data).resolve() / "Programs" / "SwarmInference")
    seen: set[Path] = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        record_path = root / "app" / "install-record.json"
        try:
            document = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(document, dict)
            and document.get("schema_version") == 1
            and document.get("installation_mode") == "native-windows"
            and Path(str(document.get("application_path", ""))).resolve() == root
            and (root / "runtime" / "Scripts" / "python.exe").is_file()
        ):
            return root, document
    return None


__all__ = ["native_install_record"]
