"""Create deterministic Colibri source and binary build fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    excluded = {".git", "__pycache__"}
    generated_names = {
        ".build-config",
        "colibri",
        "colibri.exe",
        "glm",
        "glm.exe",
        "inkling",
        "inkling.exe",
        "kimi_k3",
        "kimi_k3.exe",
        "olmoe",
        "olmoe.exe",
    }
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not excluded.intersection(path.parts)
        and path.name not in generated_names
        and path.suffix not in {".o", ".obj", ".pyc"}
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def command_version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).splitlines()
    return output[0].strip() if output else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--patches", nargs="*", default=[])
    args = parser.parse_args()
    binaries = []
    for path in sorted(item for item in args.bin.iterdir() if item.is_file()):
        binaries.append({"name": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {
        "schema_version": "experiment-009-colibri-build-v1",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "repository": "JustVugg/colibri",
        "release": "v1.4.0",
        "commit": args.commit,
        "license": "Apache-2.0",
        "source_tree_sha256": tree_hash(args.source),
        "patches": args.patches,
        "platform": platform.platform(),
        "compiler": command_version([os.environ.get("CC", "gcc"), "--version"]),
        "binaries": binaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
