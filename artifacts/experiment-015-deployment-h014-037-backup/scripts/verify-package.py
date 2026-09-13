#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
lock_path = root / "package-lock.json"
lock = json.loads(lock_path.read_text(encoding="utf-8"))
entries = lock.get("files")
if lock.get("status") != "PASS" or not isinstance(entries, dict):
    raise SystemExit("package lock is not passing")
actual = sorted(
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path.name != "package-lock.json"
)
if actual != sorted(entries):
    raise SystemExit("package file set differs from package lock")
for relative, identity in entries.items():
    path = root / relative
    if path.is_symlink() or path.stat().st_size != int(identity["bytes"]):
        raise SystemExit(f"package size/link identity differs: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != identity["sha256"]:
        raise SystemExit(f"package SHA-256 differs: {relative}")
print(json.dumps({"status": "PASS", "file_count": len(entries) + 1}, sort_keys=True))
