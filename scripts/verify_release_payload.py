"""Verify release assets against both the manifest and SHA256SUMS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from release_common import (
    ReleaseError,
    manifest_file_entries,
    parse_sha256sums,
    read_json_object,
    sha256_file,
    validate_manifest,
)


def verify_payload(
    *,
    manifest_path: Path,
    payload_directories: list[Path],
    checksums_path: Path | None,
) -> dict[str, object]:
    manifest = validate_manifest(read_json_object(manifest_path))
    verified: list[str] = []
    for entry in manifest_file_entries(manifest):
        filename = entry["filename"]
        if manifest["manifest_scope"] == "embedded-payload" and entry["sha256"].endswith("0" * 64):
            continue
        matches = [
            directory / filename
            for directory in payload_directories
            if (directory / filename).is_file()
        ]
        if not matches:
            raise ReleaseError(f"expected at least one payload file named {filename}, found none")
        for path in matches:
            if sha256_file(path) != entry["sha256"]:
                raise ReleaseError(f"release manifest hash mismatch for {filename}")
            if path.stat().st_size != entry["size_bytes"]:
                raise ReleaseError(f"release manifest size mismatch for {filename}")
        verified.append(filename)
    if checksums_path is not None:
        checksums = parse_sha256sums(checksums_path)
        for filename, expected in checksums.items():
            if filename == manifest_path.name:
                matches = [manifest_path]
            else:
                matches = [
                    directory / filename
                    for directory in payload_directories
                    if (directory / filename).is_file()
                ]
            matches = list(dict.fromkeys(path.resolve() for path in matches))
            if not matches or any(sha256_file(path) != expected for path in matches):
                raise ReleaseError(f"SHA256SUMS verification failed for {filename}")
    return {"version": manifest["version"], "verified_files": sorted(set(verified))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("release/generated/release-manifest.json")
    )
    parser.add_argument("--payload-dir", type=Path, action="append")
    parser.add_argument("--checksums", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    directories = arguments.payload_dir or [arguments.manifest.parent]
    try:
        result = verify_payload(
            manifest_path=arguments.manifest.resolve(),
            payload_directories=[path.resolve() for path in directories],
            checksums_path=arguments.checksums.resolve() if arguments.checksums else None,
        )
    except (OSError, ReleaseError) as exc:
        print(json.dumps({"status": "FAIL", "detail": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps({"status": "PASS", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
