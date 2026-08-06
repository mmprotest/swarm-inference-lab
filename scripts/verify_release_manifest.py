"""Validate strict release-manifest structure and optional payload hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from release_common import ReleaseError, read_json_object, validate_manifest, verify_manifest_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--payload-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        manifest = validate_manifest(read_json_object(arguments.manifest))
        if arguments.payload_dir is not None:
            verify_manifest_files(manifest, arguments.payload_dir.resolve())
    except (OSError, ReleaseError) as exc:
        print(json.dumps({"status": "FAIL", "detail": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "version": manifest["version"],
                "git_tag": manifest["git_tag"],
                "scope": manifest["manifest_scope"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
