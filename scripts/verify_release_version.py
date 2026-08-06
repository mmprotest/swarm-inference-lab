"""Validate the single package version against an immutable Git release tag."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from release_common import (
    ReleaseError,
    atomic_write_json,
    git_commit,
    pep440_to_git_tag,
    read_pyproject_version,
    verify_release_identity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag", help="Release tag; defaults to GITHUB_REF_NAME or version mapping."
    )
    parser.add_argument("--commit", help="Expected 40-character commit; defaults to HEAD.")
    parser.add_argument("--output", type=Path, help="Optional JSON evidence path.")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Development-only: validate mapping while permitting local source changes.",
    )
    parser.add_argument(
        "--allow-untagged",
        action="store_true",
        help="Development-only: do not require the mapped tag to exist locally.",
    )
    return parser


def default_release_tag(version: str) -> str:
    """Use a tag ref in CI, but never mistake a branch name for a release tag."""

    github_ref_name = os.environ.get("GITHUB_REF_NAME")
    if github_ref_name and github_ref_name.startswith("v"):
        return github_ref_name
    return pep440_to_git_tag(version)


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    version = read_pyproject_version()
    tag = arguments.tag or default_release_tag(version)
    commit = (arguments.commit or git_commit()).lower()
    try:
        evidence = verify_release_identity(
            version=version,
            tag=tag,
            commit=commit,
            require_clean=not arguments.allow_dirty,
            require_tag=not arguments.allow_untagged,
        )
    except ReleaseError as exc:
        print(json.dumps({"status": "FAIL", "detail": str(exc)}, sort_keys=True))
        return 1
    if arguments.output is not None:
        atomic_write_json(arguments.output, evidence)
    print(json.dumps({"status": "PASS", **evidence}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
