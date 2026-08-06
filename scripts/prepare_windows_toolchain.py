"""Download and verify the repository-pinned native Windows build tools."""

from __future__ import annotations

import argparse
import json
import subprocess

from build_windows_installer import _verified_dotnet, _verified_iscc, _verified_uv
from release_common import ReleaseError, load_toolchain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tools",
        choices=("uv", "all"),
        default="all",
        help="Fetch only uv for quality checks, or the complete native build toolchain.",
    )
    arguments = parser.parse_args(argv)
    try:
        toolchain = load_toolchain()
        uv = _verified_uv(toolchain, None)
        result = {"uv": str(uv)}
        if arguments.tools == "all":
            result.update(
                dotnet=str(_verified_dotnet(toolchain, None)),
                iscc=str(_verified_iscc(toolchain, None)),
            )
    except (OSError, ReleaseError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "FAIL", "detail": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps({"status": "PASS", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
