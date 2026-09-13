"""Prepare the ignored pinned llama.cpp worktree for the E027 native build."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


PINNED_LLAMA_COMMIT = "f1b6fbf35cfa010b0a8d6301fdfccbb7f41bd903"


def _run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def prepare(repo_root: Path, llama_root: Path) -> None:
    repo_root = repo_root.resolve()
    llama_root = llama_root.resolve()
    if not (llama_root / ".git").exists():
        raise FileNotFoundError(f"not a llama.cpp worktree: {llama_root}")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=llama_root, text=True
    ).strip()
    if commit != PINNED_LLAMA_COMMIT:
        raise RuntimeError(f"expected llama.cpp {PINNED_LLAMA_COMMIT}, found {commit}")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=llama_root, text=True
    ).strip():
        raise RuntimeError("E027 llama.cpp worktree must be clean before preparation")

    native = repo_root / "native" / "experiment_027"
    _run("git", "apply", "--check", str(native / "qwen35-stage-range.patch"), cwd=llama_root)
    _run("git", "apply", str(native / "qwen35-stage-range.patch"), cwd=llama_root)
    destination = llama_root / "examples" / "e027-stage"
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(native / "e027_stage_server.cpp", destination / "e027-stage.cpp")
    shutil.copy2(native / "CMakeLists.txt", destination / "CMakeLists.txt")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--llama-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repo_root, args.llama_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
