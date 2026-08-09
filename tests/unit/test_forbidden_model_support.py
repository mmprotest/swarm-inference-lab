from __future__ import annotations

import subprocess
from pathlib import Path


def test_forbidden_model_family_is_absent_from_active_repository_surfaces() -> None:
    forbidden = (
        "ol" + "moe",
        "ol" + "mo-e",
        "ol" + "mo_moe",
        "ol" + "mo moe",
    )
    active_roots = {
        ".github",
        "benchmarks",
        "configs",
        "docs",
        "examples",
        "installer",
        "integrations",
        "release",
        "scripts",
        "src",
        "tests",
    }
    active_files = {"pyproject.toml", "README.md", "uv.lock"}
    text_suffixes = {
        ".cfg",
        ".ini",
        ".json",
        ".md",
        ".ps1",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=False,
    )
    matches: list[str] = []
    for raw_name in result.stdout.split(b"\0"):
        if not raw_name:
            continue
        relative = Path(raw_name.decode("utf-8"))
        if relative.as_posix().startswith("artifacts/runs/"):
            continue
        if relative.name not in active_files and (
            not relative.parts or relative.parts[0] not in active_roots
        ):
            continue
        if relative.suffix.casefold() not in text_suffixes:
            continue
        text = relative.read_text(encoding="utf-8", errors="strict").casefold()
        for term in forbidden:
            if term in text:
                matches.append(f"{relative.as_posix()}: {term}")
    assert not matches, "forbidden model support references found:\n" + "\n".join(matches)
