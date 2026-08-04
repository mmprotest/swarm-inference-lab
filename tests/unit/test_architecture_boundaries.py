from __future__ import annotations

import ast
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
PROTECTED_PACKAGES = (
    "protocol",
    "transport",
    "model",
    "execution",
    "runtime",
    "coordinator",
    "worker",
    "backends",
)

# Temporary productization allowlist. Every entry is a known pre-existing
# dependency and must be removed here in the same change that removes the import.
# The productization program requires this set to be empty by Pull Request 5.
EXPERIMENT_IMPORT_ALLOWLIST = {
    (
        "swarm_inference/backends/colibri/cuda.py",
        "swarm_inference.experiments.experiment_010.colibri_expert_bank",
    ),
}


def _resolve_from_import(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    relative = path.relative_to(SOURCE_ROOT).with_suffix("")
    package = list(relative.parts[:-1])
    keep = len(package) - (node.level - 1)
    base = package[: max(0, keep)]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _experiment_imports() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    package_root = SOURCE_ROOT / "swarm_inference"
    for package in PROTECTED_PACKAGES:
        for path in (package_root / package).rglob("*.py"):
            relative = path.relative_to(SOURCE_ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    modules.append(_resolve_from_import(path, node))
                for module in modules:
                    if module == "swarm_inference.experiments" or module.startswith(
                        "swarm_inference.experiments."
                    ):
                        found.add((relative, module))
    return found


def test_core_packages_do_not_gain_experiment_dependencies() -> None:
    actual = _experiment_imports()
    unexpected = actual - EXPERIMENT_IMPORT_ALLOWLIST
    stale = EXPERIMENT_IMPORT_ALLOWLIST - actual
    assert not unexpected, f"new core-to-experiment imports require removal: {sorted(unexpected)}"
    assert not stale, f"remove resolved imports from the temporary allowlist: {sorted(stale)}"


def test_product_commands_do_not_import_or_name_experiment_011_controller() -> None:
    product_sources = [
        SOURCE_ROOT / "swarm_inference" / "cli.py",
        *(SOURCE_ROOT / "swarm_inference" / "coordinator").glob("*.py"),
        *(SOURCE_ROOT / "swarm_inference" / "worker").glob("*.py"),
        SOURCE_ROOT / "swarm_inference" / "model" / "product.py",
        SOURCE_ROOT / "swarm_inference" / "protocol" / "product.py",
        SOURCE_ROOT / "swarm_inference" / "config" / "product.py",
    ]
    violations = []
    for path in product_sources:
        source = path.read_text(encoding="utf-8")
        if "StageRingController" in source or "experiments.experiment_011" in source:
            violations.append(path.relative_to(SOURCE_ROOT).as_posix())
    assert not violations, f"product paths depend on Experiment 011: {violations}"
