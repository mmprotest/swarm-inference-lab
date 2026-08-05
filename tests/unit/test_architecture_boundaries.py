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
    "microsharding",
)

# Pull Request 5 closes the product-to-experiment boundary. This remains an
# explicit empty set so a future exception cannot be introduced invisibly.
EXPERIMENT_IMPORT_ALLOWLIST: set[tuple[str, str]] = set()


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


def test_product_telemetry_and_worker_lifecycle_have_no_experiment_imports() -> None:
    protected_roots = [
        SOURCE_ROOT / "swarm_inference" / "runtime",
        SOURCE_ROOT / "swarm_inference" / "worker",
    ]
    violations: list[str] = []
    for root in protected_roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [_resolve_from_import(path, node)]
                else:
                    continue
                if any(
                    module == "swarm_inference.experiments"
                    or module.startswith("swarm_inference.experiments.")
                    for module in modules
                ):
                    violations.append(path.relative_to(SOURCE_ROOT).as_posix())
    assert not violations, (
        f"product telemetry or worker lifecycle imports experiments: {violations}"
    )


def test_experiment_010_runtime_compatibility_names_are_canonical() -> None:
    from swarm_inference.backends.colibri import expert_bank as canonical_bank
    from swarm_inference.execution import expert as canonical_expert
    from swarm_inference.experiments.experiment_010 import codecs as legacy_codecs
    from swarm_inference.experiments.experiment_010 import (
        colibri_expert_bank as legacy_bank,
    )
    from swarm_inference.experiments.experiment_010 import expert as legacy_expert
    from swarm_inference.experiments.experiment_010 import schemas as legacy_schemas
    from swarm_inference.experiments.experiment_010 import wire as legacy_wire
    from swarm_inference.protocol import expert as canonical_protocol
    from swarm_inference.transport import expert as canonical_transport

    assert legacy_expert.ExpertStore is canonical_expert.ExpertStore
    assert legacy_expert.execute_expert is canonical_expert.execute_expert
    assert legacy_schemas.ExpertExecutionRequest is canonical_protocol.ExpertExecutionRequest
    assert legacy_schemas.ExpertExecutionResponse is canonical_protocol.ExpertExecutionResponse
    assert legacy_codecs.encode_array is canonical_transport.encode_array
    assert legacy_wire.encode_request is canonical_transport.encode_request
    assert legacy_wire.decode_response is canonical_transport.decode_response
    assert legacy_bank.scan_safetensors is canonical_bank.scan_safetensors
