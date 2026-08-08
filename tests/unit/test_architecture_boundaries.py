from __future__ import annotations

import ast
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
PROTECTED_PACKAGES = (
    "acceptance",
    "cluster",
    "commands",
    "protocol",
    "transport",
    "model",
    "execution",
    "runtime",
    "coordinator",
    "worker",
    "backends",
    "microsharding",
    "platforms",
)

# Pull Request 5 closes the product-to-experiment boundary. This remains an
# explicit empty set so a future exception cannot be introduced invisibly.
EXPERIMENT_IMPORT_ALLOWLIST: set[tuple[str, str]] = set()
LEGACY_RUNTIME_PREFIX = "swarm_inference.experiments.experiment_010.legacy_runtime"
LEGACY_RUNTIME_MODULES = {
    "__init__.py",
    "coordinator.py",
    "dispatch.py",
    "planner.py",
    "process_main.py",
    "transport.py",
    "worker.py",
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


def test_all_production_modules_import_zero_experiment_packages() -> None:
    violations: list[tuple[str, str]] = []
    package_root = SOURCE_ROOT / "swarm_inference"
    for path in package_root.rglob("*.py"):
        if "experiments" in path.relative_to(package_root).parts:
            continue
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
                    violations.append((path.relative_to(SOURCE_ROOT).as_posix(), module))
    assert not violations, f"production modules import experiments: {violations}"


def test_generic_runtime_has_no_model_specific_orchestration_coupling() -> None:
    generic_files = (
        "cluster/models.py",
        "cluster/artifacts.py",
        "cluster/orchestrator.py",
        "config/product.py",
        "worker/stage_runtime.py",
        "worker/capabilities.py",
        "coordinator/stage_planner.py",
        "coordinator/expert_planner.py",
        "coordinator/canonical_planner.py",
        "engines/registry.py",
        "engines/colibri.py",
        "engines/native_stage.py",
        "commands/run.py",
        "model/resolver.py",
        "model/shard_builder.py",
        "model/feasibility.py",
        "execution/expert.py",
        "execution/microshard.py",
    )
    violations: list[tuple[str, str]] = []
    package_root = SOURCE_ROOT / "swarm_inference"
    for relative in generic_files:
        source = (package_root / relative).read_text(encoding="utf-8").casefold()
        for marker in ("deepseek", "gemma", "glm", "kimi", "minimax", "mistral", "qwen"):
            if marker in source:
                violations.append((relative, marker))
    assert not violations, f"generic runtime contains model-specific coupling: {violations}"


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


def test_non_experiment_source_tests_and_tools_do_not_import_legacy_runtime() -> None:
    roots = [SOURCE_ROOT, REPOSITORY_ROOT / "tests", REPOSITORY_ROOT / "scripts"]
    violations: list[tuple[str, str]] = []
    for root in roots:
        for path in root.rglob("*.py"):
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            if "src/swarm_inference/experiments/experiment_010/" in relative:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    modules.append(node.module or "")
                for module in modules:
                    if module == LEGACY_RUNTIME_PREFIX or module.startswith(
                        f"{LEGACY_RUNTIME_PREFIX}."
                    ):
                        violations.append((relative, module))
    assert not violations, f"non-experiment code imports frozen runtime: {violations}"


def test_legacy_runtime_is_frozen_to_the_documented_module_set() -> None:
    legacy = SOURCE_ROOT / "swarm_inference" / "experiments" / "experiment_010" / "legacy_runtime"
    actual = {path.name for path in legacy.glob("*.py")}
    assert actual == LEGACY_RUNTIME_MODULES
    for path in legacy.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if path.name == "__init__.py":
            assert "No product features" in source or "do not add product features" in source
        else:
            assert "LEGACY_FROZEN" in source.splitlines()[0]
    forbidden = {"protocol.py", "codecs.py", "kernels.py", "schemas.py"}
    assert not (actual & forbidden), "new runtime primitives must be canonical"


def test_experiment_010_runtime_compatibility_names_are_canonical() -> None:
    from swarm_inference.execution import expert as canonical_expert
    from swarm_inference.experiments.experiment_010 import codecs as legacy_codecs
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


def test_experiment_010_frozen_public_paths_are_named_compatibility_shims() -> None:
    from swarm_inference.experiments.experiment_010 import (
        coordinator,
        dispatch,
        planner,
        transport,
        worker,
    )

    assert coordinator.StableExpertCoordinator.__module__ == f"{LEGACY_RUNTIME_PREFIX}.coordinator"
    assert dispatch.ExpertDispatcher.__module__ == f"{LEGACY_RUNTIME_PREFIX}.dispatch"
    assert planner.PositiveUtilityPlanner.__module__ == f"{LEGACY_RUNTIME_PREFIX}.planner"
    assert transport.ExpertTransportClient.__module__ == f"{LEGACY_RUNTIME_PREFIX}.transport"
    assert worker.ExpertWorkerRuntime.__module__ == f"{LEGACY_RUNTIME_PREFIX}.worker"
