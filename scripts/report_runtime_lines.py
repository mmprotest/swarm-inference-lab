"""Report physical source lines across canonical and frozen runtime boundaries."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

CANONICAL_ROOTS = (
    "backends",
    "coordinator",
    "execution",
    "microsharding",
    "model",
    "protocol",
    "runtime",
    "transport",
    "worker",
)
EVIDENCE_MODULES = (
    "batching.py",
    "bundle.py",
    "colibri_token_path.py",
    "colibri_workloads.py",
    "correction_bundle.py",
    "level_a.py",
    "level_b.py",
    "memory_analysis.py",
    "phase10_analysis.py",
    "real_path_resilience.py",
    "real_path_simulator.py",
    "reporting.py",
    "runner.py",
    "verification.py",
)


def source_lines(paths: Iterable[Path]) -> int:
    return sum(len(path.read_text(encoding="utf-8").splitlines()) for path in paths)


def runtime_line_counts(repository_root: Path) -> dict[str, int]:
    package = repository_root / "src" / "swarm_inference"
    experiment = package / "experiments" / "experiment_010"
    legacy = experiment / "legacy_runtime"
    canonical_files = [path for name in CANONICAL_ROOTS for path in (package / name).rglob("*.py")]
    experiment_files = [path for path in experiment.rglob("*.py") if legacy not in path.parents]
    return {
        "canonical_runtime_source_lines": source_lines(canonical_files),
        "experiment_source_lines": source_lines(experiment_files),
        "legacy_frozen_runtime_source_lines": source_lines(legacy.rglob("*.py")),
        "experiment_only_evidence_reporting_source_lines": source_lines(
            experiment / name for name in EVIDENCE_MODULES
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    counts = runtime_line_counts(args.repository_root.expanduser().resolve())
    if args.json:
        print(json.dumps(counts, indent=2, sort_keys=True))
    else:
        for name, value in counts.items():
            print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
