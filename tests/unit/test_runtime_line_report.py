from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def _load_report_module() -> Any:
    path = Path("scripts/report_runtime_lines.py").resolve()
    spec = importlib.util.spec_from_file_location("report_runtime_lines", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_line_report_exposes_all_non_gating_categories() -> None:
    module = _load_report_module()
    counts = module.runtime_line_counts(Path.cwd())
    assert set(counts) == {
        "canonical_runtime_source_lines",
        "experiment_source_lines",
        "legacy_frozen_runtime_source_lines",
        "experiment_only_evidence_reporting_source_lines",
    }
    assert all(isinstance(value, int) and value > 0 for value in counts.values())
    assert counts["legacy_frozen_runtime_source_lines"] < counts["experiment_source_lines"]
