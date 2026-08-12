"""Create a machine-readable Experiment 017 test receipt from pytest JUnit XML."""

from __future__ import annotations

import argparse
import json
import os
import platform
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _suite_totals(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time_seconds": 0.0}
    for suite in suites:
        totals["tests"] += int(suite.attrib.get("tests", 0))
        totals["failures"] += int(suite.attrib.get("failures", 0))
        totals["errors"] += int(suite.attrib.get("errors", 0))
        totals["skipped"] += int(suite.attrib.get("skipped", 0))
        totals["time_seconds"] += float(suite.attrib.get("time", 0.0))
    totals["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    totals["status"] = "PASS" if totals["failures"] == 0 and totals["errors"] == 0 else "FAIL"
    return totals


def create_receipt(targeted_xml: Path, full_xml: Path, output: Path) -> dict[str, Any]:
    receipt = {
        "schema_version": "experiment-017-test-results-v1",
        "python": platform.python_version(),
        "runner": "pytest",
        "targeted": {
            "command": (
                "python -m pytest -q tests/test_kda_verification.py "
                "tests/unit/test_kimi_indexed_copy_runtime.py "
                "tests/unit/test_kimi_k3_adapter.py"
            ),
            "coverage": [
                "blocks 1/2/4/7/12/16",
                "serial versus factorized KDA",
                "initial/continuing/accepted state",
                "prefix composition and reconstruction",
                "padding, aliasing, repetition, cleanup",
                "native backend capability and fail-closed fallback",
                "precision-mode configuration round trip",
            ],
            **_suite_totals(targeted_xml),
        },
        "full_repository": {
            "command": "python -m pytest -q",
            **_suite_totals(full_xml),
        },
    }
    receipt["status"] = (
        "PASS"
        if receipt["targeted"]["status"] == "PASS"
        and receipt["full_repository"]["status"] == "PASS"
        else "FAIL"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted-xml", type=Path, required=True)
    parser.add_argument("--full-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = create_receipt(arguments.targeted_xml, arguments.full_xml, arguments.output)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
