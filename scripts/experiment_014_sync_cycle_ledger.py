"""Reconcile Experiment 014 continuation tables into the JSON cycle ledger.

The Markdown report is the crash-surviving continuation record.  This tool keeps
the rich structured entries already present in the JSON ledger and replaces only
stale NOT_RUN entries or entries missing from that ledger.  It fails closed on
duplicate cycle IDs or incomplete eight-field tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

FIELDS = (
    "hypothesis",
    "implementation",
    "benchmark",
    "result",
    "inspection",
    "bottleneck",
    "decision",
    "redesign",
)
HEADER_RE = re.compile(r"^\| Field\s+\|\s+(H014-[^|]+?)\s+\|$")
ROW_RE = re.compile(
    r"^\|\s+(Hypothesis|Implementation|Benchmark|Result|Inspection|Bottleneck|Decision|Redesign)\s+\|\s*(.*?)\s*\|$"
)
PREREGISTERED_RE = re.compile(
    r"\s+\([^)]*\bpreregistered\b[^)]*\)\s*$", re.IGNORECASE
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _parse_tables(markdown: str) -> list[dict[str, str]]:
    lines = markdown.splitlines()
    cycles: list[dict[str, str]] = []
    seen: set[str] = set()
    index = 0
    while index < len(lines):
        header = HEADER_RE.match(lines[index])
        if header is None:
            index += 1
            continue

        cycle_id = PREREGISTERED_RE.sub("", header.group(1)).strip()
        if cycle_id in seen:
            raise ValueError(f"duplicate continuation cycle: {cycle_id}")
        seen.add(cycle_id)

        values: dict[str, str] = {}
        cursor = index + 1
        while cursor < len(lines):
            if cursor > index + 1 and HEADER_RE.match(lines[cursor]):
                break
            row = ROW_RE.match(lines[cursor])
            if row is not None:
                values[row.group(1).lower()] = row.group(2)
            if len(values) == len(FIELDS):
                break
            cursor += 1

        missing = [field for field in FIELDS if field not in values]
        if missing:
            raise ValueError(
                f"incomplete continuation cycle {cycle_id}: missing {', '.join(missing)}"
            )
        cycles.append({"id": cycle_id, **values})
        index = cursor + 1

    if not cycles:
        raise ValueError("no Experiment 014 cycle tables found")
    return cycles


def _is_completed_structured(cycle: dict[str, Any]) -> bool:
    result = cycle.get("result")
    return isinstance(result, dict) and result.get("verdict") not in (None, "NOT_RUN")


def _reconcile(
    ledger: dict[str, Any], markdown_cycles: list[dict[str, str]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    existing_cycles = ledger.get("cycles")
    if not isinstance(existing_cycles, list):
        raise ValueError("cycle ledger does not contain a cycles list")
    existing_by_id = {cycle.get("id"): cycle for cycle in existing_cycles}
    if len(existing_by_id) != len(existing_cycles):
        raise ValueError("existing cycle ledger contains duplicate or missing IDs")

    merged: list[dict[str, Any]] = []
    preserved: list[str] = []
    replaced: list[str] = []
    added: list[str] = []
    for parsed in markdown_cycles:
        existing = existing_by_id.get(parsed["id"])
        if isinstance(existing, dict) and _is_completed_structured(existing):
            merged.append(existing)
            preserved.append(parsed["id"])
        else:
            merged.append(parsed)
            (replaced if existing is not None else added).append(parsed["id"])

    markdown_ids = {cycle["id"] for cycle in markdown_cycles}
    orphaned = [cycle_id for cycle_id in existing_by_id if cycle_id not in markdown_ids]
    if orphaned:
        raise ValueError(f"ledger cycles absent from Markdown: {orphaned}")

    reconciled = dict(ledger)
    reconciled["cycles"] = merged
    audit = {
        "hypothesis": (
            "The Markdown continuation contains one unique complete eight-field table "
            "per Experiment 014 cycle and can recover the stale JSON ledger without "
            "altering completed structured cycles."
        ),
        "source_cycle_count": len(existing_cycles),
        "markdown_cycle_count": len(markdown_cycles),
        "reconciled_cycle_count": len(merged),
        "preserved_completed_structured_cycles": preserved,
        "replaced_stale_cycles": replaced,
        "added_continuation_cycles": added,
        "duplicate_cycle_ids": [],
        "incomplete_cycle_ids": [],
        "orphaned_ledger_cycle_ids": [],
        "verdict": "PASS",
    }
    return reconciled, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/experiment-014-kimi-k3-precluster-certification.md"),
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("artifacts/experiment-014/cycle-ledger.json"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("artifacts/experiment-014/ledger-reconciliation.json"),
    )
    args = parser.parse_args()

    report_bytes = args.report.read_bytes()
    original_ledger_bytes = args.ledger.read_bytes()
    original = json.loads(original_ledger_bytes)
    parsed = _parse_tables(report_bytes.decode("utf-8"))
    reconciled, audit = _reconcile(original, parsed)
    ledger_bytes = (json.dumps(reconciled, indent=2, ensure_ascii=False) + "\n").encode()
    audit.update(
        {
            "report_path": args.report.as_posix(),
            "report_sha256": _sha256(report_bytes),
            "ledger_path": args.ledger.as_posix(),
            "prior_ledger_sha256": _sha256(original_ledger_bytes),
            "reconciled_ledger_sha256": _sha256(ledger_bytes),
        }
    )
    receipt_bytes = (json.dumps(audit, indent=2, ensure_ascii=False) + "\n").encode()

    if args.write:
        _atomic_write(args.ledger, ledger_bytes)
        _atomic_write(args.receipt, receipt_bytes)
    elif original_ledger_bytes != ledger_bytes:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return 1

    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
