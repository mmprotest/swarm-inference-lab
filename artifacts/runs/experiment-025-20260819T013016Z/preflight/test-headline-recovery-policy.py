"""Focused offline checks for the E025 rolling headline recovery policy."""

from __future__ import annotations

import json
import runpy
import threading
from pathlib import Path
from typing import Any

RUN_ROOT = Path("artifacts/runs/experiment-025-20260819T013016Z").resolve()
PREFLIGHT = RUN_ROOT / "preflight"


class FakeTime:
    def time(self) -> float:
        return 1000.0


def main() -> int:
    observed = runpy.run_path(str(PREFLIGHT / "run-headline-observed.py"))
    clock = observed["HardReserveAcquisitionClock"](FakeTime())
    clock.arm(3700.0)
    seeded_now = clock.time()
    acquisition_deadline = min(3700.0 - 15 * 60, seeded_now + 25 * 60)
    assert acquisition_deadline == 2800.0
    assert clock.time() == 1000.0

    backbone_ids = [f"backbone-{index:02d}" for index in range(40)]
    started: set[str] = set()
    lock = threading.Lock()
    all_backbones_started = threading.Event()

    def ready_group(group_id: str) -> list[str]:
        if group_id.startswith("backbone"):
            with lock:
                started.add(group_id)
                if len(started) == len(backbone_ids):
                    all_backbones_started.set()
            assert all_backbones_started.wait(timeout=5.0)
        else:
            assert all_backbones_started.wait(timeout=5.0)
        return [group_id]

    def ready_parent_after_fragments(fragments: list[str]) -> list[str]:
        assert set(fragments) == {"fragment-0", "fragment-1"}
        return ["parent"]

    fragments, parent, backbone = observed[
        "ready_groups_with_full_backbone_concurrency"
    ](
        fragment_group_ids=["fragment-0", "fragment-1"],
        backbone_group_ids=backbone_ids,
        ready_group=ready_group,
        ready_parent_after_fragments=ready_parent_after_fragments,
    )
    assert set(backbone) == set(backbone_ids)
    assert set(fragments) == {"fragment-0", "fragment-1"}
    assert parent == ["parent"]

    preflight = runpy.run_path(str(PREFLIGHT / "run-full-fleet-preflight-only.py"))
    exclusion_receipt = preflight[
        "headline_progress_aware_failed_machine_exclusions"
    ](RUN_ROOT)
    excluded = set(exclusion_receipt["machine_ids"])
    retained = set(exclusion_receipt["retained_progressing_deadline_abort_machine_ids"])
    attempt = (
        RUN_ROOT
        / "rental"
        / "stage-4-headline-attempt-004-global-deadline-abort-active-gpu-load"
    )
    event_path = attempt / "physical-run-events-attempt-004.jsonl"
    events: list[dict[str, Any]] = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    attributable_stalls = {
        int(row["machine_id"])
        for row in events
        if row.get("event_type") == "TIMEOUT"
        and row.get("bootstrap_stage") is not None
        and float(row.get("no_progress_seconds", 0.0)) >= 240.0
    }
    assert attributable_stalls
    assert attributable_stalls <= excluded
    assert 57056 in retained
    assert 57056 not in excluded

    print(
        json.dumps(
            {
                "status": "PASS",
                "hard_acquisition_deadline_epoch": acquisition_deadline,
                "concurrently_started_backbone_groups": len(started),
                "attributable_stall_machine_count": len(attributable_stalls),
                "total_temporarily_excluded_machine_count": len(excluded),
                "progressing_deadline_abort_machine_57056_retained": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
