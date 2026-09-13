"""Finalize truthful, offline evidence for E025 Headline Attempt 005.

This script never contacts Vast and never launches model compute.  It derives
post-run evidence only from the canonical Attempt 005 event stream, lifecycle
ledger, frozen offer snapshot, controller receipts, and terminal cleanup files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import textwrap
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


RUN_ROOT = Path(__file__).resolve().parents[1]
TELEMETRY = RUN_ROOT / "telemetry"
FINAL = RUN_ROOT / "final"
RENTAL = RUN_ROOT / "rental" / "stage-4-headline"
PREFLIGHT = RUN_ROOT / "preflight"

EVENT_PATH = TELEMETRY / "physical-run-events.jsonl"
LEDGER_PATH = RENTAL / "instance-ledger.jsonl"
OFFER_PATH = PREFLIGHT / "full-fleet-offer-snapshot-attempt-005.json"
GO_PATH = PREFLIGHT / "FULL_FLEET_GO-attempt-005.json"
HEADLINE_PATH = FINAL / "headline-stage.json"
CLEANUP_PATH = RENTAL / "cleanup-verification.json"
COST_PATH = RENTAL / "rental-cost-summary.json"
WATCHDOG_PATH = RENTAL / "watchdog-receipt.json"
RECOVERY_PATH = PREFLIGHT / "headline-recovery-policy-tests.json"
TOKEN_BUDGET_PATH = PREFLIGHT / "headline-token-budget.json"

IMAGE_DIGEST = (
    "sha256:46cad5a031b98aeecdee9ba471e2e8338eb8e9e485d1d2e90a1505f8890ee8e4"
)
REQUIRED_WORKER_ROLES = 97
REQUIRED_BACKBONE_STAGES = 93
REQUIRED_SUBLAYER_WORKERS = 4

NAVY = "#17233A"
BLUE = "#2864DC"
BLUE_LIGHT = "#DCE8FF"
GOLD = "#C78400"
GOLD_LIGHT = "#FFF1CC"
ORANGE = "#D66324"
INK_MUTED = "#596579"
GRID = "#D8DEE8"
PAPER = "#F7F8FA"
WHITE = "#FFFFFF"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_ledger(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    previous = "0" * 64
    for sequence, row in enumerate(rows, start=1):
        if row.get("schema_version") != "experiment-025-vast-ledger-v1":
            errors.append(f"row {sequence}: schema mismatch")
        if row.get("sequence") != sequence:
            errors.append(f"row {sequence}: sequence mismatch")
        if row.get("previous_entry_sha256") != previous:
            errors.append(f"row {sequence}: previous digest mismatch")
        observed = str(row.get("entry_sha256", ""))
        body = dict(row)
        body.pop("entry_sha256", None)
        if canonical_sha256(body) != observed:
            errors.append(f"row {sequence}: entry digest mismatch")
        previous = observed
    return {
        "status": "PASS" if not errors else "FAIL",
        "entry_count": len(rows),
        "first_entry_sha256": rows[0].get("entry_sha256") if rows else None,
        "last_entry_sha256": rows[-1].get("entry_sha256") if rows else None,
        "error_count": len(errors),
        "errors": errors,
    }


def validate_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    prior_monotonic = -1
    prior_timestamp: datetime | None = None
    timestamp_regressions = 0
    counter_pattern = re.compile(r"^e025-(\d{9})-[0-9a-f]{32}$")
    for sequence, event in enumerate(events, start=1):
        event_id = str(event.get("event_id", ""))
        if event_id in seen_ids:
            errors.append(f"duplicate event id: {event_id}")
        seen_ids.add(event_id)
        match = counter_pattern.match(event_id)
        if match is None or int(match.group(1)) != sequence:
            errors.append(f"event {sequence}: non-canonical event id {event_id}")
        monotonic_ns = int(event.get("monotonic_ns", -1))
        if monotonic_ns < prior_monotonic:
            errors.append(f"event {sequence}: monotonic clock regressed")
        prior_monotonic = monotonic_ns
        timestamp = parse_time(str(event["timestamp_utc"]))
        if prior_timestamp is not None and timestamp < prior_timestamp:
            timestamp_regressions += 1
        prior_timestamp = timestamp
    return {
        "status": "PASS" if not errors else "FAIL",
        "event_count": len(events),
        "unique_event_id_count": len(seen_ids),
        "event_ids_unique": len(seen_ids) == len(events),
        "event_id_sequence_contiguous": not any(
            "non-canonical event id" in error for error in errors
        ),
        "monotonic_ns_nondecreasing": not any(
            "monotonic clock regressed" in error for error in errors
        ),
        "timestamp_regression_count": timestamp_regressions,
        "animation_order": "event_id / monotonic_ns",
        "error_count": len(errors),
        "errors": errors,
    }


def location_fields(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    names = {
        "geolocation",
        "geolocode",
        "country",
        "country_code",
        "region",
        "state",
        "province",
        "city",
        "datacenter",
        "datacenter_name",
        "location",
        "latitude",
        "longitude",
        "lat",
        "lon",
    }
    return {key: value for key, value in raw.items() if key in names and value is not None}


def short_gpu(value: str | None) -> str:
    return str(value or "Not supplied").replace("NVIDIA GeForce ", "")


def event_state_analysis(
    events: list[dict[str, Any]], hard_deadline: datetime
) -> dict[str, Any]:
    start = parse_time(events[0]["timestamp_utc"])
    ready_by_worker: dict[str, int | None] = {}
    changes: list[dict[str, Any]] = [
        {
            "timestamp_utc": events[0]["timestamp_utc"],
            "elapsed_minutes": 0.0,
            "ready_workers": 0,
        }
    ]
    peak_count = 0
    peak_event_id: str | None = None
    peak_timestamp: str | None = None
    peak_workers: list[str] = []
    ready_at_cutoff: dict[str, int | None] = {}

    for event in events:
        timestamp = parse_time(event["timestamp_utc"])
        if timestamp <= hard_deadline:
            ready_at_cutoff = dict(ready_by_worker)
        event_type = event.get("event_type")
        worker_id = event.get("worker_id")
        instance_id = event.get("instance_id")
        changed = False
        if event_type == "WORKER_READY" and worker_id:
            ready_by_worker[str(worker_id)] = (
                int(instance_id) if instance_id is not None else None
            )
            changed = True
        elif event_type in {
            "WORKER_UNHEALTHY",
            "WORKER_DISCONNECTED",
        } and worker_id:
            existing = ready_by_worker.get(str(worker_id))
            if str(worker_id) in ready_by_worker and (
                instance_id is None or existing == int(instance_id)
            ):
                ready_by_worker.pop(str(worker_id), None)
                changed = True
        elif event_type == "INSTANCE_DESTROYED" and instance_id is not None:
            doomed = [
                worker
                for worker, active_instance in ready_by_worker.items()
                if active_instance == int(instance_id)
            ]
            for worker in doomed:
                ready_by_worker.pop(worker, None)
                changed = True

        if changed:
            count = len(ready_by_worker)
            changes.append(
                {
                    "timestamp_utc": event["timestamp_utc"],
                    "elapsed_minutes": round(
                        (timestamp - start).total_seconds() / 60.0, 6
                    ),
                    "ready_workers": count,
                }
            )
            if count > peak_count:
                peak_count = count
                peak_event_id = str(event["event_id"])
                peak_timestamp = str(event["timestamp_utc"])
                peak_workers = sorted(ready_by_worker)
        if timestamp <= hard_deadline:
            ready_at_cutoff = dict(ready_by_worker)

    consolidated: list[dict[str, Any]] = []
    for row in changes:
        if consolidated and row["elapsed_minutes"] == consolidated[-1]["elapsed_minutes"]:
            consolidated[-1] = row
        else:
            consolidated.append(row)
    return {
        "timeline": consolidated,
        "peak_live_ready_worker_count": peak_count,
        "peak_live_ready_event_id": peak_event_id,
        "peak_live_ready_timestamp_utc": peak_timestamp,
        "peak_live_ready_worker_ids": peak_workers,
        "ready_worker_count_at_hard_deadline": len(ready_at_cutoff),
        "ready_worker_ids_at_hard_deadline": sorted(ready_at_cutoff),
    }


def base_figure(*, portrait: bool = False) -> tuple[Any, Any]:
    size = (12, 15) if portrait else (16, 9)
    figure, axis = plt.subplots(figsize=size, dpi=100)
    figure.patch.set_facecolor(PAPER)
    axis.set_facecolor(PAPER)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    return figure, axis


def save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor(), dpi=100)
    plt.close(figure)


def badge(axis: Any, text: str, *, x: float, y: float, color: str) -> None:
    axis.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=WHITE,
        fontsize=14,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": color, "edgecolor": color},
    )


def footer(axis: Any, text: str = "Attempt 005 | retained machine-readable evidence") -> None:
    axis.text(0.04, 0.035, text, fontsize=10.5, color=INK_MUTED)


def metric_card(
    axis: Any,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    value: str,
    label: str,
    edge: str = GRID,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.014",
            facecolor=WHITE,
            edgecolor=edge,
            linewidth=1.6,
        )
    )
    axis.text(
        x + width / 2,
        y + height * 0.62,
        value,
        ha="center",
        va="center",
        fontsize=27,
        fontweight="bold",
        color=NAVY,
    )
    axis.text(
        x + width / 2,
        y + height * 0.25,
        label,
        ha="center",
        va="center",
        fontsize=11,
        color=INK_MUTED,
        linespacing=1.2,
    )


def draw_hero(facts: dict[str, Any], path: Path) -> None:
    figure, axis = base_figure(portrait=True)
    axis.text(0.06, 0.94, "EXPERIMENT 025", fontsize=18, fontweight="bold", color=BLUE)
    axis.text(
        0.06,
        0.875,
        "HEADLINE ATTEMPT 005",
        fontsize=36,
        fontweight="bold",
        color=NAVY,
    )
    badge(axis, "INCOMPLETE", x=0.16, y=0.79, color=ORANGE)
    axis.text(
        0.06,
        0.70,
        "Fleet acquisition reached the\n30-minute safety cutoff.",
        fontsize=29,
        fontweight="bold",
        color=NAVY,
        linespacing=1.25,
    )
    axis.text(
        0.06,
        0.60,
        "The full fleet never became READY, so no Kimi K3 token was attempted.",
        fontsize=15,
        color=INK_MUTED,
    )
    cards = [
        (
            f"{facts['readiness']['peak_live_ready_worker_count']}/{REQUIRED_WORKER_ROLES}",
            "PEAK LIVE\nREADY ROLES",
        ),
        ("4/4", "LAYER 89 FRAGMENTS\nREADY (NOT EXECUTED)"),
        (str(facts["instances_created"]), "PAID INSTANCES\nCREATED"),
        (str(facts["replacement_count"]), "TARGETED\nREPLACEMENTS"),
    ]
    for index, (value, label) in enumerate(cards):
        row, column = divmod(index, 2)
        metric_card(
            axis,
            x=0.06 + column * 0.46,
            y=0.43 - row * 0.18,
            width=0.40,
            height=0.13,
            value=value,
            label=label,
            edge=BLUE_LIGHT if index != 1 else GOLD_LIGHT,
        )
    axis.add_patch(
        FancyBboxPatch(
            (0.06, 0.125),
            0.86,
            0.095,
            boxstyle="round,pad=0.014",
            facecolor=BLUE_LIGHT,
            edgecolor=BLUE_LIGHT,
        )
    )
    axis.text(
        0.09,
        0.175,
        "CLEANUP VERIFIED",
        fontsize=14,
        fontweight="bold",
        color=BLUE,
        va="center",
    )
    axis.text(
        0.39,
        0.175,
        "zero live E025 Vast instances",
        fontsize=16,
        color=NAVY,
        va="center",
    )
    axis.text(
        0.06,
        0.085,
        f"Ledger estimate: ${facts['cost']['active_plus_storage_usd']:.2f} active + storage | not a provider invoice",
        fontsize=11.5,
        color=INK_MUTED,
    )
    footer(axis, "Attempt 005 | PHYSICAL acquisition evidence | no inference claim")
    save_figure(figure, path)


def draw_topology(facts: dict[str, Any], path: Path) -> None:
    figure, axis = base_figure()
    axis.text(0.04, 0.93, "Attempt 005 acquisition topology", fontsize=29, fontweight="bold", color=NAVY)
    axis.text(
        0.04,
        0.88,
        "All 41 backbone groups and four fragment groups were monitored concurrently; the parent followed fragment readiness.",
        fontsize=13,
        color=INK_MUTED,
    )
    nodes = [
        (0.13, 0.60, "41", "BACKBONE\nGROUPS", BLUE),
        (0.40, 0.60, "4", "L89 FRAGMENT\nGROUPS", GOLD),
        (0.67, 0.60, "1", "L89 PARENT\nGROUP", INK_MUTED),
        (0.88, 0.60, "97", "REQUIRED\nROLES", NAVY),
    ]
    for x, y, value, label, color in nodes:
        axis.add_patch(plt.Circle((x, y), 0.066, facecolor=WHITE, edgecolor=color, linewidth=3))
        axis.text(x, y + 0.012, value, ha="center", va="center", fontsize=25, fontweight="bold", color=color)
        axis.text(x, y - 0.10, label, ha="center", va="top", fontsize=11, color=NAVY, linespacing=1.2)
    for x0, x1 in ((0.20, 0.33), (0.47, 0.60), (0.74, 0.81)):
        axis.annotate("", xy=(x1, 0.60), xytext=(x0, 0.60), arrowprops={"arrowstyle": "->", "color": GRID, "lw": 2.2})

    peak = facts["readiness"]["peak_live_ready_worker_count"]
    axis.text(0.08, 0.34, "Peak simultaneously-live readiness", fontsize=14, fontweight="bold", color=NAVY)
    axis.add_patch(FancyBboxPatch((0.08, 0.25), 0.84, 0.055, boxstyle="round,pad=0.002", facecolor="#E4E8EF", edgecolor="#E4E8EF"))
    axis.add_patch(FancyBboxPatch((0.08, 0.25), 0.84 * peak / REQUIRED_WORKER_ROLES, 0.055, boxstyle="round,pad=0.002", facecolor=BLUE, edgecolor=BLUE))
    axis.text(0.08, 0.215, f"{peak} READY", fontsize=12, fontweight="bold", color=BLUE)
    axis.text(0.92, 0.215, f"{REQUIRED_WORKER_ROLES - peak} short of full fleet", fontsize=12, ha="right", color=INK_MUTED)
    axis.text(
        0.08,
        0.12,
        f"{facts['instances_created']} created instances  |  {facts['replacement_count']} isolated replacements  |  parent never READY  |  Token 1 not run",
        fontsize=14,
        color=NAVY,
    )
    footer(axis, "Attempt 005 | topology is acquisition state, not a completed inference path")
    save_figure(figure, path)


def draw_sublayer(facts: dict[str, Any], path: Path) -> None:
    figure, axis = base_figure()
    axis.text(0.04, 0.93, "Layer 89 sub-layer readiness", fontsize=29, fontweight="bold", color=NAVY)
    axis.text(
        0.04,
        0.88,
        "Four mandatory fragment workers reached READY on four distinct machines. Attempt 005 never reached inference, so this is not execution evidence.",
        fontsize=13,
        color=INK_MUTED,
    )
    rows = facts["sublayer_workers"]
    for index, row in enumerate(rows):
        x = 0.04 + index * 0.24
        axis.add_patch(
            FancyBboxPatch(
                (x, 0.23),
                0.215,
                0.54,
                boxstyle="round,pad=0.014",
                facecolor=WHITE,
                edgecolor=GOLD,
                linewidth=2,
            )
        )
        axis.text(x + 0.018, 0.70, row["worker_id"].replace("e025-layer-089-", ""), fontsize=18, fontweight="bold", color=GOLD)
        axis.text(x + 0.018, 0.635, short_gpu(row.get("gpu_name")), fontsize=15, fontweight="bold", color=NAVY, wrap=True)
        details = [
            f"Machine {row['machine_id']}",
            f"Instance {row['instance_id']}",
            f"GPU UUID {str(row.get('gpu_uuid') or 'not supplied')[:18]}",
            f"Experts {row.get('owned_expert_count', 'not supplied')}",
            f"READY +{row['ready_elapsed_minutes']:.1f} min",
            str(row.get("raw_geolocation") or "location not supplied"),
        ]
        axis.text(x + 0.018, 0.55, "\n".join(details), fontsize=11, color=INK_MUTED, va="top", linespacing=1.55)
        badge(axis, "READY", x=x + 0.1075, y=0.275, color=BLUE)
    axis.text(
        0.04,
        0.12,
        "Readiness means checkpoint hash verified, shard loaded, worker listening, and WORKER_READY recorded. It does not mean a token traversed the worker.",
        fontsize=12.5,
        color=NAVY,
    )
    footer(axis, "Attempt 005 | Vast-reported host location metadata | no geography inferred")
    save_figure(figure, path)


def draw_inference(facts: dict[str, Any], path: Path) -> None:
    figure, axis = base_figure()
    axis.text(0.04, 0.93, "Attempt 005 inference gate", fontsize=29, fontweight="bold", color=NAVY)
    badge(axis, "NOT REACHED", x=0.84, y=0.92, color=ORANGE)
    stages = [
        ("Fleet acquisition", f"{facts['readiness']['peak_live_ready_worker_count']}/{REQUIRED_WORKER_ROLES} peak READY", ORANGE),
        ("Token 1 correctness", "not run", INK_MUTED),
        ("Token 2 state advance", "not run", INK_MUTED),
        ("Exact public sentence", "not run", INK_MUTED),
    ]
    for index, (title, detail, color) in enumerate(stages):
        y = 0.73 - index * 0.16
        axis.add_patch(plt.Circle((0.10, y), 0.025, facecolor=WHITE, edgecolor=color, linewidth=3))
        if index < len(stages) - 1:
            axis.plot([0.10, 0.10], [y - 0.03, y - 0.13], color=GRID, linewidth=3)
        axis.text(0.16, y + 0.017, title, fontsize=18, fontweight="bold", color=NAVY, va="center")
        axis.text(0.16, y - 0.027, detail, fontsize=13, color=INK_MUTED, va="center")
    token_counts = facts["token_event_counts"]
    axis.add_patch(FancyBboxPatch((0.55, 0.20), 0.39, 0.53, boxstyle="round,pad=0.018", facecolor=WHITE, edgecolor=GRID))
    axis.text(0.59, 0.66, "Canonical token events", fontsize=15, fontweight="bold", color=NAVY)
    y = 0.59
    for name, count in token_counts.items():
        axis.text(0.59, y, name.replace("_", " "), fontsize=11, color=INK_MUTED)
        axis.text(0.89, y, str(count), fontsize=12, ha="right", fontweight="bold", color=NAVY)
        y -= 0.05
    axis.text(
        0.04,
        0.09,
        "No token IDs, decoded text, latency, throughput, communication path, reduction, or Layer 89 per-token participation exists for this attempt.",
        fontsize=13,
        color=NAVY,
    )
    footer(axis, "Attempt 005 | absence is explicit; no output reconstructed from timestamps")
    save_figure(figure, path)


def draw_locations(facts: dict[str, Any], path: Path) -> None:
    counts = facts["location_counts"]
    rows = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    if len(rows) > 16:
        retained = rows[:15]
        retained.append(("Other supplied raw labels", sum(value for _, value in rows[15:])))
        rows = retained
    figure, axis = plt.subplots(figsize=(16, 9), dpi=100)
    figure.patch.set_facecolor(PAPER)
    axis.set_facecolor(PAPER)
    labels = [label for label, _ in rows][::-1]
    values = [value for _, value in rows][::-1]
    axis.barh(labels, values, color=BLUE, edgecolor=NAVY, linewidth=0.5)
    axis.set_title("Vast-reported host location metadata", loc="left", fontsize=27, fontweight="bold", color=NAVY, pad=30)
    axis.text(0, 1.02, "Distinct Attempt 005 machines by raw geolocation label; no coordinates or missing geography inferred.", transform=axis.transAxes, fontsize=12.5, color=INK_MUTED)
    axis.set_xlabel("Distinct physical machine IDs", color=INK_MUTED)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(axis="y", labelsize=10.5, colors=NAVY)
    axis.tick_params(axis="x", colors=INK_MUTED)
    for index, value in enumerate(values):
        axis.text(value + 0.15, index, str(value), va="center", fontsize=10.5, color=NAVY)
    figure.text(0.06, 0.025, "Attempt 005 | full raw fields retained in telemetry/attempt-005-node-inventory.json", fontsize=10.5, color=INK_MUTED)
    figure.subplots_adjust(left=0.23, right=0.96, top=0.83, bottom=0.12)
    save_figure(figure, path)


def draw_audit(facts: dict[str, Any], path: Path) -> None:
    figure, axis = base_figure()
    axis.text(0.04, 0.93, "E025 Attempt 005 audit proof", fontsize=29, fontweight="bold", color=NAVY)
    badge(axis, "INCOMPLETE / CLEAN", x=0.82, y=0.92, color=ORANGE)
    checks = [
        ("Canonical event log", f"{facts['event_log']['event_count']} events | validation {facts['event_log']['status']}"),
        ("Event stream SHA-256", facts["event_log_sha256"]),
        ("Rental ledger", f"{facts['ledger']['entry_count']} entries | hash chain {facts['ledger']['status']}"),
        ("Paid instances", f"{facts['instances_created']} created | {facts['instances_destroyed']} uniquely destroyed"),
        ("Lifecycle policy", "41 backbone groups concurrent | 240 s no-progress replacement"),
        ("Hard reserve", "30 min acquisition limit | 15 min inference/cleanup reserve"),
        ("Inference", "NOT REACHED | zero token execution and emission events"),
        ("Cleanup", "PASS | zero live E025 instances"),
        ("Watchdog", "stopped only after zero-live verification"),
        ("Worker image", IMAGE_DIGEST),
    ]
    for index, (label, value) in enumerate(checks):
        y = 0.82 - index * 0.071
        axis.text(0.05, y, label, fontsize=12.5, color=INK_MUTED, fontweight="bold", va="center")
        shown = value if len(value) <= 82 else value[:79] + "..."
        axis.text(0.29, y, shown, fontsize=11.5, color=NAVY, va="center", family="monospace" if "SHA" in label or "image" in label.lower() else None)
        axis.plot([0.05, 0.95], [y - 0.031, y - 0.031], color=GRID, linewidth=0.7)
    axis.text(
        0.05,
        0.075,
        "Attempt 005 does not satisfy E025 PASS. The trace freezes a failed acquisition attempt, not a completed Kimi K3 inference.",
        fontsize=12.5,
        color=NAVY,
        fontweight="bold",
    )
    footer(axis)
    save_figure(figure, path)


def report_markdown(facts: dict[str, Any]) -> str:
    peak = facts["readiness"]["peak_live_ready_worker_count"]
    deadline_ready = facts["readiness"]["ready_worker_count_at_hard_deadline"]
    cost = facts["cost"]
    return f"""# Experiment 025: Headline Attempt 005

## Technical summary

Headline Attempt 005 is **INCOMPLETE**. The corrected controller launched only after the live full-fleet balance gate returned `GO`, monitored all backbone groups concurrently, and performed {facts['replacement_count']} attributable instance replacements without destroying healthy siblings. The fleet reached a peak of **{peak} simultaneously-live READY roles out of 97 required**. All four mandatory Layer 89 fragment workers became READY on distinct physical machines, but the Layer 89 parent and the complete required fleet did not become READY before the 30-minute acquisition cutoff.

No Kimi K3 inference was attempted: Token 1, Token 2, and the exact public sentence were all **not reached**. Cleanup passed, the independent watchdog stopped only after zero-live verification, and a post-cleanup Vast query confirmed **zero live E025 instances**.

## The corrected lifecycle policy worked, but acquisition did not finish

- `READINESS_MONITORING_STARTED` records 41 backbone groups and four Layer 89 fragment groups under `ALL_GROUPS_CONCURRENT_ROLLING_REPLACEMENT`.
- The controller used a 240-second per-instance no-progress threshold and a 30-minute hard acquisition window with a 15-minute inference/cleanup reserve.
- {facts['attributable_instance_timeout_count']} instance-level no-progress timeouts were recorded and {facts['replacement_count']} replacement instances were created.
- Machine 57056, the progressing host incorrectly torn down in Attempt 004, was retained through genuine progress in Attempt 005 and reached `WORKER_READY` for `e025-layer-089-sub-02`.
- At the hard deadline, {deadline_ready} roles were still READY. Global teardown occurred only at the allowed terminal acquisition cutoff because the required 97-role fleet was incomplete.

This is evidence that the corrected dud-isolation policy operated as designed. It is not evidence that the full Kimi K3 path executed.

## Layer 89 fragments reached READY but never executed a token

Each mandatory Layer 89 fragment loaded its assigned expert shard, verified the checkpoint fingerprint, and reported a physical consumer GPU identity. Their machine IDs were {', '.join(str(row['machine_id']) for row in facts['sublayer_workers'])}, all distinct. Readiness establishes physical capacity and worker preparation in this attempt; because Token 1 was never started, it does not establish Layer 89 participation in a full-fleet forward pass.

## Scope and metric definitions

- **Required role:** one of the frozen 93 backbone stages or four Layer 89 sub-layer workers.
- **Simultaneously-live READY:** a worker's latest canonical state is `WORKER_READY`, with no subsequent matching unhealthy, disconnect, or instance-destroy event at that point in event order.
- **Paid instance:** a unique Vast instance with both ledger `CREATE_CONFIRMED` and canonical `INSTANCE_CREATED` evidence.
- **Targeted replacement:** a canonical `INSTANCE_REPLACED` event for one failed instance group.
- **Attempt duration:** the policy's 1,800-second acquisition window; the first ledger-confirmed paid instance occurred {facts['deadline']['seconds_after_policy_origin_to_first_paid_instance']:.3f} seconds after the conservative policy origin.
- **Location:** raw Vast-reported host metadata only. No missing country, city, region, or coordinates were inferred.

## Methodology and retained evidence

The post-run finalizer parsed the canonical event stream in event-ID/monotonic order, validated contiguous unique event IDs, computed the stream SHA-256, replayed worker readiness as a state machine, reconciled unique created and destroyed instance IDs, and independently validated every entry in the append-only lifecycle ledger hash chain. It then joined created instances to the frozen offer snapshot for advertised hardware, network, pricing, and raw location fields. Missing values remain explicitly absent.

The canonical trace contains {facts['event_log']['event_count']} events. It has no `TOKEN_EXECUTION_STARTED`, `MESSAGE_SEND_STARTED`, `ROUTE_COMPUTED`, `TOKEN_SAMPLE_STARTED`, or `TOKEN_EMITTED` records because inference never began. No token path or output was reconstructed from timestamps.

## Cost and cleanup

The lifecycle ledger estimates **${cost['estimated_active_rental_usd']:.4f}** active rental and **${cost['estimated_storage_usd']:.4f}** storage, or **${cost['active_plus_storage_usd']:.4f}** combined. These use actual ledger lifetimes and selected advertised rates but are **not a provider invoice**. Completed worker-download events support a separate **${cost['completed_download_ingress_lower_bound_usd']:.4f}** lower-bound ingress estimate; incomplete downloads and provider billing counters are not fully observed, so this must not be represented as total actual ingress.

All {facts['instances_created']} created Attempt 005 instance IDs have destruction evidence. Cleanup and the independent post-cleanup query both report `zero_live_e025_instances = true`; unrelated instances were not destroyed.

## Limitations and PASS-gate status

Attempt 005 cannot support the E025 headline claim. The complete physical fleet identity was never frozen, full-path model execution did not occur, numerical and two-token stateful correctness were not tested, and no public generation exists. The terminal base-summary exception names Layer 89 parent readiness, but the controlling terminal condition was the wrapper's permitted hard acquisition cutoff; the parent exception is the downstream incomplete-fleet manifestation, not a new Kimi K3 execution defect.

The canonical headline summary correctly remains `INCOMPLETE`. Its full-fleet physical worker count is zero because the implementation freezes that field only after complete acquisition; the {peak}-role figure in this report is an acquisition-state statistic derived from the canonical event stream, not a replacement PASS metric.

## Recommended next step

Do not weaken any E025 PASS gate and do not portray Attempt 005 as model execution. Preserve this attempt as acquisition evidence. Any future paid attempt should be separately authorized, retain the corrected per-instance lifecycle policy, and first use the Attempt 005 churn record to improve live-offer feasibility and alternate depth without redesigning the 97-role scientific architecture.

## Further questions

- Which Vast offer and bootstrap attributes best predict READY completion within the acquisition window?
- How much alternate depth is required to overcome the observed machine churn while retaining a 15-minute correctness and cleanup reserve?
- Can the controller emit a dedicated `ACQUISITION_HARD_DEADLINE_REACHED` event before teardown so the terminal reason is explicit rather than inferred from the recorded policy deadline and subsequent readiness aborts?
"""


def build_report_artifact(facts: dict[str, Any], generated_at: str) -> dict[str, Any]:
    readiness_rows = [
        {"state": "Required fleet", "roles": REQUIRED_WORKER_ROLES, "sort_order": 1},
        {
            "state": "Peak live READY",
            "roles": facts["readiness"]["peak_live_ready_worker_count"],
            "sort_order": 2,
        },
    ]
    churn_rows = [
        {"event": "Create requested", "count": facts["create_request_count"], "sort_order": 1},
        {"event": "Instance created", "count": facts["instances_created"], "sort_order": 2},
        {"event": "Targeted replacement", "count": facts["replacement_count"], "sort_order": 3},
        {"event": "Instance no-progress timeout", "count": facts["attributable_instance_timeout_count"], "sort_order": 4},
        {"event": "Instance destroyed", "count": facts["instances_destroyed"], "sort_order": 5},
    ]
    sublayer_rows = [
        {
            "worker": row["worker_id"].replace("e025-layer-089-", ""),
            "machine_id": row["machine_id"],
            "gpu": short_gpu(row.get("gpu_name")),
            "ready_minutes": row["ready_elapsed_minutes"],
            "raw_location": row.get("raw_geolocation") or "Not supplied",
            "execution_status": "Not reached",
        }
        for row in facts["sublayer_workers"]
    ]
    location_rows = [
        {"raw_location": label, "distinct_machines": count}
        for label, count in sorted(
            facts["location_counts"].items(), key=lambda item: (-item[1], item[0])
        )
    ]
    cost_rows = [
        {"component": "Active rental", "usd": facts["cost"]["estimated_active_rental_usd"], "basis": "Ledger lifetime x advertised active rate", "sort_order": 1},
        {"component": "Storage", "usd": facts["cost"]["estimated_storage_usd"], "basis": "Ledger lifetime x advertised storage rate", "sort_order": 2},
        {"component": "Active + storage", "usd": facts["cost"]["active_plus_storage_usd"], "basis": "Ledger estimate; not provider invoice", "sort_order": 3},
        {"component": "Completed-download ingress lower bound", "usd": facts["cost"]["completed_download_ingress_lower_bound_usd"], "basis": "Completed worker download bytes x selected offer rate", "sort_order": 4},
    ]
    gate_rows = [
        {"gate": gate, "result": "PASS" if passed else "NOT ESTABLISHED"}
        for gate, passed in sorted(facts["pass_gates"].items())
    ]

    query_sql = {
        "readiness-query": "SELECT state, roles, sort_order FROM readiness_summary ORDER BY sort_order",
        "churn-query": "SELECT event, count, sort_order FROM lifecycle_churn ORDER BY sort_order",
        "sublayer-query": "SELECT worker, machine_id, gpu, ready_minutes, raw_location, execution_status FROM sublayer_workers ORDER BY worker",
        "locations-query": "SELECT raw_location, distinct_machines FROM location_counts ORDER BY distinct_machines DESC, raw_location LIMIT 10",
        "cost-query": "SELECT component, usd, basis, sort_order FROM cost_components ORDER BY sort_order",
        "gates-query": "SELECT gate, result FROM pass_gates ORDER BY gate",
    }

    database_path = FINAL / "attempt-005-report.sqlite"
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.cursor()
        for table in (
            "readiness_summary",
            "lifecycle_churn",
            "sublayer_workers",
            "location_counts",
            "cost_components",
            "pass_gates",
        ):
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
        cursor.execute(
            "CREATE TABLE readiness_summary (state TEXT, roles INTEGER, sort_order INTEGER)"
        )
        cursor.executemany(
            "INSERT INTO readiness_summary VALUES (:state, :roles, :sort_order)",
            readiness_rows,
        )
        cursor.execute(
            "CREATE TABLE lifecycle_churn (event TEXT, count INTEGER, sort_order INTEGER)"
        )
        cursor.executemany(
            "INSERT INTO lifecycle_churn VALUES (:event, :count, :sort_order)", churn_rows
        )
        cursor.execute(
            "CREATE TABLE sublayer_workers (worker TEXT, machine_id INTEGER, gpu TEXT, ready_minutes REAL, raw_location TEXT, execution_status TEXT)"
        )
        cursor.executemany(
            "INSERT INTO sublayer_workers VALUES (:worker, :machine_id, :gpu, :ready_minutes, :raw_location, :execution_status)",
            sublayer_rows,
        )
        cursor.execute(
            "CREATE TABLE location_counts (raw_location TEXT, distinct_machines INTEGER)"
        )
        cursor.executemany(
            "INSERT INTO location_counts VALUES (:raw_location, :distinct_machines)",
            location_rows,
        )
        cursor.execute(
            "CREATE TABLE cost_components (component TEXT, usd REAL, basis TEXT, sort_order INTEGER)"
        )
        cursor.executemany(
            "INSERT INTO cost_components VALUES (:component, :usd, :basis, :sort_order)",
            cost_rows,
        )
        cursor.execute("CREATE TABLE pass_gates (gate TEXT, result TEXT)")
        cursor.executemany(
            "INSERT INTO pass_gates VALUES (:gate, :result)", gate_rows
        )
        connection.commit()

        queried: dict[str, list[dict[str, Any]]] = {}
        for source_id, sql in query_sql.items():
            queried[source_id] = [dict(row) for row in cursor.execute(sql).fetchall()]
    finally:
        connection.close()

    readiness_rows = queried["readiness-query"]
    churn_rows = queried["churn-query"]
    sublayer_rows = queried["sublayer-query"]
    location_rows = queried["locations-query"]
    cost_rows = queried["cost-query"]
    gate_rows = queried["gates-query"]

    def sql_source(
        source_id: str,
        label: str,
        description: str,
        table: str,
    ) -> dict[str, Any]:
        return {
            "id": source_id,
            "label": label,
            "path": "final/attempt-005-report.sqlite",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": query_sql[source_id],
                "description": description,
                "executed_at": generated_at,
                "tables_used": [table],
                "filters": [],
            },
        }

    sources = [
        sql_source("readiness-query", "Canonical readiness replay", "Selects the required-fleet and peak-live-readiness comparison derived from the canonical event stream.", "readiness_summary"),
        sql_source("churn-query", "Canonical lifecycle reconciliation", "Selects lifecycle counts reconciled from the canonical event stream and ledger.", "lifecycle_churn"),
        sql_source("sublayer-query", "Layer 89 readiness identities", "Selects the four physical Layer 89 readiness records derived from WORKER_READY events.", "sublayer_workers"),
        sql_source("locations-query", "Vast-reported location labels", "Selects distinct attempted-machine counts by raw Vast geolocation label.", "location_counts"),
        sql_source("cost-query", "Attempt 005 cost reconciliation", "Selects ledger-derived cost components and the completed-download ingress lower bound.", "cost_components"),
        sql_source("gates-query", "Attempt 005 terminal PASS gates", "Selects terminal gate values from the headline summary.", "pass_gates"),
        {"id": "events", "label": "Attempt 005 canonical physical event stream", "path": "telemetry/physical-run-events.jsonl"},
        {"id": "ledger", "label": "Attempt 005 append-only rental ledger", "path": "rental/stage-4-headline/instance-ledger.jsonl"},
        {"id": "offers", "label": "Frozen Attempt 005 live-offer snapshot", "path": "preflight/full-fleet-offer-snapshot-attempt-005.json"},
        {"id": "cleanup", "label": "Attempt 005 cleanup verification", "path": "rental/stage-4-headline/cleanup-verification.json"},
    ]

    summary_text = (
        "## Attempt 005 stopped before inference\n\n"
        f"**INCOMPLETE.** Peak simultaneously-live readiness was **{facts['readiness']['peak_live_ready_worker_count']} of 97 roles**. "
        "All four Layer 89 fragments reached READY on distinct machines, but the required fleet and Layer 89 parent did not finish acquisition before the hard cutoff.\n\n"
        "No Token 1, Token 2, or public generation was attempted. Cleanup passed with **zero live E025 instances**."
    )
    lifecycle_text = (
        "## Isolated replacement worked; fleet completion did not\n\n"
        f"The corrected controller monitored 41 backbone groups and four fragment groups concurrently, recorded **{facts['replacement_count']} targeted replacements**, and retained progressing siblings. "
        "The readiness curve below shows acquisition state, not inference throughput."
    )
    churn_text = (
        "## Provider churn consumed the acquisition window\n\n"
        "Created and destroyed instances reconcile one-for-one. Replacement and timeout counts show why retained healthy workers were insufficient to assemble all 97 roles before the reserve boundary."
    )
    sublayer_text = (
        "## Four Layer 89 fragments became READY, not executed\n\n"
        "Each fragment verified its checkpoint and loaded its assigned expert shard on a distinct consumer-GPU machine. The bars show minutes from run start to READY. Because Token 1 never started, these workers have no per-token execution evidence in Attempt 005."
    )
    scope_text = (
        "## Scope, definitions, and evidence classes\n\n"
        "**PHYSICAL** applies to instance lifecycle, remote worker preparation, and readiness. **NOT REACHED** applies to all inference, communication, routing, reduction, sampling, and decoded-output claims. `READY` is replayed from canonical lifecycle events and removed by matching unhealthy, disconnect, or destroy events. Location labels are raw Vast-reported host metadata; no missing geography is inferred."
    )
    location_text = (
        "## Attempted machines spanned the supplied raw location labels\n\n"
        "This comparison counts distinct attempted machine IDs by Vast's raw `geolocation` value. It is an infrastructure-distribution view, not proof of geographic accuracy or a statement about inference participation."
    )
    method_text = (
        "## Validation method\n\n"
        f"The finalizer validated **{facts['event_log']['event_count']} contiguous unique events**, replayed readiness in event-ID/monotonic order, reconciled {facts['instances_created']} unique created instances with destruction evidence, and verified all {facts['ledger']['entry_count']} ledger entries against their hash chain. The frozen event-stream SHA-256 is `{facts['event_log_sha256'][:12]}...{facts['event_log_sha256'][-12:]}`; the complete digest remains in the source evidence."
    )
    cost_text = (
        "## Cost and cleanup remained bounded\n\n"
        f"Active rental plus storage is a **${facts['cost']['active_plus_storage_usd']:.4f} ledger estimate**, not a provider invoice. The completed-download ingress value is a lower bound because incomplete downloads and provider counters are not fully observed. All Attempt 005 instances were destroyed and the watchdog stopped only after zero-live verification."
    )
    limits_text = (
        "## What this attempt does not establish\n\n"
        "The complete Kimi K3 physical path, numerical correctness, stateful two-token decode, public generation, token latency, throughput, and per-token Layer 89 participation were not measured. The 46-role peak is an acquisition-state statistic and is not an E025 PASS substitute."
    )
    next_text = (
        "## Recommended next step\n\n"
        "Preserve Attempt 005 as physical acquisition evidence and keep every scientific PASS gate unchanged. Any future paid attempt should be separately authorized and should use this attempt's offer, timeout, and replacement history to improve live-fleet feasibility while retaining the corrected per-instance policy and 15-minute reserve."
    )
    questions_text = (
        "## Further questions\n\n"
        "- Which offer and bootstrap attributes best predict READY completion?\n"
        "- How much alternate depth is required for the observed churn?\n"
        "- Should the wrapper emit a dedicated hard-deadline event before teardown?"
    )

    manifest_sources = [dict(row) for row in sources]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "E025 Headline Attempt 005: acquisition stopped before inference",
            "description": "Technical audit of the physical fleet acquisition, terminal cutoff, and cleanup.",
            "generatedAt": generated_at,
            "filters": [],
            "cards": [],
            "charts": [
                {
                    "id": "readiness-chart",
                    "title": "Required versus peak live READY roles",
                    "subtitle": "Acquisition peaked 51 roles short of the required frozen fleet",
                    "type": "bar",
                    "dataset": "readiness_summary",
                    "sourceId": "readiness-query",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "state", "type": "nominal", "label": "Fleet state"},
                        "y": {"field": "roles", "type": "quantitative", "label": "Worker roles"},
                    },
                },
                {
                    "id": "churn-chart",
                    "title": "Acquisition lifecycle counts",
                    "subtitle": "Canonical lifecycle events and unique instance identities",
                    "type": "bar",
                    "dataset": "lifecycle_churn",
                    "sourceId": "churn-query",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "event", "type": "nominal", "label": "Lifecycle measure"},
                        "y": {"field": "count", "type": "quantitative", "label": "Count"},
                    },
                },
                {
                    "id": "sublayer-chart",
                    "title": "Layer 89 fragment time to READY",
                    "subtitle": "Four distinct physical machines; execution was not reached",
                    "type": "bar",
                    "dataset": "sublayer_workers",
                    "sourceId": "sublayer-query",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "worker", "type": "nominal", "label": "Fragment worker"},
                        "y": {"field": "ready_minutes", "type": "quantitative", "label": "Minutes to READY"},
                        "tooltip": [
                            {"field": "machine_id", "type": "quantitative", "label": "Machine ID"},
                            {"field": "gpu", "type": "nominal", "label": "GPU"},
                            {"field": "raw_location", "type": "nominal", "label": "Raw Vast location"},
                        ],
                    },
                },
                {
                    "id": "locations-chart",
                    "title": "Attempted machines by raw Vast location label",
                    "subtitle": "Top 10 raw labels by distinct machine count; full inventory is retained separately",
                    "type": "bar",
                    "dataset": "location_counts",
                    "sourceId": "locations-query",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "raw_location", "type": "nominal", "label": "Vast-reported host location metadata"},
                        "y": {"field": "distinct_machines", "type": "quantitative", "label": "Distinct machines"},
                    },
                },
            ],
            "tables": [
                {
                    "id": "sublayer-table",
                    "title": "Layer 89 fragment worker identities",
                    "subtitle": "Physical readiness identities; no token execution occurred",
                    "dataset": "sublayer_workers",
                    "sourceId": "sublayer-query",
                    "density": "spacious",
                    "defaultSort": {"field": "worker", "direction": "asc"},
                    "columns": [
                        {"field": "worker", "label": "Worker", "type": "text"},
                        {"field": "machine_id", "label": "Machine ID", "format": "number"},
                        {"field": "gpu", "label": "GPU", "type": "text"},
                        {"field": "ready_minutes", "label": "Minutes to READY", "format": "number"},
                        {"field": "raw_location", "label": "Raw Vast location", "type": "text"},
                        {"field": "execution_status", "label": "Execution", "type": "text"},
                    ],
                },
                {
                    "id": "cost-table",
                    "title": "Attempt 005 cost reconciliation",
                    "subtitle": "USD estimates derived from ledger lifetimes and advertised rates",
                    "dataset": "cost_components",
                    "sourceId": "cost-query",
                    "density": "spacious",
                    "defaultSort": {"field": "component", "direction": "asc"},
                    "columns": [
                        {"field": "component", "label": "Component", "type": "text"},
                        {"field": "usd", "label": "USD", "format": "currency"},
                        {"field": "basis", "label": "Basis", "type": "text"},
                    ],
                },
                {
                    "id": "gates-table",
                    "title": "Terminal E025 PASS-gate state",
                    "subtitle": "Unreached scientific gates remain not established",
                    "dataset": "pass_gates",
                    "sourceId": "gates-query",
                    "density": "dense",
                    "defaultSort": {"field": "gate", "direction": "asc"},
                    "columns": [
                        {"field": "gate", "label": "Gate", "type": "text"},
                        {"field": "result", "label": "Terminal result", "type": "text"},
                    ],
                },
            ],
            "sources": manifest_sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# E025 Headline Attempt 005: acquisition stopped before inference"},
                {"id": "summary", "type": "markdown", "body": summary_text},
                {"id": "lifecycle", "type": "markdown", "body": lifecycle_text},
                {"id": "readiness", "type": "chart", "chartId": "readiness-chart"},
                {"id": "churn-text", "type": "markdown", "body": churn_text},
                {"id": "sublayer-text", "type": "markdown", "body": sublayer_text},
                {"id": "scope", "type": "markdown", "body": scope_text},
                {"id": "locations-text", "type": "markdown", "body": location_text},
                {"id": "method", "type": "markdown", "body": method_text},
                {"id": "cost-text", "type": "markdown", "body": cost_text},
                {"id": "limits", "type": "markdown", "body": limits_text},
                {"id": "next", "type": "markdown", "body": next_text},
                {"id": "questions", "type": "markdown", "body": questions_text},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "readiness_summary": readiness_rows,
                "lifecycle_churn": churn_rows,
                "sublayer_workers": sublayer_rows,
                "location_counts": location_rows,
                "cost_components": cost_rows,
                "pass_gates": gate_rows,
            },
            "accessIssues": [],
        },
        "sources": [dict(row) for row in sources],
        "package_info": {
            "originUrl": "artifact://experiment-025-attempt-005",
            "controls": {"edit": False, "refresh": False},
        },
    }


def generate() -> dict[str, Any]:
    generated_at = utc_now()
    events = read_jsonl(EVENT_PATH)
    ledger = read_jsonl(LEDGER_PATH)
    offer_snapshot = read_json(OFFER_PATH)
    preflight_go = read_json(GO_PATH)
    headline = read_json(HEADLINE_PATH)
    cleanup = read_json(CLEANUP_PATH)
    lifecycle_cost = read_json(COST_PATH)
    watchdog = read_json(WATCHDOG_PATH)
    recovery = read_json(RECOVERY_PATH)
    token_budget = read_json(TOKEN_BUDGET_PATH)

    event_validation = validate_events(events)
    ledger_validation = validate_ledger(ledger)
    if event_validation["status"] != "PASS":
        raise ValueError(f"canonical event validation failed: {event_validation['errors']}")
    if ledger_validation["status"] != "PASS":
        raise ValueError(f"lifecycle ledger validation failed: {ledger_validation['errors']}")

    event_hash = sha256_file(EVENT_PATH)
    (TELEMETRY / "physical-run-events.sha256").write_text(
        f"{event_hash}  physical-run-events.jsonl\n", encoding="ascii"
    )
    event_counts = Counter(str(event.get("event_type")) for event in events)
    policy = next(event for event in events if event.get("event_type") == "ACQUISITION_POLICY_APPLIED")
    monitoring = next(event for event in events if event.get("event_type") == "READINESS_MONITORING_STARTED")
    hard_deadline = datetime.fromtimestamp(float(policy["hard_acquisition_deadline_epoch"]), tz=UTC)
    policy_origin = hard_deadline.timestamp() - 1800.0

    created_ids = {
        int(event["instance_id"])
        for event in events
        if event.get("event_type") == "INSTANCE_CREATED" and event.get("instance_id") is not None
    }
    destroyed_ids = {
        int(event["instance_id"])
        for event in events
        if event.get("event_type") == "INSTANCE_DESTROYED" and event.get("instance_id") is not None
    }
    ledger_created = {
        int(row["instance_id"])
        for row in ledger
        if row.get("event") == "CREATE_CONFIRMED" and row.get("instance_id") is not None
    }
    ledger_destroyed = {
        int(row["instance_id"])
        for row in ledger
        if row.get("event") == "DESTROY_CONFIRMED" and row.get("instance_id") is not None
    }
    if created_ids != ledger_created:
        raise ValueError("event/ledger created instance sets differ")
    if not created_ids.issubset(destroyed_ids & ledger_destroyed):
        raise ValueError("not every created instance has destruction evidence")

    create_records = {
        int(row["instance_id"]): row
        for row in ledger
        if row.get("event") == "CREATE_CONFIRMED" and row.get("instance_id") is not None
    }
    destroy_records = {
        int(row["instance_id"]): row
        for row in ledger
        if row.get("event") == "DESTROY_CONFIRMED" and row.get("instance_id") is not None
    }
    offers = {
        int(row["offer_id"]): row
        for row in offer_snapshot.get("offers", [])
        if row.get("offer_id") is not None
    }
    event_locations: dict[int, dict[str, Any]] = {}
    for event in events:
        instance_id = event.get("instance_id")
        metadata = event.get("vast_reported_host_location_metadata")
        if instance_id is not None and isinstance(metadata, dict):
            event_locations[int(instance_id)] = dict(metadata)

    assignments = {
        str(event["worker_id"]): event
        for event in events
        if event.get("event_type") == "SHARD_ASSIGNED" and event.get("worker_id")
    }
    ready_by_pair = {
        (str(event["worker_id"]), int(event["instance_id"])): event
        for event in events
        if event.get("event_type") == "WORKER_READY"
        and event.get("worker_id")
        and event.get("instance_id") is not None
    }
    state_by_pair: dict[tuple[str, int], str] = {}
    role_by_pair: dict[tuple[str, int], str | None] = {}
    for event in events:
        worker_id = event.get("worker_id")
        instance_id = event.get("instance_id")
        if not worker_id or instance_id is None:
            continue
        pair = (str(worker_id), int(instance_id))
        if event.get("role"):
            role_by_pair[pair] = str(event["role"])
        if event.get("event_type") in {
            "WORKER_CONNECTING",
            "WORKER_CONNECTED",
            "WORKER_READY",
            "WORKER_UNHEALTHY",
            "WORKER_DISCONNECTED",
        }:
            state_by_pair[pair] = str(event["event_type"])

    worker_pairs = sorted(
        {
            (str(event["worker_id"]), int(event["instance_id"]))
            for event in events
            if event.get("event_type") in {"WORKER_CONNECTING", "WORKER_READY"}
            and event.get("worker_id")
            and event.get("instance_id") is not None
        }
    )
    worker_ids_by_instance: dict[int, list[str]] = defaultdict(list)
    worker_attempts: list[dict[str, Any]] = []
    run_start = parse_time(events[0]["timestamp_utc"])

    for worker_id, instance_id in worker_pairs:
        worker_ids_by_instance[instance_id].append(worker_id)
        creation = create_records[instance_id]
        offer = offers.get(int(creation.get("offer_id") or -1), {})
        raw_offer = offer.get("raw") if isinstance(offer.get("raw"), dict) else {}
        metadata = location_fields(raw_offer)
        if not metadata:
            metadata = location_fields(event_locations.get(instance_id))
        ready = ready_by_pair.get((worker_id, instance_id))
        assignment = assignments.get(worker_id, {})
        gpu = ready.get("gpu", {}) if ready else {}
        worker_attempts.append(
            {
                "worker_id": worker_id,
                "vast_instance_id": instance_id,
                "vast_machine_id": int(creation["machine_id"]),
                "offer_id": int(creation["offer_id"]),
                "instance_label": creation.get("instance_label"),
                "role": role_by_pair.get((worker_id, instance_id)) or assignment.get("role"),
                "assigned_layer": assignment.get("layer"),
                "assigned_fragment": assignment.get("fragment_id"),
                "assigned_shard_event_id": assignment.get("event_id"),
                "assigned_bytes": assignment.get("assigned_bytes"),
                "download_bytes": assignment.get("download_bytes"),
                "physical_gpu_slot": gpu.get("physical_gpu_slot", assignment.get("gpu_slot")),
                "gpu_model_advertised": creation.get("gpu_model"),
                "gpu_model_observed": gpu.get("gpu_name"),
                "gpu_uuid": gpu.get("gpu_uuid"),
                "vram_gib_advertised": creation.get("advertised_vram_gib"),
                "vram_mib_observed": gpu.get("vram_mib"),
                "cpu_name": raw_offer.get("cpu_name"),
                "cpu_cores": raw_offer.get("cpu_cores"),
                "cpu_cores_effective": raw_offer.get("cpu_cores_effective"),
                "cpu_ram_mib": raw_offer.get("cpu_ram"),
                "driver_version_advertised": raw_offer.get("driver_version"),
                "driver_version_observed": gpu.get("driver_version"),
                "cuda_max_advertised": raw_offer.get("cuda_max_good"),
                "cuda_runtime_observed": gpu.get("cuda_runtime"),
                "network": {
                    "internet_down_mbps": creation.get("advertised_internet_down_mbps"),
                    "internet_up_mbps": creation.get("advertised_internet_up_mbps"),
                    "disk_bandwidth_mbps": creation.get("advertised_disk_bandwidth_mbps"),
                    "internet_ingress_rate_usd_per_gb": creation.get("internet_ingress_rate_usd_per_gb"),
                    "internet_egress_rate_usd_per_gb": creation.get("internet_egress_rate_usd_per_gb"),
                },
                "advertised_reliability": creation.get("advertised_reliability"),
                "active_rental_rate_usd_per_hour": creation.get("active_rental_rate_usd_per_hour"),
                "storage_rate_usd_per_gb_month": creation.get("storage_rate_usd_per_gb_month"),
                "requested_disk_gb": creation.get("requested_disk_gb"),
                "vast_reported_host_location_metadata": metadata,
                "ready": ready is not None,
                "ready_timestamp_utc": ready.get("timestamp_utc") if ready else None,
                "checkpoint_fingerprint": ready.get("checkpoint_fingerprint") if ready else None,
                "image_digest": ready.get("image_digest") if ready else IMAGE_DIGEST,
                "last_recorded_state": state_by_pair.get((worker_id, instance_id)),
                "evidence_class": "PHYSICAL",
            }
        )

    instances: list[dict[str, Any]] = []
    for instance_id in sorted(created_ids):
        creation = create_records[instance_id]
        destruction = destroy_records[instance_id]
        offer = offers.get(int(creation.get("offer_id") or -1), {})
        raw_offer = offer.get("raw") if isinstance(offer.get("raw"), dict) else {}
        metadata = location_fields(raw_offer)
        if not metadata:
            metadata = location_fields(event_locations.get(instance_id))
        worker_ids = sorted(set(worker_ids_by_instance.get(instance_id, [])))
        instances.append(
            {
                "vast_instance_id": instance_id,
                "vast_machine_id": int(creation["machine_id"]),
                "offer_id": int(creation["offer_id"]),
                "instance_label": creation.get("instance_label"),
                "assigned_group": creation.get("instance_label"),
                "gpu_model": creation.get("gpu_model"),
                "gpu_count": creation.get("gpu_count"),
                "vram_gib_per_gpu": creation.get("advertised_vram_gib"),
                "cpu_name": raw_offer.get("cpu_name"),
                "cpu_cores": raw_offer.get("cpu_cores"),
                "cpu_ram_mib": raw_offer.get("cpu_ram"),
                "driver_version": raw_offer.get("driver_version"),
                "cuda_max": raw_offer.get("cuda_max_good"),
                "network": {
                    "internet_down_mbps": creation.get("advertised_internet_down_mbps"),
                    "internet_up_mbps": creation.get("advertised_internet_up_mbps"),
                    "disk_bandwidth_mbps": creation.get("advertised_disk_bandwidth_mbps"),
                },
                "advertised_reliability": creation.get("advertised_reliability"),
                "active_rental_rate_usd_per_hour": creation.get("active_rental_rate_usd_per_hour"),
                "storage_rate_usd_per_gb_month": creation.get("storage_rate_usd_per_gb_month"),
                "requested_disk_gb": creation.get("requested_disk_gb"),
                "vast_reported_host_location_metadata": metadata,
                "worker_ids": worker_ids,
                "ready_worker_ids": sorted(
                    worker
                    for worker in worker_ids
                    if (worker, instance_id) in ready_by_pair
                ),
                "created_at_utc": creation.get("creation_time"),
                "destroyed_at_utc": destruction.get("destroy_confirmed_time"),
                "final_status": destruction.get("final_status"),
                "image_digest": IMAGE_DIGEST,
                "evidence_class": "PHYSICAL",
            }
        )

    machine_location: dict[int, str] = {}
    for row in instances:
        machine_id = int(row["vast_machine_id"])
        metadata = row["vast_reported_host_location_metadata"]
        machine_location[machine_id] = str(metadata.get("geolocation") or "Not supplied")
    location_counts = Counter(machine_location.values())

    readiness = event_state_analysis(events, hard_deadline)
    ready_events = [event for event in events if event.get("event_type") == "WORKER_READY"]
    ready_workers_ever = {str(event["worker_id"]) for event in ready_events}
    sublayer_workers: list[dict[str, Any]] = []
    for index in range(REQUIRED_SUBLAYER_WORKERS):
        worker_id = f"e025-layer-089-sub-{index:02d}"
        candidates = [event for event in ready_events if event.get("worker_id") == worker_id]
        if not candidates:
            raise ValueError(f"missing required sub-layer readiness event: {worker_id}")
        ready = candidates[-1]
        instance_id = int(ready["instance_id"])
        creation = create_records[instance_id]
        offer = offers.get(int(creation.get("offer_id") or -1), {})
        raw_offer = offer.get("raw") if isinstance(offer.get("raw"), dict) else {}
        metadata = location_fields(raw_offer) or location_fields(event_locations.get(instance_id))
        assignment = ready.get("assignment") if isinstance(ready.get("assignment"), dict) else {}
        gpu = ready.get("gpu") if isinstance(ready.get("gpu"), dict) else {}
        sublayer_workers.append(
            {
                "worker_id": worker_id,
                "instance_id": instance_id,
                "machine_id": int(ready["machine_id"]),
                "gpu_name": gpu.get("gpu_name"),
                "gpu_uuid": gpu.get("gpu_uuid"),
                "vram_mib": gpu.get("vram_mib"),
                "driver_version": gpu.get("driver_version"),
                "cuda_runtime": gpu.get("cuda_runtime"),
                "physical_gpu_slot": gpu.get("physical_gpu_slot"),
                "checkpoint_fingerprint": ready.get("checkpoint_fingerprint"),
                "image_digest": ready.get("image_digest"),
                "owned_expert_count": assignment.get("owned_expert_count"),
                "owned_components": assignment.get("owned_components"),
                "source_weight_bytes": assignment.get("source_weight_bytes"),
                "ready_timestamp_utc": ready["timestamp_utc"],
                "ready_elapsed_minutes": round(
                    (parse_time(ready["timestamp_utc"]) - run_start).total_seconds() / 60.0,
                    6,
                ),
                "vast_reported_host_location_metadata": metadata,
                "raw_geolocation": metadata.get("geolocation"),
                "attempt_005_execution_count": 0,
                "attempt_005_execution_status": "NOT_REACHED",
                "evidence_class": "PHYSICAL_READINESS_NOT_EXECUTION",
            }
        )
    if len({row["machine_id"] for row in sublayer_workers}) != REQUIRED_SUBLAYER_WORKERS:
        raise ValueError("sub-layer workers did not use four distinct machine IDs")

    completed_download_events = [
        event for event in events if event.get("event_type") == "MODEL_DOWNLOAD_COMPLETED"
    ]
    observed_completed_bytes = 0
    ingress_lower_bound = 0.0
    for event in completed_download_events:
        downloaded = int(event.get("downloaded_bytes") or 0)
        observed_completed_bytes += downloaded
        creation = create_records.get(int(event["instance_id"]))
        if creation:
            rate = float(creation.get("internet_ingress_rate_usd_per_gb") or 0.0)
            ingress_lower_bound += downloaded / 1e9 * rate
    active_cost = float(lifecycle_cost["estimated_active_cost_usd"])
    storage_cost = float(lifecycle_cost["estimated_storage_cost_usd"])
    active_plus_storage = active_cost + storage_cost
    cost = {
        "schema_version": "experiment-025-attempt-005-cost-reconciliation-v1",
        "generated_at_utc": generated_at,
        "status": "PASS_ESTIMATE_ONLY",
        "instance_count": len(created_ids),
        "total_elapsed_instance_seconds": lifecycle_cost.get("total_elapsed_instance_seconds"),
        "estimated_active_rental_usd": active_cost,
        "estimated_storage_usd": storage_cost,
        "active_plus_storage_usd": active_plus_storage,
        "completed_download_event_count": len(completed_download_events),
        "completed_download_bytes": observed_completed_bytes,
        "completed_download_ingress_lower_bound_usd": ingress_lower_bound,
        "active_storage_plus_completed_download_lower_bound_usd": active_plus_storage + ingress_lower_bound,
        "provider_invoice_claimed": False,
        "actual_provider_invoice_available": False,
        "basis": "ledger instance lifetimes multiplied by selected advertised active/storage rates; completed worker download bytes multiplied by selected advertised ingress rates",
        "limitations": [
            "This is not a Vast provider invoice.",
            "Completed-download ingress is a lower bound and excludes incomplete downloads or missing provider counters.",
            "The account had unrelated activity, so balance delta is not attributed to Attempt 005.",
        ],
        "source_cost_receipt": "rental/stage-4-headline/rental-cost-summary.json",
        "source_event_log": "telemetry/physical-run-events.jsonl",
    }
    write_json(FINAL / "attempt-005-cost-reconciliation.json", cost)

    token_event_names = [
        "TOKEN_EXECUTION_STARTED",
        "MESSAGE_SEND_STARTED",
        "ROUTE_COMPUTED",
        "EXPERT_DISPATCHED",
        "REDUCTION_STARTED",
        "TOKEN_SAMPLE_STARTED",
        "TOKEN_EMITTED",
        "TOKEN_EXECUTION_COMPLETED",
    ]
    token_event_counts = {name: event_counts.get(name, 0) for name in token_event_names}
    attributable_timeouts = [
        event
        for event in events
        if event.get("event_type") == "TIMEOUT"
        and not event.get("worker_id")
        and event.get("instance_id") is not None
        and float(event.get("no_progress_seconds") or 0) >= 240.0
    ]
    first_paid = min(
        parse_time(str(row["creation_time"])) for row in create_records.values()
    )
    deadline = {
        "policy_origin_utc": datetime.fromtimestamp(policy_origin, tz=UTC).isoformat(),
        "hard_acquisition_deadline_utc": hard_deadline.isoformat(),
        "policy_window_seconds": 1800,
        "first_ledger_confirmed_paid_instance_utc": first_paid.isoformat(),
        "seconds_after_policy_origin_to_first_paid_instance": first_paid.timestamp() - policy_origin,
        "seconds_from_first_ledger_confirmed_paid_instance_to_deadline": hard_deadline.timestamp() - first_paid.timestamp(),
        "reserve_seconds": int(policy["inference_and_cleanup_reserve_seconds"]),
        "terminal_abort_timestamp_utc": next(
            event["timestamp_utc"]
            for event in reversed(events)
            if event.get("event_type") == "ABORT"
        ),
        "interpretation": "The conservative policy clock began a few seconds before the first ledger-confirmed paid instance; it did not exceed the 30-minute maximum from paid creation.",
    }

    required_event_types = [
        "OFFER_SELECTED",
        "INSTANCE_CREATE_REQUESTED",
        "INSTANCE_CREATED",
        "INSTANCE_BOOTING",
        "INSTANCE_REPLACED",
        "WORKER_CONNECTING",
        "WORKER_CONNECTED",
        "WORKER_READY",
        "WORKER_UNHEALTHY",
        "WORKER_DISCONNECTED",
        "INSTANCE_DESTROY_REQUESTED",
        "INSTANCE_DESTROYED",
        "MODEL_DOWNLOAD_STARTED",
        "MODEL_DOWNLOAD_PROGRESS",
        "MODEL_DOWNLOAD_COMPLETED",
        "MODEL_HASH_VERIFIED",
        "SHARD_ASSIGNED",
        "SHARD_LOAD_STARTED",
        "SHARD_LOADED",
        "MESSAGE_SEND_STARTED",
        "MESSAGE_SEND_COMPLETED",
        "MESSAGE_RECEIVED",
        "TOKEN_EXECUTION_STARTED",
        "LAYER_EXECUTION_STARTED",
        "FRAGMENT_EXECUTION_STARTED",
        "FRAGMENT_EXECUTION_COMPLETED",
        "LAYER_EXECUTION_COMPLETED",
        "TOKEN_EXECUTION_COMPLETED",
        "ROUTE_COMPUTED",
        "EXPERT_DISPATCHED",
        "EXPERT_RESULT_RETURNED",
        "REDUCTION_STARTED",
        "REDUCTION_COMPLETED",
        "TOKEN_SAMPLE_STARTED",
        "TOKEN_EMITTED",
        "RETRY",
        "TIMEOUT",
        "ERROR",
        "ABORT",
        "WATCHDOG_TRIGGERED",
    ]
    event_coverage = {
        name: {
            "count": event_counts.get(name, 0),
            "status": (
                "PRESENT"
                if event_counts.get(name, 0)
                else "NOT_EMITTED_INFERENCE_NOT_REACHED"
                if name in token_event_names
                or name
                in {
                    "MESSAGE_SEND_COMPLETED",
                    "MESSAGE_RECEIVED",
                    "LAYER_EXECUTION_STARTED",
                    "FRAGMENT_EXECUTION_STARTED",
                    "FRAGMENT_EXECUTION_COMPLETED",
                    "LAYER_EXECUTION_COMPLETED",
                    "EXPERT_RESULT_RETURNED",
                    "REDUCTION_COMPLETED",
                }
                else "NOT_EMITTED_NOT_APPLICABLE_OR_NOT_OBSERVED"
            ),
        }
        for name in required_event_types
    }

    facts = {
        "schema_version": "experiment-025-attempt-005-derived-facts-v1",
        "generated_at_utc": generated_at,
        "run_id": "20260819T013016Z",
        "attempt": 5,
        "status": "INCOMPLETE",
        "evidence_class": "PHYSICAL_ACQUISITION_ONLY",
        "scientific_execution_reached": False,
        "token_1_reached": False,
        "token_2_reached": False,
        "public_generation_reached": False,
        "event_log_sha256": event_hash,
        "event_log": event_validation,
        "ledger": ledger_validation,
        "event_type_counts": dict(sorted(event_counts.items())),
        "event_coverage": event_coverage,
        "instances_created": len(created_ids),
        "instances_destroyed": len(created_ids & destroyed_ids & ledger_destroyed),
        "create_request_count": event_counts["INSTANCE_CREATE_REQUESTED"],
        "replacement_count": event_counts["INSTANCE_REPLACED"],
        "attributable_instance_timeout_count": len(attributable_timeouts),
        "ready_event_count": event_counts["WORKER_READY"],
        "unique_worker_roles_ever_ready": len(ready_workers_ever),
        "readiness": readiness,
        "sublayer_workers": sublayer_workers,
        "all_four_sublayer_workers_ready": True,
        "all_four_sublayer_machine_ids_distinct": True,
        "layer89_parent_ready": any(
            event.get("event_type") == "WORKER_READY"
            and event.get("worker_id") == "e025-stage-089-parent"
            for event in events
        ),
        "monitoring": {
            "policy": monitoring.get("policy"),
            "backbone_group_count": monitoring.get("backbone_group_count"),
            "fragment_group_count": monitoring.get("fragment_group_count"),
            "all_groups_concurrent": monitoring.get("policy")
            == "ALL_GROUPS_CONCURRENT_ROLLING_REPLACEMENT",
            "per_instance_no_progress_timeout_seconds": policy.get(
                "per_instance_no_progress_timeout_seconds"
            ),
        },
        "deadline": deadline,
        "token_event_counts": token_event_counts,
        "location_counts": dict(sorted(location_counts.items())),
        "cost": cost,
        "cleanup": cleanup,
        "watchdog": watchdog,
        "preflight_go_status": preflight_go.get("status"),
        "recovery_policy_test_status": recovery.get("status"),
        "exact_public_prompt_configuration_status": token_budget.get("status"),
        "pass_gates": headline.get("pass_gates", {}),
        "immutable_worker_image": IMAGE_DIGEST,
        "controller_observer_sha256": events[0].get("observer_sha256"),
        "source_tree_sha256": events[0].get("source_tree_sha256"),
    }

    timeline_events: list[dict[str, Any]] = []
    for sequence, event in enumerate(events, start=1):
        row = dict(event)
        row["timeline_sequence"] = sequence
        row["elapsed_seconds"] = round(
            (parse_time(event["timestamp_utc"]) - run_start).total_seconds(), 6
        )
        timeline_events.append(row)
    animation_timeline = {
        "schema_version": "experiment-025-animation-timeline-v1",
        "generated_at_utc": generated_at,
        "run_id": "20260819T013016Z",
        "attempt": 5,
        "status": "INCOMPLETE",
        "source_event_log": "telemetry/physical-run-events.jsonl",
        "source_event_log_sha256": event_hash,
        "ordering": "timeline_sequence follows canonical event_id and monotonic_ns",
        "inference_not_reached": True,
        "event_count": len(timeline_events),
        "events": timeline_events,
    }
    inventory = {
        "schema_version": "experiment-025-attempt-005-node-inventory-v1",
        "generated_at_utc": generated_at,
        "run_id": "20260819T013016Z",
        "attempt": 5,
        "status": "INCOMPLETE",
        "evidence_class": "PHYSICAL_ACQUISITION_ONLY",
        "location_semantics": "Vast-reported host location metadata; missing geography is not inferred",
        "immutable_worker_image": IMAGE_DIGEST,
        "created_instance_count": len(instances),
        "physical_worker_attempt_count": len(worker_attempts),
        "ready_worker_attempt_count": sum(1 for row in worker_attempts if row["ready"]),
        "instances": instances,
        "worker_attempts": worker_attempts,
    }
    event_summary = dict(facts)
    event_summary["source_event_log"] = "telemetry/physical-run-events.jsonl"
    event_summary["source_ledger"] = "rental/stage-4-headline/instance-ledger.jsonl"
    event_summary["source_offer_snapshot"] = (
        "preflight/full-fleet-offer-snapshot-attempt-005.json"
    )
    event_summary["animation_timeline"] = "telemetry/animation-timeline.json"
    event_summary["node_inventory"] = "telemetry/attempt-005-node-inventory.json"

    write_json(TELEMETRY / "animation-event-log-summary.json", event_summary)
    write_json(TELEMETRY / "animation-timeline.json", animation_timeline)
    write_json(TELEMETRY / "attempt-005-node-inventory.json", inventory)
    write_json(FINAL / "attempt-005-terminal-result.json", facts)
    (FINAL / "EXPERIMENT_025_REPORT.md").write_text(
        report_markdown(facts), encoding="utf-8"
    )
    (FINAL / "manager-summary.md").write_text(
        "# E025 Attempt 005 manager summary\n\n"
        "Status: **INCOMPLETE**\n\n"
        f"Peak simultaneously-live READY roles: **{readiness['peak_live_ready_worker_count']} / 97**\n\n"
        "Token 1 / Token 2 / public generation: **not reached**\n\n"
        f"Estimated active rental + storage: **${active_plus_storage:.4f}** (not a provider invoice)\n\n"
        "Zero live E025 Vast instances: **true**\n",
        encoding="utf-8",
    )
    write_json(
        FINAL / "chart-map.json",
        {
            "schema_version": "experiment-025-attempt-005-chart-map-v1",
            "generated_at_utc": generated_at,
            "delivery_surface": "portable HTML plus explicitly requested static PNG evidence",
            "palette_policy": "hard two-root cap: blue and gold/orange plus neutrals",
            "charts": [
                {
                    "section": "Lifecycle",
                    "question": "How did peak readiness compare with the required fleet?",
                    "family": "Comparison & Ranking",
                    "type": "bar",
                    "fields": ["state", "roles"],
                    "claim": "Acquisition peaked below the required 97 roles.",
                    "source": "telemetry/physical-run-events.jsonl",
                },
                {
                    "section": "Churn",
                    "question": "How much instance churn occurred?",
                    "family": "Comparison & Ranking",
                    "type": "bar",
                    "fields": ["event", "count"],
                    "claim": "Targeted replacement operated, but churn consumed the window.",
                    "source": "telemetry/animation-event-log-summary.json",
                },
                {
                    "section": "Layer 89",
                    "question": "When did each fragment become READY?",
                    "family": "Comparison & Ranking",
                    "type": "bar",
                    "fields": ["worker", "ready_minutes"],
                    "claim": "All four distinct fragment workers became READY, not executed.",
                    "source": "telemetry/physical-run-events.jsonl",
                },
                {
                    "section": "Locations",
                    "question": "Which raw Vast geolocation labels were reported?",
                    "family": "Comparison & Ranking",
                    "type": "bar",
                    "fields": ["raw_location", "distinct_machines"],
                    "claim": "Attempted physical machines span the supplied raw labels.",
                    "source": "telemetry/attempt-005-node-inventory.json",
                },
            ],
        },
    )
    write_json(
        FINAL / "attempt-005-report-artifact.json",
        build_report_artifact(facts, generated_at),
    )

    draw_hero(facts, FINAL / "hero-linkedin.png")
    draw_topology(facts, FINAL / "swarm-topology.png")
    draw_sublayer(facts, FINAL / "sub-layer-proof.png")
    draw_inference(facts, FINAL / "inference-proof.png")
    draw_locations(facts, FINAL / "node-locations.png")
    draw_audit(facts, FINAL / "audit-proof.png")
    return facts


def freeze() -> dict[str, Any]:
    report_delivery = {
        "schema_version": "experiment-025-attempt-005-report-delivery-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS_WITH_DISCLOSED_BROWSER_QA_LIMITATION",
        "delivery_mode": "portable_html",
        "artifact_json": "final/attempt-005-report-artifact.json",
        "html": "final/evidence.html",
        "self_contained_builder_validation": "PASS",
        "enhanced_reader_browser_verification": "INCOMPLETE",
        "browser_verifier_code": "horizontal_overflow",
        "browser_verifier_context": (
            "On this Windows host, the packaged verifier measured document-width "
            "overflow approximately equal to the classic vertical scrollbar. The "
            "failure-only screenshot shows the enhanced reader and chart rendered, "
            "but the verifier could not issue a passed receipt. The generated HTML "
            "was not hand-edited to conceal the failure."
        ),
        "failure_screenshot": "final/report-builder-failure.png",
        "semantic_fallback_retained": True,
        "external_network_dependencies": False,
    }
    write_json(FINAL / "report-delivery-receipt.json", report_delivery)
    excluded = {
        "final/artifact-hashes.json",
        "final/render-receipt.json",
    }
    files: list[dict[str, Any]] = []
    for path in sorted(RUN_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(RUN_ROOT).as_posix()
        if relative in excluded or relative.endswith(".lock"):
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "experiment-025-artifact-hashes-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "experiment_status": "INCOMPLETE",
        "file_count": len(files),
        "files": files,
        "excluded_paths": sorted(excluded | {"**/*.lock"}),
    }
    write_json(FINAL / "artifact-hashes.json", manifest)
    expected = [
        "hero-linkedin.png",
        "swarm-topology.png",
        "sub-layer-proof.png",
        "inference-proof.png",
        "node-locations.png",
        "audit-proof.png",
    ]
    missing = [name for name in expected if not (FINAL / name).is_file()]
    html_path = FINAL / "evidence.html"
    receipt = {
        "schema_version": "experiment-025-final-render-v2",
        "generated_at_utc": utc_now(),
        "status": (
            "PASS_WITH_DISCLOSED_REPORT_BROWSER_QA_LIMITATION"
            if not missing and html_path.is_file()
            else "INCOMPLETE"
        ),
        "experiment_status": "INCOMPLETE",
        "victory_graphics_generated": False,
        "truthful_incomplete_evidence_graphics_generated": not missing,
        "missing_static_evidence": missing,
        "images": [f"final/{name}" for name in expected if (FINAL / name).is_file()],
        "report_markdown": "final/EXPERIMENT_025_REPORT.md",
        "report_artifact": "final/attempt-005-report-artifact.json",
        "report_html": "final/evidence.html" if html_path.is_file() else None,
        "report_delivery_receipt": "final/report-delivery-receipt.json",
        "animation_event_log_summary": "telemetry/animation-event-log-summary.json",
        "animation_timeline": "telemetry/animation-timeline.json",
        "event_log_sha256": sha256_file(EVENT_PATH),
        "artifact_hash_manifest": "final/artifact-hashes.json",
        "zero_live_e025_instances": read_json(CLEANUP_PATH).get(
            "zero_live_e025_instances"
        ),
        "inference_not_reached": True,
    }
    write_json(FINAL / "render-receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help="Hash the already-rendered artifacts after portable HTML packaging.",
    )
    args = parser.parse_args()
    result = freeze() if args.freeze_only else generate()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
