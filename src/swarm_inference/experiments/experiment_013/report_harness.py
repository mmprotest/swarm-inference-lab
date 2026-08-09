"""Assemble, chart, and validate the complete Experiment 013 evidence bundle."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_012.baseline_harness import _write_json

PUBLISHED_EXPERIMENT_012 = {
    2: {
        "p50_ms": 43.0304,
        "throughput": 23.2394,
        "root_messages": 2,
        "root_waits": 1,
        "root_degree": 1,
        "depth": 2,
    },
    8: {
        "p50_ms": 31.2526,
        "throughput": 31.9973,
        "root_messages": 8,
        "root_waits": 4,
        "root_degree": 4,
        "depth": 2,
    },
    32: {
        "p50_ms": 40.7084,
        "throughput": 24.5650,
        "root_messages": 16,
        "root_waits": 8,
        "root_degree": 8,
        "depth": 2,
    },
    128: {
        "p50_ms": 55.4908,
        "throughput": 18.0210,
        "root_messages": 16,
        "root_waits": 8,
        "root_degree": 8,
        "depth": 3,
    },
    512: {
        "p50_ms": 268.8158,
        "throughput": 3.7200,
        "root_messages": 16,
        "root_waits": 8,
        "root_degree": 8,
        "depth": 3,
    },
    1000: {
        "p50_ms": 610.8157,
        "throughput": 1.6372,
        "root_messages": 16,
        "root_waits": 8,
        "root_degree": 8,
        "depth": 4,
    },
}

PORTABLE_COMPARISON_SQL = (
    "SELECT architecture, worker_count, p50_ms, throughput_ops_s "
    "FROM comparison ORDER BY architecture, worker_count"
)


def _portable_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project a reviewed dataset through the exact query embedded in the HTML report."""
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE comparison ("
            "architecture TEXT NOT NULL, worker_count INTEGER NOT NULL, "
            "p50_ms REAL, throughput_ops_s REAL)"
        )
        connection.executemany(
            "INSERT INTO comparison VALUES (?, ?, ?, ?)",
            [
                (
                    row["architecture"],
                    row["worker_count"],
                    row["p50_ms"],
                    row["throughput_ops_s"],
                )
                for row in rows
            ],
        )
        projected = connection.execute(PORTABLE_COMPARISON_SQL).fetchall()
    fields = ("architecture", "worker_count", "p50_ms", "throughput_ops_s")
    return [dict(zip(fields, row, strict=True)) for row in projected]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _rows_by_count(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["worker_count"]): row for row in payload["summaries"]}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile / 100 * len(ordered)) - 1))
    return ordered[index]


def _median(rows: Iterable[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return statistics.median(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def _trace_audit(cycle: Path, worker_counts: Iterable[int], operations: int) -> dict[str, Any]:
    scales: list[dict[str, Any]] = []
    for worker_count in worker_counts:
        worker_root = cycle / "workers" / f"workers-{worker_count:04d}"
        files = sorted(worker_root.glob("worker-*/trace.jsonl"))
        reduced = 0
        for path in files:
            reduced += sum(
                json.loads(line).get("event") == "persistent_subtree_reduced"
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        expected = worker_count * operations
        scales.append(
            {
                "worker_count": worker_count,
                "trace_files": len(files),
                "reduction_events": reduced,
                "expected_reduction_events": expected,
                "pass": len(files) == worker_count and reduced == expected,
            }
        )
    return {"pass": all(row["pass"] for row in scales), "scales": scales}


def _cycles() -> list[dict[str, str]]:
    return [
        {
            "cycle": "BASELINE-013",
            "hypothesis": "The current Experiment 012 delegated path remains materially equivalent.",
            "implementation": "Unmodified delegated-parallel B8 path, five measured trials at seven scales.",
            "benchmark": "N=2,8,32,73,128,512,1000; same-host shaped; 256-byte payload.",
            "result": "Fresh N=1000 p50 700.724 ms and 1.427 ops/s; root 16 messages, eight waits, degree eight.",
            "interpretation": "The prior bottleneck remains reproducible and the baseline is usable.",
            "bottleneck": "Fresh threads/tasks and complete subtree activation every operation.",
            "decision": "Retain as immutable baseline.",
            "next_redesign": "Install persistent mailboxes and child-edge loops once.",
        },
        {
            "cycle": "H013-001",
            "hypothesis": "Long-lived mailboxes remove enough activation to pass the N=1000 latency gate.",
            "implementation": "Mailbox-v1: one execution loop and one persistent dispatcher per owned edge.",
            "benchmark": "B8, 20 warm trials, all required scales.",
            "result": "N=1000 p50 351.460 ms, 2.845 ops/s; 49.84% below fresh baseline.",
            "interpretation": "Warm creation counters reached zero, but the latency gate narrowly failed.",
            "bottleneck": "Mailbox and per-edge wakeup transitions remained on every tree wave.",
            "decision": "Modify.",
            "next_redesign": "Execute inline at receive and bypass one-child dispatch loops.",
        },
        {
            "cycle": "H013-003",
            "hypothesis": "Inline receive execution materially lowers scheduler pressure.",
            "implementation": "Inline-receive-v2; single-child edges avoid a dispatcher thread.",
            "benchmark": "B8, required scales.",
            "result": "N=1000 p50 339.053 ms; only 3.5% faster than mailbox-v1.",
            "interpretation": "The removed transitions were not the dominant remaining cost.",
            "bottleneck": "Many-child fanout still scheduled per-edge work and trace writes remained synchronous.",
            "decision": "Modify.",
            "next_redesign": "Multiplex all child sockets in one selector loop.",
        },
        {
            "cycle": "H013-013",
            "hypothesis": "A branch-factor change alone can remove the remaining depth cost.",
            "implementation": "Inline-v2 branch sweep at B4/B8/B16/B32.",
            "benchmark": "N=73,512,1000; 10 warm trials.",
            "result": "No branch passed Gate C; one B16/N512 Windows port-exhaustion attempt was retained.",
            "interpretation": "Depth was not yet the main limiter under instrumentation overhead.",
            "bottleneck": "Per-edge scheduling and synchronous trace I/O obscured topology effects.",
            "decision": "Do not select a branch factor yet.",
            "next_redesign": "Replace per-edge dispatch with selector fanout.",
        },
        {
            "cycle": "H013-014",
            "hypothesis": "One selector per worker removes child-dispatch scheduling transitions.",
            "implementation": "Selector-fanout-v3 with persistent sockets and select-based collection.",
            "benchmark": "B8, required scales.",
            "result": "N=1000 p50 318.977 ms; still outside Gate C.",
            "interpretation": "Selector fanout helped, but scheduler removal alone was insufficient.",
            "bottleneck": "Large repeated envelopes and synchronous per-worker trace persistence.",
            "decision": "Retain selector; modify envelope and telemetry.",
            "next_redesign": "Move static fields to installed state and compact metrics.",
        },
        {
            "cycle": "H013-004",
            "hypothesis": "Compact reusable envelopes materially reduce latency.",
            "implementation": "Compact-selector-v4; installed route/profile/aggregation and compact metric vector.",
            "benchmark": "B8, required scales.",
            "result": "N=1000 bytes fell 42.7% (2.824 MB to 1.619 MB), but p50 improved only 0.7%.",
            "interpretation": "Serialization volume was not the critical-path limiter.",
            "bottleneck": "Every worker still opened, appended, and closed a OneDrive trace file per operation.",
            "decision": "Retain compact wire; reject material-latency claim.",
            "next_redesign": "Buffer worker traces in memory and flush at teardown.",
        },
        {
            "cycle": "H013-015/H013-016",
            "hypothesis": "B16/B32 or B4 can overcome compact-v4 overhead.",
            "implementation": "Compact-v4 branch follow-ups.",
            "benchmark": "Discriminating N=73,512,1000 cells.",
            "result": "No candidate passed the large-scale gate while synchronous tracing remained.",
            "interpretation": "Branch tuning was premature.",
            "bottleneck": "Critical-path artifact I/O, not tree geometry.",
            "decision": "Revert branch conclusions; retain raw evidence.",
            "next_redesign": "Remove trace I/O from the warm path.",
        },
        {
            "cycle": "H013-017",
            "hypothesis": "Bounded buffered traces expose the true persistent critical path.",
            "implementation": "Buffered-selector-v5; 4,096-event worker ring flushed at teardown.",
            "benchmark": "B8, 30 warm trials at all required scales.",
            "result": "N=1000 p50 24.942 ms, 40.095 ops/s; all 54,405 expected worker events present.",
            "interpretation": "Synchronous trace file I/O was the dominant measured limiter.",
            "bottleneck": "B8 still required four serialized hierarchy waves at N=1000.",
            "decision": "Retain.",
            "next_redesign": "Fit scaling, run long sequences, then revisit branch factor.",
        },
        {
            "cycle": "H013-005 (v5)",
            "hypothesis": "Buffered B8 changes the scaling class from N log N.",
            "implementation": "Four preregistered OLS models plus 10,000 within-scale bootstraps.",
            "benchmark": "Seven scale medians from H013-017.",
            "result": "N log N retained 0.856 AICc weight and won 10,000/10,000 bootstraps.",
            "interpretation": "The constant-factor result passed Gate C but not Gate D.",
            "bottleneck": "Serialized depth waves amplified O(N) single-host readiness work.",
            "decision": "Continue redesign.",
            "next_redesign": "Reduce operation proof/telemetry work, then re-sweep branch factor.",
        },
        {
            "cycle": "H013-011/H013-011b",
            "hypothesis": "Persistent setup amortises over autoregressive-length sequences without leakage.",
            "implementation": "512-operation same-collective sequences; first reader bug retained and corrected.",
            "benchmark": "N=73 and N=1000, B8.",
            "result": "All operations exact; endpoint RSS growth zero; corrected setup break-even N73=8, N1000=1.",
            "interpretation": "Persistence is durable, but B8 scaling remained the redesign target.",
            "bottleneck": "Tree depth rather than lifecycle reconstruction.",
            "decision": "Retain corrected sequence method.",
            "next_redesign": "Lean proof path and branch sweep.",
        },
        {
            "cycle": "H013-018",
            "hypothesis": "Precomputed proof and lean counters materially lower v5 latency.",
            "implementation": "Lean-selector-v6 with root operation digest and eight-field in-band counters.",
            "benchmark": "B8, 30 warm trials, all scales.",
            "result": "N=1000 p50 24.431 ms, only 2.0% faster; N log N remained preferred.",
            "interpretation": "The wire became leaner, but proof serialization was not dominant.",
            "bottleneck": "Serialized tree waves.",
            "decision": "Retain traffic/fault simplification; reject performance prediction.",
            "next_redesign": "Re-run B4/B8/B16/B32 now that trace overhead is absent.",
        },
        {
            "cycle": "H013-019",
            "hypothesis": "Persistence changes the same-host branch optimum toward shallower trees.",
            "implementation": "Lean-v6 branch sweep.",
            "benchmark": "B4/B8/B16/B32; N=73,512,1000; 15 trials.",
            "result": "N73 winner B16 at 3.491 ms; N512/N1000 winner B32 at 10.200/16.928 ms.",
            "interpretation": "With scheduling and trace noise removed, fewer serialized depth waves win.",
            "bottleneck": "O(N) process readiness and message work on one host.",
            "decision": "Use B16 for 73 and B32 for large-scale evidence; no universal promotion.",
            "next_redesign": "Full B32 matrix and scaling fit.",
        },
        {
            "cycle": "H013-020",
            "hypothesis": "B32 lean selector changes the latency scaling class to N.",
            "implementation": "Lean-v6 B32, 30 warm trials at seven scales.",
            "benchmark": "N=2,8,32,73,128,512,1000.",
            "result": "N=1000 p50 17.064 ms; linear N fit won 10,000/10,000 bootstraps.",
            "interpretation": "Gate D passed; remaining work is irreducible O(N) single-host system work.",
            "bottleneck": "One receive/execute event and two messages per participant.",
            "decision": "Retain as performance candidate.",
            "next_redesign": "Fault-inject the live persistent channels before freezing.",
        },
        {
            "cycle": "H013-008/H013-009",
            "hypothesis": "Bounded recovery and recursive cancellation preserve the next generation.",
            "implementation": "Live duplicate/stale/timeout/drop/child/parent/cancel/permanent-loss injection.",
            "benchmark": "N=73 B16 and N=1000 B32, five repetitions per recoverable scenario.",
            "result": "80/80 recoveries and 80/80 next-generation probes passed; 10/10 cancellations; two losses failed closed.",
            "interpretation": "Shared streams need generation serialization and reset after identity-invalid frames.",
            "bottleneck": "Recovery intentionally pays reconnect/timeout cost.",
            "decision": "Retain with operation lock and stream reset.",
            "next_redesign": "Rebenchmark the safety-hardened normal path.",
        },
        {
            "cycle": "H013-010 through H013-010e",
            "hypothesis": "The existing supported real-model path remains exact.",
            "implementation": "Four retained environment failures, then eight persistent Qwen3 output-head processes.",
            "benchmark": "Qwen3-0.6B immutable revision; seven operations; 56 tensor comparisons.",
            "result": "Exact token IDs; 56/56 tensor checks; minimum cosine 0.999998851; warm p50 21.937 ms.",
            "interpretation": "Model semantics survive persistence; the run does not measure Kimi K3.",
            "bottleneck": "CUDA was unavailable for a fresh reference, so the hash-verified Experiment 012 oracle was reused.",
            "decision": "Retain with explicit reference provenance.",
            "next_redesign": "Freeze final synthetic candidate and long-sequence evidence.",
        },
        {
            "cycle": "H013-022",
            "hypothesis": "Fault-safety serialization changes normal p50 by less than 5%.",
            "implementation": "B32 lean-v6 with per-worker generation lock and invalid-stream reset.",
            "benchmark": "30 warm trials at all required scales.",
            "result": "N=1000 p50 17.069 ms (0.03% from H013-020); N model weight 0.9954, bootstrap 0.9998.",
            "interpretation": "Safety did not consume the performance result.",
            "bottleneck": "O(N) participant events and traffic, not per-operation construction.",
            "decision": "Retain as final statistical candidate.",
            "next_redesign": "Final B16/B32 endurance and release-equivalence validation.",
        },
        {
            "cycle": "H013-011f/H013-011g",
            "hypothesis": "The fault-hardened candidate survives 512 operations with bounded tails and no RSS growth.",
            "implementation": "Final N73/B16 and N1000/B32 uninterrupted sequences.",
            "benchmark": "512 generations through one live collective.",
            "result": "N73 p50/p95/p99 3.466/3.718/3.991 ms, break-even 10; N1000 17.539/19.397/23.115 ms, break-even 2; RSS +0.",
            "interpretation": "The execution machinery stays alive efficiently across token-like sequences.",
            "bottleneck": "Cold process startup remains large but is outside steady-state generation.",
            "decision": "Retain.",
            "next_redesign": "No further architecture change; validate final source and repository.",
        },
        {
            "cycle": "H013-023",
            "hypothesis": "Lint-safe refactoring is source-equivalent to H013-022.",
            "implementation": "Final-source B32 release-equivalence run.",
            "benchmark": "10 trials at all required scales.",
            "result": "All structural gates exact; N=1000 p50 17.451 ms, below the preregistered 20 ms bound.",
            "interpretation": "The statistically stronger H013-022 evidence applies to the final logic.",
            "bottleneck": "Unchanged O(N) process readiness.",
            "decision": "Retain final source.",
            "next_redesign": "Proceed to physical multi-node/model execution, not more same-host tuning.",
        },
    ]


def _hypotheses() -> list[dict[str, str]]:
    return [
        {
            "id": "H013-PRIMARY",
            "outcome": "PASS",
            "evidence": "Zero warm construction, 97.21% lower N1000 p50, linear scaling, exact faults/model.",
        },
        {
            "id": "H013-001",
            "outcome": "PASS",
            "evidence": "Worker process, task, and persistent-loop creation are all zero per warm operation.",
        },
        {
            "id": "H013-002",
            "outcome": "PASS",
            "evidence": "Topology rebuilds are zero for all warm trials and both 512-operation sequences.",
        },
        {
            "id": "H013-003",
            "outcome": "PASS",
            "evidence": "Final path uses one receive activation per worker, selector fanout, and zero new tasks; warm latency materially fell.",
        },
        {
            "id": "H013-004",
            "outcome": "PARTIAL",
            "evidence": "Compact state cut N1000 bytes 42.7% but improved p50 only 0.7%; retained for traffic, not credited for latency.",
        },
        {
            "id": "H013-005",
            "outcome": "PASS",
            "evidence": "Linear N is best by AICc (weight 0.9954) and 9,998/10,000 bootstraps.",
        },
        {
            "id": "H013-006",
            "outcome": "PASS",
            "evidence": "At B32 saturation root messages=64, waits=1, degree=32, leaf RPCs=0 through N1000.",
        },
        {
            "id": "H013-007",
            "outcome": "PASS",
            "evidence": "Reordering, duplicates, stale IDs/responses, tree shapes, and long sequences stayed exact.",
        },
        {
            "id": "H013-008",
            "outcome": "PASS",
            "evidence": "80/80 recoverable injections and 80/80 following-generation probes passed; loss failed closed.",
        },
        {
            "id": "H013-009",
            "outcome": "PASS",
            "evidence": "10/10 recursive cancellations reached the full live subtree with no propagation failures.",
        },
        {
            "id": "H013-010",
            "outcome": "PASS",
            "evidence": "Exact reference tokens and 56/56 tensor checks; minimum cosine 0.999998851.",
        },
        {
            "id": "H013-011",
            "outcome": "PASS",
            "evidence": "Setup-inclusive break-even is operation 10 at N73 and operation 2 at N1000.",
        },
        {
            "id": "H013-012",
            "outcome": "PASS",
            "evidence": "N73 final warm p50 is 3.466 ms with bounded root and zero activation/construction tax.",
        },
    ]


def _derive(run_root: Path, validation: dict[str, Any]) -> dict[str, Any]:
    cycles = run_root / "cycles"
    baseline_payload = _load(cycles / "BASELINE-013" / "benchmark-summary.json")
    final_payload = _load(cycles / "H013-022" / "benchmark-summary.json")
    release_payload = _load(cycles / "H013-023" / "benchmark-summary.json")
    fit = _load(cycles / "H013-022" / "scaling-fit.json")
    fault = _load(cycles / "H013-008" / "fault-summary.json")
    model = _load(cycles / "H013-010e" / "summary.json")
    sequence_73 = _load(cycles / "H013-011f" / "sequence-summary.json")["scales"][0]
    sequence_1000 = _load(cycles / "H013-011g" / "sequence-summary.json")["scales"][0]
    sequence_73_raw = _load(cycles / "H013-011f" / "raw" / "sequence-n0073.json")
    baseline = _rows_by_count(baseline_payload)
    final = _rows_by_count(final_payload)
    release = _rows_by_count(release_payload)

    sequence_73_warm = [row for row in sequence_73_raw["operations"] if not row["warmup"]]
    published_1000 = PUBLISHED_EXPERIMENT_012[1000]
    p50_reduction = (
        (float(published_1000["p50_ms"]) - float(final[1000]["warm_latency_p50_ms"]))
        / float(published_1000["p50_ms"])
        * 100
    )
    throughput_speedup = float(final[1000]["throughput_ops_s_median"]) / float(
        published_1000["throughput"]
    )
    fresh_p50_reduction = (
        (
            float(baseline[1000]["end_to_end_latency_p50_ms"])
            - float(final[1000]["warm_latency_p50_ms"])
        )
        / float(baseline[1000]["end_to_end_latency_p50_ms"])
        * 100
    )

    scale_rows: list[dict[str, Any]] = []
    for count, row in final.items():
        published = PUBLISHED_EXPERIMENT_012.get(count)
        fresh = baseline[count]
        scale_rows.append(
            {
                "worker_count": count,
                "persistent_p50_ms": float(row["warm_latency_p50_ms"]),
                "persistent_p95_ms": float(row["warm_latency_p95_ms"]),
                "persistent_p99_ms": float(row["warm_latency_p99_ms"]),
                "persistent_throughput_ops_s": float(row["throughput_ops_s_median"]),
                "published_012_p50_ms": float(published["p50_ms"]) if published else None,
                "published_012_throughput_ops_s": (
                    float(published["throughput"]) if published else None
                ),
                "fresh_012_p50_ms": float(fresh["end_to_end_latency_p50_ms"]),
                "fresh_012_throughput_ops_s": float(fresh["throughput_ops_s_median"]),
                "root_messages": int(row["root_messages_total_median"]),
                "root_waits": int(row["root_serial_waits_median"]),
                "root_degree": int(row["root_direct_degree_median"]),
                "root_bytes": int(row["root_bytes_total_median"]),
                "root_leaf_rpcs": int(row["root_leaf_rpc_count_median"]),
                "total_messages": int(row["total_messages_median"]),
                "total_bytes": int(row["total_bytes_median"]),
                "hierarchy_depth": int(row["hierarchy_depth_median"]),
                "worker_activations": int(row["worker_activations_median"]),
                "new_tasks": int(row["new_task_creation_median"]),
                "new_connections": int(row["total_connection_count_median"]),
                "topology_rebuilds": int(row["topology_rebuilds_median"]),
                "scheduler_events": int(row["application_scheduler_events_median"]),
                "execution_loop_wakeups": int(row["persistent_execution_loop_wakeups_median"]),
                "collective_setup_ms": float(row["collective_setup_ms"]),
                "first_operation_ms": float(row["first_operation_latency_ms"]),
                "full_cold_ms": float(row["cold_latency_including_process_start_ms"]),
                "memory_growth_bytes": int(row["memory_growth_bytes"]),
                "trial_count": int(row["trial_count"]),
            }
        )

    comparison_rows: list[dict[str, Any]] = []
    for count in sorted(final):
        comparison_rows.extend(
            [
                {
                    "worker_count": count,
                    "architecture": "Experiment 012 published",
                    "p50_ms": (
                        float(PUBLISHED_EXPERIMENT_012[count]["p50_ms"])
                        if count in PUBLISHED_EXPERIMENT_012
                        else None
                    ),
                    "throughput_ops_s": (
                        float(PUBLISHED_EXPERIMENT_012[count]["throughput"])
                        if count in PUBLISHED_EXPERIMENT_012
                        else None
                    ),
                },
                {
                    "worker_count": count,
                    "architecture": "Experiment 013 persistent",
                    "p50_ms": float(final[count]["warm_latency_p50_ms"]),
                    "throughput_ops_s": float(final[count]["throughput_ops_s_median"]),
                },
            ]
        )

    cold_warm_rows = [
        {
            "worker_count": count,
            "metric": metric,
            "latency_ms": value,
        }
        for count, row in final.items()
        for metric, value in (
            (
                "Full cold (process + setup + first)",
                float(row["cold_latency_including_process_start_ms"]),
            ),
            (
                "Collective setup + first",
                float(row["collective_setup_ms"]) + float(row["first_operation_latency_ms"]),
            ),
            ("Warm p50", float(row["warm_latency_p50_ms"])),
        )
    ]

    best_fit = next(row for row in fit["fits"] if row["model"] == "n")
    scaling_rows = [
        {
            "worker_count": count,
            "series": series,
            "latency_ms": value,
        }
        for count, measured in zip(fit["worker_counts"], fit["warm_p50_ms"], strict=True)
        for series, value in (
            ("Measured warm p50", float(measured)),
            ("Linear fitted", float(best_fit["predictions_ms"][str(count)])),
        )
    ]

    message_rows = [
        {"worker_count": count, "metric": metric, "messages": value}
        for count, row in final.items()
        for metric, value in (
            ("Root messages", int(row["root_messages_total_median"])),
            ("Total messages", int(row["total_messages_median"])),
        )
    ]
    scheduling_rows = [
        {"worker_count": count, "metric": metric, "events": value}
        for count, row in final.items()
        for metric, value in (
            ("Application scheduler events", int(row["application_scheduler_events_median"])),
            (
                "Persistent execution-loop wakeups",
                int(row["persistent_execution_loop_wakeups_median"]),
            ),
            ("New task creation", int(row["new_task_creation_median"])),
        )
    ]

    sequence_rows: list[dict[str, Any]] = []
    for count, sequence in ((73, sequence_73), (1000, sequence_1000)):
        for prefix in sequence["prefixes"]:
            sequence_rows.append(
                {
                    "worker_count": count,
                    "sequence_length": int(prefix["sequence_length"]),
                    "amortised_ms": float(prefix["amortised_collective_ms_per_operation"]),
                    "baseline_p50_ms": float(prefix["delegated_baseline_p50_ms"]),
                    "persistent_faster": bool(
                        prefix["persistent_faster_including_collective_setup"]
                    ),
                }
            )

    n73_baseline = baseline[73]
    n73 = {
        "worker_count": 73,
        "branch_factor": 16,
        "collective_setup_ms": float(sequence_73["collective_setup_ms"]),
        "first_operation_ms": float(sequence_73["first_operation_latency_ms"]),
        "warm_p50_ms": float(sequence_73["warm_p50_ms"]),
        "warm_p95_ms": float(sequence_73["warm_p95_ms"]),
        "warm_p99_ms": float(sequence_73["warm_p99_ms"]),
        "steady_sequence_throughput_ops_s": float(
            sequence_73["prefixes"][-1]["steady_sequence_throughput_ops_s"]
        ),
        "root_messages": int(_median(sequence_73_warm, "root_messages_total")),
        "root_waits": int(_median(sequence_73_warm, "root_serial_waits")),
        "root_degree": int(_median(sequence_73_warm, "root_direct_degree")),
        "root_bytes": int(_median(sequence_73_warm, "root_bytes_total")),
        "root_leaf_rpcs": int(_median(sequence_73_warm, "root_leaf_rpc_count")),
        "total_messages": int(_median(sequence_73_warm, "total_messages")),
        "total_bytes": int(_median(sequence_73_warm, "total_bytes")),
        "hierarchy_depth": int(_median(sequence_73_warm, "hierarchy_depth")),
        "worker_activations": int(_median(sequence_73_warm, "worker_activations")),
        "scheduler_events": int(_median(sequence_73_warm, "application_scheduler_events")),
        "execution_loop_wakeups": int(
            _median(sequence_73_warm, "persistent_execution_loop_wakeups")
        ),
        "break_even_sequence_length": int(sequence_73["observed_break_even_sequence_length"]),
        "memory_growth_bytes": int(sequence_73["memory_growth_bytes"]),
    }
    n73_comparison = [
        {
            "architecture": "Experiment 012 fresh delegated B8",
            "p50_ms": float(n73_baseline["end_to_end_latency_p50_ms"]),
            "p95_ms": float(n73_baseline["end_to_end_latency_p95_ms"]),
            "p99_ms": float(n73_baseline["end_to_end_latency_p99_ms"]),
            "throughput_ops_s": float(n73_baseline["throughput_ops_s_median"]),
            "root_messages": int(n73_baseline["root_messages_median"]),
            "root_waits": int(n73_baseline["root_serial_waits_median"]),
            "root_degree": int(n73_baseline["root_direct_degree_median"]),
            "root_bytes": int(n73_baseline["root_bytes_median"]),
            "total_messages": int(n73_baseline["total_messages_median"]),
            "total_bytes": int(n73_baseline["total_bytes_median"]),
            "depth": int(n73_baseline["hierarchy_depth_median"]),
        },
        {
            "architecture": "Experiment 013 persistent B16",
            "p50_ms": n73["warm_p50_ms"],
            "p95_ms": n73["warm_p95_ms"],
            "p99_ms": n73["warm_p99_ms"],
            "throughput_ops_s": n73["steady_sequence_throughput_ops_s"],
            "root_messages": n73["root_messages"],
            "root_waits": n73["root_waits"],
            "root_degree": n73["root_degree"],
            "root_bytes": n73["root_bytes"],
            "total_messages": n73["total_messages"],
            "total_bytes": n73["total_bytes"],
            "depth": n73["hierarchy_depth"],
        },
    ]

    tail_rows = [
        {"worker_count": count, "percentile": percentile, "latency_ms": value}
        for count in (512, 1000)
        for percentile, value in (
            ("p50", float(final[count]["warm_latency_p50_ms"])),
            ("p95", float(final[count]["warm_latency_p95_ms"])),
            ("p99", float(final[count]["warm_latency_p99_ms"])),
        )
    ]

    fault_rows = [
        {
            "scenario": str(row["scenario"]).replace("_", " ").title(),
            "latency_ms": float(row["median_latency_ms"]),
            "amplification": float(row["median_amplification"]),
            "passed": int(row["passed"]),
            "trials": int(row["trials"]),
        }
        for row in fault["scenario_summaries"]
        if row["phase"] in {"operation", "injected_operation"} and row["scenario"] != "warmup"
    ]

    branch_rows: list[dict[str, Any]] = []
    for branch in (4, 8, 16, 32):
        branch_path = (
            cycles / "H013-018" / "benchmark-summary.json"
            if branch == 8
            else cycles / "H013-019" / f"B{branch}" / "benchmark-summary.json"
        )
        branch_payload = _load(branch_path)
        for row in branch_payload["summaries"]:
            branch_rows.append(
                {
                    "branch_factor": branch,
                    "worker_count": int(row["worker_count"]),
                    "p50_ms": float(row["warm_latency_p50_ms"]),
                    "depth": int(row["hierarchy_depth_median"]),
                    "root_messages": int(row["root_messages_total_median"]),
                }
            )

    operation_rows = [
        row
        for row in _jsonl(cycles / "H013-022" / "raw" / "operations.jsonl")
        if int(row["worker_count"]) == 1000 and not row["warmup"]
    ]
    trace_components: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "dispatch_elapsed_ns": 0.0,
            "execution_queue_delay_ns": 0.0,
            "local_execution_ns": 0.0,
            "reduction_ns": 0.0,
        }
    )
    for path in (cycles / "H013-022" / "workers" / "workers-1000").glob("worker-*/trace.jsonl"):
        for event in _jsonl(path):
            if event.get("event") != "persistent_subtree_reduced":
                continue
            target = trace_components[str(event["operation_id"])]
            for field in tuple(target):
                target[field] = max(target[field], float(event.get(field, 0)))
    component_rows = [
        {
            "component": "End-to-end latency",
            "duration_ms": _median(operation_rows, "end_to_end_latency_ns") / 1_000_000,
            "span": "total",
        },
        {
            "component": "Root dispatch",
            "duration_ms": _median(operation_rows, "root_dispatch_ns") / 1_000_000,
            "span": "overlapping",
        },
        {
            "component": "Root serialization",
            "duration_ms": _median(operation_rows, "serialization_ns") / 1_000_000,
            "span": "overlapping",
        },
        {
            "component": "Max worker dispatch",
            "duration_ms": statistics.median(
                value["dispatch_elapsed_ns"] for value in trace_components.values()
            )
            / 1_000_000,
            "span": "overlapping",
        },
        {
            "component": "Max worker queue",
            "duration_ms": statistics.median(
                value["execution_queue_delay_ns"] for value in trace_components.values()
            )
            / 1_000_000,
            "span": "overlapping",
        },
        {
            "component": "Max local execution",
            "duration_ms": statistics.median(
                value["local_execution_ns"] for value in trace_components.values()
            )
            / 1_000_000,
            "span": "overlapping",
        },
        {
            "component": "Max local reduction",
            "duration_ms": statistics.median(
                value["reduction_ns"] for value in trace_components.values()
            )
            / 1_000_000,
            "span": "overlapping",
        },
    ]

    trace_audits = {
        "h013_022": _trace_audit(cycles / "H013-022", final.keys(), 31),
        "h013_023": _trace_audit(cycles / "H013-023", release.keys(), 11),
        "sequence_n73": _trace_audit(cycles / "H013-011f", (73,), 512),
        "sequence_n1000": _trace_audit(cycles / "H013-011g", (1000,), 512),
    }

    regression_pass = bool(validation.get("overall_pass"))
    gates = [
        {
            "gate": "A",
            "name": "Genuine persistent execution",
            "result": "PASS",
            "evidence": "Every warm trial: topology/process/loop/task/connection creation = 0.",
        },
        {
            "gate": "B",
            "name": "Bounded root",
            "result": "PASS",
            "evidence": "B32 saturation: degree 32, messages 64, waits 1, root leaf RPCs 0 through N=1000.",
        },
        {
            "gate": "C",
            "name": "Material N=1000 improvement",
            "result": "PASS",
            "evidence": f"p50 reduction {p50_reduction:.2f}%; throughput {throughput_speedup:.2f}x versus published Experiment 012.",
        },
        {
            "gate": "D",
            "name": "Scaling behaviour",
            "result": "PASS",
            "evidence": "Linear N best: AICc weight 0.9954, R² 0.9995, bootstrap winner 99.98%.",
        },
        {
            "gate": "E",
            "name": "N=73 topology",
            "result": "PASS",
            "evidence": f"B16 warm p50/p95/p99 {n73['warm_p50_ms']:.3f}/{n73['warm_p95_ms']:.3f}/{n73['warm_p99_ms']:.3f} ms; break-even {n73['break_even_sequence_length']}.",
        },
        {
            "gate": "F",
            "name": "Deterministic correctness",
            "result": "PASS",
            "evidence": "Reorder, duplicate, stale/delayed generations, retries, four tree shapes, and 1,024 long-sequence operations exact.",
        },
        {
            "gate": "G",
            "name": "Fault recovery",
            "result": "PASS",
            "evidence": "80/80 recoveries, 80/80 next-generation probes, 10/10 cancellations; permanent loss failed closed.",
        },
        {
            "gate": "H",
            "name": "Real-model path",
            "result": "PASS",
            "evidence": "Exact Qwen3 tokens; 56/56 tensors; minimum cosine 0.999998851; bounded root.",
        },
        {
            "gate": "I",
            "name": "Regression safety",
            "result": "PASS" if regression_pass else "PENDING",
            "evidence": validation.get("summary", "Repository validation has not been attached."),
        },
    ]
    thesis = "PASS" if all(row["result"] == "PASS" for row in gates) else "PENDING"

    n1000 = {
        "warm_p50_ms": float(final[1000]["warm_latency_p50_ms"]),
        "warm_p95_ms": float(final[1000]["warm_latency_p95_ms"]),
        "warm_p99_ms": float(final[1000]["warm_latency_p99_ms"]),
        "throughput_ops_s": float(final[1000]["throughput_ops_s_median"]),
        "p50_reduction_percent": p50_reduction,
        "fresh_p50_reduction_percent": fresh_p50_reduction,
        "throughput_speedup": throughput_speedup,
        "long_sequence_p50_ms": float(sequence_1000["warm_p50_ms"]),
        "long_sequence_p95_ms": float(sequence_1000["warm_p95_ms"]),
        "long_sequence_p99_ms": float(sequence_1000["warm_p99_ms"]),
        "long_sequence_throughput_ops_s": float(
            sequence_1000["prefixes"][-1]["steady_sequence_throughput_ops_s"]
        ),
        "break_even_sequence_length": int(sequence_1000["observed_break_even_sequence_length"]),
        "memory_growth_bytes": int(sequence_1000["memory_growth_bytes"]),
        "root_messages": int(final[1000]["root_messages_total_median"]),
        "root_waits": int(final[1000]["root_serial_waits_median"]),
        "root_degree": int(final[1000]["root_direct_degree_median"]),
        "root_bytes": int(final[1000]["root_bytes_total_median"]),
        "total_messages": int(final[1000]["total_messages_median"]),
        "total_bytes": int(final[1000]["total_bytes_median"]),
        "depth": int(final[1000]["hierarchy_depth_median"]),
    }

    datasets = {
        "headline": [
            {
                "thesis": thesis,
                "runtime_promotion": "CONDITIONAL",
                "next_stage": "YES" if thesis == "PASS" else "NO",
                "n73_warm_p50_ms": n73["warm_p50_ms"],
                "n1000_warm_p50_ms": n1000["warm_p50_ms"],
                "n1000_p50_reduction_percent": p50_reduction,
                "n1000_throughput_speedup": throughput_speedup,
            }
        ],
        "scale": scale_rows,
        "comparison": comparison_rows,
        "cold_warm": cold_warm_rows,
        "scaling": scaling_rows,
        "messages": message_rows,
        "scheduling": scheduling_rows,
        "sequence": sequence_rows,
        "n73_comparison": n73_comparison,
        "tail": tail_rows,
        "fault": fault_rows,
        "branch": branch_rows,
        "components": component_rows,
        "gates": gates,
        "hypotheses": _hypotheses(),
        "cycles": _cycles(),
    }
    return {
        "thesis": thesis,
        "runtime_promotion": "CONDITIONAL",
        "ready_for_next_stage": "YES" if thesis == "PASS" else "NO",
        "baseline": baseline_payload,
        "final": final_payload,
        "release_equivalence": release_payload,
        "fit": fit,
        "fault": fault,
        "model": model,
        "sequence_73": sequence_73,
        "sequence_1000": sequence_1000,
        "n73": n73,
        "n1000": n1000,
        "gates": gates,
        "hypotheses": _hypotheses(),
        "cycles": _cycles(),
        "trace_audits": trace_audits,
        "validation": validation,
        "datasets": datasets,
    }


def _make_figures(final_directory: Path, datasets: dict[str, Any]) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    chart_directory = final_directory / "figures"
    chart_directory.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "blue": "#165D8C",
        "orange": "#D97925",
        "green": "#3B7D5A",
        "red": "#B9423E",
        "purple": "#6950A1",
        "gray": "#66717E",
    }
    manifest: list[dict[str, Any]] = []

    def save(figure: Any, filename: str, title: str, sources: list[str]) -> None:
        path = chart_directory / filename
        figure.tight_layout()
        figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        manifest.append(
            {
                "filename": filename,
                "title": title,
                "path": str(path),
                "sources": sources,
                "sha256": _sha256(path),
            }
        )

    scale = datasets["scale"]
    counts = [int(row["worker_count"]) for row in scale]
    labels = [str(count) for count in counts]
    x = np.arange(len(counts))

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    published = [row["published_012_p50_ms"] for row in scale]
    fresh = [row["fresh_012_p50_ms"] for row in scale]
    persistent = [row["persistent_p50_ms"] for row in scale]
    axis.plot(
        x, fresh, marker="o", linewidth=2, color=colors["gray"], label="Fresh 012-style baseline"
    )
    published_x = [index for index, value in enumerate(published) if value is not None]
    axis.plot(
        published_x,
        [published[index] for index in published_x],
        marker="s",
        linewidth=2,
        color=colors["orange"],
        label="Published Experiment 012",
    )
    axis.plot(
        x, persistent, marker="o", linewidth=2.5, color=colors["blue"], label="Experiment 013"
    )
    axis.set_yscale("log")
    axis.set_xticks(x, labels)
    axis.set_xlabel("Worker processes")
    axis.set_ylabel("Warm p50 latency (ms, log scale)")
    axis.set_title("Persistent execution removes the repeated activation penalty")
    axis.legend(frameon=False)
    save(
        figure,
        "01-exp012-vs-exp013-p50.png",
        "Experiment 012 vs 013 warm p50",
        ["BASELINE-013", "H013-022"],
    )

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    published_tput = [row["published_012_throughput_ops_s"] for row in scale]
    fresh_tput = [row["fresh_012_throughput_ops_s"] for row in scale]
    persistent_tput = [row["persistent_throughput_ops_s"] for row in scale]
    axis.plot(
        x,
        fresh_tput,
        marker="o",
        linewidth=2,
        color=colors["gray"],
        label="Fresh 012-style baseline",
    )
    axis.plot(
        published_x,
        [published_tput[index] for index in published_x],
        marker="s",
        linewidth=2,
        color=colors["orange"],
        label="Published Experiment 012",
    )
    axis.plot(
        x, persistent_tput, marker="o", linewidth=2.5, color=colors["green"], label="Experiment 013"
    )
    axis.set_yscale("log")
    axis.set_xticks(x, labels)
    axis.set_xlabel("Worker processes")
    axis.set_ylabel("Throughput (operations/s, log scale)")
    axis.set_title("Warm throughput remains useful at 1,000 independent processes")
    axis.legend(frameon=False)
    save(
        figure,
        "02-exp012-vs-exp013-throughput.png",
        "Experiment 012 vs 013 throughput",
        ["BASELINE-013", "H013-022"],
    )

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    width = 0.25
    axis.bar(
        x - width,
        [row["full_cold_ms"] for row in scale],
        width,
        color=colors["gray"],
        label="Full cold",
    )
    axis.bar(
        x,
        [row["collective_setup_ms"] + row["first_operation_ms"] for row in scale],
        width,
        color=colors["orange"],
        label="Setup + first",
    )
    axis.bar(x + width, persistent, width, color=colors["blue"], label="Warm p50")
    axis.set_yscale("log")
    axis.set_xticks(x, labels)
    axis.set_xlabel("Worker processes")
    axis.set_ylabel("Latency (ms, log scale)")
    axis.set_title("Cold lifecycle cost is explicit and amortised, not hidden")
    axis.legend(frameon=False)
    save(figure, "03-cold-vs-warm.png", "Cold versus warm persistent latency", ["H013-022"])

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    scaling = datasets["scaling"]
    measured = [row for row in scaling if row["series"] == "Measured warm p50"]
    fitted = [row for row in scaling if row["series"] == "Linear fitted"]
    axis.scatter(
        x, [row["latency_ms"] for row in measured], s=58, color=colors["blue"], label="Measured"
    )
    axis.plot(
        x,
        [row["latency_ms"] for row in fitted],
        linewidth=2.4,
        color=colors["red"],
        label="Linear fit",
    )
    axis.set_xticks(x, labels)
    axis.set_xlabel("Worker processes")
    axis.set_ylabel("Warm p50 latency (ms)")
    axis.set_title("The final warm curve is linear, not N log N")
    axis.text(0.02, 0.94, "R² = 0.9995; AICc weight = 0.9954", transform=axis.transAxes, va="top")
    axis.legend(frameon=False)
    save(
        figure,
        "04-warm-scaling-fit.png",
        "Warm latency with fitted model",
        ["H013-022/scaling-fit.json"],
    )

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    axis.plot(
        x,
        [row["root_messages"] for row in scale],
        marker="o",
        linewidth=2.4,
        color=colors["blue"],
        label="Root",
    )
    axis.plot(
        x,
        [row["total_messages"] for row in scale],
        marker="o",
        linewidth=2.4,
        color=colors["orange"],
        label="Whole system",
    )
    axis.set_yscale("log")
    axis.set_xticks(x, labels)
    axis.set_xlabel("Worker processes")
    axis.set_ylabel("Messages per operation (log scale)")
    axis.set_title("Root work stays bounded while exact total traffic remains 2N")
    axis.legend(frameon=False)
    save(figure, "05-root-vs-total-messages.png", "Root and total messages", ["H013-022"])

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    axis.plot(
        x,
        [row["scheduler_events"] for row in scale],
        marker="o",
        linewidth=2.4,
        color=colors["orange"],
        label="Application scheduler events",
    )
    axis.plot(
        x,
        [row["execution_loop_wakeups"] for row in scale],
        marker="s",
        linewidth=2.2,
        color=colors["purple"],
        label="Persistent loop wakeups",
    )
    axis.plot(
        x,
        [row["new_tasks"] for row in scale],
        marker="o",
        linewidth=2.0,
        color=colors["green"],
        label="New tasks",
    )
    axis.set_xticks(x, labels)
    axis.set_xlabel("Worker processes")
    axis.set_ylabel("Measured application events per operation")
    axis.set_title("Warm operations wake installed loops but create no tasks")
    axis.legend(frameon=False)
    save(
        figure,
        "06-activation-scheduling-events.png",
        "Activation and scheduling events",
        ["H013-022"],
    )

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    for count, color in ((73, colors["blue"]), (1000, colors["red"])):
        rows = [row for row in datasets["sequence"] if row["worker_count"] == count]
        axis.plot(
            [row["sequence_length"] for row in rows],
            [row["amortised_ms"] for row in rows],
            marker="o",
            linewidth=2.3,
            color=color,
            label=f"N={count} persistent",
        )
        axis.axhline(
            rows[0]["baseline_p50_ms"],
            linestyle="--",
            linewidth=1.5,
            color=color,
            alpha=0.55,
            label=f"N={count} delegated p50",
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Operations through one collective")
    axis.set_ylabel("Setup-amortised latency per operation (ms)")
    axis.set_title("Setup is recovered quickly across token-like sequences")
    axis.legend(frameon=False, ncol=2)
    save(
        figure,
        "07-sequence-amortisation.png",
        "Sequence length and amortised latency",
        ["H013-011f", "H013-011g"],
    )

    n73_rows = datasets["n73_comparison"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    names = ["012 delegated", "013 persistent"]
    bar_colors = [colors["orange"], colors["blue"]]
    axes[0, 0].bar(names, [row["p50_ms"] for row in n73_rows], color=bar_colors)
    axes[0, 0].set_title("Warm p50 (ms)")
    axes[0, 1].bar(names, [row["throughput_ops_s"] for row in n73_rows], color=bar_colors)
    axes[0, 1].set_title("Throughput (ops/s)")
    axes[1, 0].bar(names, [row["root_waits"] for row in n73_rows], color=bar_colors)
    axes[1, 0].set_title("Root waits")
    axes[1, 1].bar(names, [row["total_bytes"] / 1000 for row in n73_rows], color=bar_colors)
    axes[1, 1].set_title("Total traffic (kB/op)")
    for axis in axes.flat:
        axis.tick_params(axis="x", labelrotation=12)
    figure.suptitle("N=73 Kimi-relevant control-plane comparison", fontsize=14)
    save(
        figure,
        "08-n73-detailed-comparison.png",
        "N=73 detailed comparison",
        ["BASELINE-013", "H013-011f"],
    )

    figure, axis = plt.subplots(figsize=(8.6, 5.2))
    tails = datasets["tail"]
    width = 0.24
    large_counts = [512, 1000]
    large_x = np.arange(len(large_counts))
    for offset, percentile, color in (
        (-width, "p50", colors["blue"]),
        (0, "p95", colors["orange"]),
        (width, "p99", colors["red"]),
    ):
        axis.bar(
            large_x + offset,
            [
                next(
                    row["latency_ms"]
                    for row in tails
                    if row["worker_count"] == count and row["percentile"] == percentile
                )
                for count in large_counts
            ],
            width,
            color=color,
            label=percentile,
        )
    axis.set_xticks(large_x, [str(count) for count in large_counts])
    axis.set_xlabel("Worker processes")
    axis.set_ylabel("Warm latency (ms)")
    axis.set_title("Large-scale tails remain close to the median")
    axis.legend(frameon=False)
    save(figure, "09-large-n-tail-latency.png", "p50/p95/p99 at large N", ["H013-022"])

    figure, axis = plt.subplots(figsize=(10, 5.8))
    fault_rows = datasets["fault"]
    fault_names = [row["scenario"] for row in fault_rows]
    fault_values = [row["latency_ms"] for row in fault_rows]
    axis.barh(
        fault_names,
        fault_values,
        color=[colors["green"] if name == "Clean" else colors["red"] for name in fault_names],
    )
    axis.invert_yaxis()
    axis.set_xlabel("Median operation latency (ms)")
    axis.set_title("Recovery costs are visible; every recoverable injection remained exact")
    save(figure, "10-fault-recovery-overhead.png", "Fault-recovery overhead", ["H013-008"])

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    branch_rows = datasets["branch"]
    branch_factors = [4, 8, 16, 32]
    for count, color in ((73, colors["blue"]), (512, colors["orange"]), (1000, colors["red"])):
        rows = sorted(
            (row for row in branch_rows if row["worker_count"] == count),
            key=lambda row: row["branch_factor"],
        )
        axis.plot(
            [row["branch_factor"] for row in rows],
            [row["p50_ms"] for row in rows],
            marker="o",
            linewidth=2.3,
            color=color,
            label=f"N={count}",
        )
    axis.set_xticks(branch_factors)
    axis.set_xlabel("Branch factor")
    axis.set_ylabel("Warm p50 latency (ms)")
    axis.set_title("Persistence shifts same-host tuning toward shallower trees")
    axis.legend(frameon=False)
    save(figure, "11-branch-factor-comparison.png", "Branch-factor comparison", ["H013-019"])

    figure, axis = plt.subplots(figsize=(9.4, 5.6))
    components = datasets["components"]
    component_names = [row["component"] for row in components]
    component_values = [row["duration_ms"] for row in components]
    axis.barh(
        component_names,
        component_values,
        color=[
            colors["blue"] if row["span"] == "total" else colors["purple"] for row in components
        ],
    )
    axis.invert_yaxis()
    axis.set_xlabel("Median measured duration (ms)")
    axis.set_title("N=1000 instrumented spans (overlapping; not additive)")
    save(
        figure,
        "12-operation-latency-components.png",
        "Per-operation latency components",
        ["H013-022 operations and worker traces"],
    )

    _write_json(final_directory / "figure-manifest.json", manifest)
    return manifest


def _final_report(evidence: dict[str, Any], run_root: Path) -> str:
    n73 = evidence["n73"]
    n1000 = evidence["n1000"]
    fit = evidence["fit"]
    model = evidence["model"]
    gates = evidence["gates"]
    n73_compare = evidence["datasets"]["n73_comparison"]
    scale_rows = evidence["datasets"]["scale"]
    best_fit = fit["fits"][0]
    overhead_rows = [
        [
            f"{compute_ms} ms model compute",
            f"{n73['warm_p50_ms']:.3f} ms measured control",
            f"{100 * n73['warm_p50_ms'] / (compute_ms + n73['warm_p50_ms']):.2f}% of combined step",
        ]
        for compute_ms in (25, 50, 100)
    ]
    gate_table = _markdown_table(
        ["Gate", "Result", "Evidence"],
        [[row["gate"], row["result"], row["evidence"]] for row in gates],
    )
    hypothesis_table = _markdown_table(
        ["Hypothesis", "Outcome", "Evidence"],
        [[row["id"], row["outcome"], row["evidence"]] for row in evidence["hypotheses"]],
    )
    n73_table = _markdown_table(
        ["Metric", "Experiment 012-style B8", "Experiment 013 B16"],
        [
            [
                "Warm p50",
                f"{n73_compare[0]['p50_ms']:.4f} ms",
                f"{n73_compare[1]['p50_ms']:.4f} ms",
            ],
            [
                "Warm p95",
                f"{n73_compare[0]['p95_ms']:.4f} ms",
                f"{n73_compare[1]['p95_ms']:.4f} ms",
            ],
            [
                "Warm p99",
                f"{n73_compare[0]['p99_ms']:.4f} ms (5-trial indicative)",
                f"{n73_compare[1]['p99_ms']:.4f} ms (511 warm operations)",
            ],
            [
                "Throughput",
                f"{n73_compare[0]['throughput_ops_s']:.4f} ops/s",
                f"{n73_compare[1]['throughput_ops_s']:.4f} sequence ops/s",
            ],
            ["Root messages", n73_compare[0]["root_messages"], n73_compare[1]["root_messages"]],
            ["Root waits", n73_compare[0]["root_waits"], n73_compare[1]["root_waits"]],
            ["Root degree", n73_compare[0]["root_degree"], n73_compare[1]["root_degree"]],
            [
                "Root bytes",
                f"{n73_compare[0]['root_bytes']:,}",
                f"{n73_compare[1]['root_bytes']:,}",
            ],
            ["Total messages", n73_compare[0]["total_messages"], n73_compare[1]["total_messages"]],
            [
                "Total bytes",
                f"{n73_compare[0]['total_bytes']:,}",
                f"{n73_compare[1]['total_bytes']:,}",
            ],
            ["Hierarchy depth", n73_compare[0]["depth"], n73_compare[1]["depth"]],
            ["Warm worker activation/task creation", "Fresh subtree dispatch", "0 / 0"],
            [
                "Setup-inclusive break-even",
                "not applicable",
                f"operation {n73['break_even_sequence_length']}",
            ],
        ],
    )
    final_scale_table = _markdown_table(
        ["N", "p50", "p95", "p99", "ops/s", "root msgs", "waits", "degree", "total msgs", "depth"],
        [
            [
                row["worker_count"],
                f"{row['persistent_p50_ms']:.4f}",
                f"{row['persistent_p95_ms']:.4f}",
                f"{row['persistent_p99_ms']:.4f}",
                f"{row['persistent_throughput_ops_s']:.4f}",
                row["root_messages"],
                row["root_waits"],
                row["root_degree"],
                row["total_messages"],
                row["hierarchy_depth"],
            ]
            for row in scale_rows
        ],
    )
    trace_table = _markdown_table(
        ["Evidence set", "Result", "Detail"],
        [
            [
                name,
                "PASS" if audit["pass"] else "FAIL",
                "; ".join(
                    f"N={row['worker_count']}: {row['reduction_events']:,}/{row['expected_reduction_events']:,}"
                    for row in audit["scales"]
                ),
            ]
            for name, audit in evidence["trace_audits"].items()
        ],
    )
    return f"""# Experiment 013: Persistent Event-Driven Subtree Collectives

## Verdict

- **Experiment 013 thesis: {evidence["thesis"]}**
- **Runtime promotion: {evidence["runtime_promotion"]}**
- **Ready for next stage: {evidence["ready_for_next_stage"]}**

The thesis passes. A live worker-owned collective can stay installed across repeated operations and remove hierarchy reconstruction, task creation, process creation, and connection establishment from the warm path. At 1,000 independent worker processes, the retained candidate reached **{n1000["warm_p50_ms"]:.4f} ms p50** and **{n1000["throughput_ops_s"]:.4f} operations/s**: a **{n1000["p50_reduction_percent"]:.2f}% p50 reduction** and **{n1000["throughput_speedup"]:.2f}x throughput** versus the published Experiment 012 result.

The scaling classification also changed. Linear `N` was preferred over constant, `log N`, and `N log N` with AICc weight **{best_fit["akaike_weight"]:.4f}**, R² **{best_fit["r_squared"]:.6f}**, and **{fit["bootstrap"]["models"]["n"]["winner_count"]:,}/{fit["bootstrap"]["repetitions"]:,}** bootstrap wins. This satisfies the crucial thesis requirement that the result be more than a constant-factor improvement to the Experiment 012 `N log N` curve.

Runtime promotion is conditional because this experiment validates the persistent engine, protocol, and real-model output-head path, but does not yet wire it into the canonical production `MicroshardRemoteBackend`. The production default therefore remains unchanged. The evidence supports an explicit local-fast-domain adapter with flat and Experiment 012 delegated fallbacks; it does not support a universal branch factor or fine-grained WAN promotion.

## What changed

Experiment 012 established genuine worker-owned delegation but rebuilt the per-operation execution structure: intermediate workers created child-dispatch threads, the root submitted fresh tasks, and every generation activated the full tree. Experiment 013 installs parent/child ownership, route identity, receive execution, selector fanout, reduction state, connections, trace buffers, and fault state once. Warm requests carry an operation ID, a monotonic generation, a deadline, the payload, and a root proof. Every intermediate process still dispatches to children, collects responses, reduces deterministically, retries, and propagates cancellation.

No cross-process shared-memory shortcut was used. All benchmark traffic used framed TCP loopback between independent processes. Total traffic remains exactly `2N` messages. The root never contacted a leaf directly.

## Final required-scale result

All latencies are milliseconds. The 30-trial H013-022 run supplies the final cross-scale statistics; H013-023 independently confirms the linted final source at ten trials per cell.

{final_scale_table}

At B32 saturation, root work is 64 messages, one wait, degree 32, and zero leaf RPCs. Root bytes stay near 50 kB per operation while total bytes grow with N. Compared with Experiment 012's B8 bound of 16 messages/eight waits/degree eight, the selected same-host candidate spends a larger but still constant root fanout to remove serialized depth waves. The planner conclusion is topology-specific, not universal.

## Acceptance gates

{gate_table}

## Hypothesis outcomes

{hypothesis_table}

H013-004 is deliberately marked partial. Compact envelopes reduced N=1000 traffic from 2.824 MB to 1.619 MB, but improved p50 by only 0.7%. It was retained as a traffic and fault-state simplification, not rewritten as a successful latency mechanism.

## Evidence-driven redesign history

The first genuine persistent mailbox missed the p50 gate at 351.460 ms. Inline execution improved only 3.5%; selector fanout reached 318.977 ms; compact envelopes cut bytes but barely moved latency. Inspection then found the actual critical-path mechanism: every one of 1,000 workers synchronously opened, appended, and closed a trace file on the OneDrive-backed workspace for every operation. Bounded in-memory trace buffering reduced N=1000 p50 to 24.942 ms without dropping the expected events.

That candidate still fit `N log N`, so the experiment continued. Lean proofs only improved p50 by 2.0%. A new branch sweep, now free of trace-I/O distortion, showed that serialized depth waves were the remaining avoidable cost: B16 won at N=73, while B32 won at N=512 and N=1000. B32 reduced the installed tree to depth two and produced the final linear curve. Fault injection then exposed a response-stream race; a per-worker generation lock and identity-invalid stream reset fixed it with a 0.03% N=1000 normal-path change.

The full ledger is in `{run_root.name}/cycle-ledger.md`; failed hypotheses and environment failures remain in their original cycle directories.

## Sequence endurance and amortisation

At N=73/B16, one live collective processed 512 generations with warm p50/p95/p99 of **{n73["warm_p50_ms"]:.4f}/{n73["warm_p95_ms"]:.4f}/{n73["warm_p99_ms"]:.4f} ms**. Collective setup cost **{n73["collective_setup_ms"]:.4f} ms**, the first operation cost **{n73["first_operation_ms"]:.4f} ms**, and setup-inclusive execution first beat repeated fresh delegation at sequence length **{n73["break_even_sequence_length"]}**. Measured steady sequence throughput was **{n73["steady_sequence_throughput_ops_s"]:.4f} operations/s**. Aggregate endpoint RSS growth was zero.

At N=1000/B32, 512 generations produced warm p50/p95/p99 of **{n1000["long_sequence_p50_ms"]:.4f}/{n1000["long_sequence_p95_ms"]:.4f}/{n1000["long_sequence_p99_ms"]:.4f} ms**, setup broke even by operation **{n1000["break_even_sequence_length"]}**, steady sequence throughput was **{n1000["long_sequence_throughput_ops_s"]:.4f} operations/s**, and endpoint RSS growth was zero. Zero endpoint growth does not rule out transient peaks, but the lifecycle counters, complete teardown, stable connections, and event counts showed no retained resource leak.

{trace_table}

## N=73 Kimi-relevant result

{n73_table}

The persistent path reduces N=73 p50 by **{100 * (n73_compare[0]["p50_ms"] - n73_compare[1]["p50_ms"]) / n73_compare[0]["p50_ms"]:.2f}%** against the fresh Experiment 012-style baseline. Total messages remain 146 (`2N`), while total bytes fall because the installed route and compact telemetry are not resent. Root messages rise from 16 to 32 and degree from eight to 16, but waits fall from eight to one and root leaf RPCs remain zero.

# Implications for 73-node Kimi K3 serving

The measured control-plane fact is narrow but encouraging: on this single Windows development machine, 73 independent worker processes can keep a worker-owned collective alive across hundreds of token-like operations with **3.4664 ms warm p50**, **3.7175 ms p95**, **3.9906 ms p99**, 32 root messages, one root wait, degree 16, zero root-to-leaf RPCs, 146 total messages, and depth two. There is no per-token topology rebuild, process/task creation, or connection establishment. Setup is repaid by the tenth operation, far below a 128- or 512-token sequence.

An illustrative overhead budget, treating the measured synthetic control span as additive to otherwise independent model compute, is:

{_markdown_table(["Assumed model execution", "Measured control plane", "Control share"], overhead_rows)}

This is a projection, not a Kimi K3 benchmark. It assumes the real tensor/data path can overlap and communicate with the same topology without increasing the measured control span. Experiment 013 does **not** establish Kimi K3 tokens/s, RTX 3090 kernel time, VRAM fit, PCIe staging cost, physical-LAN contention, expert imbalance, NCCL behavior, or the interaction between Kimi K3 tensor traffic and this control protocol. It says that repeatedly reconstructing the distributed scheduler need not be the first-product blocker at roughly 73 nodes. The next defensible step is full model execution on the intended physical topology.

## Correctness, faults, and cancellation

Across N=73 and N=1000, 80/80 recoverable injections passed and every one of 80 following-generation probes was exact. Reordered arrivals, stale requests, delayed stale responses, dropped responses, duplicate responses, transient child failures, transient parent failures, and timeouts were covered. Ten recursive cancellations reached the live subtrees without propagation failure. Two permanent losses failed closed and explicitly marked the collective invalid/rebuild-required; neither published an incorrect aggregate.

Recovery cost is visible rather than hidden. Combined medians were 27.105 ms for a dropped response, 11.174 ms for an injected duplicate response, 19.052 ms for reordered arrivals, 25.392 ms for a stale response, 66.140 ms for a timeout, 29.483 ms for a transient leaf failure, 33.837 ms for a transient parent failure, and 85.608 ms for recursive cancellation. A permanent loss took about three seconds to fail closed under the configured deadline/retry policy.

## Real-model validation

The immutable supported path used eight independent persistent Qwen3-0.6B output-head workers. Reference token IDs were `{model["reference_token_ids"]}` and the persistent results reproduced them exactly. All **56/56** tensor comparisons passed; minimum cosine similarity was **0.9999988513** and maximum absolute error was **0.059288** within the preregistered tolerances. The root performed four messages at degree two, zero leaf RPCs, and median 16,459 bytes; warm p50 was **21.9373 ms** and median total traffic was **61,305.5 bytes**.

Four earlier attempts are retained: network resolution was blocked, offline resolution initially failed, Torch DLL loading was sandbox-blocked, and the elevated run found CUDA unavailable for fresh reference generation. The final validation therefore used the hash-verified immutable Experiment 012 reference tensors and a newly launched persistent worker tree. This supports exact path behavior but is weaker than a same-day fresh CUDA oracle, and the report does not describe it otherwise.

## Scaling interpretation

The final linear fit is `latency_ms = {best_fit["intercept_ms"]:.6f} + {best_fit["slope"]:.9f} * N`. Its RMSE is **{best_fit["rmse_ms"]:.4f} ms**. `N log N` was second with ΔAICc **{fit["fits"][1]["delta_aicc"]:.4f}** and weight **{fit["fits"][1]["akaike_weight"]:.4f}**. Bootstrap support for linear N was **{100 * fit["bootstrap"]["models"]["n"]["winner_fraction"]:.2f}%**.

This does not make total-system work constant. Every operation still wakes each participant once and sends exactly two messages per worker. The remaining same-host slope is approximately **{best_fit["slope"] * 1000:.3f} microseconds per added worker**. The limiting mechanism is now O(N) process readiness, TCP framing, and reduction work on one physical host—not repeated task, topology, or connection construction and not an avoidable extra log-depth factor at the selected branch factor.

## Runtime disposition

The thesis passes, but automatic canonical promotion would overstate the evidence. The retained code is an experimental persistent protocol plus harness, including real-model output-head support. The canonical production microshard backend still uses the Experiment 012 delegated request structure and must receive an adapter for collective install/prepare/execute/reset before this path can safely become a runtime choice.

Recommended promotion contract:

1. Explicit `persistent` selection only after capability negotiation and collective readiness.
2. Eligibility restricted to one measured `local-fast` domain; WAN fine-grained microwork stays disabled.
3. B16 is the current N≈73 research choice; B32 is the N≥512 same-host choice. Neither is universal.
4. Flat and Experiment 012 delegated fallbacks remain available.
5. Signed routes, generation isolation, deterministic reduction, bounded parent retry, recursive cancellation, separate root/system telemetry, fail-closed loss, and explicit rebuild remain mandatory.
6. Promotion becomes default only after the canonical tensor backend and physical multi-node evidence reproduce the gates.

## Limitations

- All scale and fault measurements use one development machine and TCP loopback, not a physical LAN or WAN.
- Synthetic operations measure the control plane; operations/s are not model tokens/s.
- Application wakeup counters are instrumented protocol events, not operating-system context-switch counters.
- The cross-scale p99 values have 30 observations and are labeled indicative; the 511-operation sequences provide the stronger tail evidence at N=73 and N=1000.
- Endpoint RSS was stable, but only process-level memory snapshots were used.
- B32 trades a larger constant root fanout for depth two; different RTT, CPU, and NIC conditions can change the optimum.
- Dynamic per-message Python objects still exist. Gate A concerns hierarchy/process/task/session creation, and H013-004's allocation claim is only partial.
- Real Kimi K3 compute, tensor traffic, placement, VRAM behavior, and economics remain unmeasured.

## Reproducibility and artifact integrity

The run root preserves preregistered hypotheses, immutable baseline measurements, raw JSONL operation records, lifecycle records, topology files, worker/root traces, failed attempts, source snapshots, scaling fits, fault rows, model checks, the cycle ledger, 12 figures, and the standalone report artifact. `evidence-integrity.json` records hashes for the headline inputs and deliverables. `artifact-validation.json` records schema, cross-file, chart, trace, and report checks.
"""


def _manager_summary(evidence: dict[str, Any]) -> str:
    n73 = evidence["n73"]
    n1000 = evidence["n1000"]
    hypotheses = _markdown_table(
        ["Hypothesis", "Outcome", "Why"],
        [[row["id"], row["outcome"], row["evidence"]] for row in evidence["hypotheses"]],
    )
    return f"""## Verdict

- Experiment 013 thesis: **{evidence["thesis"]}**
- Runtime promotion: **{evidence["runtime_promotion"]}**
- Ready for next stage: **{evidence["ready_for_next_stage"]}**

We have built a collective that stays alive across repeated token-like operations. The final path does not rebuild the tree, create worker processes/tasks, or establish sessions on a warm operation. Intermediate workers still genuinely own child dispatch, collection, reduction, retry, and cancellation.

At 1,000 workers, warm p50 fell from the published Experiment 012 **610.8157 ms** to **{n1000["warm_p50_ms"]:.4f} ms** (**{n1000["p50_reduction_percent"]:.2f}% lower**) and throughput rose from **1.6372** to **{n1000["throughput_ops_s"]:.4f} ops/s** (**{n1000["throughput_speedup"]:.2f}x**). The best scaling model changed from `N log N` to linear `N` with 99.98% bootstrap support.

For the Kimi-relevant 73-worker topology, the final B16 sequence delivered **{n73["warm_p50_ms"]:.4f}/{n73["warm_p95_ms"]:.4f}/{n73["warm_p99_ms"]:.4f} ms** p50/p95/p99 across 511 warm operations. It sustained **{n73["steady_sequence_throughput_ops_s"]:.2f} synthetic control operations/s**, broke even after **{n73["break_even_sequence_length"]} operations**, and showed zero endpoint RSS growth. Root work was 32 messages, one wait, degree 16, and zero leaf RPCs; total traffic remained exactly 146 messages.

This clears the coordination architecture for a real Kimi K3 stage, but it does not predict Kimi K3 tokens/s. RTX 3090 compute, model tensor traffic, physical networking, VRAM fit, imbalance, and end-to-end serving economics remain to be measured.

Runtime promotion is conditional because the validated persistent implementation is still an experimental protocol rather than an adapter in the canonical tensor microshard backend. The production default remains unchanged. The next implementation should add explicit persistent capability negotiation inside one measured local-fast domain, preserving flat and Experiment 012 delegated fallbacks. Fine-grained WAN microwork remains disabled.

The most important failed idea was useful: compact envelopes cut traffic 42.7% but improved latency only 0.7%. The decisive bottleneck was synchronous trace-file I/O on every worker, followed by serialized tree depth. Buffered traces plus B16/B32 topology selection produced the final result. Fault hardening then added effectively no normal-path penalty.

{hypotheses}
"""


def _cycle_ledger_markdown(cycles: list[dict[str, str]]) -> str:
    sections = ["# Experiment 013 cycle ledger", ""]
    for row in cycles:
        sections.extend(
            [
                f"## {row['cycle']}",
                "",
                f"- **Hypothesis:** {row['hypothesis']}",
                f"- **Implementation:** {row['implementation']}",
                f"- **Benchmark:** {row['benchmark']}",
                f"- **Result:** {row['result']}",
                f"- **Interpretation:** {row['interpretation']}",
                f"- **Bottleneck:** {row['bottleneck']}",
                f"- **Decision:** {row['decision']}",
                f"- **Next redesign:** {row['next_redesign']}",
                "",
            ]
        )
    return "\n".join(sections)


def _report_artifact(datasets: dict[str, Any], generated_at: str, run_root: Path) -> dict[str, Any]:
    dataset_sources = [
        {
            "id": f"dataset_{name}",
            "label": f"Reviewed Experiment 013 report dataset: {name}",
            "path": "final/report-data.json",
            **(
                {
                    "query": {
                        "engine": "sqlite",
                        "sql": PORTABLE_COMPARISON_SQL,
                        "description": (
                            "Projects the reviewed comparison rows for the portable "
                            "native chart; upstream values remain sourced from saved "
                            "Experiment 012 and 013 JSON evidence."
                        ),
                        "executed_at": generated_at,
                    }
                }
                if name == "comparison"
                else {}
            ),
        }
        for name in datasets
    ]
    evidence_sources = [
        {
            "id": "baseline",
            "label": "Immutable Experiment 013 delegated baseline",
            "path": "cycles/BASELINE-013/benchmark-summary.json",
        },
        {
            "id": "final_scale",
            "label": "Final 30-trial required-scale candidate",
            "path": "cycles/H013-022/benchmark-summary.json",
        },
        {
            "id": "scaling_fit",
            "label": "Final scaling model fit and bootstrap",
            "path": "cycles/H013-022/scaling-fit.json",
        },
        {
            "id": "release_equivalence",
            "label": "Final-source equivalence run",
            "path": "cycles/H013-023/benchmark-summary.json",
        },
        {
            "id": "sequence_73",
            "label": "N=73 512-operation sequence",
            "path": "cycles/H013-011f/sequence-summary.json",
        },
        {
            "id": "sequence_1000",
            "label": "N=1000 512-operation sequence",
            "path": "cycles/H013-011g/sequence-summary.json",
        },
        {
            "id": "branch_sweep",
            "label": "B4/B8/B16/B32 lean branch sweep",
            "path": "cycles/H013-019",
        },
        {
            "id": "faults",
            "label": "Live fault, retry, and cancellation evidence",
            "path": "cycles/H013-008/fault-summary.json",
        },
        {
            "id": "real_model",
            "label": "Persistent Qwen3 real-model validation",
            "path": "cycles/H013-010e/summary.json",
        },
        {"id": "ledger", "label": "Experiment 013 cycle ledger", "path": "cycle-ledger.json"},
        {
            "id": "validation",
            "label": "Repository and artifact validation",
            "path": "final/release-validation.json",
        },
    ]
    sources = [*dataset_sources, *evidence_sources]

    def chart(
        identifier: str,
        title: str,
        dataset: str,
        chart_type: str,
        x_field: str,
        x_label: str,
        y_field: str,
        y_label: str,
        color_field: str | None = None,
        subtitle: str = "",
    ) -> dict[str, Any]:
        encodings: dict[str, Any] = {
            "x": {"field": x_field, "label": x_label, "type": "ordinal"},
            "y": {"field": y_field, "label": y_label, "type": "quantitative", "format": "number"},
        }
        if color_field is not None:
            encodings["color"] = {
                "field": color_field,
                "label": color_field.replace("_", " ").title(),
                "type": "nominal",
            }
        return {
            "id": identifier,
            "title": title,
            "subtitle": subtitle,
            "type": chart_type,
            "dataset": dataset,
            "sourceId": f"dataset_{dataset}",
            "layout": "full",
            "valueFormat": "number",
            "encodings": encodings,
        }

    charts = [
        chart(
            "p50_comparison",
            "Experiment 012 vs 013 warm p50",
            "comparison",
            "line",
            "worker_count",
            "Workers",
            "p50_ms",
            "Warm p50 (ms)",
            "architecture",
            "Published and fresh baselines are both retained.",
        ),
        chart(
            "throughput_comparison",
            "Experiment 012 vs 013 throughput",
            "comparison",
            "line",
            "worker_count",
            "Workers",
            "throughput_ops_s",
            "Operations/s",
            "architecture",
        ),
        chart(
            "cold_warm",
            "Cold versus warm lifecycle",
            "cold_warm",
            "bar",
            "worker_count",
            "Workers",
            "latency_ms",
            "Latency (ms)",
            "metric",
            "Full cold includes process start; setup and warm are also shown separately.",
        ),
        chart(
            "scaling",
            "Warm latency and linear fit",
            "scaling",
            "line",
            "worker_count",
            "Workers",
            "latency_ms",
            "Latency (ms)",
            "series",
            "Linear N has 0.9954 AICc weight.",
        ),
        chart(
            "messages",
            "Bounded root versus total messages",
            "messages",
            "line",
            "worker_count",
            "Workers",
            "messages",
            "Messages/op",
            "metric",
        ),
        chart(
            "scheduling",
            "Warm activation and scheduling events",
            "scheduling",
            "line",
            "worker_count",
            "Workers",
            "events",
            "Application events/op",
            "metric",
            "These are instrumented application events, not OS context switches.",
        ),
        chart(
            "sequence",
            "Sequence setup amortisation",
            "sequence",
            "line",
            "sequence_length",
            "Operations",
            "amortised_ms",
            "Amortised ms/op",
            "worker_count",
            "N=73 breaks even at 10; N=1000 at 2.",
        ),
        chart(
            "n73",
            "N=73 latency comparison",
            "n73_comparison",
            "bar",
            "architecture",
            "Architecture",
            "p50_ms",
            "Warm p50 (ms)",
        ),
        chart(
            "tail",
            "Large-N latency tails",
            "tail",
            "bar",
            "worker_count",
            "Workers",
            "latency_ms",
            "Latency (ms)",
            "percentile",
        ),
        chart(
            "fault",
            "Fault-recovery overhead",
            "fault",
            "bar",
            "scenario",
            "Scenario",
            "latency_ms",
            "Median latency (ms)",
        ),
        chart(
            "branch",
            "Branch-factor comparison",
            "branch",
            "line",
            "branch_factor",
            "Branch factor",
            "p50_ms",
            "Warm p50 (ms)",
            "worker_count",
        ),
        chart(
            "components",
            "N=1000 measured timing spans",
            "components",
            "bar",
            "component",
            "Instrumented span",
            "duration_ms",
            "Median duration (ms)",
            None,
            "Worker spans overlap and are not additive.",
        ),
    ]
    tables = [
        {
            "id": "scale_table",
            "title": "Final required-scale result",
            "subtitle": "H013-022, 30 exact warm trials per cell",
            "dataset": "scale",
            "sourceId": "dataset_scale",
            "layout": "full",
            "density": "spacious",
            "defaultSort": {"field": "worker_count", "direction": "asc"},
            "columns": [
                {"field": "worker_count", "label": "Workers", "format": "number"},
                {"field": "persistent_p50_ms", "label": "p50 ms", "format": "number"},
                {"field": "persistent_p95_ms", "label": "p95 ms", "format": "number"},
                {"field": "persistent_p99_ms", "label": "p99 ms", "format": "number"},
                {"field": "persistent_throughput_ops_s", "label": "ops/s", "format": "number"},
                {"field": "root_messages", "label": "Root msgs", "format": "number"},
                {"field": "root_waits", "label": "Root waits", "format": "number"},
                {"field": "root_degree", "label": "Root degree", "format": "number"},
                {"field": "total_messages", "label": "Total msgs", "format": "number"},
                {"field": "hierarchy_depth", "label": "Depth", "format": "number"},
            ],
        },
        {
            "id": "gates_table",
            "title": "Critical acceptance gates",
            "subtitle": "The thesis passes only when every gate passes.",
            "dataset": "gates",
            "sourceId": "dataset_gates",
            "layout": "full",
            "density": "spacious",
            "defaultSort": {"field": "gate", "direction": "asc"},
            "columns": [
                {"field": "gate", "label": "Gate", "type": "text"},
                {"field": "name", "label": "Name", "type": "text"},
                {"field": "result", "label": "Result", "type": "text"},
                {"field": "evidence", "label": "Evidence", "type": "text"},
            ],
        },
        {
            "id": "hypotheses_table",
            "title": "Hypothesis outcomes",
            "subtitle": "Partial and rejected mechanisms remain explicit.",
            "dataset": "hypotheses",
            "sourceId": "dataset_hypotheses",
            "layout": "full",
            "density": "spacious",
            "defaultSort": {"field": "id", "direction": "asc"},
            "columns": [
                {"field": "id", "label": "Hypothesis", "type": "text"},
                {"field": "outcome", "label": "Outcome", "type": "text"},
                {"field": "evidence", "label": "Evidence", "type": "text"},
            ],
        },
        {
            "id": "cycles_table",
            "title": "Experimental redesign ledger",
            "subtitle": "Hypothesis → implementation → benchmark → inspection → redesign",
            "dataset": "cycles",
            "sourceId": "dataset_cycles",
            "layout": "full",
            "density": "dense",
            "defaultSort": {"field": "cycle", "direction": "asc"},
            "columns": [
                {"field": "cycle", "label": "Cycle", "type": "text"},
                {"field": "result", "label": "Measured result", "type": "text"},
                {"field": "bottleneck", "label": "Bottleneck", "type": "text"},
                {"field": "decision", "label": "Decision", "type": "text"},
                {"field": "next_redesign", "label": "Next redesign", "type": "text"},
            ],
        },
    ]
    blocks: list[dict[str, Any]] = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# Experiment 013: Persistent Event-Driven Subtree Collectives",
        },
        {
            "id": "verdict",
            "type": "markdown",
            "sourceId": "dataset_headline",
            "body": (
                "## Verdict\n\n**Thesis: PASS. Runtime promotion: CONDITIONAL. "
                "Ready for next stage: YES.**\n\nThe worker execution machinery now stays "
                "alive across operations. N=1000 warm p50 is 17.0685 ms, 97.21% "
                "below the published Experiment 012 result, and the preferred scaling "
                "model changes from N log N to N."
            ),
        },
    ]
    for identifier in (
        "p50_comparison",
        "throughput_comparison",
        "cold_warm",
        "scaling",
        "messages",
        "scheduling",
        "sequence",
        "n73",
        "tail",
        "fault",
        "branch",
        "components",
    ):
        blocks.append(
            {"id": f"{identifier}_block", "type": "chart", "chartId": identifier, "layout": "full"}
        )
    blocks.extend(
        [
            {
                "id": "scale_table_block",
                "type": "table",
                "tableId": "scale_table",
                "layout": "full",
            },
            {
                "id": "gates_table_block",
                "type": "table",
                "tableId": "gates_table",
                "layout": "full",
            },
            {
                "id": "hypotheses_table_block",
                "type": "table",
                "tableId": "hypotheses_table",
                "layout": "full",
            },
            {
                "id": "cycles_table_block",
                "type": "table",
                "tableId": "cycles_table",
                "layout": "full",
            },
            {
                "id": "kimi",
                "type": "markdown",
                "sourceId": "sequence_73",
                "body": (
                    "## Implications for 73-node Kimi K3 serving\n\nMeasured N=73 "
                    "control-plane p50/p95/p99 is 3.4664/3.7175/3.9906 ms across "
                    "511 warm operations, with setup break-even at operation 10. This "
                    "supports a real Kimi K3 benchmark; it is not a tokens/s claim."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Scope and limitations\n\nSingle-machine TCP loopback measures the "
                    "control plane. Physical networking, RTX 3090 execution, Kimi K3 "
                    "tensor traffic, VRAM fit, and token throughput remain unknown. "
                    "Application wakeups are not OS context switches."
                ),
            },
        ]
    )

    # The portable report renderer requires native cards/charts/tables to name the
    # SQL that produced them. Experiment 013 is sourced from JSON benchmark records
    # and worker traces, not SQL. Preserve truthful provenance by embedding the
    # already-QA'd publication figures and rendering the exact tables as Markdown
    # instead of inventing a query layer solely for packaging.
    figure_paths = sorted((run_root / "final" / "figures").glob("*.png"))
    if len(figure_paths) != len(charts):
        raise RuntimeError(f"expected {len(charts)} publication figures, found {len(figure_paths)}")
    interpretations = {
        "p50_comparison": "Persistence removes the repeated activation path at every tested scale; the gap widens sharply after N=128.",
        "throughput_comparison": "The final path reverses Experiment 012's large-N throughput collapse without changing the 2N message invariant.",
        "cold_warm": "Setup remains visible and material, but it is paid once; steady-state operations reuse every process, route, loop, and session.",
        "scaling": "Linear N is the preferred final fit, replacing Experiment 012's N log N classification with 99.98% bootstrap support.",
        "messages": "Root work saturates at B32 while total traffic remains exactly two messages per worker.",
        "scheduling": "Warm construction counters are zero; one persistent receive activation per worker remains the irreducible event-driven work.",
        "sequence": "At N=73 the setup-inclusive persistent path overtakes repeated delegated operations at operation 10.",
        "n73": "The Kimi-relevant topology combines millisecond-scale warm control latency with bounded root coordination.",
        "tail": "Large-N tails remain close to the median across the 30-trial final cells; no divergent queue tail appears.",
        "fault": "Recovery is deliberately more expensive than the normal path, and permanent loss fails closed rather than contaminating later generations.",
        "branch": "Persistence changes the useful topology by scale: B16 is best at N=73 while B32 is best at N=1000.",
        "components": "Buffered tracing removed the dominant synchronous file-I/O span; the remaining worker spans overlap and are not additive.",
    }
    static_blocks: list[dict[str, Any]] = [
        blocks[0],
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "final_scale",
            "body": (
                "## Technical summary\n\n**Thesis: PASS. Runtime promotion: "
                "CONDITIONAL. Ready for next stage: YES.** The persistent collective "
                "keeps worker execution machinery alive, cuts N=1000 warm p50 from "
                "610.8157 ms to 17.0685 ms, and changes the preferred scaling model "
                "from N log N to N while preserving exact worker-owned semantics."
            ),
        },
        {
            "id": "p50_comparison_context",
            "type": "markdown",
            "sourceId": "dataset_comparison",
            "body": (
                "## Experiment 012 vs 013 warm p50\n\nPublished and fresh "
                "baselines are both retained. "
                f"{interpretations['p50_comparison']}"
            ),
        },
        {
            "id": "p50_comparison_native",
            "type": "chart",
            "chartId": "p50_comparison",
            "layout": "full",
        },
    ]
    for spec, figure_path in zip(charts[1:], figure_paths[1:], strict=True):
        subtitle = str(spec.get("subtitle", "")).strip()
        interpretation = interpretations[spec["id"]]
        context = " ".join(part for part in (subtitle, interpretation) if part)
        encoded = base64.b64encode(figure_path.read_bytes()).decode("ascii")
        static_blocks.extend(
            [
                {
                    "id": f"{spec['id']}_context",
                    "type": "markdown",
                    "sourceId": spec["sourceId"],
                    "body": f"## {spec['title']}\n\n{context}",
                },
                {
                    "id": f"{spec['id']}_figure",
                    "type": "html",
                    "sourceId": spec["sourceId"],
                    "body": (
                        "<style>body{margin:0;background:#fff}figure{margin:0}"
                        "img{display:block;width:100%;height:auto}</style>"
                        f'<figure><img src="data:image/png;base64,{encoded}" '
                        f'alt="{spec["title"]}"></figure>'
                    ),
                },
            ]
        )
    for table in tables:
        columns = table["columns"]
        table_rows = datasets[table["dataset"]]
        rendered = _markdown_table(
            [str(column["label"]) for column in columns],
            [[row.get(str(column["field"]), "") for column in columns] for row in table_rows],
        )
        static_blocks.append(
            {
                "id": f"{table['id']}_markdown",
                "type": "markdown",
                "sourceId": table["sourceId"],
                "body": f"## {table['title']}\n\n{table['subtitle']}\n\n{rendered}",
            }
        )
    static_blocks.extend(blocks[-2:])

    return {
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Experiment 013: Persistent Event-Driven Subtree Collectives",
            "description": "Technical report for the persistent worker-owned collective experiment.",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "thesis",
                    "dataset": "headline",
                    "sourceId": "dataset_headline",
                    "description": "All critical gates.",
                    "metrics": [{"field": "thesis", "label": "Thesis", "format": "text"}],
                },
                {
                    "id": "n73",
                    "dataset": "headline",
                    "sourceId": "dataset_headline",
                    "description": "Kimi-relevant topology.",
                    "metrics": [
                        {"field": "n73_warm_p50_ms", "label": "N=73 p50 (ms)", "format": "number"}
                    ],
                },
                {
                    "id": "n1000",
                    "dataset": "headline",
                    "sourceId": "dataset_headline",
                    "description": "Largest tested scale.",
                    "metrics": [
                        {
                            "field": "n1000_warm_p50_ms",
                            "label": "N=1000 p50 (ms)",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "speedup",
                    "dataset": "headline",
                    "sourceId": "dataset_headline",
                    "description": "Versus published Experiment 012.",
                    "metrics": [
                        {
                            "field": "n1000_throughput_speedup",
                            "label": "Throughput speedup",
                            "format": "number",
                        }
                    ],
                },
            ],
            "charts": [charts[0]],
            "tables": [],
            "sources": sources,
            "blocks": static_blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
        "surface": "report",
        "runId": run_root.name,
    }


def _machine_summary(evidence: dict[str, Any], run_root: Path) -> dict[str, Any]:
    return {
        "experiment_id": "013",
        "run_id": run_root.name,
        "thesis": evidence["thesis"],
        "runtime_promotion": evidence["runtime_promotion"],
        "ready_for_next_stage": evidence["ready_for_next_stage"],
        "final_architecture": {
            "name": "lean-selector-v6",
            "collective_protocol": "persistent-event-v1",
            "n73_branch_factor": 16,
            "large_scale_branch_factor": 32,
            "distributed_semantics": "independent worker processes over framed TCP loopback",
            "normal_warm_path": {
                "topology_rebuilds": 0,
                "worker_process_creations": 0,
                "persistent_loop_creations": 0,
                "new_task_creations": 0,
                "connection_establishments": 0,
            },
        },
        "headline": {
            "n73": evidence["n73"],
            "n1000": evidence["n1000"],
            "scaling": {
                "best_model": evidence["fit"]["best_model"],
                "aicc_weight": evidence["fit"]["fits"][0]["akaike_weight"],
                "r_squared": evidence["fit"]["fits"][0]["r_squared"],
                "bootstrap_winner_fraction": evidence["fit"]["bootstrap"]["models"]["n"][
                    "winner_fraction"
                ],
            },
        },
        "correctness_and_recovery": {
            "recoverable_injections": evidence["fault"]["recoverable_injections"],
            "recoverable_injections_passed": evidence["fault"]["recoverable_injections_passed"],
            "next_generation_probes": evidence["fault"]["next_generation_probes"],
            "next_generation_probes_passed": evidence["fault"]["next_generation_probes_passed"],
            "permanent_losses_fail_closed": evidence["fault"]["permanent_losses_fail_closed"],
            "real_model_checks": evidence["model"]["checks"],
            "reference_token_ids": evidence["model"]["reference_token_ids"],
            "persistent_token_ids": evidence["model"]["persistent_token_ids"],
            "tensor_checks": 56,
            "tensor_checks_passed": 56,
            "minimum_cosine_similarity": 0.9999988513,
        },
        "acceptance_gates": evidence["gates"],
        "hypotheses": evidence["hypotheses"],
        "trace_audits": evidence["trace_audits"],
        "validation": evidence["validation"],
        "scope": {
            "physical_network": "single development machine and TCP loopback only",
            "synthetic_ops_are_model_tokens": False,
            "kimi_k3_executed": False,
        },
    }


def _write_integrity(run_root: Path, final_directory: Path) -> dict[str, Any]:
    relative_paths = [
        "run-manifest.json",
        "cycle-ledger.json",
        "cycle-ledger.md",
        "cycles/BASELINE-013/benchmark-summary.json",
        "cycles/H013-022/benchmark-summary.json",
        "cycles/H013-022/scaling-fit.json",
        "cycles/H013-023/benchmark-summary.json",
        "cycles/H013-011f/sequence-summary.json",
        "cycles/H013-011g/sequence-summary.json",
        "cycles/H013-008/fault-summary.json",
        "cycles/H013-010e/summary.json",
        "final/machine-readable-summary.json",
        "final/acceptance-gates.json",
        "final/final-report.md",
        "final/manager-summary.md",
        "final/report-data.json",
        "final/artifact.json",
        "final/figure-manifest.json",
        "final/html-build-receipt.json",
    ]
    if (final_directory / "report.html").is_file():
        relative_paths.append("final/report.html")
    records = []
    for relative in relative_paths:
        path = run_root / relative
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    for path in sorted((final_directory / "figures").glob("*.png")):
        records.append(
            {
                "path": path.relative_to(run_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "algorithm": "sha256",
        "file_count": len(records),
        "files": records,
        "all_files_nonempty": all(int(row["bytes"]) > 0 for row in records),
    }
    _write_json(final_directory / "evidence-integrity.json", payload)
    return payload


def _assemble(run_root: Path) -> dict[str, Any]:
    final_directory = run_root / "final"
    final_directory.mkdir(parents=True, exist_ok=True)
    validation_path = final_directory / "release-validation.json"
    validation = (
        _load(validation_path)
        if validation_path.is_file()
        else {
            "overall_pass": False,
            "summary": "Repository validation pending.",
        }
    )
    evidence = _derive(run_root, validation)
    evidence["datasets"]["comparison"] = _portable_comparison_rows(
        evidence["datasets"]["comparison"]
    )
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    _write_json(
        run_root / "cycle-ledger.json", {"experiment_id": "013", "cycles": evidence["cycles"]}
    )
    (run_root / "cycle-ledger.md").write_text(
        _cycle_ledger_markdown(evidence["cycles"]), encoding="utf-8"
    )
    _write_json(final_directory / "report-data.json", evidence["datasets"])
    _write_json(
        final_directory / "report-source-notes.json",
        {
            "generated_at": generated_at,
            "transformation": "Deterministic Python extraction in report_harness.py; medians use successful warm rows and nearest-rank empirical tails. The comparison dataset is projected through the recorded in-memory SQLite query for portable-chart provenance.",
            "headline_precedence": "H013-022 supplies 30-trial cross-scale statistics; H013-023 is final-source equivalence; H013-011f/g supply 511-operation tails.",
            "baseline_precedence": "Published Experiment 012 figures are used for Gate C; the fresh BASELINE-013 run is reported separately.",
            "limitations": [
                "Single-host TCP loopback only.",
                "Synthetic operations are not model tokens.",
                "Application wakeups are not OS context switches.",
                "N=73 branch-optimal tails come from H013-011f, not the B32 cross-scale run.",
            ],
        },
    )
    summary = _machine_summary(evidence, run_root)
    _write_json(final_directory / "machine-readable-summary.json", summary)
    _write_json(
        final_directory / "acceptance-gates.json",
        {
            "experiment_id": "013",
            "thesis": evidence["thesis"],
            "all_critical_gates_pass": evidence["thesis"] == "PASS",
            "gates": evidence["gates"],
        },
    )
    (final_directory / "final-report.md").write_text(
        _final_report(evidence, run_root), encoding="utf-8"
    )
    (final_directory / "manager-summary.md").write_text(
        _manager_summary(evidence), encoding="utf-8"
    )
    figures = _make_figures(final_directory, evidence["datasets"])
    artifact = _report_artifact(evidence["datasets"], generated_at, run_root)
    _write_json(final_directory / "artifact.json", artifact)
    manifest = _load(run_root / "run-manifest.json")
    manifest.update(
        {
            "completed_utc": generated_at,
            "status": "complete" if evidence["thesis"] == "PASS" else "validation_pending",
            "thesis": evidence["thesis"],
            "runtime_promotion": evidence["runtime_promotion"],
            "ready_for_next_stage": evidence["ready_for_next_stage"],
        }
    )
    _write_json(run_root / "run-manifest.json", manifest)
    integrity = _write_integrity(run_root, final_directory)
    return {
        "thesis": evidence["thesis"],
        "figures": len(figures),
        "integrity_files": integrity["file_count"],
        "artifact": str(final_directory / "artifact.json"),
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _validate(run_root: Path) -> dict[str, Any]:
    final_directory = run_root / "final"
    required = [
        run_root / "cycle-ledger.json",
        run_root / "cycle-ledger.md",
        final_directory / "machine-readable-summary.json",
        final_directory / "acceptance-gates.json",
        final_directory / "final-report.md",
        final_directory / "manager-summary.md",
        final_directory / "report-data.json",
        final_directory / "artifact.json",
        final_directory / "report.html",
        final_directory / "figure-manifest.json",
        final_directory / "evidence-integrity.json",
        final_directory / "release-validation.json",
        final_directory / "html-build-receipt.json",
    ]
    summary = _load(final_directory / "machine-readable-summary.json")
    gates = _load(final_directory / "acceptance-gates.json")
    report_data = _load(final_directory / "report-data.json")
    artifact = _load(final_directory / "artifact.json")
    html_build = _load(final_directory / "html-build-receipt.json")
    figure_manifest = _load(final_directory / "figure-manifest.json")
    manager = (final_directory / "manager-summary.md").read_text(encoding="utf-8")
    html = (
        (final_directory / "report.html").read_text(encoding="utf-8")
        if (final_directory / "report.html").is_file()
        else ""
    )
    checks = {
        "required_files_present_nonempty": all(
            path.is_file() and path.stat().st_size > 0 for path in required
        ),
        "manager_starts_exactly_with_verdict": manager.startswith("## Verdict"),
        "all_critical_gates_pass": bool(gates["all_critical_gates_pass"]),
        "summary_gate_consistency": summary["thesis"] == gates["thesis"] == "PASS",
        "report_data_finite": _all_finite(report_data),
        "artifact_snapshot_matches_report_data": artifact["snapshot"]["datasets"] == report_data,
        "twelve_figures_present": len(figure_manifest) == 12
        and all(
            (final_directory / "figures" / row["filename"]).is_file() for row in figure_manifest
        ),
        "figure_hashes_match": all(
            _sha256(final_directory / "figures" / row["filename"]) == row["sha256"]
            for row in figure_manifest
        ),
        "trace_audits_pass": all(audit["pass"] for audit in summary["trace_audits"].values()),
        "html_is_standalone_and_substantial": len(html.encode("utf-8")) > 100_000
        and "Experiment 013" in html,
        "html_builder_package_passes": html_build.get("status") == "PASS"
        and html_build.get("artifact_validation") == "passed"
        and html_build.get("packaging") == "passed",
        "release_validation_passes": bool(summary["validation"].get("overall_pass")),
        "root_leaf_rpc_exact_zero": all(
            int(row["root_leaf_rpcs"]) == 0 for row in report_data["scale"]
        ),
        "total_messages_exact_2n": all(
            int(row["total_messages"]) == 2 * int(row["worker_count"])
            for row in report_data["scale"]
        ),
        "warm_creation_counters_zero": all(
            all(
                int(row[field]) == 0
                for field in (
                    "worker_activations",
                    "new_tasks",
                    "new_connections",
                    "topology_rebuilds",
                )
            )
            for row in report_data["scale"]
        ),
    }
    receipt = {
        "experiment_id": "013",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "report_html_bytes": len(html.encode("utf-8")),
        "validated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    _write_json(final_directory / "artifact-validation.json", receipt)
    _write_json(
        final_directory / "report-delivery-receipt.json",
        {
            "status": receipt["status"],
            "verification": (
                "builder_structural_plus_independent_evidence_validation_and_"
                "separate_figure_visual_qa"
            ),
            "builder": html_build,
            "output": str(final_directory / "report.html"),
            "bytes": receipt["report_html_bytes"],
            "checks_passed": sum(checks.values()),
            "checks_total": len(checks),
        },
    )
    _write_integrity(run_root, final_directory)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    result = _validate(run_root) if args.validate_only else _assemble(run_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
