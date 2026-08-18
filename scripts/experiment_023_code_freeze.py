"""Seal repaired E023 code after the control pilot and before headline execution."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_022.io import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from swarm_inference.experiments.experiment_023.freeze import (  # noqa: E402
    CONTROL_INVENTORIES,
    FROZEN_CONSTANTS,
)
from swarm_inference.experiments.experiment_023.repair_validation import (  # noqa: E402
    validate_repair_inputs_read_only,
)
from swarm_inference.experiments.experiment_023.serving_objective import (  # noqa: E402
    score_recorded_runs_under_primary_slo,
)

PILOT_ATTEMPT = "repair-control-pilot-v1"
CODE_FILES = (
    "src/swarm_inference/experiments/experiment_022/manifest_correctness.py",
    "src/swarm_inference/experiments/experiment_023/analysis.py",
    "src/swarm_inference/experiments/experiment_023/correctness.py",
    "src/swarm_inference/experiments/experiment_023/finalize.py",
    "src/swarm_inference/experiments/experiment_023/models.py",
    "src/swarm_inference/experiments/experiment_023/repair_validation.py",
    "src/swarm_inference/experiments/experiment_023/replica_planner.py",
    "src/swarm_inference/experiments/experiment_023/reproducibility.py",
    "src/swarm_inference/experiments/experiment_023/runner.py",
    "src/swarm_inference/experiments/experiment_023/serving_objective.py",
    "scripts/experiment_023_code_freeze.py",
    "scripts/experiment_023_correctness.py",
    "scripts/experiment_023_finalize.py",
    "scripts/experiment_023_run.py",
    "tests/test_experiment_023.py",
    "tests/test_experiment_023_completion.py",
)


@dataclass(frozen=True, slots=True)
class _Run:
    status: str
    concurrency: int
    target_rows_per_second: float
    p50_pass_latency_ms: float
    p95_pass_latency_ms: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _score(
    rows: list[dict[str, str]],
    *,
    inventory_id: str,
    arm: str,
    u_strong_c1_p95_ms: float,
):
    selected = [
        row
        for row in rows
        if row["inventory_id"] == inventory_id
        and row["arm"] == arm
        and row["network_mode"] == "SHARED_NIC"
    ]
    costs = {float(row["abstract_node_cost"]) for row in selected}
    if len(costs) != 1:
        raise RuntimeError(f"non-constant cost for {inventory_id}/{arm}")
    runs = {
        int(row["concurrency"]): _Run(
            status=row["status"],
            concurrency=int(row["concurrency"]),
            target_rows_per_second=float(row["target_rows_per_second"]),
            p50_pass_latency_ms=float(row["p50_pass_latency_ms"]),
            p95_pass_latency_ms=float(row["p95_pass_latency_ms"]),
        )
        for row in selected
    }
    return score_recorded_runs_under_primary_slo(
        runs,
        u_strong_c1_p95_ms=u_strong_c1_p95_ms,
        abstract_cost=next(iter(costs)),
    )


def _control_gate(attempt_root: Path) -> dict[str, Any]:
    summary = _read_json(attempt_root / "attempt-summary.json")
    if summary.get("inventory_ids") != list(CONTROL_INVENTORIES):
        raise RuntimeError("control pilot did not run exactly the three frozen controls")
    if summary.get("thresholds_sha256") != canonical_sha256(FROZEN_CONSTANTS):
        raise RuntimeError("control pilot frozen constants hash changed")
    arm_rows = _read_csv(attempt_root / "serving/arm-results.csv")
    results: list[dict[str, Any]] = []
    for inventory_id in CONTROL_INVENTORIES:
        reference = next(
            row
            for row in arm_rows
            if row["inventory_id"] == inventory_id
            and row["arm"] == "U_STRONG"
            and row["network_mode"] == "SHARED_NIC"
            and int(row["concurrency"]) == 1
        )
        c1_p95 = float(reference["p95_pass_latency_ms"])
        u_strong = _score(
            arm_rows,
            inventory_id=inventory_id,
            arm="U_STRONG",
            u_strong_c1_p95_ms=c1_p95,
        )
        for arm in ("FLEX_FREE", "FLEX_POOL"):
            candidate = _score(
                arm_rows,
                inventory_id=inventory_id,
                arm=arm,
                u_strong_c1_p95_ms=c1_p95,
            )
            efficiency = 100.0 * (
                candidate.rows_per_second_per_abstract_cost
                / u_strong.rows_per_second_per_abstract_cost
                - 1.0
            )
            throughput = 100.0 * (
                candidate.target_rows_per_second
                / u_strong.target_rows_per_second
                - 1.0
            )
            results.append(
                {
                    "inventory_id": inventory_id,
                    "arm": arm,
                    "selected_slo_concurrency": candidate.selected_concurrency,
                    "efficiency_uplift_percent": efficiency,
                    "raw_slo_throughput_uplift_percent": throughput,
                    "efficiency_gate_pass": efficiency >= -5.0,
                    "throughput_gate_pass": throughput >= -5.0,
                }
            )
    superset = _read_csv(attempt_root / "validation/flex-pool-superset.csv")
    coverage = _read_csv(
        attempt_root / "validation/flex-pool-search-coverage.csv"
    )
    superset_pass = len(superset) == 3 and all(
        row["status"] == "PASS" for row in superset
    )
    coverage_pass = bool(coverage) and all(
        row["flex_free_in_final_pool_envelope"] == "True" for row in coverage
    )
    u_matches = 0
    for inventory_id in CONTROL_INVENTORIES:
        old = _read_json(
            REPO_ROOT
            / "artifacts/experiment-023/attempts/deterministic-run-1/plans"
            / inventory_id
            / "U_STRONG.json"
        )
        new = _read_json(
            attempt_root / "plans" / inventory_id / "U_STRONG.json"
        )
        u_matches += old["canonical_plan_sha256"] == new["canonical_plan_sha256"]
    passed = (
        all(
            row["efficiency_gate_pass"] and row["throughput_gate_pass"]
            for row in results
        )
        and superset_pass
        and coverage_pass
        and u_matches == 3
    )
    return {
        "schema_version": "experiment-023-repair-control-gate-v1",
        "experiment_id": "023",
        "status": "PASS" if passed else "FAIL",
        "attempt": PILOT_ATTEMPT,
        "controls": results,
        "flex_pool_superset_pass_3_of_3": superset_pass,
        "flex_free_in_final_pool_envelope_all_rows": coverage_pass,
        "u_strong_run1_exact_matches": u_matches,
        "elapsed_seconds": float(summary["elapsed_seconds"]),
    }


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return (result.stdout + result.stderr).strip()


def main() -> int:
    validation = validate_repair_inputs_read_only(REPO_ROOT)
    attempt_root = (
        REPO_ROOT / "artifacts/experiment-023/attempts" / PILOT_ATTEMPT
    )
    gate = _control_gate(attempt_root)
    atomic_write_json(attempt_root / "validation/control-gate.json", gate)
    if gate["status"] != "PASS":
        raise RuntimeError("MODEL_INVALID: repaired control pilot failed")
    files = []
    for relative in CODE_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    protocol = REPO_ROOT / "artifacts/experiment-023/repair/repair-protocol.json"
    physical = _read_json(protocol)["physical_validation_reuse"]["files"]
    receipt = {
        "schema_version": "experiment-023-repair-code-freeze-v1",
        "experiment_id": "023",
        "status": "FROZEN_BEFORE_AUTHORITATIVE_HEADLINE",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "git_head": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "files": files,
        "file_count": len(files),
        "files_canonical_sha256": canonical_sha256(files),
        "frozen_constants_sha256": canonical_sha256(FROZEN_CONSTANTS),
        "e022_inventory_suite_sha256": validation[
            "e022_inventory_suite_sha256"
        ],
        "e022_input_hashes_sha256": validation["e022_input_hashes_sha256"],
        "physical_artifact_hashes": physical,
        "repair_protocol_sha256": sha256_file(protocol),
        "control_pilot_gate_path": (
            "artifacts/experiment-023/attempts/repair-control-pilot-v1/"
            "validation/control-gate.json"
        ),
        "control_pilot_gate_sha256": sha256_file(
            attempt_root / "validation/control-gate.json"
        ),
        "control_pilot": gate,
    }
    destination = REPO_ROOT / "artifacts/experiment-023/repair/repair-code-freeze.json"
    atomic_write_json(destination, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
