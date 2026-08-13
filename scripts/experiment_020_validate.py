"""Run the corrected Experiment 020 replay and accounting gates."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_019.placement import PlacementSpec
from swarm_inference.experiments.experiment_019.simulation import (
    NETWORK_PROFILES,
    SimulationConfiguration,
)
from swarm_inference.experiments.experiment_020.replay import (
    run_single_resource_replay,
)
from swarm_inference.experiments.experiment_020.simulation import (
    accounting_reconciliation,
    measured_service,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "experiment-020"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    attention = _read(ARTIFACT / "physical" / "attention-raw.json")
    legacy_expert = _read(
        ROOT / "artifacts" / "experiment-019" / "physical" / "expert-stripe-raw.json"
    )
    grouped_calibration = _read(
        ARTIFACT / "physical" / "expert-grouped-robust-calibration.json"
    )
    grouped_heldout = _read(
        ARTIFACT / "physical" / "expert-grouped-robust-heldout.json"
    )
    other_calibration = _read(
        ARTIFACT / "physical" / "other-shards-robust-calibration.json"
    )
    other_heldout = _read(
        ARTIFACT / "physical" / "other-shards-robust-heldout.json"
    )
    protocol_calibration = _read(
        ARTIFACT / "runtime" / "protocol-heldout-raw.json"
    )
    protocol_heldout = _read(
        ARTIFACT / "runtime" / "protocol-validation-raw.json"
    )
    full_grouped = _read(ARTIFACT / "physical" / "expert-grouped-raw.json")
    full_other = _read(ARTIFACT / "physical" / "other-shards-raw.json")

    replay = run_single_resource_replay(
        attention_calibration=attention,
        attention_heldout=attention,
        legacy_expert=legacy_expert,
        grouped_calibration=grouped_calibration,
        grouped_heldout=grouped_heldout,
        other_calibration=other_calibration,
        other_heldout=other_heldout,
        protocol_calibration=protocol_calibration,
        protocol_heldout=protocol_heldout,
    )
    _write_csv(
        ARTIFACT / "validation" / "single-resource-replay.csv", replay["rows"]
    )
    _write_json(
        ARTIFACT / "validation" / "single-resource-replay.json", replay
    )

    reconciliations = []
    for block in (7, 12, 16):
        for chunk in (1, 2, 4):
            accounting_grouped = grouped_calibration if chunk == 1 else full_grouped
            accounting_other = other_calibration if chunk == 1 else full_other
            service = measured_service(
                attention,
                legacy_expert,
                accounting_grouped,
                accounting_other,
                protocol_calibration,
                degree=8,
                rows=chunk,
            )
            reconciliation = accounting_reconciliation(
                service,
                SimulationConfiguration(
                    placement=PlacementSpec(20, 8, 8, chunk),
                    block=block,
                    chunk=chunk,
                    local_profile=NETWORK_PROFILES["canonical_fast_local"],
                    inter_pod_profile=NETWORK_PROFILES["canonical_inter_pod"],
                ),
            )
            reconciliations.append(reconciliation)
    accounting = {
        "schema_version": "experiment-020-accounting-suite-v1",
        "configurations_checked": len(reconciliations),
        "all_compute_invariants_pass": all(
            row["compute_invariant_pass"] for row in reconciliations
        ),
        "all_network_invariants_pass": all(
            row["network_invariant_pass"] for row in reconciliations
        ),
        "no_cost_disappears": all(row["status"] == "PASS" for row in reconciliations),
        "reconciliations": reconciliations,
        "status": (
            "PASS" if all(row["status"] == "PASS" for row in reconciliations) else "FAIL"
        ),
    }
    _write_json(
        ARTIFACT / "validation" / "accounting-reconciliation.json", accounting
    )
    _write_json(
        ARTIFACT / "simulation" / "accounting-reconciliation.json", accounting
    )
    result = {
        "single_resource_replay": replay["status"],
        "replay_errors": replay["validation"],
        "accounting": accounting["status"],
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if replay["status"] == accounting["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
