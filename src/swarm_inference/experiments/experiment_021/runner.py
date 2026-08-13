"""Resumable zero-rental orchestration for Experiment 021."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_019.checkpoint import CheckpointCatalog
from swarm_inference.experiments.experiment_019.placement import PlacementSpec, build_placement

from .charts import generate_charts
from .finalize import finalize_experiment
from .market import run_read_only_market_inventory
from .placement import (
    EXPERT_ALLOCATION_OVERHEAD_FACTOR,
    materialize_placement_artifacts,
)
from .runtime import materialize_runtime_receipts, run_control_plane_scaling
from .simulation import (
    run_accounting,
    run_concurrency,
    run_heterogeneity,
    run_main_sweep,
    run_network_envelope,
    run_whole_layer_control,
)
from .validation import run_ordered_physical_replay

PHASES = (
    "placement",
    "validation",
    "simulation",
    "runtime",
    "control-plane",
    "market",
    "charts",
    "finalize",
)


def _placements(checkpoint: Path) -> dict[int, Any]:
    catalog = CheckpointCatalog(checkpoint)
    catalog.records()
    specs = {8: (8, 2), 4: (8, 1), 2: (16, 1), 1: (32, 1)}
    return {
        cap: build_placement(
            catalog,
            PlacementSpec(
                cap,
                degree,
                depth,
                1,
                hardware_class="INDEPENDENT_MACHINE_RTX5090_SHARD_SERVICE",
                expert_allocation_overhead_factor=EXPERT_ALLOCATION_OVERHEAD_FACTOR,
            ),
        )
        for cap, (degree, depth) in specs.items()
    }


def _best_rows(artifact_root: Path) -> dict[tuple[int, str], dict[str, str]]:
    with (artifact_root / "simulation" / "sweep.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for cap in (8, 4, 2, 1):
        for regime in ("A", "B", "C", "D", "E"):
            values = [
                row
                for row in rows
                if int(float(row["memory_cap_gib"])) == cap
                and row["regime"] == regime
            ]
            result[(cap, regime)] = max(
                values, key=lambda row: float(row["exact_tok_s_per_user"])
            )
    return result


def run_phase(
    phase: str,
    *,
    repo: Path,
    checkpoint: Path,
    force: bool,
) -> dict[str, Any]:
    artifact_root = repo / "artifacts" / "experiment-021"
    artifact_root.mkdir(parents=True, exist_ok=True)
    if phase == "placement":
        target = artifact_root / "placement" / "worker-manifest-8g.json"
        if target.is_file() and not force:
            return {"phase": phase, "status": "RESUMED", "artifact": str(target)}
        _catalog, placements, receipt = materialize_placement_artifacts(
            checkpoint, artifact_root / "placement"
        )
        return {
            "phase": phase,
            "status": receipt["status"],
            "workers": {cap: len(value.workers) for cap, value in placements.items()},
        }
    if phase == "validation":
        target = artifact_root / "physical" / "ordered-workloads.json"
        if target.is_file() and not force:
            receipt = json.loads(target.read_text(encoding="utf-8"))
        else:
            receipt = run_ordered_physical_replay(
                repo,
                artifact_root,
                checkpoint=checkpoint,
            )
        return {
            "phase": phase,
            "status": receipt["status"],
            "model_validation": receipt["model_validation"],
        }
    if phase == "simulation":
        placements = _placements(checkpoint)
        validation = json.loads(
            (artifact_root / "physical" / "ordered-workloads.json").read_text(
                encoding="utf-8"
            )
        )
        validated = validation["status"] == "PASS"
        target = artifact_root / "simulation" / "sweep.csv"
        if target.is_file() and not force:
            best = _best_rows(artifact_root)
        else:
            _rows, best, _critical = run_main_sweep(
                repo,
                placements,
                artifact_root,
                validated=validated,
            )
        run_accounting(repo, placements, best, artifact_root)
        run_network_envelope(
            repo,
            placements[8],
            best[(8, "B")],
            artifact_root,
            validated=validated,
        )
        run_heterogeneity(
            repo,
            placements[8],
            best[(8, "B")],
            artifact_root,
            validated=validated,
        )
        run_concurrency(
            repo,
            placements[8],
            best[(8, "B")],
            artifact_root,
            validated=validated,
        )
        run_whole_layer_control(repo, artifact_root)
        return {
            "phase": phase,
            "status": "DIAGNOSTIC_MODEL_INVALID" if not validated else "PASS",
            "admissible": validated,
        }
    if phase == "runtime":
        receipt = materialize_runtime_receipts(repo, artifact_root)
        return {"phase": phase, "status": receipt["status"]}
    if phase == "control-plane":
        manifest = json.loads(
            (artifact_root / "placement" / "worker-manifest-8g.json").read_text(
                encoding="utf-8"
            )
        )
        rows = run_control_plane_scaling(
            artifact_root,
            natural_worker_count=int(manifest["summary"]["worker_count"]),
        )
        return {
            "phase": phase,
            "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
            "rows": len(rows),
        }
    if phase == "market":
        placements = _placements(checkpoint)
        snapshot, feasibility, audit = run_read_only_market_inventory(
            artifact_root, placements
        )
        return {
            "phase": phase,
            "status": feasibility["status"],
            "offers": snapshot["offer_count"],
            "gpu_rentals": audit["gpu_rentals"],
            "vast_mutations": audit["vast_resource_mutations"],
        }
    if phase == "charts":
        receipt = generate_charts(artifact_root)
        return {"phase": phase, "status": receipt["status"], "charts": receipt["chart_count"]}
    if phase == "finalize":
        receipt = finalize_experiment(repo, artifact_root)
        return {"phase": phase, "status": receipt["outcome"], "report": receipt["report"]}
    raise ValueError(f"unknown phase {phase}")


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", *PHASES), default="all")
    parser.add_argument("--checkpoint", type=Path, default=Path("F:/models/Kimi-K3"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="forbidden in zero-rental E021; retained only to fail closed",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _arguments(argv)
    if arguments.apply:
        raise SystemExit(
            "E021_ZERO_RENTAL: --apply is forbidden; no Vast mutation path exists"
        )
    repo = Path(__file__).resolve().parents[4]
    phases = PHASES if arguments.phase == "all" else (arguments.phase,)
    receipts = []
    for phase in phases:
        receipt = run_phase(
            phase,
            repo=repo,
            checkpoint=arguments.checkpoint.resolve(),
            force=arguments.force,
        )
        receipts.append(receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
    # MODEL_INVALID is a scientific outcome rather than a process crash.  A
    # completed run returns zero; required gate status lives in summary.json.
    return 0


__all__ = ["main", "run_phase"]
