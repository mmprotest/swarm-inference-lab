"""Crash-tolerant, category-strict Experiment 010 evidence persistence."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from swarm_inference.experiments.experiment_010.schemas import EvidenceCategory

REQUIRED_FILES = (
    "README.md",
    "manifest.json",
    "environment.json",
    "repository_fingerprint.json",
    "colibri_dependency.json",
    "colibri_build.json",
    "colibri_cuda_build.json",
    "colibri_patch_manifest.json",
    "colibri_external_dispatch_build.json",
    "colibri_expert_worker_build.json",
    "expert_bank_manifest.json",
    "microshard_bank_manifest.json",
    "hardware_profile.json",
    "storage_profile.json",
    "pcie_profile.json",
    "model_inventory_level_a.json",
    "model_inventory_level_b.json",
    "level_b_gate_17_validation.json",
    "level_b_model_acquisition.json",
    "level_b_workload_summary.csv",
    "kimi_fixture_inventory.json",
    "worker_capabilities.json",
    "worker_budgets.json",
    "topology_inventory.json",
    "expert_ownership.json",
    "tensor_inventory.json",
    "microshard_inventory.json",
    "transport_profiles.json",
    "transport_achieved.csv",
    "codec_results.csv",
    "whole_expert_results.csv",
    "microshard_results.csv",
    "colibri_rpc_token_results.csv",
    "colibri_rpc_boundary_errors.csv",
    "forbidden_local_loads.csv",
    "real_model_cuda_results.csv",
    "real_model_failure_results.csv",
    "real_model_corruption_results.csv",
    "data_plane_results.csv",
    "coalescing_results.csv",
    "configuration_matrix.csv",
    "capacity_accounting.json",
    "routing_trace_summary.json",
    "routing_events.csv",
    "batching_results.csv",
    "prefill_plan.json",
    "decode_plan.json",
    "mixed_service_plan.json",
    "planner_candidates.json",
    "planner_results.csv",
    "worker_marginal_utility.csv",
    "failure_schedule.json",
    "failure_results.csv",
    "corruption_schedule.json",
    "verification_results.csv",
    "reputation_history.csv",
    "simulator_calibration.json",
    "simulator_validation.csv",
    "simulator_predictions.csv",
    "break_even_surface.csv",
    "kimi_operator_results.csv",
    "kimi_projections.csv",
    "correctness_results.json",
    "token_comparisons.json",
    "resource_timeseries.csv",
    "memory_residency_timeseries.csv",
    "page_fault_results.csv",
    "reuse_distance_curves.csv",
    "simulator_behavioral_parity.json",
    "full_run_completeness.json",
    "telemetry.ndjson",
    "verdict.json",
    "report.md",
    "SHA256SUMS.txt",
    "reproduce.ps1",
)


def timestamped_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid4().hex[:8]}"


class Experiment010Bundle:
    def __init__(self, root: Path, *, resume: bool) -> None:
        self.root = root.expanduser().resolve()
        if self.root.exists() and not resume and any(self.root.iterdir()):
            raise FileExistsError(f"non-empty evidence bundle requires resume: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("logs", "traces", "plots"):
            (self.root / name).mkdir(exist_ok=True)
        self.checkpoint_path = self.root / "checkpoint.json"
        if resume and self.checkpoint_path.is_file():
            self.checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8-sig"))
        else:
            self.checkpoint = {
                "schema_version": "experiment-010-checkpoint-v1",
                "completed_stages": [],
                "completed_configurations": [],
                "commands": [],
                "processes": [],
                "failures": [],
            }
        self.save_checkpoint()

    def _replace(self, relative: str, data: bytes) -> None:
        target = (self.root / relative).resolve()
        if self.root != target and self.root not in target.parents:
            raise ValueError("evidence path escaped bundle root")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.partial")
        temporary.write_bytes(data)
        os.replace(temporary, target)

    def write_text(self, relative: str, text: str) -> None:
        self._replace(relative, text.encode("utf-8"))

    def write_json(self, relative: str, payload: Any) -> None:
        self.write_text(
            relative,
            json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        )

    def write_csv(
        self,
        relative: str,
        rows: Iterable[dict[str, Any]],
        *,
        fields: list[str] | None = None,
    ) -> None:
        materialized = list(rows)
        columns = fields or sorted({field for row in materialized for field in row})
        if not columns:
            columns = ["status", "reason", "category"]
            materialized = [
                {
                    "status": "NOT_MEASURED",
                    "reason": "no observations were produced",
                    "category": None,
                }
            ]
        target = (self.root / relative).resolve()
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.partial")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in materialized:
                category = row.get("category")
                if category is not None:
                    EvidenceCategory(str(category))
                writer.writerow(
                    {
                        field: (
                            json.dumps(value, sort_keys=True, separators=(",", ":"))
                            if isinstance(value, (dict, list, tuple))
                            else value
                        )
                        for field, value in row.items()
                    }
                )
        os.replace(temporary, target)

    def write_ndjson(self, relative: str, rows: Iterable[dict[str, Any]]) -> None:
        lines = []
        for row in rows:
            category = row.get("category")
            if category is not None:
                EvidenceCategory(str(category))
            lines.append(json.dumps(row, sort_keys=True, default=str, allow_nan=False))
        self.write_text(relative, "\n".join(lines) + ("\n" if lines else ""))

    def save_checkpoint(self) -> None:
        self.checkpoint["updated_at_utc"] = datetime.now(UTC).isoformat()
        self.write_json("checkpoint.json", self.checkpoint)

    def complete_stage(self, stage: str) -> None:
        values = self.checkpoint.setdefault("completed_stages", [])
        if stage not in values:
            values.append(stage)
        self.save_checkpoint()

    def complete_configuration(self, configuration: str) -> None:
        values = self.checkpoint.setdefault("completed_configurations", [])
        if configuration not in values:
            values.append(configuration)
        self.save_checkpoint()

    def is_configuration_complete(self, configuration: str) -> bool:
        return configuration in self.checkpoint.get("completed_configurations", [])

    def record_command(
        self,
        command: list[str],
        *,
        environment: dict[str, str | None],
        exit_code: int | None,
        started_at: str,
        completed_at: str | None,
    ) -> None:
        self.checkpoint.setdefault("commands", []).append(
            {
                "command": command,
                "environment": environment,
                "exit_code": exit_code,
                "started_at": started_at,
                "completed_at": completed_at,
            }
        )
        self.save_checkpoint()

    def record_process(self, payload: dict[str, Any]) -> None:
        self.checkpoint.setdefault("processes", []).append(payload)
        self.save_checkpoint()

    def record_failure(
        self,
        stage: str,
        error: str,
        *,
        exit_code: int | None = None,
        supported: bool | None = None,
    ) -> None:
        self.checkpoint.setdefault("failures", []).append(
            {
                "stage": stage,
                "error": error,
                "exit_code": exit_code,
                "supported": supported,
                "captured_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        self.save_checkpoint()

    def audit(self) -> dict[str, Any]:
        missing = [name for name in REQUIRED_FILES if not (self.root / name).is_file()]
        empty = [
            name
            for name in REQUIRED_FILES
            if (self.root / name).is_file() and (self.root / name).stat().st_size == 0
        ]
        return {
            "required_count": len(REQUIRED_FILES),
            "missing": missing,
            "empty": empty,
            "complete": not missing and not empty,
            "routing_event_format": "csv because pyarrow is not a project dependency",
        }

    def artifact_manifest(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(item for item in self.root.rglob("*") if item.is_file()):
            if ".partial" in path.name or path.name == "manifest.json":
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
        return rows


def create_bundle_root(output: Path, *, explicit: bool) -> Path:
    if explicit:
        return output.expanduser().resolve() / "experiment_010"
    return (
        output.expanduser().resolve() / f"{timestamped_run_id()}-experiment-010" / "experiment_010"
    )
