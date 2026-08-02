"""Crash-tolerant evidence bundle persistence for Experiment 008."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

REQUIRED_FILES = (
    "README.md",
    "manifest.json",
    "environment.json",
    "hardware_profile.json",
    "model_preflight.json",
    "tensor_inventory.json",
    "tensor_tiles.json",
    "expert_trace_summary.json",
    "expert_activation_matrix.csv",
    "expert_coactivation.csv",
    "baseline_search.json",
    "candidate_plans.json",
    "prefill_plan.json",
    "decode_plan.json",
    "adaptive_plan.json",
    "cost_model_predictions.csv",
    "benchmark_results.csv",
    "ablation_results.csv",
    "correctness_results.json",
    "residency_accounting.json",
    "resource_timeseries.csv",
    "report.md",
    "verdict.json",
    "reproduce.ps1",
)


def timestamped_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid4().hex[:8]}"


class EvidenceBundle:
    def __init__(self, root: Path, *, resume: bool) -> None:
        self.root = root.expanduser().resolve()
        if self.root.exists() and not resume and any(self.root.iterdir()):
            raise FileExistsError(
                f"output bundle already exists and is not empty: {self.root}; use --resume"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("profiler_trace", "logs", "plots"):
            (self.root / name).mkdir(exist_ok=True)
        self.checkpoint_path = self.root / "checkpoint.json"
        self.checkpoint = (
            self._load_checkpoint()
            if resume
            else {
                "schema_version": "experiment-008-checkpoint-v1",
                "completed_stages": [],
                "completed_configurations": [],
                "failures": [],
                "updated_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        self.save_checkpoint()

    def _load_checkpoint(self) -> dict[str, Any]:
        if not self.checkpoint_path.is_file():
            return {
                "schema_version": "experiment-008-checkpoint-v1",
                "completed_stages": [],
                "completed_configurations": [],
                "failures": [],
                "updated_at_utc": datetime.now(UTC).isoformat(),
            }
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Experiment 008 checkpoint is not a JSON object")
        return payload

    def _replace_bytes(self, relative: str, data: bytes) -> None:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.partial")
        temporary.write_bytes(data)
        os.replace(temporary, target)

    def write_text(self, relative: str, text: str) -> None:
        self._replace_bytes(relative, text.encode("utf-8"))

    def write_json(self, relative: str, payload: Any) -> None:
        self.write_text(
            relative,
            json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        )

    def write_csv(
        self, relative: str, rows: Iterable[dict[str, Any]], *, fields: list[str] | None = None
    ) -> None:
        materialized = list(rows)
        if fields is None:
            fields = sorted({key for row in materialized for key in row})
        if not fields:
            fields = ["status", "reason"]
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.partial")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in materialized:
                writer.writerow(
                    {
                        key: (
                            json.dumps(value, sort_keys=True, separators=(",", ":"))
                            if isinstance(value, (list, dict))
                            else value
                        )
                        for key, value in row.items()
                    }
                )
        os.replace(temporary, target)

    def save_checkpoint(self) -> None:
        self.checkpoint["updated_at_utc"] = datetime.now(UTC).isoformat()
        self.write_json("checkpoint.json", self.checkpoint)

    def complete_stage(self, stage: str) -> None:
        stages = self.checkpoint.setdefault("completed_stages", [])
        if stage not in stages:
            stages.append(stage)
        self.save_checkpoint()

    def complete_configuration(self, configuration: str) -> None:
        configurations = self.checkpoint.setdefault("completed_configurations", [])
        if configuration not in configurations:
            configurations.append(configuration)
        self.save_checkpoint()

    def record_failure(
        self,
        *,
        stage: str,
        error: str,
        exit_code: int | None = None,
        status: str = "FAILED",
    ) -> None:
        self.checkpoint.setdefault("failures", []).append(
            {
                "stage": stage,
                "status": status,
                "error": error,
                "exit_code": exit_code,
                "captured_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        self.save_checkpoint()

    def is_stage_complete(self, stage: str) -> bool:
        return stage in self.checkpoint.get("completed_stages", [])

    def is_configuration_complete(self, configuration: str) -> bool:
        return configuration in self.checkpoint.get("completed_configurations", [])

    def audit_required(self) -> dict[str, Any]:
        missing = [name for name in REQUIRED_FILES if not (self.root / name).is_file()]
        return {
            "required_file_count": len(REQUIRED_FILES),
            "missing": missing,
            "complete": not missing,
        }


def create_bundle_root(output_root: Path, *, explicit: bool) -> Path:
    if explicit:
        return output_root.expanduser().resolve() / "experiment_008"
    return (
        output_root.expanduser().resolve()
        / f"{timestamped_run_id()}-experiment-008"
        / "experiment_008"
    )
