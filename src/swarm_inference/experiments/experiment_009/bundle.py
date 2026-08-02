"""Crash-tolerant evidence bundle persistence for Experiment 009."""

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
    "colibri_dependency.json",
    "colibri_patch_manifest.json",
    "colibri_build.json",
    "capability_report.json",
    "model_inventory.json",
    "tensor_inventory.json",
    "expert_inventory.json",
    "native_quantization_inventory.json",
    "storage_inventory.json",
    "hardware_profile.json",
    "colibri_resource_plan.json",
    "swarm_resource_plan.json",
    "routing_trace_summary.json",
    "expert_activation.csv",
    "expert_coactivation.csv",
    "expert_transitions.csv",
    "tier_residency.csv",
    "cache_events.csv",
    "storage_events.csv",
    "telemetry.ndjson",
    "replay_tokens.json",
    "tuning_candidates.json",
    "tuning_results.csv",
    "heldout_policy_results.csv",
    "correctness_results.json",
    "adapter_overhead_results.csv",
    "microshard_descriptors.json",
    "benchmark_results.csv",
    "verdict.json",
    "report.md",
    "reproduce.ps1",
)


def timestamped_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid4().hex[:8]}"


class EvidenceBundle:
    def __init__(self, root: Path, *, resume: bool) -> None:
        self.root = root.expanduser().resolve()
        if self.root.exists() and not resume and any(self.root.iterdir()):
            raise FileExistsError(f"non-empty evidence bundle requires --resume: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("logs", "plots"):
            (self.root / name).mkdir(exist_ok=True)
        self.checkpoint_path = self.root / "checkpoint.json"
        if resume and self.checkpoint_path.is_file():
            self.checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        else:
            self.checkpoint = {
                "schema_version": "experiment-009-checkpoint-v1",
                "completed_stages": [],
                "completed_configurations": [],
                "failures": [],
            }
        self.save_checkpoint()

    def _replace(self, relative: str, data: bytes) -> None:
        target = self.root / relative
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
        columns = fields or sorted({key for row in materialized for key in row})
        if not columns:
            columns = ["status", "reason"]
            materialized = [{"status": "NOT_AVAILABLE", "reason": "no measured rows"}]
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.partial")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
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
        completed = self.checkpoint.setdefault("completed_stages", [])
        if stage not in completed:
            completed.append(stage)
        self.save_checkpoint()

    def complete_configuration(self, configuration: str) -> None:
        completed = self.checkpoint.setdefault("completed_configurations", [])
        if configuration not in completed:
            completed.append(configuration)
        self.save_checkpoint()

    def is_stage_complete(self, stage: str) -> bool:
        return stage in self.checkpoint.get("completed_stages", [])

    def is_configuration_complete(self, configuration: str) -> bool:
        return configuration in self.checkpoint.get("completed_configurations", [])

    def record_failure(
        self, stage: str, error: str, *, exit_code: int | None = None, timeout: bool = False
    ) -> None:
        self.checkpoint.setdefault("failures", []).append(
            {
                "stage": stage,
                "error": error,
                "exit_code": exit_code,
                "timeout": timeout,
                "captured_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        self.save_checkpoint()

    def audit(self) -> dict[str, Any]:
        missing = [name for name in REQUIRED_FILES if not (self.root / name).is_file()]
        return {"required": len(REQUIRED_FILES), "missing": missing, "complete": not missing}


def create_bundle_root(output: Path, *, explicit: bool) -> Path:
    if explicit:
        return output.expanduser().resolve() / "experiment_009"
    return (
        output.expanduser().resolve() / f"{timestamped_run_id()}-experiment-009" / "experiment_009"
    )
