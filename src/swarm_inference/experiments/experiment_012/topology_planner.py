"""Evidence-calibrated branch-factor planner for Experiment 012.

The planner is deliberately small.  H012-007 tests one mechanism only: an
additive activation-plus-depth-times-RTT model calibrated from the immutable
H012-006 same-host measurements.  Held-out measurements are evaluation data,
never planner inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_012.baseline_harness import _sha256_file
from swarm_inference.microworker_protocol import LinkProfile

CALIBRATION_HYPOTHESIS_ID = "H012-006"
CALIBRATION_PROFILE = "same_host_shaped"
CALIBRATION_RTT_MS = 0.1
CANDIDATE_BRANCH_FACTORS = (8, 32)


def calibrate_from_summary(
    summary_path: Path,
    *,
    worker_count: int = 128,
    branch_factors: tuple[int, ...] = CANDIDATE_BRANCH_FACTORS,
) -> dict[str, Any]:
    """Extract activation costs from the declared H012-006 control only."""

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("hypothesis_id") != CALIBRATION_HYPOTHESIS_ID:
        raise ValueError("planner calibration source is not H012-006")

    candidates: list[dict[str, Any]] = []
    for branch_factor in branch_factors:
        matches = [
            row
            for row in summary.get("summaries", [])
            if row.get("network_profile") == CALIBRATION_PROFILE
            and int(row.get("worker_count", -1)) == worker_count
            and int(row.get("branch_factor", -1)) == branch_factor
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one H012-006 calibration row for B={branch_factor}, got {len(matches)}"
            )
        row = matches[0]
        latency_ms = float(row["end_to_end_latency_p50_ms"])
        depth = int(row["hierarchy_depth_median"])
        candidates.append(
            {
                "branch_factor": branch_factor,
                "hierarchy_depth": depth,
                "same_host_latency_p50_ms": latency_ms,
                "activation_ms": latency_ms - depth * CALIBRATION_RTT_MS,
            }
        )

    return {
        "schema_version": "1.0",
        "mechanism": "activation_ms + hierarchy_depth * rtt_ms",
        "calibration_hypothesis_id": CALIBRATION_HYPOTHESIS_ID,
        "calibration_profile": CALIBRATION_PROFILE,
        "calibration_rtt_ms": CALIBRATION_RTT_MS,
        "worker_count": worker_count,
        "source": {
            "path": str(summary_path),
            "sha256": _sha256_file(summary_path),
            "bytes": summary_path.stat().st_size,
        },
        "candidates": candidates,
        "excluded_inputs": "all non-same-host H012-006 rows and all H012-007 measurements",
    }


def choose_branch_factor(calibration: dict[str, Any], profile: LinkProfile) -> dict[str, Any]:
    """Choose the lowest predicted latency, breaking exact ties toward lower degree."""

    scores = [
        {
            "branch_factor": int(candidate["branch_factor"]),
            "hierarchy_depth": int(candidate["hierarchy_depth"]),
            "activation_ms": float(candidate["activation_ms"]),
            "predicted_latency_ms": float(candidate["activation_ms"])
            + int(candidate["hierarchy_depth"]) * profile.rtt_ms,
        }
        for candidate in calibration["candidates"]
    ]
    selected = min(
        scores, key=lambda score: (score["predicted_latency_ms"], score["branch_factor"])
    )
    return {
        "network_profile": profile.name,
        "rtt_ms": profile.rtt_ms,
        "upload_mbps": profile.upload_mbps,
        "download_mbps": profile.download_mbps,
        "jitter_ms": profile.jitter_ms,
        "selected_branch_factor": selected["branch_factor"],
        "selected_predicted_latency_ms": selected["predicted_latency_ms"],
        "candidate_scores": scores,
        "tie_break": "lower branch factor",
    }


def choose_profiles(
    calibration: dict[str, Any], profiles: tuple[LinkProfile, ...]
) -> list[dict[str, Any]]:
    return [choose_branch_factor(calibration, profile) for profile in profiles]


__all__ = [
    "CALIBRATION_HYPOTHESIS_ID",
    "CALIBRATION_PROFILE",
    "CALIBRATION_RTT_MS",
    "CANDIDATE_BRANCH_FACTORS",
    "calibrate_from_summary",
    "choose_branch_factor",
    "choose_profiles",
]
