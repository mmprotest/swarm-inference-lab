"""Frozen Experiment 024 constants and immutable-input validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_022.completion_inputs import (
    load_frozen_inventories,
)

EXPERIMENT_ID = "024"

HIDDEN = 7168
LATENT = 3584
TOPK = 16
DEGREE = 8
FLOAT_BYTES = 4
ROUTE_ID_BYTES = 4
ROUTE_WEIGHT_BYTES = 4
REMOTE_WORKERS = 7

STAGE_A_ROWS = (1, 2, 4)
STAGE_A_CONCURRENCY = (1, 8, 32, 128)
STAGE_A_PRIMARY_CONCURRENCY = 32
DECODE_CONCURRENCY_LEVELS = (1, 4, 16, 64, 128)

FAST_FABRIC_LATENCY_MS = 0.25
FAST_FABRIC_BANDWIDTH_GBPS = 25.0
FAST_FABRIC_SOFTWARE_OVERHEAD_MS = 0.04

KIMI_OUTPUT_API_PRICE_USD_PER_M = 15.0
PRIMARY_CONTRIBUTOR_PAYOUT_USD_PER_ACTIVE_NODE_HOUR = 0.15
PAYOUT_SENSITIVITY_USD_PER_ACTIVE_NODE_HOUR = (0.05, 0.10, 0.15, 0.25, 0.50)
TARGET_COST_LEVELS_USD_PER_M = (15.0, 12.0, 9.0, 7.5, 5.0, 3.0)

PRIMARY_DECODE_SLO_MULTIPLIER = 4.0
DECODE_SLO_SENSITIVITY_MULTIPLIERS = (2.0, 8.0)
PHYSICAL_RELATIVE_L2_MAX = 2e-6
SERVICE_VALIDATION_MEDIAN_ERROR_PERCENT_MAX = 10.0
SERVICE_VALIDATION_MAX_ERROR_PERCENT_MAX = 15.0
COMMUNICATION_LOWER_BOUND_RATIO_MAX = 1.03
STAGE_A_NETWORK_STRESS_PERCENT = 10.0
COMMODITY_AVAILABLE_NODE_BUDGETS = (96, 128, 160, 192, 224, 256, 320)
WHOLE_LAYER_INCAPABLE_COMPUTE_SHARE_MIN = 0.95
SWARM_D_MAX_REGRESSION_VS_CURRENT_PERCENT = 5.0
CALIBRATION_WARMUP = 20
CALIBRATION_ITERATIONS = 100
FUSION_WARMUP = 20
FUSION_ITERATIONS = 200

CHECKPOINT = Path(r"F:\models\Kimi-K3")
REPAIRED_SERVICE_RELATIVE_PATH = Path(
    "artifacts/experiment-022/completion/validation/"
    "repaired-resident-service.csv"
)
CANDIDATE_CATALOG_RELATIVE_PATH = Path(
    "artifacts/experiment-022/completion/rerun/candidate-catalog.json"
)


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest without mutating the input."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Phase0Audit:
    """Mechanical immutable-input audit performed before E024 implementation."""

    status: str
    mandatory_failure_id: str | None
    reason: str | None
    checkpoint_exists: bool
    transformer_layer_count: int | None
    e022_inventory_count: int
    e022_inventory_hashes_valid: bool
    e023_final_verdict: str | None
    candidate_catalog_path: str
    candidate_catalog_sha256: str
    repaired_service_path: str
    repaired_service_sha256: str
    layer_zero_p8_candidate_count: int
    layer_zero_admitted_p8_candidate_count: int
    layer_zero_candidates: tuple[dict[str, Any], ...]
    complete_p8_only_placement_possible: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _checkpoint_layer_count(checkpoint: Path) -> int | None:
    config_path = checkpoint / "config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    value = text_config.get("num_hidden_layers")
    return int(value) if value is not None else None


def _catalog_layer_zero_p8(catalog: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for candidate in catalog.get("candidates", []):
        if int(candidate.get("layer", -1)) != 0 or int(candidate.get("degree", -1)) != DEGREE:
            continue
        rows.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "candidate_type": str(candidate["candidate_type"]),
                "correctness_status": str(candidate.get("correctness_status")),
                "headline_eligible": bool(candidate.get("headline_eligible")),
                "production_native_binding": bool(
                    candidate.get("production_native_binding")
                ),
                "chunk_sizes_physically_validated": list(
                    candidate.get("chunk_sizes_physically_validated", [])
                ),
                "service_status": str(candidate.get("service_status")),
            }
        )
    return tuple(sorted(rows, key=lambda row: row["candidate_id"]))


def _candidate_is_admitted(candidate: dict[str, Any]) -> bool:
    return (
        candidate["headline_eligible"] is True
        and candidate["production_native_binding"] is True
        and candidate["correctness_status"] == "PASS"
        and candidate["chunk_sizes_physically_validated"] == [1, 2, 4]
    )


def audit_immutable_inputs(repo_root: Path) -> Phase0Audit:
    """Audit the frozen inputs and fail on the first complete-P8 impossibility.

    A valid Stage B placement must choose one physically admitted P8 candidate
    for every transformer layer.  Network size and memory cannot repair an
    empty candidate set, so this check soundly precedes placement search.
    """

    repo_root = repo_root.resolve()
    inventories, inventory_audit = load_frozen_inventories(repo_root)
    inventory_count = len(inventories)

    e023_summary_path = repo_root / "artifacts/experiment-023/summary.json"
    e023_verdict = None
    if e023_summary_path.exists():
        e023_verdict = str(
            json.loads(e023_summary_path.read_text(encoding="utf-8")).get(
                "final_verdict"
            )
        )

    catalog_path = repo_root / CANDIDATE_CATALOG_RELATIVE_PATH
    repaired_path = repo_root / REPAIRED_SERVICE_RELATIVE_PATH
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    layer_zero = _catalog_layer_zero_p8(catalog)
    admitted = tuple(row for row in layer_zero if _candidate_is_admitted(row))

    checkpoint_exists = CHECKPOINT.is_dir()
    layer_count = _checkpoint_layer_count(CHECKPOINT) if checkpoint_exists else None
    inventory_hashes_valid = (
        inventory_count == 27 and inventory_audit.get("status") == "PASS"
    )
    base_valid = (
        checkpoint_exists
        and layer_count == 93
        and inventory_hashes_valid
        and e023_verdict == "NO_WEDGE"
        and repaired_path.is_file()
    )
    complete_p8 = base_valid and bool(admitted)

    failure_id: str | None = None
    reason: str | None = None
    if not base_valid:
        failure_id = "HISTORICAL_IMMUTABLE_INPUT_VALIDATION"
        reason = "One or more frozen historical/checkpoint prerequisites failed."
    elif not admitted:
        failure_id = "NO_PRODUCTION_NATIVE_P8_CANDIDATE_FOR_LAYER_0"
        reason = (
            "The frozen E022 catalog has no physically admitted, production-native "
            "degree-8 candidate for transformer layer 0. E024 forbids WHOLE_LAYER "
            "for every transformer layer, so no complete 93-layer P8-only placement "
            "can exist at any commodity node budget."
        )

    return Phase0Audit(
        status="PASS" if failure_id is None else "MODEL_INVALID",
        mandatory_failure_id=failure_id,
        reason=reason,
        checkpoint_exists=checkpoint_exists,
        transformer_layer_count=layer_count,
        e022_inventory_count=inventory_count,
        e022_inventory_hashes_valid=inventory_hashes_valid,
        e023_final_verdict=e023_verdict,
        candidate_catalog_path=CANDIDATE_CATALOG_RELATIVE_PATH.as_posix(),
        candidate_catalog_sha256=sha256_file(catalog_path),
        repaired_service_path=REPAIRED_SERVICE_RELATIVE_PATH.as_posix(),
        repaired_service_sha256=sha256_file(repaired_path),
        layer_zero_p8_candidate_count=len(layer_zero),
        layer_zero_admitted_p8_candidate_count=len(admitted),
        layer_zero_candidates=layer_zero,
        complete_p8_only_placement_possible=complete_p8,
    )


def frozen_constants() -> dict[str, Any]:
    """Return the preregistered constants in JSON-compatible form."""

    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in globals().items()
        if name.isupper() and isinstance(value, (str, int, float, tuple))
    }


__all__ = [
    "CALIBRATION_ITERATIONS",
    "CALIBRATION_WARMUP",
    "CHECKPOINT",
    "COMMODITY_AVAILABLE_NODE_BUDGETS",
    "COMMUNICATION_LOWER_BOUND_RATIO_MAX",
    "DECODE_CONCURRENCY_LEVELS",
    "DEGREE",
    "EXPERIMENT_ID",
    "FLOAT_BYTES",
    "FUSION_ITERATIONS",
    "FUSION_WARMUP",
    "HIDDEN",
    "KIMI_OUTPUT_API_PRICE_USD_PER_M",
    "LATENT",
    "PAYOUT_SENSITIVITY_USD_PER_ACTIVE_NODE_HOUR",
    "PRIMARY_CONTRIBUTOR_PAYOUT_USD_PER_ACTIVE_NODE_HOUR",
    "PRIMARY_DECODE_SLO_MULTIPLIER",
    "REMOTE_WORKERS",
    "ROUTE_ID_BYTES",
    "ROUTE_WEIGHT_BYTES",
    "STAGE_A_CONCURRENCY",
    "STAGE_A_PRIMARY_CONCURRENCY",
    "STAGE_A_ROWS",
    "TARGET_COST_LEVELS_USD_PER_M",
    "TOPK",
    "Phase0Audit",
    "audit_immutable_inputs",
    "frozen_constants",
    "sha256_file",
]
