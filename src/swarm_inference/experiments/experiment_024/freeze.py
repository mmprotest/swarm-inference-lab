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
TARGET_COST_LEVELS_USD_PER_M = (15.0, 12.0, 9.0, 7.50, 5.0, 3.0)

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
DENSE_LAYER0_CALIBRATION_ITERATIONS = 200
FUSION_WARMUP = 20
FUSION_ITERATIONS = 200

COMMODITY_WORKER_MEMORY_BYTES = 10_422_845_440
LAYER_ZERO_WHOLE_CANDIDATE_ID = "layer-00:WHOLE_LAYER:p1"
LAYER_ZERO_WHOLE_RESIDENT_BYTES = 2_549_338_530
P8_REQUIRED_LAYER_IDS = tuple(range(1, 93))
EXPECTED_CANDIDATE_CATALOG_SHA256 = (
    "3f1d8e8519fb2b759b7ec5678bfc1458a782a49a6e438eeb1b26d2077258ccd7"
)
EXPECTED_REPAIRED_SERVICE_SHA256 = (
    "ee240937dfc36a6ce04161812f61ff0021359a2861ecb783772d44e2614bcf2b"
)

CHECKPOINT = Path(r"F:\models\Kimi-K3")
REPAIRED_SERVICE_RELATIVE_PATH = Path(
    "artifacts/experiment-022/completion/validation/"
    "repaired-resident-service.csv"
)
CANDIDATE_CATALOG_RELATIVE_PATH = Path(
    "artifacts/experiment-022/completion/rerun/candidate-catalog.json"
)
GENERATOR_CONFIG_RELATIVE_PATH = Path(
    "artifacts/experiment-022/inventories/generator-config.json"
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
    """Mechanical immutable-input and corrected-architecture audit."""

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
    commodity_worker_memory_bytes: int
    layer_zero_whole_candidate_id: str
    layer_zero_whole_candidate_admitted: bool
    layer_zero_whole_resident_bytes: int | None
    layer_zero_whole_fits_commodity: bool
    p8_required_layer_count: int
    p8_required_layer_ids: tuple[int, ...]
    p8_admitted_candidate_counts_by_layer: dict[int, int]
    p8_admitted_candidate_ids_by_layer: dict[int, tuple[str, ...]]
    whole_layer_resident_bytes_by_layer: dict[int, int]
    whole_layer_feasible_layer_ids_on_commodity: tuple[int, ...]
    whole_layer_infeasible_layer_ids_on_commodity: tuple[int, ...]
    complete_commodity_candidate_coverage: bool
    complete_commodity_architecture_candidate_coverage: bool
    whole_layer_only_commodity_model_feasible: bool

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


def _candidate_is_physically_admitted(candidate: dict[str, Any]) -> bool:
    return (
        candidate.get("headline_eligible") is True
        and candidate.get("production_native_binding") is True
        and candidate.get("correctness_status") == "PASS"
        and list(candidate.get("chunk_sizes_physically_validated", [])) == [1, 2, 4]
    )


def _admitted_p8(candidate: dict[str, Any]) -> bool:
    return (
        int(candidate.get("degree", -1)) == DEGREE
        and candidate.get("candidate_type") != "WHOLE_LAYER"
        and _candidate_is_physically_admitted(candidate)
    )


def _admitted_whole(candidate: dict[str, Any]) -> bool:
    return (
        int(candidate.get("degree", -1)) == 1
        and candidate.get("candidate_type") == "WHOLE_LAYER"
        and _candidate_is_physically_admitted(candidate)
    )


def _catalog_candidates_by_layer(catalog: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result = {layer: [] for layer in range(93)}
    for candidate in catalog.get("candidates", []):
        layer = int(candidate.get("layer", -1))
        if layer in result:
            result[layer].append(candidate)
    return result


def _whole_candidate(
    candidates: list[dict[str, Any]], *, candidate_id: str | None = None
) -> dict[str, Any] | None:
    rows = [
        row
        for row in candidates
        if _admitted_whole(row)
        and (candidate_id is None or str(row.get("candidate_id")) == candidate_id)
    ]
    if len(rows) != 1:
        return None
    return rows[0]


def _resident_bytes(candidate: dict[str, Any] | None) -> int | None:
    if candidate is None:
        return None
    values = [int(value) for value in candidate.get("resident_memory_bytes", [])]
    return values[0] if len(values) == 1 else None


def audit_immutable_inputs(repo_root: Path) -> Phase0Audit:
    """Validate the corrected one-whole-plus-92-P8 commodity architecture."""

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
    generator_path = repo_root / GENERATOR_CONFIG_RELATIVE_PATH
    catalog_sha = sha256_file(catalog_path) if catalog_path.is_file() else ""
    repaired_sha = sha256_file(repaired_path) if repaired_path.is_file() else ""
    catalog = (
        json.loads(catalog_path.read_text(encoding="utf-8"))
        if catalog_path.is_file()
        else {"candidates": []}
    )
    candidates_by_layer = _catalog_candidates_by_layer(catalog)

    commodity_memory = -1
    if generator_path.is_file():
        generator = json.loads(generator_path.read_text(encoding="utf-8"))
        commodity_memory = int(generator["memory_classes_bytes"]["sub_layer"])

    layer_zero = _whole_candidate(
        candidates_by_layer[0], candidate_id=LAYER_ZERO_WHOLE_CANDIDATE_ID
    )
    layer_zero_resident = _resident_bytes(layer_zero)
    layer_zero_admitted = layer_zero is not None
    layer_zero_fits = (
        layer_zero_resident is not None and layer_zero_resident <= commodity_memory
    )

    p8_rows = {
        layer: tuple(
            sorted(
                (
                    row
                    for row in candidates_by_layer[layer]
                    if _admitted_p8(row)
                ),
                key=lambda row: str(row["candidate_id"]),
            )
        )
        for layer in P8_REQUIRED_LAYER_IDS
    }
    p8_counts = {layer: len(rows) for layer, rows in p8_rows.items()}
    p8_ids = {
        layer: tuple(str(row["candidate_id"]) for row in rows)
        for layer, rows in p8_rows.items()
    }

    whole_rows = {
        layer: _whole_candidate(candidates_by_layer[layer]) for layer in range(93)
    }
    whole_resident = {
        layer: resident
        for layer, row in whole_rows.items()
        if (resident := _resident_bytes(row)) is not None
    }
    whole_feasible = tuple(
        layer
        for layer in range(93)
        if whole_resident.get(layer, commodity_memory + 1) <= commodity_memory
    )
    whole_infeasible = tuple(
        layer
        for layer in range(93)
        if whole_resident.get(layer, 0) > commodity_memory
    )

    checkpoint_exists = CHECKPOINT.is_dir()
    layer_count = _checkpoint_layer_count(CHECKPOINT) if checkpoint_exists else None
    inventory_hashes_valid = (
        inventory_count == 27 and inventory_audit.get("status") == "PASS"
    )
    historical_valid = (
        checkpoint_exists
        and layer_count == 93
        and inventory_hashes_valid
        and e023_verdict == "NO_WEDGE"
        and repaired_path.is_file()
        and generator_path.is_file()
        and catalog_sha == EXPECTED_CANDIDATE_CATALOG_SHA256
        and repaired_sha == EXPECTED_REPAIRED_SERVICE_SHA256
        and commodity_memory == COMMODITY_WORKER_MEMORY_BYTES
    )

    p8_coverage = all(p8_counts[layer] >= 1 for layer in P8_REQUIRED_LAYER_IDS)
    whole_catalog_coverage = len(whole_resident) == 93
    layer_zero_values_exact = layer_zero_resident == LAYER_ZERO_WHOLE_RESIDENT_BYTES
    complete_coverage = (
        historical_valid
        and layer_zero_admitted
        and layer_zero_values_exact
        and layer_zero_fits
        and p8_coverage
        and whole_catalog_coverage
        and whole_feasible == (0,)
        and whole_infeasible == P8_REQUIRED_LAYER_IDS
    )

    failure_id: str | None = None
    reason: str | None = None
    if not historical_valid or not whole_catalog_coverage:
        failure_id = "HISTORICAL_IMMUTABLE_INPUT_VALIDATION"
        reason = "One or more frozen historical, checkpoint, catalog, service, or memory prerequisites failed."
    elif not layer_zero_admitted or not layer_zero_values_exact:
        failure_id = "NO_ADMITTED_WHOLE_LAYER_CANDIDATE_FOR_DENSE_LAYER_0"
        reason = (
            "The exact frozen layer-00:WHOLE_LAYER:p1 candidate is missing, not physically "
            "admitted, or does not carry the frozen resident-memory value."
        )
    elif not layer_zero_fits:
        failure_id = "DENSE_LAYER_0_DOES_NOT_FIT_COMMODITY_WORKER"
        reason = "The admitted dense layer-0 whole candidate exceeds commodity memory."
    else:
        missing = next(
            (layer for layer in P8_REQUIRED_LAYER_IDS if p8_counts[layer] < 1),
            None,
        )
        if missing is not None:
            failure_id = f"MISSING_ADMITTED_P8_CANDIDATE_FOR_LAYER_{missing}"
            reason = f"Transformer layer {missing} lacks an admitted degree-8 candidate."
        elif whole_feasible != (0,) or whole_infeasible != P8_REQUIRED_LAYER_IDS:
            failure_id = "NON_DENSE_LAYER_WHOLE_FITS_COMMODITY_WORKER"
            reason = (
                "The frozen catalog and commodity memory do not yield exactly layer 0 as "
                "whole-layer feasible and layers 1 through 92 as whole-layer infeasible."
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
        candidate_catalog_sha256=catalog_sha,
        repaired_service_path=REPAIRED_SERVICE_RELATIVE_PATH.as_posix(),
        repaired_service_sha256=repaired_sha,
        commodity_worker_memory_bytes=commodity_memory,
        layer_zero_whole_candidate_id=LAYER_ZERO_WHOLE_CANDIDATE_ID,
        layer_zero_whole_candidate_admitted=layer_zero_admitted,
        layer_zero_whole_resident_bytes=layer_zero_resident,
        layer_zero_whole_fits_commodity=layer_zero_fits,
        p8_required_layer_count=len(P8_REQUIRED_LAYER_IDS),
        p8_required_layer_ids=P8_REQUIRED_LAYER_IDS,
        p8_admitted_candidate_counts_by_layer=p8_counts,
        p8_admitted_candidate_ids_by_layer=p8_ids,
        whole_layer_resident_bytes_by_layer=whole_resident,
        whole_layer_feasible_layer_ids_on_commodity=whole_feasible,
        whole_layer_infeasible_layer_ids_on_commodity=whole_infeasible,
        complete_commodity_candidate_coverage=complete_coverage,
        complete_commodity_architecture_candidate_coverage=complete_coverage,
        whole_layer_only_commodity_model_feasible=len(whole_feasible) == 93,
    )


def frozen_constants() -> dict[str, Any]:
    """Return preregistered scalar and tuple constants in JSON-compatible form."""

    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in globals().items()
        if name.isupper() and isinstance(value, (str, int, float, tuple))
    }


__all__ = [
    "CALIBRATION_ITERATIONS",
    "CALIBRATION_WARMUP",
    "CANDIDATE_CATALOG_RELATIVE_PATH",
    "CHECKPOINT",
    "COMMODITY_AVAILABLE_NODE_BUDGETS",
    "COMMODITY_WORKER_MEMORY_BYTES",
    "COMMUNICATION_LOWER_BOUND_RATIO_MAX",
    "DECODE_CONCURRENCY_LEVELS",
    "DEGREE",
    "DENSE_LAYER0_CALIBRATION_ITERATIONS",
    "EXPERIMENT_ID",
    "FLOAT_BYTES",
    "FUSION_ITERATIONS",
    "FUSION_WARMUP",
    "HIDDEN",
    "KIMI_OUTPUT_API_PRICE_USD_PER_M",
    "LATENT",
    "LAYER_ZERO_WHOLE_CANDIDATE_ID",
    "LAYER_ZERO_WHOLE_RESIDENT_BYTES",
    "P8_REQUIRED_LAYER_IDS",
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
    "WHOLE_LAYER_INCAPABLE_COMPUTE_SHARE_MIN",
    "Phase0Audit",
    "audit_immutable_inputs",
    "frozen_constants",
    "sha256_file",
]
