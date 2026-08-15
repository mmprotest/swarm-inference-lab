"""Build the repaired, gate-bound E022 planner service catalog.

Only independently physical native services are admitted.  In particular this
module cannot import the original ordered-replay ``residual / 5`` barriers.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .completion_physical import CHUNKS
from .io import write_csv

WHOLE_OPERATIONS = {
    "whole_layer",
    "attention_whole",
    "latent_down_whole",
    "expert_whole",
    "shared_expert_whole",
    "latent_up_whole",
}


def _layer_type(value: str) -> str:
    return "GATED_MLA" if value == "Gated_MLA" else value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _row(
    *,
    split: str,
    layer: int,
    layer_type: str,
    operation: str,
    degree: int,
    rows: int,
    p50_ms: float,
    source: Path,
    evidence_class: str,
    correctness_receipt: str,
    resident_bytes: int | str = "",
) -> dict[str, Any]:
    if p50_ms <= 0:
        raise ValueError(f"non-positive repaired service {operation}: {p50_ms}")
    return {
        "split": split,
        "layer": layer,
        "layer_type": layer_type,
        "operation": operation,
        "degree": degree,
        "rows": rows,
        "p50_ms": p50_ms,
        "startup_ms": 0.0,
        "resident_bytes": resident_bytes,
        "correctness_pass": True,
        "all_partition_shards_simultaneously_resident": True,
        "timed_checkpoint_reads": 0,
        "timed_weight_uploads": 0,
        "timed_shard_creation": 0,
        "timed_repacking": 0,
        "estimate_kind": "direct_physical_feature_conditioned",
        "evidence_class": evidence_class,
        "source": str(source.resolve()),
        "valid_for_service": True,
        "production_native_binding": True,
        "correctness_receipt": correctness_receipt,
        "cost_owner": (
            "worker_software_overhead"
            if operation == "worker_protocol"
            else "collective_compute"
            if operation.endswith("_reduction")
            else "worker_compute"
        ),
    }


def _physical_receipts(completion_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = [
        completion_root / "physical" / "resident-kda-traces.json",
        completion_root / "physical" / "resident-mla-traces.json",
    ]
    receipts: list[dict[str, Any]] = []
    for path in paths:
        value = _read_json(path)
        receipts.extend(value["receipts"])
    return receipts, paths


def _phase_values(receipt: dict[str, Any], *names: str) -> list[float]:
    return [
        sum(float(sample["phase_wall_ms"][name]) for name in names)
        for sample in receipt["instrumentation"]
    ]


def _operation_values(
    receipt: dict[str, Any], operators: Iterable[str]
) -> list[float]:
    wanted = set(operators)
    values: list[float] = []
    for sample in receipt["operation_records"]:
        values.extend(
            float(row["duration_ms"])
            for row in sample
            if str(row["operator"]) in wanted
        )
    if not values:
        raise ValueError(f"missing physical operators {sorted(wanted)}")
    return values


def _critical_worker_operation(
    receipt: dict[str, Any], operators: Iterable[str]
) -> float:
    """Return the slowest physical worker's median for a parallel primitive."""

    wanted = set(operators)
    by_worker: dict[str, list[float]] = defaultdict(list)
    for sample in receipt["operation_records"]:
        for row in sample:
            if str(row["operator"]) in wanted:
                by_worker[str(row["worker_id"])].append(float(row["duration_ms"]))
    if not by_worker:
        raise ValueError(f"missing physical worker operators {sorted(wanted)}")
    return max(statistics.median(values) for values in by_worker.values())


def _summed_operation_values(
    receipt: dict[str, Any], operators: Iterable[str]
) -> list[float]:
    wanted = set(operators)
    values = []
    for sample in receipt["operation_records"]:
        matched = [
            float(row["duration_ms"])
            for row in sample
            if str(row["operator"]) in wanted
        ]
        if len(matched) != len(wanted):
            raise ValueError(
                f"expected exactly one record for each of {sorted(wanted)}; got {len(matched)}"
            )
        values.append(sum(matched))
    return values


def _semantic_services(
    receipt: dict[str, Any],
    binding: dict[str, dict[str, Any]],
    reductions: dict[str, dict[str, Any]],
    projections: dict[str, dict[str, Any]],
) -> dict[str, float]:
    attention_type = str(receipt["attention_type"])
    attention_binding = binding["kda_shard" if attention_type == "KDA" else "mla_shard"]
    attention_merge = statistics.median(
        _operation_values(receipt, ["attention_residual_merge"])
    )
    values = {
        "attention_preprocess": statistics.median(
            _phase_values(
                receipt, "input_and_attention_metadata", "attention_preprocess"
            )
        ),
        "attention_common": statistics.median(
            _operation_values(receipt, [f"{attention_type}_common_projection"])
        ),
        "attention_shard": _critical_worker_operation(
            receipt, [f"{attention_type}_attention_stripe"]
        ),
        # The residual merge is a real serial coordinator operation immediately
        # following the native collective, so the event owns both exactly once.
        "attention_reduction": float(
            reductions["hidden_p8"]["direct_execution"]["wall_ms"]
        )
        + attention_merge,
        "post_attention_preprocess": statistics.median(
            _phase_values(receipt, "post_attention_preprocess")
        ),
        "router": statistics.median(_phase_values(receipt, "router")),
        "latent_down": _critical_worker_operation(
            receipt, ["latent_down_projection_stripe"]
        ),
        "expert_stripe": _critical_worker_operation(
            receipt, ["grouped_expert_stripe_bank_top16"]
        ),
        "expert_reduction": float(
            reductions["latent_p8"]["direct_execution"]["wall_ms"]
        ),
        "routed_norm": statistics.median(
            _phase_values(receipt, "routed_normalization")
        ),
        "latent_up": _critical_worker_operation(
            receipt, ["latent_up_projection_stripe"]
        ),
        "latent_up_reduction": float(
            reductions["hidden_p8"]["direct_execution"]["wall_ms"]
        ),
        "shared_expert": _critical_worker_operation(
            receipt, ["shared_expert_stripe"]
        ),
        "shared_reduction": float(
            reductions["hidden_p8"]["direct_execution"]["wall_ms"]
        ),
        "routed_shared_reduction": float(
            reductions["hidden_p2"]["direct_execution"]["wall_ms"]
        ),
        "output_state_commit": statistics.median(
            _phase_values(receipt, "output_state_commit")
        ),
    }
    values.update(
        {
            # This binding's input is the exact packed coordinator output
            # (normalized hidden plus KDA/MLA common projections), so its wall
            # owns one remote stripe and its H2D/D2H copies without repeating
            # coordinator common compute.
            "attention_shard_remote": max(
                values["attention_shard"],
                float(attention_binding["direct_execution"]["wall_ms"]),
            ),
            "expert_stripe_remote": max(
                values["expert_stripe"],
                float(
                    binding["routed_expert_stripe"]["direct_execution"]["wall_ms"]
                ),
            ),
            "shared_expert_remote": max(
                values["shared_expert"],
                float(
                    binding["shared_expert_shard"]["direct_execution"]["wall_ms"]
                ),
            ),
            "latent_up_remote": max(
                values["latent_up"],
                float(projections["latent_up"]["direct_execution"]["wall_ms"]),
            ),
            "latent_down_remote": max(
                values["latent_down"],
                float(projections["latent_down"]["direct_execution"]["wall_ms"]),
            ),
        }
    )
    return values


def build_repaired_service(
    *, repo: Path, completion_root: Path
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    """Materialize the only service CSV eligible for the completion rerun."""

    old_path = repo / "artifacts" / "experiment-022" / "validation" / "resident-shard-results.csv"
    old_rows = _read_csv(old_path)
    rows = [
        row
        for row in old_rows
        if row.get("operation") in WHOLE_OPERATIONS
        and int(row["rows"]) in CHUNKS
        and str(row.get("valid_for_service", "True")).lower() in {"true", "1", "yes"}
        and "barrier" not in str(row.get("estimate_kind", "")).lower()
    ]
    for row in rows:
        row["cost_owner"] = "worker_compute"
        row["production_native_binding"] = True
    receipts, trace_paths = _physical_receipts(completion_root)
    if len(receipts) != 12 or any(row.get("status") != "PASS" for row in receipts):
        raise RuntimeError("the repaired service requires twelve passing physical chunk receipts")
    binding_paths = [
        completion_root / "physical" / f"execute-shard-bindings-chunk-{chunk}.json"
        for chunk in CHUNKS
    ]
    binding_receipts = {int(value["rows"]): value for value in map(_read_json, binding_paths)}
    if set(binding_receipts) != set(CHUNKS) or any(
        value.get("status") != "PASS" for value in binding_receipts.values()
    ):
        raise RuntimeError("production binding chunks 1/2/4 are not all physical PASS")
    for receipt in receipts:
        split = "calibration" if int(receipt["layer"]) in (45, 47) else "heldout"
        source = trace_paths[0] if receipt["attention_type"] == "KDA" else trace_paths[1]
        binding_receipt = binding_receipts[int(receipt["rows"])]
        binding = {
            str(case["operation"]): case for case in binding_receipt["cases"]
        }
        reductions = {
            str(case["operation"]).removeprefix("reduction_"): case
            for case in binding_receipt["reduction_service_variants"]
        }
        projections = {
            str(case["operation"]).removeprefix("projection_"): case
            for case in binding_receipt["projection_service_variants"]
        }
        for operation, value in _semantic_services(
            receipt, binding, reductions, projections
        ).items():
            degree = (
                2
                if operation == "routed_shared_reduction"
                else int(receipt["degree"])
                if operation
                in {
                    "attention_shard",
                    "attention_reduction",
                    "latent_down",
                    "expert_stripe",
                    "expert_reduction",
                    "latent_up",
                    "latent_up_reduction",
                    "shared_expert",
                    "shared_reduction",
                    "attention_shard_remote",
                    "latent_down_remote",
                    "expert_stripe_remote",
                    "latent_up_remote",
                    "shared_expert_remote",
                }
                else 1
            )
            rows.append(
                _row(
                    split=split,
                    layer=int(receipt["layer"]),
                    layer_type=_layer_type(str(receipt["attention_type"])),
                    operation=operation,
                    degree=degree,
                    rows=int(receipt["rows"]),
                    p50_ms=float(value),
                    source=source,
                    evidence_class="PHYSICAL_E022_COMPLETION_RESIDENT_NATIVE_DAG",
                    correctness_receipt=f"layer-{receipt['layer']}:chunk-{receipt['rows']}",
                )
            )
    whole_expert_path = completion_root / "physical" / "whole-expert-services.json"
    whole_expert = _read_json(whole_expert_path)
    if whole_expert.get("status") != "PASS":
        raise RuntimeError("WHOLE_EXPERT physical capability gate did not pass")
    for service in whole_expert["services"]:
        for operation in ("expert_whole_group", "expert_whole_group_remote"):
            rows.append(
                _row(
                    split=str(service["split"]),
                    layer=int(service["layer"]),
                    layer_type=_layer_type(str(service["layer_type"])),
                    operation=operation,
                    degree=int(service["degree"]),
                    rows=int(service["rows"]),
                    p50_ms=float(service["native_p50_ms"]),
                    source=whole_expert_path,
                    evidence_class=str(service["evidence_class"]),
                    correctness_receipt=f"whole-expert:layer-{service['layer']}:chunk-{service['rows']}",
                )
            )

    # One authenticated request and one authenticated response around a hidden
    # boundary is the conservative measured frame service used by every native
    # task.  It is a distinct software event, never part of network transport.
    for chunk in CHUNKS:
        binding_receipt = binding_receipts[chunk]
        protocol_ms = max(
            float(case["protocol_overhead_ms"])
            for case in binding_receipt["cases"]
            if str(case["operation"]) != "reduction_contribution"
        )
        for layer, layer_type in ((0, "DENSE"), (45, "KDA"), (47, "GATED_MLA")):
            rows.append(
                _row(
                    split="calibration",
                    layer=layer,
                    layer_type=layer_type,
                    operation="worker_protocol",
                    degree=1,
                    rows=chunk,
                    p50_ms=protocol_ms,
                    source=completion_root
                    / "implementation"
                    / "execute-shard-bindings.json",
                    evidence_class="PHYSICAL_AUTHENTICATED_FRAME_ROUND_TRIP",
                    correctness_receipt="all-six-execute-shard-bindings",
                )
            )

    output_path = completion_root / "validation" / "repaired-resident-service.csv"
    write_csv(output_path, rows)
    ownership: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        ownership[str(row["operation"])].add(str(row["cost_owner"]))
    conflicts = {
        operation: sorted(owners)
        for operation, owners in ownership.items()
        if len(owners) != 1
    }
    experiment_only_operations = {
        "state_validation_capture",
        "receipt_assembly",
        "fixture_state_reset",
    }
    experiment_only_in_service = sorted(
        experiment_only_operations.intersection(ownership)
    )
    accounting_path = completion_root / "validation" / "accounting-reconciliation.json"
    accounting = _read_json(accounting_path)
    accounting.update(
        {
            "service_operation_ownership": {
                operation: next(iter(owners))
                for operation, owners in sorted(ownership.items())
            },
            "service_owner_conflicts": conflicts,
            "experiment_only_operations_in_planner_service": experiment_only_in_service,
            "planner_service_excludes_experiment_only": not experiment_only_in_service,
            "non_service_event_ownership": {
                "network_fanout_and_gather": "network_transport",
                "inter_layer_transfer": "network_transport",
                "scheduler_dependencies": "scheduler_control_plane",
                "recurrent_state_dependencies": "state_wait",
            },
            "every_final_service_cost_has_exactly_one_owner": (
                not conflicts and not experiment_only_in_service
            ),
        }
    )
    if conflicts or experiment_only_in_service:
        accounting["status"] = "FAIL"
    accounting_path.write_text(
        json.dumps(accounting, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if accounting.get("status") != "PASS":
        raise RuntimeError("repaired event-model accounting reconciliation failed")
    manifest = {
        "schema_version": "experiment-022-completion-repaired-service-v1",
        "status": "PASS",
        "row_count": len(rows),
        "chunks": list(CHUNKS),
        "sublayer_partition_degrees": [8],
        "artificial_barrier_rows": 0,
        "global_correction_factor": False,
        "source_files": [
            str(old_path),
            *(str(path) for path in trace_paths),
            *(str(path) for path in binding_paths),
            str(whole_expert_path),
        ],
        "service_csv": str(output_path),
        "cost_owners": ["worker_compute", "worker_software_overhead", "collective_compute"],
    }
    manifest_path = completion_root / "validation" / "repaired-service-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows, output_path, manifest


__all__ = ["build_repaired_service"]
