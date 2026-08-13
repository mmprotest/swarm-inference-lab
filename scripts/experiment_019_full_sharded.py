"""Execute and persist the Experiment 019 full sharded correctness oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from swarm_inference.experiments.experiment_019.sharded_graph import ShardedK3Graph
from swarm_inference.experiments.experiment_018.analysis import parse_oracle_routes
from swarm_inference.experiments.experiment_019.sharded_graph import RELATIVE_L2_GATE


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--kda-shard-library", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--layer-limit", type=int, default=93)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--recertify-existing",
        action="store_true",
        help="Re-evaluate only the stored validation predicates; do not execute compute.",
    )
    return parser.parse_args()


def _recertify_existing(path: Path, routes_path: Path) -> dict[str, object]:
    original_bytes = path.read_bytes()
    receipt = json.loads(original_bytes)
    expected_routes = parse_oracle_routes(routes_path)
    mismatches: list[dict[str, object]] = []
    routes_exact = True
    maximum_error = 0.0
    for layer in receipt["layers"]:
        maximum_error = max(
            maximum_error,
            float(layer["oracle_correctness"]["relative_l2_error"]),
        )
        route = layer.get("routes")
        if route is None:
            continue
        layer_id = int(layer["layer"])
        observed = tuple(int(value) for value in route["selected_expert_ids"])
        expected = expected_routes[layer_id][0]
        equal = observed == expected
        route["ordered_ids_match_oracle"] = equal
        routes_exact = routes_exact and equal
        if not equal:
            mismatches.append(
                {
                    "layer": layer_id,
                    "observed": list(observed),
                    "expected": list(expected),
                }
            )
    head = receipt.get("head")
    if head is not None:
        maximum_error = max(
            maximum_error,
            float(head["oracle_logits"]["relative_l2_error"]),
        )
    direct_gate = receipt["direct_loader_gate"]
    initial_status = receipt["status"]
    receipt["maximum_relative_l2_error"] = maximum_error
    receipt["routes_exact"] = routes_exact
    receipt["status"] = (
        "PASS"
        if bool(receipt["complete_93_layer_graph"])
        and maximum_error <= RELATIVE_L2_GATE
        and routes_exact
        and head is not None
        and bool(head["greedy_token_match"])
        and int(direct_gate["full_material_tensors_over_review_threshold"]) == 0
        else "FAIL"
    )
    receipt["validator_recertification"] = {
        "initial_validator_status": initial_status,
        "reason": "Corrected type-unstable JSON-list versus oracle-tuple route comparison.",
        "physical_compute_reexecuted": False,
        "stored_physical_execution_reused": True,
        "original_receipt_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "route_rows_checked": sum(
            1 for layer in receipt["layers"] if layer.get("routes") is not None
        ),
        "route_mismatch_count": len(mismatches),
        "route_mismatches": mismatches,
        "relative_l2_gate_unchanged": RELATIVE_L2_GATE,
    }
    return receipt


def main() -> int:
    arguments = _arguments()
    if arguments.recertify_existing:
        receipt = _recertify_existing(
            arguments.output, arguments.oracle_root / "routes.txt"
        )
    else:
        graph = ShardedK3Graph(
            arguments.checkpoint,
            arguments.cuda_library,
            arguments.kda_shard_library,
            degree=arguments.degree,
            device=arguments.device,
        )
        try:
            receipt = graph.execute_full_oracle(
                arguments.oracle_root / "hidden-trace.f32",
                arguments.oracle_root / "routes.txt",
                arguments.oracle_root / "prefill-logits.f32",
                layer_limit=arguments.layer_limit,
            )
        finally:
            graph.close()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, arguments.output)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "executed_layers": receipt["executed_layers"],
                "maximum_relative_l2_error": receipt["maximum_relative_l2_error"],
                "routes_exact": receipt["routes_exact"],
                "wall_seconds": receipt["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
