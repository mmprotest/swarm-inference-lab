"""Zero-spend full-graph and four-fragment Kimi K3 rehearsal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_014.full_cuda import (
    benchmark_streamed_cuda_graph,
)
from swarm_inference.experiments.experiment_014.sub_layer_microwork import (
    benchmark_real_expert_microwork,
)

from .io import atomic_write_json, read_json, sha256_file, utc_now


def run_local_rehearsal(
    *,
    checkpoint: Path,
    cuda_library: Path,
    oracle_root: Path,
    graph_certification: Path,
    placement_path: Path,
    test_receipt_path: Path,
    native_dispatch_path: Path,
    full_graph_output: Path,
    sub_layer_output: Path,
    output_path: Path,
) -> dict[str, Any]:
    full = read_json(full_graph_output) if full_graph_output.is_file() else None
    if full is None or full.get("status") != "PASS":
        full = benchmark_streamed_cuda_graph(
            checkpoint,
            cuda_library,
            oracle_root / "hidden-trace.f32",
            oracle_root / "routes.txt",
            full_graph_output,
            oracle_logits_path=oracle_root / "prefill-logits.f32",
            layer_limit=93,
            prompt_token_ids=(163584, 18699),
            decode_token_id=11,
            device=0,
            relative_error_gate=3e-3,
            cycle_id="E025-LOCAL-FULL-93",
            oracle_layer_count=93,
        )
    sub_layer = read_json(sub_layer_output) if sub_layer_output.is_file() else None
    if sub_layer is None or sub_layer.get("status") != "PASS":
        sub_layer = benchmark_real_expert_microwork(
            checkpoint,
            cuda_library,
            oracle_root / "hidden-trace.f32",
            oracle_root / "routes.txt",
            graph_certification,
            sub_layer_output,
            layer=89,
            workers=4,
            device=0,
            warmup=7,
            iterations=20,
            cycle_id="E025-LOCAL-LAYER89-FOUR-WORKER",
        )
    placement = read_json(placement_path)
    tests = read_json(test_receipt_path)
    native_dispatch = read_json(native_dispatch_path)
    sub_workers = sub_layer.get("memory", {}).get("workers", [])
    gates = {
        "all_93_real_layers_executed": full.get("status") == "PASS"
        and full.get("fixture", {}).get("layer_limit") == 93,
        "greedy_endpoint_token_exact": full.get("prefill", {}).get(
            "sampled_token_id"
        )
        == 11,
        "stateful_second_token_exact": full.get("correctness", {}).get(
            "stateful_decode_executed"
        )
        is True,
        "all_routes_exact": full.get("correctness", {}).get("routing_equality")
        is True,
        "full_graph_numerical_gate": float(
            full.get("correctness", {}).get("maximum_layer_relative_l2_error", 1.0)
        )
        <= float(full.get("correctness", {}).get("relative_error_gate", 0.0)),
        "four_disjoint_real_expert_partitions": len(sub_workers) == 4
        and sub_layer.get("correctness", {}).get("status") == "PASS",
        "sub_layer_exact_reduction": float(
            sub_layer.get("correctness", {}).get("maximum_relative_l2_error", 1.0)
        )
        <= 1e-7,
        "sub_layer_state_exact": sub_layer.get("correctness", {}).get(
            "state_fingerprint_exact"
        )
        is True,
        "every_fragment_smaller_than_layer": sub_layer.get("memory", {}).get(
            "every_worker_less_than_complete_layer"
        )
        is True,
        "exact_frozen_manifest_coverage": placement.get("status") == "PASS"
        and all(placement.get("acceptance_gates", {}).values()),
        "no_monolithic_layer89_fallback": placement.get("sub_layer_proof", {})
        .get("negative_control", {})
        .get("frozen_placement_without_sub_layer_group_valid")
        is False,
        "current_execute_shard_functional_test": tests.get("status") == "PASS"
        and tests.get("production_dispatch_functional_test") is True,
        "real_native_execute_shard_process_rehearsal": native_dispatch.get("status")
        == "PASS",
    }
    payload = {
        "schema_version": "experiment-025-local-rehearsal-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "evidence_class": "LOCAL_SINGLE_GPU_LOGICAL_REHEARSAL",
        "headline_physical_swarm_claimed": False,
        "source_tree_sha256": tests.get("source_tree_sha256"),
        "checkpoint": str(checkpoint.resolve()),
        "cuda_library": str(cuda_library.resolve()),
        "cuda_library_sha256": sha256_file(cuda_library.resolve()),
        "full_graph": full,
        "sub_layer": sub_layer,
        "native_dispatch": native_dispatch,
        "gates": gates,
        "interpretation": (
            "The exact real Kimi task graph, endpoint, second-token state, and current "
            "four-way layer-89 ownership are locally safe to rent; no distributed "
            "physical claim follows from this rehearsal."
        ),
    }
    atomic_write_json(output_path, payload)
    return payload


__all__ = ["run_local_rehearsal"]
