from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from swarm_inference.experiments.experiment_022.inventories import (
    FAMILY_COUNTS,
    materialize_inventory_suite,
)
from swarm_inference.experiments.experiment_022.io import read_json
from swarm_inference.experiments.experiment_022.model_graph import (
    build_model_graph,
    candidate_catalog,
    candidate_memory,
)
from swarm_inference.experiments.experiment_022.models import (
    ALLOWED_BY_LEVEL,
    NetworkPeer,
    NodeCapability,
    PartitionKind,
    PlannerLevel,
)
from swarm_inference.experiments.experiment_022.native_dispatch import (
    CallableResidentPrimitive,
    NativeShardDispatcher,
    ShardRequest,
    ShardTaskType,
    decode_shard_result,
    encode_shard_request,
)

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "artifacts" / "experiment-022"
CHECKPOINT = Path("F:/models/Kimi-K3")


def _model():
    if not CHECKPOINT.is_dir():
        pytest.skip("local Kimi K3 checkpoint is unavailable")
    return build_model_graph(
        CHECKPOINT,
        whole_layer_service_csv=REPO
        / "artifacts"
        / "experiment-018"
        / "physical"
        / "layer-service.csv",
    )


def test_adaptive_action_space_is_cumulative_and_contains_whole() -> None:
    assert ALLOWED_BY_LEVEL[PlannerLevel.A] == frozenset({PartitionKind.WHOLE_LAYER})
    for left, right in zip(
        (PlannerLevel.A, PlannerLevel.B, PlannerLevel.C, PlannerLevel.D),
        (PlannerLevel.B, PlannerLevel.C, PlannerLevel.D, PlannerLevel.E),
        strict=True,
    ):
        assert ALLOWED_BY_LEVEL[left] <= ALLOWED_BY_LEVEL[right]
    assert ALLOWED_BY_LEVEL[PlannerLevel.E] == frozenset(PartitionKind)


def test_node_capability_is_measurement_driven_and_rejects_faster_than_reference() -> None:
    node = NodeCapability(
        node_id="measured-0",
        accelerator_memory_bytes=8 * 1024**3,
        system_memory_bytes=32 * 1024**3,
        compute_profile={"reference": 0.8},
        memory_bandwidth_profile={"reference": 0.8},
        supported_precisions=("int4", "float32"),
        network_peers={
            "measured-1": NetworkPeer("measured-1", 1.0, 10.0, 0.05, "medium")
        },
        reliability=0.99,
        cost=1.0,
        cached_shards=(),
        runtime_capabilities=("WHOLE_LAYER", "EXPERT_SHARD"),
        locality_group="g0",
    )
    assert NodeCapability.from_dict(node.as_dict()) == node
    value = node.as_dict()
    value["compute_profile"] = {"reference": 1.01}
    with pytest.raises(ValueError, match=r"slower-or-equal|must be in"):
        NodeCapability.from_dict(value)


def test_inventory_suite_is_frozen_complete_and_slower_or_equal(tmp_path: Path) -> None:
    model = _model()
    first, inventories = materialize_inventory_suite(tmp_path, model)
    second, repeated = materialize_inventory_suite(tmp_path, model)
    assert first["suite_sha256"] == second["suite_sha256"]
    assert len(inventories) == len(repeated) == sum(FAMILY_COUNTS.values()) == 27
    assert {family: sum(value.family == family for value in inventories) for family in FAMILY_COUNTS} == FAMILY_COUNTS
    assert all(node.compute_multiplier <= 1 for value in inventories for node in value.nodes)
    config = read_json(tmp_path / "inventories" / "generator-config.json")
    assert config["created_before_planner_evaluation"] is True
    assert config["external_resource_queries"] == 0


def test_execute_shard_dispatches_registered_resident_primitive() -> None:
    dispatcher = NativeShardDispatcher("worker-022")
    dispatcher.register(
        "layer-45:kda:p4:s0",
        ShardTaskType.KDA_SHARD,
        CallableResidentPrimitive(
            "test-native-kda",
            lambda values, _request: values * np.float32(2),
            native=True,
        ),
    )
    request = ShardRequest(
        assignment_id="layer-45:kda:p4:s0",
        task_type=ShardTaskType.KDA_SHARD,
        layer=45,
        shard_index=0,
        degree=4,
        rows=1,
        input_shape=(1, 4),
        state_id="state-45",
    )
    values = np.arange(4, dtype=np.float32).reshape(1, 4)
    result, output = decode_shard_result(
        dispatcher.execute_payload(encode_shard_request(request, values))
    )
    np.testing.assert_array_equal(output, values * 2)
    assert result.native_primitive == "test-native-kda"
    assert result.native_invocation_count == 1
    assert dispatcher.audit[0]["whole_layer_fallback"] is False


def test_candidate_memory_reconciles_semantic_split_ownership() -> None:
    layer = _model().layers[45]
    routed = layer.component_bytes["routed_expert"]
    attention = layer.component_bytes["attention_shardable"]
    attention_common = layer.component_bytes["attention_common"]
    attention_replicated = layer.component_bytes["attention_replicated"]
    projection = layer.component_bytes["projection"]
    shared = layer.component_bytes["shared_expert"]
    split_bytes = {
        PartitionKind.EXPERT_SHARD: routed,
        PartitionKind.ATTENTION_PROJECTION_SHARD: attention,
        PartitionKind.FULL_MIXED_STRIPE: routed + attention + projection + shared,
    }
    for kind, split in split_bytes.items():
        resident, checkpoint = candidate_memory(layer, kind, 4)
        assert len(resident) == len(checkpoint) == 4
        assert sum(checkpoint) == layer.checkpoint_bytes
        assert checkpoint[0] >= layer.checkpoint_bytes - split
        assert sum(resident) >= layer.resident_bytes
        if kind is PartitionKind.ATTENTION_PROJECTION_SHARD:
            assert checkpoint[0] >= attention_common + attention_replicated
            assert all(value > 0 for value in resident[1:])


def test_candidate_catalog_exposes_every_required_completion_field() -> None:
    rows = candidate_catalog(_model())["candidates"]
    required = {
        "candidate_type",
        "layer",
        "partition_degree",
        "resident_memory_bytes",
        "worker_compute_dag",
        "state_ownership",
        "collective_dag",
        "network_payload",
    }
    assert rows
    assert all(required <= row.keys() for row in rows)
    assert all(row["candidate_type"] == row["partition_type"] for row in rows)
    assert all(row["partition_degree"] == row["degree"] for row in rows)


def test_required_validation_and_optimizer_artifacts_pass() -> None:
    if not (ARTIFACT / "run-result.json").is_file():
        pytest.skip("full Experiment 022 benchmark has not completed")
    validation = read_json(ARTIFACT / "validation" / "model-validation.json")
    assert validation["status"] == "PASS"
    assert validation["normalization_applied"] is False
    ordered = validation["ordered_validation"]
    assert ordered["median_percent"] <= 5
    assert ordered["p90_percent"] <= 10
    assert ordered["maximum_percent"] <= 15
    oracle_rows = list(
        __import__("csv").DictReader(
            (ARTIFACT / "validation" / "optimizer-small-oracle.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    )
    assert oracle_rows
    assert all(row["status"] == "PASS" for row in oracle_rows)
    assert all(float(row["objective_gap_percent"]) <= 1 for row in oracle_rows)


def test_final_artifact_inventory_is_complete_and_honest() -> None:
    if not (ARTIFACT / "summary.json").is_file():
        pytest.skip("Experiment 022 finalization has not completed")
    suite = read_json(ARTIFACT / "inventories" / "inventory-suite.json")
    assert suite["inventory_count"] == 27
    assert len(suite["inventories"]) == 27
    assert len(list((ARTIFACT / "planner" / "placements").glob("*.json"))) == 27 * 5
    summary = read_json(ARTIFACT / "summary.json")
    environment = read_json(ARTIFACT / "environment.json")
    assert summary["physical_heterogeneous_swarm_tested"] is False
    assert summary["gpu_rentals"] == 0
    assert environment["vast_queries"] == environment["vast_mutations"] == 0
    assert summary["evidence_class"].startswith("PHYSICALLY GROUNDED MODEL")
    assert (REPO / "docs" / "experiments" / "EXPERIMENT_022_REPORT.md").is_file()
    assert len(list((ARTIFACT / "charts").glob("chart-*.png"))) == 12

    required = [
        "summary.json",
        "truth-table.json",
        "environment.json",
        "model-metadata.json",
        "source-manifest.json",
        "commands.txt",
        "test-results.json",
        "failure-log.json",
        "validation/resident-shard-results.csv",
        "validation/ordered-dag-validation.csv",
        "validation/heldout-validation.csv",
        "validation/model-validation.json",
        "validation/optimizer-small-oracle.csv",
        "validation/optimizer-convergence.csv",
        "inventories/generator-config.json",
        "inventories/seeds.json",
        "inventories/inventory-suite.json",
        "planner/whole-layer-results.csv",
        "planner/adaptive-results.csv",
        "planner/ablation-results.csv",
        "planner/candidate-catalog.json",
        "dynamic/join-useful.csv",
        "dynamic/join-harmful.csv",
        "dynamic/slowdown.csv",
        "dynamic/network-degradation.csv",
        "dynamic/node-loss.csv",
        "correctness/primitive-results.json",
        "correctness/full-93-representative.json",
        "control-plane/scaling.csv",
        "control-plane/capability-discovery.json",
        "analysis/throughput-uplift.csv",
        "analysis/capacity-unlocks.csv",
        "analysis/target-crossings.csv",
        "analysis/sublayer-usage.csv",
        "analysis/memory-utilization.csv",
        "analysis/critical-path.csv",
    ]
    missing = [value for value in required if not (ARTIFACT / value).is_file()]
    assert not missing
    for family, count in FAMILY_COUNTS.items():
        family_root = ARTIFACT / "inventories" / family
        assert family_root.is_dir()
        assert len(list(family_root.glob("*.json"))) == count

    manifests = [read_json(path) for path in (ARTIFACT / "planner/placements").glob("*.json")]
    assert len(manifests) == 135
    for manifest in manifests:
        if not manifest["feasible"]:
            continue
        reconciliation = manifest["checkpoint_reconciliation"]
        assert reconciliation["gap_bytes"] == 0
        assert reconciliation["overlap_bytes"] == 0
        transformer_layers = {
            int(piece["piece"].rsplit("_", 1)[-1])
            for node in manifest["nodes"]
            for piece in node["pieces"]
            if piece["piece"].startswith("transformer_layer_")
        }
        assert transformer_layers == set(range(93))
        assert all(
            node["assigned_memory_bytes"] <= node["available_memory_bytes"]
            for node in manifest["nodes"]
        )

    assert len(list(__import__("csv").DictReader((ARTIFACT / "planner/whole-layer-results.csv").open(encoding="utf-8")))) == 27
    assert len(list(__import__("csv").DictReader((ARTIFACT / "planner/adaptive-results.csv").open(encoding="utf-8")))) == 27
    assert len(list(__import__("csv").DictReader((ARTIFACT / "planner/ablation-results.csv").open(encoding="utf-8")))) == 135
    for filename in (
        "join-useful.csv",
        "join-harmful.csv",
        "slowdown.csv",
        "network-degradation.csv",
        "node-loss.csv",
    ):
        rows = list(
            __import__("csv").DictReader(
                (ARTIFACT / "dynamic" / filename).open(encoding="utf-8")
            )
        )
        assert len(rows) == 5
        assert all(row["status"] == "PASS" for row in rows)


def test_inventory_suite_file_hash_matches_preregistered_suite() -> None:
    if not (ARTIFACT / "inventories" / "inventory-suite.json").is_file():
        pytest.skip("inventory suite has not been materialized")
    suite = json.loads(
        (ARTIFACT / "inventories" / "inventory-suite.json").read_text(encoding="utf-8")
    )
    assert suite["suite_sha256"] == "3e949a8eee0a71e128493f64e0be903bd373d3baad4be86a8869d87279d4bb49"
