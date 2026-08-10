from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from swarm_inference.experiments.experiment_014.distribution import (
    DistributionError,
    materialize_worker_package,
)
from swarm_inference.experiments.experiment_014.oracle import (
    inspect_route_trace,
    inspect_state_trace,
    inspect_trace,
)
from swarm_inference.experiments.experiment_014.placement import (
    MemoryPolicy,
    PlacementUnit,
    solve_node_counts,
)
from swarm_inference.experiments.experiment_014.support import (
    _SOURCE_MARKERS,
    build_model_support_matrix,
)


def test_oracle_trace_validates_complete_finite_steps(tmp_path: Path) -> None:
    trace = tmp_path / "trace.f32"
    # Two steps, three transformer layers plus the final normalized hidden row.
    values = [float(value) / 10 for value in range(1, 17)]
    trace.write_bytes(struct.pack("<16f", *values))

    result = inspect_trace(trace, hidden_size=2, layer_count=3)

    assert result["forward_steps"] == 2
    assert result["all_finite"] is True
    assert len(result["layers"]) == 3
    assert all(layer["status"] == "PASS" for layer in result["layers"])
    assert len(result["final_hidden"]) == 2


def test_route_and_state_receipts_require_complete_layers(tmp_path: Path) -> None:
    routes = tmp_path / "routes.txt"
    routes.write_text(
        "0 0 1 2:0.6000 4:0.4000\n"
        "1 0 2 1:0.5500 3:0.4500\n"
        "2 0 1 2:0.5100 7:0.4900\n"
        "3 0 2 0:0.7000 6:0.3000\n",
        encoding="utf-8",
    )
    route_result = inspect_route_trace(
        routes,
        layer_count=3,
        first_dense_layer_count=1,
        expert_count=8,
        top_k=2,
    )
    assert route_result["status"] == "PASS"
    assert route_result["complete_route_calls"] == 4

    states = tmp_path / "states.txt"
    states.write_text(
        "0 0 kda 8 0000000000000001 2\n"
        "0 1 mla 8 0000000000000002 2\n"
        "0 2 kda 8 0000000000000003 2\n"
        "1 0 kda 8 0000000000000011 2\n"
        "1 1 mla 16 0000000000000012 4\n"
        "1 2 kda 8 0000000000000013 2\n",
        encoding="ascii",
    )
    state_result = inspect_state_trace(states, layer_count=3, kda_layers={0, 2})
    assert state_result["status"] == "PASS"
    assert state_result["changed_step_transitions"] == 1


def test_support_matrix_stays_fail_closed_without_oracle(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = {
        "text_config": {
            "num_hidden_layers": 2,
            "first_k_dense_replace": 1,
            "linear_attn_config": {"kda_layers": [1], "full_attn_layers": [2]},
        }
    }
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    source = tmp_path / "kimi_k3.c"
    source.write_text("\n".join(_SOURCE_MARKERS.values()), encoding="utf-8")
    output = tmp_path / "support.json"

    receipt = build_model_support_matrix(checkpoint, source, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert receipt["status"] == "FAIL"
    assert payload["summary"]["static_reference_paths"] == 2
    assert payload["summary"]["real_weight_correctness_passed"] == 0
    assert all(row["production_sm86_execution"] == "BLOCKER" for row in payload["layers"])


def test_node_solver_separates_absolute_minimum_from_operational_recommendation() -> None:
    units = [
        PlacementUnit(
            unit_id=f"expert-{index}",
            kind="routed_expert",
            layer=1,
            attention_type="kda",
            routed_expert=index,
            weight_bytes=400,
        )
        for index in range(3)
    ]
    policy = MemoryPolicy(
        physical_vram_bytes=1000,
        cuda_context_bytes=0,
        runtime_library_bytes=0,
        workspace_bytes=0,
        activation_buffer_bytes=0,
        reduction_buffer_bytes=0,
        communication_buffer_bytes=0,
        serving_runtime_bytes=0,
    )

    result = solve_node_counts(units, policy, candidates=(2, 3))

    assert result["absolute_minimum_node_count"] == 2
    assert result["recommended_operational_node_count"] == 3
    assert result["candidates"][0]["headroom_scenarios"]["10pct"]["feasible"] is True


def test_worker_materialization_is_exact_atomic_and_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    shard = source / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"HEAD" + struct.pack("<2f", 1.5, -2.5) + b"TAIL")
    shard_sha256 = hashlib.sha256(shard.read_bytes()).hexdigest()
    placement = tmp_path / "placement.json"
    placement.write_text(
        json.dumps(
            {
                "workers": [
                    {
                        "worker_id": "worker-000",
                        "checkpoint_fingerprint": "fixture",
                        "assignment_units": [
                            {
                                "tensors": [
                                    {
                                        "name": "fixture.weight",
                                        "dtype": "F32",
                                        "shape": [2],
                                        "physical_bytes": 8,
                                        "safetensors_file": shard.name,
                                        "byte_range": [4, 12],
                                    }
                                ]
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    package = tmp_path / "worker.safetensors"

    receipt = materialize_worker_package(
        placement,
        "worker-000",
        source,
        package,
        verify_source_hashes={shard.name: shard_sha256},
    )

    assert receipt["status"] == "PASS"
    assert receipt["tensor_bytes"] == 8
    assert package.is_file()
    assert not package.with_suffix(".safetensors.partial").exists()
    with pytest.raises(DistributionError, match="source shard hash rejected"):
        materialize_worker_package(
            placement,
            "worker-000",
            source,
            tmp_path / "rejected.safetensors",
            verify_source_hashes={shard.name: "0" * 64},
        )
