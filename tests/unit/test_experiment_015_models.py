from __future__ import annotations

import numpy as np
import pytest

from swarm_inference.experiments.experiment_015.dcp import (
    benchmark_dcp_combination,
    combine_attention_partials,
    dcp_partial_payload_bytes,
    full_attention,
    shard_attention,
)
from swarm_inference.experiments.experiment_015.network import (
    MicrocellGeometry,
    NetworkProfile,
    activation_payload_bytes,
)
from swarm_inference.experiments.experiment_015.packing import (
    HardwareClass,
    worker_slots,
)
from swarm_inference.experiments.experiment_015.routing import parse_route_trace
from swarm_inference.experiments.experiment_015.service_model import (
    ArchitectureServiceModel,
    load_service_evidence,
)


def test_network_contract_reproduces_experiment_014_coarse_edge() -> None:
    coarse = NetworkProfile("coarse", rtt_ms=5.0, bandwidth_gbps=10.0)
    assert activation_payload_bytes(1) == 258_048
    assert coarse.service_ms(258_048) == pytest.approx(3.3168264)


def test_microcell_geometry_preserves_all_boundaries() -> None:
    expected = {
        1: (93, 92, 0),
        2: (47, 46, 46),
        4: (24, 23, 69),
        8: (12, 11, 81),
    }
    for depth, counts in expected.items():
        geometry = MicrocellGeometry(93, depth)
        assert (
            geometry.cell_count,
            geometry.coarse_boundaries,
            geometry.internal_boundaries,
        ) == counts
        assert geometry.coarse_boundaries + geometry.internal_boundaries == 92


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_dcp_logsumexp_combination_matches_full_attention(workers: int) -> None:
    rng = np.random.default_rng(15006)
    scores = rng.normal(size=(3, 64, 127)) * 4.0
    values = rng.normal(size=(3, 64, 127, 16))
    reference = full_attention(scores, values)
    score_shards = np.array_split(scores, workers, axis=-1)
    value_shards = np.array_split(values, workers, axis=-2)
    observed = combine_attention_partials(
        [
            shard_attention(score_shard, value_shard)
            for score_shard, value_shard in zip(
                score_shards, value_shards, strict=True
            )
        ]
    )
    np.testing.assert_allclose(observed, reference, rtol=2e-14, atol=2e-14)


def test_dcp_payload_uses_real_kimi_geometry() -> None:
    assert dcp_partial_payload_bytes(query_rows=1) == 33_280
    assert dcp_partial_payload_bytes(query_rows=8) == 266_240


def test_dcp_component_benchmark_is_explicitly_scoped(tmp_path) -> None:
    receipt = benchmark_dcp_combination(tmp_path / "dcp.json")
    assert receipt["status"] == "PASS"
    assert receipt["evidence_class"] is None
    assert receipt["scientific_result"] is False
    assert receipt["complete_kimi_reference_gate"] == "NOT_RUN"
    assert receipt["maximum_relative_l2_error"] <= 1e-12


def test_worker_packing_does_not_infer_compute_capacity() -> None:
    hardware = HardwareClass("generic-24gb", 24.0)
    assert worker_slots(hardware, 3_931_060_224) == 5
    assert hardware.compute_relative is None
    assert hardware.hourly_price_usd is None


def test_service_model_reproduces_baseline_and_fast_domain_limit() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    evidence = load_service_evidence(root)
    model = ArchitectureServiceModel(evidence)
    baseline = model.microcell_latency(1)
    assert baseline["evidence_class"] == "SHAPED"
    assert baseline["dependency_latency_ms"] == pytest.approx(
        evidence.baseline_end_to_end_ms
    )
    depth8 = model.microcell_latency(8)
    assert depth8["coarse_boundaries"] == 11
    assert depth8["dependency_bound_tok_s"] > baseline["dependency_bound_tok_s"]
    assert depth8["dependency_bound_tok_s"] < 5.0


def test_perfect_block7_speculation_is_still_below_interactive_bound() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    model = ArchitectureServiceModel(load_service_evidence(root))
    bound = model.speculation_upper_bound(
        block_size=7,
        accepted_tokens_per_target_pass=8.0,
        cell_depth=8,
    )
    assert bound["dependency_bound_tok_s"] < 5.0
    assert (
        bound["aggregate_tok_s_per_paid_gpu_equivalent_excluding_draft"]
        <= model.evidence.baseline_aggregate_tok_s
        / model.evidence.baseline_paid_gpu_equivalents
    )


def test_retained_route_trace_has_exact_three_token_layer_grain() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    rows = parse_route_trace(root / "artifacts/experiment-014/oracle-full-93/routes.txt")
    assert len(rows) == 92 * 3
    assert {row.layer for row in rows} == set(range(1, 93))
    assert {row.position for row in rows} == {0, 1, 2}
    assert all(len(row.experts) == 16 for row in rows)
