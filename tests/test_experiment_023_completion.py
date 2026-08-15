from __future__ import annotations

import csv
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarm_inference.experiments.experiment_022.models import (
    Inventory,
    LayerAssignment,
    LayerSpec,
    LayerType,
    NetworkPeer,
    NodeCapability,
    PartitionKind,
    PlacementPlan,
    PlannerLevel,
)
from swarm_inference.experiments.experiment_023 import baseline
from swarm_inference.experiments.experiment_023.analysis import evaluate_verdict
from swarm_inference.experiments.experiment_023.freeze import (
    FROZEN_CONSTANTS,
    HEADLINE_INVENTORIES,
    validate_e023_freeze,
)
from swarm_inference.experiments.experiment_023.hedging import (
    HEDGE_SEEDS,
    HedgeDeadlineInputs,
    common_random_draw_indices,
    decide_hedge_launch,
)
from swarm_inference.experiments.experiment_023.serving_engine import _eligible_passes

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "artifacts/experiment-023"
ATTEMPT = ARTIFACT / "attempts/deterministic-run-1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _verdict_rows(
    efficiency: list[float],
    *,
    throughput: float = 1.0,
    legacy: float = 12.0,
    replica: bool = True,
) -> list[dict[str, object]]:
    families = (
        ["memory-fragmented"] * 3
        + ["compute-heterogeneous"] * 6
        + ["network-heterogeneous"] * 6
        + ["full-mixed"] * 3
    )
    assert len(efficiency) == len(families) == 18
    return [
        {
            "inventory_id": f"fixture-{index:02d}",
            "family": family,
            "efficiency_uplift_percent": value,
            "throughput_uplift_percent": throughput,
            "legacy_efficiency_uplift_percent": legacy,
            "flex_pool_actual_replica_used": replica,
        }
        for index, (family, value) in enumerate(zip(families, efficiency, strict=True))
    ]


@pytest.mark.parametrize(
    ("rows", "failures", "expected"),
    (
        (_verdict_rows([25.0] * 18), ("fixture-invalid",), "MODEL_INVALID"),
        (_verdict_rows([25.0] * 18), (), "YES_GENERAL_WEDGE"),
        (
            _verdict_rows([25.0] * 3 + [0.0] * 15, legacy=6.0),
            (),
            "YES_CONDITIONAL_WEDGE",
        ),
        (
            _verdict_rows([25.0] + [0.0] * 17, legacy=0.0),
            (),
            "CAPABILITY_SIGNAL_ONLY",
        ),
        (_verdict_rows([0.0] * 18, legacy=0.0), (), "NO_WEDGE"),
    ),
)
def test_verdict_fixtures_cover_every_exact_category(
    rows: list[dict[str, object]], failures: tuple[str, ...], expected: str
) -> None:
    assert evaluate_verdict(rows, validity_failures=failures)["final_verdict"] == expected


def test_actual_verdict_is_mechanical_model_invalid() -> None:
    summary = json.loads((ARTIFACT / "summary.json").read_text(encoding="utf-8"))
    truth = json.loads((ARTIFACT / "truth-table.json").read_text(encoding="utf-8"))
    assert summary["final_verdict"] == truth["Final verdict"] == "MODEL_INVALID"
    failures = truth["Mandatory control gate"]["failures"]
    assert any(
        row["inventory_id"] == "coarse-friendly-03"
        and row["arm"] == "FLEX_POOL"
        and row["efficiency_uplift_percent"] < -5.0
        and row["throughput_uplift_percent"] < -5.0
        for row in failures
    )


def test_frozen_inputs_revalidate_without_changing_e022() -> None:
    validated = validate_e023_freeze(REPO)
    assert validated["status"] == "PASS"
    assert validated["e022_verdict"] == "MODEL_INVALID"
    assert validated["inventory_count"] == 27


def test_complete_arm_schema_concurrency_and_target_row_accounting() -> None:
    rows = _read_csv(ARTIFACT / "serving/arm-results.csv")
    required = {
        "inventory_id",
        "family",
        "cohort",
        "arm",
        "network_mode",
        "concurrency",
        "status",
        "target_passes_measured",
        "target_rows_measured",
        "measurement_window_ms",
        "target_rows_per_second",
        "p50_pass_latency_ms",
        "p95_pass_latency_ms",
        "abstract_node_cost",
        "rows_per_second_per_abstract_cost",
        "network_bytes",
        "network_bytes_per_target_row",
        "worker_compute_ms",
        "worker_compute_ms_per_target_row",
        "nodes_used",
        "replica_nodes",
        "replica_count",
        "replica_checkpoint_bytes",
        "replica_resident_bytes",
        "new_nodes_activated",
        "maximum_compute_utilization",
        "maximum_tx_utilization",
        "maximum_rx_utilization",
        "plan_sha256",
    }
    assert len(rows) == 945
    assert required.issubset(rows[0])
    assert not any(not key or key.startswith("Unnamed") for key in rows[0])
    assert all(row["status"] == "PASS" for row in rows)
    assert all(
        int(row["target_rows_measured"]) == 17 * int(row["target_passes_measured"])
        for row in rows
    )
    shared = [row for row in rows if row["network_mode"] == "SHARED_NIC"]
    for inventory_id in HEADLINE_INVENTORIES:
        for arm in (
            "U_STRONG",
            "FLEX_FREE_NO_ALT",
            "FLEX_FREE",
            "FLEX_POOL_NO_ALT",
            "FLEX_POOL",
        ):
            assert {
                int(row["concurrency"])
                for row in shared
                if row["inventory_id"] == inventory_id and row["arm"] == arm
            } == {1, 8, 32, 64, 128}


def test_warmup_passes_are_excluded_by_global_t0() -> None:
    passes = [
        SimpleNamespace(slot_id=0, pass_index=0, start_ms=0.0, finish_ms=4.0),
        SimpleNamespace(slot_id=1, pass_index=0, start_ms=0.0, finish_ms=5.0),
        SimpleNamespace(slot_id=0, pass_index=1, start_ms=4.0, finish_ms=10.0),
        SimpleNamespace(slot_id=1, pass_index=1, start_ms=5.0, finish_ms=12.0),
        SimpleNamespace(slot_id=0, pass_index=2, start_ms=10.0, finish_ms=14.0),
        SimpleNamespace(slot_id=1, pass_index=2, start_ms=12.0, finish_ms=16.0),
        SimpleNamespace(slot_id=0, pass_index=3, start_ms=14.0, finish_ms=18.0),
    ]
    t0, eligible = _eligible_passes(passes)
    assert t0 == 12.0
    assert [(row.slot_id, row.pass_index) for row in eligible] == [(1, 2), (0, 3)]


def test_latency_budget_uses_only_u_strong_c1_in_each_network_mode() -> None:
    arm_rows = _read_csv(ARTIFACT / "serving/arm-results.csv")
    saturation = _read_csv(ARTIFACT / "serving/saturation-summary.csv")
    references = {
        (row["inventory_id"], row["network_mode"]): float(row["p95_pass_latency_ms"])
        for row in arm_rows
        if row["arm"] == "U_STRONG" and int(row["concurrency"]) == 1
    }
    for row in saturation:
        reference = references[(row["inventory_id"], row["network_mode"])]
        assert float(row["u_strong_c1_p95_pass_latency_ms"]) == reference
        assert float(row["slo_2.0x_latency_budget_ms"]) == 2.0 * reference


def test_future_stochastic_service_is_not_visible_to_hedge_policy() -> None:
    signature = inspect.signature(decide_hedge_launch)
    assert "realized_service_ms" not in signature.parameters
    prediction = HedgeDeadlineInputs(10.0, 2.0, 5.0, 3.0)
    assert prediction.deadline_ms == 20.0
    assert decide_hedge_launch(
        prediction, observed_contribution_arrival_ms=19.0
    ).launch is False
    assert decide_hedge_launch(
        prediction, observed_contribution_arrival_ms=21.0
    ).launch is True
    assert common_random_draw_indices(
        seed=HEDGE_SEEDS[0], task_count=16, sample_count=200
    ) == common_random_draw_indices(
        seed=HEDGE_SEEDS[0], task_count=16, sample_count=200
    )


def _relocation_node(node_id: str, peers: tuple[str, ...]) -> NodeCapability:
    return NodeCapability(
        node_id=node_id,
        accelerator_memory_bytes=100_000,
        system_memory_bytes=200_000,
        compute_profile={"reference": 1.0},
        memory_bandwidth_profile={"reference": 1.0},
        supported_precisions=("MXFP4",),
        network_peers={
            peer: NetworkPeer(peer, 1.0, 10.0) for peer in peers if peer != node_id
        },
        reliability=1.0,
        cost=1.0,
        cached_shards=(),
        runtime_capabilities=("WHOLE_LAYER",),
        locality_group="fixture",
    )


@pytest.mark.parametrize(
    "inventory_id",
    ("full-mixed-01", "full-mixed-02", "full-mixed-03", "full-mixed-05"),
)
def test_relocation_discovers_injected_useful_node_for_known_e022_cases(
    monkeypatch: pytest.MonkeyPatch, inventory_id: str
) -> None:
    source = f"source-{inventory_id}"
    joined = f"useful-joined-{inventory_id}"
    inventory = Inventory(
        inventory_id=inventory_id,
        family="full-mixed",
        seed=1,
        nodes=(
            _relocation_node(source, (source, joined)),
            _relocation_node(joined, (source, joined)),
        ),
        evidence_class="SYNTHETIC",
        generator_version="test",
        scenario="useful node injected",
    )
    layers = [
        LayerSpec(
            layer_id=index,
            layer_type=LayerType.KDA,
            checkpoint_bytes=100,
            resident_bytes=200,
            component_bytes={"routed_expert": 80, "other": 20},
            tensor_count=1,
            attnres_snapshot=False,
        )
        for index in range(2)
    ]
    plan = PlacementPlan(
        inventory_id=inventory_id,
        planner_level=PlannerLevel.A,
        chunk_rows=1,
        assignments=[
            LayerAssignment(
                layer_id=0,
                partition_kind=PartitionKind.EXPERT_SHARD,
                degree=1,
                node_ids=(source,),
                memory_by_node={source: 200},
                checkpoint_bytes_by_node={source: 100},
                coordinator_node_id=source,
                candidate_id="layer-00:EXPERT_SHARD:p1",
            ),
            LayerAssignment(
                layer_id=1,
                partition_kind=PartitionKind.WHOLE_LAYER,
                degree=1,
                node_ids=(source,),
                memory_by_node={source: 200},
                checkpoint_bytes_by_node={source: 100},
                coordinator_node_id=source,
                candidate_id="layer-01:WHOLE_LAYER:p1",
            ),
        ],
        endpoint_memory_by_node={},
        endpoint_checkpoint_bytes_by_node={},
        feasible=True,
    )

    class FakeEvaluator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def evaluate(self, candidate: PlacementPlan) -> None:
            candidate.exact_tok_s_per_user = (
                2.0 if candidate.assignments[1].node_ids == (joined,) else 1.0
            )

    monkeypatch.setattr(baseline, "PlacementEvaluator", FakeEvaluator)
    repaired, actions = baseline.repair_whole_layer_relocations(
        SimpleNamespace(layers=layers),
        inventory,
        SimpleNamespace(),
        SimpleNamespace(),
        plan,
        starting_candidate="INJECTED_USEFUL_FIXTURE",
    )
    assert repaired.assignments[1].node_ids == (joined,)
    assert len(actions) == 1
    assert actions[0]["destination_node_id"] == joined
    assert actions[0]["relative_gain_percent"] == 100.0


def test_actual_planner_limits_and_no_forced_control_actions() -> None:
    rows = _read_csv(ARTIFACT / "serving/replica-actions.csv")
    accepted: dict[tuple[str, str], int] = {}
    for row in rows:
        if row["accepted"] == "True":
            key = (row["inventory_id"], row["arm"])
            accepted[key] = accepted.get(key, 0) + 1
            assert float(row["relative_gain_percent"]) >= 0.5
            assert float(row["objective_after"]) > float(row["objective_before"])
    assert all(value <= int(FROZEN_CONSTANTS["planner_max_accepted_layer_actions"]) for value in accepted.values())
    assert accepted.get(("coarse-friendly-02", "FLEX_POOL"), 0) == 0


def test_physical_sample_and_memory_gates_are_preserved() -> None:
    receipt = json.loads(
        (ARTIFACT / "physical/duplicate-expert-group-correctness.json").read_text(
            encoding="utf-8"
        )
    )
    samples = _read_csv(ARTIFACT / "physical/service-samples.csv")
    assert receipt["status"] == receipt["primary_gate"] == "PASS"
    assert len(receipt["cases"]) == 12
    assert len(samples) == 2400
    assert max(row["a_b_relative_l2"] for row in receipt["cases"]) <= 2e-6
    assert max(row["memory_error_percent"] for row in receipt["copy_identity"]) <= 5.0
    assert receipt["hedge_service_drift"] is True
    assert receipt["hedging_conclusion_suppressed"] is True


def test_required_report_sections_are_present_in_exact_order() -> None:
    report = (REPO / "docs/experiments/EXPERIMENT_023_REPORT.md").read_text(
        encoding="utf-8"
    )
    headings = re.findall(r"^## .+$", report, flags=re.MULTILINE)
    assert headings == [
        "## Verdict",
        "## Executive Summary",
        "## Frozen Hypothesis",
        "## Why This Experiment Exists",
        "## Evidence Boundary",
        "## Physical Replica Validation",
        "## Strong Unique Baseline",
        "## Serving Model",
        "## Sparse Flexibility Mechanism",
        "## Primary Results",
        "## Family Results",
        "## Optionality Ablation",
        "## Zero-New-Node Result",
        "## Capacity Cohort",
        "## Legacy Network Robustness",
        "## Hedging Diagnostic",
        "## Correctness",
        "## Memory and Cost Accounting",
        "## Limitations",
        "## What E023 Proves",
        "## What E023 Does Not Prove",
        "## Recommendation for Experiment 024",
        "## Reproduction",
    ]
    first_verdict = report.split("## Verdict", 1)[1].split("\n\n", 2)[1]
    assert "mechanically determined primary category is **MODEL_INVALID**" in first_verdict


def test_required_tree_and_explicit_not_run_receipts_exist() -> None:
    for index, name in enumerate(
        (
            "efficiency-uplift",
            "throughput-uplift",
            "throughput-vs-cost",
            "replica-memory-vs-uplift",
            "family-summary",
            "saturation-curves",
            "replica-selection",
            "hedging-tail-tradeoff",
        ),
        1,
    ):
        path = ARTIFACT / "charts" / f"chart-{index:02d}-{name}.png"
        assert path.is_file() and path.stat().st_size > 0
    assert len(list((ARTIFACT / "plans").glob("*/*.json"))) == 135
    for inventory_id in (
        "memory-fragmented-03",
        "compute-heterogeneous-04",
        "network-heterogeneous-01",
        "full-mixed-02",
        "coarse-friendly-01",
    ):
        receipt = json.loads(
            (ARTIFACT / "correctness" / f"{inventory_id}.json").read_text(
                encoding="utf-8"
            )
        )
        assert receipt["status"] == "NOT_RUN_AFTER_MANDATORY_CONTROL_FAILURE"
        assert receipt["complete_93_layer_traversal_executed"] is False
