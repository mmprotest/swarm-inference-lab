"""Refresh the five Experiment 022 dynamic adaptation scenarios."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_022.dynamic import (  # noqa: E402
    run_dynamic_scenarios,
)
from swarm_inference.experiments.experiment_022.inventories import (  # noqa: E402
    materialize_inventory_suite,
)
from swarm_inference.experiments.experiment_022.io import (  # noqa: E402
    atomic_write_json,
    read_json,
    write_csv,
)
from swarm_inference.experiments.experiment_022.model_graph import (  # noqa: E402
    build_model_graph,
)
from swarm_inference.experiments.experiment_022.models import PlannerLevel  # noqa: E402
from swarm_inference.experiments.experiment_022.planner import (  # noqa: E402
    OptimizerConfiguration,
    SharedPlacementOptimizer,
)
from swarm_inference.experiments.experiment_022.validation import (  # noqa: E402
    materialize_validation,
)


def main() -> int:
    artifact = REPO_ROOT / "artifacts" / "experiment-022"
    model = build_model_graph(
        Path("F:/models/Kimi-K3"),
        whole_layer_service_csv=REPO_ROOT
        / "artifacts/experiment-018/physical/layer-service.csv",
    )
    _suite, inventories = materialize_inventory_suite(artifact, model)
    service, _validation = materialize_validation(REPO_ROOT, artifact, model)
    optimizer = SharedPlacementOptimizer(
        model,
        service,
        OptimizerConfiguration(
            proposal_budget=24,
            exact_evaluation_budget=6,
            restarts=3,
            candidate_groups_per_action=4,
            randomization=0.08,
            seed=22022,
        ),
    )
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for inventory in [
        value for value in inventories if value.family == "full-mixed"
    ][:5]:
        initial = optimizer.optimize(inventory, PlannerLevel.E).plan
        if not initial.feasible:
            continue
        for scenario, row in run_dynamic_scenarios(
            inventory, initial, optimizer
        ).items():
            rows[scenario].append(row)
    names = {
        "JOIN_USEFUL": "join-useful.csv",
        "JOIN_HARMFUL": "join-harmful.csv",
        "SLOWDOWN": "slowdown.csv",
        "NETWORK_DEGRADATION": "network-degradation.csv",
        "NODE_LOSS": "node-loss.csv",
    }
    for scenario, filename in names.items():
        write_csv(artifact / "dynamic" / filename, rows[scenario])
    useful = bool(rows["JOIN_USEFUL"]) and all(
        bool(row["beneficial_join_admitted_and_improved"])
        for row in rows["JOIN_USEFUL"]
    )
    harmful = bool(rows["JOIN_HARMFUL"]) and all(
        bool(row["harmful_join_ignored"]) for row in rows["JOIN_HARMFUL"]
    )
    status = (
        useful
        and harmful
        and all(
            row["status"] == "PASS"
            for scenario_rows in rows.values()
            for row in scenario_rows
        )
    )
    result_path = artifact / "run-result.json"
    result = read_json(result_path)
    result["dynamic_status"] = "PASS" if status else "FAIL"
    result["dynamic_useful_join_non_regression"] = useful
    result["dynamic_harmful_join_ignored"] = harmful
    atomic_write_json(result_path, result)
    print(
        json.dumps(
            {
                "status": "PASS" if status else "FAIL",
                "useful_join_admitted": useful,
                "harmful_join_ignored": harmful,
                "case_count": sum(len(value) for value in rows.values()),
            },
            sort_keys=True,
        )
    )
    return 0 if status else 1


if __name__ == "__main__":
    raise SystemExit(main())
