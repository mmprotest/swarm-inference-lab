"""Acceptance evaluation over measured or emulated experiment rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

from swarm_inference.config.models import ExperimentConfig


@dataclass(frozen=True, slots=True)
class CriterionResult:
    name: str
    status: str
    observed: Any
    required: Any
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def criterion(
    name: str,
    passed: bool,
    *,
    observed: Any,
    required: Any,
    reason: str,
) -> CriterionResult:
    return CriterionResult(
        name=name,
        status="PASS" if passed else "FAIL",
        observed=observed,
        required=required,
        reason=reason,
    )


def evaluate_experiment(
    *,
    config: ExperimentConfig,
    summaries: list[dict[str, Any]],
    scaling_rows: list[dict[str, Any]],
) -> list[CriterionResult]:
    acceptance = config.acceptance
    primary = max(
        summaries,
        key=lambda row: (
            int(row["concurrent_request_count"]),
            int(row["node_count"]),
        ),
    )
    results = [
        criterion(
            "completed_verified_requests",
            int(primary["completed_verified_requests"]) > 0,
            observed=primary["completed_verified_requests"],
            required="> 0",
            reason="Only successfully completed, verified requests count.",
        ),
        criterion(
            "throughput_milestone",
            (
                float(primary["aggregate_verified_output_tokens_s"])
                >= acceptance.minimum_aggregate_verified_tokens_s
                and float(primary["simulated_duration_s"]) >= acceptance.minimum_duration_s
                and int(primary["node_count"]) > 1
            ),
            observed={
                "tokens_s": primary["aggregate_verified_output_tokens_s"],
                "duration_s": primary["simulated_duration_s"],
                "workers": primary["node_count"],
            },
            required={
                "tokens_s": f">= {acceptance.minimum_aggregate_verified_tokens_s}",
                "duration_s": f">= {acceptance.minimum_duration_s}",
                "workers": "> 1",
            },
            reason="The milestone must be sustained after warm-up and use multiple workers.",
        ),
        criterion(
            "steady_state_stage_utilisation",
            float(primary["minimum_stage_utilisation"]) >= acceptance.minimum_stage_utilisation,
            observed=primary["minimum_stage_utilisation"],
            required=f">= {acceptance.minimum_stage_utilisation}",
            reason="Uses the least-utilised stage, not the mean.",
        ),
        criterion(
            "capacity_balance",
            float(primary["capacity_imbalance"]) <= acceptance.maximum_capacity_imbalance,
            observed=primary["capacity_imbalance"],
            required=f"<= {acceptance.maximum_capacity_imbalance}",
            reason="Maximum service-capacity gap relative to the largest stage.",
        ),
        criterion(
            "accepted_request_completion",
            float(primary["completion_fraction"]) >= acceptance.minimum_completion_fraction,
            observed=primary["completion_fraction"],
            required=f">= {acceptance.minimum_completion_fraction}",
            reason="Completion requires successful verification.",
        ),
    ]
    eligible_scaling = [row for row in scaling_rows if int(row["concurrent_requests"]) >= 16]
    by_concurrency: dict[int, list[dict[str, Any]]] = {}
    for row in eligible_scaling:
        by_concurrency.setdefault(int(row["concurrent_requests"]), []).append(row)
    doubling_gains: list[float] = []
    marginal_values: list[float] = []
    for rows in by_concurrency.values():
        ordered = sorted(rows, key=lambda row: int(row["node_count"]))
        lookup = {int(row["node_count"]): row for row in ordered}
        for node_count, row in lookup.items():
            doubled = lookup.get(node_count * 2)
            if doubled is not None and float(row["throughput"]) > 0:
                doubling_gains.append(float(doubled["throughput"]) / float(row["throughput"]))
        marginal_values.extend(float(row["marginal_throughput"]) for row in ordered[1:])
    consecutive_doublings_pass = len(doubling_gains) >= 2 and any(
        first >= acceptance.minimum_doubling_gain and second >= acceptance.minimum_doubling_gain
        for first, second in pairwise(doubling_gains)
    )
    results.extend(
        [
            criterion(
                "two_consecutive_capacity_doublings",
                consecutive_doublings_pass,
                observed=doubling_gains,
                required=(
                    f"two consecutive gains >= {acceptance.minimum_doubling_gain} "
                    "at concurrency >= 16"
                ),
                reason="Aggregate throughput is compared only at matching concurrency.",
            ),
            criterion(
                "nonnegative_useful_node_marginal_throughput",
                bool(marginal_values) and all(value >= -1e-12 for value in marginal_values),
                observed=marginal_values,
                required="all >= 0",
                reason="A negative observed step fails even if a later step recovers.",
            ),
        ]
    )
    if config.faults.churn_rate_per_hour > 0:
        active_failures = int(primary.get("failures_during_active_requests", 0))
        results.append(
            criterion(
                "active_churn_exposure",
                active_failures > 0,
                observed=active_failures,
                required="> 0 worker failures while requests are active",
                reason="Post-workload failures are not resilience evidence.",
            )
        )
    return results


def project_acceptance_status(
    *,
    config: ExperimentConfig,
    experiment_criteria: list[CriterionResult],
) -> list[CriterionResult]:
    """Report every project-level gate without treating unrun work as passing."""

    experiment_by_name = {item.name: item for item in experiment_criteria}
    simulation = config.execution_mode.value == "simulation"
    not_real_reason = (
        "Not demonstrated by this simulation run; physical or real-model evidence is required."
    )
    missing_criterion = CriterionResult(
        name="missing",
        status="FAIL",
        observed=None,
        required=None,
        reason="criterion was not evaluated",
    )
    doubling = experiment_by_name.get("two_consecutive_capacity_doublings", missing_criterion)
    nonnegative_marginal = experiment_by_name.get(
        "nonnegative_useful_node_marginal_throughput", missing_criterion
    )
    stage_utilisation = experiment_by_name.get("steady_state_stage_utilisation", missing_criterion)
    capacity_balance = experiment_by_name.get("capacity_balance", missing_criterion)
    throughput_milestone = experiment_by_name.get("throughput_milestone", missing_criterion)
    request_completion = experiment_by_name.get("accepted_request_completion", missing_criterion)
    churn_exposure = experiment_by_name.get("active_churn_exposure", missing_criterion)
    results = [
        criterion(
            "capacity:model_larger_than_each_worker",
            config.model.layer_count * config.model.bytes_per_layer
            > max(node.memory_bytes for node in config.nodes),
            observed={
                "model_bytes": config.model.layer_count * config.model.bytes_per_layer,
                "largest_worker_bytes": max(node.memory_bytes for node in config.nodes),
            },
            required="model_bytes > every worker memory cap",
            reason="Synthetic capacity evidence only."
            if simulation
            else "Measured capacity evidence.",
        ),
        criterion(
            "capacity:no_worker_loads_full_model",
            config.model.stage_count > 1
            and max(
                (
                    (end - start) * config.model.bytes_per_layer
                    for start, end in _layer_ranges(
                        config.model.layer_count, config.model.stage_count
                    )
                ),
                default=config.model.layer_count * config.model.bytes_per_layer,
            )
            < config.model.layer_count * config.model.bytes_per_layer,
            observed={"stage_count": config.model.stage_count},
            required="each assigned shard < full model",
            reason="Logical memory accounting in this execution mode.",
        ),
        criterion(
            "capacity:distributed_real_model_valid_output",
            False,
            observed="not run" if simulation else "not independently validated",
            required="valid Qwen3 output from at least two workers",
            reason=not_real_reason,
        ),
        criterion(
            "correctness:greedy_token_identity",
            False,
            observed="not run",
            required="distributed token IDs exactly equal reference",
            reason=not_real_reason,
        ),
        criterion(
            "correctness:stage_tolerance",
            False,
            observed="not run",
            required="all intermediate outputs within configured tolerance",
            reason=not_real_reason,
        ),
        criterion(
            "correctness:cache_replay_preserves_output",
            False,
            observed="replay timing only" if simulation else "not run",
            required="subsequent output unchanged after replay",
            reason="A timing simulation is not numerical correctness evidence.",
        ),
        criterion(
            "correctness:corrupt_workers_detected",
            False,
            observed="run may contain probabilistic audit events",
            required="deterministic corrupt-worker integration test passes",
            reason="Project acceptance is asserted by the dedicated executable test.",
        ),
        criterion(
            "correctness:invalid_shard_hash_rejected",
            False,
            observed="not part of this run",
            required="dedicated shard-integrity test passes",
            reason="Run artifacts do not replace the correctness test.",
        ),
        criterion(
            "scaling:two_consecutive_doublings",
            doubling.status == "PASS",
            observed=doubling.observed,
            required="two consecutive >= 1.6x gains",
            reason="Simulation evidence remains labelled simulation.",
        ),
        criterion(
            "scaling:adding_eligible_worker_never_reduces_throughput",
            nonnegative_marginal.status == "PASS",
            observed=nonnegative_marginal.observed,
            required="all observed marginals >= 0",
            reason="Compared within this run's execution mode only.",
        ),
        criterion(
            "scaling:stage_utilisation",
            stage_utilisation.status == "PASS",
            observed=stage_utilisation.observed,
            required=">= 70%",
            reason="Least-utilised stage during the simulated run.",
        ),
        criterion(
            "scaling:balanced_stage_capacity",
            capacity_balance.status == "PASS",
            observed=capacity_balance.observed,
            required="<= 20% gap",
            reason="Configured/measured service-capacity balance.",
        ),
        criterion(
            "throughput:20_verified_tokens_s_for_5_minutes",
            throughput_milestone.status == "PASS",
            observed=throughput_milestone.observed,
            required=">=20 tokens/s, >1 worker, >=300s",
            reason="Execution-mode label is preserved.",
        ),
        criterion(
            "resilience:99_percent_completion_at_10_percent_hourly_churn",
            (
                abs(config.faults.churn_rate_per_hour - 0.10) < 1e-12
                and request_completion.status == "PASS"
                and churn_exposure.status == "PASS"
            ),
            observed={
                "churn_rate": config.faults.churn_rate_per_hour,
                "completion": request_completion.observed,
                "failures_during_active_requests": churn_exposure.observed,
            },
            required="churn=0.10/hour, active failures>0, and completion>=0.99",
            reason="Requires the specified churn arm, not extrapolation.",
        ),
        criterion(
            "resilience:throughput_degradation_below_20_percent",
            False,
            observed="no paired no-churn baseline in this run",
            required="<20% degradation versus matched baseline",
            reason="A paired baseline comparison is required.",
        ),
        criterion(
            "resilience:failed_stage_reconstructed",
            False,
            observed="timing events only",
            required="exact replay integration output remains correct",
            reason="Numerical replay must be demonstrated outside simulation.",
        ),
        criterion(
            "resilience:corrupt_workers_quarantined",
            False,
            observed="probabilistic simulation is not dedicated test evidence",
            required="dedicated corruption experiment and audit evidence",
            reason="No claim is made from an arm that may sample no corrupt operation.",
        ),
    ]
    return results


def _layer_ranges(layer_count: int, stage_count: int) -> list[tuple[int, int]]:
    base, remainder = divmod(layer_count, stage_count)
    cursor = 0
    result = []
    for index in range(stage_count):
        width = base + (1 if index < remainder else 0)
        result.append((cursor, cursor + width))
        cursor += width
    return result
