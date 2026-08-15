# Experiment 023: Exact Sparse Expert Optionality

## Verdict

The mechanically determined primary category is **MODEL_INVALID**. The frozen FLEX_POOL planner accepted three C32-improving actions for the negative control `coarse-friendly-03`, but the final latency-constrained evaluation regressed SLO efficiency by -77.09% and raw SLO throughput by -77.59%. Section 54 makes a regression worse than 5% caused by accepted FLEX actions a planner implementation invalidity, so no E023 performance conclusion can be promoted.

The completed 27-inventory deterministic run is retained below as **diagnostic, non-promotable evidence**. The experiment stopped at the first mandatory gate; deterministic run 2, five full 93-layer correctness receipts, diagnostic copy-choice ablations, and stochastic hedging were not run.

## Executive Summary

- **Primary question:** unanswered because the serving planner failed a mandatory negative-control safety property.
- **Diagnostic headline numbers:** median FLEX_POOL efficiency uplift was +0.00%; 0 of 18 cases reached at least 20%; median raw SLO throughput uplift was +0.00%.
- **Worst headline observations:** efficiency was +0.00% and raw throughput was +0.00%. These values cannot establish a wedge because the model is invalid.
- **Physical prerequisite:** all 12 duplicate-group comparisons passed; maximum A/B relative L2 was 0 and maximum replica-memory error was 3.809%.
- **Next step:** fix only the invalid evidence path and rerun E023 with the exact same frozen hypothesis and thresholds.

## Frozen Hypothesis

The preregistered hypothesis was that a heterogeneous resource pool is more useful when excess memory creates alternative exact execution paths than when every added worker creates another mandatory dependency. The primary gate required at least a 20% serving-efficiency wedge under the frozen 18-inventory cohort and all listed robustness and validity checks. Thresholds were frozen before headline generation and were not changed.

## Why This Experiment Exists

E022 remained frozen as `MODEL_INVALID` and showed that additional nodes could become mandatory dependencies while its old placement search could miss stronger unique placements. E023 therefore constructed `U_STRONG` from the complete frozen A/B/C/D/E envelope, deterministic whole-layer relocation repair, and unique P8 refinement before permitting optional exact expert-group copies.

The architectural inversion being tested was substitutability: a logical expert group still executes exactly once in the deterministic arm, but a non-clairvoyant router may choose either of two exact resident copies.

## Evidence Boundary

- **PHYSICAL:** local RTX 5090 duplicate expert-group exactness, memory residency, and service samples using `F:/models/Kimi-K3`.
- **PHYSICALLY GROUNDED MODEL:** the multi-request 17-row target-pass execution model using frozen repaired E022 services.
- **SHAPED NETWORK:** frozen directed-link timing with E023 shared TX/RX NIC calendars.
- **SYNTHETIC:** verdict fixtures and other explicit test-only cases; none can promote the experiment.

This was not a physical distributed K3 run. Target rows are target-pass work units, not generated user tokens, and abstract node cost is not currency.

## Physical Replica Validation

Two sequentially instantiated resident copies were tested for KDA layer 89 and gated-MLA layer 91, groups 0 and 7, and row counts 1, 2, and 4. All 12 cases had identical ownership, routes, route weights, expert ranges, finite results, zero timed checkpoint reads, zero whole-layer fallback, zero persistent state, and relative L2 within the unchanged `2e-6` gate. Maximum A/B relative L2 was 0; maximum relative L2 against the frozen E022 whole-expert reference was 0.

The standalone memory estimator's maximum physical error was 3.809%, below the frozen 5% gate. The service collection contains 2,400 timed executions. All 12 service cells drifted by more than 10% from E022, with maximum absolute drift 71.93%; this does not invalidate deterministic E023 but suppresses hedging.

Source: `artifacts/experiment-023/physical/duplicate-expert-group-correctness.json` and `service-samples.csv`.

## Strong Unique Baseline

Every inventory considered all five frozen E022 manifests, their deterministic relocation-repaired variants, and a deterministic `U_P8_REFINE` candidate. The final `U_STRONG` applied the frozen 95%-of-fastest envelope and cost-efficiency rule. No randomized reconstruction of E022 placements was used.

Source: `artifacts/experiment-023/baseline/baseline-envelope.csv`, `baseline-relocations.csv`, and `unique-refinement-actions.csv`.

## Serving Model

The new interval-calendar engine modeled one non-preemptive compute resource per worker, directed links, shared source TX NICs, shared destination RX NICs, closed-loop concurrency 1/8/32/64/128, and persistent weights with independent per-slot logical state. All five fixed legacy compatibility cases matched E022 in integer counters and deterministic timing within `1e-9`.

Each 2.0x latency budget was derived only from `U_STRONG` C1 p95 in the corresponding network mode. The 945 required arm rows and 189 saturation summaries are complete and finite.

Source: `artifacts/experiment-023/validation/engine-compatibility.csv` and `serving/arm-results.csv`.

## Sparse Flexibility Mechanism

Only stateless `WHOLE_EXPERT:p8` logical groups could be copied, with at most one alternate. The runtime exhaustively enumerated the `2^k` copy masks using only current calendar reservations and deterministic services, then committed the mask minimizing fork-join completion. Physical arrival never changed the canonical logical reduction order `0..7`.

The figure shows how often final representative plans selected alternates. It demonstrates that the modeled router exercised optional paths; it does not repair the failed control gate.

![Alternate-copy selection by layer and group](../../artifacts/experiment-023/charts/chart-07-replica-selection.png)

## Primary Results

There is **no valid primary performance result**. Diagnostic-run-1 produced a median 18-inventory efficiency uplift of +0.00%, with 0 cases at or above 20%. Median raw SLO throughput uplift was +0.00%; 0 of 18 headline inventories actually selected at least one alternate.

The first chart shows the frozen 20% reference and the second separates raw throughput from cost efficiency. Both are diagnostic because the invalid planner means the experiment cannot decide whether exact optionality creates a reliable wedge.

![Efficiency uplift across headline inventories](../../artifacts/experiment-023/charts/chart-01-efficiency-uplift.png)

![Raw SLO throughput uplift across headline inventories](../../artifacts/experiment-023/charts/chart-02-throughput-uplift.png)

Abstract cost and SLO throughput are shown jointly below. Each used node is counted once, including replica-only nodes; no dollar interpretation is made.

![SLO throughput versus abstract node cost](../../artifacts/experiment-023/charts/chart-03-throughput-vs-cost.png)

## Family Results

No family can qualify while the mandatory validity gate is failed. If the same diagnostic rows were considered without the validity short-circuit, the frozen family predicates would have produced the states recorded in `truth-table.json`; those counterfactual checks are not verdicts.

| Family | n | Median efficiency | ≥20% cases | Median raw throughput | Median legacy efficiency |
|---|---:|---:|---:|---:|---:|
| memory-fragmented | 3 | +0.00% | 0 | +0.00% | +0.00% |
| compute-heterogeneous | 6 | +0.00% | 0 | +0.00% | +0.00% |
| network-heterogeneous | 6 | +0.00% | 0 | +0.00% | +0.00% |
| full-mixed | 3 | +0.00% | 0 | +0.00% | +0.00% |

The plot preserves individual observations around each family median so small cohorts are not hidden by aggregation.

![Family efficiency-uplift distributions](../../artifacts/experiment-023/charts/chart-05-family-summary.png)

## Optionality Ablation

The diagnostic median FLEX_POOL-versus-FLEX_POOL_NO_ALT efficiency difference was +0.00%. Because the planner failed a mandatory control and full correctness/reproducibility were not completed, this cannot establish that optional routing—rather than P8 decomposition, search behavior, or an invalid SLO trade-off—caused a gain.

Replica memory and total architecture uplift are plotted together to expose scale and outliers rather than imply a causal memory-response curve.

![Replica memory versus diagnostic uplift](../../artifacts/experiment-023/charts/chart-04-replica-memory-vs-uplift.png)

Source: `artifacts/experiment-023/analysis/optionality-ablation.csv`.

## Zero-New-Node Result

The diagnostic FLEX_FREE median efficiency uplift was +0.00%. The formal `ZERO_NEW_NODE_WEDGE` flag is **not adjudicated and is reported false** because the experiment is `MODEL_INVALID`; the counterfactual gate state is retained separately in `summary.json` and `truth-table.json`.

## Capacity Cohort

The six exploratory capacity inventories had diagnostic median efficiency uplift +0.00% and median raw SLO throughput uplift +0.00%. 3 of 6 selected an alternate. These cases are excluded from the primary 18-inventory threshold and cannot promote a verdict.

Source: `artifacts/experiment-023/analysis/capacity-exploratory.csv`.

## Legacy Network Robustness

Under `LEGACY_DIRECTED_LINK`, diagnostic median FLEX_POOL efficiency uplift across the headline cohort was +0.00%, with 18 of 18 non-negative cases. The robustness gate is not evaluated for promotion after the mandatory invalidity.

The five-point SHARED_NIC saturation curves below show the actual modeled throughput observations used before SLO filtering for fixed representatives.

![Saturation curves for fixed representatives](../../artifacts/experiment-023/charts/chart-06-saturation-curves.png)

## Hedging Diagnostic

Hedging was not run. E023 stopped before Phase 10, and the physical sample pool also carried `HEDGE_SERVICE_DRIFT` in every service cell. Therefore mean/median throughput, p95/p99 latency, extra compute/network, launch rate, and duplicate-win rate do not exist; `TAIL_DIAGNOSTIC_POSITIVE` is false and no conclusion is claimed.

![Hedging diagnostic status](../../artifacts/experiment-023/charts/chart-08-hedging-tail-tradeoff.png)

## Correctness

The physical duplicate-group correctness prerequisite passed. The five mandatory final 93-layer `FLEX_POOL` correctness traversals were **not run** after the negative-control stop, so E023 has no full-plan correctness conclusion. The five required JSON receipts exist as explicit not-run records and must not be mistaken for passes.

## Memory and Cost Accounting

All 135 final plan manifests passed the completed per-node memory and abstract-cost reconciliation in deterministic run 1. Replica checkpoint bytes are separated from unique model checkpoint bytes; replica-only nodes are charged once; multiple pieces on one node do not multiply its cost. Every manifest preserves zero replica persistent state, at most two copies per logical group, `WHOLE_EXPERT:p8` ownership, and canonical group reduction order.

These accounting passes do not override the planner-control failure.

## Limitations

- The accepted-action objective used C32 throughput (or throughput per abstract cost), while the primary metric applied a latency budget derived from U_STRONG C1. `coarse-friendly-03` exposed a mismatch large enough to invalidate the planner.
- The deterministic full run was not repeated after the stop, so exact rerun reproducibility is unadjudicated.
- Full 93-layer correctness and the two required diagnostic routing ablations are unadjudicated.
- Hedging was not executed and its physical service pool drifted from E022.
- All fleet/network results are modeled or shaped, not a physical multi-machine deployment.

## What E023 Proves

E023 proves that the tested local K3 expert-group copies are numerically substitutable within the frozen gate and that their standalone resident-memory estimator is within 5% on the local RTX 5090. It also proves exact single-pass compatibility for the new legacy engine representatives and records a complete first deterministic concurrency evaluation with reconciled memory and cost.

E023 also identifies a concrete evidence-path defect: local C32 planner acceptance did not protect latency-constrained serving performance on a required negative control.

## What E023 Does Not Prove

E023 does not answer whether exact sparse expert-group optionality creates a real 20% serving-efficiency wedge. It does not prove full-plan correctness, deterministic rerun identity, tail-hedging benefit, real LAN/WAN behavior, real cluster throughput, API token throughput, dollar economics, or physical operation across hundreds of workers.

## Recommendation for Experiment 024

**Fix only the invalid evidence path and rerun E023 with the exact same frozen hypothesis and thresholds.**

This is the required `MODEL_INVALID` branch. Do not broaden E024, increase search budgets, or change thresholds to rescue the result.

## Reproduction

Use the repository's active interpreter `C:\Users\Simon\OneDrive\Documents\Python Scripts\swarm-inference-lab\.venv\Scripts\python.exe` with `PYTHONPATH=src`. The immutable freeze is in `artifacts/experiment-023/freeze/`; deterministic-run-1 remains in `artifacts/experiment-023/attempts/deterministic-run-1/`; published diagnostic tables, manifests, and audits are under `artifacts/experiment-023/`. The recorded deterministic run consumed 10520.510 seconds. A total end-to-end experiment wall-clock time was not recorded.

The executable entry points are `scripts/experiment_023_freeze.py`, `experiment_023_physical.py`, `experiment_023_run.py`, `experiment_023_correctness.py`, `experiment_023_finalize.py`, and `experiment_023_test.py`. `commands.txt` records the phase and validation commands. The machine-readable decision is `truth-table.json`; prose cannot override it.
