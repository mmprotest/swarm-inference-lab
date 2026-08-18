# Experiment 023 Repair Protocol

## Status and historical record

This protocol was frozen before any repaired headline result. The original Experiment 023 attempt, `deterministic-run-1`, remains permanently recorded as `MODEL_INVALID`. It is immutable and is not promoted, rewritten, or deleted. Its published record is archived at `artifacts/experiment-023/archive/model-invalid-run-1/`; the archive contains 164 hash-verified files and manifest SHA-256 `97868c59904da914a9bd4e9732d07a7df633d7542fe6c7173bb1515800b5b2b8`.

The repair is still Experiment 023. The authoritative repaired attempt is `deterministic-run-2`; the independent reproducibility attempt is `deterministic-run-3`. The three-control pilot is `repair-control-pilot-v1`.

## Frozen scientific question and hypothesis

The question remains:

> On the same frozen Kimi K3 resource inventories and physically grounded service model repaired in Experiment 022, can selective duplicate residency of stateless `WHOLE_EXPERT:p8` expert groups, combined with non-clairvoyant queue-aware runtime routing, improve latency-constrained steady-state serving throughput per abstract cost unit by at least 20% relative to the strongest unique-residency baseline?

The hypothesis remains that extra heterogeneous resources should create alternative execution paths rather than additional mandatory dependencies. This is the same preregistered claim expressed in the frozen artifact as: “A heterogeneous resource pool is more useful when excess memory creates alternative execution paths than when every additional worker creates another mandatory dependency.”

No scientific threshold, cohort, inventory, baseline, runtime routing rule, shared-NIC rule, physical method, or inference technique changes in this repair.

## Frozen-contract verification

`canonical_sha256(FROZEN_CONSTANTS)` is unchanged from run 1:

`0b5e502e22bb0fca8cbb0dc0d8d0f80b3abd3610c34f2634e399900fc9c45f4b`

All values in `FROZEN_CONSTANTS` remain byte-for-byte and value-for-value unchanged, including seed 23023, 17 target rows, P8, replica counts 2/4/8, concurrency 1/8/32/64/128, C32 planning, two primary candidates, six accepted layer actions, +0.5% gain, 1% throughput regression, 95% economic floor, 2.0x primary latency, the 1.5x/2.0x/4.0 sensitivities, `2e-6` correctness, 5% replica-memory error, the 20% wedge gates, the -5% headline guards, and the frozen hedging thresholds.

The immutable E022 manifest contains 181 files. A read-only verification found 0 missing, size-mismatched, or hash-mismatched inputs. Its canonical input hash is `4887745f5e40bb943faba1ff705e19deaa42bfa04a444a45229de9550d2eec6b`; the inventory-suite hash remains `3e949a8eee0a71e128493f64e0be903bd373d3baad4be86a8869d87279d4bb49`.

The four frozen physical artifacts exactly match their run-1 hashes. The physical prerequisite remains PASS: 12/12 cases, maximum A/B relative L2 0, maximum memory-estimation error 3.8087815298161036%, and 2,400 service samples. The maximum service drift remains 71.93110756234796%, so hedging remains suppressed as `HEDGE_SERVICE_DRIFT`; the physical measurement is not rerun merely to change that outcome.

## Why run 1 was invalid

Run 1 exposed two evidence-path defects and a finalization limitation.

First, the planner accepted FLEX actions using C32 throughput or C32 throughput per abstract cost, while the headline selected the fastest point on the full concurrency ladder that met a latency budget equal to twice U_STRONG C1 p95. `coarse-friendly-03` demonstrated that a C32-improving plan could narrowly miss the C8 hard SLO and fall to C1, producing a legitimate control failure.

Second, FLEX_POOL was not an operational superset of FLEX_FREE. Completion-time-only pruning of two primary groups and one greedy alternate map could hide already-paid, lower-cost FLEX_FREE-feasible solutions after unused fast nodes were admitted.

Third, the finalizer was hard-coded to package the stopped `deterministic-run-1` failure and could not validate a repaired authoritative attempt.

Only those invalid planner, evaluation, correctness-completion, analysis, and finalization paths are repaired.

## Preregistered repair algorithm

### One canonical primary-SLO scorer

One public `score_plan_under_primary_slo` implementation will be shared by action acceptance, branch and envelope comparison, saturation summaries, headline analysis, and negative-control analysis. It will evaluate SHARED_NIC at exactly concurrency 1, 8, 32, 64, and 128; require every run to PASS; set the budget to `2.0 * U_STRONG C1 p95`; retain only p95-eligible points; maximize target rows per second; and, when points are within 0.5% of the maximum, choose the smallest concurrency. It will compute existing abstract node cost and target rows per second per abstract cost.

C32 remains frozen and may order or diagnose candidates, but it cannot accept an action.

### True-objective action gates

FLEX_FREE maximizes primary-SLO raw throughput. FLEX_POOL maximizes primary-SLO throughput per abstract cost. Each accepted action must improve that true objective by at least 0.5%, retain at least 99% of the current plan’s SLO-selected throughput, and retain at least 99% of the original U_STRONG SLO-selected throughput. A candidate with no eligible primary-SLO concurrency is rejected as `NO_PRIMARY_SLO_ELIGIBLE_CONCURRENCY`.

Qualifying actions are selected deterministically by larger SLO objective gain, fewer added replica resident bytes, fewer newly activated nodes, lower layer ID, fewer replicas, lexicographically smaller primary tuple, then lexicographically smaller alternate assignment. C32 screening and complete SLO results are both preserved in `replica-actions.csv`, with unambiguous names.

### Primary and alternate search coverage

FLEX_FREE remains restricted to U_STRONG-used nodes and retains at most two primary P8 groups ranked by predicted completion.

FLEX_POOL first constructs the raw feasible primary set, then retains at most two unique endpoints: `FASTEST` and `LOWEST_RESULTING_COST`. Resulting cost is measured after replacing the layer with the primary P8 layout and before alternates.

For each FLEX_POOL primary layout and replica count, at most two unique alternate maps are evaluated: `NO_NEW_NODE`, restricted to nodes already used by the candidate primary plan and its replicas, and `UNRESTRICTED_FASTEST`, using all eligible inventory nodes. Both use the frozen deterministic group-cost and tie-break ordering. No weighted latency/cost coefficient or new hyperparameter is introduced.

### FLEX_POOL is a real superset

FLEX_FREE runs first. FLEX_POOL then searches two branches: `POOL-U` from U_STRONG and `POOL-FREE` from the final repaired FLEX_FREE plan after removing only the FLEX_FREE node-set restriction. Inherited FLEX_FREE actions do not count as newly accepted FLEX_POOL actions.

The final FLEX_POOL envelope contains exactly: U_STRONG cloned as FLEX_POOL, final FLEX_FREE cloned as FLEX_POOL, final POOL-U, and final POOL-FREE. Candidates below 99% of U_STRONG SLO throughput are discarded. The winner maximizes SLO throughput per abstract cost, then higher raw throughput, lower cost, fewer replicas, fewer used nodes, and canonical plan SHA-256.

All 27 inventories must pass the 1e-12-relative superset audit against both U_STRONG and FLEX_FREE and the cumulative U_STRONG throughput guard. FLEX_FREE must be present in every final envelope. A no-op remains feasible.

### Unchanged runtime and pure ablations

`routing.py::select_fork_join_routes` is unchanged: exhaustive copy masks, present calendars only, canonical group order, no lookahead, and current fork-join completion minimization. SHARED_NIC continues to reserve `link:A->B`, `nic_tx:A`, and `nic_rx:B` with the existing E022 duration and workload semantics.

`FLEX_FREE_NO_ALT` and `FLEX_POOL_NO_ALT` are derived only by removing alternate residency from the final repaired parents. P8 conversion, primary assignment, base placement, and chunk rows are preserved.

### Correctness and finalization

The five frozen final representatives use Experiment 022’s production-native authenticated 93-layer manifest correctness mechanism. An E023 wrapper may change only physical worker destination. Where a logical group has an alternate, it forces alternate execution when `(layer_id + logical_group_id + chunk_index) % 2 == 1`. Logical group identity, ownership, routes, weights, reduction slot, and canonical reduction order remain unchanged.

The finalizer will accept explicit primary and reproducibility attempts, validate every mandatory gate, collect all validity failures, and call the unchanged E023 verdict tree. It will not assume a control failure or any expected scientific outcome.

## Stop rules

The repaired run stops as `MODEL_INVALID` if the constants hash changes, any of 27 U_STRONG plan hashes differs from run 1, any pilot or authoritative control violates the frozen -5% efficiency or raw-throughput guard, any FLEX_POOL superset invariant fails, any full correctness representative fails, reproducibility fails, or another mandatory validity gate fails.

If a genuine implementation bug is discovered after the code freeze, the incomplete authoritative attempt is superseded in full, the code freeze is updated, and the 27-inventory run restarts from inventory 1. No partial headline results may be retained.

## Evidence boundary

The reused duplicate-group evidence is PHYSICAL. The multi-inventory serving results remain a PHYSICALLY GROUNDED MODEL with SHAPED NETWORK conditions. This repair does not add a physical multi-machine deployment, rented hardware, or a new inference method.
