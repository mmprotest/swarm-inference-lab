# Experiment 024: Repaired Kimi K3 Swarm Performance-Cost Frontier

## Verdict

The authoritative repaired E024 verdict is **MODEL_INVALID**.

The repaired architecture passed Phase 0, physical calibration, held-out service validation, physical D correctness, token-semantics validation, Stage A, and Stage B. Stage B then produced a canonical point only for COMMODITY_GOOD. The frozen full-correctness runner requires the canonical COMMODITY_REGIONAL D point. Because no such point existed, the mandatory two-token run could not start without selecting a substitute after performance results were known. No substitute was selected, no frozen code was changed, and the experiment failed closed with:

```text
NO_CANONICAL_REGIONAL_POINT_FOR_FULL_CORRECTNESS
```

This failure does not say that the two generated tokens were numerically wrong: no two-token physical execution started. It says the repaired attempt did not satisfy every preregistered validity gate, so its Stage B observations cannot produce a valid commercial Swarm verdict.

## Why Attempt 1 Was Invalid

Attempt 1 required a production-native degree-8 candidate for every transformer layer, including Kimi K3's structurally different dense layer 0. No production-native layer-0 P8 candidate existed, so Attempt 1 ended at Phase 0 as:

```text
MODEL_INVALID
NO_PRODUCTION_NATIVE_P8_CANDIDATE_FOR_LAYER_0
performance_results_seen = false
```

No calibration, physical correctness, Stage A performance, Stage B performance, output-token throughput, or economics existed in Attempt 1. The rule was therefore repaired before performance was observed. The complete Attempt 1 root artifact set and its report are preserved unchanged under `artifacts/experiment-024/archive/model-invalid-run-1/`; its 69-file archive manifest has SHA-256 `d0d5f63cdb3308acdca56b10782fb973575a44ff2e626e2c90ef8098c6cce6cc`.

Layer 0 is a dense layer with a frozen resident footprint of 2,549,338,530 bytes (2.374256523 GiB), which fits comfortably on the 10,422,845,440-byte (9.70703125 GiB) commodity worker class. Every layer from 1 through 92 needs approximately 17 GiB as a whole layer and cannot fit on that class. All 92 retain physically admitted, production-native degree-8 candidates.

No economic threshold, network definition, contributor payout, communication transformation, or performance-cost frontier rule changed in the repair.

## Corrected Commodity Architecture

> The E024 commodity Swarm uses one ordinary commodity worker to execute Kimi K3's small dense layer 0 as an admitted whole layer. The remaining 92 transformer layers cannot fit whole on the frozen commodity worker class and are executed exclusively through physically admitted degree-8 sub-layer candidates.

The architecture is therefore **92/93 transformer layers fine-grained, with the one naturally small dense bootstrap layer executed whole on an ordinary commodity worker.**

The Phase-0 audit established:

| Gate | Result |
| --- | ---: |
| Layer-0 candidate | `layer-00:WHOLE_LAYER:p1` |
| Layer-0 degree/type | 1 / WHOLE_LAYER |
| Layer-0 admitted and physically validated | PASS |
| Layer-0 resident memory | 2.374256523 GiB |
| Commodity worker memory | 9.70703125 GiB |
| Whole-layer-feasible IDs | `[0]` |
| Layers 1-92 with admitted P8 coverage | 92/92 |
| Admitted P8 candidates per MoE layer | 4 |
| Layers 1-92 whole-layer-infeasible | 92/92 |
| Complete whole-layer-only commodity K3 placement | INFEASIBLE |
| Complete repaired candidate coverage | PASS |

All frozen commodity placements contain exactly one whole transformer layer, layer 0, and exactly 92 P8 transformer layers, layers 1 through 92. No layer 1-92 uses a whole-layer fallback. Layer 0 uses an ordinary paid commodity node and receives no special memory, multiplier, network, or cost treatment.

## Evidence Classes

| Evidence | Class |
| --- | --- |
| Dense-layer, whole-layer, P8, and fusion calibration | PHYSICAL |
| Physical D correctness | PHYSICAL, single-device sequential workers |
| Stage A and Stage B | PHYSICALLY GROUNDED MODEL WITH SHAPED NETWORK |
| Two-token full correctness | NOT RUN: mandatory validity failure |
| Physical multi-machine Swarm | Not instantiated |

The modeled Swarm is not presented as a physical multi-machine deployment.

## Fresh Physical Calibration

All timed calibration used resident weights with no timed checkpoint reads.

The direct production-native `layer-00:WHOLE_LAYER:p1` calibration ran 20 warmups and 200 recorded iterations for each row count:

| Rows | Physical p50 service, ms |
| ---: | ---: |
| 1 | 3.13075 |
| 2 | 6.26535 |
| 4 | 12.56450 |

All 600 dense-layer samples were finite. The measured persistent allocation was 708,837,376 bytes, maximum observed allocation was 742,391,808 bytes, the catalog resident charge remained 2,549,338,530 bytes, and all were within the commodity memory class. Commodity service was scaled only by `physical_service_ms / compute_multiplier`; the concentrated reference used the same calibrated primitive at multiplier 1.0.

Fresh P8 and whole-layer calibration produced 1,200 samples each, and local fusion calibration passed. Held-out validation covered layers 89 and 91 at rows 1, 2, and 4:

| Service class | Median error | Maximum error |
| --- | ---: | ---: |
| P8 | 2.147526709% | 2.760982067% |
| Whole-layer | 2.862424966% | 6.268137747% |
| Combined | 2.147526709% | 6.268137747% |

No correction factor or post-result normalization was applied. The resulting 4,977-row service table has SHA-256 `9d40d0974de9168348119daf53d91fcb0938531fbfaabc0e9c6fc72e920767d6`.

## Physical Correctness and Token Semantics

Physical D correctness passed all six layer/row cases for layers 89 and 91 and rows 1, 2, and 4. Routes, route weights, state reconciliation, AttnRes reconciliation, canonical reduction order 0 through 7, finite output, production-native dispatch, and no-hot-read requirements passed. The observed output relative-L2 range was approximately `1.39e-8` through `9.03e-8`.

The token-semantics audit passed and records:

```text
layer_0_execution_kind = WHOLE_LAYER
layer_0_degree = 1
layers_1_92_execution_degree = 8
```

One completed decode step still means one newly generated output token after embedding, layer 0 whole execution, 92 P8 layer executions, final norm, LM head, greedy argmax, and state commit. No speculative or preloaded future-token semantics were introduced.

## Stage A Communication Result

Stage A completed all 288 frozen cells. The communication formulas and theoretical byte ledger remained unchanged:

| Arm | Bytes/row |
| --- | ---: |
| A_CURRENT | 1,004,416 |
| B_RETAIN_HIDDEN | 803,712 |
| C_SLICE_LATENT | 715,904 |
| D_FUSE_OUTPUT | 515,200 |
| Fixed-placement lower bound | 502,712 |

The A-to-D reduction was 48.7065120428%. D divided by the lower bound was 1.0248412610, satisfying the frozen 1.03 gate. Median Stage A gap closure was 13.5857889389%; maximum gap closure was 27.6410091680%; maximum regression was 0%.

This is a physically grounded shaped-network result, not a physical distributed-network measurement.

## Concentrated Reference

The frozen concentrated reference used 46 nodes, whole-layer execution for all 93 transformer layers, multiplier 1.0, FAST_FABRIC, E022 multi-layer memory, and deterministic contiguous packing.

Its C1 p95 output-token latency was 427.262505680 ms. The primary commodity SLO budget was four times that value, 1,709.050022720 ms. Reference SLO throughput was 57.3058556602 output tok/s.

## Stage B Frontier

Stage B built 42 repaired commodity placements, found 24 memory-feasible placements, evaluated 72 screening rows and 105 full decode rows, and retained four Pareto-frontier rows. Every commodity row used the repaired one-whole-layer-0-plus-92-P8 architecture. The frozen scenario networks, node budgets, compute multiplier cycle, closed-loop decode, microbatch limit, concurrency ladder, and cost equations were unchanged.

Only COMMODITY_GOOD had an SLO-feasible canonical point. COMMODITY_REGIONAL and COMMODITY_WAN had feasible placements and completed modeled decode rows, but their lowest full-ladder p95 latencies were 1,989.198045467 ms and 6,048.222746000 ms respectively, both above the 1,709.050022720-ms primary budget. They therefore had no canonical points under the frozen selection rule.

| Scenario | Canonical architecture | Budget / C | Output tok/s | Retention | Active nodes | $/M @ $0.15 | % of Kimi | Perf-cost leverage | Max payout/node-h @ $15/M |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GOOD | SWARM_D_OPT | 192 / C1 | 0.6501359572 | 0.0113450179 | 192 | 12,305.118508800 | 82,034.123391998% | 1.3829632612e-5 | $0.000182850738 |
| REGIONAL | No SLO-feasible canonical point | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| WAN | No SLO-feasible canonical point | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

The GOOD point's retention is 1.1345017882% when expressed as a percentage. Its active-node cost is $28.80/hour. Correctly dividing that hourly cost by `0.6501359572 * 3,600` output tokens/hour produces $12,305.118508800/M, an API-cost ratio of 820.341233920. It is not commercially cheaper than the frozen $15/M Kimi benchmark.

No scenario is below $15/M. The reported medians have a denominator of one canonical point and therefore equal GOOD:

| Metric | Canonical median (n=1) |
| --- | ---: |
| Performance retention | 0.0113450178816 |
| Cost/M | $12,305.118508800 |
| API cost ratio | 820.341233920 |
| Performance-cost leverage | 1.3829632612e-5 |
| Overall whole-layer-incapable compute share | 0.998155269577 |
| P8-required whole-layer-incapable compute share | 1.0 |
| Max uniform payout at $15/M | $0.000182850738/node-h |
| Max uniform payout at $7.50/M | $0.000091425369/node-h |
| Max uniform payout at $3/M | $0.000036570148/node-h |

The overall whole-layer-incapable share is diagnostic because layer 0 legitimately executes whole. The authoritative fine-grained metric excludes layer 0 from its denominator; its value is 1.0 and passes the frozen 0.95 gate.

## CURRENT Versus D

On the identical frozen GOOD D placement, changing execution semantics from CURRENT to D increased modeled output throughput by 30.1270127875% and reduced modeled cost per million by 23.1520052156%. Placement and service-table hashes were identical, so this is an execution-only causal check.

The Stage A mechanism gate passed. It cannot override a mandatory validity failure.

## Mandatory Two-Token Failure

The preregistered R18 check requires the canonical COMMODITY_REGIONAL D point for two consecutive physical one-row autoregressive steps. The frozen canonical-point artifact contained only COMMODITY_GOOD. The runner raised `StopIteration` before physical execution.

Selecting GOOD, a noncanonical REGIONAL row, or a new REGIONAL placement after observing Stage B would be a result-driven change. None was selected. T1 and T2 are therefore null, `physical_two_token_execution_started=false`, and full autoregressive correctness is not established for this attempt.

This mandatory failure mechanically overrides the otherwise passing mechanism gate and fixes the final verdict at MODEL_INVALID.

## Reproducibility and Quality

Stage A reproducibility passed with an exact 288-row digest of `a5ded150ac8b84d2d817d9a763321732b920816b3f638a6fa6a1b9c4ee5a347e`. Stage B reproducibility passed for the saved GOOD canonical point using its frozen placement with no replanning. These checks do not supply the missing REGIONAL correctness anchor.

| Check | Result |
| --- | --- |
| Focused E022-E024 tests | 102 passed in 72.92s |
| Full repository tests | 1,364 passed, 13 skipped in 270.05s |
| E024 compileall | PASS |
| E024/core Ruff | 0 findings |
| Repository Ruff before | 725 findings |
| Repository Ruff after | 725 findings |
| New repository Ruff findings | 0 |
| Chart visual QA | PASS |
| Frozen code changed after performance | No |

The repaired code-freeze covers 43 files and has payload hash `4906e25658f7a62560c33f6ea508529079be10102d77463b92d50a98fbbecc26`.

## What E024 Establishes

E024 establishes that the corrected catalog and memory architecture is structurally valid: layer 0 fits whole on an ordinary commodity worker, all 92 larger layers require and possess admitted P8 execution, and a complete whole-layer-only commodity K3 placement is impossible. It also establishes the A-to-D communication reduction, physical calibration evidence, physical D block correctness, and the reported shaped-network Stage A and Stage B observations.

It does not establish full two-token autoregressive correctness for the repaired canonical architecture, a valid commercial wedge verdict, or physical multi-machine throughput. The preserved GOOD model point is dramatically more expensive than the Kimi API benchmark and cannot support a commercial claim even before the global correctness veto.

## Recommendation for E025

**DO NOT START E025.** Preregister a further E024 repair that defines a deterministic full-correctness anchor even when a network scenario has no SLO-feasible canonical commercial point, freeze that rule before rerunning performance, and complete the mandatory two-token check before advancing experiments.

The repair must not choose an anchor using the already observed Stage B results. A scientifically clean rule could be tied to a predeclared scenario and deterministic feasible placement independent of commercial canonical selection, but that choice belongs in a new preregistration before any rerun.

## Reproduction and Artifacts

The authoritative artifacts are under `artifacts/experiment-024/`. Important entry points are:

- `summary.json`
- `truth-table.json`
- `failure-log.json`
- `validation/final-audit.json`
- `calibration/dense-layer0-whole-samples.csv`
- `calibration/dense-layer0-whole-service.json`
- `stage-b/decode-serving-results.csv`
- `stage-b/performance-cost-frontier.csv`
- `charts/`

Total recorded wall-clock runtime was 9,498.060844 seconds (2:38:18.061).
