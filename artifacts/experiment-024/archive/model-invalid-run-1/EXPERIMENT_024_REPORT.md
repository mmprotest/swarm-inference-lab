# Experiment 024: Communication-Avoiding Kimi K3 Swarm Economics

## Verdict

The mechanical verdict is **MODEL_INVALID**. The frozen E022 catalog has no physically admitted, production-native degree-8 candidate for transformer layer 0. E024 forbids WHOLE_LAYER for every transformer layer, so no complete 93-layer P8-only placement can exist at any commodity node budget.

## Executive Summary

Phase 0 found zero admitted degree-8 candidates for transformer layer 0. Because the experiment requires all 93 transformer layers to use P8 and forbids whole-layer fallback, no valid Stage B deployment can be constructed. The run stopped before physical calibration or performance modeling.

| Scenario | Architecture | Output tok/s | Perf. retained | Active nodes | $/M @ $0.15/node-h | % of Kimi cost | Perf/$ leverage | Max payout @ $15/M | Whole-layer-incapable compute |
| -------- | ------------ | -----------: | -------------: | -----------: | -----------------: | -------------: | --------------: | -----------------: | ----------------------------: |
| GOOD | NOT EVALUATED | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| REGIONAL | NOT EVALUATED | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| WAN | NOT EVALUATED | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| MEDIAN | NOT EVALUATED | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## The Actual Swarm Thesis

The intended thesis remains whether fragmented ordinary compute can serve exact Kimi K3 output tokens at commercially competitive cost. This invalid run does not adjudicate that thesis.

## Commercial Benchmark

The frozen benchmark is **$15.00 per million output tokens**. It was not refreshed.

## Evidence Boundary

The only new E024 evidence is a deterministic immutable-input audit. No PHYSICAL_LOCAL, PHYSICALLY_GROUNDED_MODEL, SHAPED_NETWORK, or CONTROLLED_REFERENCE performance result was produced.

## Output Token Definition

The code preserves the frozen definition: one completed decode row generates one token after embedding, 93 transformer layers, final norm, LM head, greedy argmax, and state commit. It was not executed because Phase 0 failed.

## Fresh Physical Calibration

Not run after the mandatory Phase 0 failure.

## Current Communication Problem

The frozen accounting remains A=1,004,416 bytes/row and D=515,200 bytes/row. These are theoretical geometry checks, not measured results.

## Communication Lower Bound

The fixed-placement uncoded communication payload lower bound under the frozen P8 decomposition. is 502,712 bytes/row; D/lower-bound is 1.024841261000.

## A/B/C/D Transformations

Frozen payloads are A=1,004,416, B=803,712, C=715,904, and D=515,200 bytes/row. No Stage A performance execution occurred.

## Physical D Correctness

Not run after the mandatory Phase 0 failure.

## Stage A Results

No valid Stage A results were generated.

## Concentrated Fast Reference

Not constructed; no reference denominator exists.

## Commodity Worker Definition

The frozen worker-memory rule remains E022 `memory_classes(model)["sub_layer"]`: 10,422,845,440 bytes (9.70703125 GiB) per worker. The pool was not constructed because candidate admission had already failed.

## Commodity Placement

The catalog contains 2 layer-0 P8 candidate records and 0 admitted records. Both recorded candidates are `INELIGIBLE_UNVALIDATED`; all E022 completed placements therefore use `WHOLE_LAYER:p1` for layer 0.

## Autoregressive Decode Serving

Not run; a complete P8-only model placement is a prerequisite.

## Performance-Cost Frontier

No frontier exists for this invalid attempt.

## COMMODITY_GOOD

Not evaluated.

## COMMODITY_REGIONAL

Not evaluated.

## COMMODITY_WAN

Not evaluated.

## Cost per Million Output Tokens

Not calculated without valid output-token throughput.

## Contributor Payout Frontier

Not calculated without valid output-token throughput.

## Performance Retained versus API Cost

Not calculated; both axes require a valid Stage B result.

## Whole-Layer-Incapable Compute

Not calculated; no transformer execution schedule exists.

## SWARM_CURRENT versus SWARM_D

Not evaluated.

## Two-Token Autoregressive Correctness

Not run after the mandatory Phase 0 failure.

## Reproducibility

The invalidity is reproduced from candidate catalog SHA-256 `3f1d8e8519fb2b759b7ec5678bfc1458a782a49a6e438eeb1b26d2077258ccd7` and repaired service SHA-256 `ee240937dfc36a6ce04161812f61ff0021359a2861ecb783772d44e2614bcf2b`.

## Limitations

1. This experiment covers output-token decode economics only.
2. Prompt-prefill and input-token economics are not included.
3. No real WAN cluster was used.
4. No contributor churn was modeled.
5. The intended network model is deterministic and shaped.
6. Contributor payout is an assumption, not an observed market price.
7. The concentrated reference is not an H100 benchmark.
8. Intended physical compute calibration is from a single RTX 5090.
9. A resulting $/M would be a modeled serving-cost estimate grounded in physical compute measurements; no such estimate was produced in this invalid run.

## What E024 Proves

The frozen E022 inputs cannot instantiate the required complete 93-layer P8-only Stage B architecture.

## What E024 Does Not Prove

It does not prove or disprove communication avoidance, throughput, correctness, or commercial economics.

## Recommendation for E025

Fix only the invalid evidence path by defining and physically admitting an exact production-native P8 candidate for transformer layer 0, then rerun identical E024 scientific and economic assumptions.

## Reproduction

Run `python scripts/experiment_024_freeze.py` from the repository root. The command must return `MODEL_INVALID` while the frozen catalog is unchanged.
