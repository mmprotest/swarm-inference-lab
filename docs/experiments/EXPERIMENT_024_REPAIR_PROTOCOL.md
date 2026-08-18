# Experiment 024 Repair Protocol

## Status and historical record

This protocol is frozen before any repaired performance work. Experiment 024 attempt 1 remains permanently recorded as `MODEL_INVALID` with mandatory failure `NO_PRODUCTION_NATIVE_P8_CANDIDATE_FOR_LAYER_0`. Its hash-verified 69-file record is archived at `artifacts/experiment-024/archive/model-invalid-run-1/`; the archive manifest SHA-256 is `d0d5f63cdb3308acdca56b10782fb973575a44ff2e626e2c90ef8098c6cce6cc`.

Attempt 1 stopped before calibration, physical correctness, Stage A performance, Stage B performance, token throughput, or economics. Its frozen record says `performance_results_seen = false`. No performance result was used to design this repair. The repaired run remains Experiment 024 and is the authoritative scientific attempt; it is not Experiment 025.

## Why the rule is repaired

Attempt 1 required all 93 transformer layers to use physically admitted degree-8 sub-layer execution. That rule was over-constrained and did not represent the Swarm thesis.

Kimi K3 layer 0 is structurally different from layers 1 through 92. Its frozen admitted candidate, `layer-00:WHOLE_LAYER:p1`, has resident memory of 2,549,338,530 bytes (approximately 2.374 GiB) and fits comfortably on the frozen 10,422,845,440-byte (9.70703125 GiB) commodity worker. By contrast, every layer from 1 through 92 requires approximately 17 GiB as a whole layer and therefore cannot execute whole on that commodity class.

The corrected architecture is preregistered as:

- layer 0: exactly `layer-00:WHOLE_LAYER:p1`, executed on one ordinary commodity worker;
- layers 1 through 92: physically admitted, production-native degree-8 sub-layer execution only, with `WHOLE_LAYER` strictly forbidden.

Thus the repaired architecture has one naturally small whole layer and 92 necessarily fine-grained layers. A complete whole-layer-only commodity K3 placement remains infeasible.

## Frozen scientific question and hypothesis

The commercial question is unchanged:

> Can ordinary fragmented workers generate sustained correct Kimi K3 output tokens fast enough that, after every active worker is paid and throughput is correctly included in the denominator, cost per million output tokens is below the frozen $15/M Kimi API benchmark?

The falsifiable hypothesis is that at least two of the three frozen network scenarios contain an SLO-feasible, exact, fully costed point below $15/M under the corrected one-whole-plus-92-P8 architecture.

Any mandatory candidate, memory, calibration, correctness, architecture, reconciliation, reproducibility, or immutable-input failure makes the repaired attempt `MODEL_INVALID` before scientific interpretation.

## Unchanged experimental contract

This repair changes only the layer-0 validity rule and metrics that directly depend on it.

1. No economic threshold changes.
2. No network changes.
3. No contributor-price changes.
4. No communication-transformation changes.
5. No performance-cost-frontier changes.
6. No output-token-economics changes.
7. No additional research mechanism is introduced.

The frozen commercial values remain: $15/M API price; $0.15 per active node-hour primary payout; payout sensitivity $0.05/$0.10/$0.15/$0.25/$0.50; target costs $15/$12/$9/$7.50/$5/$3 per million; concurrency 1/4/16/64/128; primary decode SLO multiplier 4.0; sensitivity multipliers 2.0 and 8.0; and commodity node budgets 96/128/160/192/224/256/320. GOOD, REGIONAL, WAN, A/B/C/D communication formulas, Stage A, autoregressive token semantics, concentrated reference, and commercial verdict logic remain frozen.

## Evidence boundary

Fresh dense-layer-0 whole service for rows 1, 2, and 4 will be measured physically using resident real Kimi K3 weights and the existing production-native candidate. The distributed serving frontier remains a physically grounded model under the frozen shaped network scenarios. Evidence classes will be reported explicitly; modeled swarm results will not be described as a physical multi-node deployment.
