# Experiment promotion ledger

This ledger is the human-readable index for the machine-readable
[`promotion_manifest.yaml`](../benchmarks/canonical/promotion_manifest.yaml). The manifest is
authoritative: every promoted or rejected mechanism has an owner and a release gate. Archived
experiment bundles remain immutable evidence; production code must not import them.

| Experiment | Status | Canonical product consequence |
|---|---|---|
| 001 | completed | Direct persistent peer data plane, measured replica planning, bounded reservations, and idle as a valid outcome. |
| 002 | completed | Immutable real-model stage ownership, exact direct BF16 execution, stage-local KV caches, and the native Qwen3 adapter. |
| 003 | completed | Persistent warm workers; startup, acquisition, residency, queue depth, and concurrency affect replica count. |
| 004 | completed | Dense-Qwen3 GPU-native/cache/compile/CUDA-graph modes are exactness-gated canonical candidates, never universal defaults. |
| 005 | no completed experiment | No capability or performance claim is admitted. |
| 006 | completed | Tensor and expert microsharding are capacity-first; speed requires new matched physical evidence. |
| 007 | corrected result | CPU background inference may add service throughput; synchronous CPU expert placement defaults off; idle is always eligible. |
| 008 | mixed result | Residency, movement, prefill/decode phase facts, and bounded candidate search are canonical; paging, prefetch, and adaptive caching remain rejectable. |
| 009 | PASS_STRONG | Colibri is a first-class probed engine; routing-aware placement is conditional; rejected prefetch and inferred capabilities remain disabled. |
| 010 | all 17 gates | Whole experts, native expert microshards, hybrid placement, deterministic reduction, isolation, detection, quarantine, and replay remain inside stages. |
| 011 | completed | The persistent direct contiguous stage ring is mandatory; compression and speculation are exact but auto-off without positive utility. |

## Disposition semantics

- `REQUIRED`: the canonical runtime owns and exercises the mechanism.
- `AVAILABLE_CONDITIONAL`: the runtime exposes it, but admission needs exactness and positive
  matched-runtime utility where the manifest says so.
- `REJECTED_DEFAULT`: the planner excludes it unless an operator explicitly forces a diagnostic
  test; forced tests do not become reusable performance evidence automatically.
- `EVIDENCE_ONLY`: fixtures, simulations, reports, superseded evidence, and fault injectors stay
  outside the production execution path.

## Permanent invariants

The coordinator never relays steady-state hidden states. A connected worker is not automatically a
participant. A distributed or optional mechanism must add required memory, independent compute,
aggregate throughput, expert/microshard ownership, beneficial storage/cache, or beneficial
background capacity. Otherwise the worker is assigned `idle` with an explanation.

Performance results from a failed exactness run are ineligible. Quantized exactness is relative to
the selected immutable quantized artifact and pinned reference runtime, not to BF16. An active
request never changes execution engine; replay recovery fails closed if the accepted prefix cannot
be reproduced exactly.

## Release contracts

The contract files in [`benchmarks/canonical`](../benchmarks/canonical/) freeze reference models,
hardware classes, correctness gates, performance gates, planner gates, and required telemetry.
They describe releases; they do not provide alternate implementations. Experiment 005 has an
explicit `NO_COMPLETED_EXPERIMENT` contract and no fabricated gate.
