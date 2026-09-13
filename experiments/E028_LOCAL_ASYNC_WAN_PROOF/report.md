# E028 — Local Async WAN Proof

## Verdict
**FAIL**

## Headline result
Trace-driven three-stage WAN simulation projects 12.63 committed tok/s at 60 ms RTT with K=1, W=8; 1.20× the strongest synchronous speculative control, 36.3% of the best zero-WAN throughput. WAN wait is 11.6% and discarded target compute is 74.0%. These are **trace-driven simulation results**, derived from one RTX 5090, not measured three-GPU throughput. Decision: **STOP_CURRENT_SWARM_ARCHITECTURE**.

## Correctness
The final protocol completed 128 physical runs and 32,768 committed tokens: eight fixed prompts, 256 tokens each, every K/W combination, and serial controls. The two long-context inputs contain 2,037 and 2,281 tokens; neither required a reduced output length. Exact synchronous/async agreement: **True**. Natural rollback events: **7,324**. Forced events: **360** on the first conversational prompt, across positions 1, 2, 3 and windows 2, 4, 8; stress rollback failures: **0**; confirmed state corruption: **0**. The stress run made **1,146 stage comparisons of both complete and partial-state fingerprints**, checked restored sequence positions and cached-prefix output checksums, and compared continuation with the clean control. Stale chunks never commit. The extra W=8 shaped-network run also matched exactly and reached 8 unfinished target verifications at once. The W=8 and W=16 implementations fit locally; sampled peak total GPU memory was 27.73 GiB.

## Simulator validation
Independent E028 measurements supplied service times. The initial serial WAN-60 check missed the gate at 10.57% because native service increased during idle gaps. A separate fixed prompt then supplied periodic native service measurements for the non-speculative path; the table below uses fresh held-out validation runs. The initial attempt is preserved in `archives/validation_attempt_1/`. Actual local sleeps injected each WAN hop and the parallel rollback controls. No timing correction was fitted to the validation runs.

| Condition | Measured | Predicted | Absolute error |
|---|---:|---:|---:|
| WAN-30, K=0, W=1 | 5.654 s | 5.271 s | 6.79% |
| WAN-60, K=0, W=1 | 10.092 s | 9.854 s | 2.36% |
| WAN-60, K=3, W=1 | 5.162 s | 4.934 s | 4.42% |

Serial validity gate: **True** (every condition ≤10%). This validates serial accounting, not real multi-GPU contention or WAN transport.

## Performance
The target is the existing Qwen3.8-27B Q4_K_M GGUF, with its local Q8_0 MTP artifact. Measured layer profiling selected contiguous ranges [0,22), [22,45), [45,64); all simulator service times are actual operations on those stages. The virtual coordinator/drafter has independently charged GPU-capable MTP service measured on the same 5090.

Fixed configuration below: **K=1, W=8**, selected by aggregate 60 ms throughput. K denotes draft tokens beyond a leading token; full chunks contain K+1 rows. Verification uses scalar target kernels coalesced into one wire chunk; no fused-batch speedup is assumed. Ratios use the strongest SPEC_SYNC configuration at each RTT. The zero-WAN reference is the best LOCAL speculative configuration: K=1, W=4, 34.83 committed tok/s. Rates pool total committed tokens / total elapsed time over eight prompts and three fixed jitter seeds. Decode includes drafting, pipeline fill/drain, rejection, rollback and transfers; prefill is separate in TTFT.

| RTT (ms) | Committed tok/s | vs best SPEC_SYNC | Zero-WAN retained | WAN wait | Discarded compute |
|---:|---:|---:|---:|---:|---:|
| 0 | 34.72 | 1.10× | 99.7% | 0.0% | 54.3% |
| 30 | 18.13 | 1.22× | 52.0% | 7.9% | 70.6% |
| 60 | 12.63 | 1.20× | 36.3% | 11.6% | 74.0% |
| 100 | 8.93 | 1.16× | 25.6% | 18.7% | 75.0% |
| 150 | 6.50 | 1.13× | 18.7% | 29.3% | 75.2% |

The SERIAL control projects 6.52 committed tok/s at 60 ms. At 60 ms, the same-K speedup is 1.34×; the stronger all-K synchronous comparison is 1.20×. Failed required checks: PASS/speedup_60. PASS_STRONG check details are in `summary.json`. All 1,920 per-run projections and 80 aggregates are retained; the six figures use one fixed K.

## Did async hide WAN latency?
No. Moving from 60 to 100 ms RTT reduces W=8 throughput by 29.3%, almost the same loss as W=1 (30.3%). W=4 loses 30.4%. The required asynchronous speedup is not reached, and throughput remains strongly sensitive to RTT. Consult the throughput, wait and discarded-compute curves together: stage activity includes work that can later be invalidated. The functional traces explicitly record launches preceding older verification completion; concurrency is represented in the actual scheduler.

At the same fixed K, the RTT sensitivity is:

| W | 30 ms tok/s | 60 ms tok/s | 100 ms tok/s | 150 ms tok/s | 100/60 ms throughput |
|---:|---:|---:|---:|---:|---:|
| 1 | 13.98 | 9.42 | 6.56 | 4.76 | 69.7% |
| 4 | 18.34 | 12.39 | 8.63 | 6.23 | 69.6% |
| 8 | 18.13 | 12.63 | 8.93 | 6.50 | 70.7% |

At W=4 and W=8, discarded target compute at 60 ms is 57.5% and 74.0%, respectively.

## Genericity
**True**. Scheduler, transport, chunk invalidation and rollback use generic tokens, activations and llama state APIs. The native adapter queries public RoPE metadata for text positions. Existing architecture-specific model graph support remains inside the pinned llama.cpp build; E028 adds no model-specific numerical kernels or model-name scheduling branch. Only this model/backend combination was physically tested.

## Main bottleneck after E028
At the best setting, only 15.5% of launched speculative input positions become committed, and 74.0% of target compute is discarded. Steady virtual stage occupancy is A=55.4%, B=50.3%, C=38.1%. Provisional-draft rejection and invalidation/refill cycles leave WAN verification/rollback dependencies on the committed-token path. Resident scalar verification and the independently charged MTP drafter also consume measured compute. Increasing raw stage activity is useful only when it yields more committed tokens.

## Decision
**STOP_CURRENT_SWARM_ARCHITECTURE**

Artifacts: `summary.json`, `correctness_results.json`, `rollback_stress_results.json`, `stage_profile.json`, `simulator_validation.json`, `wan_sweep_results.jsonl`, `wan_aggregate.json`, `plots/`, and reproducible sources listed in `provenance/`. No paid resources or additional hosts were used.
