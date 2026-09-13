# E029 — Local Commit-Anchored WAN Tree

## Verdict
FAIL

## Decision
DRAFTER_LIMITED

## Headline
FAIL: real WAN-aware trees project 10.68 committed tok/s at 60 ms RTT, 3.84 tokens/traversal and 30.0% discarded target compute; oracle ceiling 18.33 tok/s. DRAFTER_LIMITED. These are trace-driven projections of three independent stage resources from real RTX 5090 measurements, **not physical three-GPU throughput**. Gate findings: Mean commits/traversal 3.835 below PASS minimum 4.0; 60 ms throughput 10.676 below PASS requirement 13.144 tok/s (1.25x sealed SPEC_SYNC); 60 ms throughput below sealed ASYNC 12.632 tok/s

## What changed from E028
Each explicit scored tree starts at an immutable committed root. The coordinator commits its longest valid path before constructing the next tree. There are no unresolved future rounds. E028's sealed artifacts were imported by hash; no E028 benchmark was rerun. Native DFlash2 supplies a scored candidate lattice through a generic drafter interface.

## Correctness
216 physical policy/prompt runs generated 55,296 committed tokens: 256 for each of the eight fixed prompts in every condition, with exact agreement against E029's same-engine scalar references. All 16 branch tests passed, including two/four siblings, deeper paths, rejected alternatives, unchanged committed-root hashes, and deterministic continuations. Confirmed cross-branch contamination: 0.

Packed and sequence-fork compatibility attempts changed greedy continuations. The final worker uses canonical scalar node execution, device-resident partial-state checkpoints, and KV-tail truncation. Branched trees replay the accepted path on commit; linear trees use shared-prefix sequence checkpoints. Every call, copy, restore and replay is charged. The live checkpoint stack grows with depth; full contexts are not copied per node. The draft service is charged at measured GPU latency as a coordinator resource, following E028's abstraction; no target/draft compute overlap occurs across a commit-anchored round. Remote drafter placement itself was not physically tested. Distributed scheduling/transport remains model-generic; backend-specific draft feature handling is confined to its adapter. E029 prefill uses 256-row batches; one long-context scalar reference differs from E028's 512-row prefill reference. Comparisons use the sealed E028 throughput values and E029's exact internal correctness control.

## Real tree verification cost
E028's measured contiguous stage boundaries are reused: [0,22), [22,45), [45,64). Physical RTX 5090 stage measurements: N=8: 237.2 ms median across the three stage calls; N=16: 499.7 ms median across the three stage calls; N=32: 900.7 ms median across the three stage calls. These values exclude commit replay, which is charged separately in every WAN projection. Peak VRAM was 26.13 GiB. Native DFlash2's median proposal latency was 4.17 ms; its five target feature taps require 102,400 serialized bytes per committed token in addition to stage activations. CUDA-event-only timing is unavailable; traces record synchronized native execution and host/service overhead.

The simulator reused E028's resource and directed-link event machinery. For a 32-token STATIC_TREE condition (N=8, D=4, 11 rounds), independent unshaped service measurements predicted **6.310 s**, versus **6.314 s** measured with actual local 60 ms RTT delays: **0.06% error**, within ±10%. Replaying the shaped run's observed service durations gives 5.25% error, also within tolerance. Physical commit calls share one GPU; the virtual model permits independent stage commits. No fitted normalization was applied.

## WAN results
The real policy's best 60 ms operating point is N≤32, D≤8. That ceiling is held fixed below while WAN-aware tree shape adapts to RTT. Links use 100 Mbps and seeded jitter; LOCAL uses unlimited bandwidth. A round charges four activation/control traversal hops plus parallel commit requests and committed-feature replies. No next round overlaps it.

| RTT (ms) | Projected committed tok/s | Tokens/traversal | Discarded compute | WAN wait |
|---:|---:|---:|---:|---:|
| 0 | 34.86 | 2.54 | 2.5% | 0.0% |
| 30 | 14.85 | 3.68 | 23.2% | 47.9% |
| 60 | 10.68 | 3.84 | 30.0% | 59.3% |
| 100 | 7.85 | 3.93 | 35.2% | 67.4% |
| 150 | 5.92 | 3.98 | 39.3% | 73.4% |

| 60 ms condition | Projected committed tok/s | Tokens/traversal | Discarded compute |
|---|---:|---:|---:|
| E028 SERIAL | 6.52 | 1.00 | 0.0% |
| E028 SPEC_SYNC | 10.52 | 2.90 | 21.3% |
| E028 ASYNC | 12.63 | 0.59 | 74.0% |
| E029 Linear | 9.55 | 3.54 | 39.8% |
| E029 Static | 5.87 | 3.25 | 50.5% |
| E029 Probability | 6.68 | 3.34 | 45.7% |
| E029 WAN-aware | 10.68 | 3.84 | 30.0% |
| E029 Oracle ceiling | 18.33 | 8.00 | 0.0% |

| Real WAN-aware gate | Observed | PASS requirement |
|---|---:|---:|
| Committed tokens/traversal | 3.835 | >= 4.0 |
| Discarded target compute | 30.0% | <= 40% |
| Throughput / sealed SPEC_SYNC | 1.015 | >= 1.25 |
| Throughput / sealed ASYNC | 0.845 | >= 1.00 |
| 100 ms / 60 ms throughput | 73.5% | >= 55% |
| Gain over strongest probability tree | 59.7% | >= 10% (failure gate) |

## Useful work per traversal
The primary WAN-aware result commits 3.84 tokens per traversal (path p50 3.0, p95 8.0); verifies 5.52 nodes per round; and uses 35.80 ms of target/state compute per committed token. Its useful target-compute share is 70.0%. Physical target calls and logical WAN rounds are recorded separately, including accepted-path replay. E029's discarded work is off-path verification/state work, not unresolved-round invalidation.

## Did tree speculation solve E028's wasted-compute problem?
No for the real drafter. Root anchoring removes causal invalidation across rounds, but the useful accepted path still does not justify the measured work and WAN round cost.

## WAN-aware optimization
At 60 ms, the best WAN-aware point improves throughput by 59.7% over the strongest simple probability tree. Its optimizer uses draft path scores and measured scalar-node, draft and state service costs, plus activation/feature bytes, RTT and bandwidth; it prunes when an expansion reduces estimated committed tokens per second. Speculative children and branch allocation use only draft scores; the first executed anchor is the greedy token already known from the previous committed round. It is charged and counted only when executed, with no free bonus tokens. Selection estimates are saved beside actual utility. 99.8% of selected rounds are single chains; the strongest E029 linear control projects 9.55 tok/s. Any chain-dominated result is evidence about adaptive commit-anchored blocks, not a demonstrated benefit from branching.

The cost profile crosses all nine N/D combinations. The decision workload uses the fixed pairs (8,4), (16,6), (32,8); N and D effects in those throughput curves are therefore coupled. N=64 was not required and was not run. Per-prompt results and three fixed jitter seeds are retained; the seeds replay the same physical traces and are not independent hardware repetitions. Aggregate rates divide total tokens by total elapsed time.

## Oracle ceiling
The diagnostic oracle commits 8.00 tokens per round and projects 18.33 tok/s at 60 ms, with 0.0% discarded compute. It uses target reference tokens and **zero draft-generation/injection cost**, while retaining measured target/state costs and the same feature-transfer protocol. It is an architectural ceiling, not a real drafter result. Its 100/150 ms throughput retention is 78.2%/61.4%, and target/state compute costs 24.83 ms per committed token. PASS gates: True; PASS_STRONG gates: True.

## Bottleneck
The remaining cost is the accepted progress per synchronous round relative to WAN transit, off-path scalar verification, partial-state copying and commit work. The real 60 ms run spends 59.3% waiting only on the network; target/state compute costs 35.80 ms per committed token. Useful candidate coverage, rather than pipeline occupancy, determines whether that cost is amortized.

## Final decision
DRAFTER_LIMITED. Real candidates fail the gates while the ideal-coverage oracle passes strongly. At the current mean round duration, PASS requires at least 4.72 committed tokens per round versus 3.84 measured (1.23× coverage). This is an optimistic lower bound: additional committed-feature traffic and state work can raise it. Discarded target compute must remain at most 40%.
