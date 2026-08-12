# Experiment 015: Break the Depth Barrier

Experiment 015 is an architecture research failure, not a negative commercial proof. It did not establish a correct, representative speculative Kimi path or implement the distributed EP/DCP/TP mechanisms needed to decide the full search. It did establish several useful upper bounds that prevent an expensive physical canary.

## Research outcome

The answer to the ultimate simultaneous question is **not demonstrated**. The best admitted architecture-level result is an eight-layer microcell **SHAPED** result at 1.1766 dependency-bound tok/s and 1.0446 aggregate tok/s per paid-GPU-equivalent. It misses both primary thresholds.

More strongly, the optimistic zero-draft-cost, perfect-acceptance block-7 plus depth-8 bound is only 2.1838 tok/s. That bound proves the currently modeled batch-verification service cannot reach 5 tok/s merely by adding DSpark and reducing slow boundaries.

## Evidence contract

* **MEASURED** means locally executed real Kimi weights and real CUDA. There are no exceptions.
* **SHAPED** means measured payloads/timings under synthetic transport.
* **VALIDATED MODEL** means a model validated against held-out real execution; the component service model reached 3.09% median error. Results that add synthetic topology remain **SHAPED**.
* **PROJECTED** means an architecture or economic consequence of those inputs.

CPU correctness diagnostics are marked `scientific_result: false` and have no evidence class.

External DSpark, vLLM, DeepEP, EPLB, DCP, and offloading claims appear only in the research-source ledger. They are not Swarm results.

## Immutable B015-000 control

The Experiment 014 capture remains unchanged: 93 stages, one layer per stage, safe batch 8, batch 9 fail-closed, 97.15 aggregate tok/s, 0.9503 dependency-bound tok/s, and 1.0446 aggregate tok/s per paid-GPU-equivalent. At the configured $0.15/GPU-hour this is $39.89/M output tokens.

Its dependency path is 747.16 ms compute plus 92 coarse boundaries. At depth 8, the service model exposes only 11 coarse boundaries but retains 81 internal boundaries and every layer's compute.

## Hypothesis cycles

The complete hypothesis → implementation → benchmark → inspection → redesign record is in [cycle-ledger.md](../artifacts/experiment-015/cycle-ledger.md). Failed directions are not hidden.

## Component findings

### Speculation

The pinned 7.12 GB public DSpark checkpoint executed real weights against retained real Kimi hidden states. Blocks 1/2/3/5/7 were finite and deterministic, and greedy/stochastic transaction tests covered full acceptance, partial acceptance, first rejection, full rejection, EOS, cancellation, rollback, repetition, and request isolation.

That is not enough to certify speculative inference. No representative workload received target verification, so accepted length, positional acceptance, target traversals/output token, draft GPU cost, rollback cost, and end-to-end speedup are **NOT ESTABLISHED**. The CPU reference timing is correctness instrumentation, not capacity evidence.

### Pipeline-aware speculation and cancellation

This branch stopped at its prerequisite. The pinned vLLM source explicitly raises for DSpark plus pipeline parallelism, while Swarm lacks a passing synchronous DSpark baseline. Transaction cancellation had zero corruption in unit tests, but layers/CUDA/messages avoided were not benchmarked.

### Hierarchical microcells

Depth 1/2/4/8 produced 0.9503/1.0668/1.1365/1.1766 tok/s in the shaped topology model. Depth 8 is best within the preregistered sweep. It is a useful 23.8% latency sensitivity, not a depth-barrier solution.

### Expert microwork and overlap

The real Kimi sweep remains 2/4/8/16 workers. Four workers are best, at 3.931 GB tracked per worker and 78.81% relative complete-layer throughput. A zero-replacement-cost subtraction of host copies, serialization, exposed transport, and imbalance gives a 95.56% ceiling. Since no device-resident transport implementation ran, the >=90% research target failed.

BF16 and FP16 were checked only as real-hidden-row transport round trips; their full-layer correctness gates did not run. FP8/MXFP8 is unsupported by the retained Kimi transport kernel. No low-precision format was promoted.

### Decode Context Parallelism

Stable max/denominator/numerator combination reproduced unsharded attention to machine precision for degrees 1/2/4/8. The shaped Kimi timing model predicts an 8K DCP8 stage gain of 40.47%. Complete Kimi MLA state, cancellation/recovery, aggregate capacity, and paid compute did not run, so neither the beneficial context threshold nor optimal degree is established.

### Targeted TP

Real CUDA TP1 operation timings were combined with a shaped 0.25 ms/25 Gbps collective. KDA q/output and the MLA projection group did not beat TP1; LM-head TP4 had a 1.63x operation sensitivity. It saves too little of the 1,052 ms path to justify unmeasured extra workers, so no TP was retained.

### Expert placement, prediction, and residency

The available trace contains only three target tokens across 92 MoE layers. Static modulo placement had a p95 hottest/coldest ratio of 8x. A trace-fitted plan reduced mean critical selections by 10.16% in-sample, which is not held-out evidence. The previous-token predictor achieved only 17.09% precision/recall and was rejected. Replication, dynamic migration, and steady-state tiered residency were stopped for insufficient trace history.

### Packing and economics

Memory packing is modeled independently of compute. The 0.915 GiB 16-way shard is the smallest measured logical worker, but it delivered poor layer efficiency. A 24 GiB device can memory-pack all four 3.931 GB expert shards, yet one device cannot provide four-way concurrent compute. Unknown hardware prices, bandwidth, and compute are never imputed from VRAM.

The best admitted projection still requires 93 paid-GPU-equivalents, about 2232 GiB nominal VRAM under the legacy 24 GiB stage assumption, and costs $39.89/M output tokens.

## Required serving metrics

| Metric | Best justified value | Evidence |
| --- | ---: | --- |
| Dependency-bound tok/s | 1.1766 | SHAPED |
| Aggregate tok/s / paid-GPU-equivalent | 1.0446 | PROJECTED |
| Cost/M output | $39.89 | PROJECTED, configurable economics |
| TTFT | NOT ESTABLISHED | no end-to-end serving run |
| p50/p95/p99 inter-token latency | NOT ESTABLISHED | no Experiment 015 end-to-end run |
| Logical EP memory/worker | 3.661 GiB | MEASURED |
| Total active GPU-equivalents | 93 | PROJECTED |
| Network bytes/token | 23,740,416 | PROJECTED FP32 92-boundary payload |
| Synchronizations/token | 92 total; 11 coarse at depth 8 | SHAPED |
| GPU utilization | NOT ESTABLISHED | no physical topology |
| Communication overlap | NOT ESTABLISHED | no resident async implementation |
| Numerical fidelity | control PASS; speculative/DCP full path NOT ESTABLISHED | mixed, explicitly scoped |

## Answers to the 30 hard questions

1. Speculation's Swarm contribution is not measured; the perfect combined upper bound is 2.18 tok/s.
2. Accepted tokens/target traversal on our workloads: **NOT ESTABLISHED**.
3. Async vs ordinary speculation: **NOT ESTABLISHED**.
4. Microcells reduce slow boundaries from 92 to 11 at depth 8 and improve modeled speed 23.8%.
5. The best tested microcell depth is 8.
6. Expert microwork did not improve beyond 78.8% in a measured implementation.
7. Reaching the 95.6% ceiling requires persistent device buffers, direct packed activations, device routing metadata, and async device transport; this remains a hypothesis.
8. Smallest measured shard: 0.915 GiB; smallest economically useful footprint: **NOT ESTABLISHED**.
9. Four expert workers remain best measured.
10. Retained practical network domain: <=0.5 ms RTT and >=2.5 Gbps; redesigned admission is not certified.
11. Speculative batching economics: **NOT ESTABLISHED**; batch-8 verification already consumes aggregate capacity.
12. DCP's full Kimi effect: **NOT ESTABLISHED**; shaped stage sensitivity is positive.
13. DCP break-even context: **NOT ESTABLISHED**.
14. Optimal DCP workers: **NOT ESTABLISHED**.
15. Only the LM head showed a TP sensitivity; none passed a complete-stage gate.
16. KDA/MLA collective overhead outweighed savings by TP2/4 in the modeled internal network; LM head preferred TP4.
17. Real-route skew was material in the three-token fixture, with p95 hottest/coldest ratio 8x.
18. Hot-expert replication: **NOT ESTABLISHED**.
19. Dynamic placement is not justified without representative drift traces.
20. Previous-token route prediction was too weak at 17.09% precision/recall.
21. Removable expert VRAM under steady-state tiering: **NOT ESTABLISHED**.
22. Tiered-residency throughput loss: **NOT ESTABLISHED**.
23. Tiering tok/s/$ benefit: **NOT ESTABLISHED**.
24. Best measured complete architecture: B015-000 control; best justified shaped result: depth-8 cells.
25. No EP/DCP/TP/speculation combination passed admission.
26. Depth-8 sensitivity assumes <=0.25 ms/25 Gbps internal and 5 ms/10 Gbps coarse links.
27. Memory-only packing spans 8/12/16/24/32/48 GiB classes; compute-qualified worker sizes are not established.
28. Best justified dependency prediction: 1.1766 tok/s.
29. Best justified aggregate efficiency: 1.0446 tok/s/GPU-equivalent.
30. Projected GPU cost: $39.89/M output tokens at $0.15/GPU-hour.

## Stop decision and Experiment 016

The experiment stops because all unimplemented branches either depend on the failed standard-DSpark gate or cannot enter the Pareto frontier without representative traces and complete Kimi correctness. No GPU fleet was rented, no RTX 3090 canary ran, and shaped data is not described as physical.

No Experiment 016 physical-validation plan is produced. Physical validation would be premature. The next work should remain local and produce representative DSpark acceptance plus at least one real resident EP or DCP implementation.

## Final regression record

The complete repository suite passed 1,113 tests with 13 explicitly gated skips. The new focused Experiment 015 suite passed 22 tests. Ruff and Mypy passed. The exact final Kimi CUDA binary then executed all 93 layers for prefill and stateful decode against the pinned independent oracle: routing equality was exact and maximum layer relative L2 was 9.922e-07 under the unchanged 2.0e-06 gate. Its 3666.3-second streamed duration is correctness-only and is not used as serving capacity evidence.
