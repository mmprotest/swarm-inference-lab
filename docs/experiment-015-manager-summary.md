## Verdict

* Experiment 015: **FAIL** — unresolved implementation/correctness evidence
* Best architecture: no qualifying Experiment 015 architecture; B015-000 remains the only complete control. The best justified sensitivity is eight-layer microcells.
* Speculative decoding: real public DSpark weights produced finite deterministic draft blocks; representative target acceptance and target-distribution equivalence were not established.
* Pipeline-aware speculation: **NOT ESTABLISHED**; the synchronous prerequisite failed and the pinned public runtime rejects DSpark pipeline parallelism.
* Best accepted tokens/target pass: **NOT ESTABLISHED** (public external cross-workload mean: 3.85, not Swarm evidence)
* Best dependency-bound decode: **1.1766 tok/s** (SHAPED, microcells only)
* Baseline dependency decode: **0.95 tok/s**
* Dependency speedup: **1.238x**
* Best aggregate tok/s/GPU-equivalent: **1.0446**
* Baseline: **1.04**
* Throughput/$ speedup: **1.000x**
* Break-even target: **2.78**
* 50%-margin target: **5.56**
* Best microcell depth: **8 layers** (SHAPED, not physical)
* Best expert microwork size: **4 workers**
* Microwork memory/worker: **3.661 GiB tracked**
* Best microwork layer efficiency: **78.81% MEASURED**
* Minimum microwork network: **<=0.5 ms RTT and >=2.5 Gbps** for the retained Experiment 014 practical domain; the redesigned direct-buffer minimum is not established.
* DCP gain: **40.47% SHAPED** at 8K/DCP8; no complete Kimi DCP result
* Best DCP degree: **NOT ESTABLISHED** (DCP8 is only the best latency sensitivity)
* Useful TP operations: **LM head only in a SHAPED TP4 sensitivity; none retained**
* Expert load imbalance: **p95 hottest/coldest ratio 8.0x** on a non-representative three-token trace
* Expert replication gain: **NOT ESTABLISHED**
* Route-prediction usefulness: **REJECTED**; precision/recall 17.09%
* Minimum useful GPU worker memory: **NOT ESTABLISHED**; 0.915 GiB is the smallest measured shard, while 8 GiB is the smallest modeled class with safe room for the promoted four-way ready delta.
* Paid-GPU-equivalent requirement: **93**
* Projected cost/M output at $0.15/hr: **$39.89**
* >=5 tok/s/user target: **FAIL**
* >=2.78 tok/s/GPU target: **FAIL**
* >=10 tok/s/user stretch: **FAIL**
* >=5.56 tok/s/GPU stretch: **FAIL**
* Ready for physical architecture validation: **NO**
* Full 93-layer Kimi CUDA regression: **PASS** — 93/93 layers, exact routing, stateful decode, maximum relative L2 9.922e-07

1. **Why was the 93-stage architecture slow?** Every token crosses 93 serial transformer layers, 92 coarse synchronizations, and about 747 ms of compute; aggregate batching cannot remove this dependency path.
2. **How much did speculation help?** It did not produce an admissible Swarm speedup. Even the zero-draft-cost, perfect block-7/depth-8 upper bound is only 2.18 tok/s.
3. **How much did reducing coarse boundaries help?** Depth 8 reduced modeled latency from 1,052.3 to about 849.9 ms, a 23.8% speed improvement.
4. **Did microworkers become economically useful?** No. Four-way EP remained 78.81% of the one-GPU layer and concurrent compute cost was not reduced by memory packing.
5. **What specifically improved microwork efficiency?** No implementation improvement was retained. Removing individually measured host/serialization overhead yields a 95.56% optimistic ceiling, not a benchmark.
6. **Did DCP fix the MLA context bottleneck?** Not yet. Exact component reduction passed, and an 8K/DCP8 shaped model showed 40.5% stage gain, but full Kimi state/capacity did not run.
7. **Where did TP help?** Only the LM head in a shaped aggressive-network sensitivity; KDA projection collectives outweighed compute saved.
8. **How important was expert imbalance?** Static placement showed material in-fixture imbalance; a fitted plan cut mean critical selections by 10.16%, but the trace was too small for promotion.
9. **Could route prediction hide communication?** The simple predictor could usefully anticipate only 17.09% of predicted expert activations, so no.
10. **Could GPU expert residency be materially reduced?** Not established; three tokens cannot measure a cache hit rate or steady-state H2D cost.
11. **Which combination won?** None. Microcells alone are the only retained model improvement.
12. **What is now the main bottleneck?** Multi-row target verification service across all 93 layers, followed by unresolved resident EP/DCP implementations.
13. **What hardware/network topology does the winning architecture need?** There is no winning architecture. The microcell sensitivity assumes <=0.25 ms/25 Gbps internally and 5 ms/10 Gbps between cells while retaining 93 paid GPU-equivalents.
14. **Does it reach interactive serving?** No: 1.1766 tok/s vs 5 tok/s.
15. **Does it reach commercial break-even at $0.15/GPU-hour?** No: 1.0446 tok/s/GPU-equivalent vs 2.78.
16. **What should Experiment 016 physically validate?** Nothing yet. First complete representative DSpark verification and a real resident EP or DCP kernel locally; only then select hardware from the measured topology.
