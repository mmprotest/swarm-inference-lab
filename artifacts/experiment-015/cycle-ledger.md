# Experiment 015 cycle ledger

Failed and stopped branches are retained. Results are not promoted across evidence classes.

| Field | Required evidence |
| --- | --- |
| ID | H015-001A |
| Hypothesis | Pinned public Kimi K3 DSpark can draft deterministic candidate blocks from real target hidden states. Expected: finite deterministic blocks 1/2/3/5/7 and exact repeat |
| Implementation | CPU BF16 public-weight reference plus transactional commit/rollback semantics |
| Benchmark | pinned revisions/hashes; retained real Kimi hidden trace; blocks 1/2/3/5/7 |
| Result | finite deterministic blocks and repeat passed; acceptance/distribution equivalence not established |
| Inspection | real DSpark weights executed; target verification was absent |
| Bottleneck | representative 93-layer target pass cost and no public DSpark pipeline support |
| Decision | MODIFY |
| Redesign | bound target verification before implementing pipeline speculation |

| ID | H015-001B |
| Hypothesis | Block-7 verification can reach at least 5 tok/s with perfect acceptance after boundary reduction. Expected: >=5 dependency-bound tok/s |
| Implementation | shaped target-batch service upper bound with depth-8 cells and zero draft cost |
| Benchmark | immutable batch-8 KDA/MLA timings; perfect 8 outputs/pass |
| Result | 2.1838 tok/s |
| Inspection | target verification widens each pass to batch 8 and raises traversal service time |
| Bottleneck | target block-verification service, not only boundary latency |
| Decision | REVERT |
| Redesign | a faster verification kernel/architecture is required before DSpark can hit 5 tok/s |

| ID | H015-002A |
| Hypothesis | Asynchronous drafting improves ordinary DSpark after the synchronous path passes. Expected: >10% over synchronous DSpark at equal target work |
| Implementation | occupancy/admission contract; no distributed runtime mutation |
| Benchmark | baseline stage-time model and pinned vLLM DSpark pipeline guard |
| Result | NOT ESTABLISHED |
| Inspection | ordinary DSpark acceptance/GPU draft latency gate failed first |
| Bottleneck | missing synchronous baseline and unsupported public PP combination |
| Decision | REVERT |
| Redesign | resume only after H015-001 passes |

| ID | H015-003A |
| Hypothesis | Depth-4/8 cells materially reduce exposed slow-boundary latency. Expected: >=20% dependency speed improvement for depth 8 |
| Implementation | separate internal/coarse domains in a held-out-validated component model with shaped network |
| Benchmark | depth 1/2/4/8, actual 258,048-byte Kimi boundary |
| Result | depth 8: 1.1766 tok/s, 23.8% faster |
| Inspection | compute and 81 internal boundaries remain sequential |
| Bottleneck | 747 ms compute plus internal synchronization |
| Decision | RETAIN |
| Redesign | combine only with a verification method that reduces target passes |

| ID | H015-004A |
| Hypothesis | Direct resident expert dispatch achieves >=90% complete-layer throughput. Expected: >=90% measured relative throughput |
| Implementation | overhead decomposition and real hidden-row transport precision checks |
| Benchmark | immutable 2/4/8/16 worker real Kimi sweep; promoted four-worker repeat |
| Result | 78.81% measured; 95.56% optimistic subtraction bound |
| Inspection | host/network/serialization costs are material but replacement cost was not measured |
| Bottleneck | missing device-resident transport implementation |
| Decision | REVERT |
| Redesign | implement persistent device buffers before another performance claim |

| ID | H015-005A |
| Hypothesis | Expert transport overlap hides a significant fraction of communication. Expected: >=25% exposed communication hidden |
| Implementation | admission surface only |
| Benchmark | retained timing decomposition |
| Result | NOT ESTABLISHED |
| Inspection | no asynchronous send/receive implementation ran |
| Bottleneck | lack of independent resident network/compute streams |
| Decision | REVERT |
| Redesign | requires H015-004 resident buffers first |

| ID | H015-006A |
| Hypothesis | Exact context-shard sufficient statistics reproduce full softmax attention. Expected: relative L2 <=1e-12 for degrees 1/2/4/8 |
| Implementation | stable max/denominator/numerator reduction |
| Benchmark | deterministic 1K synthetic context, uneven-capable shards |
| Result | maximum relative L2 1.119e-15 |
| Inspection | component exactness passed; complete Kimi state path not exercised |
| Bottleneck | no real distributed Kimi MLA kernel |
| Decision | MODIFY |
| Redesign | integrate into persistent MLA worker before capacity claims |

| ID | H015-006B |
| Hypothesis | DCP4+ reduces 8K MLA service by >=25%. Expected: >=25% shaped stage gain |
| Implementation | measured context-scan fit plus actual partial payload network model |
| Benchmark | 1K/4K/8K/16K x DCP1/2/4/8 |
| Result | 8K DCP8 shaped gain 40.47% |
| Inspection | worker cost/capacity and complete-stage correctness remain unknown |
| Bottleneck | distributed combine and paid-compute validation |
| Decision | REVERT |
| Redesign | real Kimi DCP kernel and conditional context gate |

| ID | H015-007A |
| Hypothesis | Only operations larger than their collectives benefit from TP. Expected: >1.0x complete-operation speedup at TP2/4 |
| Implementation | real CUDA TP1 timings plus shaped internal collective |
| Benchmark | KDA q/output, MLA projection group, LM head; TP1/2/4 |
| Result | LM head TP4 sensitivity 1.63x; KDA/MLA candidates did not win |
| Inspection | isolated head gain is small in the 1,052 ms model path |
| Bottleneck | collective overhead and lack of full-stage implementation |
| Decision | REVERT |
| Redesign | retain LM-head TP as a later endpoint-only candidate |

| ID | H015-008A |
| Hypothesis | Measured load placement reduces critical expert worker demand. Expected: >=10% held-out critical-load reduction |
| Implementation | static modulo and greedy trace-fitted placement |
| Benchmark | three real tokens across 92 MoE layers |
| Result | 10.16% in-sample gain; no held-out trace |
| Inspection | fit and evaluation reused the same tiny trace |
| Bottleneck | representative route history |
| Decision | REVERT |
| Redesign | collect multi-workload routes before replication/migration |

| ID | H015-009A |
| Hypothesis | Previous-token routes predict enough experts to hide dispatch. Expected: >=70% precision and recall |
| Implementation | same-layer previous-token top-16 predictor |
| Benchmark | 184 transitions in retained route trace |
| Result | precision/recall 17.09% |
| Inspection | 82.91% of predicted activation bytes were wasted |
| Bottleneck | low temporal route overlap |
| Decision | REVERT |
| Redesign | do not train a learned predictor until representative traces exist |

| ID | H015-010A |
| Hypothesis | 25-75% expert residency improves paid-GPU efficiency without decode collapse. Expected: positive tok/s/$ after H2D misses |
| Implementation | cold-start per-layer LRU replay and transfer accounting |
| Benchmark | 100/75/50/25% residency over three target tokens |
| Result | NOT ESTABLISHED |
| Inspection | three tokens cannot estimate steady-state hit rate or throughput |
| Bottleneck | representative route locality and real H2D overlap |
| Decision | REVERT |
| Redesign | collect long traces before whole-expert or partial-expert residency |

| ID | H015-011A |
| Hypothesis | DSpark plus microcells exceeds both primary targets. Expected: >=5 tok/s and >=2.78 tok/s/GPU-equivalent |
| Implementation | admission-gated architecture model; no unpassed components multiplied |
| Benchmark | A-M search plus perfect-acceptance upper bound |
| Result | perfect block-7/depth-8 bound 2.18 tok/s and 1.04 tok/s/GPU excluding draft |
| Inspection | verification batch service consumes the same rows used by aggregate batch 8 |
| Bottleneck | target verification service and unchanged paid compute |
| Decision | REVERT |
| Redesign | requires fundamentally faster target verification, not another topology-only combination |

| ID | H015-012A |
| Hypothesis | Operation nanobatching improves utilization without violating 5 tok/s cadence. Expected: >10% throughput at dependency >=5 tok/s |
| Implementation | prerequisite gate only |
| Benchmark | immutable safe batch 8 capacity and dependency model |
| Result | NOT EXECUTED; dependency prerequisite failed |
| Inspection | baseline already exploits batch 8 for aggregate capacity |
| Bottleneck | individual dependency path |
| Decision | REVERT |
| Redesign | resume only after a >=5 tok/s architecture exists |

| ID | H015-PRIMARY |
| Hypothesis | A speculative hierarchical sub-layer architecture materially improves both axes. Expected: >=5 tok/s/user and >=2.78 tok/s/GPU-equivalent |
| Implementation | immutable baseline, component gates, held-out validation, shaped topology, admission-gated A-M search |
| Benchmark | all retained local evidence and explicit upper bounds |
| Result | FAIL: best admitted projection 1.1766 tok/s and 1.0446 tok/s/GPU-equivalent |
| Inspection | no speculative/EP/DCP/TP/residency component passed complete implementation gates |
| Bottleneck | multi-token target verification service plus unresolved distributed implementations |
| Decision | REVERT |
| Redesign | do not proceed to physical validation; build a local complete verifier first |
