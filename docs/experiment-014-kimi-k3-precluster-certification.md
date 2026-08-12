<!-- EXPERIMENT-014-FINAL-SUMMARY:BEGIN -->

# Experiment 014 final pre-cluster report

## Verdict

* Experiment 014 pre-cluster certification: **PASS**
* Complete Kimi CUDA graph: **PASS**
* RTX 3090 sm_86 package: **READY** (physical execution remains NOT RUN)
* Canonical persistent Kimi runtime: **PASS**
* Safe certified batch: **8** (first rejected batch: 9, before CUDA)
* Sub-layer microwork: **FUNCTIONAL BUT NOT CURRENTLY ECONOMIC**
* Smallest useful worker VRAM: **8 GiB** for an optional four-way expert partition inside a fast domain
* Recommended fleet size: **93**
* Recommended topology: **WHOLE-LAYER**
* Maximum worker VRAM: **22.609 GiB planned on 24 GiB**
* Minimum worker headroom: **3.791 GiB total; 1.391 GiB remains beyond the 10% safety reserve**
* KDA p50: **2.975 ms** (production batch 1)
* MLA p50: **2.704 ms** (production batch 1)
* Sub-layer distributed layer p50: **4.189 ms**
* Real coarse boundary bytes: **258,048 payload / 258,283 mean wire**
* Real microwork bytes: **289,550.6 mean total / 100,352 critical path per token**
* Required coarse network: **<= 5.0 ms RTT and >= 10.0 Gbps**
* Required microwork network: **<= 0.5 ms RTT and >= 2.5 Gbps**
* Capacity model held-out error: **0.553% local median APE**
* Projected RTX 3090 capacity retention: **39.434% wall**
* Projected aggregate output throughput: **97.150 tok/s**
* Projected per-user decode: **0.950 tok/s** at the admitted coarse edge
* Fleet cost/hour: **$4.65** at $0.05/GPU-hour
* Cost/M output: **$13.296**
* Margin at $15/M: **11.363%**
* Remote distribution: **PASS**
* Bootstrap: **PASS** locally / physical clean node NOT RUN
* Recovery: **PASS**
* Final rehearsal: **PASS**
* Experiment 015 package: **READY**
* Ready to rent GPUs: **YES — rent the single RTX 3090 canary first; full-fleet activation remains locked**

## Required sub-layer summary

* Sub-layer microwork execution: **FUNCTIONAL BUT NOT CURRENTLY ECONOMIC**
* Smallest tested worker footprint: **0.983 GB (0.915 GiB)**
* Fraction of complete layer per smallest microworker: **5.377%**
* Best sub-layer worker count per layer: **4**
* Sub-layer layer-throughput relative to one-GPU baseline: **78.809%**
* Sub-layer capacity retention: **84.153%** at logical batch 8
* Maximum viable microwork RTT: **0.5 ms tested** (0.646 ms exact at 100 Gbps)
* Minimum viable microwork bandwidth: **2.5 Gbps tested** (1.986 Gbps exact at 0.25 ms)
* Expert-routing imbalance: **6:2 hottest:coldest selections (3.0x)** in the retained three-position trace
* Recommended use of microworkers: **ONLY INSIDE FAST DOMAINS; not in the initial fleet**
* Experiment 015 topology: **WHOLE-LAYER**

The initial fleet is economically viable only near the measured low-price case. The modeled break-even GPU price is approximately **$0.056/hour** at full modeled utilization and a $15/M selling price; the $0.08/hour and higher cases lose money. Sequential dependency depth also limits one stream to 0.950 tok/s even though aggregate batch-8 capacity is 97.150 tok/s.


## Final questions answered

1. **Can the complete model execute correctly?** Yes. The exact promoted binary executes all 93 real layers, all 11 CUDA operation classes, final norm, head and sampling with exact routes, stateful decode and maximum full-graph relative L2 error below 1e-6.
2. **Can the first physical cluster test be launched safely?** Yes, beginning with one RTX 3090 canary. The static package cannot activate the fleet without a physical certificate, exact canary ELF, 93 node admissions and the cost guard.
3. **What batching design should be used?** Incremental, fail-closed batch 8. Batch 1/2/4/8 passed on late KDA and MLA stages; batch 9 is the first production rejection and is rejected before CUDA.
4. **Did real Kimi sub-layer microworkers work?** Functionally yes: four independent persistent process partitions executed real selected MXFP4 experts and deterministic reduction exactly. Physical multi-GPU efficiency was not claimed.
5. **Can a worker store less than one complete layer?** Yes. Exact four-way workers held 3.931 GB each; the smallest tested 16-way worker held 0.983 GB.
6. **What is the smallest useful footprint?** The smallest tested footprint is 0.983 GB, but the smallest meaningful deployable class is 8 GiB for a four-way partition plus runtime reserves. It is optional, not economic in the initial fleet.
7. **When is sub-layer work worthwhile?** Only inside a fine domain at <=0.5 ms RTT and >=2.5 Gbps. Even there, the same-GPU exact layer retained 78.809% of the resident baseline, so it buys memory reduction rather than speed.
8. **What is the dominant bottleneck?** Aggregate capacity is limited by row-serial 8K contextual Gated-MLA service. Per-user cadence is limited by 93 sequential stages plus 92 coarse edges; economics is limited by rental price.
9. **How many GPUs should be rented?** Rent one RTX 3090 first. If it passes, admit the remaining nodes for an exact 93-GPU fleet; do not activate a partial or over-price fleet.
10. **What physical topology should be used?** Candidate B, a 93-worker whole-layer pipeline with embedding+dense packed into the first worker and final/head packed into the last. Fine expert groups remain an optional later canary.
11. **What network is required?** Coarse workers require <=5.0 ms RTT and >=10.0 Gbps as one coupled admission rule. Optional fine groups require <=0.5 ms and >=2.5 Gbps.
12. **What throughput is predicted?** 97.150 aggregate output tok/s at batch 8 and 0.950 tok/s for one dependency-bound stream at the admitted coarse edge.
13. **How trustworthy is the model?** Local held-out median error is 0.553%, well inside 10%. RTX 3090 transfer is still model-based rather than held out, so the canary must validate it.
14. **Does the economics work?** At $0.05/GPU-hour, yes narrowly: $13.296/M and 11.363% margin at $15/M. Break-even is about $0.056/GPU-hour; it fails at $0.08/hour and above under this model.
15. **What remains unknown until physical GPUs run?** Actual Linux sm_86 correctness and timing, 5090-to-3090 transfer error, thermal/clock behavior, physical network jitter/contention, full-fleet cadence, and real clean-node provisioning time.
16. **What launches Experiment 015?** Verify `package-lock.json`; install and build the runtime on one 3090; prepare workers 000/089/091/092; run `run-3090-canary.sh`; publish the exact ELF and physical certificate immutably; rent the remainder only if canary, network and <=$0.05/hour cost gates pass; install that qualified ELF on every node; prepare and qualify each assigned stage; exchange public fingerprints; start workers; then run `bind-and-deploy.sh`.

**PRE-CLUSTER CERTIFICATION COMPLETE. Experiment 015 is the full physical Kimi K3 cluster test. No further synthetic architecture experiment is required.**

<!-- EXPERIMENT-014-FINAL-SUMMARY:END -->

# Superseded pre-continuation report (historical)

> The body below is retained as crash-surviving history. Its original top-level FAIL, blockers, node count and manager answers are stale and are superseded by the evidence-backed final section above.

## Verdict

* Experiment 014 pre-cluster certification: **FAIL**
* Full 93-layer Kimi graph supported: **YES — in the correctness-oriented serial engine**
* Full real-weight serial Kimi execution: **PASS**
* Stateful decode semantics: **PASS**
* Canonical persistent tensor runtime: **FAIL**
* RTX 3090 sm_86 package ready: **NO**
* 24 GB worker placement feasible: **NO — modeled feasibility exists, but physical-runtime memory safety is not certified**
* Minimum safe worker count: **N/A — model-only lower bound is 81**
* Recommended Experiment 015 worker count: **N/A — no fleet is approved; the provisional checkpoint-aligned design uses 96**
* Partial worker model distribution: **FAIL — exact local materialization passes, but remote release-based acquisition is incomplete**
* Clean-node bootstrap: **FAIL**
* 73-node logical rehearsal: **N/A — 73 is rejected by the memory model; the 96-node metadata rehearsal passes**
* Capacity model held-out error: **N/A — no validated GPU Kimi capacity model exists**
* Projected capacity retention: **N/A**
* Projected aggregate output throughput: **N/A**
* Projected per-user decode speed: **N/A**
* Projected GPU cost/M output: **N/A**
* Gross margin at $15/M: **N/A**
* Experiment 015 deployment package: **NOT READY**
* Ready to rent physical cluster: **NO**

Question A, “Can we run it?”, is **not yet certified**. The checkpoint mathematics executes end to end, but no production Kimi CUDA stage engine is connected to the distributed runtime and no Kimi-critical `sm_86` package exists.

Question B, “Is it likely to be economically useful?”, is **unanswered**. There are no real Kimi GPU component measurements, held-out service-model validation, or measured Kimi network profiles from which to make a defensible RTX 3090 throughput projection.

### Decisive blockers

1. `third_party/colibri/c/kimi_k3.c` executes the complete Kimi graph through CPU code with a partial Vulkan matrix-multiply path. The generic CUDA backend is not called by the Kimi target, and the Kimi build does not link a production CUDA implementation. The compatibility audit therefore reports 11 Kimi-critical operations as `BLOCKER`, zero as `CERTIFIED_FOR_SM86`.
2. The Swarm runtime cannot instantiate `k3-3090-placement-manifest.json` as real Kimi layer stages. The 96-worker rehearsal validates metadata, identities, routes, ordering, and state ownership with typed placeholders; it is not a production worker execution.
3. Experiment 013 connection persistence was promoted to reusable canonical TCP clients, but delegated execution still creates child tasks per operation and real Kimi activations/experts have not traversed the canonical persistent path.
4. Clean Linux bootstrap, immutable release lock, remote shard acquisition, Kimi-state recovery, scheduler/API lifecycle, real traffic replay, and the validated capacity/economic model remain incomplete.

These are locally solvable software and deployment problems. An RTX 3090 canary cannot legitimately defer them to rented hardware.

## Evidence

### Checkpoint identity and census

The immutable local checkpoint at `F:\Models\Kimi-K3` is authoritative.

| Item | Measured value |
|---|---:|
| Checkpoint revision | `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` |
| Checkpoint semantic fingerprint | `f320c7e1483f06130ca93afc7ac501754498575bb1efd6c62c357a39f8c84737` |
| `config.json` SHA-256 | `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213` |
| Safetensors index SHA-256 | `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd` |
| Safetensors shards | 96 |
| Tensors | 497,220 |
| Required text-generation tensors | 497,052 |
| Unclassified required tensors | **0** |
| Required physical tensor bytes | 1,559,965,606,912 |
| All physical tensor bytes | 1,560,860,324,864 |
| Required logical bytes | 5,644,076,852,224 |
| Transformer layers | 93: 69 KDA, 24 Gated MLA |
| Routed experts / selected experts | 896 / 16 |
| Shared experts | 2 |
| Hidden / latent dimensions | 7,168 / 3,584 |
| Vocabulary / configured context | 163,840 / 1,048,576 |

All required tensor metadata, shapes, dtypes, files, byte ranges, roles, layers, and expert identities are classified in `artifacts/experiment-014/k3-checkpoint-census.json`. Physical SHA-256 recomputation covered 5 of 96 immutable source shards (69,881,452,400 bytes) before it was stopped after the decisive deployment blockers were established; the incomplete receipt is explicit in `k3-checkpoint-integrity.json`. The metadata census passes Gate 1, but this is not a claim that all 1.56 TB was reread and rehashed locally.

### Proof A: full real-weight serial oracle

The serial oracle loaded the real official checkpoint a layer at a time and executed the complete text graph:

`tokenizer -> embeddings -> layers 0..92 -> final norm -> LM head -> logits -> sampler -> next stateful decode`

Measured receipt:

| Check | Result |
|---|---:|
| Real transformer layers executed | 93/93 |
| Input/state positions executed | 3 |
| Hidden-state rows captured | 282 |
| MoE routing calls | 276 |
| Routed expert selections | 4,416 |
| KDA and MLA state transitions changed | 2/2 |
| Logit bytes | 1,310,720 |
| Finite hidden states/logits | PASS |
| Sampled token ID | 11, valid |
| Wall time | 2,073.42 s |
| Hidden trace SHA-256 | `95012ceb...` |
| Routing trace SHA-256 | `4d0fbf...` |
| Logit SHA-256 | `0e681a...` |

This proves native checkpoint MXFP4 weight decoding and complete model connectivity with FP32 reference activations. It does **not** certify the requested production MXFP8 activation path or GPU throughput.

An independent NumPy implementation checked a deterministic real-weight four-layer slice containing three KDA layers, one Gated MLA layer, the dense first layer, and MoE layers. With floating activation reference mode, per-layer relative L2 error was `3.390e-6`, `2.291e-6`, `1.389e-6`, and `1.919e-6`; cosine similarity was approximately one. An earlier integer-dot activation comparison produced worst relative L2 error `4.924e-3`, correctly exposing the activation-quantization approximation rather than hiding it.

Official tokenizer/chat-template comparison found and repaired a real defect: the serving renderer did not preserve Kimi's default typed `thinking_effort=max` semantics. Three single/multi-turn fixtures now match the checkpoint tokenizer token-for-token, including low/default/no-thinking behavior and EOS ID 163586. Tool-call content is deliberately rejected for the initial product rather than silently mishandled.

### Placement result: 73 is falsified

The solver uses a 24 GiB physical device, 3.5 GiB fixed runtime/state reserve, and evaluates 0%, 5%, and 10% headroom. The provisional serving state is context 8,192 with four streams; it is a memory-model input, not a scheduler certification.

| Nodes | 10% headroom result | Tightest remaining bytes |
|---:|---|---:|
| 64 | FAIL | negative |
| 72 | FAIL | negative |
| 73 | **FAIL** | -1,994,629,530 |
| 80 | **FAIL** | -127,244,186 |
| 81 | model-only minimum | non-negative |
| 88 | PASS in model | 1,662,576,742 |
| 96 | PASS in model | 2,417,109,094 |
| 112 | PASS in model | 5,467,104,870 |
| 128 | PASS in model | 7,207,512,166 |

The provisional 96-node layout was selected because it gives an auditable checkpoint-aligned coarse partition: embeddings on worker 0, layers 0–92 on workers 1–93, final residual/norm on worker 94, and LM head on worker 95. Every required tensor is assigned exactly once. This is the strongest current placement candidate, not a safe physical recommendation: production CUDA context, kernel workspace, communication buffers, activation format, state allocation, and fragmentation have not been measured in the missing Kimi GPU engine.

### Proof B: complete logical deployment rehearsal

The generated 96-node plan contains 100 ordered operations and 97 stage transitions. Mechanical equivalence checks pass for serial/distributed operation order, tensor identities, router/reduction semantics, KDA/MLA state transitions, final norm, head, and sampling. The logical rehearsal admitted all 96 placeholder workers and completed two generations with 384 events while exercising all 93 declared state transitions.

The 73-node rehearsal intentionally fails admission because its memory plan is infeasible. Neither rehearsal executes production Kimi kernels or proves physical performance.

### Partial distribution proof

The distribution manifest maps every worker to exact immutable checkpoint files and tensor byte ranges. It specifies revision-pinned official source URLs, source hashes, resumable whole-source-shard acquisition, exact range extraction, corrupted-source rejection, package hashing, atomic activation, and cache reuse.

One real worker package was materialized locally for worker 94: three tensors, 43,008 tensor bytes, 43,520 package bytes, package SHA-256 `fa6bf3ce...`, with the source shard verified before atomic activation. Unit tests also prove corrupted-source rejection. Median and worst cold source download per node are both 16,990,916,912 bytes for the checkpoint-aligned plan. Remote credentials, clean-node fetch, retry against a real object source, and release bootstrap were not exercised, so the end-to-end distribution gate fails.

### sm_86 compatibility audit

The development GPU is an RTX 5090 (`cc 12.0`, 32,607 MiB, driver 591.86, CUDA 13.0), which cannot certify Ampere behavior. Source inspection found:

* the generic Colibri CUDA backend contains an Ampere WMMA target declaration;
* the Kimi engine does not call that backend and its target does not link it;
* the Kimi Vulkan path only offloads a subset of matrix multiplications and native expert operations, leaving KDA, MLA, state, routing, and other graph work on CPU;
* no explicit Kimi `sm_86` binary was produced, and the local Windows environment lacks the MSVC compiler required by the installed CUDA toolchain.

Accordingly, `rtx3090-compatibility-matrix.json` records 11 blockers and zero certified Kimi-critical CUDA operations. This is a software integration blocker, not an unknown RTX 3090 speed result.

### Regression evidence

* Root unit/integration suite: **1,064 passed, 13 skipped** in 187.57 seconds.
* Focused Experiment 014/canonical transport suite: **13 passed**.
* Colibri OpenAI serving suite: **109 passed** in 12.554 seconds.
* Ruff on the Experiment 014 and canonical transport changes: **PASS**.

These regressions establish that the new evidence tooling and persistent connection reuse did not break the tested repository behavior. They do not upgrade the missing production features to PASS.

## Hard-gate ledger

| Gate | Status | Reason |
|---:|---|---|
| 1. Complete checkpoint understood | PASS | 497,052/497,052 required tensors classified; zero unclassified. |
| 2. 93/93 layer support | PASS | Every layer executed in the serial oracle and appears in the support matrix. |
| 3. Full real-weight graph | PASS | All real checkpoint layers, norm, head, logits, and sampling executed. |
| 4. Stateful generation | PASS | A subsequent decode position advanced KDA/MLA state. |
| 5. Component correctness | PASS | Independent real-weight mixed-layer reference passed at documented tolerance. |
| 6. Canonical persistent runtime | **FAIL** | Reusable sockets pass, but per-operation child tasks remain and real Kimi stages are absent. |
| 7. Safe physical placement | **FAIL** | 73 is infeasible; 81/96 are memory-model results without production-runtime allocation measurements. |
| 8. No orphan tensors | PASS | Exact 96-node tensor coverage and ownership check passes. |
| 9. sm_86 package | **FAIL** | Kimi CUDA backend is not wired/built; 11 critical blockers. |
| 10. Partial distribution | **FAIL** | Exact local materialization passes; remote clean-node workflow is untested/incomplete. |
| 11. Clean bootstrap | **FAIL** | No release-based Linux bootstrap reaches READY with real Kimi weights. |
| 12. Exact cluster plan | PASS | Complete machine-readable placement/DAG/topology artifacts exist and are equivalent to serial order. |
| 13. Full logical rehearsal | PASS | The 96-worker metadata plan reaches READY and traverses the full logical DAG. |
| 14. Real tensor traffic | **FAIL** | Boundary payloads were derived, not benchmarked through the production Kimi path. |
| 15. Network requirements | **FAIL** | No measured payload replay/capacity model supports hard rental thresholds. |
| 16. Real Kimi recovery | **FAIL** | Protocol tests exist, but KDA/MLA state recovery and replacement were not proven end to end. |
| 17. Capacity model | **FAIL** | No real Kimi GPU measurements or held-out error result. |
| 18. Economic projection | **FAIL** | Throughput is not validated, so cost/token and margin are intentionally not projected. |
| 19. Experiment 015 preregistered | **FAIL** | A blocked draft exists; performance and network thresholds cannot yet be registered honestly. |
| 20. Deployment package | **FAIL** | The package is an explicit fail-closed blocker stub, not a runnable fleet deployment. |

Result: **8 PASS, 12 FAIL**. Gate details are also machine-readable in `artifacts/experiment-014/acceptance-gates.json`.

## Measured versus modeled versus unknown

Measured evidence includes checkpoint metadata, complete serial execution, independent mixed-layer correctness, tokenizer/chat semantics, local package extraction, logical route coverage, and regression results.

Modeled evidence includes the 81-node lower bound, 96-node provisional layout, per-node source bytes, and memory headroom. These are not physical allocation or performance measurements.

Unknown and deliberately reported as `N/A`: RTX 3090 Kimi kernel speed, production activation format performance, TTFT, decode speed, aggregate throughput, overlap, real Kimi network demand, capacity retention, cost/M tokens, gross margin, context/concurrency frontier, and cold-start duration.

At $0.165/GPU-hour, arithmetic target lines are:

| Fleet | GPU cost/hour | Break-even at $15/M | 50% infrastructure-margin line |
|---:|---:|---:|---:|
| 73 | $12.045 | 223.06 aggregate tok/s | 446.11 aggregate tok/s |
| 96 | $15.840 | 293.33 aggregate tok/s | 586.67 aggregate tok/s |

These are target lines only, not projections.

## Research loops and redesigns

1. **Checkpoint hypothesis:** metadata would match the documented architecture. The census confirmed 93 layers and exact local values, classified every required tensor, and made the local revision authoritative.
2. **Full-graph hypothesis:** streaming could connect all real layers. A complete 2,073-second run passed and advanced state into another decode position.
3. **Independent-correctness hypothesis:** the serial math would match a separately implemented reference. The first integer-dot comparison exposed a `4.924e-3` quantization delta; a float-activation reference isolated that approximation and passed near `1e-6` relative error.
4. **Conversation hypothesis:** the existing renderer matched the official template. Exact token comparison falsified it; two boundary/attribute redesigns produced exact token equality for all three fixtures.
5. **73-worker hypothesis:** the checkpoint would fit with operational reserve. Exact packing falsified 73 and 80; the modeled lower bound moved to 81 and the auditable candidate to 96.
6. **sm_86 hypothesis:** generic CUDA support implied Kimi compatibility. Build-graph and call-site inspection falsified it: the Kimi graph never invokes that CUDA backend.
7. **Persistent-runtime hypothesis:** connection persistence completed Experiment 013 promotion. Reuse tests passed, but inspection found per-operation child task creation and no real Kimi stage binding, leaving the gate failed.

## Continuation cycle ledger

Experiment 014 continues from the fail-closed verdict above. Earlier PASS results are retained and are not repeated unless a production-runtime change requires regression validation.

| Field | H014-025a |
|---|---|
| Hypothesis | At least 6 of the 11 Kimi-critical operation classes can be completed by wiring existing Colibri CUDA primitives, without a new mathematical CUDA kernel. |
| Implementation | Added a reproducible source/API/build audit and emitted `artifacts/experiment-014/k3-cuda-operation-matrix.json`; no execution class was promoted from inspection alone. |
| Benchmark | Audited `kimi_k3.c`, `backend_cuda.cu`, `backend_cuda.h`, the Kimi Make target, the retained RTX 5090 generic-CUDA receipt, and the packaged native-MXFP4 fixture. The retained generic CUDA fixture had relative L2 error `2.708e-7`, but used synthetic int4 weights and is only evidence that the generic backend executes. |
| Result | **FALSIFIED**: 5/11 classes are wiring-only candidates; 6/11 need an adapted or new Kimi kernel; 0/11 are CUDA-ready. Focused generator test: 1 passed. |
| Inspection | Existing CUDA fully covers generic GEMM, router selection, RMSNorm, LM-head-style GEMM, and internal fixed-order reduction pieces. MLA has a reusable causal attention core. The packaged `coli_kimi_mxfp4.dll` is a CPU/OpenMP fixture around `quant.h`, not a CUDA runtime. The retained `coli_cuda.dll` is an `sm_120` development build, not the required `sm_86` package. |
| Bottleneck | Missing Kimi-specific data-plane semantics are concentrated in native MXFP4 plus SiTU, KDA short-convolution/recurrent state, embedding gather, Gated-MLA output gating, and AttnRes mixing. The Kimi target also does not link the CUDA loader. |
| Decision | **RETAIN** the operation-matrix audit; do not treat generic CUDA presence or the CPU MXFP4 fixture as Kimi CUDA coverage. |
| Redesign | H014-025b: extend the existing resident CUDA tensor/expert path with the native Kimi MXFP4 layout and SiTU epilogue, wire only the routed-expert fixture first, then compare one real Kimi expert against the serial oracle with explicit backend identity and transfer timing. |

| Field | H014-025b |
|---|---|
| Hypothesis | The resident CUDA expert path can consume byte-exact Kimi E2M1/UE8M0 group-32 tensors, execute SiTU on device, and match the serial FP32-activation oracle within `3e-4` relative L2 error without a CPU fallback or layout conversion. |
| Implementation | Added CUDA tensor format 7, byte UE8M0 scale accounting, a native MXFP4 dot-product branch, a SiTU kernel, a fail-closed Kimi expert entry point, and a real-checkpoint benchmark using layer 1 expert 0. |
| Benchmark | Real `w1/w2/w3` tensors from `model-00002-of-000096.safetensors`, batch 1, latent 3,584, intermediate 3,072, deterministic FP32 input seed 14025, compiled CPU serial oracle versus direct `sm_120` CUDA on the RTX 5090. |
| Result | **SUPPORTED for arithmetic**: relative L2 `3.904e-7`, cosine `0.9999999999999252`, max absolute error `5.821e-9`; all 17,547,264 tensor bytes remained native packed. The first timing receipt was not retained because detailed event time incorrectly remained zero. |
| Inspection | Direct DLL dispatch made CPU fallback impossible, output fingerprints were stable, and there was no conversion launch or persistent dequantized copy. Cross-runtime environment mutation did not switch telemetry inside the MSVC DLL: every mode behaved as production telemetry. |
| Bottleneck | The arithmetic path was correct; measurement-mode control across Python and the independently linked MSVC C runtime was invalid. |
| Decision | **RETAIN** CUDA MXFP4/SiTU arithmetic and correctness evidence; **INVALIDATE** the first timing receipt. |
| Redesign | H014-025c: replace environment selection with an explicit telemetry-mode ABI and make its invariants part of the benchmark PASS condition. |

| Field | H014-025c |
|---|---|
| Hypothesis | An explicit DLL telemetry control will produce zero minimal counters, production counters without CUDA events, nonzero detailed segment timings, and bit-identical outputs in all modes. |
| Implementation | Added `coli_cuda_kimi_set_telemetry(0/1/2)`, resettable counters, strict harness invariants, and reran the immutable real expert for 100 retained calls per mode after 10 warmups. |
| Benchmark | `artifacts/experiment-014/cuda/h014-025c-real-mxfp4-expert-profiled.json`; same real expert/input/oracle as H014-025b. |
| Result | **SUPPORTED**: minimal calls `0`; production calls `100` with zero event time; detailed calls `100` with `0.10510 ms` kernel time. CUDA correctness remained relative L2 `3.904e-7` with bit-identical outputs across modes. Warm minimal p50/p95/p99 was `0.1783/0.1858/0.1925 ms`. |
| Inspection | Detailed segments were H2D `0.00626 ms`, kernels `0.10510 ms`, D2H `0.03860 ms`, with `0.03319 ms` exposed host/synchronization overhead. Production telemetry changed p50 by `-0.11%` (noise); detailed events added `2.72%`. |
| Bottleneck | Kernel work was 58.9% of detailed wall time and D2H plus exposed host/synchronization was 40.3%; the aggregate kernel trace still did not identify which operation to optimize. |
| Decision | **RETAIN** the explicit telemetry ABI, correctness result, minimal/production timings, and segmented transfer trace. |
| Redesign | H014-025d: instrument gate, up, SiTU and down as separate CUDA-event phases before selecting an optimization. |

| Field | H014-025d |
|---|---|
| Hypothesis | The three MXFP4 GEMMs consume more than 95% of routed-expert kernel time, making SiTU immaterial. |
| Implementation | Added per-phase CUDA event accounting and changed mode selection to a relaxed atomic load so minimal telemetry does not take the statistics mutex. |
| Benchmark | `artifacts/experiment-014/cuda/h014-025d-real-mxfp4-expert-phases.json`; 20 warmups and 200 retained calls per mode on the same immutable real expert. |
| Result | **FALSIFIED**: GEMMs consumed `94.551%`, not more than 95%; SiTU consumed `0.00618 ms` or `5.449%`. Gate/up/down were `0.05032/0.02822/0.02864 ms`; total kernel time was `0.11336 ms`. |
| Inspection | Equal-sized gate and up tensors did not have equal time. The first projection carried an unexplained `~0.0221 ms` penalty and alone represented 44.4% of aggregate kernel time. Minimal p50 was `0.1800 ms`; production telemetry overhead was `1.19%`; detailed overhead was `7.83%`. |
| Bottleneck | The dominant uncertainty is whether the first-projection penalty is launch/stream position or gate-specific data behavior; optimizing SiTU first would attack the smaller term. |
| Decision | **RETAIN** phase instrumentation and the native path; do not yet implement a broad MXFP4 rewrite. |
| Redesign | H014-025e: reverse gate/up launch order without changing mathematics. If the penalty follows first position, test a fused gate/up launch; if it remains gate-specific, inspect data/cache behavior instead. |

| Field | H014-025e |
|---|---|
| Hypothesis | The `~0.022 ms` first-projection penalty is caused by launch/stream position rather than gate tensor data; reversing gate/up order will move the penalty to up while preserving output. |
| Implementation | Added a research-only projection-order control and logical phase attribution; no mathematical operation, weight layout, or reduction order changed. |
| Benchmark | Fresh-process gate-first and up-first runs, each with 20 warmups and 200 retained calls per telemetry mode; artifacts `h014-025e-gate-first.json` and `h014-025e-up-first.json`. |
| Result | **SUPPORTED**: gate changed `0.05194→0.02818 ms`; up changed `0.02822→0.04954 ms`. The penalty followed first position. Total kernel changed `0.11498→0.11254 ms`, minimal p50 `0.18145→0.18000 ms`, and output fingerprints were identical. |
| Inspection | The logical tensor placed second consistently took about `0.0282 ms`; the first launch took `0.0495–0.0519 ms`. Tensor identity explains neither the asymmetry nor most of the total. |
| Bottleneck | A launch/stream-position penalty inflates the first of two independent MXFP4 projections; rewriting tensor layout is not supported by this evidence. |
| Decision | **RETAIN** the diagnostic control; keep production order unchanged until a fused launch is benchmarked. |
| Redesign | H014-025f: fuse gate/up into one MXFP4 kernel launch that shares activation reads. Require at least 15% lower combined phase and warm p50 with unchanged numerical thresholds, otherwise leave it disabled. |

| Field | H014-025f |
|---|---|
| Hypothesis | A fused gate/up MXFP4 launch will reduce both the combined projection phase and warm end-to-end p50 by at least 15%, with unchanged correctness. |
| Implementation | Added a native paired MXFP4 kernel that keeps each dot-product reduction order, shares activation reads, and replaces two projection launches with one; benchmark switch remained explicit during comparison. |
| Benchmark | Same-binary unfused/fused runs, each with 50 warmups and 500 retained calls per mode; artifacts `h014-025f-unfused.json` and `h014-025f-fused.json`. |
| Result | **FALSIFIED at the stage threshold**: combined projection improved `15.09%`, but minimal p50 improved only `6.23%` (`0.17885→0.16770 ms`). P95 improved `7.90%`, p99 `20.18%`, total kernel `9.80%`; output fingerprints and relative L2 `3.904e-7` were unchanged. |
| Inspection | Fused pair `0.06638 ms`, SiTU `0.00479 ms`, down `0.02868 ms`, total kernel `0.10193 ms`. Detailed D2H was `0.03964 ms` and exposed host/synchronization `0.03507 ms`, limiting the wall benefit. |
| Bottleneck | After fusion, host boundary and synchronization cost is material relative to kernel time; further SiTU work targets less than 3% of minimal stage wall. |
| Decision | **RETAIN** fusion as the production default because all retained percentiles improved without correctness loss, while recording the preregistered 15% end-to-end claim as false. |
| Redesign | H014-025g: add a device-resident Kimi expert entry point and measure chained stage execution without H2D/D2H per operation; retain it only if persistent semantics eliminate the measured boundary cost. |

| Field | H014-025g |
|---|---|
| Hypothesis | A device-pointer Kimi expert entry point with resident weights, scratch, input, and output eliminates the measured per-operation boundary cost, improving synchronized p50 by at least 15% and queued service time by at least 25%. |
| Implementation | Added a fail-closed device-resident expert ABI with no internal transfer or synchronization, then extended the real-weight harness with one-time upload/download, synchronized-generation, and queued-service measurements. |
| Benchmark | `artifacts/experiment-014/cuda/h014-025g-resident-expert.json`; 50 warmups and 500 retained batch-1 calls against the same layer 1 expert 0 oracle. |
| Result | **SUPPORTED**: synchronized p50 `0.0818 ms`, `50.81%` below host-bound `0.1663 ms`; queued service `0.071743 ms/call`, `56.86%` lower. Resident output was bit-identical to host CUDA and relative L2 remained `3.904e-7`. |
| Inspection | Per-call H2D and D2H were both zero; only a 14,336-byte one-time input upload and validation download occurred. Synchronized p95/p99 were `0.0828/0.0847 ms`. |
| Bottleneck | The host boundary was confirmed as a major component-fixture bottleneck. The remaining expert service term is device compute plus explicit generation synchronization; the full Kimi worker lifecycle remains unbound. |
| Decision | **RETAIN** the device-resident ABI and promote it as the required data plane for canonical persistent stages. This passes routed-expert local CUDA semantics but not the broader persistent-worker gate. |
| Redesign | H014-025h: cross-build this exact native path for `sm_86` with embedded capability metadata, inspect cubin/PTX and exports, and execute its PTX fallback on the local RTX 5090 against the immutable real fixture. |

| Field | H014-025h |
|---|---|
| Hypothesis | The retained native MXFP4/SiTU path can be packaged as an `sm_86` cubin with `compute_86` PTX, reject pre-`sm_86` devices, and execute the same real-weight fixture through forward PTX on the local RTX 5090. |
| Implementation | Added explicit minimum-capability/forward-PTX metadata, fail-closed capability negotiation, parameterized `sm_86` build output, and a machine-readable binary audit. |
| Benchmark | `h014-025h-sm86-package-final.json`, `rtx3090-sm86-certification.json`, and the packaged `coli_cuda-sm86.dll`; 20 warmups and 100 retained calls on the immutable layer 1 expert 0 fixture. |
| Result | **SUPPORTED for routed expert**: current 603,648-byte binary SHA-256 `91c665a3...324e3d`; `sm_86` cubin and `compute_86` PTX present; pre-`sm_86` rejected; real-weight relative L2 `3.904e-7`; resident p50 `0.0777 ms`. Global certificate remained **FAIL**. |
| Inspection | `cuobjdump` found `quant_matmul`, `mxfp4_matmul_pair`, and `kimi_situ_mul` in the required artifact contract. The package executed through forward PTX on the RTX 5090 with no CPU fallback. This is build/package evidence for Ampere, not a physical RTX 3090 timing claim. |
| Bottleneck | Ten Kimi-critical operation classes still lacked complete real-weight and package evidence; physical `sm_86` behavior remains an Experiment 015 single-node-canary measurement. |
| Decision | **RETAIN** the `sm_86`+PTX package contract and routed-expert certification; do not promote the global CUDA or RTX 3090 gates. |
| Redesign | H014-025i: bind the existing resident router to real Kimi weights, correction bias, and retained Kimi activation before deciding whether its generic implementation is usable. |

| Field | H014-025i |
|---|---|
| Hypothesis | The existing CUDA router can consume real Kimi router weights and correction bias, reproduce exact top-16 IDs, and keep selected-weight relative L2 below `1e-6` without CPU fallback. |
| Implementation | Added byte-range BF16 safetensor loading, a resident CUDA router binding, the exact Kimi top-16 CPU oracle, and a layer 1 benchmark using retained activation `x0.f32`. |
| Benchmark | `h014-025i-real-router.json`; dimensions `896 x 7168`, top 16, 50 warmups, 500 retained calls. |
| Result | **SUPPORTED correctness, PERFORMANCE BLOCKED**: exact selected IDs, selected-weight relative L2 `5.161e-8`; p50/p95/p99 `4.1092/4.2566/4.3204 ms`. |
| Inspection | Arithmetic and Kimi parameter wiring were correct, but source inspection found the exact top-16 scan running on one CUDA thread. Total latency was disproportionate to the 25.7 MB resident matrix. |
| Bottleneck | Serial selector suspected, but its contribution versus GEMV and D2H was not measured. |
| Decision | **RETAIN** the fixture and binding; withhold performance certification. |
| Redesign | H014-025j: add phase events and require evidence that selection consumes more than 80% before changing it. |

| Field | H014-025j |
|---|---|
| Hypothesis | The single-thread exact top-16 selector consumes more than 80% of real Kimi router CUDA time. |
| Implementation | Added detailed CUDA-event counters for logits, selection, and D2H without changing minimal/production routing. |
| Benchmark | `h014-025j-router-profile.json`; 50 warmups, 500 retained calls, 100 detailed-profile calls. |
| Result | **SUPPORTED**: selector `3.94640 ms/call`, or `94.96%`; logits `0.08054 ms`; D2H `0.12889 ms`; p50 `4.0927 ms`; routing remained exact. |
| Inspection | The measurement localized the problem to the O(E*K) serial scan, not router weight layout or GEMV. |
| Bottleneck | Exact top-16 selection on one CUDA thread. |
| Decision | **RETAIN** phase evidence; replace only the selector. |
| Redesign | H014-025k: use a cooperative deterministic reduction, requiring at least 80% lower selector time and wall p50 with identical routing. |

| Field | H014-025k |
|---|---|
| Hypothesis | A cooperative deterministic selector preserves exact IDs/weights while reducing both selector time and wall p50 by at least 80%. |
| Implementation | Replaced the serial selector with a 256-thread block using explicit choice-descending/expert-id-ascending comparison and fixed-order postprocessing. |
| Benchmark | `h014-025k-router-parallel.json` against H014-025j; same fixture, 50 warmups, 500 retained calls, 100 profiled calls. |
| Result | **SUPPORTED**: exact IDs; selected-weight relative L2 `5.161e-8`; selection `3.94640 -> 0.03528 ms` (`99.11%` reduction); p50 `4.0927 -> 0.0961 ms` (`97.65%` reduction); p95/p99 `0.09981/0.14115 ms`. |
| Inspection | Post-redesign logits/selection/D2H were `0.04683/0.03528/0.04791 ms`; the router is no longer dominated by serialized selection. |
| Bottleneck | The host-returning fixture is now mixed across GEMV, deterministic selection, and the synchronous 132-byte result readback. |
| Decision | **RETAIN** the parallel selector as production behavior. |
| Redesign | H014-025l: certify the router kernels in the exact `sm_86` cubin/PTX artifact, then advance to a real production-format grouped-int4 dense projection. |

| Field | H014-025l |
|---|---|
| Hypothesis | The exact binary exercised by the passing real router fixture contains both router kernels in its `sm_86` cubin/`compute_86` PTX package and satisfies the routed-expert capability contract. |
| Implementation | Extended certification to consume operation-specific retained evidence, demand matching binary hashes/capability metadata, inspect router PTX entries, and promote only fully evidenced rows. |
| Benchmark | `h014-025k-router-parallel.json`, `rtx3090-sm86-certification.json`, and `k3-cuda-operation-matrix.json`; current binary SHA-256 `91c665a3...324e3d`. |
| Result | **SUPPORTED for router**: `pipe_router_logits` and `pipe_router_select` are present in the required package; exact routing and error gates passed. CUDA matrix/certificate now show 2 certified and 9 blocked classes; global status remains **FAIL**. |
| Inspection | Router and routed-expert receipts identify the same binary, and capability negotiation remains fail-closed below `sm_86`. No class was promoted from source presence alone. |
| Bottleneck | Nine operation classes remain; lifecycle binding is deferred until P0 local coverage is complete. |
| Decision | **RETAIN** router as `CUDA_READY/CERTIFIED`; keep the global gates failed. |
| Redesign | H014-025m: test the existing grouped-int4 GEMV on a real Kimi dense projection in the production quantized representation. |

| Field | H014-025m |
|---|---|
| Hypothesis | The resident grouped-int4 CUDA GEMV consumes Kimi's exact int4-g64 load representation and matches an independent production-quantized oracle within `1e-5` relative L2 without CPU fallback. |
| Implementation | Added production-equivalent int4-g64 quantization, independent dequantized and source-BF16 oracles, a resident GEMV binding, and isolated kernel profiling. |
| Benchmark | `h014-025m-real-dense-int4.json`; real layer 1 `f_a_proj`, `[128,7168]`, 50 warmups, 500 retained calls. |
| Result | **SUPPORTED**: CUDA relative L2 `4.141e-7`, cosine `0.999999999999915`; p50/p95/p99 `0.0163/0.0374/0.0402 ms`; kernel `0.01333 ms`; resident weights 516,096 bytes. |
| Inspection | Production int4 versus source BF16 was separately `0.11427` relative L2; that is quantization loss, not CUDA execution error. The microprojection reached only `38.72 GB/s`, insufficient to characterize large projection behavior. |
| Bottleneck | Suspected small-workload underutilization. |
| Decision | **RETAIN** correctness path, but require a large real fixture before certification. |
| Redesign | H014-025n: require at least 4x higher effective bandwidth on a production-size real projection with unchanged fidelity. |

| Field | H014-025n |
|---|---|
| Hypothesis | A real `[12288,7168]` projection increases effective bandwidth at least 4x over H014-025m if the small result is underfill, while preserving the `1e-5` execution gate. |
| Implementation | Reused the immutable harness with real layer 1 `q_proj`; no kernel change. One CLI-label plumbing failure produced no benchmark and was fixed before retention. |
| Benchmark | `h014-025n-real-dense-large.json`; 20 warmups and 100 retained calls. |
| Result | **SUPPORTED**: CUDA relative L2 `3.832e-7`; p50/p95/p99 `0.1024/0.1040/0.1115 ms`; kernel `0.09871 ms`; 49,545,216 resident bytes; `501.92 GB/s`, a `12.96x` increase. |
| Inspection | Kernel time is `96.40%` of synchronized p50. The size response supports a memory-traffic-dominated large decode GEMV, while the small fixture was underfilled. The exact current package regressed routed-expert/router receipts and retained all correctness gates. |
| Bottleneck | Large grouped-int4 projection weight traffic; physical RTX 3090 bandwidth remains a canary measurement. |
| Decision | **RETAIN** generic grouped-int4 CUDA and promote `dense_projection` to `CUDA_READY/CERTIFIED` using both fixtures. Matrix/certificate: 3 certified, 8 blocked, global **FAIL**. |
| Redesign | H014-025o: test final RMS normalization with real Kimi weights and activation through the resident generic CUDA primitive. |

| Field | H014-025o |
|---|---|
| Hypothesis | Resident generic CUDA RMSNorm matches serial FP32-activation final normalization with real Kimi scale weights at relative L2 `<=2e-6`. |
| Implementation | Added a real final-weight/activation fixture and a reusable default-stream CUDA event timer. |
| Benchmark | `h014-025o-real-final-norm.json`; dimension 7,168; 50 warmups, 500 retained calls, 100 individually profiled calls. |
| Result | **SUPPORTED correctness; TIMING INVALIDATED**: relative L2 `8.901e-8`, max absolute `2.384e-7`, cosine `0.9999999999999994`; wall p50 `0.0213 ms`. |
| Inspection | Per-call event time `0.01994 ms` contradicted minimally instrumented queued service `0.01242 ms`; measurement overhead materially determined the apparent kernel result. |
| Bottleneck | Instrumentation perturbation, before operation classification. |
| Decision | **RETAIN** arithmetic and **INVALIDATE** per-call event timing. |
| Redesign | H014-025p: time 500 launches inside one event interval; predict more than 30% per-call-event perturbation and agreement with minimal queued service within 10%. |

| Field | H014-025p |
|---|---|
| Hypothesis | Per-call CUDA events inflate norm service by more than 30%; batched-event service agrees with minimal queued service within 10%. |
| Implementation | Added batched-event measurement and an explicit retained-timing contract; per-call events remain diagnostic only. |
| Benchmark | `h014-025p-final-norm-telemetry.json`; 50 warmups, 500 retained calls, 100 individual event samples, 500 calls inside one event pair. |
| Result | **SUPPORTED**: per-call event p50 `0.020064 ms`; batched service `0.012317 ms`; minimal queued service `0.012325 ms`; perturbation `62.90%`; batched/minimal difference `0.07%`; relative L2 unchanged at `8.901e-8`. |
| Inspection | Synchronized wall p50 `0.0213 ms` leaves roughly `0.009 ms` exposed host/synchronization beyond the batched device-service term. This is a mixed tiny-kernel/launch/synchronization operation, not a saturated bandwidth result. |
| Bottleneck | Tiny batch-1 work plus launch and generation synchronization. |
| Decision | **RETAIN** batched service, reject per-call timing, promote `final_norm` to `CUDA_READY/CERTIFIED`. Matrix/certificate: 4 certified, 7 blocked, global **FAIL**. |
| Redesign | H014-025q: test token embedding gather from real Kimi BF16 weights through a resident custom CUDA path with explicit one-row transfer accounting. |

| Field | H014-025q |
|---|---|
| Hypothesis | A fully resident BF16 Kimi embedding table gathers actual and boundary token IDs on CUDA with bit-exact FP32 output and no warm host transfer. |
| Implementation | Added BF16 tensor format 8, resident device-ID gather, exact range hashing, and a six-ID fixture including generated token 11 and special/boundary IDs. |
| Benchmark | `h014-025q-real-embedding.json`; full `[163840,7168]` table, 50 warmups, 500 calls. |
| Result | **FALSIFIED by implementation defect**: 2,348,810,240 resident bytes were correct, but CUDA output was non-finite and timing was invalid. |
| Inspection | The generic activation uploader coerced int32 IDs to float32; the kernel read their float bit patterns as out-of-range IDs and emitted NaNs. Table layout and ID bounds were not the cause. |
| Bottleneck | Host adapter dtype coercion. |
| Decision | **RETAIN** BF16 storage/kernel; reject the adapter and timing. |
| Redesign | H014-025r: preserve token-ID bytes and demand bit-exact output without changing the kernel. |

| Field | H014-025r |
|---|---|
| Hypothesis | Byte-preserving int32 upload eliminates H014-025q's NaNs and makes all six rows bit exact. |
| Implementation | Added a dtype-preserving device upload and used it only for token IDs. |
| Benchmark | `h014-025r-real-embedding-byte-upload.json`; identical table/IDs/call counts. |
| Result | **SUPPORTED**: bit exact, relative L2 `0`; p50/p95/p99 `0.0101/0.0286/0.03275 ms`; batched device service `0.007293 ms`; exact resident bytes `2,348,810,240`. |
| Inspection | Warm weight loads, H2D, and D2H are zero. The one-time upload was `806.84 ms` (`2.91 GB/s`) and belongs to LOAD, not warm EXECUTE. |
| Bottleneck | One-time table materialization/upload; warm gather is launch-scale. |
| Decision | **RETAIN** resident BF16 embedding and byte-preserving control-data upload. |
| Redesign | H014-025s: test the real production-int8 LM head against retained full-model activation/logits. |

| Field | H014-025s |
|---|---|
| Hypothesis | Resident int8 CUDA GEMV reproduces retained 163,840-way serial logits within `3e-4` relative L2 and preserves argmax token 11. |
| Implementation | Added streamed production-equivalent per-row int8 quantization and used retained final-normalized trace row 187 (fingerprint `a7b4ce8a...`) with retained logits row 1. |
| Benchmark | `h014-025s-real-lm-head.json`; `[163840,7168]`, 20 warmups, 100 retained calls. |
| Result | **SUPPORTED**: CUDA/serial relative L2 `2.885e-7`; CUDA/independent quantized `2.193e-7`; argmax `11 == 11`; p50/p95/p99 `1.6279/1.6393/1.6426 ms`; resident bytes `1,175,060,480`. |
| Inspection | Production int8 versus source BF16 contributed `0.004912` relative L2. Batched device service was `1.6172 ms` and effective weight bandwidth `726.61 GB/s`, consuming nearly all wall p50. |
| Bottleneck | LM-head weight traffic. |
| Decision | **RETAIN** generic resident int8 GEMV for LM head; package promotion awaits current-binary recertification. |
| Redesign | H014-025t: adapt the resident Kimi SiTU expert path to real grouped-int4 shared-expert weights and compare the complete shared-expert MLP. |

| Field | H014-025t |
|---|---|
| Hypothesis | The resident Kimi SiTU expert path can reuse generic grouped-int4 CUDA GEMV for real shared-expert weights and match the production-quantized oracle within `1e-5` relative L2. |
| Implementation | Extended the resident expert ABI to accept int4-g64 weights while preserving routed-expert MXFP4-g32 behavior, then bound all three real layer-1 shared-expert projections to an independent quantized oracle. |
| Benchmark | `h014-025t-real-shared-expert.json`; `[7168 -> 6144 -> 7168]`, 20 warmups, 100 retained calls, exact `sm_86` + `compute_86` PTX binary SHA-256 `760d84c3...633bdc`. |
| Result | **SUPPORTED**: CUDA/production-quantized relative L2 `6.356e-7`, cosine `0.9999999999997993`; p50/p95/p99 `0.1688/0.170705/0.1794 ms`; resident bytes `74,317,824`. |
| Inspection | Batched device service `0.157748 ms` and minimal queued service `0.157774 ms` agree within `0.02%`; effective weight bandwidth is `471.12 GB/s`. Production int4 versus source BF16 is separately `0.17920` relative L2, so quantization policy—not CUDA arithmetic—is the semantic-risk term. |
| Bottleneck | Grouped-int4 weight traffic. The large quantization delta is not an execution-path defect and must remain visible in whole-graph comparison. |
| Decision | **RETAIN** the shared-expert adapter; package promotion waits for same-binary certification. |
| Redesign | H014-025u: test deterministic real MoE reduction on CUDA using real router weights/IDs and real expert outputs before entering stateful attention work. |

| Field | H014-025u |
|---|---|
| Hypothesis | The existing fixed-order resident CUDA reduction reproduces the weighted sum of 16 actual Kimi routed-expert outputs within `2e-6` relative L2 and is deterministic. |
| Implementation | Exported Colibri's existing `weighted_sum_rows` kernel through a fail-closed resident ABI. The fixture derived the real layer-1 top-16 route, formed the real routed latent input, and evaluated all 16 selected real MXFP4 experts with the independent native oracle. |
| Benchmark | `h014-025u-real-moe-reduction.json`; actual expert IDs `878,704,325,513,468,805,788,187,83,142,412,533,84,263,564,144`; `[16,3584]`; 50 warmups, 500 retained calls. |
| Result | **SUPPORTED**: exact routing; reduction relative L2 `7.120e-8`; max absolute error `4.657e-10`; repeated output bit exact; p50/p95/p99 `0.0121/0.063545/0.100143 ms`. |
| Inspection | Batched device service `0.007649 ms` and queued service `0.007856 ms` agree within `2.7%`; synchronized tail is host/driver noise. The 244 KB operation reaches only `31.87 GB/s`, confirming underfill rather than a bandwidth ceiling. |
| Bottleneck | Kernel launch and generation synchronization for a small fixed-order vector reduction. |
| Decision | **RETAIN** the existing deterministic reduction and resident wrapper; defer matrix promotion until stable same-binary certification. |
| Redesign | H014-025v: audit and test the real AttnRes attention/residual state transition shared by KDA and MLA layers. |

| Field | H014-025v |
|---|---|
| Hypothesis | A custom resident Kimi AttnRes score/softmax/mix kernel plus existing RMSNorm reproduces retained full-model decode trace row 281 within `2e-6` relative L2. |
| Implementation | Reconstructed generation step 2 from token-11 embedding, eight real block snapshots, the real final prefix, and actual output AttnRes/final-norm weights; implemented only the missing score/softmax/mix primitive. |
| Benchmark | `h014-025v-real-attnres.json`; `[8,7168]` snapshots plus prefix; 50 warmups, 500 retained calls. |
| Result | **SUPPORTED correctness**: fixture/serial `4.989e-8`; CUDA mix/reference `4.960e-8`; CUDA final/serial `1.612e-8`; repeated output bit exact; p50/p95/p99 `0.1033/0.106105/0.1075 ms`. |
| Inspection | AttnRes mix is `0.080126 ms` of `0.092505 ms` combined device service (`86.6%`). Source inspection shows nine candidate reductions serialized inside the first correct kernel. |
| Bottleneck | Sequential candidate reductions and repeated whole-block barriers. |
| Decision | **RETAIN** semantics/fixture; **MODIFY** kernel scheduling before promotion. |
| Redesign | H014-025w: compute candidates with one warp each; require `>=40%` lower mix service without weakening correctness or repeatability. |

| Field | H014-025w |
|---|---|
| Hypothesis | One-warp-per-candidate scoring lowers AttnRes mix service at least `40%` while preserving correctness and bit repeatability. |
| Implementation | Replaced nine sequential 256-thread block reductions with nine simultaneous warp reductions; softmax and output mixing were unchanged. |
| Benchmark | `h014-025w-attnres-warp-parallel.json` against H014-025v; identical retained decode fixture and call counts. |
| Result | **FALSIFIED speed gate**: correctness remained `1.612e-8` relative L2 and bit exact across repeats, but device mix fell only `0.080126 -> 0.067488 ms` (`15.77%`, required `>=40%`). |
| Inspection | Candidate scheduling removed only `12.64 us`; the kernel still performs roughly 129k FP64 multiply-adds. On this consumer GPU that arithmetic, not serialization, is now dominant. |
| Bottleneck | FP64 score accumulation throughput. |
| Decision | **MODIFY** accumulator precision; retain the warp mapping but not this performance result. |
| Redesign | H014-025x: use FP32 warp accumulation; require `>=40%` lower mix time than H014-025w while preserving `<=2e-6` serial error and repeatability. |

| Field | H014-025x |
|---|---|
| Hypothesis | FP32 warp accumulation lowers AttnRes mix service at least `40%` versus H014-025w while retaining `<=2e-6` serial error and bit repeatability. |
| Implementation | Changed only score mean-square/dot accumulation and square root from FP64 to FP32 in the warp-parallel kernel. |
| Benchmark | `h014-025x-attnres-fp32-warp.json` against H014-025w; identical retained decode state and 500 calls. |
| Result | **SUPPORTED**: mix `0.067488 -> 0.018434 ms` (`72.69%` lower); combined AttnRes+norm `0.030801 ms`; CUDA/serial relative L2 `1.612e-8`; repeated output bit exact; p50/p95/p99 `0.0388/0.0429/0.057212 ms`. |
| Inspection | A one-line precision-class change removed most remaining time while moving relative error only about `2.5e-12`; this confirms FP64 throughput as the prior bottleneck. |
| Bottleneck | Post-redesign service is launch/synchronization scale; defer fusion until a full-layer profile makes it material. |
| Decision | **RETAIN** FP32 warp-parallel AttnRes; same-binary package promotion remains pending. |
| Redesign | H014-025y: audit existing CUDA projection, convolution, normalization, and recurrent-state primitives against real KDA semantics. |

| Field | H014-025y |
|---|---|
| Hypothesis | One custom resident CUDA kernel can cover only the missing depthwise-convolution/recurrent KDA core and match a three-step production-quantized oracle within `2e-5` relative L2. |
| Implementation | Derived real layer-1 projected inputs independently, then fused only causal conv windows, q/k normalization, alpha/beta, 96 recurrent state updates, per-head output norm, and full-rank gate. Eight linear transforms stayed outside this custom kernel. |
| Benchmark | `h014-025y-real-kda-core.json`; three consecutive inputs, 20 warmups, 100 retained stateful calls. |
| Result | **SUPPORTED**: per-step errors `2.821e-7/2.138e-7/2.902e-7`; state error `1.421e-7`; convolution windows bit exact; reset replay bit exact; p50/p95/p99 `0.0511/0.05231/0.060003 ms`. |
| Inspection | Device and queued service are `0.043131/0.043138 ms`. Updating 6.29 MB recurrent plus 0.59 MB conv state reaches about `583.47 GB/s` under the four-touch state model. |
| Bottleneck | Persistent recurrent-state memory traffic. |
| Decision | **RETAIN** the custom stateful core. |
| Redesign | H014-025z: wire all eight real projections through existing resident GEMV and benchmark the complete KDA attention stage. |

| Field | H014-025z |
|---|---|
| Hypothesis | Eight existing resident GEMVs plus the retained state core reproduce a complete real Kimi KDA attention stage over three stateful steps within `3e-5` relative L2. |
| Implementation | Wired q/k/v/g/output through int4-g64 CUDA GEMV and f_a/f_b/b through FP32 CUDA GEMV; activations and all state remain resident. |
| Benchmark | `h014-025z-real-kda-stage.json`; three consecutive steps; 20 warmups, 100 retained calls; nine kernels/call. |
| Result | **SUPPORTED**: per-step errors `9.507e-7/4.931e-7/5.211e-7`; state `2.906e-7`; replay bit exact; p50/p95/p99 `0.8428/0.853555/0.875014 ms`; device `0.829307 ms`. |
| Inspection | Seven input projections `0.610413 ms` (`73.6%`), core `0.043965 ms` (`5.3%`), output `0.095365 ms` (`11.5%`). Their isolated sum is `0.749743 ms`, leaving a measured `0.07956 ms` cross-phase/cache/scheduling term. |
| Bottleneck | Projection weight traffic plus an unresolved 9.6% cross-phase term. |
| Decision | **RETAIN** complete KDA correctness/wiring; profile before optimizing. |
| Redesign | H014-025aa: attribute individual unchanged projections and at least `90%` of complete device service before considering fusion. |

| Field | H014-025aa |
|---|---|
| Hypothesis | Individual unchanged projection profiles plus the KDA core attribute at least `90%` of complete KDA device service. |
| Implementation | Added per-matrix batched event profiles without changing arithmetic and kept production-order grouped measurements in the same run. |
| Benchmark | `h014-025aa-kda-projection-profile.json`; same complete layer-1 fixture and current binary as H014-025z. |
| Result | **FALSIFIED**: isolated matrices plus core attribute only `68.84%` (`0.57057/0.82885 ms`). Seven input matrices cost `0.60992 ms` in production order but only `0.43130 ms` summed in isolated loops, a `41.41%` understatement. |
| Inspection | Repeating one 3-50 MB matrix makes it cache-warm; cycling the 260.5 MB stage working set does not. Full interleaving adds another `0.07966 ms` beyond grouped phase sums. The isolated profile is therefore not capacity-model evidence. |
| Bottleneck | Cold/cycling projection-weight traffic, not a single slow mathematical primitive. |
| Decision | **RETAIN** production-order KDA timing and current implementation; reject isolated profiles and do not fuse merely to reduce launch count. |
| Redesign | H014-025ab: audit Gated MLA against existing absorb-attention, RMSNorm, projection, cache, and gating CUDA primitives. |

| Field | H014-025ab |
|---|---|
| Hypothesis | Existing CUDA GEMV, RMSNorm, and absorb attention plus minimal cache/gate adapters reproduce a three-step real-weight Gated MLA stage within `3e-5`, with a bit-exact NoPE cache check against the independent oracle. |
| Implementation | Reused all six int8 GEMVs, generic RMSNorm, and existing absorb-attention. Added asynchronous latent/NoPE cache wiring, an unsynchronized resident attention entry point, and only the missing sigmoid gate. |
| Benchmark | `h014-025ab-real-mla-stage.json`; real layer-3 weights and three real serial hidden rows; 20 warmups, 100 retained calls. |
| Result | **FALSIFIED VALIDATION GATE**: output relative L2 `5.879e-7`, latent cache `2.736e-7`, rope cache `1.609e-7`, replay bit exact, but CPU-reference rope bytes were not bit identical. Device service was `0.539272 ms`. |
| Inspection | The cache-copy gate compared bytes produced by two different projection reductions. A D2D copy must match its CUDA source exactly; an independent CPU projection should be compared numerically. The runtime mathematics passed. |
| Bottleneck | Observation provenance, not MLA execution. |
| Decision | **MODIFY** only the cache observation; retain the runtime pending a correctly scoped bit test. |
| Redesign | H014-025ac: observe the originating CUDA compressed-KV row and test its NoPE tail bit-for-bit against the resident cache. |

| Field | H014-025ac |
|---|---|
| Hypothesis | The NoPE cache is bit exact to its originating CUDA projection while complete output and both caches remain within `3e-5` of the independent oracle. |
| Implementation | Captured CUDA compressed-KV rows before reset; production arithmetic was unchanged. |
| Benchmark | `h014-025ac-real-mla-stage.json`; same immutable layer-3 fixture and exact sm_86+PTX DLL as H014-025ab. |
| Result | **SUPPORTED**: output `5.879e-7`, latent `2.736e-7`, rope `1.609e-7`; CUDA-source NoPE copy and replay both bit exact. p50/p95/p99 `0.5534/0.56715/0.584056 ms`; device `0.540446 ms`. |
| Inspection | Projection/query norm costs `0.233874 ms` (`43.3%`), attention+gate `0.109308 ms` (`20.2%`), output projection `0.066044 ms` (`12.2%`), and cache append `0.015999 ms` (`3.0%`). The phase sum is `0.425224 ms`; cycling/cache scheduling leaves `0.115222 ms` (`21.3%`). The stage holds 232,452,352 projection-weight bytes and 2,304 cache bytes/token/layer/request. |
| Bottleneck | Projection-weight traffic dominates overall; absorb attention is the largest non-projection phase and still pays a separate sigmoid launch/context round-trip. |
| Decision | **RETAIN** complete stateful Gated MLA CUDA semantics. |
| Redesign | H014-025ad: fuse sigmoid gating into the attention epilogue and require at least `5%` lower complete-stage device service with unchanged correctness. |

| Field | H014-025ad |
|---|---|
| Hypothesis | Fusing sigmoid gating into the absorb-attention epilogue lowers complete MLA device service by at least `5%` without changing correctness. |
| Implementation | Added the gate operand to the existing single-token absorb kernel and removed the separate sigmoid launch in the experimental binary. |
| Benchmark | `h014-025ad-mla-fused-gate.json` against immutable H014-025ac; identical fixture, 20 warmups, 100 retained calls. |
| Result | **FALSIFIED**: device service `0.540446 -> 0.535388 ms`, only `0.936%` lower. Attention+gate `0.109308 -> 0.105516 ms` (`3.47%`). Correctness stayed at `5.879e-7`, cache copy and replay stayed bit exact. |
| Inspection | The removed launch/round-trip saved `0.00506 ms`; the 5% gate required `0.02702 ms`. One tail outlier also raised p99 to `1.068719 ms`. |
| Bottleneck | Projection-weight traffic and cross-phase cache effects, not the standalone sigmoid kernel. |
| Decision | **REVERT** fusion; retain the simpler separately testable sigmoid primitive. |
| Redesign | H014-025ae: rebuild the reverted sm_86+PTX package and qualify all 11 operation classes on one exact binary. |

| Field | H014-025ae |
|---|---|
| Hypothesis | All 11 Kimi-critical operation classes retain real-weight correctness and no-fallback CUDA identity on one exact sm_86+compute_86 deployment binary. |
| Implementation | Rebuilt the reverted separately gated MLA implementation once, reran all 12 real-weight/state fixtures against that DLL, and added a fail-closed aggregate certifier for evidence hashes, backend identity, numerical gates, capability negotiation, and PTX/SASS symbols. |
| Benchmark | `h014-025ae-same-binary-qualification.json`, 12 retained fixtures, `k3-cuda-operation-matrix.json`, and `rtx3090-sm86-certification.json`; exact binary SHA-256 `628662f5...e8f7b`. |
| Result | **SUPPORTED**: 11/11 `CUDA_READY`, 11/11 `CERTIFIED`, zero blockers, one binary hash. Maximum component relative L2 was `9.507e-7`; routing was exact; embedding and stateful replay gates were bit exact. The package contains every required entry point in sm_86 cubin and compute_86 PTX, with no Blackwell cubin. |
| Inspection | Every fixture explicitly identifies a CUDA backend, forbids CPU fallback, rejects sm_75, accepts sm_86, and hashes to the deployed DLL. This is component/package evidence, not an assertion that the local sm_120 GPU physically executed sm_86 SASS. Physical sm_86 execution remains assigned to the single-3090 canary. |
| Bottleneck | Component CUDA coverage is complete. The unresolved P0 dependency is full 93-layer production dispatch plus subsequent decode through these primitives; P1 persistent stage ownership remains separate. |
| Decision | **RETAIN** the exact DLL and promote all 11 component rows. Do not promote the full-graph or physical-canary gates. |
| Redesign | H014-025af: drive the existing complete real Kimi graph through the certified CUDA operation surface and prove 93/93 layer coverage plus subsequent decode without CPU mathematical fallback. |

| Field | H014-025af |
|---|---|
| Hypothesis | The certified CUDA primitives compose into a real four-layer Kimi slice with exact routing and at most `0.003` relative L2 error across prompt and decode. |
| Implementation | Added a strict one-context, one-layer-at-a-time CUDA graph runner over real checkpoint weights and persistent KDA/MLA state. |
| Benchmark | `cuda/h014-025af-real-4-layer-graph.json`; layers 0–3, three positions, nine routed calls, exact deployment DLL. |
| Result | **FALSIFIED**: maximum error `0.005905`; routing comparison failed; all nine applicable operation classes nevertheless executed on CUDA with no CPU mathematical fallback. |
| Inspection | The trace parser read route-call ordinal as layer, and direct scale division changed `0.023–0.030%` of representative int4 packed weights relative to the serial loader's rounded reciprocal multiply. |
| Bottleneck | Adapter fidelity, not a missing CUDA primitive. |
| Decision | **MODIFY** only route parsing, quantizer rounding, and diagnostic granularity. |
| Redesign | H014-025ag repeats the same fixture with exact serial load/trace semantics and the unchanged gate. |

| Field | H014-025ag |
|---|---|
| Hypothesis | Exact loader rounding and trace coordinates restore exact routes and bring the slice inside `0.003`. |
| Implementation | Matched `kimi_k3.c`'s FP32 reciprocal-multiply packing, derived global positions from route occurrence order, and added per-position metrics. |
| Benchmark | `cuda/h014-025ag-exact-load-routing.json`; identical layers, tokens, routes, and binary. |
| Result | **FALSIFIED numerical gate**: all nine routes became exact and layer-0 error fell to `4.120e-7`, but maximum graph error was `0.003306`. |
| Inspection | Error begins after the first routed expert (`<=0.002986`) and matches the documented scale of the serial oracle's optional `K3_IDOT=1` activation quantization. Certified CUDA and the independent mathematical reference use FP32 expert activations. |
| Bottleneck | Reference precision mismatch in routed MXFP4 experts. |
| Decision | **RETAIN** corrected adapters; do not weaken the gate. |
| Redesign | H014-025ah creates an explicit `K3_IDOT=0` slice oracle and tightens the comparison gate to `3e-5`. |

| Field | H014-025ah |
|---|---|
| Hypothesis | The identical graph matches a checkpoint-faithful FP32-activation MXFP4 serial oracle within `3e-5`, with exact routing and stateful decode. |
| Implementation | Added an explicit serial-oracle IDOT selector and partial-oracle stride support; CUDA arithmetic was unchanged. |
| Benchmark | `cuda/h014-025ah-idot0-4-layer-graph.json` against `oracle-layer-4-idot0`; three positions, nine routes, exact DLL SHA `628662f5...e8f7b`. |
| Result | **SUPPORTED**: maximum error `8.057e-7`; 9/9 routes exact; stateful decode passed; nine applicable operation classes executed with zero CPU mathematical fallbacks. |
| Inspection | Removing only the CPU approximation reduced the discrepancy by roughly four orders of magnitude at layer scope. Materialization dominates the `149.75 s` wall time, so this is correctness evidence, not service-rate evidence. |
| Bottleneck | Full P0 now requires the same test over 93/93 layers plus final norm, LM head, sampling, and decode. |
| Decision | **RETAIN** the streamed graph implementation and promote the representative composition gate only. |
| Redesign | H014-025ai generates an aligned full oracle and runs the complete CUDA graph. |

| Field | H014-025ai |
|---|---|
| Hypothesis | The exact deployment CUDA binary matches an immutable full 93-layer `K3_IDOT=0` oracle with maximum relative L2 error `<=0.003`, exact top-16 routing, final norm/head/sampling coverage, stateful decode, and no CPU mathematical fallback. |
| Implementation | Generated and hashed the full aligned oracle; retained H014-025ah CUDA mathematics unchanged; added sparse stderr-only progress telemetry every eight layers. |
| Benchmark | `cuda/h014-025ai-full-93-layer-graph.json`; 93 layers, three positions, 276 routing calls, 4,416 expert selections; exact DLL SHA `628662f5...e8f7b`. |
| Result | **FALSIFIED (routing equality)**. All 11 classes and 93/93 layers executed; maximum relative L2 `5.127e-4 < 0.003`; sampled token `11` and stateful decode passed. Four of 276 ordered routes differed: two one-expert set changes and two final-pair order swaps. |
| Inspection | The first set divergence is prompt position 1 at layer 26 (`542 -> 654`), after layer 25 differed by only `3.490e-7`; layer output then reached `5.291e-4`. Layer 30 changed `888 -> 67`. Final hidden/logits errors were only `3.332e-6`/`2.235e-6` with logits cosine `0.9999999999975`. The CPU router uses index-order FP32 accumulation; CUDA used a 128-thread tree. Streamed wall was `3671.68 s`: `2501.34 s` layer loading and `1042.40 s` layer execution, not capacity evidence. |
| Bottleneck | Near-tied router cutoffs are unstable under the parallel FP32 reduction order, so exact selected-expert semantics are not yet certified. |
| Decision | **MODIFY router only**. Retain all non-router CUDA paths and the preregistered numerical gate; do not promote the full graph gate. |
| Redesign | H014-025aj tests deterministic per-expert index-order router accumulation through layer 30 before another full run. |

| Field | H014-025aj |
|---|---|
| Hypothesis | Replacing only the parallel router-logit reduction with one-thread-per-expert input-index-order accumulation reproduces all 90 ordered top-16 routes through layer 30 across the same three positions, stays within relative L2 `<=0.003`, and uses no CPU mathematical fallback. |
| Implementation | Replaced only the router logit kernel and launch geometry with deterministic per-expert input-index-order accumulation; selection and every other operation remained unchanged. Built a separate candidate DLL. |
| Benchmark | `cuda/h014-025aj-deterministic-router-31-layer.json`; layers 0–30, three positions, 90 routes, immutable full oracle. |
| Result | **SUPPORTED**: 90/90 ordered routes exact; maximum relative L2 `4.656e-7`; stateful decode passed; candidate DLL SHA `c93b8f9d...1044`. |
| Inspection | The former layer-26/layer-30 set changes disappeared and the numerical spike fell by about three orders of magnitude. The real component fixture stayed correct, but router p50 rose from `0.0964 ms` to `0.4154 ms` (4.31x, +`0.319 ms`). |
| Bottleneck | Reduction-order correctness is resolved through layer 30. Full-graph generalization and same-binary package qualification remain. |
| Decision | **RETAIN** deterministic routing: exact expert IDs are mandatory and the absolute latency cost is small relative to current stage time. |
| Redesign | H014-025ak repeats the complete 93-layer graph with the retained candidate. |

| Field | H014-025ak |
|---|---|
| Hypothesis | The unchanged deterministic-router candidate executes all 93 layers, all 276 ordered routes, final norm/head, sampled token `11`, and subsequent decode within relative L2 `<=0.003`, with no CPU mathematical fallback. |
| Implementation | No code change; repeated the immutable H014-025ai fixture with candidate DLL SHA `c93b8f9d...1044`. |
| Benchmark | `cuda/h014-025ak-deterministic-router-full-93-layer.json`; 93 layers, three positions, 276 routes, 4,416 selections. |
| Result | **SUPPORTED**: 93/93 layers, 276/276 ordered routes, 4,416/4,416 expert IDs, all 11 classes, sampled token `11`, and stateful decode passed; maximum relative L2 `9.922e-7`; no CPU math fallback. |
| Inspection | Final hidden/logits errors were `6.177e-7`/`4.742e-7`, logits cosine `0.999999999999901`. Wall was `3206.40 s`, including `2273.72 s` summed layer loading and `826.85 s` execution; streamed I/O variance makes this correctness evidence, not service-rate evidence. |
| Bottleneck | Full graph semantics and routing are resolved. The new hash still needs all 11 component receipts and sm_86 package audit rerun. |
| Decision | **RETAIN** deterministic routing and promote the 93-layer graph gate subject to same-binary package recertification. |
| Redesign | H014-025al recertifies all components and sm_86 metadata on exact candidate hash `c93b8f9d...1044`. |

| Field | H014-025al |
|---|---|
| Hypothesis | Exact candidate `c93b8f9d...1044` retains 11/11 real-weight CUDA component gates, sm_86 cubin plus compute_86 PTX coverage, fail-closed capability negotiation, and can replace the deployment DLL. |
| Implementation | Reran all 12 immutable real-weight/state component fixtures on the new candidate, inspected every required PTX/SASS symbol, copied the candidate byte-for-byte over the deployment DLL, and reran the fail-closed aggregate audit against the deployed path. Prior-hash receipts were not reused. |
| Benchmark | `cuda/h014-025al-same-binary-qualification.json` SHA `ab7c6904...70a8`, `k3-cuda-operation-matrix.json` SHA `5c01acc4...e03`, and `rtx3090-sm86-certification.json` SHA `33369c55...655`; deployed DLL SHA `c93b8f9d...1044`. |
| Result | **SUPPORTED**: 12/12 fixtures passed, 11/11 classes are `CUDA_READY` and `CERTIFIED`, zero blockers, one exact binary hash, maximum component relative L2 `9.507e-7`, exact routing, and bit-exact embedding/replay invariants. |
| Inspection | All receipts identify a no-fallback NVIDIA CUDA backend, minimum compute capability 86, sm_86 acceptance, and sm_75 rejection. `cuobjdump` found every required entry point in sm_86 SASS and compute_86 PTX and no Blackwell cubin. Eleven non-router warm p50s were stable; deterministic routing rose from `0.0964` to `0.4160 ms` (4.315x, +`0.3196 ms`), as predicted by index-order accumulation. The complete 93-layer run makes the semantic benefit decisive and the absolute cost small. |
| Bottleneck | P0 is closed. The canonical runtime still streams/rematerializes weights and lacks a persistent resident real-Kimi stage lifecycle. |
| Decision | **RETAIN AND DEPLOY** exact DLL `c93b8f9d...1044`; preserve physical sm_86 execution for the preregistered single-RTX-3090 canary. |
| Redesign | H014-026 joins this CUDA graph with Experiment 013 persistence: immutable assignment, one-time materialize/load/prepare, resident weights/state/buffers, successive warm generations, and zero infrastructure or model rebuilds. |

| Field | H014-026a |
|---|---|
| Hypothesis | A canonical persistent worker can load final Kimi layer 92 plus final AttnRes/norm/head once, retain all 896 experts and request state on CUDA, and process three consecutive oracle-derived positions within relative L2 `<=3e-5`, with exact token 11 and zero warm infrastructure/model/buffer reconstruction. |
| Implementation | Added `native-cuda:0` control identity and a minimal final-stage executor behind the canonical loader seam. It preloads all layer/endpoint weights, 896 experts, and session state/buffers, then executes canonical packed messages and token results. A first `234.4 s` attempt reached READY/warm-up but was invalidated by a telemetry-key error; only that lookup changed before the retained rerun. |
| Benchmark | `persistent/h014-026a-final-stage.json` SHA `f9ac8c82...b440`; positions 0, 1, and stateful decode 2; deployed DLL `c93b8f9d...1044`. |
| Result | **SUPPORTED**: max layer/final relative L2 `2.505e-7`; max retained-logits relative L2 `3.181e-7`; 3/3 routes and sampled IDs `220,11,374` exact; finite three-position MLA state; all aggregate and per-generation warm rebuild counters zero. Warm wall p50/p95/p99 `4.918/5.004/5.012 ms`; device p50 `4.646 ms`. |
| Inspection | The worker held all 896 experts and reported `17,258,968,832` tracked tensor/vector bytes. Actual free VRAM fell by `19,407,044,608` bytes, revealing `2,148,075,776` bytes of uncounted CUDA context/allocator/workspace overhead. Each incoming canonical boundary carried `258,048` payload bytes (`~259.2 KB` framed); each token result carried 8 payload bytes (`~1.89 KB` framed). |
| Bottleneck | Warm execution is correct; production P1 remains blocked by custom-loader-only integration, final-stage-only scope, and understated native VRAM accounting. |
| Decision | **RETAIN PROTOTYPE, MODIFY INTEGRATION/ACCOUNTING**. |
| Redesign | H014-026b uses measured native VRAM delta, registers the Kimi adapter with exact DLL pinning, and proves a non-final stage emits the complete Kimi boundary with the same zero-rebuild lifecycle. |

| Field | H014-026b (preregistered) |
|---|---|
| Hypothesis | A registered Kimi adapter with worker-pinned checkpoint identity and exact DLL hash can load the same final stage through the default canonical loader, preserve H014-026a correctness/zero-rebuild behavior, and report native resident memory as the measured CUDA free-memory delta rather than the tracked tensor sum. |
| Implementation | Added `kimi_k3_cuda` to the packaged native-adapter registry; pinned config, Safetensors index, content fingerprint, tokenizer, and exact native-library SHA-256 through worker-owned control data; routed loading through `PersistentStageRuntime`'s default loader; changed only native residency reporting from the tracked tensor sum to the measured positive CUDA free-memory delta. Kimi arithmetic and fixtures were unchanged. |
| Benchmark | `persistent/h014-026b-registered-final-stage.json` SHA `eddbb53d...f1d2`, using H014-026a as the immutable regression baseline; 19 focused adapter/runtime tests. |
| Result | **SUPPORTED**: default-loader stage execution preserved max layer/final relative L2 `2.505e-7`, logits `3.180e-7`, exact 3/3 routes, sampled IDs `220,11,374`, valid three-position state, and zero aggregate/per-generation warm reconstruction counters. Wall p50/p95 `5.005/5.016 ms`; device p50 `4.669 ms`. |
| Inspection | Tracked CUDA allocations remained `17,258,968,832` bytes while the measured free-memory delta was `19,407,044,608` bytes, so admission now includes the previously hidden `2,148,075,776` bytes. H014-026a and H014-026b numerical metrics were identical; the `0.087 ms` wall-p50 difference and `0.023 ms` device-p50 difference are run variance, not an arithmetic change. The real assignment independently resolved to exactly 5,402 tensors and `18,915,537,408` source bytes. |
| Bottleneck | Registration, identity pinning, and native load-time memory admission are resolved. The retained executor still accepts only final layer 92, returns a compact final-stage result rather than the complete inter-stage Kimi boundary, and lives behind an experiment implementation module. |
| Decision | **RETAIN** identity/library pinning, registered default loading, and measured native residency. Do not promote the full P1 gate yet. |
| Redesign | H014-026c promotes a production executor and proves representative non-final KDA and Gated-MLA stages emit the complete canonical boundary with state continuity and zero warm reconstruction. |

| Field | H014-026c (preregistered) |
|---|---|
| Hypothesis | One production `KimiK3StageExecutor` can own either a representative non-final KDA layer or Gated-MLA layer, keep all assigned weights/state resident, execute three consecutive oracle-derived positions within relative L2 `<=3e-5` with exact routes, emit the complete `float32 [1,9,7168]` boundary required by its successor, and retain zero warm infrastructure/model/buffer reconstruction. |
| Implementation | Added packaged `swarm_inference.execution.kimi_k3_stage.KimiK3StageExecutor`, parameterized checkpoint-aligned non-embedding layer ownership, allocated KDA or MLA state per request during PREPARE, and made every non-final result preserve/update the full eight-row AttnRes snapshot payload. Kernel and quantization code were unchanged. |
| Benchmark | `persistent/h014-026c-nonfinal-stages.json` SHA `70c7d94a...59ca`; real layer 1 (KDA+MoE) and layer 3 (Gated-MLA+MoE), positions 0/1/2, plus final-stage regression SHA `efd17422...d28d`, immutable full oracle, deployed DLL `c93b8f9d...1044`, and 19 focused tests. |
| Result | **SUPPORTED**: layer-1/layer-3 maximum boundary relative L2 `1.619e-7`/`3.002e-7`; 6/6 routes exact; every residual row bit exact; every emitted boundary `float32 [1,9,7168]`; KDA/MLA state advanced to length 3, remained finite, nonzero, and isolated; all aggregate/per-generation warm rebuild counters zero. The generalized layer-92 regression retained `2.505e-7` error and exact tokens/routes. |
| Inspection | Layer 1 tracked/reported `16,112,520,448`/`18,264,096,768` CUDA bytes; layer 3 `16,083,851,008`/`18,228,445,184`. Cold loads were `205.38/207.22 s` and visibly storage-bound. Three retained calls gave wall p50 `30.881/27.494 ms` and device p50 `30.694/27.285 ms`; this cycle was designed for lifecycle/correctness, so those three-call values are not promoted to the capacity model. Canonical forwarded payloads were exactly `258,048` bytes and about `260.0 KB` framed. |
| Bottleneck | Non-final KDA/MLA ownership and boundary semantics are resolved. P1 still lacks a registered stage-zero embedding+dense path, and the packaged executor entry still inherits implementation from the experiment module rather than owning production code directly. |
| Decision | **RETAIN** the complete boundary contract and attention-specific request state. Do not close P1 or use the three-call timings for throughput. |
| Redesign | H014-026d adds the missing stage-zero embedding plus dense layer lifecycle without changing retained KDA/MLA/final arithmetic. |

| Field | H014-026d (preregistered) |
|---|---|
| Hypothesis | A registered stage-zero worker can preload the real BF16 embedding and dense layer-0 weights, accept three successive token IDs, execute embedding, attention, dense MLP, and AttnRes on CUDA, emit the complete boundary within relative L2 `<=3e-5` with bit-exact residual snapshots, and retain zero warm infrastructure/model/buffer reconstruction. |
| Implementation | Allowed exact stage-0 ownership, preloaded the BF16 embedding table, added one persistent device token buffer, wired the existing CUDA embedding gather and dense-MLP branch, and initially packed each one-token request as rank-one `int64 [1]`. No kernel, quantizer, or nonzero-stage arithmetic changed. |
| Benchmark | Failed attempt receipt `persistent/h014-026d-stage-zero-failure.json`; real token IDs `163584,18699,11`, exact 24-tensor/`4,690,023,424`-byte assignment and deployed DLL. |
| Result | **FALSIFIED BEFORE EXECUTION**: real weights reached CUDA load, then warmup was rejected with `ValueError: stage zero requires a rank-two int64 token tensor`; no numerical result was retained. |
| Inspection | The public stage-zero validator deliberately requires the established `[batch, sequence]` token contract. The executor accepts one element, but the harness encoded `[1]`; this is an input-shape adapter defect, not evidence about embedding or dense-layer arithmetic. |
| Bottleneck | Canonical fixture shape. |
| Decision | **MODIFY HARNESS ONLY**; retain the runtime implementation unjudged. |
| Redesign | H014-026e reshapes the identical token IDs to canonical `int64 [1,1]` and repeats all unchanged gates. |

| Field | H014-026e (preregistered) |
|---|---|
| Hypothesis | Packing each unchanged one-token input as canonical `int64 [1,1]` lets the registered stage-zero path reach CUDA and pass the H014-026d numerical, boundary, state, and zero-reconstruction gates without any arithmetic change. |
| Implementation | Changed only each immutable token fixture from rank-one `[1]` to canonical rank-two `[1,1]`; runtime arithmetic and the exact binary were unchanged. |
| Benchmark | `persistent/h014-026e-stage-zero.json` SHA `c8a2a4e7...a9e7`; real token IDs `163584,18699,11`, exact 24-tensor/`4,690,023,424`-byte assignment, and H014-026c regression SHA `70c7d94a...59ca`. |
| Result | **SUPPORTED**: maximum boundary relative L2 `2.752e-7`; all three embedding residual rows bit exact; three emitted `float32 [1,9,7168]` boundaries; KDA state length 3/finite/nonzero; zero aggregate/per-generation warm reconstruction counters. Tracked/reported CUDA bytes were `3,018,802,944`/`3,034,578,944`; wall/device p50 `2.870/2.687 ms`. |
| Inspection | The one-line shape correction moved execution through the existing public validator and CUDA path exactly as predicted. Incoming canonical token payloads were 8 bytes (`~1.16 KB` framed); outgoing activation payloads were `258,048` bytes (`~259.6 KB` framed). The source assignment includes a `2,348,810,240`-byte BF16 embedding and no experts. |
| Bottleneck | All functional stage roles now have real persistent lifecycle evidence. The only remaining P1 promotion issue is that the packaged production entry class inherits its implementation from the experiment module. |
| Decision | **RETAIN** stage-zero embedding+dense support and the canonical rank-two token contract. |
| Redesign | H014-026f mechanically moves the retained implementation under the production execution package, makes the experiment harness a wrapper, and reruns stage-zero, KDA, MLA, and final regressions. |

| Field | H014-026f |
|---|---|
| Hypothesis | Moving the unchanged persistent Kimi implementation under `swarm_inference.execution` and leaving only experiment wrappers preserves all H014-026e stage-zero, H014-026c KDA/MLA, and H014-026b final correctness/state/lifecycle gates on the exact DLL. |
| Implementation | Mechanically moved the unchanged executor to `swarm_inference.execution.kimi_k3_stage`, left the experiment module as a thin wrapper, rewired the registered adapter and canonical stage runtime, and added cycle provenance to the retained harnesses. No arithmetic, allocation, protocol, or fixture semantics changed. |
| Benchmark | `persistent/h014-026f-production-promotion.json`, aggregating fresh final receipt SHA `aeeaad0b...3881`, nonfinal receipt SHA `97947e5d...4d90`, and stage-zero receipt SHA `d772ea40...043d`, all on exact DLL SHA `c93b8f9...1044`. A direct import audit resolved the registered executor to `swarm_inference.execution.kimi_k3_stage.PersistentKimiStageExecutor` with MRO ending directly at `builtins.object`. |
| Result | **SUPPORTED / P1 PASS**: stage zero, layer 1 KDA+MoE, layer 3 Gated-MLA+MoE, and layer 92/final/head/sampling all passed. Worst retained relative L2 was `3.180e-7`; expert routes and sampled tokens were exact; stage-zero residual rows were bit exact; KDA/MLA state advanced through three positions; every warm topology/process/task/thread/connection/load/materialization/persistent-allocation delta was zero. |
| Inspection | The mechanical promotion did not change numerical or lifecycle behavior. The registered product adapter constructs a production-package class rather than inheriting an experiment implementation. The final, nonfinal, and stage-zero receipts retain exact native binary identity and report zero CPU mathematical fallbacks. |
| Bottleneck | No promotion bottleneck was observed. Cold checkpoint loading remains material but is correctly outside warm execution. The three-call warm samples are lifecycle gates, not a capacity benchmark; the next unresolved bottleneck is resident real-layer service time and its phase composition. |
| Decision | **RETAIN** the promoted production executor and close P1 canonical persistent Kimi stage runtime as **PASS**. |
| Redesign | H014-027a profiles the canonical resident real-weight MoE layer with statistically meaningful warm runs and compares detailed, production, and minimal telemetry before any optimization or capacity projection. |

| Field | H014-027a |
|---|---|
| Hypothesis | With all 896 real experts resident, the 16 routed MXFP4 experts plus the shared expert account for more than 70% of batch-1 canonical device service time in both a KDA+MoE layer and a Gated-MLA+MoE layer; production telemetry changes warm wall p50 by no more than 5% relative to minimal telemetry. |
| Implementation | Added measurement-only telemetry selection, opt-in cumulative router statistics, and CUDA-event phase timing around the routed and shared resident-expert groups. The direct source audit showed that the resident expert entry point deliberately has no detailed counter path, so the design retained the certified DLL and used its existing event API. No arithmetic, scheduling, tensor layout, state, or weight changed. |
| Benchmark | `performance/h014-027a-resident-stage-profile.json` SHA `d05f8d18...054f` plus derived inspection receipt. Real layers 1 and 3, exact DLL, modes minimal/production/detailed, 20 warmups + 100 retained state-contiguous calls/mode, first three positions checked against the oracle. |
| Result | **HYPOTHESIS FALSIFIED; BENCHMARK PASS**. KDA production wall p50/p95/p99 was `3.204/3.583/3.649 ms`; MLA was `2.920/3.158/3.322 ms`. Expert share was only `47.73%` and `52.34%`, below the predicted `>70%`. Production telemetry overhead versus minimal was `+0.31%` and `-0.12%`, passing the `<=5%` gate. Worst oracle relative L2 was `3.002e-7`, routes and cross-mode outputs were exact, and warm lifecycle deltas were zero. |
| Inspection | KDA decomposed to `47.73%` expert, `30.77%` dense, `13.36%` router, `8.14%` other; MLA to `52.34%`, `18.34%`, `14.98%`, `14.33%`. Thus weight-reading kernels jointly accounted for `91.86%`/`85.67%`, at expert effective source-weight bandwidth `374.1`/`377.8 GB/s`. Detailed telemetry added `6.38%`/`5.41%`, so it is retained only for phase attribution; production/minimal results define service rate. One-second `nvidia-smi` sampling aliased the short MLA block to zero and is not used as compute utilization. |
| Bottleneck | **Mixed weight-reading decode service**, not an expert-only kernel. The prior three-call `~27-31 ms` lifecycle sample is not steady-state evidence; after 20 calls the same canonical layers are `~2.9-3.2 ms`. Expert working-set reuse versus a cold/residency effect remains unresolved. |
| Decision | **RETAIN** telemetry controls and steady service results; reject an expert-only rewrite. |
| Redesign | H014-027b sweeps fixed-16 versus rotating-896 real expert working sets on the same resident layer and latent input to determine whether cache/reuse explains the cold/warm gap. |

| Field | H014-027b |
|---|---|
| Hypothesis | On a fully resident real layer-1 expert set, cycling all 896 experts raises 16-expert-group device p50 by at least `2x` relative to a repeatedly reused real top-16 group. If supported, expert working-set reuse explains a material part of the earlier `~30 ms` cold service; if falsified, the next cycle must inspect clocks, paging, or driver scheduling instead of changing expert kernels. |
| Implementation | Used the unchanged resident expert primitive and one oracle-seeded latent activation. Bracketed a rotating permutation of all 896 experts with fixed real top-16 blocks. No production arithmetic, kernel, weight, layout, route, or scheduler change. |
| Benchmark | `performance/h014-027b-expert-working-set.json` SHA `28023f4c...676c`. Fixed pre/post: 20 warmups + 100 retained groups; rotating: 56 warmups + 112 retained groups. Each timed group was 16 real experts, 48 kernels, zero H2D/D2H/D2D. |
| Result | **HYPOTHESIS FALSIFIED; BENCHMARK PASS**. Fixed-pre/rotating/fixed-post device p50 was `1.187776/1.188208/1.188048 ms`; rotating/fixed ratio `1.000249x`, not `>=2x`. Effective source-weight bandwidth was `232.48/232.57/231.88 GB/s`; the bracketing output fingerprints were identical; warm lifecycle deltas were zero. The first three post-load canonical calls remained `36.663/33.660/27.785 ms`, a cold-median/steady-p50 ratio of `11.173x`, while correctness and routes remained exact. |
| Inspection | Touching every resident expert does not reduce or increase steady group time, so neither a 16-expert cache working set nor per-expert first access explains the cold canonical calls. The cold cost belongs to a broader first-compute/device-readiness path. |
| Bottleneck | **Cold CUDA device readiness / scheduling transient**, exact submechanism still unresolved; not expert cache reuse. |
| Decision | **RETAIN** the working-set falsification and do not redesign expert layout for this symptom. |
| Redesign | H014-027c primes the device with an unrelated resident add kernel before the first canonical call. If this removes the transient, PREPARE must warm the CUDA device before READY; otherwise trace JIT/paging/driver scheduling. |

| Field | H014-027c |
|---|---|
| Hypothesis | At least `100 ms` of unrelated device-resident CUDA add work after LOAD but before the first Kimi call reduces the median of the first three canonical layer-1 calls by at least `5x` and to at most `2x` the H014-027a steady device p50. This would identify inadequate device warmup/readiness rather than Kimi expert working-set reuse. |
| Implementation | After a fresh exact-DLL load, uploaded two zero activation rows and ran the existing generic resident add kernel until CUDA events measured at least `100 ms`; no expert weights or Kimi state were touched. |
| Benchmark | `performance/h014-027c-device-warmup.json` SHA `89a76325...9f0b`, compared mechanically to retained H014-027b cold SHA `28023f4c...676c` and H014-027a steady SHA `d05f8d18...054f`. The warmup was 12,288 add launches, `114.266 ms` device / `114.486 ms` wall, and 57,344 one-time H2D bytes. |
| Result | **SUPPORTED**: canonical cold median improved `33.660 -> 3.419 ms` (`9.844x`) and the warmed median was `1.135x` steady p50, passing both `>=5x` and `<=2x` gates. Calls 0/1/2 were `4.059/3.419/3.405 ms`; routes and numerics remained exact; lifecycle deltas were zero. |
| Inspection | An unrelated generic kernel eliminates the Kimi cold transient without touching expert weights. This rules out expert cache reuse and identifies the READY boundary as premature: memory-resident is not yet compute-ready. The one-time detailed warmup is excluded from service rate. |
| Bottleneck | **CUDA device not compute-ready at the existing READY boundary.** |
| Decision | **RETAIN** a measured generic compute warmup as PREPARE work. |
| Redesign | H014-027d integrates the bounded warmup in the production executor, publishes its count/time/temporary memory, and validates fresh registered KDA and MLA stages without a harness-side Kimi warmup. |

| Field | H014-027d |
|---|---|
| Hypothesis | A one-time production PREPARE warmup of at least `100 ms` makes the first three fresh registered layer-1 and layer-3 canonical calls median `<=2x` their H014-027a steady device p50, while adding `<=500 ms` load wall, `<=64 KiB` temporary VRAM, exactly one prepare-warmup count, no warm-path allocation/reload/rebuild, and no correctness or route regression. |
| Implementation | Added the planned two-buffer generic-add hook, but placed it immediately after weight upload/synchronization and before final weight-digest, source-byte, and ownership validation. Buffers were freed and all metrics published as planned. |
| Benchmark | `performance/h014-027d-integrated-readiness.json` SHA `ed621142...89f`; fresh registered layers 1 and 3, no harness-side Kimi warmup, exact oracle calls 0-2 and exact DLL. |
| Result | **FALSIFIED / FAIL retained**. Warmup gates all passed (`101.99-103.29 ms` device, `102.63-104.15 ms` wall, 57,344 bytes temporary and recovered, count 1), correctness/routes/lifecycle passed, but first-three medians were `13.422 ms` KDA and `12.154 ms` MLA: `4.455x`/`4.451x` steady, above the `<=2x` gate. |
| Inspection | The generic warmup was followed by final checkpoint/source ownership work before LOAD returned. The device became partially cold again before READY. H014-027c succeeded because its warmup immediately preceded execution. The mechanism is now specifically the gap between compute priming and READY, not the primitive or duration. |
| Bottleneck | **Warmup placement before remaining CPU-side PREPARE work.** |
| Decision | **MODIFY**; retain the production metrics/hook but move it, unchanged, to the last PREPARE action. |
| Redesign | H014-027e moves only the warmup call after ownership construction and reruns the identical fresh KDA/MLA gate. |

| Field | H014-027e |
|---|---|
| Hypothesis | Moving the unchanged measured warmup to the final PREPARE action, after all ownership/source validation, makes fresh KDA and MLA first-three medians `<=2x` steady while retaining all H014-027d time/memory/count/correctness/lifecycle gates. |
| Implementation | Moved the warmup to the last executor-constructor statement. A CLI wiring error failed to pass the requested cycle ID, so `h014-027e-integrated-readiness.json` embedded `H014-027d`; inspection receipt SHA `fdda3365...05f8` marks it diagnostic-only and excludes it from retained gate evidence. |
| Benchmark | **INVALID PROVENANCE**. Diagnostic run used the intended exact fixtures/DLL but cannot satisfy the retained H014-027e receipt contract. |
| Result | Diagnostic-only: warmup time/memory/correctness/lifecycle gates passed, but KDA/MLA first-three medians were `14.836/30.665 ms` (`4.925x/11.229x` steady). The placement prediction would have failed even absent the provenance bug. |
| Inspection | The executor constructor is not the worker READY boundary. `PersistentStageRuntime.load_stage` performs ownership and memory validation, status construction, capability synchronization, and execution-runner startup after the loader returns. Readiness priming must be owned by that canonical lifecycle. |
| Bottleneck | **Warmup is still upstream of the canonical worker READY boundary.** |
| Decision | **MODIFY** and exclude the malformed raw receipt from gate evidence. |
| Redesign | H014-027f exposes an idempotence-checked executor `prepare_for_ready()` hook and calls it from `PersistentStageRuntime` after loader validation, immediately before loaded status/READY publication. |

| Field | H014-027f |
|---|---|
| Hypothesis | When canonical `PersistentStageRuntime` invokes the unchanged `>=100 ms` compute-readiness hook after executor ownership/memory validation and before publishing loaded/READY state, fresh KDA and MLA first-three medians are `<=2x` H014-027a steady p50, with exactly one hook call, `<=500 ms` wall, `<=64 KiB` recovered temporary VRAM, exact correctness/routes/state, and zero warm reconstruction. |
| Implementation | Constructor stopped self-warming; executor exposed idempotence-checked `prepare_for_ready()`; canonical `PersistentStageRuntime` invoked it after ownership/resident-memory validation. Fixed CLI cycle propagation. No CUDA or Kimi arithmetic change. |
| Benchmark | Valid `performance/h014-027f-integrated-readiness.json` SHA `69521b7e...10af`, embedded `H014-027f`, fresh registered real layers 1 and 3. |
| Result | **FALSIFIED / mixed FAIL**. Both generic warmups passed (`113.2-114.1 ms` device, `<=114.96 ms` wall, 57,344 recovered bytes, count 1), and correctness/routes/lifecycle passed. KDA first calls were `29.707/31.180/31.317 ms`, median `10.350x` steady (**FAIL**); subsequently loaded MLA was `3.163/3.082/3.068 ms`, median `1.129x` steady (**PASS**). |
| Inspection | A generic add warmup can leave the first stage-specific kernels cold and is not a deterministic readiness certificate. The second stage benefits from the preceding real KDA work, explaining the mixed result. READY must include the actual assigned Kimi call path, not a proxy kernel. |
| Bottleneck | **Stage-specific first-use readiness**, not generic CUDA activity. |
| Decision | **MODIFY**: retain the canonical lifecycle hook but make it execute and discard an isolated assigned-stage Kimi fixture. |
| Redesign | H014-027g adds a temporary 20-position stage-specific session after generic warmup, destroys its state, and validates a new oracle session for contamination and first-call latency. |

| Field | H014-027g |
|---|---|
| Hypothesis | Adding 20 assigned-stage Kimi fixture calls inside the canonical PREPARE hook makes fresh KDA and MLA oracle-call medians `<=2x` steady in the same process, while the temporary session is fully released, active sessions return to zero, the later oracle state/routes/outputs remain exact, PREPARE wall stays `<=1 s`, temporary VRAM stays `<=8 MiB`, and warm reconstruction remains zero. |
| Implementation | Added the isolated 20-position assigned-stage session exactly as planned. It executes after the generic add, closes before READY, restores serving execute count, removes research records, and reports timing/fingerprint/memory. Non-embedding roles use a zero boundary; stage zero uses token 0. |
| Benchmark | `performance/h014-027g-integrated-readiness.json` SHA `7a16e58d...9fb4`; fresh layers 1/3. Required role regressions: final SHA `1ba6e579...5ce1`, nonfinal SHA `aa8c165f...cfd3`, stage-zero SHA `14c3db6a...bc57`. |
| Result | **SUPPORTED / PASS**. KDA calls `3.303/3.000/3.022 ms`, median `1.003x` steady; MLA `3.049/2.742/2.717 ms`, median `1.004x`. Total PREPARE wall `273.532/323.571 ms`; temporary VRAM `6,291,456/2,097,152` bytes. Active fixture sessions returned to 0, memory recovered, serving count restored, records removed, outputs finite. All oracle outputs/routes/states remained exact. Stage-zero/KDA/MLA/final role regressions passed with worst relative L2 `3.002e-7` and all warm lifecycle deltas zero. |
| Inspection | Stage-specific first-use is the readiness requirement. An isolated disposable request state prevents contamination and provides a real local fixture. The generic add is now only a proxy preceding the sufficient stage-specific fixture and contributes roughly `110-115 ms` to PREPARE. |
| Bottleneck | **Resolved by assigned-stage PREPARE execution.** Remaining opportunity: remove redundant generic priming. |
| Decision | **RETAIN** canonical stage-specific PREPARE fixture and the worker-owned readiness hook. |
| Redesign | H014-027h removes the generic add from PREPARE and reruns the same first-call gate; retain the removal only if readiness and isolation remain passing while PREPARE falls materially. |

| Field | H014-027h |
|---|---|
| Hypothesis | The 20-call assigned-stage fixture alone is sufficient: removing the generic add keeps fresh KDA/MLA first-three medians `<=2x` steady, all isolation/correctness/lifecycle gates passing, total PREPARE wall `<=500 ms`, temporary VRAM `<=8 MiB`, and reduces median PREPARE wall by at least `25%` versus H014-027g. |
| Implementation | Removed the generic-add invocation; kept the 20-call assigned-stage fixture and isolation behavior unchanged. |
| Benchmark | `performance/h014-027h-integrated-readiness.json` SHA `5b533e29...e8c8`, compared to H014-027g SHA `7a16e58d...9fb4`. |
| Result | **PARTIALLY SUPPORTED; WALL-REDUCTION PREDICTION FALSIFIED**. KDA/MLA first-three medians were `2.993/2.833 ms` (`0.993x/1.038x` steady); isolation/correctness/lifecycle and `<=500 ms`/`<=8 MiB` gates passed. Mean PREPARE wall fell only `298.551 -> 281.855 ms` (`5.592%`), not `>=25%`. |
| Inspection | Generic work was functionally redundant, but removing it shifted first-use cost into the real fixture rather than eliminating it. The assigned-stage fixture is the correct readiness work and now owns the cold cost transparently. |
| Bottleneck | **Assigned-stage cold first calls inside PREPARE.** |
| Decision | **RETAIN** generic removal for simpler, semantically direct readiness despite falsifying the magnitude prediction. |
| Redesign | H014-027i reduces the assigned-stage fixture from 20 calls to 3, matching the observed cold convergence window, and reruns the identical gate. |

| Field | H014-027i |
|---|---|
| Hypothesis | Three isolated assigned-stage fixture calls are sufficient to make fresh KDA/MLA first-three medians `<=2x` steady, preserve every isolation/correctness/lifecycle/memory gate, keep PREPARE wall `<=250 ms`, and reduce mean PREPARE wall by at least `50%` versus H014-027h's `281.855 ms`. |
| Implementation | Changed the isolated assigned-stage fixture and matching validator from 20 calls to 3; input, state reset, CUDA, weights, and serving execution remained unchanged. |
| Benchmark | `performance/h014-027i-integrated-readiness.json` SHA `a89e2065...008c`, fresh registered layer 1 KDA and layer 3 Gated MLA, compared to retained H014-027a steady p50 and H014-027h PREPARE wall. |
| Result | **FALSIFIED / FAIL**. Correctness, routing, isolation, memory recovery, and warm lifecycle gates passed, but KDA retained calls were `16.963/13.214/12.958 ms` (median `4.386x` steady) and MLA calls were `12.145/11.773/11.745 ms` (`4.311x`). PREPARE wall was `95.105/83.598 ms`; the `68.300%` mean wall reduction passed but readiness did not. |
| Inspection | Three real fixture calls absorb only the largest cold transient. Both architectural classes then decay through a materially slower `11-17 ms` band, so readiness convergence is multi-phase rather than complete after the first three calls. The failure is not explained by correctness, routing, request leakage, allocation leakage, or lifecycle reconstruction. |
| Bottleneck | **Residual stage-specific CUDA first-use/convergence after call 3.** |
| Decision | **REVERT/MODIFY** the three-call count; do not use this configuration at READY. Retain the artifact as failed evidence. |
| Redesign | H014-027j tests 10 calls, the midpoint of the falsified 3-call and sufficient 20-call bounds, with unchanged semantics and a preregistered readiness/wall gate. |

| Field | H014-027j |
|---|---|
| Hypothesis | Ten isolated assigned-stage fixture calls are sufficient to make fresh KDA/MLA first-three medians `<=2x` steady while preserving every isolation/correctness/lifecycle/memory gate, keeping PREPARE wall `<=250 ms`, and reducing mean PREPARE wall by at least `25%` versus H014-027h's `281.855 ms`. |
| Implementation | Changed the isolated assigned-stage fixture and matching validator from 3 calls to 10; fixture inputs, reset behavior, CUDA, weights, and serving execution remained unchanged. |
| Benchmark | `performance/h014-027j-integrated-readiness.json` SHA `2cd2a94c...1bde`, exact H014-027a steady comparator and H014-027h wall comparator. |
| Result | **PARTIALLY SUPPORTED / FAIL**. Both roles became execution-ready: KDA retained calls `3.148/3.081/3.003 ms`, median `1.023x` steady; MLA `2.756/2.743/2.801 ms`, median `1.009x`. Correctness, routing, isolation, memory recovery, and warm lifecycle passed. MLA PREPARE passed at `218.306 ms`; KDA was `265.458 ms`, `15.458 ms` above the wall gate. Mean PREPARE wall was `241.882 ms`, only `14.182%` below H014-027h, not `>=25%`. |
| Inspection | Ten calls eliminate the residual readiness transient seen at three calls, but the aggregate summary discarded call order. KDA fixture device time was `254.413 ms` with `2.980/31.598/39.558 ms` min/p50/max; MLA was `212.476 ms` with `2.719/27.572/36.353 ms`. These distributions prove a late sharp convergence but do not identify its exact call. |
| Bottleneck | **An unlocated late cold-to-steady transition inside fixture calls 4-10; KDA one-time PREPARE wall, not retained execution, fails the gate.** |
| Decision | **MODIFY**. Do not relax the `250 ms` gate and do not yet retain 10 as minimal. Add the missing ordered timing trace before selecting a smaller count. |
| Redesign | H014-027k repeats the unresolved 10-call configuration with ordered per-call device telemetry only. If both roles enter and remain in the `<=2x` steady band by call 8, test 8; otherwise retain the observed lower bound and redesign from the actual trace. |

| Field | H014-027k |
|---|---|
| Hypothesis | In a fresh 10-call assigned-stage PREPARE fixture, calls 8-10 are each `<=2x` the retained H014-027a steady p50 for both KDA and MLA, and the following three oracle calls remain `<=2x`; therefore the cold tail is exhausted by call 8 even though H014-027j's aggregate timing could not show it. |
| Implementation | Added the ordered fixture device-time values to the PREPARE receipt. Call count, fixture inputs, stage math, request isolation, CUDA execution, and readiness validation remained unchanged. |
| Benchmark | `performance/h014-027k-integrated-readiness.json` SHA `07d4132c...28a`, fresh registered layers 1/3 with immutable H014-027a steady p50. |
| Result | **SUPPORTED / PASS**. KDA calls 1-7 were `37.378/36.393/35.645/29.093/31.217/38.604/17.092 ms`, followed by calls 8-10 at `2.996/2.982/2.991 ms`; its oracle calls were `3.144/3.028/2.981 ms` (`1.005x` median) and PREPARE wall `246.785 ms`. MLA calls 1-7 were `30.788/30.612/26.690/31.093/26.047/30.716/28.808 ms`, followed by `2.772/2.715/2.697 ms`; oracle calls were `2.811/2.705/2.705 ms` (`0.990x`) and wall `218.818 ms`. All correctness, route, isolation, memory, and lifecycle gates passed. |
| Inspection | Both roles show the same sharp transition: the first seven assigned-stage executions are cold and execution 8 is steady. The result also explains H014-027i: three calls stopped within the cold plateau. The required priming count is the number before the first served call, so the next minimum candidate is 7, not 8. |
| Bottleneck | **Seven-call stage-specific CUDA cold plateau before a stable call-8 transition.** |
| Decision | **RETAIN** ordered timing evidence and the isolated fixture design; reduce the disposable PREPARE calls to 7 for a direct readiness test. |
| Redesign | H014-027l makes seven disposable calls so the first post-READY call is execution 8, then reruns the identical correctness/readiness/wall/lifecycle gate. |

| Field | H014-027l |
|---|---|
| Hypothesis | Seven isolated assigned-stage PREPARE calls are sufficient for both KDA and MLA: each of the first three post-READY oracle calls is `<=2x` H014-027a steady p50, every correctness/routing/isolation/memory/lifecycle gate passes, temporary VRAM remains `<=8 MiB`, and each PREPARE wall remains `<=250 ms`. |
| Implementation | Changed the disposable assigned-stage fixture and matching validator from 10 calls to 7 and retained ordered device telemetry. Fixture inputs, reset behavior, CUDA, weights, and serving execution remained unchanged. |
| Benchmark | `performance/h014-027l-integrated-readiness.json` SHA `db65cd5e...f39e`; chained role regressions: final SHA `5dcbbb4f...f1ce`, non-final SHA `42cacf38...6b76`, stage-zero SHA `921d51fc...4468`. |
| Result | **SUPPORTED / PASS**. KDA fixture `36.816/29.991/36.278/25.524/2.976/2.971/2.979 ms`, PREPARE wall `148.685 ms`, and post-READY `3.176/3.091/3.062 ms` (`1.026x` steady). MLA fixture `27.761/29.182/26.148/29.662/30.813/16.379/2.719 ms`, wall `167.337 ms`, and post-READY `2.759/2.820/2.705 ms` (`1.010x`). Temporary VRAM was `6/2 MiB`; all correctness, routing, isolation, memory, and lifecycle gates passed. Final/non-final/stage-zero regressions passed with worst relative L2 `3.002e-7`, exact routes, bit-exact stage-zero residual rows, and zero warm lifecycle deltas. |
| Inspection | Convergence position varies: KDA reached steady at call 5 in this run, MLA at call 7, while H014-027k measured both at call 8 after seven cold calls. Seven disposable calls therefore cover the observed worst cold tail and make the first served call at least call 8. Testing six would knowingly expose a first request to a call-7 transient observed in H014-027k and is not a safe optimization. |
| Bottleneck | **READY-boundary cold execution is resolved by seven isolated assigned-stage calls; no cold work remains in the measured warm serving path.** |
| Decision | **RETAIN** seven-call production PREPARE fixture, ordered readiness evidence, and worker-owned hook. P1 remains PASS after regression. |
| Redesign | Continue P2 with representative early/middle/late KDA and MLA stage profiling, using READY workers and excluding one-time LOAD/PREPARE from warm service time. |

| Field | H014-027m |
|---|---|
| Hypothesis | For batch-1 warm decode, layer depth is not a material service-time driver within an attention class: early/middle/late production device p50 differs by at most `10%` and p95 by at most `15%` for KDA and separately for Gated MLA, while all real-oracle correctness, routing, and zero warm-lifecycle gates pass. |
| Implementation | Added a measurement-only production profiler with cycle-scoped worker/topology/session/load IDs. It reused retained early layers 1/3 and executed only unresolved real layers 45/47/89/91. No kernel, layout, state, routing, or scheduling change. |
| Benchmark | `performance/h014-027m-depth-profile.json` SHA `c01b2ddd...baec`; fresh layers used 20 warmups and 100 retained calls, with H014-027a SHA `d05f8d18...054f` supplying early evidence. |
| Result | **SUPPORTED / PASS**. KDA layers 1/45/89 device p50 were `3.013/3.052/3.073 ms` and p95 `3.381/3.434/3.435 ms`, spreads `2.010%/1.606%`. MLA layers 3/47/91 p50 were `2.731/2.793/2.815 ms` and p95 `2.930/3.149/3.117 ms`, spreads `3.094%/7.467%`. All correctness, route, and zero warm-lifecycle gates passed. Layer 89 was slowest at `3.073 ms` device p50. |
| Inspection | Architecture class, not layer depth, explains the stable warm stage difference; KDA is consistently slower than MLA. Fresh LOAD elapsed `214.7-275.3 s`, far above retained early `18.8-21.7 s`; this is a distinct host materialization/environment observation and is excluded from warm service time, but must be revisited in P7 bootstrap work. |
| Bottleneck | **Warm service: architecture-class weight-reading path, not depth. Startup: anomalous host materialization latency, not yet diagnosed.** |
| Decision | **RETAIN** KDA/MLA architecture-class representatives and layer 89 as the measured slowest complete non-final layer. Do not add per-depth constants to the capacity model. |
| Redesign | H014-027n measures multiple live decode streams on layer 89 under the unchanged synchronous scheduler before implementing batching. |

| Field | H014-027n |
|---|---|
| Hypothesis | With the current synchronous canonical stage scheduler, round-robin concurrency from 1 to 16 live real-Kimi request states increases aggregate layer-89 service rate by at most `10%`, while 16-stream per-stream cadence falls to at most one eighth of the single-stream rate; pre/post single-stream drift remains `<=5%`, and correctness, state isolation, routing, memory recovery, and warm lifecycle all pass. |
| Implementation | Added the measurement-only round-robin driver over one registered READY layer-89 worker with 1/2/4/8/16 sessions and a final single-stream control, 20 warmups and 100 retained calls per stream. No batching, overlap, kernel, weight, state, or scheduling change. |
| Benchmark | `performance/h014-027n-concurrency-baseline.json` SHA `8fbed1a0...dd63`; rates are explicitly single-stage operations/s, not output tok/s. |
| Result | **PARTIALLY SUPPORTED / FAIL**. Aggregate rates for 1/2/4/8/16 streams were `297.329/297.339/297.700/298.596/298.846 ops/s`; the pre/post control mean was `296.405`, so 16 streams changed aggregate rate only `+0.824%`. Per-stream rate at 16 was `18.678 ops/s`, `6.301%` of baseline. Median inter-completion/queue exposure grew from `3.307/0.006 ms` at one stream to `53.495/50.155 ms` at 16. Control drift was `0.622%`. Session allocation grew from `6` to `126 MiB`; request state from `6,881,280` to `110,100,480` bytes. All numerical, routing, cross-stream equality, and memory-recovery gates passed. The 8-stream window alone created two OS threads; every other lifecycle counter/window, including 16 streams and the post control, was zero. |
| Inspection | Synchronous round-robin is conclusively serial: more live state does not raise aggregate stage capacity and linearly converts capacity into queue delay. The two threads appear once at an execution/concurrency threshold and persist, because the later 16-stream window creates none; the aggregate snapshot does not locate whether sampling or CUDA execution caused them. |
| Bottleneck | **Primary: synchronous serial scheduler. Certification sub-blocker: unlocated one-time two-thread creation at the 8-stream window.** |
| Decision | **RETAIN** the concurrency baseline as evidence against unbatched concurrency, but **do not proceed to batching yet**; inspect the lifecycle transition first. |
| Redesign | H014-027o adds phase snapshots around sampler start, retained CUDA execution, and sampler stop, repeats the failed sweep, and includes a second 8-stream control to determine whether the thread creation is compute-triggered and one-time. |

| Field | H014-027o |
|---|---|
| Hypothesis | The two H014-027n OS threads are one-time lazy parent-process workers created during retained CUDA execution—not during research sampler start/stop—and persist afterward; the first 8-stream window reproduces the creation while a later 8-stream repeat creates zero additional threads. |
| Implementation | Added parent thread/process snapshots immediately before/after sampler start, after retained execution, and after sampler stop, plus a repeated 8-stream configuration. Stage math, inputs, state, CUDA, weights, and synchronous scheduling remained unchanged. |
| Benchmark | `performance/h014-027o-concurrency-lifecycle-trace.json` SHA `b6a7391e...0779`, same real layer-89 fixtures and stream grid as H014-027n. |
| Result | **FALSIFIED / VALID DIAGNOSTIC PASS**. All retained configurations, including both 8-stream windows, reported zero thread creation and passed correctness/memory/lifecycle. Parent thread counts remained `46/46/46/46` across sampler-start/execute/sampler-stop for 1/2/4/8 streams, then were already `48/48/48/48` at 16 streams and thereafter. Every sampler start created two child processes and every stop removed them; no parent thread came from sampling. |
| Inspection | The two parent threads arose after the first 8-stream sampler-stop snapshot but before the 16-stream retained-window snapshot: session close, record cleanup, next session open, or warmup. They are not created by retained EXECUTE. The post-compute process snapshot was accidentally included in H014-027o's block wall timer, depressing short-window rates; therefore H014-027o service rates are excluded and H014-027n remains the scheduler baseline. |
| Bottleneck | **One-time parent-thread transition in the request teardown/open/warmup boundary; exact phase unresolved. H014-027n synchronous scheduling remains the measured performance bottleneck.** |
| Decision | **RETAIN** H014-027o only for lifecycle phase evidence; do not use its rate measurements and do not yet modify PREPARE. |
| Redesign | H014-027p moves the process snapshot outside the service timer and traces before/after session open, warmup, retained execution, close, and research-record cleanup. |

| Field | H014-027p |
|---|---|
| Hypothesis | The two one-time parent threads are created while closing the first 8-session request set, persist through record cleanup and the next configuration, and are not created during session open, warmup, retained CUDA execution, or sampling. |
| Implementation | Added request-lifecycle snapshots before/after open, after warmup, after retained execution, after close, and after record cleanup. Moved the post-compute process snapshot outside the service timer. No stage/runtime semantic change. |
| Benchmark | `performance/h014-027p-request-lifecycle-trace.json` SHA `9d37dc61...73c3`, same layer-89 fixture/grid as H014-027n. |
| Result | **SUPPORTED / PASS**. Parent threads were `47/47/47/47/49/49` across before-open/after-open/after-warmup/after-retained/after-close/after-cleanup for the first 8-stream set. Only close created thread IDs `508508` and `861152`; both persisted. Every later configuration, including 16 streams and the repeated 8-stream set, remained at 49 with zero creation. All semantic, memory, and retained lifecycle gates passed. Corrected single-stream baseline was `300.859 ops/s`, 16-stream aggregate changed `-0.344%`, per-stream retention was `6.228%`, and pre/post drift `0.530%`, corroborating H014-027n. |
| Inspection | The threads are a deterministic one-time native request-resource teardown effect, not per-token EXECUTE, session open, warmup, sampling, or research-record cleanup. It is locally solvable by exercising the reset boundary before READY rather than accepting first-user RESET side effects. |
| Bottleneck | **First multi-session RESET REQUEST lazily creates two persistent native threads; synchronous serialization remains the performance bottleneck.** |
| Decision | **RETAIN** the corrected trace and move the exact trigger to PREPARE before any user request. |
| Redesign | H014-027q adds one disposable eight-session open/close PREPARE fixture, reports its transient memory/thread evidence, and reruns the full request-lifecycle sweep. |

| Field | H014-027q |
|---|---|
| Hypothesis | Opening and closing eight disposable request sessions during PREPARE absorbs the two-thread first-reset transition before READY; the subsequent 1/2/4/8/16/8/1 request sweep creates zero parent threads in every open, warmup, EXECUTE, close, cleanup, and retained-lifecycle phase, while correctness/memory pass, PREPARE wall remains `<=250 ms`, and maximum transient VRAM remains `<=64 MiB`. |
| Implementation | Added an isolated eight-session open/close fixture after the seven-call stage fixture. It executed no model operation, recorded thread/memory/wall evidence, and freed all state before READY. |
| Benchmark | Early-stop gate `performance/h014-027q-integrated-readiness.json` SHA `32afea5d...781c`; real registered KDA layer 1 and MLA layer 3. The downstream concurrency sweep was correctly not run after this gate failed. |
| Result | **FALSIFIED / FAIL**. Empty reset close created zero threads in both roles (`44/44/44` KDA and `43/43/43` MLA before-open/after-open/after-close), so it did not absorb the H014-027p trigger. Post-READY calls remained exact and steady: KDA `3.137/3.004/2.989 ms` (`0.997x`), MLA `2.748/2.727/2.865 ms` (`1.006x`); lifecycle/memory passed. KDA stage/reset/total wall was `240.494/51.738/292.231 ms`; MLA `221.900/33.209/255.110 ms`; both exceeded `250 ms`. Transient VRAM was `62/10 MiB`. |
| Inspection | Merely allocating and freeing eight request states is insufficient; the resources must be touched by real CUDA execution before close. The additive design also duplicates PREPARE work and cannot meet the wall budget under this run's cold plateau. |
| Bottleneck | **Thread trigger requires touched multi-session resources; additive stage-plus-reset PREPARE is wall-inefficient.** |
| Decision | **REVERT/MODIFY** the empty reset fixture. Preserve its failed artifact; do not relax the 250 ms gate. |
| Redesign | H014-027r replaces both fixtures with one eight-session composite: execute one real assigned-stage call per disposable session, then close all eight together. |

| Field | H014-027r |
|---|---|
| Hypothesis | A single composite PREPARE fixture that opens eight sessions, executes one real assigned-stage call in each, and closes all eight both (a) exhausts the seven-call cold plateau and (b) triggers the two close-time threads before READY; first-three KDA/MLA oracle calls are `<=2x` steady, PREPARE wall is `<=250 ms`, transient VRAM `<=64 MiB`, all state/memory/isolation gates pass, and no model work is duplicated in a second fixture. |
| Implementation | Replaced both fixtures with eight one-call disposable sessions, restoring serving counters/records and recording ordered device, output, thread, wall, and memory evidence. |
| Benchmark | Early-stop gate `performance/h014-027r-integrated-readiness.json` SHA `fb96657f...1bc7`; downstream sweep/regressions correctly not run after failure. |
| Result | **PARTIALLY SUPPORTED / FAIL**. Post-READY correctness/readiness passed: KDA oracle `3.132/3.031/3.401 ms` (`1.040x`), MLA `3.067/2.743/2.705 ms` (`1.004x`). MLA composite PREPARE passed at `218.774 ms`, `10 MiB`. KDA failed wall at `261.524 ms`, `62 MiB`; its fixture calls were `31.479/32.119/20.614/12.929/13.592/13.407/12.949/13.156 ms`. KDA parent threads changed `27 -> 46` during execution (19 lazy threads), but neither role created threads at close. |
| Inspection | Eight global calls exhaust serving cold work, but distributing them across fresh request states adds a `~13 ms` touched-session plateau and still does not reproduce the close-time threads that appeared only after sustained request use in H014-027p. This is a worse PREPARE than H014-027l. |
| Bottleneck | **Touched multi-session first-use overhead; sustained-use RESET threads are not cheaply primeable within the wall gate.** |
| Decision | **REVERT** to the retained H014-027l seven-call single-session PREPARE. Treat H014-027p's exactly-once close threads as explicit request-specific RESET activity, not warm per-generation infrastructure recreation. |
| Redesign | H014-027s regression-validates the reverted PREPARE and the exact separation: all EXECUTE windows zero; the first sustained eight-session RESET may create the measured two threads once; the repeat creates none. |

| Field | H014-027s |
|---|---|
| Hypothesis | Reverting to the retained seven-call single-session PREPARE restores KDA/MLA READY wall `<=250 ms`, transient VRAM `<=8 MiB`, and first-three latency `<=2x` steady; in the full request sweep every retained EXECUTE lifecycle delta is zero, exactly two persistent threads may be created only by the first sustained eight-session RESET REQUEST, and the repeated reset creates zero. |
| Implementation | Exactly reverted to H014-027l PREPARE/validator and kept the request-phase trace. No serving math, CUDA, weight, state, or scheduler change. |
| Benchmark | READY `performance/h014-027s-integrated-readiness.json` SHA `0a656654...07ca`; request trace `performance/h014-027s-request-lifecycle.json` SHA `731dc8cb...736b`. |
| Result | **READINESS SUPPORTED; EXACT RESET SUBPHASE FALSIFIED / OVERALL FAIL**. KDA/MLA READY passed at `232.380/180.270 ms`, `6/2 MiB`, post-READY `1.002x/1.010x` steady. Every retained EXECUTE lifecycle delta was zero. First eight-session counts were `46/46/46/46/46/48` across before-open/after-open/after-warmup/after-retained/after-close/after-record-cleanup: two threads appeared after cleanup, not immediately at close. All later phases including repeated reset stayed 48 with zero creation. Baseline was `300.174 ops/s`; 16-stream aggregate `-0.292%`, per-stream retention `6.232%`, drift `0.729%`. |
| Inspection | H014-027p observed two threads at the after-close snapshot; H014-027s observed the same count one snapshot later. The native creation is asynchronous within the bounded first RESET teardown interval, so claiming one exact observation subphase is not stable. Both runs agree on zero open/warmup/EXECUTE activity, exactly two across close+cleanup, persistence, and zero on repeat. |
| Bottleneck | **Performance: synchronous serial scheduler. Lifecycle: bounded exactly-once asynchronous RESET teardown activity, not per-generation recreation.** |
| Decision | **MODIFY the evidence contract, not the runtime**: evaluate close+cleanup as one RESET boundary. Retain the seven-call PREPARE. |
| Redesign | H014-027t mechanically cross-validates the phase-insensitive RESET contract over H014-027p and H014-027s before batching. |

| Field | H014-027t |
|---|---|
| Hypothesis | Across both independent real-weight request traces H014-027p and H014-027s, the first sustained eight-session RESET creates exactly two persistent parent threads in the union of close and immediate record-cleanup phases, all open/warmup/retained-EXECUTE phases create zero, and the repeated eight-session RESET creates zero; therefore the activity is bounded request-specific teardown rather than warm-generation infrastructure recreation. |
| Implementation | Added an evidence-only validator over the two immutable receipts/hashes. No CUDA run and no runtime change. |
| Benchmark | `performance/h014-027t-reset-contract.json` SHA `a4454e4c...0201`; H014-027p SHA `9d37dc61...73c3`, H014-027s SHA `731dc8cb...736b`. Restored-role regressions: final SHA `2640286b...8d40`, non-final SHA `86ff8a61...2d3`, stage-zero SHA `cf1e86ba...d291`. |
| Result | **SUPPORTED / PASS**. Both traces mechanically pass: exactly two persistent thread IDs in the first close+cleanup union, zero during open/warmup/retained EXECUTE, zero outside the first reset, zero on repeated reset, and all correctness/memory/retained lifecycle gates pass. Restored final/non-final/stage-zero roles passed with worst relative L2 `3.002e-7`, exact routes, bit-exact stage-zero residual rows, and zero warm lifecycle deltas. |
| Inspection | The stable semantic boundary is RESET REQUEST, not the scheduling instant at which an asynchronous native thread becomes visible. Normal generations recreate no process/thread/task/topology/connection/weight/materialization/buffer infrastructure. |
| Bottleneck | **P1 lifecycle blocker resolved. Remaining P2 bottleneck: synchronous serial stage scheduling.** |
| Decision | **RETAIN** seven-call PREPARE and bounded RESET contract; proceed to existing primitive batch capability before changing the canonical scheduler. |
| Redesign | H014-027u tests real batch-2 dense/expert primitives with actual layer-89 activations and weights. |

| Field | H014-027u |
|---|---|
| Hypothesis | Existing generic resident CUDA `batch=2` execution reuses Kimi weights efficiently: for real layer-89 latent-down dense, one selected MXFP4 routed expert, and the shared expert, one batch-2 call provides at least `1.5x` pair throughput versus two batch-1 calls, with relative L2 `<=1e-6`, cosine `>=0.999999`, zero per-call H2D/D2H in the timed window, and no weight/layout conversion. |
| Implementation | Added a benchmark-only contiguous two-row adapter using the existing `execute_dense(..., batch=2)` and `execute_resident(..., batch=2)` handles from one registered READY layer-89 worker. It brackets batch-2 with serial-pre and serial-post controls. No kernel, weight, layout, canonical stage, state, or scheduler change. |
| Benchmark | `performance/h014-027u-batch2-primitives.json` SHA `cff15d16...b1d4`; real oracle-derived layer-89 MLP/latent activations, 20 warmup and 100 retained pairs per mode, exact DLL SHA `c93b8f9d...044`, serial-pre/batch2/serial-post CUDA events, zero timed H2D/D2H, and recovered temporary allocations. |
| Result | **HYPOTHESIS FALSIFIED; EXECUTION PASS**. Batch-2 was bit-exact to serial for all three primitives (maximum absolute and relative L2 error `0`, cosine `1.0` within floating representation), routing matched the oracle, and the full fixture relative L2 was `6.139e-8`. Device pair speedups were only `1.1093x` latent-down dense, `1.1315x` selected routed expert, and `1.0736x` shared expert; wall speedups were `1.1014x`, `1.1257x`, and `1.0727x`. Batch-2 device p50 was `0.064704/0.129312/0.301472 ms`, respectively. |
| Inspection | The native source maps `S` to `blockIdx.y`: both `quant_matmul` and fused `mxfp4_matmul_pair` launch independent output blocks for each row. Batch-2 halves launch count (dense `2 -> 1`; a fused expert pair `6 -> 3`) but each row still reloads the weights. This matches the small measured gain. The shared expert's duplicated-weight effective traffic reached `1,753.0 GB/s`; latent-down reached `1,588.2 GB/s`. The routed expert's `271.4 GB/s` aggregate reflects its three projections plus intervening SiTU work and is mixed rather than a simple single-GEMV bandwidth measure. |
| Bottleneck | **Existing batch dimension is row-parallel launch amortization, not cross-row weight reuse. Dense/shared paths are bandwidth-saturated; routed expert is mixed weight traffic, reduction/nonlinearity, and launch cost.** |
| Decision | **RETAIN** correctness and the existing batch API, but reject it as sufficient evidence for a canonical batching redesign. Do not claim the preregistered `1.5x` gain. |
| Redesign | H014-027v measures the unchanged primitives at batch `1/2/4/8/16` to determine the scaling ceiling and justify—or reject—the first row-cooperative native kernel. |

| Field | H014-027v |
|---|---|
| Hypothesis | Because the current native grid assigns one independent block set per row and cannot deliberately reuse weights across rows, batch `2/4/8/16` will provide less than `1.25x` row-throughput speedup versus batch 1 for each real layer-89 primitive, despite bit-exact outputs, and per-row latency will plateau rather than fall materially. |
| Implementation | Added the planned measurement-only scaling driver over the same registered layer-89 handles, with a post-run batch-1 control and explicit workspace accounting. The first version did not emit per-size progress or serialize partial results before the whole primitive list completed. Native and production math were unchanged. |
| Benchmark | Failed-run receipt `performance/h014-027v-batch-scaling-failure.json` SHA `d8242cb6...364c`; exact H014-027u input SHA `cff15d16...b1d4`, exact CUDA DLL SHA `c93b8f9d...044`, batches `1/2/4/8/16`, 20 warmups and 100 retained calls requested. |
| Result | **BLOCKED / FAIL; HYPOTHESIS NOT EVALUABLE**. After real layer-89 LOAD and advancement beyond the latent-down primitive, the selected routed-expert warmup raised `Kimi resident expert MLP launch: unknown error`. No partial timings were retained. The driver then disappeared: `nvidia-smi` reported no devices and `GPU is lost ... reboot required`. |
| Inspection | The Python stack proves failure inside the selected expert primitive, but the first harness did not identify the exact batch. Windows System recorded 601 `nvlddmkm` event-153 and 60 event-14 entries in the one-hour query, concentrated at `13:33:28-13:33:32 +10:00`. A targeted batch-4 follow-up could not initialize CUDA because the device was already lost, so it is invalid as a batch-4 result. This is stronger safety evidence than a mere rejected call. |
| Bottleneck | **An uncertified native expert batch above the retained batch-2 result can lose the local GPU; exact first failing size remains unknown until recovery.** |
| Decision | **HALT GPU WORK** until reboot. Do not rerun the multi-size sweep. First make the native API fail closed above the last certified expert batch (`2`) and expose that capability mechanically. |
| Redesign | H014-027w adds a certified-max-batch query and pre-launch `S<=2` guard to both Kimi expert entry points. After reboot it must prove batch 1/2 unchanged, batch 4 rejected before launch, and post-rejection GPU health intact. |

| Field | H014-027w |
|---|---|
| Hypothesis | Advertising `max_certified_batch=2` and rejecting `S>2` before scratch allocation or kernel launch prevents recurrence of the H014-027v device-loss class: real batch 1 and 2 remain numerically/performance equivalent to the retained binary, a batch-4 call returns a bounded unsupported-batch error, and `nvidia-smi` plus a subsequent batch-1 fixture remain healthy. |
| Implementation | Added the minimum safety change in both host and device-resident Kimi expert entry points, an exported capability query, and Python-side preflight (including a conservative `2` ceiling for legacy binaries). Built a quarantined candidate without overwriting the retained deployment DLL. After reboot, a bounded harness ran only `1 -> 2 -> rejected 4 -> 1`; an armed Python sentinel made the rejected request physically incapable of crossing the native boundary, and the retained DLL ran separately at only batch 1/2. The first validation attempt used the wrong `gate/down/up` handle order and was retained as invalid fixture evidence; only that adapter tuple was corrected to the certified `gate/up/down` ABI. |
| Benchmark | Build receipt `performance/h014-027w-failclosed-batch-candidate.json` SHA `eb10c0c5...e22f`; valid GPU receipt `performance/h014-027w-failclosed-batch-validation.json` SHA `4992a0da...2303`; invalid harness receipt `performance/h014-027w-invalid-harness-handle-order.json` SHA `a513ac98...b053`. Real layer-89 expert 803 and checkpoint weights; 20 warmups + 100 retained calls per safe mode; candidate SHA `c666e37c...1968`; retained SHA `c93b8f9d...1044`; RTX 5090 UUID pinned before/after. Required exact-binary regressions then produced 12/12 passing component receipts and `cuda/h014-027w-same-binary-qualification.json` SHA `0c95463f...662a`; complete graph `cuda/h014-027w-regression-full-93-layer.json` SHA `0c0a655a...4199`; final/non-final/stage-zero receipts SHA `f258dd02...ec97` / `7ad0f72b...f68f` / `8f7ef623...7515`; promotion receipt `cuda/h014-027w-promotion.json` SHA `37b5b5cb...55cc`. |
| Result | **SUPPORTED / H014-027w PASS / PROMOTED**. The native query advertised `2`. Candidate batch-1 pre/post device p50 was `0.073504/0.073504 ms`; batch-2 p50 was `0.128992 ms`. Batch 4 returned exactly `requested=4, certified_max=2` with zero native-sentinel calls, unchanged native statistics, zero request/persistent memory growth, and a passing CUDA synchronize. Batch-2 rows were bit exact to the two candidate batch-1 controls; all candidate batch-1/2 outputs were bit exact to the retained DLL. Candidate/retained p50 ratios were `0.999565` batch 1 and `0.999504` batch 2. On the same exact hash, 11/11 CUDA classes and sm_86/compute_86 packaging passed; the full graph executed 93/93 layers, 276/276 exact routes, 4,416/4,416 routed experts, final/head/sampling and stateful decode at maximum layer relative L2 `9.922e-7`; all four P1 roles passed with worst error `3.002e-7` and zero warm lifecycle reconstruction. Deployment now hashes to `c666e37c...1968`, returns certified maximum `2`, and `nvidia-smi` remains healthy on the same UUID. |
| Inspection | The invalid first attempt failed before retained batch evidence because the harness swapped projection handles; the already-certified uploader and real-expert fixture proved the ABI error, and post-cleanup `nvidia-smi` returned to the original free-memory state. With only that fixture correction, the bounded rerun demonstrated that the new guard has no measurable numerical or service regression and prevents the unsafe native launch. No batch above 2 executed. The graph rerun took `3812.71 s`, remained load-dominated, and exactly reproduced the retained numerical maximum. Promotion atomically retained the prior DLL, regenerated the canonical operation matrix, sm_86 certificate and native manifest, and verified the deployed capability. Generic `.lib/.exp` files predate the new export and are explicitly excluded from runtime certification; they will be regenerated after H014-028 stops changing the binary. |
| Bottleneck | **Existing batch execution is safe but still row-parallel and reloads weights per row; cross-row weight reuse is the measured unresolved P2 bottleneck.** No P0/P1 regression blocker remains on the promoted binary. |
| Decision | **RETAIN AND PROMOTE** exact DLL `c666e37c...1968`; keep `max_certified_batch=2`; retain the invalid harness and GPU-loss receipts; do not reopen the invalid multi-size sweep or delete the guard. |
| Redesign | H014-028 implements the minimum row-cooperative CUDA primitive that deliberately reuses one real Kimi weight tile across rows, then certifies batch `1 -> 2 -> 4 -> 8 -> 16` one size at a time with a health checkpoint after every newly attempted size. |

| Field | H014-028a |
|---|---|
| Hypothesis | A two-row cooperative real MXFP4 expert kernel that traverses each gate/up/down weight once for two token rows provides at least `1.50x` pair throughput versus two batch-1 calls, remains bit exact to serial and the deployed binary, and changes batch-1 p50 by no more than `5%`. |
| Implementation | Added templated two-row gate/up-pair and single-projection CUDA kernels. Each block owns one output and accumulates both rows from one decoded MXFP4 nibble/UE8M0 scale. Batch 1 remains on the exact H014-027w kernel path; the exported ceiling remains `2`; the deployed DLL was not changed. Added an incremental harness that persists each phase and runs no batch above 2. |
| Benchmark | `performance/h014-028a-row-cooperative-batch2.json` SHA `4c90e0b9...46c2`; quarantined candidate SHA `e0f94e1d...2119`; deployed control SHA `c666e37c...1968`; real layer-89 expert 803, 50 warmups and 300 retained calls per mode, CUDA events with zero timed H2D/D2H. New kernels were inspected in sm_86 SASS and compute_86 PTX. |
| Result | **SUPPORTED / PASS**. Candidate batch-2 device p50 was `0.079312 ms` versus `0.133056 ms` deployed. Pair-throughput speedup was `1.911438x` versus the deployed path's `1.146825x`; candidate/deployed batch-2 ratio was `0.596080`. Candidate batch-1 mean p50 was `0.075800 ms` versus `0.076296 ms` deployed (`0.993499x`). Batch 2 was bit exact to both candidate serial rows and the deployed binary. Candidate p95/p99 were `0.092162/0.369724 ms`; memory recovered within the gate and the same GPU UUID remained healthy. |
| Inspection | Deliberate cross-row traversal reuse converts the previously modest launch-only gain into near-ideal batch-2 aggregate throughput. The high p99 is an outlier tail and must be tracked at later sizes; architectural traversal reuse is proven, but physical DRAM byte reduction is not inferred without profiler counters. |
| Bottleneck | **Batch 2:** largely resolved weight traversal duplication; remaining tail jitter and per-row accumulation/reduction overhead. **Unresolved:** whether register/shared-memory growth and fanout remain stable at batch 4 and beyond. |
| Decision | **RETAIN IN QUARANTINE** and build a separate exact-batch-4 candidate. Do not promote yet and do not execute batch 4 through the current max-2 binary. |
| Redesign | H014-028b instantiates the same row-cooperative design for exactly four rows, raises only the quarantined candidate ceiling to 4, and tests batch 1, 2, then 4 with synchronization, known-safe fixtures, VRAM and `nvidia-smi` checks before considering batch 8. |

| Field | H014-028b |
|---|---|
| Hypothesis | The exact four-row cooperative real MXFP4 expert path provides at least `3.00x` aggregate throughput versus four batch-1 calls, remains bit exact, preserves the already certified batch-1/2 paths within `5%`, and leaves CUDA, VRAM and the physical GPU healthy for a post-target batch-1 fixture. |
| Implementation | Instantiated four-row gate/up-pair and down-projection kernels in a separate candidate, raised only that candidate's maximum to 4, added an exact supported-size export for `(1,2,4)`, and added a non-launching sticky-CUDA-error query. Unsupported batch 3 and 8 requests were armed with a native-call sentinel and rejected in Python before allocation/launch. The deployed binary remained unchanged. |
| Benchmark | `performance/h014-028b-row-cooperative-batch4.json` SHA `401a84a5...8ff7`; candidate SHA `7f7f2ad7...d360`; prior batch-2 candidate SHA `e0f94e1d...2119`; real layer-89 expert 803, 50 warmups and 300 retained calls per mode, exact order `1 -> 2 -> 4 -> 1`, then safe serial controls. Batch-4 templates were inspected in sm_86 SASS and compute_86 PTX. |
| Result | **SUPPORTED / PASS**. Batch-4 device p50/p95/p99 were `0.080384/0.082112/0.086328 ms`; wall p50/p95/p99 were `0.086700/0.088605/0.094941 ms`. Four serial p50 values summed to `0.294368 ms`, giving `3.662022x` aggregate speedup, `49,761.146 rows/s`, and `0.020096 ms` per-row service. Output was bit exact to all four serial rows. Batch-1/2 ratios versus the prior binary were `0.964541/0.952190`, both bit exact. Synchronize, sticky error, post-target known-safe output, memory, exact-size rejection and same-UUID GPU-health gates all passed. |
| Inspection | Weight traversal remains nearly constant-cost through four rows, and the batch-2 tail outlier did not recur at batch 4. Exact-size capability negotiation closes the safety ambiguity of a maximum-only contract: untested intermediate or larger sizes cannot reach native CUDA. |
| Bottleneck | **At batch 4, projection service is still dominated by one shared weight traversal; added row accumulation is small.** The next risk is register/shared-memory pressure at eight rows, not correctness or launch safety. |
| Decision | **RETAIN IN QUARANTINE** and build a separate exact-batch-8 candidate. Do not promote or expose batch 8 through this binary. |
| Redesign | H014-028c instantiates exactly eight rows and requires at least `5.00x` aggregate throughput, bit-exact output, prior-size preservation, and the identical post-target CUDA/VRAM/known-safe/`nvidia-smi` protocol before batch 16 is considered. |

| Field | H014-028c |
|---|---|
| Hypothesis | The exact eight-row cooperative real MXFP4 expert path provides at least `5.00x` aggregate throughput versus eight batch-1 calls, remains bit exact, preserves batch 1/2/4 within `5%`, and passes the full post-target CUDA-error, known-safe fixture, VRAM and same-GPU health protocol. |
| Implementation | Instantiated eight-row gate/up-pair and down-projection kernels in a new candidate and exposed only exact sizes `(1,2,4,8)`. No deployed or prior candidate binary was mutated. The same incremental harness completed every prior size before arming batch 8, then rejected batch 3 and 16 before the native boundary. |
| Benchmark | `performance/h014-028c-row-cooperative-batch8.json` SHA `16181f58...42e4`; candidate SHA `70d65478...a398`; prior batch-4 candidate SHA `7f7f2ad7...d360`; real layer-89 expert 803, 50 warmups and 300 retained calls per mode, exact order `1 -> 2 -> 4 -> 8 -> 1`, with eight serial controls. Batch-8 templates were present in sm_86 SASS and compute_86 PTX. |
| Result | **SUPPORTED / PASS**. Batch-8 device p50/p95/p99 were `0.109584/0.111459/0.114272 ms`; wall values were `0.115900/0.118200/0.126943 ms`. Eight serial device p50 values summed to `0.587520 ms`, yielding `5.361367x`, `73,003.358 rows/s`, and `0.013698 ms` per row. Outputs were bit exact. Batch-1/2/4 ratios to the prior binary were `0.962232/0.962993/0.964797`, all bit exact. Sticky CUDA state, synchronization, post-target batch 1, memory, exact-size preflight and same-UUID `nvidia-smi` all passed. |
| Inspection | Cross-row reuse remains materially useful at eight rows, but scaling efficiency has fallen from `91.6%` of ideal at batch 4 to `67.0%` at batch 8. The p95/p99 remain tight, so the bend is repeatable kernel work rather than tail instability. |
| Bottleneck | **Growing per-thread accumulator/register demand and shared-memory reduction work now expose row-scaling cost.** Weight traversal is still amortized, but no longer near constant-cost. |
| Decision | **RETAIN IN QUARANTINE** and build one final exact-batch-16 candidate. Do not infer batch-16 viability from batch 8. |
| Redesign | H014-028d tests exactly 16 rows with a preregistered `8.00x` aggregate-speedup floor (50% ideal efficiency), bit-exact prior-size preservation, and the identical fail-closed health protocol. The result will set the primitive ceiling. |

| Field | H014-028d |
|---|---|
| Hypothesis | The exact 16-row cooperative real MXFP4 expert path provides at least `8.00x` aggregate throughput versus 16 batch-1 calls (at least 50% ideal efficiency), remains bit exact, preserves batch 1/2/4/8 within `5%`, and leaves CUDA, VRAM and the physical GPU healthy for a known-safe fixture. |
| Implementation | Instantiated 16-row gate/up-pair and down-projection kernels in a new candidate with the exact supported set `(1,2,4,8,16)`. The harness completed each smaller size before arming 16, queried sticky CUDA state after it, ran batch 1 afterward, and rejected batch 3/32 before native CUDA. No prior/deployed binary was modified. |
| Benchmark | `performance/h014-028d-row-cooperative-batch16.json` SHA `9505e06e...180f`; candidate SHA `cbaa45f1...88ff`; prior batch-8 candidate SHA `70d65478...a398`; real layer-89 expert 803, 50 warmups and 300 retained calls, exact order `1 -> 2 -> 4 -> 8 -> 16 -> 1`, with 16 serial controls. Batch-16 templates were present in sm_86 SASS and compute_86 PTX. |
| Result | **EFFICIENCY HYPOTHESIS FALSIFIED; SAFETY/CORRECTNESS PASS**. Batch-16 device p50/p95/p99 were `0.178112/0.180290/0.181641 ms`; wall values were `0.184200/0.186805/0.189981 ms`. Sixteen serial p50 values summed to `1.182160 ms`, yielding `6.637172x`, below `8.00x`, but still `89,831.117 rows/s` and `0.011132 ms` per row. Output and all prior sizes were bit exact; prior-size timing ratios were `0.962327/0.951386/0.968592/0.981487`. Every CUDA error, synchronization, safe-fixture, memory, preflight and same-UUID health gate passed. |
| Inspection | Batch 16 is a safe, higher-capacity primitive, but only `41.5%` efficient relative to ideal and raises call latency `62.5%` over batch 8 for `23.1%` more aggregate rows/s. The stable p95/p99 indicate a deterministic kernel-resource bend, not jitter. |
| Bottleneck | **Batch-16 row accumulation/reduction resource pressure dominates the marginal capacity gain.** The exact compiled register/shared-memory mechanism still requires inspection; complete-stage effects must decide whether the extra capacity is useful. |
| Decision | **CERTIFY BATCH 16 AS SAFE BUT DO NOT SELECT IT FROM PRIMITIVE EVIDENCE.** Retain batch 8 as the provisional efficiency ceiling and retain batch 16 for complete-stage comparison. No batch above 16 will be attempted. |
| Redesign | H014-028e inspects compiled resource usage and attempted hardware counters to identify the batch-16 bend. Then H014-029 benchmarks complete real layer-89 KDA+MoE and late Gated-MLA+MoE at batch 1/8/16 before fixing the production batch. |

| Field | H014-028e |
|---|---|
| Hypothesis | The batch-16 efficiency bend is consistent with the fused gate/up kernel crossing an sm_86 residency boundary: its theoretical warp-occupancy upper bound falls from `100%` at batch 8 to at most `50%` at batch 16, while the single down kernel remains at `100%` and neither kernel spills to local/stack memory. |
| Implementation | Added an evidence-only parser over `cuobjdump --dump-resource-usage`, calculated documented sm_86 static occupancy bounds for 256-thread blocks, joined the immutable batch-2/4/8/16 receipts, and attempted Nsight Compute counter access. No CUDA kernel executed and no binary changed. |
| Benchmark | `performance/h014-028e-row-cooperative-resource-inspection.json` SHA `78e6623b...40bd`; exact batch-16 binary SHA `cbaa45f1...88ff`; all eight pair/single template variants found. Nsight Compute 2025.3.1 returned `ERR_NVGPUCTRPERM`, retained explicitly as a blocked counter attempt. |
| Result | **SUPPORTED / PASS**. Pair-kernel resources progress `32/4 KiB`, `36/8 KiB`, `40/16 KiB`, `56/32 KiB` registers-per-thread/shared-per-block at batch `2/4/8/16`. Static sm_86 resident blocks are `6/6/6/3`, giving warp-occupancy upper bounds `100/100/100/50%`. The single kernel remains six blocks/SM and `100%` through batch 16. Batch-16 local and stack bytes are zero. Observed ideal efficiency falls `67.017% -> 41.482%` from batch 8 to 16. |
| Inspection | The fused gate/up pair—not the down projection—crosses both a shared-memory and register residency boundary at 16. This is compiled sm_86 evidence and matches the stable local timing bend. Achieved occupancy and physical DRAM bytes were not measured because counter access is disabled; no values are inferred from source or effective bandwidth. |
| Bottleneck | **Batch-16 fused gate/up occupancy pressure; no compiler spill.** |
| Decision | **RETAIN batch 8 as the primitive efficiency candidate and batch 16 as a safe capacity comparator.** Select between them only with complete-stage evidence. Do not attempt a larger expert batch. |
| Redesign | H014-029 groups actual per-row routed expert selections by expert ownership for complete layer-89 KDA+MoE and late Gated-MLA+MoE batches, measures real overlap/reuse, and compares batch 1/8/16 end-to-end. |

| Field | H014-029a |
|---|---|
| Hypothesis | A complete production layer-89 control on the H014-028d candidate will preserve the latest retained serial oracle before any row-cooperative batch is armed, after which real-route grouping can be tested for at least `1.5x` complete-stage aggregate throughput. |
| Implementation | Added a fixed pre-READY batch workspace and a production-stage static-batch path with independent attention state, exact per-row routing, expert-ID gather/cooperative-execute/scatter, unchanged deterministic reduction, batched latent projections/shared expert, and atomic per-size receipts. The serial control ran before batch 1. |
| Benchmark | Failed control `performance/h014-029a-complete-stage-batch-layer89.json` SHA `ceccb24b...2125`; inspection `performance/h014-029a-invalid-control-inspection.json` SHA `e3abe8e3...f891`. No batch size executed. Candidate/deployed layer-89 shared-expert isolation receipts SHA `9dd2212c...7e3` / `9b6cb5e8...d5d` used the same activation and real weights. |
| Result | **INVALID HARNESS INPUT / NO CUDA-BATCH RESULT**. Serial routes were exact but reported maximum relative L2 `2.218e-4`. The configured trace SHA was `95012ceb...6ae`, while H014-027w's exact-binary graph certificate names deterministic trace SHA `0a432e25...a8d`. Candidate and deployed shared-expert outputs were bit exact (`sha256:c5266f4e...a534`) with identical `4.039e-7` oracle error. CUDA and `nvidia-smi` remained healthy. |
| Inspection | The arithmetic-regression hypothesis was falsified by the direct shared-expert binary comparison. The control paired the later deterministic-router binary lineage with a superseded non-idot0 trace; its expected boundaries did not describe the configured execution. |
| Bottleneck | **Harness provenance, not CUDA arithmetic.** The first harness trusted caller-supplied oracle paths without mechanically joining them to the exact passing graph receipt. |
| Decision | **RETAIN AS AN INVALID ATTEMPT.** Do not treat its numerical mismatch as an H014-028d regression, do not hide it, and do not modify CUDA from this result. |
| Redesign | Require a passing graph-certification receipt and exact trace/routes SHA equality before loading weights or touching CUDA, then repeat unchanged as H014-029b. |

| Field | H014-029b |
|---|---|
| Hypothesis | On the graph-certified deterministic oracle, real-route row grouping plus batched latent/shared work preserves serial semantics and yields at least `1.5x` complete layer-89 aggregate throughput at one incrementally certified size without GPU or lifecycle degradation. |
| Implementation | Added the fail-closed graph/oracle provenance join, then ran the production executor in exact order `1 -> 2 -> 4 -> 8 -> 16`. Every size received three-generation batch-versus-serial output/route/state comparison, retained p50/p95/p99 timing, phase timing, VRAM/lifecycle evidence, CUDA synchronization/sticky-error checks, a known-safe standard batch-1 fixture, and `nvidia-smi` before the next size. |
| Benchmark | `performance/h014-029b-complete-stage-batch-layer89.json` SHA `af3d11f9...f858`; candidate SHA `cbaa45f1...88ff`; graph receipt SHA `0c0a655a...4199`; trace/routes SHA `0a432e25...a8d` / `c734d864...1288`; 10 warmups, 50 retained calls and five phase calls per size. Persistent workspace was `8,486,912` bytes; stage residency `18,272,485,376` bytes; KDA state `6,881,280` bytes/stream. |
| Result | **MATERIAL-SPEED HYPOTHESIS FALSIFIED; BATCHES 1/2/4/8 PASS; BATCH 16 FAILS LIFECYCLE.** Batch 1/2/4/8/16 device p50 were `3.272/6.374/11.616/21.182/39.384 ms`, aggregate `305.6/313.8/344.4/377.7/406.3 rows/s`, and throughput gain versus the serial `3.039 ms` p50 was `0.929/0.954/1.047/1.148/1.235x`. All sizes were bit exact to serial with exact routes/state, healthy CUDA and a passing safe fixture. Batch 16 alone created one persistent parent thread in the aggregate retained window, so its complete-stage gate failed. |
| Inspection | Even the optimistic three-trace row rotation reached only `72.6%` repeated hits and `2.089` effective routed rows/native call at batch 8; batch 16 reached `86.3%` repeats and `3.343` rows/call but remained below `1.5x`. Batch-16 phase p50 was `17.784 ms` attention/pre-MoE, `6.924 ms` router, `6.482 ms` routed compute, `3.659 ms` dispatch+collection, and `2.555 ms` shared expert. Primitive reuse therefore moved the complete-stage bottleneck to serial attention/routing and gather/scatter. |
| Bottleneck | **Row-serial attention and routing dominate; per-selection D2D dispatch/collection is also exposed.** Routed-expert compute is no longer sufficient to make the complete stage scale materially. |
| Decision | **RETAIN batches through 8 as safe complete-stage executions; do not certify batch 16 at complete-stage level.** Keep the candidate quarantined and do not select a production batch from the optimistic three-route corpus alone. |
| Redesign | H014-029c localizes the batch-16 thread transition between retained execution, diagnostic profiling and state inspection before fixing the ceiling. |

| Field | H014-029c |
|---|---|
| Hypothesis | The batch-16 parent thread seen by H014-029b is research-side activity created during phase profiling or state inspection, not retained complete-stage execution. |
| Implementation | Added OS-thread/process/task/connection/model lifecycle snapshots immediately before retained execution, after retained calls, after phase profiling and after state inspection. CUDA math, batching, fixtures, iteration counts and size order were unchanged. |
| Benchmark | `performance/h014-029c-batch16-thread-localization-layer89.json` SHA `accb702e...6321`; same exact candidate/oracle, 10 warmups, 50 retained calls and five phase calls. All earlier sizes were again persisted and checked before batch 16. |
| Result | **FALSIFIED / DIAGNOSTIC FAIL AT BATCH 16.** Batch 1/2/4/8 again passed; batch 16 remained bit exact and GPU healthy but created two new persistent thread IDs (`29140`, `31692`) during retained execution. Phase profiling and state inspection each created zero. Batch-8 p50 was `21.142 ms`, `378.4 rows/s`, `1.152x`; batch 16 remained below the material floor. |
| Inspection | The batch-16 lifecycle transition is real retained-path host activity in this process, not an artifact of the diagnostic or state download. It is unnecessary to absorb into READY because the size already loses the preregistered efficiency test and crosses the compiled occupancy boundary. |
| Bottleneck | **Batch-16 host lifecycle transition plus the already measured fused-kernel occupancy cliff; no correctness or GPU-health failure.** |
| Decision | **REJECT batch 16 as a complete-stage production size and stop.** Batch 8 is the maximum locally certified complete-stage size; no batch above 16 is attempted. |
| Redesign | Certify the other required late architecture class only through the passing `1/2/4/8` prefix, without re-executing batch 16. |

| Field | H014-029d |
|---|---|
| Hypothesis | The same real-route batch-8 design preserves exact late Gated-MLA+MoE semantics and yields at least `1.5x` aggregate complete-stage throughput without state, lifecycle, memory or GPU degradation. |
| Implementation | Reused the unchanged production batch executor on real layer 91, permitting a strict passing prefix of a superset-capable binary so only `1 -> 2 -> 4 -> 8` executed. The graph/oracle SHA join, per-size serial equivalence, state fingerprints, phase decomposition, safe fixture and post-size health gates remained identical. |
| Benchmark | `performance/h014-029d-complete-stage-batch-layer91.json` SHA `8eddb73d...1509`; 10 warmups, 50 retained and five phase calls; candidate SHA `cbaa45f1...88ff`; stage residency `18,241,028,096` bytes, persistent batch workspace `8,486,912` bytes and measured MLA state `149,760` bytes/stream at the configured context. |
| Result | **CORRECTNESS/SAFETY PASS; MATERIAL-SPEED HYPOTHESIS FALSIFIED.** Batch 1/2/4/8 device p50 were `3.024/5.627/10.219/18.743 ms`, p95 `3.349/5.979/10.677/19.550 ms`, p99 `3.402/6.032/11.023/20.040 ms`, and aggregate `330.7/355.4/391.4/426.8 rows/s`. Batch-8 gain over serial `2.784 ms` was `1.188x`. Every batched output was bit exact to serial, routes and state fingerprints matched, lifecycle deltas were zero, and all CUDA/VRAM/known-safe/`nvidia-smi` gates passed. |
| Inspection | Batch-8 route overlap was `71.875%`, effective reuse `2.088` rows/native call and only `36.0` unique experts per 128 selections. Despite this favorable repeated three-trace corpus, phase p50 remained `6.518 ms` attention/pre-MoE, `3.475 ms` router, `4.644 ms` routed experts, `1.821 ms` dispatch+collection and `1.295 ms` shared expert. Architecture class changes absolute service but not the scaling mechanism. |
| Bottleneck | **Cross-row reuse is absent from attention and router, which together dominate the remaining batch service; D2D per-selection fanout is second-order but material.** |
| Decision | **RETAIN safe complete-stage batch 8 as the current maximum, but do not call static batching materially capacity-improving and do not start a continuous-batching scheduler.** Production-size selection remains pending a diverse-route corpus and attention/router redesign. |
| Redesign | H014-030 tests the measured bottleneck: row-cooperative KDA/MLA projection and batched-router weight reuse at batch 8 must lift complete-stage gain to at least `1.5x` while preserving independent state. If it cannot, retain batch 1 for latency and batch 8 only as a modest capacity mode, then proceed to the mandatory sub-layer program. |

| Field | H014-030a |
|---|---|
| Hypothesis | One real-weight batched router launch and one D2H result transfer preserve exact per-row top-16 semantics and cut batch-8 router service by at least `2x` versus eight serial native calls. |
| Implementation | Generalized the logits/select kernels over rows and added a fail-closed `coli_cuda_pipe_router_batch` ABI. The harness used eight distinct retained Kimi trace rows, compared the new serial path bit-for-bit with H014-028d, and certified only `1 -> 2 -> 4 -> 8`; batch 3 was rejected before native entry and batch 16 was not run. |
| Benchmark | `performance/h014-030a-batched-router.json` SHA `1fe22281...714f`; candidate SHA `cd7a10ec...a5b4`; 30 warmups and 200 retained calls/size; real layer-89 router/bias and trace SHA `0a432e25...a8d`. |
| Result | **PASS.** Batch-8 serial/batched wall p50 were `3.3572/0.4200 ms`, a `7.9933x` speedup and `19,047.6 rows/s`. IDs, weights and effective counts were bit exact at every size; CUDA synchronization, sticky state, safe serial fixture, VRAM and `nvidia-smi` passed after each increment. |
| Inspection | Batch-8 device decomposition was `0.372960 ms` logits, `0.032649 ms` selection and `0.047218 ms` D2H per call. Launch/D2H amortization worked; logits compute now dominates the primitive. |
| Bottleneck | **Complete-stage attention/pre-MoE and routed experts, not the batched router.** |
| Decision | **RETAIN in the quarantine candidate and test complete stages.** Do not promote from a primitive result. |
| Redesign | H014-030b/c rerun graph-proven layer-89 KDA and layer-91 MLA complete-stage batches against the unchanged H014-029 controls. |

| Field | H014-030b |
|---|---|
| Hypothesis | The router redesign preserves exact layer-89 semantics and lifts batch-8 aggregate complete-stage gain materially; the retained global material gate remains `1.5x`. |
| Implementation | Reused the exact complete-stage harness and graph/oracle SHA join with candidate SHA `cd7a10ec...a5b4`; only the router ABI/path changed. |
| Benchmark | `performance/h014-030b-complete-stage-batch-layer89.json` SHA `b4fe5737...4b68`; 10 warmups, 50 retained and five phase calls at `1/2/4/8`. |
| Result | **CORRECTNESS/SAFETY PASS; 1.5x HYPOTHESIS FALSIFIED.** Batch-8 device p50 fell from H014-029b's `21.1824` to `18.2174 ms`; gain rose from `1.1478x` to `1.3258x` (`439.14 rows/s`). All outputs were bit exact; routes/state/lifecycle and every GPU health gate passed. |
| Inspection | Router p50 fell from `3.4743` to `0.5483 ms`. Attention/pre-MoE became `8.6433 ms`, routed experts `4.7525 ms`, and dispatch+collection `1.9101 ms`. |
| Bottleneck | **Row-independent KDA projections dominate the complete-stage critical path.** |
| Decision | **RETAIN router; do not declare batching material yet.** |
| Redesign | Confirm the architecture-class effect on late Gated-MLA, then target projection weight reuse rather than another router change. |

| Field | H014-030c |
|---|---|
| Hypothesis | The batched router preserves exact late MLA semantics but complete-stage batch 8 still must reach `1.5x` before static batching is considered materially useful. |
| Implementation | Ran the passing `1/2/4/8` prefix on real layer 91 with the same candidate, oracle, state and lifecycle checks. |
| Benchmark | `performance/h014-030c-complete-stage-batch-layer91.json` SHA `81da45d5...63b0`; serial oracle error `8.971e-8`; 10/50/five warm/retained/phase calls. |
| Result | **CORRECTNESS/SAFETY PASS; MATERIAL GATE FALSIFIED.** Batch-8 device p50 was `16.6579 ms`, `480.25 rows/s`, and `1.3597x` gain versus serial; prior gain was `1.1883x`. Output, routes and state were exact and lifecycle/GPU checks passed. |
| Inspection | Attention/pre-MoE remained `6.5778 ms`, experts `4.6484 ms`, shared expert `1.2895 ms`, and router only `0.5777 ms`. |
| Bottleneck | **Row-independent attention projections now dominate both architecture classes.** |
| Decision | **RETAIN router and form a projection-reuse hypothesis.** |
| Redesign | H014-030d isolates one measured large real KDA projection before modifying the production attention path. |

| Field | H014-030d |
|---|---|
| Hypothesis | A batch-8 kernel that loads each real layer-89 KDA q-projection weight once for eight rows is bit exact and at least `2x` faster than the independent-row batch kernel. |
| Implementation | Added exact compile-time `2/4/8` row accumulators with the same per-thread input order and 256-way reduction tree; the ABI rejects unsupported sizes before CUDA. No production call site changed. |
| Benchmark | `performance/h014-030d-dense-row-reuse-layer89-q.json` SHA `56885721...7cc`; candidate SHA `5d33aded...0120`; real `7168 -> 12288` grouped-int4 q projection, eight distinct trace rows, 30 warmups and 200 retained calls/size. |
| Result | **PASS.** Batch-8 baseline/reuse device p50 were `0.806736/0.363056 ms`, `2.2221x`, with bit-exact outputs. Batch 4 reached slightly higher aggregate throughput (`22,486.1` versus `22,035.2 rows/s`). All safety/health checks passed. |
| Inspection | Compiled batch 4 uses 29 registers/4 KiB shared; batch 8 uses 40 registers/8 KiB with no spill. The resource rise explains the throughput bend without invalidating batch 8. |
| Bottleneck | **At projection level, row accumulator/resource pressure; at stage level, integration is now justified.** |
| Decision | **RETAIN for one production attention integration.** |
| Redesign | Batch stateless KDA/MLA projections while keeping KDA/MLA cache/state cores strictly session-owned. |

| Field | H014-030e |
|---|---|
| Hypothesis | Batched KDA projections plus the retained router/expert reuse push layer-89 batch-8 aggregate throughput over `1.5x` with exact state and zero warm lifecycle deltas. |
| Implementation | Allocated fixed attention scratch before READY; batched stateless q/k/v/gate/decay/output projections in exact `8/4/2/1` chunks; retained per-session KDA core/state, reduction and residual ordering. Workspace grew from `8,486,912` to `13,219,840` bytes. |
| Benchmark | `performance/h014-030e-complete-stage-batch-layer89.json` SHA `bc1dda9a...261a`; candidate SHA `5d33aded...0120`; exact graph/oracle, `1/2/4/8`, 10/50/five calls. |
| Result | **PASS / MATERIAL STATIC BATCHING PROVEN.** Batch-8 device p50 `13.6709 ms`, `585.18 rows/s`, `2.0337x` aggregate gain. Batch 4 already reached `1.7062x`. Outputs/routes/state were bit exact; lifecycle deltas and all CUDA/VRAM/health checks were zero/pass. |
| Inspection | Attention/pre-MoE fell to `4.2944 ms`; routed experts became the largest phase at `4.7646 ms`. Router was `0.6268 ms`; the 13.22 MB workspace is negligible beside `18.279 GB` stage residency. |
| Bottleneck | **Routed experts now narrowly dominate KDA batch-8 service.** |
| Decision | **RETAIN batch 8 as a materially useful static capacity mode, pending diverse-route validation.** |
| Redesign | Certify the same production path on late MLA before starting continuous batching. |

| Field | H014-030f |
|---|---|
| Hypothesis | The row-reuse attention design preserves late MLA cache/output semantics and clears `1.5x` complete-stage gain at batch 8. |
| Implementation | Applied the same fixed-workspace design to batched query/kv/gate/output projections while leaving cache append and absorb state session-owned. |
| Benchmark | `performance/h014-030f-complete-stage-batch-layer91.json` SHA `8c1093c3...1640`; exact graph/oracle, `1/2/4/8`, 10/50/five calls; workspace `11,374,592` bytes. |
| Result | **PASS.** Batch-8 device p50 `13.1691 ms`, `607.48 rows/s`, `1.9012x`; batch 4 `1.6016x`. Outputs/routes/MLA state were bit exact and lifecycle/GPU gates passed. |
| Inspection | Experts dominate at `4.8265 ms`, attention/pre-MoE is `3.8794 ms`, and router is `0.5436 ms`. The same mechanism generalizes across attention classes. |
| Bottleneck | **Routed expert work, followed by attention and per-selection gather/scatter.** |
| Decision | **RETAIN; static batching is materially capacity-improving.** Production size still awaits diverse routes and scheduler evidence. |
| Redesign | Open continuous batching and the mandatory real sub-layer expert-parallel program. |

| Field | H014-SUB-001a |
|---|---|
| Hypothesis | Two persistent disjoint expert partitions can execute one real layer exactly while each owns less than the complete layer. |
| Implementation | Loaded a complete resident reference, two isolated 448-expert CUDA worker processes and an expertless parent coordinator, then began the seven-call collective PREPARE fixture. |
| Benchmark | Failed receipt `sub-layer/h014-sub-001a-two-worker-real-expert.json` SHA `aa122cf4...6313`; inspection SHA `dc3e1adf...b9ac`. Reference and both workers reached READY; no distributed correctness/performance call completed. |
| Result | **INVALID HARNESS / NO MICROWORK RESULT.** First dispatch rejected returned geometry. GPU stayed healthy and all process memory was recovered. |
| Inspection | Worker code used the checkpoint-derived latent width `3584`; the collective had a hard-coded `512`. Shutdown also consumed a queued RESULT before CLOSED and attempted to JSON-serialize its ndarray. |
| Bottleneck | **Harness geometry and terminal-frame draining, not CUDA arithmetic.** |
| Decision | **RETAIN FAILED ATTEMPT.** Do not count it as a sub-layer correctness failure. |
| Redesign | Derive collective width from the coordinator and drain queued result frames until CLOSED; repeat unchanged as H014-SUB-001b. |

| Field | H014-SUB-001b |
|---|---|
| Hypothesis | Corrected two-worker execution resolves and executes all 16 real selected experts exactly once, preserves full-layer output/state, and puts every worker below complete-layer memory. |
| Implementation | Persistent processes own experts by `expert_id mod 2`; each holds 448 real checkpoint experts and one fixed input/output workspace. The parent retains router, latent projections, shared experts, deterministic reduction and residual semantics; real `3584`-float activations/results cross measured pipe frames. |
| Benchmark | `sub-layer/h014-sub-001b-two-worker-real-expert.json` SHA `5d9bd1b3...d53f`; 7-call collective PREPARE, three graph-certified correctness generations, 10 warmups and 30 retained calls. |
| Result | **LOGICAL SUB-LAYER CORRECTNESS/MEMORY PASS.** Full-layer error `0.0`; IDs, route weights, ownership, all 16 calls and KDA state exact. Each worker holds `7,861,417,984` bytes (`43.008%` of the `18,278,776,832`-byte complete resident layer). Warm lifecycle deltas are zero. |
| Inspection | Same-GPU distributed/reference complete-layer p50 were `4.1462/3.2635 ms` (`78.710%` throughput). Expert roundtrip was `1.7261 ms`, critical worker device `0.6604 ms`; 229,376 output bytes and 259,523 mean framed bytes/token expose transport. |
| Bottleneck | **Process transport/coordination, not correctness or worker compute.** |
| Decision | **RETAIN as logical proof; physical multi-GPU efficiency remains unproven.** |
| Redesign | Increment to four disjoint workers with explicit memory, critical-path and retention predictions. |

| Field | H014-SUB-002 |
|---|---|
| Hypothesis | Four workers keep exactness, fall below 25% layer memory, improve critical worker p50 at least 35% from two workers (`<=0.42925 ms`), and retain at least 70% same-GPU layer throughput. |
| Implementation | Four persistent processes each own 224 experts. A receipt-plumbing bug parsed but failed to forward the two performance thresholds; a separate audit preserves the preregistered command thresholds without changing numerical evidence. |
| Benchmark | Raw receipt `sub-layer/h014-sub-002-four-worker-real-expert.json` SHA `02f9b829...e5bf`; threshold audit SHA `93c20a3f...8fdb`; identical 7/3/10/30 PREPARE/correctness/warm/retained protocol. |
| Result | **CORRECTNESS/MEMORY/RETENTION PASS; CRITICAL-PATH HYPOTHESIS FALSIFIED.** Error `0.0`; `3,930,830,848` bytes/worker (`21.505%`); retention `81.371%`; critical worker p50 `0.4780 ms`, only `27.6%` better than two workers. |
| Inspection | Real routes place 2–6 selected experts on a contacted worker. Mean framed bytes rose to `289,551` and all four workers were contacted. |
| Bottleneck | **Routing imbalance plus fixed per-worker messaging prevents proportional compute scaling.** |
| Decision | **RETAIN characterization; do not claim the performance prediction passed.** |
| Redesign | Test eight workers with a weaker evidence-based critical-path and retention gate. |

| Field | H014-SUB-003 |
|---|---|
| Hypothesis | Eight workers remain exact, put each worker below 12% of full-layer residency, keep critical worker p50 `<=0.42 ms`, and retain at least 65% same-GPU layer throughput. |
| Implementation | Eight persistent processes each own 112 disjoint real experts; all are admitted sequentially before PREPARE. |
| Benchmark | `sub-layer/h014-sub-003-eight-worker-real-expert.json` SHA `68fccd46...1de7`; same 7/3/10/30 protocol and health checks. |
| Result | **PASS.** Error/state delta `0`; `1,965,537,280` bytes/worker (`10.753%`); critical device p50 `0.2900 ms`; distributed p50 `4.6184 ms`; retention `79.416%`. |
| Inspection | Fanout averages `7.867/8`, exposed coordination `0.7591 ms`, roundtrip `2.0455 ms`, and mean framed traffic `347,608` bytes/token. Compute improves while the collective worsens. |
| Bottleneck | **Fanout, serialization and collection.** |
| Decision | **RETAIN and execute the final required 16-worker point; do not extrapolate.** |
| Redesign | H014-SUB-004 tests the smallest static equal partition and stops at 16. |

| Field | H014-SUB-004 |
|---|---|
| Hypothesis | Sixteen workers remain exact, put each below 6% of full-layer residency, keep critical p50 `<=0.25 ms`, and retain at least 55% same-GPU layer throughput. |
| Implementation | Sixteen isolated persistent processes each own 56 experts. Admission remained sequential and fail-closed; no topology above 16 was attempted. |
| Benchmark | `sub-layer/h014-sub-004-sixteen-worker-real-expert.json` SHA `7e303916...2dd4`; same real layer/trace and 7/3/10/30 protocol. |
| Result | **PASS.** Error/state delta `0`; `982,890,496` bytes/worker (`5.377%`); critical device p50 `0.1602 ms`; complete-layer p50 `5.1531 ms`; retention `71.620%`; all 16 workers closed healthy. |
| Inspection | Only 1–2 selected experts land on a contacted worker, but mean fanout is `10.867`, exposed coordination `0.9330 ms`, roundtrip `2.4166 ms`, and framed traffic `392,660` bytes/token. Defined sub-layer efficiency falls to `3.057%`. |
| Bottleneck | **Coordination overwhelms the now-small expert compute.** |
| Decision | **RETAIN as the scaling endpoint and stop adding workers.** |
| Redesign | Replay measured payloads under physical RTT/bandwidth profiles and select the useful domain/topology. |

| Field | H014-SUB-005 |
|---|---|
| Hypothesis | Real measured expert payloads define a non-empty physical-link region retaining at least 90% of resident complete-layer capacity. |
| Implementation | Built a conservative counterfactual replay: measured parent non-expert p50 + measured critical worker device p50 + measured loopback coordination base + RTT + critical-path payload/bandwidth. Evaluated loopback evidence, RTT `0.1/0.25/0.5/1/2/5/10/20 ms` and bandwidth `0.1/0.25/0.5/1/2.5/5/10/25/100 Gbps` for all four topologies. |
| Benchmark | Scaling/network JSON `sub-layer/h014-sub-005-scaling-network.json` SHA `dbb6cb5e...4b96`; 288-row matrix CSV SHA `2c7fb355...8be`; chart SHA `7ba292a2...2f3`. This is a replay, not a physical multi-GPU measurement. |
| Result | **PASS / USEFUL REGION EXISTS ONLY IN FAST DOMAINS.** At 100 Gbps, modeled 90%-capacity max RTT is `0.521/0.646/0.536/0.495 ms` for `2/4/8/16`; max tested is `0.5 ms` except 16 (`0.25 ms`). At `0.25 ms`, exact minimum bandwidth is `4.446/1.986/1.963/1.383 Gbps` (tested tiers `5/2.5/2.5/2.5`). |
| Inspection | Four workers provide the widest RTT envelope and `3.661 GiB` worker state. Strict break-even RTT at 100 Gbps is only `0.159/0.237/0.129/0.085 ms`; WAN profiles collapse capacity. |
| Bottleneck | **Sub-millisecond latency, then serialization/fanout; bandwidth is secondary above a few Gbps.** |
| Decision | **RETAIN FOUR-WORKER EXPERT GROUPS ONLY INSIDE FAST DOMAINS.** Physical multi-GPU canary remains mandatory before fleet use. |
| Redesign | Test shared-expert placement, sub-layer batching, failure/recovery and a small physical two-GPU canary package; compare with coarse whole-layer transport before final topology selection. |

| Field | H014-031a |
|---|---|
| Hypothesis | A persistent position-aware FIFO scheduler can combine eight independent real Kimi layer-89 decode streams, including unequal cache positions, while remaining bit exact to serial execution, retaining at least `1.7x` aggregate device capacity versus batch 1, completing the same number of retained tokens per active stream, and supporting fail-closed cancellation plus fresh-slot reuse without state interference. |
| Implementation | Extended the certified batch executor to validate one cache position per row and added a bounded FIFO over persistent stage sessions. The scheduler creates no worker threads, tasks or connections and changes no CUDA arithmetic or native batch ceiling. |
| Benchmark | `performance/h014-031a-continuous-batch-layer89.json` SHA `7847f570...382f`; real layer 89, eight streams staggered at positions `0..7`, three exact serial-comparison rounds, 10 warmup and 50 retained rounds, cancellation/replacement, safe fixture and GPU checks. |
| Result | **CORE HYPOTHESIS PASS; LATENCY-TAIL EVIDENCE INVALID.** Output error `0`, exact routes and state; device p50/p95/p99 `13.319/13.601/13.679 ms`; batch-1 p50 `3.2966 ms`; capacity gain `1.9800x`; `600.63` device rows/s and `559.60` wall rows/s. Every stream completed 50 rows, lifecycle deltas were zero, cancellation removed the pending row, stale submission failed closed, replacement state was exact, and CUDA remained healthy. |
| Inspection | Normal queue p50 was `0.0449 ms`, but the first retained lifecycle snapshot ran after enqueue and injected a `79.5988 ms` pause. Therefore queue/formation/response/cadence p99 from this receipt are not valid. Routing achieved `2.0867` rows/native expert call; eight KDA states consumed `55,050,240` bytes. |
| Bottleneck | **Complete-stage CUDA service in normal rounds; measurement ordering, not CUDA, contaminated the retained host tail.** |
| Decision | **RETAIN position-aware scheduler, exactness, capacity, fairness, state and recovery evidence. Do not retain the queue/response tail.** |
| Redesign | H014-031b moves the unchanged lifecycle snapshot before enqueue and repeats the same workload once. |

| Field | H014-031b |
|---|---|
| Hypothesis | Moving retained lifecycle inspection before FIFO enqueue removes the instrumentation outlier while preserving exact health and at least `1.7x` capacity; queue-formation p99 will be below `0.5 ms` and end-to-end response p99 below `16 ms`. |
| Implementation | Moved the retained process/lifecycle snapshot before requests enter the FIFO. Scheduler policy, state offsets, CUDA path, fixtures and iteration counts were unchanged. |
| Benchmark | `performance/h014-031b-continuous-batch-layer89.json` SHA `e434b1c2...bdf3`; the exact H014-031a protocol repeated once. |
| Result | **EXECUTION/CAPACITY/QUEUE PASS; RESPONSE-TAIL HYPOTHESIS FALSIFIED.** Exact output/routes/state; capacity `2.0312x`; device p50/p95/p99 `13.103/14.235/14.810 ms`; `610.53` device and `565.30` wall rows/s. Formation p99 `0.1464 ms` passed, but response p99 `16.3180 ms` missed the `16 ms` gate by `0.3180 ms`. Fairness remained `1.0`, cancellation/reuse passed, lifecycle remained zero and CUDA stayed healthy. |
| Inspection | The H014-031a 79 ms artifact disappeared, validating the measurement-order diagnosis. Per-stream cadence p50/p95/p99 was identical across all streams at `14.883/16.114/16.600 ms`. |
| Bottleneck | **Device p99 `14.810 ms`, then host boundary delivery; queue formation is negligible.** |
| Decision | **RETAIN continuous batch 8 as the production capacity candidate, but modify host delivery before closing its tail gate.** |
| Redesign | H014-031c removes one redundant copy of the already-owned `2,064,384`-byte batch boundary; no CUDA or scheduler-policy change. |

| Field | H014-031c |
|---|---|
| Hypothesis | Removing the scheduler's redundant host copy of the already-owned `2,064,384`-byte batch output preserves exact execution and device service while reducing response p99 below `16 ms`; formation p99 remains below `0.5 ms` and capacity remains at least `1.7x`. |
| Implementation | Removed only the redundant `copy()` of the executor-owned fresh NumPy batch boundary. Row views retain ownership; CUDA, FIFO policy, fixtures and state semantics were unchanged. The harness now reports execution status separately from hypothesis gates. |
| Benchmark | `performance/h014-031c-continuous-batch-no-copy-layer89.json` SHA `189505b5...b33d`; unchanged eight-stream layer-89 protocol, 10 warmup and 50 retained rounds. |
| Result | **PASS.** Exact output/routes/state, cancellation/reuse and all health gates pass. Capacity was `1.9963x`; device p50/p95/p99 `13.120/13.358/13.539 ms`; formation p99 `0.1304 ms`; response p99 `15.0725 ms`; wall/device aggregate `567.11/609.77 rows/s`. |
| Inspection | Per-stream cadence p50/p95/p99 was `14.225/14.735/15.351 ms`, identical for all eight streams; queue p99 was `0.0965 ms`. Removing the redundant `2,064,384`-byte copy reduced the observed host tail without affecting device service or lifecycle. |
| Bottleneck | **Complete-stage CUDA service (`13.539 ms` p99), not FIFO formation.** |
| Decision | **RETAIN position-aware continuous batch 8 and the no-copy delivery path.** This is the safe production capacity candidate; batch 1 remains the minimum-latency mode. |
| Redesign | Separate decode and prefill workloads, characterize sub-layer batching/shared-expert placement/recovery, then compare the fine collective with retained coarse transport before capacity modeling. |

| Field | H014-032a |
|---|---|
| Hypothesis | Two independently owned persistent CUDA workers can keep real Kimi stages 0 and 1 simultaneously resident, transmit the canonical `float32 [1,9,7168]` boundary over one persistent length-prefixed TCP connection, and reproduce the graph-certified layer-1 output/routes/state with no warm lifecycle recreation; exposed loopback transport and serialization p50 will be below `1.5 ms`. |
| Implementation | Added two isolated production executors and persistent length-prefixed TCP links. Stage 1 loaded before stage 0; both completed PREPARE and remained simultaneously alive before root admission. The harness incorrectly invoked stage zero through `execute_decode(token_ids=...)`. |
| Benchmark | Failed receipt `coarse/h014-032a-stage0-stage1-tcp.json` SHA `d67f21d7...5235`; no user CUDA execution or retained transport benchmark occurred. |
| Result | **FALSIFIED / INVALID EXECUTION BENCHMARK.** Simultaneous residency succeeded at `18,278,776,832 + 3,049,259,008 = 21,328,035,840` bytes with both processes alive and `8,985 MiB` reported free. The first user call failed in Python with `TypeError` before stage-0 CUDA because embedding-owner tokens belong to `execute_prefill`, not `execute_decode`. GPU health recovered to `30,318 MiB` free and `nvidia-smi` remained normal. |
| Inspection | Cleanup attempted a session-close frame after the worker had already closed its socket, so the outer receipt failure became `ConnectionResetError`; the nested retained traceback preserves the authoritative `TypeError`. |
| Bottleneck | **Harness API misuse and error-preservation ordering; not memory, CUDA or transport.** |
| Decision | **RETAIN only simultaneous-residency and failure evidence. Do not use for correctness or performance.** |
| Redesign | H014-032b changes the single stage-zero call to `execute_prefill` and suppresses cleanup socket errors so primary failures remain authoritative. |

| Field | H014-032b |
|---|---|
| Hypothesis | With stage zero invoked through its public embedding-owner `execute_prefill` API, the unchanged simultaneous two-worker TCP slice reproduces graph-certified layer-1 execution and exposes less than `1.5 ms` loopback transport/serialization p50. |
| Implementation | Cleanup errors were suppressed correctly, but a non-contextual patch changed stage 1's earlier textual `execute_decode` occurrence to `execute_prefill`; stage 0 remained unchanged. |
| Benchmark | Failed receipt `coarse/h014-032b-stage0-stage1-tcp.json` SHA `69704cf1...1329`; both workers again reached simultaneous READY, but no user CUDA execution occurred. |
| Result | **FALSIFIED / INVALID EXECUTION BENCHMARK.** The exact stage-0 `TypeError` was now preserved as the top-level failure, proving the cleanup redesign. The intended API correction was not applied to stage 0, and an unreached secondary stage-1 API defect was introduced. GPU health remained normal with `30,318 MiB` free after teardown. |
| Inspection | The failed patch selected by occurrence order rather than layer-specific context. This is a code-review failure, not evidence about coarse transport. |
| Bottleneck | **Incorrect patch targeting.** |
| Decision | **RETAIN failure and error-preservation evidence only.** |
| Redesign | H014-032c patches both sites with explicit layer context: stage 0 `execute_prefill(token_ids=...)`, stage 1 `execute_decode(hidden_states=...)`. |

| Field | H014-032c |
|---|---|
| Hypothesis | With both layer-specific public calls explicitly correct, the unchanged simultaneous two-worker TCP slice reproduces graph-certified layer-1 execution and exposes less than `1.5 ms` loopback transport/serialization p50. |
| Implementation | Explicitly bound stage 0 to `execute_prefill(token_ids=...)` and stage 1 to `execute_decode(hidden_states=...)`; the source locations were reviewed before GPU load. No CUDA, transport or fixture behavior changed. |
| Benchmark | `coarse/h014-032c-stage0-stage1-tcp.json` SHA `3afdd182...f970`; three correctness positions, 10 warmup and 50 retained calls through two isolated processes and one persistent TCP edge. |
| Result | **PASS.** Inter-stage activation fingerprints were bit exact on every call; graph-relative output error was `2.7521e-7`; routes and both KDA states passed. Stage-0/stage-1 device p50 were `2.1672/3.0811 ms`; end-to-end wall p50/p95/p99 `6.5452/6.8633/7.3470 ms`; exposed transport+serialization p50/p95/p99 `0.4098/0.5891/0.6145 ms`. Warm lifecycle deltas were zero and CUDA/VRAM/`nvidia-smi` passed. |
| Inspection | Two workers simultaneously held `3,049,259,008` and `18,278,776,832` bytes (`21,328,035,840` total) with `8,985 MiB` free. The actual activation was `258,048` bytes; its production-direction frame was `258,283` wire bytes. The `258,860`-byte reverse frame is harness collection, not production reverse traffic. |
| Bottleneck | **Stage-1 compute (`3.0811 ms` p50); loopback edge exposure is `0.4098 ms`.** |
| Decision | **RETAIN as the real logical coarse distributed Kimi slice.** It is two independent processes over real TCP on one GPU, not a physical multi-GPU timing claim. |
| Redesign | H014-032d replays the measured one-way `258,283`-byte frame under coarse RTT/bandwidth profiles and separates capacity from per-stream latency. |

| Field | H014-032d |
|---|---|
| Hypothesis | The measured FP32 coarse frame has a non-empty practical network region that retains at least 90% of ideal overlapped two-stage pipeline capacity; latency and bandwidth thresholds will be materially looser than the sub-layer expert collective. |
| Implementation | Replayed measured stage p50, `0.40975 ms` loopback/serialization base and the one-way `258,283`-byte wire frame with `edge = base + RTT/2 + bytes/bandwidth`; ideal pipeline capacity uses the maximum of edge and slowest-stage service. |
| Benchmark | `coarse/h014-032d-network-analysis.json` SHA `d0463d4c...b92e`; 72-row CSV SHA `18e6f860...ed4c`; visually verified chart SHA `a0435da2...942e`; FP32 edge class SHA `c844fc06...5fec`. Profiles cover all preregistered RTT/bandwidth pairs. |
| Result | **PASS.** Exact 90%-capacity maximum RTT at 100 Gbps is `5.9860 ms`; at `0.25 ms`, exact minimum bandwidth is `0.7153 Gbps`. The tested `5 ms / 5 Gbps` admission point retains `92.719%` capacity. No tested bandwidth is viable at 10 or 20 ms RTT. |
| Inspection | The coarse exact RTT envelope is `9.27x` the four-worker fine envelope (`5.986/0.646 ms`); its exact 0.25-ms bandwidth floor is lower (`0.715/1.986 Gbps`). At 5 ms RTT, FP32 requires `4.023 Gbps`, so bandwidth remains material for cheaper links and per-stream latency. |
| Bottleneck | **RTT once edge service exceeds the `3.0811 ms` slow stage; bandwidth below a few Gbps at multi-millisecond RTT.** |
| Decision | **RETAIN FP32 edge class provisionally at <=5 ms RTT and >=5 Gbps.** This is capacity admission, not yet the final per-user cadence requirement. |
| Redesign | H014-032e tests actual FP16/BF16 boundary round-trips through real stage-1 CUDA under the existing strict `3e-5` output and exact-route gates. |

| Field | H014-032e |
|---|---|
| Hypothesis | FP16 serialization of the actual real stage-0 boundary halves coarse payload bytes while preserving exact layer-1 routes and graph-relative output error `<=3e-5`; BF16 is characterized under the same gate but is not assumed to pass. |
| Implementation | Kept stages 0 and 1 simultaneously resident in one process, reproduced all three H014-032c stage-0 fingerprints, and advanced isolated stage-1 FP32/FP16/BF16 state streams after host codec round-trip. No CUDA math changed. |
| Benchmark | `coarse/h014-032e-boundary-formats.json` SHA `2be608db...3d01`; three stateful real boundaries plus 100 retained host codec iterations/format. |
| Result | **EXECUTION PASS; FP16 HYPOTHESIS FALSIFIED.** FP32 graph error was `2.7521e-7` and passed. FP16 halved payload `258,048 -> 129,024` bytes and kept exact routes, but error was `1.7285e-4` (`5.76x` over gate). BF16 kept exact routes but error was `1.3920e-3` (`46.4x` over gate). All states were finite/isolated, lifecycle deltas zero and GPU health passed. |
| Inspection | FP16/BF16 codec p50 was `0.1815/0.1441 ms`; smaller payload cannot compensate for violating the fixed numerical contract. Exact routing alone is insufficient because residual/state values diverge. |
| Bottleneck | **Numerical precision, not codec speed.** |
| Decision | **REJECT FP16 AND BF16; RETAIN FP32 as the final canonical coarse boundary.** Do not run compressed TCP or weaken the gate. |
| Redesign | Continue with sub-layer batching, shared-expert placement and fail-closed recovery using the now-final separate coarse/fine edge semantics. |

| Field | H014-SUB-006 (preregistered) |
|---|---|
| Hypothesis | A four-worker persistent expert group can incrementally execute real batch `1 -> 2 -> 4 -> 8` with exact routes/output/state, and batch 8 will reuse repeated expert weights at least `2.0` rows/native call, reduce worker messages per row by at least 75% versus distributed batch 1, retain at least 50% of the resident one-GPU batch-8 complete-layer throughput, and improve aggregate distributed capacity at least `1.5x` over eight serial distributed rows. |
| Implementation | **NOT RUN.** Proposed minimum change adds an `EXECUTE_BATCH` command to the existing disjoint worker protocol and uses the production coordinator's external expert seam. Router, shared experts, reduction, residual and state remain at the parent; each worker owns the same 224 experts as H014-SUB-002. |
| Benchmark | **NOT RUN.** Real layer 89, exact graph fixtures, full resident reference measured then released, four persistent worker processes, incremental 1/2/4/8 correctness and retained service, per-size CUDA sync/health/VRAM/`nvidia-smi`; report unique/repeated experts, worker batches, dispatch/reduction bytes, native calls, load balance, message reduction and complete-layer relative throughput. |
| Result | **NOT RUN.** |
| Inspection | **NOT RUN.** |
| Bottleneck | **UNKNOWN pending real batched collective.** |
| Decision | **PENDING.** |
| Redesign | Stop at the first failing size. If exact but uneconomic, retain characterization and keep batch reuse local to whole-layer stages. |

| Field | H014-SUB-006a |
|---|---|
| Hypothesis | The new external-expert seam and batched persistent worker command can reach the preregistered incremental batch-1 correctness gate without changing resident whole-layer behavior. |
| Implementation | Added an optional external expert dispatcher to the production complete-stage batch path and an `EXECUTE_BATCH` command to the existing persistent disjoint expert workers; all routing, shared-expert, reduction, residual and state logic remained parent-owned and unchanged. |
| Benchmark | `sub-layer/h014-sub-006a-batch-harness-failure.json` SHA `4b49aaab...1cf67`; real layer 89, exact retained graph trace, batch sizes armed incrementally. |
| Result | **FAIL before batch 1.** The resident executor loaded all 896 experts, then the local reference path raised `AttributeError: _complete_expert_ownership`; no distributed expert CUDA call ran. Post-failure `nvidia-smi` remained measured with `29,821 MiB` free. |
| Inspection | The fail-closed local/external branch referred to a cached ownership attribute that the constructor never defined, although the authoritative `_expert_ownership` set was already present. This is harness integration evidence, not evidence against batched microwork. |
| Bottleneck | **Uninitialized Python ownership flag before CUDA execution.** |
| Decision | **MODIFY; retain the failed receipt.** Replace the nonexistent flag with an exact comparison against the existing ownership set. No CUDA source, weight assignment or numerical gate changes. |
| Redesign | H014-SUB-006b predicts that deriving completeness as `len(_expert_ownership) == 896` restores the resident reference and reaches worker startup. |

| Field | H014-SUB-006b |
|---|---|
| Hypothesis | Deriving complete ownership from the authoritative resident expert set restores the unchanged resident batch reference and allows the four-worker collective to start. |
| Implementation | Replaced only the missing cached flag with `len(self._expert_ownership) != self.config.experts` in the local batch fail-closed check. |
| Benchmark | `sub-layer/h014-sub-006b-four-worker-batch.json` SHA `e293fbba...5b98`; real layer 89 resident batches `1/2/4/8`, followed by four-worker startup. |
| Result | **PARTIAL PASS, THEN HARNESS FAIL.** The real resident reference completed all four sizes; batch-8 wall p50 was `14.77355 ms`, `34.75` mean unique experts were selected and native reuse was `2.09150` rows/call. Worker creation then stopped before loading workers because `_start_worker` requires a multiprocessing context as its first positional argument and all remaining arguments keyword-only. GPU health remained measured with `29,821 MiB` free. |
| Inspection | The ownership fix was correct and resident batching did not regress. The remaining failure was a deterministic Python call-signature mismatch; no distributed batch executed and there was no CUDA degradation. |
| Bottleneck | **Worker launcher signature, not Kimi arithmetic, memory or GPU stability.** |
| Decision | **MODIFY; retain both the failed receipt and its valid resident baseline.** |
| Redesign | H014-SUB-006c passes an explicit spawn context and keyword arguments, then reruns the unchanged incremental stop-on-failure protocol. |

| Field | H014-SUB-006c |
|---|---|
| Hypothesis | Supplying the existing worker launcher with its required spawn context and keyword-only inputs is sufficient to reach and safely execute the unchanged four-worker batch `1 -> 2 -> 4 -> 8` protocol. |
| Implementation | Added `multiprocessing.get_context("spawn")` and corrected only the `_start_worker` invocation syntax. No execution semantics or acceptance thresholds changed. |
| Benchmark | `sub-layer/h014-sub-006c-four-worker-batch.json` SHA `b57f205b...b863`; real layer 89, 5 warmup plus 20 retained calls/size, three-position exact/state comparison, per-size synchronization, safe batch-1 fixture, worker sticky-error checks, VRAM and `nvidia-smi`. |
| Result | **EXECUTION PASS; HYPOTHESIS SUPPORTED.** Batches `1/2/4/8` were bit exact with identical state fingerprints, all 16 experts/row exactly once, zero lifecycle deltas and four healthy post-size fixtures. Batch-8 wall p50/p95/p99 was `21.0363/21.8450/22.2909 ms` (`380.30 rows/s`), versus resident `14.8471 ms` (`538.83 rows/s`): `70.579%` throughput retention. Reuse was `2.09150` rows/native call, messages/row fell `87.5%`, and capacity was `2.01717x` eight serial distributed batch-1 rows; all four preregistered gates passed. |
| Inspection | At `B=1/2/4/8`, complete-layer relative throughput was `66.315/64.127/70.562/70.579%`; messages/row were `8/4/2/1`; mean unique experts were `16/27.45/34.75/34.75`; and per-row transport remained about `290.3/288.7/287.9/287.4 KB`. At batch 8 the profiled external phase was `12.5445 ms`, collective p50 `10.6804 ms`, critical worker `8.6441 ms`, exposed coordination `1.7118 ms`, and returned expert rows `1,835,008` bytes. Four workers each held `3,932,536,832` tracked bytes (`21.514%` of the complete-layer resident bytes). |
| Bottleneck | **Worker-local per-expert host transfer sequencing.** The batch worker uploads and downloads around every expert group; its critical path consumes `80.9%` of collective p50. Message count is no longer the main local limit, and full expert-output payload remains irreducible without moving reduction. |
| Decision | **RETAIN safe four-worker batching and its measured 70.6% capacity retention, but do not freeze this worker execution layout.** The trace proves overlap/reuse for these real routes, not a production prompt distribution. |
| Redesign | H014-SUB-006d tests a persistent worker task buffer: upload sparse activation rows once, device-gather all task inputs, execute grouped experts at stable offsets, and download one contiguous output buffer. |

| Field | H014-SUB-006d |
|---|---|
| Hypothesis | Replacing per-expert host upload/download loops with one worker-level H2D upload, device gather, grouped native execution and one D2H collection preserves bit-exact batch `1/2/4/8` semantics and improves batch-8 complete-layer capacity by at least `1.20x` versus H014-SUB-006c, reaches at least `80%` resident batch-8 throughput, and does not increase messages or payload bytes. |
| Implementation | Added a persistent `128 x 3584` task-input buffer per worker, one activation upload, device copies into expert-group order, the same exact `8/4/2/1` native chunks, and one contiguous output download. Parent routing/reduction/shared expert/state and protocol framing were unchanged. |
| Benchmark | `sub-layer/h014-sub-006d-buffered-four-worker-batch.json` SHA `75f78ba4...c44c`; same incremental fixture and fixed H014-SUB-006c comparator. |
| Result | **HARNESS FAIL after real batch-1 execution; HYPOTHESIS NOT EVALUATED.** Four buffered workers loaded at `3,934,371,840` tracked bytes each (`21.524%` of complete-layer residency). Control flow crossed batch-1 correctness and completed its retained performance calls, then aggregation raised `KeyError: activation_h2d_device_ms`. The receipt did not atomically retain the local correctness object before aggregation, so those numerical calls are not accepted as retained certification. All four workers reported clean CUDA state and exit code 0; `nvidia-smi` remained measured. |
| Inspection | The new phase fields were mistakenly added to the legacy single-row `dispatch()` record rather than `dispatch_batch()`. The worker responses contained them, but the batched collective discarded them. This is record plumbing, not a CUDA or arithmetic failure. It also exposed an evidence-retention flaw: a passed size must be written immediately after correctness, before optional performance aggregation. |
| Bottleneck | **Telemetry schema wiring and insufficient per-phase atomic persistence.** |
| Decision | **MODIFY; retain the failed receipt; do not use its unpersisted batch-1 numbers.** Move phase fields to the batched collective record and persist each correctness result before performance. |
| Redesign | H014-SUB-006e predicts those recorder-only fixes are sufficient to evaluate the unchanged buffered execution path under the original d thresholds. |

| Field | H014-SUB-006e |
|---|---|
| Hypothesis | Correctly forwarding buffered worker phase metrics through `dispatch_batch()` and atomically persisting each passed correctness phase will allow the unchanged H014-SUB-006d execution design to complete incremental certification; batch 8 must still be bit exact, `<=17.53021 ms`, at least `80%` of resident throughput, and no larger in tensor payload or messages than H014-SUB-006c. |
| Implementation | Moved four worker phase fields from the legacy one-row record to the batched record and added an atomic `CORRECTNESS_PASS_PERFORMANCE_PENDING` receipt after every size. No CUDA operation, tensor, ownership, reduction, state or acceptance threshold changed. |
| Benchmark | `sub-layer/h014-sub-006e-buffered-four-worker-batch.json` SHA `d8aa7821...f285`; same real layer 89, exact trace, four 224-expert workers, `1 -> 2 -> 4 -> 8`, 5 warmup plus 20 retained calls, fixed H014-SUB-006c comparator and per-size health gates. |
| Result | **EXECUTION PASS; PERFORMANCE HYPOTHESIS FALSIFIED.** All sizes were bit exact with identical states, all selected experts exactly once, zero lifecycle deltas and healthy post-size fixtures. Batch-8 wall p50/p95/p99 was `17.7114/19.2441/22.6355 ms` (`451.69 rows/s`) versus resident `14.6208 ms` (`547.22 rows/s`), retaining `82.542%`. Tensor payload was unchanged at `286,720 bytes/row` and messages stayed `1/row`, but capacity gain over H014-SUB-006c was `1.18772x`, below the preregistered `1.20x`. |
| Inspection | The buffered design materially improved 006c: complete batch-8 p50 fell `15.81%`, external-phase p50 fell `12.5445 -> 8.8357 ms`, collective fell `10.6804 -> 7.4277 ms`, and critical worker fell `8.6441 -> 5.4305 ms`. At batch 8, critical gather+expert device time was `5.3995 ms`, output D2H wall `0.1158 ms`, and exposed coordination `1.4965 ms`. Worker footprint increased only `1,835,008 bytes` to `3,934,371,840` (`21.524%` of a complete layer). Tails remain noisy: p99 is `22.6355 ms`. |
| Bottleneck | **Grouped expert compute/device gather is now dominant, with an avoidable extra profiled H2D synchronization and cross-context tail jitter.** |
| Decision | **MODIFY AND RETAIN THE BUFFERED LAYOUT.** Its 82.5% capacity retention is better than 006c, but the stated 1.20x claim remains false and may not be rounded up. |
| Redesign | H014-SUB-006f removes one worker synchronization by profiling H2D+gather+expert as a single interval. This is the last test of this local synchronization mechanism; if it misses, accept 006e. |

| Field | H014-SUB-006f |
|---|---|
| Hypothesis | Removing the separate per-worker H2D profiling `profile_end()` and measuring input transfer, device gather and grouped expert compute in one CUDA interval preserves bit-exact `1/2/4/8` execution and is sufficient to meet the unchanged batch-8 `<=17.53021 ms` (`>=1.20x` versus H014-SUB-006c) gate while retaining at least `80%` resident throughput; batch-8 p99 must not exceed H014-SUB-006c's `22.29093 ms`. |
| Implementation | Removed one CUDA event synchronization per worker request and reported a combined input+gather+expert interval plus H2D enqueue wall and output D2H wall. Buffering, kernels, messages, tensors and semantics remained unchanged. |
| Benchmark | `sub-layer/h014-sub-006f-single-interval-four-worker-batch.json` SHA `6119f19e...a108`; same incremental real layer-89 fixture, fixed H014-SUB-006c p50/p99 comparators, exact/state/lifecycle/health gates and 5/20 timing protocol. |
| Result | **EXECUTION PASS; HYPOTHESIS SUPPORTED.** Batches `1/2/4/8` were bit exact with identical state fingerprints, every selected expert exactly once, zero warm lifecycle deltas and clean per-size CUDA/`nvidia-smi` checks. Batch-8 wall p50/p95/p99 was `17.4095/18.5731/19.0253 ms` (`459.52 rows/s`) versus resident `14.7880 ms` (`540.98 rows/s`): `84.942%` whole-layer throughput retention. Capacity gain over H014-SUB-006c was `1.20832x`; p99 beat the fixed `22.29093 ms` comparator. Messages and tensor payload remained `1/row` and `286,720 bytes/row`. |
| Inspection | Relative whole-layer throughput for `B=1/2/4/8` was `90.790/86.228/86.214/84.942%`; reuse was `1.000/1.1658/1.5184/2.0915` rows/native call. At batch 8, external phase p50 was `9.0802 ms`, collective `7.1793 ms`, critical input+gather+expert `5.6303 ms`, exposed coordination `1.1779 ms`, H2D enqueue `0.1305 ms`, and output D2H `0.1115 ms`. Ideal-four-way expert efficiency was `21.076%`; the complete-layer ratio is much stronger because non-expert parent work is unchanged. The three retained route positions assigned `105/107/78/94` selections to workers 0-3: hottest/coldest `1.3718x`. |
| Bottleneck | **Grouped expert compute plus route-dependent critical-worker load.** Synchronization removal is no longer the leading local mechanism; the fine edge still returns `1,835,008` expert-output bytes per batch-8 layer. |
| Decision | **RETAIN H014-SUB-006f as the best safe logical four-worker batch design.** It is a one-GPU multi-context proof, not physical four-GPU efficiency evidence. Production use remains restricted to the measured fast-domain network envelope and requires the Experiment 015 multi-GPU canary. |
| Redesign | H014-SUB-007 measures shared-expert placement. Then characterize recovery and use broader real route traces before considering ownership replication or reassignment. |

| Field | H014-SUB-007 |
|---|---|
| Hypothesis | Keeping the two fused shared experts resident on the parent stage is lower-latency and lower-network-cost than moving them to, or replicating them across, microworkers; their real compute can later overlap routed-worker service without adding another fine-edge payload. |
| Implementation | Added a fail-closed evidence join that reads the exact layer-89 safetensors header, reproduces production grouped-int4 resident bytes, consumes real H014-030e shared phase timings, H014-SUB-006f complete/collective service and H014-SUB-005 fine-edge limits, then evaluates parent, one routed worker, separate shared worker and four-way replication. No model execution changed. |
| Benchmark | `sub-layer/h014-sub-007-shared-expert-placement.json` SHA `7aeaca0f...3bd`; exact tensor metadata digest `8692c313...f242`, real B1/B2/B4/B8 shared phase, batch-8 transport, RTT `0/0.1/0.25/0.5/1/2/5/10/20 ms` at 100 Gbps and bandwidth `1/2.5/5/10/25/100 Gbps` at 0.25 ms. Remote projections intentionally assume zero software/framing overhead, so they are optimistic lower bounds. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** The three shared matrices total `264,241,152` BF16 source bytes and `74,317,824` runtime grouped-int4 bytes (`0.06921 GiB`). Real shared device p50 for `B=1/2/4/8` is `0.2421/0.3692/0.6748/1.3167 ms`. Remote placement adds `28,672` input plus `28,672` output bytes/row (`458,752` tensor bytes at batch 8) and two messages. A selected routed worker rises to at least `4,009,148,416` bytes (`21.933%` of complete-layer residency); four-way replication adds `222,953,472` duplicate weight bytes. |
| Inspection | Current parent-serial batch-8 p50 is `17.4095 ms`; perfect parent overlap projects `16.0854 ms` (`1.0823x` upper-bound gain). At 100 Gbps a remote shared service remains hidden behind the measured routed collective only through an optimistic `5.8185 ms` RTT, but it can at best tie the parent-overlap projection and always adds network. At 10/20 ms its optimistic complete projection is `20.2669/30.2669 ms`. A separate shared worker's `0.06964 GiB` is a tensor+buffer lower bound that excludes CUDA context/runtime memory. |
| Bottleneck | **Remote shared placement is dominated by the zero-network parent overlap: identical compute plus unavoidable activation/output transport.** |
| Decision | **RETAIN BOTH SHARED EXPERTS ON THE PARENT. REJECT remote placement and replication for the measured topology.** This is an evidence-backed placement decision, not a claim that overlap is already implemented. |
| Redesign | H014-SUB-008 tests the remaining opportunity directly: split persistent routed dispatch into start/collect and execute the unchanged parent-resident shared expert during worker service. |

| Field | H014-SUB-008 |
|---|---|
| Hypothesis | Starting the persistent routed collective, executing the parent-resident shared expert while workers run, then collecting routed outputs preserves bit-exact `1/2/4/8` semantics and hides at least 50% of the batch-8 shared-expert wall time: complete p50 must be `<=16.74745 ms` and p99 must not exceed H014-SUB-006f's `19.02525 ms`. |
| Implementation | Split the persistent collective into `start_batch`/`collect_batch`; after fanout the parent enqueues the unchanged shared kernel into existing session-owned buffers, then collects routed outputs. No per-call process/thread/connection/task infrastructure was added; routing, experts, reduction, residual, messages, payloads, weights and states remained unchanged. |
| Benchmark | `sub-layer/h014-sub-008-parent-shared-overlap.json` SHA `cbd919ad...6299`; incremental real layer-89 `1/2/4/8`, exact/state/lifecycle/health gates, 5 warmup plus 20 retained calls, fixed H014-SUB-006f p50/p99 comparators. |
| Result | **EXECUTION PASS; OVERLAP HYPOTHESIS FALSIFIED.** All sizes were bit exact with identical states, every expert exactly once, zero lifecycle deltas and healthy post-size checks. Batch-8 p50/p95/p99 was `17.4499/18.9221/19.1202 ms`, versus H014-SUB-006f `17.4095/.../19.0253 ms`: the run saved `-0.0404 ms` (`-3.051%` of shared wall). Both fixed performance gates failed; messages remained unchanged. |
| Inspection | The parent submitted shared work in a `0.1704 ms` CPU overlap window, but same-device contexts contended. Batch-8 collection wait rose to `7.9213 ms`; routed critical-worker device time rose `5.6303 -> 7.0936 ms`. Under phase synchronization, shared wall rose from isolated `1.3241` to `4.7080 ms`. Thus the coordination semantics work, but one physical GPU cannot demonstrate the projected separate-GPU overlap. |
| Bottleneck | **Same-GPU CUDA-context contention fully consumes the theoretical shared/routed overlap.** |
| Decision | **REJECT overlap for the certified logical one-GPU layout; retain serial parent shared execution from H014-SUB-006f.** Keep the dormant start/collect seam for recovery testing and the physical multi-GPU canary, where parent and workers occupy distinct GPUs. |
| Redesign | H014-SUB-009 tests fail-closed recovery of that persistent start/collect protocol before any topology is approved. |

| Field | H014-SUB-009 |
|---|---|
| Hypothesis | The persistent real expert collective rejects timeout, worker loss, partial response, duplicate response and stale generation before reduction; explicit cancellation drains/discards the generation, retry on a restored ownership set is exact, and a fresh slot remains reusable with no stale frame or state leakage. |
| Implementation | Added bounded response polling, immutable handle state, explicit cancel/drain, pre-dispatch stale-frame rejection and test-only response faults. Loss injection was constrained to exit only after the worker completed and synchronized real CUDA work. The production message payload is unchanged when no test fault is requested. |
| Benchmark | Real layer-89 batch-1 expert computation on four persistent 224-expert partitions using DLL SHA `5d33aded...f20120`. Executed an exact baseline, duplicate-frame injection plus a fresh next generation, then explicit cancellation; attempted the required safe fixture immediately after cancellation. Evidence was retained after every phase in `h014-sub-009a-cancel-profiler-failure.json` (SHA `e5c241e6...569a1`). |
| Result | **FAIL; RETAINED.** Baseline output/routes were exact. One duplicate frame was identified and discarded, and the next generation was exact. Cancellation drained all four workers, discarded all `16` tasks / `229,376` output bytes, exposed no reduction, left cache sequence length `0`, and set the handle to `CANCELLED`. The following safe request failed at `profile_begin` before useful execution with `CUDA event profiler rejected begin`. All four workers shut down with clear CUDA error state; after cleanup `nvidia-smi` showed `1,870 MiB` used / `30,318 MiB` free and the device remained available. Timeout/loss/partial/stale were therefore not run. |
| Inspection | Cancellation correctly cleaned transport frames and mathematical state, but its deliberate exception escaped while the coordinator's whole-stage CUDA event interval was active. The normal success path closes that interval at the end of the layer; the exception path did not. The failure was a persistent-runtime profiler lifecycle leak, not corrupt expert output, a worker CUDA error, or a GPU-loss recurrence. |
| Bottleneck | **Missing exception cleanup for an active coordinator CUDA profiling interval.** |
| Decision | **MODIFY.** Retain duplicate and cancellation protocol semantics; do not accept recovery overall. Preserve the failed receipt and close the event interval on any failed phase before propagating the original error. |
| Redesign | H014-SUB-009b predicts that minimum profiler cleanup will make the immediate safe fixture exact without rebuilding the coordinator or worker group, after which the still-unrun partial/stale/timeout/loss and restored-owner retries can proceed one at a time. |

| Field | H014-SUB-009b (preregistered) |
|---|---|
| Hypothesis | Closing an active CUDA profiling interval when a stage phase raises will preserve the already-correct cancel/drain behavior and allow the same persistent coordinator and four-worker collective to execute an exact fresh request; all incomplete, stale, timed-out or lost-worker generations will then fail before reduction and exact execution will resume only after restoring a complete ownership group. |
| Implementation | Minimum change: `run_phase` ends the active event interval in its exception path and re-raises the original fault. No arithmetic, routing, ownership, normal transport payload or successful execution path is changed. |
| Benchmark | Repeated the bounded real layer-89 batch-1 sequence on four persistent 224-expert partitions. The loss worker exited only after real expert execution, CUDA event completion and D2H synchronization. Each completed scenario was atomically retained; a complete ownership group was restored after each poison fault. Receipt: `h014-sub-009b-recovery.json`, final SHA `39705bb1...ef89f5`; exact DLL SHA `5d33aded...f20120`. |
| Result | **PASS.** All 11 acceptance gates passed. Baseline was exact. One duplicate frame was discarded and the following generation remained exact. Cancellation drained `4/4` workers, all `16` tasks and `229,376` expert-output bytes, performed no reduction, left cache sequence length `0`, and the immediate same-group safe fixture was exact. Partial, stale, `100 ms` timeout and post-CUDA worker loss each returned no stage output, left cache length `0`, marked the handle `FAILED`, and poisoned the group. After each complete-owner restoration the output fingerprint was exactly `sha256:f676fb0f...6828e5`, routes matched, and every selected expert executed once. Final `4/4` workers reported clear CUDA error state; post-process `nvidia-smi` returned to `1,870 MiB` used / `30,318 MiB` free. |
| Inspection | Mathematical fail-closed behavior is correct: the runtime never reduces an incomplete generation and never silently continues with a missing expert owner. Duplicate/cancel faults are recoverable without rebuilding the group. Poison faults require complete ownership restoration. Across five groups, worker load was `3.633-3.673 s`; because the current launcher starts four workers sequentially, restored-group load averaged `14.590 s`. Each worker held `3,934,371,840` tracked bytes; the four-worker domain held `15,737,487,360` bytes. The `360` coordinator buffer-allocation delta is session-state allocation across deliberately new/cancelled sessions, not warm-generation buffer recreation. This remains a logical multi-process proof on one GPU, not physical multi-GPU recovery timing. |
| Bottleneck | **Full expert-domain weight reload dominates service restoration after a poisoned generation (~14.59 s with the sequential launcher).** Detection itself is bounded (`100 ms` for timeout); duplicate and cancellation have no reload penalty. |
| Decision | **RETAIN the hardened persistent collective and exception-safe profiler lifecycle.** Recovery gate PASS for the logical real-CUDA topology. Require a complete ownership set before retry; never reduce a partial set. Parallel load/replacement belongs to deployment hardening, not the normal decode path. |
| Redesign | H014-SUB-010 measures actual retained Kimi routing imbalance before considering reassignment or hot-expert replication. No ownership strategy will be changed from the three-position layer-89 sample alone. |

| Field | H014-SUB-010 (preregistered) |
|---|---|
| Hypothesis | Across the canonical checkpoint-faithful `K3_IDOT=0` trace, four-way static modulo ownership is sufficiently balanced that aggregate worker selections differ by at most `5%`, and a capacity-preserving per-layer assignment trained on the first two token steps cannot improve the held-out third-step mean critical worker load by at least `10%`. If both predictions hold, three-token evidence does not justify reassignment or hot-expert replication. |
| Implementation | Added a source-backed, CUDA-free route analyzer. It verifies the passing serial receipt, explicit `K3_IDOT=0`, route path/SHA/call/selection counts and exact three-step ordering; reconstructs all per-call worker loads; and compares static modulo ownership with a per-layer greedy mapping trained only on positions 0/1. The candidate holds exactly `224` experts per worker and adds zero expert bytes. No production assignment or weight changed. |
| Benchmark | Canonical `oracle-full-93-idot0/routes.txt` SHA `c734d864...91288`: real prompt `Hi`, two prompt tokens plus one generated-token step, `92` MoE layers, `276` calls, `4,416` selections. Held-out position 2 was not used to build the candidate. Evidence: `sub-layer/h014-sub-010-routing-imbalance.json` SHA `b929b8e4...8982c3`; 552-row call CSV SHA `3221390f...d3a6`. |
| Result | **CHARACTERIZATION PASS; COMPOUND HYPOTHESIS FALSIFIED.** Static aggregate worker totals were `1,131 / 1,157 / 1,080 / 1,048`, a `9.873%` range of the mean (`1.104x` hottest/coldest), so the `<=5%` prediction failed. Per-call critical selected-expert count was mean/p50/p95/p99/max `6.203/6/8/9/9` versus ideal `4`; mean `ideal/observed` efficiency was `66.331%`. The equal-memory candidate improved training mean critical load `15.425%` but held-out mean only `0.896%`, with held-out p95 unchanged at `8`; it failed the `>=10%` usefulness threshold. |
| Inspection | Instantaneous imbalance is material: four calls placed `9/16` experts on one worker. It is not evidence of a persistent owner hotspot. Of `82,432` layer/expert pairs, only `3,889` appeared: `3,389` once, `473` twice and `27` in all three steps. Mean unique experts per layer were `42.272/48` (three-step reuse `1.143x`). The candidate changed `61,605/82,432` assignments (`74.734%`) to fit two steps and did not generalize. Layer 89 itself totaled `13/14/10/11` selections, `1.4x` hottest/coldest, with critical mean `5.667`. Because each expert is a distinct tensor per layer, global expert-ID frequency across layers is not a hot-weight signal. |
| Bottleneck | **Per-token stochastic fanout imbalance on the critical worker (mean 6.203 experts versus ideal 4), not a proven persistent hot expert.** |
| Decision | **RETAIN static disjoint `expert_id mod 4`; REJECT the learned reassignment and do not replicate experts from three tokens.** The routing-imbalance characterization gate passes, but broader real prompt traffic remains required before any hot-expert policy. |
| Redesign | Proceed with separate decode and long-context prefill/state-capacity measurement on the retained static topology. Treat broader prompt-route capture as a serving validation requirement; only reopen reassignment/replication if a larger trace shows held-out benefit. |

| Field | H014-033a (preregistered) |
|---|---|
| Hypothesis | Because layer 89 uses recurrent KDA rather than a growing attention cache, real complete-stage sequential prefill remains context invariant through 16K: last-256 device p50 at 16K is at most `1.10x` the last-256 p50 at 1K, cumulative 16K prompt throughput retains at least `90%` of 1K throughput, and session attention state remains exactly `6,881,280` bytes. |
| Implementation | Added an incremental long-context stage harness using the unchanged production executor, exact DLL and full real layer-89 weights. Each context used a new isolated session and cyclically replayed the three immutable real boundaries. Research records were popped after every call, one post-prefill decode was measured, and a separate exact three-position safe fixture plus CUDA/driver checks gated advancement. |
| Benchmark | Contexts `1,024 -> 4,096 -> 8,192 -> 16,384`, retained after each. Full stage load/READY `21,071.444 ms`; resident device bytes `18,278,776,832`; batch workspace `13,219,840`. Exact H014-030e lineage and DLL SHA `5d33aded...f20120`. Final receipt `performance/h014-033a-prefill-layer89-kda.json`, SHA `4f0c125d...322a7`. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** All 14 per-context gates passed at all four sizes. Device p50/p95/p99 was `3.316/3.443/3.747 ms` at 1K and `3.309/3.384/3.532 ms` at 16K. Last-256 p50 changed `3.338800 -> 3.314112 ms` (`0.992606x`, gate `<=1.10x`). Prompt service was `271.664 / 289.550 / 290.075 / 289.932 tok/s`; 16K/1K retention `106.724%` (gate `>=90%`). Stage-local TTFT was `3.773 / 14.150 / 28.245 / 56.513 s`. State remained exactly `6,881,280` bytes at every context and advanced to cache sequence `16,385` after the final decode. |
| Inspection | KDA is truly recurrent in the production runtime: neither logical state bytes, measured session allocation (`6,291,456` bytes), nor terminal service grew with context. Per-call weight load/materialization/buffer-allocation deltas were zero; session memory recovered exactly after every close. Every first-three output/route and post-context safe fixture was exact; final outputs/states were finite and every route executed 16 experts once. Live checks correctly showed the resident 18.279 GB stage; process exit restored `nvidia-smi` to `1,870 MiB` used / `30,318 MiB` free. H014-030e's phase profile remains the phase decomposition; this cycle tested context scaling rather than duplicating it. The repeated-boundary fixture is not a diverse natural-language 16K prompt. |
| Bottleneck | **Fixed complete-stage per-token CUDA compute (~3.31 ms), not context-state growth or cache traffic.** |
| Decision | **RETAIN the KDA recurrent state/capacity model.** KDA prefill characterization PASS. Do not average its constant service with Gated MLA. |
| Redesign | H014-033b measures late Gated MLA separately, where cache bytes and attention work grow with context. |

| Field | H014-033b |
|---|---|
| Hypothesis | Unlike recurrent KDA, layer-91 Gated MLA shows measurable context-scan growth: 16K last-256 device p50 is at least `1.10x` but no more than `2.00x` its 1K value, cumulative 16K prompt throughput retains at least `70%` of 1K, and prepared state grows exactly `2,304` bytes per token (`(context+1) x 2,304` including the post-prefill decode slot). |
| Implementation | Extended the H014-033a harness only with the preregistered terminal-ratio band and exact `2,304`-byte/token MLA state gate. The layer-91 production executor and certified H014-030d DLL remained unchanged. Each context was atomic and advanced only after correctness, state, lifecycle, memory, CUDA and driver checks. |
| Benchmark | Attempted `1,024 -> 4,096 -> 8,192 -> 16,384`, batch 1, exact H014-030f reference and real layer-91 weights/activations. Raw receipt `performance/h014-033b-prefill-layer91-mla.json` SHA `904719c7...842`; inspection receipt `h014-033b-inspection.json`. |
| Result | **FAIL; HYPOTHESIS FALSIFIED BEFORE 16K.** 1K and 4K passed every gate. Device p50/p95/p99 was `3.082/3.360/6.772 ms` at 1K and `3.530/4.125/4.202 ms` at 4K; prompt service was `289.158` and `270.360 tok/s`; stage-local TTFT was `3.545` and `15.155 s`. The 8K prompt completed, but its mandatory next decode requested `T=8,193` and the native MLA absorb call rejected it. 16K was not attempted. |
| Inspection | Source inspection found the exact fail-closed condition `T > 8192`. Batch-1 attention stores `qa[K] + cl[K] + scores[T]` in dynamic shared memory and performs max/exp/sum/normalization on CUDA thread 0. At `K=512`, shared demand is `36,868` bytes at `T=8,193` and `69,636` bytes at `T=16,385`. The failure occurred before an unsafe launch; after stage close the RTX 5090 was healthy at `2,367 MiB` used, and this was not OOM, numerical divergence or a recurrence of H014-027v. |
| Bottleneck | **A hard 8,192-context native launch contract backed by a T-sized shared-memory score vector; the wrapper neither queries nor opts into larger per-block shared memory.** |
| Decision | **RETAIN the failed receipt and retain the 8,192 guard in certified H014-030d.** Do not claim 8K-plus-decode or 16K support from that binary. |
| Redesign | H014-033c tests a guarded opt-in shared-memory launch. It must query the real device limit, reject allocations beyond it before CUDA work, preserve exact results below the old ceiling and advance incrementally through the same contexts. |

| Field | H014-033c |
|---|---|
| Hypothesis | The current GPU's reported opt-in dynamic shared-memory limit is at least `69,636` bytes, so a capacity-checked batch-1 MLA launch can safely support `T=16,385`; all `1K -> 4K -> 8K -> 16K` contexts plus one decode pass exact correctness/health gates, while 16K terminal device p50 remains no more than `2.00x` the 1K value and cumulative throughput retains at least `70%`. |
| Implementation | Added retained default/opt-in shared-memory capabilities to each CUDA device context and a query ABI. Only batch-1 MLA may now reach `T=16,385`; requests above the reported opt-in limit reject before launch, and above-default launches require successful `cudaFuncSetAttribute`. Batch/ragged ceilings and arithmetic stayed unchanged. A candidate-regression harness switch explicitly records the new SHA while retaining exact reference gates. |
| Benchmark | Candidate DLL SHA `c3bdb40d...ae326`, `sm_86` minimum plus forward PTX. RTX 5090 reported `49,152` default and `101,376` opt-in bytes; `T=16,385, K=512` required `69,636`, leaving `31,740`. Real layer 91 then advanced atomically through `1K -> 4K -> 8K -> 16K`. Raw receipt `h014-033c-prefill-layer91-mla16k.json` SHA `8ff01603...2b48`; capability SHA `6dfa436f...31b9`. |
| Result | **CHARACTERIZATION PASS; PERFORMANCE HYPOTHESIS FALSIFIED.** Every context and post-prefill decode passed all exact correctness, state, lifecycle, memory, CUDA, safe-fixture and driver gates. Last-256 device p50 was `3.257 / 4.128 / 5.298 / 7.636 ms`; 16K/1K was `2.344476x` (gate `<=2.00x`). Prompt service was `285.292 / 268.305 / 231.829 / 182.567 tok/s`; retention was `63.993%` (gate `>=70%`). Stage-local TTFT was `3.593 / 15.271 / 35.342 / 89.751 s`; state was exactly `(context+1) x 2,304` bytes. |
| Inspection | The old failure is contained: 8K plus `T=8,193` and 16K plus `T=16,385` both completed, and process exit restored `nvidia-smi` to `1,870 MiB` used / `30,318 MiB` free. Only MLA attention receives `T`; downstream dimensions remain fixed. The last-window stage delta from 1K to 16K is therefore `4.379120 ms` inside the context-dependent attention path. Source has two parallel `O(T*K)` cache scans around a thread-0 exp/sum/normalize loop; this cycle does not yet identify which dominates. |
| Bottleneck | **Safe launch capacity is solved; context-dependent MLA attention/cache work now dominates, but serial softmax versus parallel score/value scans is unresolved.** |
| Decision | **RETAIN the guarded capacity mechanism and 16K correctness evidence; do not yet promote this candidate as final.** Its latency and throughput-retention claims failed. |
| Redesign | H014-033d isolates serial elementwise softmax cost without changing ordered summation or the two cache scans. |

| Field | H014-033d |
|---|---|
| Hypothesis | Parallelizing only the per-score `expf` and normalization work across 256 threads, while preserving the serial max/sum order and both `O(T*K)` scans, keeps all reference fingerprints bit-identical and lowers 16K last-256 complete-stage device p50 by at least `10%` versus H014-033c (`<=6.872616 ms`). |
| Implementation | Reused `cl[0]` as temporary scalar storage, preserved thread-0 max and sum in their historical order, and parallelized only independent `expf` and division loops with barriers. Score scan, value scan, attention arithmetic, guard and shared-memory size were unchanged. The change was subsequently reverted from source after inspection. |
| Benchmark | Candidate SHA `efba8f32...f1f5e`; same `sm_86`+PTX/capability contract and exact real layer-91 `1K -> 4K -> 8K -> 16K` sequence. Raw SHA `7d865ebe...0020`; direct comparison against H014-033c. |
| Result | **CHARACTERIZATION PASS; PERFORMANCE HYPOTHESIS FALSIFIED.** Every gate passed and final output, decode output and state fingerprints matched H014-033c at every context. Last-256 p50 improved by `0.877% / 3.321% / 5.092% / 7.171%`; 16K was `7.636240 -> 7.088624 ms`, below the required 10%. Terminal ratio remained `2.195610x` and throughput retention `65.944%`; 16K prompt service was `192.411 tok/s`, TTFT `85.159 s`. |
| Inspection | The monotonic benefit proves serialized exp/normalization is a real context term, but it is not the dominant one. Even after removing most of that work from thread 0, both H014-033c performance gates still fail. CUDA and `nvidia-smi` remained healthy and process exit again restored `1,870 MiB` used / `30,318 MiB` free. |
| Bottleneck | **The two `O(T*K)` score and weighted-value cache scans dominate the remaining Gated-MLA context cost; the easy elementwise softmax change accounts for only 7.171% of complete-stage 16K service.** |
| Decision | **REVERT H014-033d under its preregistered 10% threshold. RETAIN H014-033c's capacity-checked 16K mechanism.** Do not optimize another easy primitive without complete-stage evidence. |
| Redesign | Run required P0/P1 regression certification on H014-033c, then use the measured context curve in P4. Reopen a cache-scan kernel redesign only if the validated fleet model shows material impact. |

| Field | H014-034a (preregistered) |
|---|---|
| Hypothesis | The H014-033c capacity-checked batch-1 MLA absorb is also sufficient for the production complete-stage row-cooperative path because that path preserves independent caches and invokes the guarded absorb once per row. At a real cache length of `8,192`, batches `1 -> 2 -> 4 -> 8` will therefore pass exact output/route/state and post-size CUDA/driver health gates without any CUDA change, and batch 8 will retain at least `1.25x` aggregate device capacity versus batch 1 measured at the same context. |
| Implementation | Added only the preregistered incremental long-context batch harness. No CUDA kernel, shared-memory limit, routing, state, or batch guard changed. |
| Benchmark | Raw receipt `performance/h014-034a-contextual-batch-layer91-8k.json` SHA `f9a013e5...b20da`; inspection receipt `h014-034a-inspection.json`. The RTX 5090 and exact H014-033c candidate were healthy before the attempted run. |
| Result | **HARNESS FAIL BEFORE BATCH 1; HYPOTHESIS NOT EVALUATED.** The layer loaded, then load-evidence construction raised `AttributeError` because the recorder called `device_shared_memory_limits()` while `_CudaRuntime` exposes the retained query as the `shared_memory_limits` property. No contextual batch size was armed or executed. |
| Inspection | The failure is isolated to evidence plumbing. Source inspection still shows `_execute_attention_batch` loops over session-owned MLA cache appends/absorbs, so the guarded H014-033c ABI remains the relevant native call. The runtime closed, and post-process `nvidia-smi` returned to `1,870 MiB` used / `30,318 MiB` free. |
| Bottleneck | **Incorrect recorder attribute name before contextual CUDA execution.** |
| Decision | **MODIFY; retain the failed receipt.** Read the existing property; change no scientific threshold or execution path. |
| Redesign | H014-034b reruns the unchanged `1 -> 2 -> 4 -> 8` protocol and `>=1.25x` batch-8 capacity hypothesis after the recorder-only fix. |

| Field | H014-034b (preregistered) |
|---|---|
| Hypothesis | With the capability recorder reading the existing `_CudaRuntime.shared_memory_limits` property, the unchanged H014-033c production path passes exact output/route/state and post-size health gates at 8K for batches `1 -> 2 -> 4 -> 8`; batch 8 retains at least `1.25x` aggregate device capacity versus batch 1 at the same context. |
| Implementation | Replaced only the erroneous method call with the existing `shared_memory_limits` property. The execution path and gates were otherwise unchanged. |
| Benchmark | Raw receipt `performance/h014-034b-contextual-batch-layer91-8k.json` SHA `31e7c980...79934`; inspection receipt `h014-034b-inspection.json`. Batch 1 was armed; later sizes remained disarmed after its gate failed. |
| Result | **REAL BATCH-1 EXECUTION PASS; EVIDENCE GATE FAIL.** The 8K+decode output was bit-identical to H014-033c, all 16 experts executed once, lifecycle/state-length/state-bytes/finite/zero-suffix/safe-fixture/CUDA/VRAM/driver gates passed. Last-256 prefill p50 was `5.22053 ms`; retained device p50/p99 was `5.30424/5.35687 ms` (`188.528 rows/s`). Only the cross-receipt state fingerprint failed. |
| Inspection | H014-033c prepared exactly `8,193` slots, equal to its active length. H014-034b prepared `8,218` so it could continue into warm timing. `session_state_evidence` hashes the complete allocation, including the zero suffix; unequal shapes therefore hash differently even though the active length was `8,193`, output was exact, and the suffix was verified zero. This is not mathematical divergence. The process closed to `1,870 MiB` used / `30,318 MiB` free. |
| Bottleneck | **State evidence conflates active mathematical state with prepared zero-capacity suffix.** |
| Decision | **MODIFY; retain the real batch-1 evidence but do not arm batch 2.** Add a separately labeled active-prefix fingerprint; keep the full-allocation fingerprint and all state gates. |
| Redesign | H014-034c compares equal-shaped active-prefix state while retaining full-allocation integrity evidence, then reruns the unchanged incremental protocol. |

| Field | H014-034c (preregistered) |
|---|---|
| Hypothesis | A separately labeled active-prefix state fingerprint will match H014-033c at 8K for the phase-zero row while the full-allocation fingerprint continues to cover the prepared zero suffix; with that evidence correction, real batches `1 -> 2 -> 4 -> 8` pass all unchanged execution/health gates and batch 8 retains at least `1.25x` same-context batch-1 device capacity. |
| Implementation | Added `active_prefix_fingerprint` while retaining the full-allocation hash, bytes, finite and zero-suffix evidence; changed only the cross-receipt comparison to the active prefix. CUDA and workload were unchanged. |
| Benchmark | Raw receipt `performance/h014-034c-contextual-batch-layer91-8k.json` SHA `ffcef0de...adc02`; inspection receipt `h014-034c-inspection.json`. Batch 1 passed and atomically armed batch 2; batches 4/8 remained disarmed after batch 2 stopped. |
| Result | **BATCH 1 PASS; BATCH-2 MATHEMATICS/HEALTH PASS; MEMORY HIGH-WATER GATE FAIL.** Batch 1 retained device p50 `5.26800 ms` (`189.825 rows/s`). At batch 2, both outputs/routes were bit exact, active-prefix hashes matched H014-033c, paired full-state hashes matched, all state/lifecycle/CUDA/safe-fixture/driver gates passed, and validation device service was `8.54858 ms`. The only failed gate was session memory recovery. |
| Inspection | Batch 2 left exactly `2,097,152` bytes resident after its sessions closed (`14,186,184,704 -> 14,184,087,552` free). This is the native runtime's first-use batch scratch high-water allocation: session states freed, their full hashes matched, and process close returned to `1,870 MiB` used / `30,318 MiB` free. Measuring recovery against a pre-first-use baseline incorrectly labels persistent scratch as leaked session memory. |
| Bottleneck | **Unprimed native batch-size scratch residency in the memory baseline.** |
| Decision | **RETAIN batch-1 pass; MODIFY warm-baseline protocol; do not arm batch 4.** Persistent scratch must be measured and stable, not hidden or counted as session state. |
| Redesign | H014-034d primes each size twice with the exact three-position fixture, records the first-use high-water bytes, requires no further growth on the repeat, then measures 8K session recovery against that stable baseline. |

| Field | H014-034d (preregistered) |
|---|---|
| Hypothesis | The batch-2 residual `2,097,152` bytes are a bounded first-use native scratch reserve rather than a session-state leak. For each incrementally armed size, two exact three-position primes will reach a stable high-water mark (second-prime growth `0` bytes); measured against that post-prime baseline, real 8K sessions at batches `1/2/4/8` recover their state memory and pass all exact/health gates, while batch 8 retains at least `1.25x` same-context batch-1 capacity. |
| Implementation | Before each 8K size, opened short-lived sessions twice, executed the canonical three exact boundaries, closed/synchronized, recorded free VRAM and required the second pass not to lower it. The long-session baseline was then taken at the stable high-water mark. Full and active-prefix state hashes were both retained; CUDA/model math was unchanged. |
| Benchmark | `performance/h014-034d-contextual-batch-layer91-8k.json` SHA `bd0dc855...0ebc3`; exact H014-033c DLL; real layer 91; strict `1 -> 2 -> 4 -> 8`; 8,192 preload rounds plus one reference decode, five warmup and 20 retained calls/size. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** Every size passed every output/route/state/lifecycle/memory/CUDA/safe-fixture/VRAM/driver gate. Device p50/p95/p99 and aggregate rows/s were: B1 `5.2890/5.6346/5.6382 ms`, `189.071`; B2 `8.6651/8.8201/8.9556`, `230.810`; B4 `15.8664/15.9782/15.9803`, `252.106`; B8 `29.7613/30.0107/30.0375`, `268.805`. Batch-8 capacity was `1.421714x` batch 1, above the `1.25x` gate. |
| Inspection | B2 and B4 each established a bounded `2,097,152`-byte first-use scratch increment; every second prime grew `0` bytes and all long-session memory recovered to its post-prime baseline. Batch-8 wall p50/p95/p99 was `30.9519/31.3308/31.3460 ms`; state was `18,934,272` bytes/stream (`151,474,176` total). Reuse rose to `3.10303` rows/native call with `36` mean unique experts. Compared with H014-030f's short-cache B8, device service grew `2.25993x` and capacity retained only `44.249%` (`607.481 -> 268.805 rows/s`). The three-boundary paired fixture is exact but not a natural-prompt route distribution. Process exit was healthy at `1,864 MiB` used / `30,324 MiB` free. |
| Bottleneck | **Eight row-serial `O(T*K)` MLA cache scans; weight reuse still helps MoE/dense work, but attention growth consumes most of the short-cache gain.** |
| Decision | **RETAIN batch 8 as the best measured 8K capacity point and retain explicit size priming for READY.** Do not use the short-context `607.5 rows/s` as long-context capacity. Candidate remains quarantined pending final P0/P1 regression. |
| Redesign | H014-034e measures the persistent continuous FIFO with eight independently owned 8K streams, including queue, response, cadence, fairness, cancellation and slot reuse. |

| Field | H014-034e (preregistered) |
|---|---|
| Hypothesis | After exact batch-8 workspace priming and an 8K real-state preload, the existing persistent FIFO can continuously serve eight independent layer-91 streams with exact paired state/output, equal completion counts, zero warm lifecycle deltas, device capacity at least `95%` of H014-034d static batch 8 (`>=255.365 rows/s`), response p99 below `35 ms`, batch-formation p99 below `0.5 ms`, and correct cancellation/slot reuse. |
| Implementation | Reused the production scheduler unchanged. Added a contextual harness that preloaded eight persistent sessions through scheduler batch-8 dispatch, removed retained executor records during preload, then measured immediate FIFO batches. CUDA/scheduler policy/arithmetic were unchanged. |
| Benchmark | `performance/h014-034e-contextual-continuous-layer91-8k.json` SHA `ae3b95fb...f6d3a`; real layer 91, exact H014-033c candidate, 8,192 preload rounds, five warmup and 50 retained rounds, cancellation/replacement and full post-run health. |
| Result | **EXECUTION PASS; COMPOUND PERFORMANCE HYPOTHESIS FALSIFIED.** Device p50/p95/p99 was `31.4985/31.7526/32.2427 ms`; device/wall capacity `253.980/246.363 rows/s`. Static retention was `94.4848%`, narrowly below the fixed `95%` gate. Response p50/p95/p99 was `32.5478/33.1607/33.8318 ms` (pass); formation p99 `0.2215 ms` (pass); queue p99 `0.1820 ms`; per-stream cadence p50/p99 `32.9126/34.4097 ms`; completion counts were exactly equal. |
| Inspection | Output/route pairs, active/full state pairs, the H014-033c 8K reference, all 16 experts, lifecycle, cancellation, stale rejection, slot reuse, session-memory recovery, safe fixture, CUDA and driver health passed. Eight streams held `19,001,088` bytes each (`152,008,704` total). Batch-8 workspace first-use residency was `4,194,304` bytes and repeat growth `0`. The decisive comparator mismatch is routing overlap: H014-034d static used phase map `[0,0,1,1,2,2,0,0]` and reuse `3.10303`; this FIFO used `[0,1,2,0,1,2,0,1]` and reuse `2.08809`. Thus the `5.515%` capacity gap is not identified as FIFO overhead. Process exit returned to `1,864 MiB` used / `30,324 MiB` free. |
| Bottleneck | **Route-overlap-dependent expert weight reuse plus 8K MLA CUDA service; exposed FIFO formation/queue time is sub-millisecond.** |
| Decision | **RETAIN execution, cancellation and conservative `253.980 rows/s` capacity; do not round `94.4848%` to the passing threshold.** The scheduler-overhead question remains confounded. |
| Redesign | H014-034f changes only the input phase map to exactly match H014-034d, then reruns the same FIFO protocol to isolate scheduler overhead from route reuse. |

| Field | H014-034f (preregistered) |
|---|---|
| Hypothesis | With the continuous streams assigned H014-034d's exact phase map `[0,0,1,1,2,2,0,0]`, routing reuse returns to `3.10303` rows/native call, device capacity improves by at least `3%` over H014-034e and reaches at least `95%` of the matched static batch-8 capacity, while response p99 remains below `35 ms`, formation p99 below `0.5 ms`, and all exact/state/fairness/lifecycle/cancellation/health gates pass. |
| Implementation | Parameterized only harness fixture selection so each stream used H014-034d's exact phase. Production FIFO, ownership, CUDA, batch path, context, warmup, retained count and gates were unchanged. |
| Benchmark | `performance/h014-034f-contextual-continuous-matched-layer91-8k.json` SHA `8754fb1d...6aa6f`; same 8K preload, five warmup, 50 retained rounds, cancellation/replacement and post-run health. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** Device p50/p95/p99 was `29.7510/30.4581/31.1630 ms`; device/wall capacity `268.899/259.483 rows/s`. This is `100.0348%` of matched static capacity and `1.058740x` H014-034e, passing both fixed floors. Response p99 was `32.8758 ms`, formation p99 `0.1713 ms`, queue p99 `0.1312 ms`, and per-stream cadence p99 `32.8983 ms`; all passed. |
| Inspection | Reuse returned to `3.09179` rows/native call (near H014-034d's `3.10303`), explaining the controlled capacity recovery. Exact output/route/state pairs, 8K reference, fairness, zero lifecycle deltas, cancellation, stale rejection, slot reuse, safe fixture, session-memory recovery, CUDA and driver health passed. Eight streams held `152,008,704` state bytes total. Batch-8 workspace retained `4,194,304` first-use bytes then grew `0`. Final process health was `1,864 MiB` used / `30,324 MiB` free. |
| Bottleneck | **Complete-stage CUDA service: row-serial 8K MLA cache scans plus route-dependent expert reuse. FIFO formation and queue exposure are negligible.** |
| Decision | **RETAIN the existing scheduler and production batch 8. STOP scheduler tuning.** Use `253.980 rows/s` as the conservative tested route-mix capacity and `268.899 rows/s` as the matched-route capacity; model route-overlap sensitivity explicitly. |
| Redesign | H014-035 builds the bottleneck-aware P4 model from depth, context, batch, coarse/fine transport and state evidence, preregisters held-out predictions, and only then executes model validation. |

| Field | H014-035a (preregistered before implementation or held-out execution) |
|---|---|
| Hypothesis | A source-backed, architecture-class model fitted only to retained H014-027m and H014-033c evidence predicts three previously unmeasured real slices—KDA layer 65 warm batch-1 device p50, Gated-MLA layer 67 warm batch-1 device p50, and Gated-MLA layer 91 terminal 2,048-context last-256 device p50—with median absolute percentage error `<=10%`; every held-out execution must also pass exact output/routing, lifecycle, CUDA, safe-fixture and driver-health gates. |
| Implementation | Added separate `model-preregister` and `model-validate` commands. The former fits device p50 versus layer index within each architecture class using H014-027m layers `1/45/89` and `3/47/91`, and terminal MLA device p50 versus context tokens using H014-033c `1K/4K/8K/16K`; the latter requires the frozen binary/source hashes and never refits. |
| Benchmark | `performance/h014-035a-performance-model-preregistration.json` SHA `d3583fdc...74bda78`; calculation only over the immutable passing H014-027m/H014-033c receipts; **no held-out slice was executed or read.** |
| Result | **PASS; predictions frozen.** KDA layer 65: `3.059541 ms`; Gated MLA layer 67: `2.799040 ms`; layer-91 2,048-context last-256: `3.546724 ms`. The fixed acceptance statistic is median APE `<=10%`, denominator measured value. |
| Inspection | Depth-fit training R² was `0.973182` KDA and `0.927860` MLA; context-fit R² was `0.9999987`. The holdouts are interpolative and span both depth and context effects; individual errors and all raw distributions will remain visible. |
| Bottleneck | This cycle is preregistration, so no new bottleneck claim is made. |
| Decision | **RETAIN the immutable prediction receipt and proceed without refitting.** |
| Redesign | H014-035b executes only the three frozen holdouts. If median APE exceeds `10%`, inspect the residual by architecture/context term and change only the falsified component in a separately preregistered cycle. |

| Field | H014-035b (preregistered against H014-035a SHA `d3583fdc...74bda78`) |
|---|---|
| Hypothesis | The frozen predictions `3.059541/2.799040/3.546724 ms` for KDA-65, MLA-67 and MLA-91@2K respectively achieve median APE `<=10%`, with every real held-out correctness, lifecycle, state, CUDA, safe-fixture and driver-health gate passing. |
| Implementation | Measurement only: load the exact H014-033c DLL, profile unseen layers `65` and `67` in production telemetry, then execute an unseen 2,048-token layer-91 stateful context. Persist after every slice; compute errors only after all slices pass. |
| Benchmark | `performance/h014-035b-performance-model-heldout-validation.json` SHA `c285d5ab...ed1866`; exact H014-033c candidate DLL SHA `c3bdb40d...2ae326`; depth holdouts used ten warmups and 50 retained calls; context holdout used 2,048 sequential real CUDA state transitions, terminal last-256 p50, a known-safe exact fixture, error-state synchronization and `nvidia-smi`. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** Measured versus predicted p50 was KDA-65 `3.077472` versus `3.059541 ms` (`0.5826%` APE), MLA-67 `2.814592` versus `2.799040 ms` (`0.5526%`), and MLA-91@2K `3.554336` versus `3.546724 ms` (`0.2142%`). Median APE was `0.5526%`, far below `10%`; maximum APE was `0.5826%`. |
| Inspection | Layer-65 device p50/p95/p99 was `3.0775/3.2863/3.9176 ms`; layer 67 was `2.8146/2.8680/2.9039 ms`; 2K terminal last-256 was `3.5543/3.6400/3.7437 ms`. Maximum relative L2 errors were `2.62e-8` and `3.66e-8`, routes were exact, warm lifecycle deltas were zero, 2K state was exactly `4,720,896` bytes, and safe fixture/CUDA/error-state/VRAM-recovery/driver-health gates passed. Final process health was `2,361 MiB` used / `29,827 MiB` free. No refit occurred. |
| Bottleneck | **Architecture-class service plus linear MLA cache scan length; layer depth itself contributes less than one percent residual at the held-out points.** |
| Decision | **RETAIN the validated local model for topology and capacity search.** The hard held-out target is passed, but RTX 3090 timing and physical-network contention remain explicitly unvalidated. |
| Redesign | H014-036 expands this validated local basis into operation-class RTX 3090 factors, coarse/fine/hybrid topology candidates, state/VRAM constraints and held-open physical-canary uncertainty. |

| Field | H014-036a (preregistered) |
|---|---|
| Hypothesis | Applying separate published RTX 5090-to-3090 ceilings—`1792/936 = 1.91453x` for measured weight-memory traffic and `104.8/35.6 = 2.94382x` for FP32/context-scan compute, while host/synchronization and network stay `1x`—predicts conservative 8K aggregate capacity between `90` and `120 output tok/s` (`35–50%` of measured RTX-5090 contextual capacity). The canonical production packing of stage zero, 91 middle stages and the final layer/head into `93` RTX 3090 workers fits 24 GiB with 10% safety and at least `1 GiB` additional unallocated VRAM, has the best equal-price throughput per dollar of candidates A–D, and cannot reach `5 tok/s/user` at the admitted `5 ms / 5 Gbps` coarse edge. |
| Implementation | Added a source-hashed model that scales detailed KDA/MLA expert+dense+router weight traffic by bandwidth, scales unattributed/context-scan math by FP32 throughput, and leaves measured host/edge time unscaled. It explicitly models endpoint service, batch-8 route-mix capacity, prefill, state, measured CUDA/resident bytes, extra reserves and 10% safety; it compares A: 96-node checkpoint-aligned, B: 93-node canonical packed whole-layer, C: 461 dedicated 8-GB fine workers, and D: 185-node hybrid packed domains. |
| Benchmark | `performance/h014-036a-capacity-topology-economics.json` SHA `fe1f0e0a...0e884e`; candidate CSV SHA `21f2ee6f...dc663`; economics CSV SHA `0eaaa0f2...d73b`; inspected chart SHA `71e10cec...76e9c`. Inputs are the exact retained receipts listed above plus NVIDIA's published `936/1792 GB/s` and `35.6/104.8 FP32 TFLOPS`. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** Projected 8K batch-8 device/wall capacity is `98.627/97.457 tok/s`; device/wall retention is `38.832/39.558%`. The final packed worker plans `24,269,986,202` bytes including `2,576,980,378` bytes of explicit safety and leaves `1,499,817,574` bytes (`1.397 GiB`) additionally unallocated. Candidate B is the sole recommendation and best equal-price throughput/$; coarse-admission end-to-end decode is `1,053.31 ms`, or `0.9494 tok/s/user`, so the 5 tok/s target fails as predicted. |
| Inspection | KDA's measured weight-memory fraction is `91.864%` and mixed slowdown `1.9983x`; MLA's is `85.667%` and short-context slowdown `2.0621x`; the 8K scan increment uses the conservative `2.9438x` compute ceiling. Candidate A/B/C/D are `96/93/461/185` workers at `97.457/97.457/76.048/76.048 tok/s`. At `$0.165/GPU-h`, B costs `$15.345/h`, `$43.737/M`, and margin at `$15/M` is `-191.58%`; only `$0.05/GPU-h` is positive (`$13.254/M`, `11.64%`). Dedicated 8-GB fine workers at `$0.05/h` still cost `$84.194/M` and must fall below `$0.02597/h` merely to match B. Projected prefill at 1K/4K/8K/16K is `131.57/121.05/100.16/74.60 tok/s`, with wavefront TTFT `8.83/34.88/82.83/220.66 s`; one-stream state grows `0.499/0.651/0.853/1.257 GiB`. The chart was visually inspected after correcting label overlap. |
| Bottleneck | **Row-serial 8K Gated-MLA state scan, followed by the serial final-layer/head endpoint. Fine expert fanout reduces worker memory but not the parent/context bottleneck.** |
| Decision | **RETAIN candidate B: 93-worker WHOLE-LAYER topology. Do not put sub-layer groups in the initial fleet.** Preserve the four-worker fine path as a functional low-latency canary only. Economics at the nominal `$0.165/h` price fails; the cost guard must not authorize a full rental on performance readiness alone. |
| Redesign | H014-037 replaces the stale 96-node placement/distribution package with the exact 93-worker production packing, worker-specific tensor acquisition, clean Linux bootstrap and fail-closed admission/cost logic. The single-3090 canary remains mandatory because the GPU-transfer factors are projections, not physical timing. |

| Field | H014-036b (context-aligned memory audit) |
|---|---|
| Hypothesis | Evaluating production 8K state separately for each worker role will show that the final KDA endpoint remains the maximum 24-GiB placement and retains at least `1 GiB` unallocated after the separate 10% safety reserve. |
| Implementation | No runtime or model change. Audited stage-zero, KDA, Gated-MLA and final totals with their own measured eight-stream 8K state and retained resident-byte receipts. |
| Benchmark | `performance/h014-036b-context-aligned-memory-audit.json`; batch `8`, context `8192`, exact role-resident bytes, common CUDA baseline, `448 MiB` explicit workspace/communication/scheduler reserves and `2,576,980,378` bytes safety. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** Stage-zero/KDA/MLA/final planned totals are `7,897,520,538 / 23,141,718,426 / 23,202,029,978 / 24,269,986,202` bytes. The final role leaves `1,499,817,574` bytes (`1.397 GiB`) unallocated after safety. |
| Inspection | The suspected context mismatch was falsified. `55,050,240` bytes is the correct eight-stream KDA state for the peak final endpoint. Gated-MLA requires `151,013,376` bytes at 8K, but has `1,163,919,360` fewer resident bytes and remains `1,067,956,224` bytes below the final endpoint. |
| Bottleneck | Final KDA layer plus norm/head resident weights, not MLA state. |
| Decision | **RETAIN H014-036a unchanged.** H014-037 must record role-specific state on each worker. |
| Redesign | Continue H014-037a; no arithmetic or runtime redesign is justified. |

| Field | H014-037a (preregistered) |
|---|---|
| Hypothesis | Mechanically merging the immutable 96-worker tensor manifest into the selected 93-stage production packing (`old 0+1`, `old 2..92`, `old 93+94+95`) preserves exact one-time ownership of all `497,052` required tensors and `1,559,965,606,912` source bytes, leaves every 24-GiB worker feasible with at least `1 GiB` beyond a separate 10% safety reserve, and enables worker-scoped remote acquisition that passes partial resume, transient retry, SHA-256 verification, corruption rejection, warm-cache zero-download and atomic package activation without fetching the complete checkpoint. |
| Implementation | Added a final-manifest builder that reuses every immutable tensor byte range, with source mapping `0+1`, `2..92`, `93+94+95`. Added content-addressed remote acquisition keyed by immutable revision/SHA, `.partial` HTTP Range resume, bounded retries, cache hardlink/copy views and atomic activation through the deterministic exact-range Safetensors packager. |
| Benchmark | Full real `93`-worker manifest SHA `fdb09a39...05549b`, distribution SHA `68fde734...d3767a5`, and compact receipt `distribution/h014-037a-final-placement-distribution.json`. Fault fixture SHA `e25a9102...e6b7a`: 1-MiB source, 65,536-byte preseeded partial, one injected `503`, two corrupted bodies and one warm-cache repeat. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** Exactly `93` workers own all `497,052` tensors, `1,559,965,606,912` tensor bytes and layers `0..92` once, with no duplicate or orphan. The minimum post-safety unallocated VRAM is `1,499,817,574` bytes. All eleven acquisition fixture gates pass. |
| Inspection | Worker 0 owns embedding+layer 0 (`4,690,023,424` bytes); workers 1..91 own their exact whole layers; worker 92 owns layer 92+final/head (`18,915,537,408` bytes). KDA/MLA eight-stream state is `55,050,240 / 151,013,376` bytes. Median/worst cold downloads are `16,990,916,912 / 21,265,171,248` bytes from one/two immutable source shards versus the `1,560,041,353,768`-byte checkpoint; no worker downloads the checkpoint. Range resume survived the injected 503, corruption never activated, and the verified-cache repeat made zero HTTP requests. |
| Bottleneck | **Cold source-shard transfer**, not exact tensor extraction: median `15.824 GiB`, worst `19.805 GiB`; warm generation performs no acquisition. |
| Decision | **RETAIN and promote the 93-worker placement and worker-scoped acquisition contract.** |
| Redesign | H014-037b packages clean Linux bootstrap, hardware/network admission, PREPARE/canary/READY, cost guard, whole-stage recovery and exact logical rehearsal around these immutable assignments. |

| Field | H014-037b (preregistered) |
|---|---|
| Hypothesis | A source-locked, no-checkout Linux canary package can fail closed before model activation on wrong GPU, compute capability, VRAM, network class, immutable assignment, runtime hash or cost; the exact 93-worker graph can rehearse multiple stateful generations with every worker exercised; and whole-stage timeout, loss, partial, duplicate, stale-generation and cancellation faults expose no incomplete output, after which exact replacement ownership can resume. |
| Implementation | **Pending.** Build the release wheel and CUDA-source bundle, Linux worker/coordinator bootstrap, hardware/network admission, one-3090 canary, optional two-GPU fine canary, cost guard, artifact collector/cleanup, exact execution plan, logical rehearsal and coarse-stage recovery fixture. The Linux CUDA object remains build-on-canary and must be hash-locked and fully qualified before fleet activation because this Windows host cannot emit or execute an ELF CUDA runtime. |
| Benchmark | **Pending.** Static package validator; bounded hardware/cost fixtures; exact 93-worker/93-layer/82,432-expert ownership rehearsal for three generations; six injected whole-stage response faults plus replacement; shell syntax checks where available. |
| Result | **PENDING.** |
| Inspection | Inspect graph coverage, routes, state transitions, all worker operation counts, fault output exposure, retry/replacement identity, bootstrap ordering, immutable hashes, cost decision and absence of repository-checkout dependencies. |
| Bottleneck | Pending measurement. |
| Decision | Promote the package to **READY FOR SINGLE-3090 CANARY** only if all local gates pass; never call the Linux CUDA binary certified before the physical canary builds and qualifies it. |
| Redesign | H014-038 qualifies the exact final Windows binary, regenerates operation/sm_86 evidence and runs full repository regressions; final reporting then distinguishes local pre-canary completion from physical Linux/3090 unknowns. |

| Field | H014-037b1 (preregistered before implementation) |
|---|---|
| Hypothesis | Carrying an exact native-runtime path/SHA on every bound stage assignment, preserving it through initial and replacement `LoadStageRequest`s, allowing a CUDA worker to advertise `native-cuda:0` while probing the same device as `cuda:0`, and pinning the activated worker snapshot with a canonical model-identity document is sufficient to make the existing transactional product runtime load the Kimi CUDA executor without any CPU mathematical fallback. A retained dry-run fixture must prove all `93/93` assignments are exact and that both initial and recovery requests carry the same runtime identity; malformed or absent runtime/assignment/model identity must reject. |
| Implementation | Added a paired native-library path/SHA contract to `PlanWorkerAssignment`, one canonical request builder shared by initial deployment and recovery, production-batch propagation, `native-cuda:0` CUDA probing, configured snapshot identity propagation through CLI/service/runtime, pre-registration snapshot hash/worker/fingerprint checks, and a deterministic binder from the immutable placement plus `swarm workers --json` to the existing typed `ProductStagePlan`. Kimi arithmetic, routing, READY, tensor transport and placement were unchanged. |
| Benchmark | `deployment/h014-037b1-deployment-identity-fixture.json` SHA `23f130f5...da2d7be`; exact final placement SHA `82c0f280...2c85e0`; `93` assignments, `93` initial and `93` recovery requests, typed JSON round trip, production batch `8`, context `8,192`, worker-local Linux runtime path and candidate SHA `c3bdb40d...ae326`. Negative controls removed one worker and injected wrong device, wrong model fingerprint, unhealthy registration, endpoint mismatch, malformed runtime SHA and a missing path/SHA pair member. |
| Result | **PASS; HYPOTHESIS SUPPORTED FOR THE LOGICAL PRODUCT PATH.** All `13/13` gates passed. Every assignment and all `186` load requests carried exact model revision/fingerprint, adapter, native-runtime path/SHA, `native-cuda:0`, FP32, resident fast path, batch `8` and context `8,192`; recovery alone advanced route generation. All seven negative controls rejected before execution. |
| Inspection | The actual product blocker was not Kimi compute. Three identity seams were incomplete: assignment identity stopped before the coordinator request, `native-cuda:0` was accepted by CLI but rejected by worker-service validation, and worker registration did not attest the configured worker-only snapshot. Initial and recovery now use the same request constructor, preventing drift. The activated snapshot remains additionally checked by the Kimi adapter for exact tensors and stage ownership before load, and `prepare_for_ready` remains the seven-call READY seam. |
| Bottleneck | **Physical Linux runtime qualification.** This Windows host cannot build or execute the final ELF CUDA library; its path and digest must be replaced by the canary-built, fully re-certified Linux identity before a real fleet plan can bind. |
| Decision | **RETAIN.** The 93-worker logical deployment identity path is fail-closed and ready to package. Do not treat the fixture endpoints or Windows candidate SHA as a deployable Linux fleet plan. |
| Redesign | Complete H014-037b packaging and static validation. H014-038 then qualifies the exact final Windows binary/regressions; Experiment 015 first builds and certifies the Linux sm_86 runtime on one physical 3090 and binds the real registered worker document before any fleet activation. |

| Field | H014-037b2 (preregistered after retained baseline failure) |
|---|---|
| Hypothesis | A stage-zero-only snapshot containing the exact Kimi tokenizer allowlist, with every asset SHA-256 bound into `model-identity.json` and reverified immediately before `AutoTokenizer(..., trust_remote_code=True)`, will make product text tokenization work without granting unaudited checkpoint code execution. It must produce the exact retained Kimi token IDs; a modified asset, missing asset, wrong worker or non-stage-zero use must reject. |
| Implementation | Atomic activation now copies only `tokenizer_config.json`, `tiktoken.model`, `tokenization_kimi.py` and `encoding_k3.py` to stage zero, binds every SHA-256 and `owns_embeddings` into `model-identity.json`, and re-verifies the exact allowlist at worker startup and immediately before local custom-code loading. Non-stage-zero snapshots remain tensor-only. |
| Benchmark | Baseline failure: `deployment/h014-037b2-tokenizer-baseline-failure.json`. Implemented seam receipt: `deployment/h014-037b2-tokenizer-product-seam.json` SHA `de015c45bf3c4f62c2e73f58f873ca4cfc9c5388ad08c15d4f52f65b214841c0`; real checkpoint metadata, three prompts, exact source/snapshot IDs, and wrong-worker/tampered/missing/non-stage-zero/expanded-allowlist negative controls. |
| Result | **SECURITY/LINEAGE PASS; PRODUCT-SEMANTICS FAIL.** All four authentic assets (`2,837,736` bytes) matched source hashes, source and activated-snapshot raw token IDs were exact for all prompts, and all five negative controls rejected. The retained `Hi` oracle requires two prompt tokens, but the custom tokenizer returned raw `[18699]`; one acceptance gate failed. |
| Inspection | The native Kimi serving path explicitly inserts `m->c.bos` before encoding (`kimi_k3.c:1534`), whereas `TikTokenTokenizer(..., add_special_tokens=True)` does not add BOS. The product seam authenticated the correct tokenizer but exposed a separate adapter semantic: raw checkpoint tokenization is not the complete engine prompt contract. |
| Bottleneck | **Missing explicit Kimi BOS insertion after authenticated raw tokenization.** This is not an asset-trust or vocabulary mismatch. |
| Decision | **MODIFY; retain the failed receipt.** Keep the exact-hash trust boundary unchanged and do not reinterpret the one-token result as equivalent. |
| Redesign | H014-037b2a tests one repository-owned Kimi prompt adapter that prepends the authenticated tokenizer's BOS exactly once when `add_special_tokens=True`, preserves raw IDs when false, and rejects an invalid BOS configuration. |

| Field | H014-037b2a (preregistered) |
|---|---|
| Hypothesis | Applying the native engine's explicit BOS rule after verified Kimi tokenization will produce `[163584, 18699]` for `Hi`, match the retained two-token prompt oracle, avoid duplicate BOS when already present, and leave `add_special_tokens=False` raw output unchanged. |
| Implementation | Added centralized `apply_kimi_prompt_special_tokens`; the Kimi product path now requests raw checkpoint tokens and applies the native engine's explicit BOS rule exactly once. Generic adapters are unchanged. Invalid BOS rejects. |
| Benchmark | `deployment/h014-037b2a-tokenizer-product-seam.json` SHA `85dec874d487117eb344155306a2a11c052684aa6adee5ce18fffc286ab18e02`; the exact H014-037b2 fixture plus exact BOS ID, exactly-once BOS, no-special-token identity and invalid-BOS controls. A sandbox-only native-extension import failure occurred before execution on the first attempt; the same command ran outside that restriction and retained the scientific result. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** All `15/15` gates passed. `Hi` became exactly `[163584, 18699]`; all three activated-snapshot product vectors matched the authenticated source vectors after the same adapter; already-prefixed input was unchanged; `add_special_tokens=False` preserved raw `[18699]`; invalid BOS and all five prior identity/asset controls rejected. |
| Inspection | The one-token discrepancy was prompt framing, not tokenizer corruption. The stage-zero snapshot still contains exactly four tokenizer assets totaling `2,837,736` bytes, and no other worker receives executable tokenizer code. Tokenizer construction remains a cold cached lifecycle cost, outside warm CUDA service. |
| Bottleneck | **Resolved locally:** native/product BOS framing. Remaining deployment bottleneck is packaging and physical Linux runtime qualification. |
| Decision | **RETAIN the hash-pinned tokenizer and explicit Kimi-only BOS adapter.** Text admission is now runnable and fail-closed for the tested plain-prompt contract. |
| Redesign | Continue H014-037b deployment-package validation. Exercise text submission and full chat framing again on the physical 3090 canary; do not broaden the tokenizer-code allowlist. |

| Field | H014-037b3 (preregistered after package audit) |
|---|---|
| Hypothesis | A platform-specific Linux ELF can safely replace the Windows evidence binary only when a physical RTX 3090 qualification certificate binds its own SHA-256 to the exact three-file native-source manifest, compile flags, placement SHA, Windows reference SHA, sm_86 SASS, compute_86 PTX, batch guard, all 11 Kimi operation classes, P1 stage roles, READY and post-test CUDA health. The production binder must accept that distinct ELF SHA and reject a bare hash, logical fixture certificate, wrong platform, changed source, changed placement or any failed gate. |
| Implementation | Added an immutable three-file native-source manifest and exact `nvcc` argument contract; added a physical Linux qualification-certificate schema with an exact 18-gate allowlist, 11 operation classes and four stage roles; changed the production binder to accept only a validated physical certificate. Logical fixture certificates are accepted only through a private test flag. |
| Benchmark | `deployment/h014-037b3-cross-platform-runtime-identity.json` SHA `4d11dc228a62781ca85fedf534df4930ba6fbbad58cd43155fb87b24596dc128`; 93 assignments and 186 initial/recovery requests using synthetic Linux SHA `5a81bed7...e693b`, intentionally distinct from Windows `c3bdb40d...ae326`. Twenty-one gates include bare/fixture/wrong-platform/source/placement/failed-gate/Windows-hash-reuse controls. |
| Result | **PASS; HYPOTHESIS SUPPORTED LOGICALLY.** All `21/21` gates passed. The distinct ELF identity propagated to every assignment and request; the production validator rejected the logical certificate itself, proving it cannot authorize a fleet. All seven prior worker/plan controls also remained fail-closed. |
| Inspection | Cross-platform equivalence is now source/build/physical-evidence equivalence rather than impossible byte equality. The source bundle comprises `backend_cuda.cu` SHA `8dd70504...c6a8`, `backend_cuda.h` `b2a5b735...2ca7`, and `backend_gpu_compat.h` `bc1cd457...d455`; source-manifest SHA is `fe4aa930...4ac95`. No local fixture can mint `physical_3090` evidence. |
| Bottleneck | **Physical qualification remains intentionally open:** the ELF SHA and physical certificate do not exist until the single-3090 canary runs. |
| Decision | **RETAIN the certificate-only production binder.** Reject the old bare `--native-runtime-sha256` interface. |
| Redesign | Package the exact sources/build contract and a canary that alone emits the physical certificate after all 18 gates. Then bind registered fleet workers to that certificate. |

| Field | H014-037b4 (preregistered before implementation) |
|---|---|
| Hypothesis | A hash-locked, no-repository-checkout Experiment 015 package can contain the release wheel, exact Kimi CUDA sources/build contract, immutable placement/distribution manifests, worker-scoped acquisition, clean Linux bootstrap, cost/network admission, artifact collection and cleanup, while making fleet binding impossible until a physical RTX 3090 canary emits the exact H014-037b3 certificate. A static validator must accept the complete package and reject a removed file, changed byte, broadened native source set, relaxed cost guard, repository-checkout dependency, logical certificate or unqualified fleet launch path. |
| Implementation | **PENDING.** Add the minimum physical-canary runner and deterministic package builder/validator. The runner must exercise the production stage loader sequentially for stage zero, real layer-89 KDA+MoE, real layer-91 Gated MLA+MoE and final/head/sampling; verify batch 8, over-limit rejection before CUDA, exact seven-call PREPARE, warm lifecycle, operation/stage matrices and post-test GPU health; and emit a production certificate only when all 18 gates pass. This Windows cycle will package and validate that procedure but will not manufacture physical evidence. |
| Benchmark | **PENDING.** Build one package into `artifacts/experiment-015-deployment/`; verify every byte through `package-lock.json`; execute its static validator and isolated negative controls; run Python/unit/CLI checks and shell syntax parsing where a Linux shell is available. Retain a package-validation receipt with exact wheel, source-manifest, placement, distribution, canary-fixture and script hashes. |
| Result | **PENDING.** Physical canary gates remain NOT RUN by construction on this Windows/non-3090 host. |
| Inspection | Inspect bootstrap order, worker-specific tensor scope, certificate-only fleet activation, source/build equivalence, package closure, cost/network fail-closed behavior, READY/canary sequencing and absence of development-checkout assumptions. |
| Bottleneck | **Unknown locally:** candidate risks are incomplete package closure or a canary that bypasses the production load/identity path. Physical RTX 3090 performance and ELF execution remain Experiment 015 measurements. |
| Decision | Retain only if all local package gates and tamper controls pass. Never translate static validation into a physical CUDA PASS. |
| Redesign | If local validation passes, complete H014-038 exact final Windows-binary regression and regenerate the final report as locally complete but explicitly pending the preregistered physical single-3090 canary before fleet rental. |

| Field | H014-037b4a (preregistered after retained fixture-build failure) |
|---|---|
| Hypothesis | Encoding the dense stage-zero role as exactly three empty expert selections, while preserving real top-16 routes for layers 89, 91 and 92, is sufficient for the compact physical-canary fixture bundle to build and pass every source/hash/shape gate. |
| Implementation | Changed only the route-array construction for layer 0 from an invalid oracle lookup to an empty `int64 [3,0]` array. No CUDA, checkpoint, numerical reference, MoE route or package rule changed. |
| Benchmark | Baseline `deployment/h014-037b4-fixture-baseline-failure.json` SHA `9767cd15...89cb5`; corrected real checkpoint/oracle bundle `h014-037b4-physical-canary-fixtures.npz` SHA `e06008bd...6a109`, manifest SHA `6fc34ade...24529`. The same command read `F:\\models\\Kimi-K3`, the retained three-position trace/routes/logits and passing placement/P0 references. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** Sixteen allowlisted arrays compressed to `4,877,706` bytes. Layer 0 is exactly `int64 [3,0]`; layers 89, 91 and 92 are each exactly `int64 [3,16]`. Every array records shape, dtype and raw-byte SHA-256, and the NPZ, placement, config, index, oracle and reference receipts are hash-bound. |
| Inspection | The canonical route trace contains routed-MoE calls; layer 0 is dense and its production execution record correctly has no selected experts. The correction removed only a false router assumption and preserved all 144 real representative expert IDs. |
| Bottleneck | **Harness-only unconditional route lookup for a non-MoE layer.** |
| Decision | **RETAIN the compact bundle and exact dense/MoE distinction.** Physical execution remains NOT RUN in the manifest. |
| Redesign | Resume H014-037b4 package construction after the compact fixture bundle passes. |

| Field | H014-037b4b (preregistered after package-path inspection) |
|---|---|
| Hypothesis | Resolving a relative placement reference from the distribution manifest's own directory and packaging the placement as its exact sibling will make worker-scoped acquisition portable to a clean Linux node without changing any of the 93 assignments, 497,052 tensor owners, immutable shard URLs/hashes or source bytes. Absolute, missing, traversal and hash-mismatched placement references must reject. |
| Implementation | **PENDING.** Add one path-resolution helper to remote acquisition and have the package builder canonicalize only `placement_manifest` to the sibling filename while retaining `placement_manifest_sha256`. Do not modify shard acquisition, tensor extraction or activation. |
| Benchmark | Baseline `deployment/h014-037b4b-absolute-placement-path-failure.json` records the Windows-only path. Validate the canonical package manifest, replay the existing resume/retry/corruption/cache fixture, verify all 93 requirements, and run relative/missing/traversal/hash-mismatch controls. |
| Result | **PENDING.** |
| Inspection | The retained distribution evidence is mathematically complete but its operational placement lookup is tied to this development checkout. |
| Bottleneck | **Absolute development-machine placement path, not model distribution or tensor ownership.** |
| Decision | Retain only if portability changes no ownership or source identity and all path controls fail closed. |
| Redesign | Resume deterministic package assembly after the portable manifest seam passes. |

| Field | H014-037b4c (preregistered after first complete-package inspection) |
|---|---|
| Hypothesis | Publishing the exact canary-built Linux ELF and certificate as immutable SHA-256-addressed artifacts, then running one bounded real assigned-stage PREPARE/smoke canary on every worker snapshot, is sufficient to produce a valid FLEET admission receipt on all 93 nodes without recompiling an unqualified runtime or weakening certificate-only binding. A wrong ELF hash, logical certificate, wrong assignment, unsafe batch or failed assigned-stage execution must reject before worker registration. |
| Implementation | **PENDING.** Add the minimum worker-local assigned-stage canary and package scripts for exact qualified-runtime acquisition, certificate validation, post-canary worker requirements, assigned-stage canary, CUDA-health inspection and FLEET admission. The single RTX 3090 canary remains the only source of the physical Linux certificate; fleet nodes consume its exact ELF and certificate. |
| Benchmark | **PENDING.** Run bounded logical fixtures for stage-zero, KDA, MLA and final role construction; validate the rebuilt package; and add tamper controls for wrong qualified-runtime SHA and missing assigned-stage receipt. Physical per-node execution remains NOT RUN on this Windows host. |
| Result | **PENDING.** |
| Inspection | The first package passed byte/source/path/cost/certificate closure, but its worker launch precondition (`fleet-admission.json`) had no producer and independently compiled fleet ELFs were not mechanically tied to the canary hash. |
| Bottleneck | **Missing operational seam from the single physical canary artifact to exact per-node runtime identity and assigned-stage FLEET admission.** |
| Decision | Do not retain the first package as runnable. Retain its positive and negative evidence as the baseline that exposed the lifecycle gap. |
| Redesign | If the new seam passes, rebuild the exact wheel/package and rerun all package controls before H014-038. |

| Field | H014-037b4d (preregistered after qualified-runtime package inspection) |
|---|---|
| Hypothesis | A standalone standard-library package-lock verifier executed before wheel installation, exact RTX 3090 hardware preflight before dependency/model transfer, atomic qualified-runtime activation, public-fingerprint-only worker trust, and certificate/runtime revalidation immediately before worker registration will close the remaining clean-node and identity seams. A tampered package, wrong hardware, copied private identity workflow, partial runtime, or stale/mismatched certificate must fail before registration. |
| Implementation | **PENDING.** Add only those bootstrap checks and public-identity export/import steps; do not change topology, arithmetic, batching, placement or certificate gates. |
| Benchmark | **PENDING.** Rebuild the wheel/package; execute the standalone verifier; run the complete package validator and tamper controls; inspect every generated launch script; and run focused unit/CLI tests. Linux shell parsing is attempted only if a usable Linux shell exists. |
| Result | **PENDING.** |
| Inspection | The H014-037b4c package correctly introduced exact ELF distribution and per-node assigned-stage FLEET admission, but package verification depended on code inside the package, runtime installation was not atomic, hardware preflight occurred after Python installation, and the trust helper used `--identity` for documents intended to contain only public metadata. |
| Bottleneck | **Bootstrap trust ordering and public/private identity mismatch, not CUDA or placement.** |
| Decision | Do not retain the second package as the final handoff. |
| Redesign | Retain only after byte tampering and private-key transport are mechanically excluded before installation/registration. |

| Field | H014-038 (preregistered before exact-final-binary execution) |
|---|---|
| Hypothesis | The exact final Windows CUDA candidate `coli_cuda-sm86-h014-033c-mla16k-candidate.dll` SHA-256 `c3bdb40d49a1b0e512ddb1e84485e0bdcf9d2e6ac6fa4049d81ff2ecd12ae326`, together with the production execution-record change that exposes routed-expert completion, preserves every established P0/P1 numerical, routing, state, READY, lifecycle and fail-closed batch property. On that one hash, all 11 Kimi operation classes, sm_86 SASS plus compute_86 PTX, the 93-layer graph, stateful decode, stage zero, representative non-final KDA/MLA stages and final/head/sampling must pass, batch 9 must reject before unsafe routed-expert CUDA, and pre/post CUDA plus `nvidia-smi` health must remain normal. |
| Implementation | No arithmetic redesign. Re-run the retained real-checkpoint fixtures against only the exact candidate hash; regenerate component receipts, operation matrix, sm_86 certificate, full-graph/P1 receipts and one promotion receipt. Promote manifests/package inputs only if every gate passes. |
| Benchmark | **PENDING.** Real checkpoint `F:\\models\\Kimi-K3`; retained independent activation/trace/route/logit fixtures; RTX 5090 UUID `GPU-8e79b31d-7efe-e59d-a370-0e0add109e49`; bounded component timings; complete 93-layer graph and stateful second token; exact seven-call PREPARE for all four roles; batch-limit query and pre-launch rejection; CUDA synchronize and GPU identity/free-memory checks before and after. |
| Result | **PENDING.** |
| Inspection | Compare numerical/routing/state fingerprints to retained H014-027w evidence and inspect operation coverage, cubin/PTX targets, routed-expert execution counts, lifecycle deltas, CUDA error state and VRAM recovery. |
| Bottleneck | Unknown until the exact final hash is exercised. Any mismatch is a release blocker, not evidence that may be inherited from H014-027w. |
| Decision | Promote only on unanimous PASS; otherwise retain the candidate as quarantined and redesign from the first failing gate. |
| Redesign | If supported, rebuild the exact 93-worker manifests and Experiment 015 package around the promoted identity, rerun package controls and full repository regressions, then regenerate all final Experiment 014 outputs. |

| Field | H014-038a (preregistered after retained full-graph fixture failure) |
|---|---|
| Hypothesis | H014-038's graph failure is caused entirely by omitting Kimi's required BOS token from the harness input, not by the final CUDA binary. Re-running the unchanged binary and unchanged deterministic oracle with exact prompt IDs `[163584, 18699]` will restore layer-0 agreement, exact routing and the established approximately `1e-6` graph error; if it does not, the binary remains unqualified. |
| Implementation | No source, binary, arithmetic, oracle or tolerance change. Retain the failed one-token receipt as `cuda/h014-038-invalid-missing-bos-full-93-layer.json`; change only the CLI invocation to append both checkpoint-authoritative token IDs in order. |
| Benchmark | Invalid receipt SHA `e422d76081f8db8e171bbc5d261a98583293f88026e16d4e9deb04f9fb8aae48` executed all 93 layers but used prompt IDs `[18699]`; divergence began at dense layer 0 (`0.966289` relative L2), then routing diverged at layer 1 and maximum error reached `9.958841`. GPU UUID, free VRAM and P8 health were unchanged afterward. Corrected run uses the same exact H014-038 graph command with `--prompt-token-id 163584 --prompt-token-id 18699`. |
| Result | **PENDING.** |
| Inspection | The oracle trace SHA was already correct (`0a432e25...a8d`); layer-0 divergence precedes any router or expert, isolating the mismatch to input embedding semantics. This reproduces the BOS seam independently identified by H014-037b2a. |
| Bottleneck | **Harness input omitted the model's required BOS token.** This is an invalid fixture, not CUDA evidence. |
| Decision | Retain the failure and retry exactly once with checkpoint-authoritative special-token application. Do not change the numerical gate. |
| Redesign | If corrected execution passes, continue H014-038. If layer 0 still differs, stop and inspect embedding/input construction before any further CUDA work. |

| Field | H014-038b (preregistered after H014-038a external timeout) |
|---|---|
| Hypothesis | H014-038a would have completed successfully if the external process allowance exceeded its cold checkpoint-streaming wall time. The identical corrected command with a 90-minute allowance will finish all five remaining decode layers, retain exact routing, remain below the unchanged `2e-6` error gate, and emit a complete atomic receipt without changing binary, model, oracle or arithmetic. |
| Implementation | No code or benchmark-parameter change. Increase only the shell process allowance from 60 to 90 minutes. Retain `cuda/h014-038a-timeout-after-decode88.json` rather than treating the partial stdout as a PASS. |
| Benchmark | H014-038a completed prefill `93/93` with exact routing and maximum relative L2 `9.92226908e-7`; it reached decode `88/93` with exact routing and maximum relative L2 `9.77985551e-7`, then the external command timed out at `3600.2 s` before atomic receipt creation. Post-timeout GPU health returned to the same UUID with `30,338 MiB` free and P8. H014-038b repeats the exact command with a 5,400-second process bound. |
| Result | **PENDING.** |
| Inspection | The missing-BOS hypothesis is already numerically supported through 181 layer executions; the remaining uncertainty is complete receipt closure, final five decode layers, head/sampling and stateful aggregate gates. |
| Bottleneck | **Cold checkpoint streaming exceeds the original external timeout.** Warm serving latency is measured by separate resident-stage evidence. |
| Decision | Retain only a complete receipt. Timeout output remains diagnostic, never promotable evidence. |
| Redesign | On PASS, continue P1 regression. On another timeout, add measured incremental receipt persistence/resume as a new harness hypothesis rather than extending the timeout blindly again. |

| Field | H014-038c (preregistered after registered-stage identity rejection) |
|---|---|
| Hypothesis | The P1 failure is caused solely by using the pre-placement H014-026b identity document after the runtime began requiring worker-scoped assignment identity. A new identity that preserves every checkpoint/model/tokenizer field and adds worker `k3-worker-092` plus its exact final-placement assignment SHA `df71446e...d8cd90` will pass the registered final-stage load and retain all established numerical, READY and lifecycle properties. |
| Implementation | Do not weaken `PersistentStageRuntime`. Retain the rejection in `persistent/h014-038c-stale-identity-failure.json`; create `h014-038-worker-092-model-identity.json` by adding only the final placement's exact worker and assignment identity to the unchanged H014-026b checkpoint identity. Re-run the identical final-stage command. |
| Benchmark | First attempt failed before weight load or CUDA with `configured model identity has no valid assignment SHA-256`. Corrected identity is bound to final placement `h014-037a-final-physical-placement.json` worker 92 assignment SHA `df71446e982e05d07033ea0141d55ef30c7436719976deaabdd8b30d57d8cd90`; same final candidate binary, deterministic oracle and registered production loader. |
| Result | **PENDING.** |
| Inspection | This rejection was introduced by the later deployment-identity hardening and therefore required explicit regression rather than inheritance from H014-027w. It is a successful fail-closed control, but leaves P1 execution untested until the scoped identity passes. |
| Bottleneck | **Stale regression fixture identity, not model loading or CUDA.** |
| Decision | Retain the fail-closed validator and replace only the stale fixture document. |
| Redesign | If the corrected final role passes, use role-correct scoped identities for the representative non-final and stage-zero P1 regressions; never reuse an identity whose assignment does not match the tested worker role. |

| Field | H014-038d (preregistered after H014-038c final-role PASS) |
|---|---|
| Hypothesis | Resolving a worker-scoped identity from a directory by exact `k3-worker-NNN-model-identity.json` name, and rejecting a missing file, wrong worker ID or invalid assignment digest before load, is sufficient for the aggregate layer-1 KDA, layer-3 Gated-MLA and layer-0 stage-zero P1 fixtures to use their exact final-placement assignment identities without changing runtime, arithmetic or lifecycle behavior. |
| Implementation | Modify only the persistent regression harness: permit its existing `--identity-manifest` path to be a directory; resolve the exact layer worker document; validate worker ID and 64-hex assignment SHA before constructing `PersistentStageRuntime`; retain direct-file compatibility for a correctly scoped file; record identity path/hash/assignment in each stage receipt. Create scoped documents for workers 000/001/003 from the unchanged checkpoint identity plus final-placement assignment SHAs. |
| Benchmark | Unit controls for directory resolution, direct-file compatibility, missing identity, wrong worker and malformed assignment. Then run the same registered non-final and stage-zero real-checkpoint regressions on final binary SHA `c3bdb40d...ae326`, using placement assignments worker 000 `ba236357...1c154`, worker 001 `93cad06e...fdb15`, and worker 003 `a353515f...7fd86`. |
| Result | **PENDING.** |
| Inspection | H014-038c proved the final role itself passes at maximum relative L2 `2.505e-7`, exact routing, `19,419,627,520` resident bytes and all-zero warm deltas. The remaining problem is aggregate harness identity selection across distinct workers. |
| Bottleneck | **One CLI identity path was reused across multiple worker roles; the runtime correctly requires worker-scoped assignment identity.** |
| Decision | Retain only if all identity controls fail closed and all three real roles pass without arithmetic changes. |
| Redesign | If supported, finish H014-038 promotion. If a stage fails after identity resolution, diagnose its first model/CUDA/lifecycle gate separately. |

| Field | H014-038e (preregistered after final batch-contract inspection) |
|---|---|
| Hypothesis | The final executor incorrectly exposes native-supported batch 16 as production-certified capacity even though complete contextual certification and the selected serving topology stop at batch 8. Capping the production executor at `min(8, native_supported_max)` will preserve real complete-stage B8 execution and make B9 reject before CUDA with unchanged execution records/counters, healthy error state and a passing post-rejection B1 fixture. |
| Implementation | Add one explicit production batch ceiling constant (`8`) and distinguish production-certified from native-supported sizes in lifecycle evidence. Extend the already required real P1 non-final regression to retain the seven-call PREPARE receipt, execute one real B8 call, attempt B9 under before/after sentinels, synchronize CUDA, and run one known-safe real B1 call. No kernel, weight, route, reduction, state or tolerance change. |
| Benchmark | Baseline H014-038c final-stage lifecycle exposes `batch_capacity=16` and native sizes `[1,2,4,8,16]`, while the Experiment 015 canary code requires B9 rejection. Unit-test the pure capacity rule and pre-CUDA B9 ordering. Re-run real layer-1 KDA+MoE and layer-3 Gated-MLA+MoE with exact worker identities, final binary SHA `c3bdb40d...ae326`, real B8 inputs/routes/weights, B9 sentinel, post-guard B1 and post-run CUDA/VRAM health. |
| Result | **PENDING.** |
| Inspection | The fixed row-cooperative workspace is already allocated in the executor constructor before PREPARE/READY. The mismatch is semantic exposure of an uncertified complete-stage size, not a late-allocation or kernel-capacity problem. |
| Bottleneck | **Native capability was conflated with the selected production certification ceiling.** |
| Decision | Retain only if B8 passes unchanged, B9 reaches no CUDA work, the safe fixture passes, and all P1/READY/lifecycle gates remain green. |
| Redesign | If supported, treat safe certified batch as 8 and continue exact final promotion. If B8 regresses, revert the cap change and inspect capacity wiring before further execution. |

| Field | H014-038f (preregistered after H014-038e strict-memory failure) |
|---|---|
| Hypothesis | H014-038e's only failure is a fixed `2,097,152`-byte CUDA allocator/context plateau on first batch-session use, not growing request state. Accepting at most 4 MiB initial allocator retention while requiring no more than 1 MiB additional growth after the post-guard B1 will preserve all safety gates and make both real stages pass. |
| Implementation | Retain the failed receipt as `persistent/h014-038e-strict-memory-threshold-failure.json`. Change only the fixture's memory interpretation: record exact initial-retained and post-safe-additional bytes; allow initial retention `<=4 MiB`; require post-safe additional retention `<=1 MiB`. Do not change executor allocation, CUDA, batch cap, state, routes or numerical thresholds. |
| Benchmark | H014-038e receipt SHA `4229d417...9e9d`: layer 1 B8 `10.655904 ms`, max relative L2 `7.743e-8`; layer 3 B8 `24.120159 ms`, max relative L2 `2.600e-7`; both exact routes/selected-once/zero lifecycle deltas; both B9 guards and safe B1 passed. Each retained exactly `2,097,152` bytes after B8 close and zero additional bytes after safe B1. Re-run the same two-stage fixture and inspect both deltas plus post-run `nvidia-smi`. |
| Result | **PENDING.** |
| Inspection | Identical 2 MiB retention on different attention architectures, followed by no further growth, is inconsistent with per-session KDA/MLA state leakage and consistent with a fixed CUDA allocation granule/cache plateau. |
| Bottleneck | **Over-strict 1 MiB fixture tolerance, not batch execution or state cleanup.** |
| Decision | Retain only if the repeat remains within both preregistered bounds; otherwise investigate allocation ownership rather than widening tolerance again. |
| Redesign | On PASS, rerun final/stage-zero roles on the new production cap, promote the exact software/binary identity and rebuild the package. |

| Field | H014-038g (preregistered after stage-zero PREPARE recovery failure) |
|---|---|
| Hypothesis | Stage zero's seven-call PREPARE failure is the same fixed 2 MiB CUDA allocator plateau characterized by H014-038e/f, not unreleased stage state. Recording exact retained bytes and accepting at most 4 MiB will make PREPARE pass while all existing finite-output, isolated-session, record-removal, execute-count-restoration and active-session-zero gates remain mandatory. |
| Implementation | Retain `persistent/h014-038f-stage-zero-prepare-memory-failure.json`. Add a named 4 MiB PREPARE recovery tolerance; record `retained_after_close_bytes` and the exact tolerance in the production PREPARE receipt; change no session close, CUDA free, state, arithmetic or lifecycle behavior. |
| Benchmark | Failed receipt SHA `7cc5cef2...e2b9`: seven PREPARE calls completed; output finite; records removed; serving count restored; active sessions zero; stage output relative L2 `2.752e-7`; warm lifecycle zero. Free memory changed `29,388,439,552 → 29,386,342,400`, exactly `2,097,152` bytes. Unit-test the recovery bound, rerun stage zero, then inspect exact retained bytes and post-run `nvidia-smi`. |
| Result | **PENDING.** |
| Inspection | The amount and non-growth match the independently repeated KDA/MLA batch allocator plateau. The prior boolean lost this distinction by demanding byte-exact `cudaMemGetInfo` recovery. |
| Bottleneck | **Exact-byte CUDA free-memory equality in PREPARE evidence, not unreleased Kimi state.** |
| Decision | Retain only if exact retention remains `<=4 MiB`; never suppress or omit the retained-byte value. |
| Redesign | On PASS, re-run the final dependency chain as needed, then produce the H014-038 promotion/integrity receipt. |

| Field | H014-038h (preregistered after late native-thread lifecycle failure) |
|---|---|
| Hypothesis | The final-role lifecycle failure is caused by a native CPU helper that is started asynchronously by the first canonical boundary pack/unpack after the seven-call CUDA fixture. Exercising one bit-exact canonical `float32 [1,9,7168]` transport round trip inside PREPARE and requiring the complete OS-thread set to remain unchanged for five consecutive 50 ms samples will force that one-time helper startup before READY; the subsequent retained generations will then create zero OS/Python threads, processes, tasks, connections, buffers, weights or model materializations. |
| Implementation | Retained `persistent/h014-038g-late-native-thread-failure.json` (SHA-256 `81214478ae4d2abe452176ce15967c745cf36fa08a300f31dfeffe851b89d8a7`). Extended, but did not replace, the exact seven-call assigned-stage PREPARE fixture with one no-compression pack/unpack of its real output boundary and a bounded two-second thread-quiescence gate. Recorded the transport checksum/bytes, bit-exact round trip, thread-set transitions and quiescence duration. No CUDA arithmetic, routing, weights, session state, batch path or numerical threshold changed. |
| Benchmark | Exact final role, worker-092 identity, final binary SHA `c3bdb40d...ae326`, seven PREPARE executions, one transport round trip, one existing warmup generation and three retained generations. The original failure had `66` OS threads before and `67` after, added TID `33096`, unchanged Python thread IDs, zero per-generation deltas, exact routing and maximum relative L2 `2.505e-7`. H014-038h used a mismatched non-`idot0` oracle family; its receipt is retained only as lifecycle/configuration evidence. |
| Result | **LIFECYCLE PREDICTION SUPPORTED; NUMERICAL BENCHMARK INVALID.** The `258,048`-byte round trip was bit exact, the thread set stabilized, and whole-window/per-call thread creation became zero. The apparent `3.056e-3` numerical error was later proven by H014-038j to come from the wrong oracle family, so it is not a CUDA result. |
| Inspection | The extra thread appeared only between the last per-call snapshot and the immediately following whole-window snapshot; each individual generation reported `thread_creation=0`, and process/task/connection/allocation/weight/materialization deltas were all zero. That timing falsifies a synchronous per-generation thread allocation and isolates a delayed native initialization or observation race. |
| Bottleneck | **READY does not currently certify process-thread quiescence after canonical transport initialization.** |
| Decision | Retain only if the transport round trip is bit exact, PREPARE reaches the preregistered bounded quiescence gate, and an exact real-role replay has zero whole-window and per-generation lifecycle deltas. A timeout or later thread creation remains a release failure; do not suppress the lifecycle counter. |
| Redesign | If supported, replay non-final and stage-zero roles against the same final source before promotion. If falsified, instrument the exact transition that creates the thread and identify its native owner before changing the READY contract again. |

| Field | H014-038i (preregistered after H014-038h numerical failure) |
|---|---|
| Hypothesis | H014-038h eliminated the late thread but invalidated the established READY effect by placing a measured `450.583 ms` CPU-only quiescence interval after all seven CUDA warm calls. Moving the unchanged canonical transport/quiescence work between assigned-stage calls six and seven will keep exactly seven isolated real CUDA calls, make the seventh call the final READY action, restore final-role numerical error below `3e-5` and the prior steady device plateau, while retaining zero whole-window/per-generation lifecycle deltas. |
| Implementation | Retained `persistent/h014-038h-post-warm-quiescence-numerical-failure.json` (SHA-256 `f5d3a96b8aacd5a4ec8b0e2cddc9911961be482e4949c9ab9dc91843a7b24146`). Reordered only the already implemented transport/quiescence block: execute calls 0..5, round-trip call 5's real boundary and reach thread quiescence, then execute call 6 last. Preserved the exact seven-call count, transport data, thread gate, memory gate, records/counter restoration, routes, arithmetic, binary and tolerances. |
| Benchmark | Exact H014-038h final-role configuration. H014-038h produced bit-exact `258,048`-byte transport, stable `47`-thread set for five samples, zero warm lifecycle deltas and exact routes/sampled tokens, but maximum layer/final relative L2 rose to `3.056107e-3`; retained device p50 rose from H014-038g's `4.574240 ms` to `5.398848 ms`. PREPARE output fingerprint remained exactly `sha256:f04993ef...26511`; GPU UUID/free memory/P8 health remained normal. |
| Result | **INVALID BENCHMARK CONFIGURATION; HYPOTHESIS NOT TESTED BY THIS RECEIPT.** H014-038i repeated H014-038h's exact mismatched inputs/outputs and `3.056107e-3` comparison, while lifecycle remained zero. H014-038j subsequently tested the retained ordering against the correct oracle and passed. |
| Inspection | The only new interval between the unchanged seven-call fixture and serving was the thread-quiescence delay. The coincident timing and accuracy regression, with unchanged fixture output and routes, is consistent with re-entering the measured CUDA cold plateau and inconsistent with changed weights or routing. |
| Bottleneck | **READY ordering: one-time CPU lifecycle stabilization must occur before the final assigned-stage warm call, not after it.** |
| Decision | Retain only if one exact replay simultaneously passes numerical, routing, sampling, state, thread/process/task/connection/allocation, PREPARE and CUDA-health gates. Otherwise the cold-plateau explanation is falsified and the final role remains quarantined. |
| Redesign | If supported, replay non-final and stage-zero roles. If falsified, instrument native phase outputs and state after each transition before another READY change. |

| Field | H014-038j (preregistered after oracle-lineage mismatch discovery) |
|---|---|
| Hypothesis | H014-038h/i's `3.056107e-3` comparison failure is entirely an invalid mixed-oracle replay: the commands used the older `oracle-full-93` trace/route/logit family instead of the canonical deterministic-router `oracle-full-93-idot0` family used by H014-038g and the exact-binary graph certificate. Replaying H014-038i source with all three `idot0` inputs will restore H014-038g's three exact input fingerprints, numerical error below `3e-5`, exact routes/sampling/state and zero warm lifecycle deltas. |
| Implementation | No source, binary, arithmetic, READY, tolerance or oracle change. Preserved H014-038h/i receipts as invalid configuration evidence. Changed only the three CLI oracle paths as one lineage-locked set: hidden trace SHA `0a432e25...b1a8d`, routes SHA `c734d864...91288`, and logits SHA `8cd947eb...0248`. |
| Benchmark | Exact worker-092 final-role replay. The H014-038g input fingerprints were `4c234e5c...ab27`, `05bf26ed...7375`, `efea0186...c798`; H014-038h/i instead used `fa963c60...03d`, `1ce904bd...eb08`, `e82ba880...f0c2`. H014-038h and H014-038i produced identical input, layer and final fingerprints despite different PREPARE ordering, proving the prior comparison result followed the selected inputs rather than the ordering change. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** Receipt `persistent/h014-038-regression-final-stage.json` SHA `b3d61ed48aed36b24a4e2468c1b05b5d71a7e332b2eb69ed969f5768b5a5be50`. All three input fingerprints exactly matched H014-038g; maximum layer/final relative L2 was `2.504679e-7`, maximum logits relative L2 `3.180334e-7`; routes and sampled tokens were exact; state was finite/isolated; every whole-window and per-generation lifecycle delta was zero. PREPARE retained seven calls, bit-exact `258,048`-byte transport, five stable thread samples and zero new threads after the final warm call. Post-run GPU UUID/free-memory/P8 health was normal. |
| Inspection | H014-038a's own preregistration names the canonical trace SHA `0a432e25...b1a8d`; the replay command accidentally selected SHA `95012ceb...6ae`. A numerical comparison to a different trace family cannot test H014-038h/i. |
| Bottleneck | **Benchmark lineage selection, not Kimi execution.** The CLI accepted individually valid but scientifically mismatched oracle files without a family manifest. |
| Decision | Treat H014-038h/i numerical verdicts as INVALID, not failed CUDA hypotheses. Retain H014-038i source only if the lineage-correct run passes every gate. |
| Redesign | After the valid replay, bind oracle trace/route/logit hashes as a single manifest in final promotion so future mixed-family commands fail closed. |

| Field | H014-038k (preregistered role-complete READY replay) |
|---|---|
| Hypothesis | The lineage-correct PREPARE transport/quiescence design is role-independent: real layer-1 KDA+MoE, layer-3 Gated-MLA+MoE and stage-zero embedding+dense workers will each retain exactly seven assigned-stage warm calls, bit-exact canonical transport, bounded thread quiescence, correct output/state, zero warm lifecycle creation and healthy CUDA. The two non-final roles will additionally preserve real B8 correctness, pre-CUDA B9 rejection, post-guard safe B1 and bounded allocator retention. |
| Implementation | No further source or binary change. Replay the current production executor with exact scoped worker identities and the canonical `idot0` trace/routes. Run non-final KDA/MLA first; allow stage zero to consume that exact passing receipt, preserving the fail-closed dependency chain. |
| Benchmark | Real checkpoint, final binary SHA `c3bdb40d...ae326`, worker 001/003/000 assignment identities, seven-call PREPARE per role, three retained generations, real non-final B8/B9/B1 fixtures and post-run GPU health. Numerical, routing, lifecycle, memory and batch limits remain unchanged. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** Non-final receipt `persistent/h014-038-regression-nonfinal-stages.json` SHA `4d3abb45bd27fc54aa37e1f0b98fdd4a46686f4967e22d4c4a69d2d04204d14c`; stage-zero receipt SHA `52774b0ae914762a8fb6fecd97d5d4bdeae4a29234a94929db8f409fcbf97b29`. All roles passed exact identity, seven-call PREPARE, canonical transport, thread quiescence, correctness/state and zero lifecycle creation. KDA/MLA maximum errors were `1.619126e-7 / 3.001679e-7`; stage zero was `2.751787e-7` with bit-exact residual rows. Real B8 was `9.846656 / 9.574912 ms` (`812.459 / 835.517 rows/s`), exact routes and selected-once; B9 rejected pre-CUDA and safe B1 passed. |
| Inspection | Each non-final B8 close retained exactly `2,097,152` bytes and the subsequent safe B1 retained zero additional bytes. Stage-zero PREPARE retained `2,097,152` bytes, below the preregistered 4 MiB ceiling; KDA/MLA PREPARE retained zero. All quiescence gates observed five stable samples; no later thread appeared. Post-chain RTX 5090 UUID, `30,248 MiB` free VRAM and P8 state were normal. |
| Bottleneck | **Resolved locally:** READY/lifecycle and batch-contract certification now pass. Physical Linux/sm_86 execution remains an Experiment 015 canary unknown. |
| Decision | **RETAIN current PREPARE ordering, production batch cap 8 and allocator tolerances.** H014-038 may advance to fail-closed promotion and package regeneration. |
| Redesign | Create the oracle-family manifest and exact promotion/integrity receipt, then rebuild all placement/package/report artifacts around the promoted source/binary identity. |

| Field | H014-038l (preregistered final-source lint replay) |
|---|---|
| Hypothesis | Replacing the batch-phase cancellation cleanup's `try/except BaseException: pass` with the semantically identical `contextlib.suppress(BaseException)` will satisfy the repository lint gate without changing successful CUDA execution, error propagation, READY, B8/B9/B1 behavior, numerical output or lifecycle. |
| Implementation | Import `contextlib` and rewrite only the nested best-effort `runtime.profile_end()` cleanup. The original exception is still re-raised; no success-path line, CUDA kernel, binary, gate or tolerance changes. |
| Benchmark | Focused unit/lint suite, then exact H014-038j/k final → non-final → stage-zero dependency replay on the same canonical `idot0` family and final binary hash. Compare core errors, routes, batch guard, PREPARE and lifecycle to H014-038j/k. |
| Result | **PENDING.** |
| Inspection | A mechanical lint change still changes the production Python source and therefore cannot inherit the prior P1 source certificate. Binary component/full-graph evidence remains valid because the DLL is unchanged. |
| Bottleneck | Final source cleanliness, not execution. |
| Decision | Promote only from the post-rewrite P1 receipts. |
| Redesign | On PASS, run promotion preflight and emit the source/oracle/promotion manifests. |

| Field | H014-038m (preregistered after native-thread trigger isolation) |
|---|---|
| Hypothesis | The recurring warm thread is lazy PyTorch CPU-pool expansion used only by boundary tensor copies, not Kimi CUDA. Configuring the persistent Kimi worker's intra-op and inter-op pools to one thread before any executor tensor work will prevent all later pool growth, keep the canonical `258,048`-byte pack/unpack bit exact, change steady transport p50 by no more than `10%`, and preserve all final/KDA/MLA/stage-zero numerical, batch, state and lifecycle gates. |
| Implementation | Added one process-global, lock-protected, idempotent fail-closed CPU transport configuration invoked at executor construction. Required `torch.get_num_threads()==1` and `torch.get_num_interop_threads()==1`; an incompatible previously initialized inter-op pool rejects. Recorded the exact values in executor lifecycle/PREPARE evidence. CUDA kernels, binary, weights, routing and tensor representation were unchanged. |
| Benchmark | Retain a two-process diagnostic comparing default versus configured pools over canonical pack/unpack. Initial observation: default PyTorch reported `20/20` and one exact round trip created `19` OS threads; configured `1/1` created `0` even after the same round trip plus a 64 MiB reduction. Then replay the exact H014-038l P1 chain and cancellation tests. |
| Result | **COMPOUND HYPOTHESIS FALSIFIED.** Diagnostic `persistent/h014-038m-cpu-thread-pool-diagnostic.json` SHA `0c13fa5b...35d6a` passed: default `20/20` created `19` threads and p50 `0.2319 ms`; configured `1/1` created zero threads and p50 `0.2016 ms` (`0.86934x`), bit exact. Fresh final repeat 1 passed (SHA `4f519242...10197`), but repeat 2 failed lifecycle (SHA `42625406...85a9b`): `29 -> 30` OS threads, added TID `32120`, unchanged Python thread IDs, exact `1/1` contract and `2.505e-7` numerical error. |
| Inspection | PREPARE ended with `47` threads, the production warmup later had `66` (`+19`), and retained generation 3 reached `67` (`+1`), while Python thread IDs were unchanged. The `19`-thread increment exactly matches a 20-lane native pool excluding the caller. Five time-only stability samples cannot prove an untriggered lazy pool is initialized. |
| Bottleneck | **Lazy CPU thread-pool policy in transport copies, not CUDA compute or worker/task construction.** |
| Decision | **RETAIN the 1/1 transport contract** because it removes a measured 19-thread pool and improves latency, but do not claim READY containment or promote yet. Retain quiescence only as a secondary check. |
| Redesign | H014-038n captures Windows thread description, start address/module and CPU-time evidence at every lifecycle snapshot to identify the remaining singleton before another READY change. |

| Field | H014-038n (preregistered native-thread ownership trace) |
|---|---|
| Hypothesis | The remaining intermittent `+1` is a persistent native helper outside Python and outside the now-disabled PyTorch pools. Recording each OS thread's description, Win32 start address, mapped module and user/system CPU time at the existing lifecycle snapshots will attribute the added TID to one module and exact generation boundary without changing execution; the owner will determine whether PREPARE must trigger it or whether the observer is counting an external runtime thread. |
| Implementation | Extended `_process_snapshot` on Windows to enrich the already collected thread IDs through read-only `OpenThread`, `GetThreadDescription`, `NtQueryInformationThread(ThreadQuerySetWin32StartAddress)` and `GetMappedFileNameW`. Added per-generation attribution and a one-second/20-sample post-generation observation. Preserved the existing thread-creation gate and execution semantics. |
| Benchmark | Repeat the exact final role in fresh processes until either two consecutive zero-thread runs or one attributed `+1` is retained; compare with H014-038l/m TIDs and per-generation transitions. GPU health and all prior correctness gates remain mandatory. |
| Result | **CHARACTERIZATION PASS.** Traces 1/2 passed (`110e1db3...836d`, `08accbf6...c076`); trace 3 failed as intended for attribution (`7129cdec...fc97a`). The added TID `29264` appeared during retained position 1, persisted through the one-second post window, started at the generic `ntdll.dll` worker thunk, had `0.015625 s` system CPU and no description; no further post-window transition occurred. Numerical error remained `2.505e-7`. |
| Inspection | Source inspection found the missing ownership seam: model load, seven-call PREPARE and every serving execution independently call `asyncio.to_thread`. PREPARE therefore is not guaranteed to use the same host thread as serving, and the elastic default executor may choose another worker. CUDA/native per-host-thread initialization can consequently occur after READY even though the mathematical fixture ran seven times. The old `python_thread_ids` field records `threading.ident`, not Windows native IDs, so it could not correlate TID `29264`. |
| Bottleneck | **PREPARE and serving do not share one worker-owned compute thread.** |
| Decision | Retain the attribution evidence; do not suppress the singleton or increase sleeps. Replace the elastic default executor for stage-owned blocking/CUDA work. |
| Redesign | H014-038o creates one persistent max-one compute executor, initializes it before READY, and sends load, PREPARE, session lifecycle, execute and close through that same native thread. |

| Field | H014-038o (preregistered worker-owned compute thread) |
|---|---|
| Hypothesis | A persistent `ThreadPoolExecutor(max_workers=1)` owned by each `PersistentStageRuntime`, created before model load and used for loader, seven-call PREPARE, session allocation/free, every CUDA execute and close, will make PREPARE exercise the exact serving host thread. It will eliminate default-executor thread growth across at least three fresh final-role processes and all non-final/stage-zero roles, while preserving event-loop task counts, numerical/state correctness, B8/B9/B1 behavior and shutdown. |
| Implementation | Added the named max-one worker-owned compute executor, replaced every stage-runtime `asyncio.to_thread` submission, routed load/PREPARE/session/execute/close through it, shut it down after unload, and recorded native TID, Python ident and submission count. Added per-execution host-thread receipts and a unit fixture spanning loader, PREPARE, session, execution and close. |
| Benchmark | Unit-test same-thread load/PREPARE/execute/close and fail-closed shutdown. Then require three independent final-role passes, KDA/MLA/stage-zero dependency replay, exact OS/Python-native thread evidence and post-run GPU health. |
| Result | **HYPOTHESIS FALSIFIED; PARTIAL MECHANISM RETAINED.** Focused runtime/identity tests passed `27/27`. Fresh final-role receipt `persistent/h014-038o-final-repeat-1.json` SHA `410c3340...7086` preserved exact routing and `2.50468e-7` maximum relative L2 error. PREPARE's seven calls and all three retained generations executed on the same native TID `35072`; `compute_thread_consistent=true`. Nevertheless, one non-Python native TID `30424` appeared at post-generation sample 4 (`200 ms`), persisted, and made warm `thread_creation=1`. GPU UUID, availability and free memory remained healthy. Repeat 2 was correctly not run. |
| Inspection | The same-thread prediction was directly disproven: no CUDA/lifecycle call escaped TID `35072`, yet the late singleton remained. It starts at the generic `ntdll.dll` worker thunk with zero sampled CPU time and appears only after the end-to-end serving path. This narrows the trigger to work still executed on the event-loop thread (tensor unpack/pack, response/token framing or another post-compute native helper), or to an asynchronously created native helper whose trigger is not host-thread ownership. |
| Bottleneck | **Unprepared native work in or asynchronously triggered by the real message transport/response path, not elastic compute-thread selection.** |
| Decision | Retain the worker-owned compute executor because it now mechanically proves load/PREPARE/serving CUDA thread ownership and bounds concurrency. Do not promote and do not weaken the zero-thread gate. |
| Redesign | H014-038p isolates delayed thread creation around the exact canonical pack/unpack path with a one-second observation, then instruments the end-to-end message phases only if standalone transport does not reproduce it. |

| Field | H014-038p (preregistered delayed transport-thread isolation) |
|---|---|
| Hypothesis | With PyTorch CPU pools fixed at `1/1`, the canonical `float32 [1,9,7168]` pack/unpack path still creates the observed singleton asynchronously after the immediate measurement. A one-second, 20-sample post-transport observation in a fresh process will reproduce exactly one persistent non-Python OS thread while preserving bit-exact payloads. |
| Implementation | Added a retained two-process diagnostic with idle control and 250 exact canonical round trips under `torch 1/1`, plus immediate and 20 x 50 ms OS/Python-native snapshots. Production READY was unchanged. |
| Benchmark | Run default and `1/1` subprocesses, retain exact payload/latency/thread transitions, and inspect whether the singleton appears without checkpoint or CUDA execution. |
| Result | **HYPOTHESIS FALSIFIED.** Receipt `persistent/h014-038p-delayed-transport-thread-diagnostic.json` SHA `1ab0e4a0...357c0` passed diagnostic integrity but `hypothesis_reproduced=false`: the idle process added zero threads; the `1/1` transport process added zero immediately and zero during the one-second window after 250 bit-exact `258,048`-byte round trips. |
| Inspection | Canonical boundary serialization alone cannot explain H014-038o's singleton. Adding an event-loop transport warmup would therefore be speculative and was rejected. The trigger requires another part of the real end-to-end path, most plausibly CUDA submission/completion or token-result construction. |
| Bottleneck | The trigger is conditional on the complete real stage path, not standalone boundary pack/unpack. |
| Decision | Retain diagnostic evidence only; make no READY change. |
| Redesign | H014-038q inserts an optional diagnostic observer at exact end-to-end phase boundaries and waits after each phase in one disposable message. |

| Field | H014-038q (preregistered exact serving-phase isolation) |
|---|---|
| Hypothesis | The singleton is asynchronously triggered by real final-stage CUDA execution/completion, not by input unpack or token-result packing. In a disposable exact final-stage message, a 500 ms observation after input unpack will remain stable, the observation after compute will add exactly one persistent non-Python OS thread, and the response-pack observation will add none. |
| Implementation | Added the disabled-by-default observer at the three exact phase boundaries and a diagnostic callback that captured OS/Python-native metadata across 500 ms per phase. Added focused observer ordering coverage. Normal runtime behavior is unchanged when no observer is supplied. |
| Benchmark | Execute one fresh registered final-stage diagnostic against the canonical `idot0` oracle and exact candidate DLL; retain phase transitions, correctness, compute TID, CUDA health and the existing process-wide observation. |
| Result | **HYPOTHESIS FALSIFIED; CHARACTERIZATION PASS.** Receipt `persistent/h014-038q-serving-phase-thread-diagnostic.json` SHA `a5b18630...b67a` passed. The first added thread was TID `31104` during the warmup request's `input_unpacked` observation (`28 -> 29`), before that request's CUDA execution. `compute_complete` and `response_built` added zero; every later phase added zero. The complete benchmark then passed with zero retained warm deltas, exact compute-thread ownership, `2.50468e-7` error and stable GPU UUID. |
| Inspection | The current user-message CUDA execution and token response are exonerated: the helper was already pending before compute began. Standalone unpack cannot reproduce it, so the trigger lies earlier in the exact lifecycle interval: final PREPARE work, route installation, or user-session CUDA allocation. The 500 ms pause itself allowed the helper to become resident before the lifecycle baseline, explaining the otherwise clean benchmark. |
| Bottleneck | READY is released before an asynchronously scheduled pre-compute native helper becomes resident. |
| Decision | Retain the optional observer for diagnostics; do not infer the exact trigger or add an arbitrary delay yet. |
| Redesign | H014-038r observes separately after load/PREPARE, route installation, warmup-session allocation and message construction. |

| Field | H014-038r (preregistered pre-compute lifecycle isolation) |
|---|---|
| Hypothesis | The seventh assigned-stage PREPARE call asynchronously schedules the singleton; a 500 ms observation immediately after `load_stage`/PREPARE will capture it before route installation or user-session allocation, while all later pre-compute lifecycle observations remain stable. |
| Implementation | Added disabled-by-default benchmark observer hooks after the four pre-compute lifecycle boundaries and reused 500 ms native/Python thread attribution. Production READY was unchanged. |
| Benchmark | Run one fresh exact final-stage process, identify the first lifecycle interval with a persistent added TID, verify canonical correctness and GPU health, and compare the thread count with PREPARE's final snapshot. |
| Result | **HYPOTHESIS FALSIFIED AT 500 ms; INTERVAL NARROWED.** Receipt `persistent/h014-038r-readiness-phase-thread-diagnostic.json` SHA `01456990...7cc4` passed. `load_prepare_complete` stayed `28 -> 28` for 500 ms. During the immediately following `route_installed` observation, TID `24264` appeared (`28 -> 29`); session-open and message-build intervals stayed stable. The benchmark passed with zero warm thread delta, exact routing, `2.50468e-7` error and stable GPU identity. |
| Inspection | The helper arrives between approximately 0.5 and 1.0 seconds after PREPARE, but this design cannot distinguish a long PREPARE delay from route installation as the trigger because the route was installed between the two windows. Session allocation and message building are exonerated. |
| Bottleneck | One unresolved timing/trigger ambiguity between delayed final-PREPARE native initialization and route installation. |
| Decision | Do not modify READY or route semantics from this ambiguous interval. |
| Redesign | H014-038s holds the process after PREPARE for 1.5 seconds with 50 ms samples before installing any route. |

| Field | H014-038s (preregistered post-PREPARE delay versus route trigger) |
|---|---|
| Hypothesis | The helper is delayed work scheduled by the seventh real PREPARE execution, not by route installation. With no route installed, a 1.5-second post-PREPARE observation sampled every 50 ms will add exactly one persistent non-Python TID after the first 500 ms; route/session/message observations will then add zero. |
| Implementation | Reused the benchmark lifecycle hook with 30 x 50 ms before route installation and 10 x 50 ms controls after route, session open and message build. Production code was unchanged. |
| Benchmark | One fresh exact final-stage process with retained transition time, module metadata, correctness, lifecycle and GPU health. |
| Result | **POINT-PREDICTION FALSIFIED; TIMING RACE CONFIRMED BY CROSS-RUN INSPECTION.** Receipt `persistent/h014-038s-post-prepare-thread-diagnostic.json` SHA `f14f68f0...552ef` passed with no transition in any interval. Crucially, PREPARE already ended with `29` threads and every later snapshot remained `29`; correctness was `2.50468e-7`, lifecycle was zero and GPU health was stable. H014-038q/r ended PREPARE with `28` and observed the same-shaped singleton later. |
| Inspection | The helper's initialization time is nondeterministic: sometimes it becomes resident during the existing PREPARE work, sometimes only 0.5-1.0 seconds later. A route-specific hypothesis is unsupported because this run had the helper before route installation. The current five-sample quiescence is also ordered before call 7, so it cannot bound work scheduled by the final warm call. |
| Bottleneck | **A PREPARE ordering/window race:** thread stability is declared before the final assigned-stage call and after too short a stable window. |
| Decision | Retain the combined q/r/s evidence. Replace the ordering/window, not the seven-call fixture or zero-thread warm gate. |
| Redesign | H014-038t moves quiescence after call 7, requires a 1.5-second observation floor and 20 stable 50 ms samples, then demands three independent final-role passes. |

| Field | H014-038t (preregistered post-final-call READY quiescence) |
|---|---|
| Hypothesis | Running the unchanged seven real assigned-stage calls first, then observing OS threads for at least 1.5 seconds and requiring 20 consecutive stable 50 ms samples will absorb the measured nondeterministic native-helper initialization before READY. Three independent exact final-role processes will then show one compute TID, zero per-generation and delayed-window thread deltas, unchanged `<=3e-5` numerical fidelity, and stable CUDA health. |
| Implementation | Moved quiescence from call 6 to after call 7 and required a 1.5-second floor, 20 stable samples and 4-second timeout. Preserved seven calls, transport, CPU `1/1`, binary and correctness gates. |
| Benchmark | Focused unit tests, then three fresh exact final-role processes. Inspect PREPARE transitions, host compute TID, one-second post-generation observation, memory recovery, CUDA error state and `nvidia-smi` after each. Only then replay KDA/MLA/stage-zero. |
| Result | **MECHANISM PROVEN; BOUND REJECTED AFTER TWO PASSES.** Trial 1 SHA `6c1a9130...92fd6` caught TID `32232` at `924.782 ms`, held READY to `2,596.763 ms`, then passed with zero warm/delayed thread growth and `2.50468e-7` error. Trial 2 SHA `76023897...e61a4` caught TID `31436` at `1,666.196 ms`, held READY to `3,487.982 ms`, and likewise passed. GPU health stayed stable. Trial 3 was intentionally not run. |
| Inspection | Moving the observer after call 7 correctly catches the helper and makes serving clean. However trial 2's arrival exceeded the 1.5-second floor. It was caught only because 20 samples took longer than their nominal 1 second on this host; relying on observer overhead would make the safety bound machine-dependent. |
| Bottleneck | The minimum observation bound is too short relative to the measured `1,666.196 ms` maximum arrival. |
| Decision | Retain the ordering and stable-sample logic; reject the 1.5-second floor and stop before trial 3. |
| Redesign | H014-038u uses a 3.5-second floor (just over 2x the measured maximum) and an 8-second fail-closed timeout, then restarts the three-process proof. |

| Field | H014-038u (preregistered measured-margin READY bound) |
|---|---|
| Hypothesis | A post-call-7 observation floor of 3.5 seconds, more than twice the measured maximum helper arrival (`1,666.196 ms`), plus 20 stable samples and an 8-second fail-closed timeout, will contain the helper independently of sampler overhead. Three fresh final-role processes will pass all numerical, compute-thread, delayed-window, memory and GPU-health gates. |
| Implementation | Changed only the quiescence floor to 3.5 seconds and timeout to 8 seconds and updated exact gates. Preserved call count, kernels, binary, transport and tolerances. |
| Benchmark | Three independent registered final-stage processes with inspection and `nvidia-smi` after each, followed by exact KDA/MLA/stage-zero replay. |
| Result | **PASS.** Three independent final-role receipts passed: repeat 1 SHA `9ccb3016...1f739`, repeat 2 SHA `9166991b...d1823`, canonical SHA `2e4d146c...f6c2e`. The helper arrived at `1,586.738`, `1,696.946`, and `1,797.046 ms`; READY released at `3,590.522`, `3,555.395`, and `3,666.667 ms`. Every process had one exact compute TID, zero retained and delayed-window thread growth, exact routes and `2.50468e-7` maximum error. KDA/MLA replay SHA `ca7475c1...5473f` passed, including real B8, pre-CUDA B9 rejection, post-guard B1, stable allocator plateau, same compute TID and zero lifecycle deltas. Stage-zero SHA `9b995943...c83a` passed with `2.75179e-7` error. GPU UUID/free memory remained stable after every increment. |
| Inspection | The measured-margin window catches the same singleton deterministically before READY without depending on sampler overhead. The maximum observed arrival was `1,797.046 ms`, leaving `1,702.954 ms` margin to the floor. No later worker, task, connection, weight, materialization, buffer or thread lifecycle change occurred. |
| Bottleneck | The late native helper is now a bounded one-time READY cost; steady execution remains resident GPU compute. |
| Decision | **RETAIN.** H014-038u closes the intermittent P1 lifecycle blocker and supplies the exact final P1 source receipts for promotion. |
| Redesign | Run exact-binary promotion preflight, full source regressions and package regeneration; any resulting source change requires targeted replay. |

| Field | H014-038v (preregistered production-boundary extraction) |
|---|---|
| Hypothesis | The four full-suite architecture failures are static dependency violations, not CUDA or numerical defects. Mechanically extracting the already-tested CUDA ABI/runtime and graph-loader core from Experiment 014 into production execution modules, leaving experiment benchmark wrappers, and routing tokenizer policy through adapter-owned optional hooks will make all architecture-boundary tests pass without changing binary calls, tensor bytes, weights, routing, arithmetic or lifecycle. |
| Implementation | Mechanically extracted the CUDA ABI/runtime into `execution/kimi_cuda_runtime.py` and the graph loader/runner into `execution/kimi_k3_graph_runtime.py`; converted the Experiment 014 modules to benchmark wrappers; removed production-to-experiment imports. Replaced the generic runtime's architecture-name branch with optional adapter `load_tokenizer`/`encode_prompt` hooks implemented by the K3 adapter. Moved `MXFP4Tensor` to the production model package and fixed only reported first-party lint. |
| Benchmark | Hash/AST-check the extracted source relationship, run architecture and tokenizer/runtime unit tests, repository-wide Ruff and full pytest. Because production source identity changes, replay the exact final/KDA/MLA/stage-zero P1 chain after static gates pass; CUDA component/full-graph DLL evidence remains valid because native source/binary are unchanged. |
| Result | **STATIC BOUNDARY PASS; COMPATIBILITY IMPLEMENTATION FALSIFIED.** The four former architecture tests and focused tokenizer/runtime tests passed (`19 passed`); first-party Ruff passed with zero findings. The first full-suite attempt then failed during collection because the wrapper did not re-export `_quantize_bf16_grouped_int4` and `_CheckpointReader`. |
| Inspection | Python wildcard/selective imports do not preserve underscore-prefixed legacy symbols automatically. Production ownership is clean, but two Experiment 014 consumers proved the wrapper compatibility surface incomplete before any behavioral or CUDA test ran. |
| Bottleneck | Experiment-wrapper API compatibility, not production dependency direction, CUDA, numerical execution or lifecycle. |
| Decision | **RETAIN the production extraction and adapter hooks; MODIFY the wrapper exports.** Promotion remains blocked until a complete static/behavioral gate and P1 replay pass. |
| Redesign | H014-038w explicitly re-exports only the inventoried legacy private names from the production graph module, then restarts the complete suite. |

| Field | H014-038w (preregistered wrapper compatibility repair) |
|---|---|
| Hypothesis | The H014-038v collection failures are caused solely by an incomplete Experiment 014 compatibility wrapper: explicitly importing the five inventoried private graph symbols (`_CheckpointReader`, `_LayerResources`, `_parse_oracle_routes`, `_pointer_offset`, `_quantize_bf16_grouped_int4`) will restore every current legacy consumer while leaving production imports, CUDA calls, tensor bytes and runtime behavior unchanged. |
| Implementation | Added the four missing names to the existing explicit import from `execution.kimi_k3_graph_runtime`; `_parse_oracle_routes` was already present. Declared the compatibility surface in `__all__` so the re-exports are intentional and lint-visible. Added no duplicate implementation and changed no production code. |
| Benchmark | Re-run the two collection-failing test modules first, repeat first-party Ruff, then restart the complete pytest suite with a retained JUnit artifact. On static/behavioral PASS, replay final/KDA/MLA/stage-zero P1 against the exact extracted source. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** The two formerly collection-failing modules passed `13/13`; first-party Ruff passed with zero findings; the complete repository suite passed `1,089`, skipped `13` explicitly physical/configuration-gated tests, and had zero failures in `175.10 s`. Retained JUnit: `artifacts/experiment-014/h014-038w-full-pytest.xml`. |
| Inspection | Inventory across Experiment 014 and tests found seven imports from the wrapper: the two public runner names plus exactly the five private names listed in the hypothesis. Explicitly exporting that exact set restored compatibility while architecture-boundary tests continued to pass. The full-repository Ruff command still reports only pre-existing findings inside the separately versioned `third_party/colibri` submodule; `ruff check src scripts tests` is clean. |
| Bottleneck | **Resolved:** explicit compatibility export coverage. |
| Decision | **RETAIN.** Static dependencies, wrapper compatibility and all software regressions pass. Promotion remains blocked only on the preregistered exact-source P1 runtime replay. |
| Redesign | Replay final, KDA/MLA and stage-zero P1 receipts against H014-038w source, then run fail-closed promotion preflight. |

| Field | H014-038x (preregistered extraction performance regression) |
|---|---|
| Hypothesis | H014-038w's three-sample layer-1 KDA timing (`15.089 ms` p50; B8 `47.793 ms`) is a transient short-window/clock-state observation rather than an extraction regression. The existing standardized resident-stage profile with 20 warmups and 100 retained iterations on the exact H014-038w source will recover layer-1 KDA and layer-3 Gated-MLA p50 to within 25% of their established `3.013 / 2.731 ms` baselines (`<=3.766 / <=3.414 ms`) while preserving exact routing and numerical gates. |
| Implementation | **None.** Do not change code or clocks. Run the existing production profile unchanged and retain the result separately from the P1 correctness receipt. |
| Benchmark | RTX 5090, exact candidate SHA `c3bdb40d...ae326`, real checkpoint, canonical `idot0` trace/routes, registered layer-1/layer-3 identities, 20 warmups and 100 retained iterations per stage. Inspect device p50/p95/p99, wall time, correctness, routes, lifecycle and post-run GPU health. |
| Result | **INVALID BENCHMARK CONFIGURATION; HYPOTHESIS NOT TESTED.** The H014-038w P1 chain itself passed: final receipt SHA `7645a1fb...c990a`, maximum error `2.504679e-7`; non-final SHA `87315dfe...bbfab`, errors `1.619126e-7 / 3.001679e-7`, exact B8 routes, pre-CUDA B9 rejection, safe B1 and zero post-safe allocator growth; stage-zero SHA `ce05797d...36f23`, error `2.751787e-7`, bit-exact residual rows and zero warm lifecycle creation. The profile command failed before stage load/CUDA because its legacy interface accepts one model-identity file, not the scoped identity directory supplied. |
| Inspection | The anomalous KDA receipt has only three retained samples after a 3.5-second READY observation; MLA in the same process family remained close to baseline, and all mathematical/lifecycle checks passed. The failed profile produced no timing datum and cannot decide performance neutrality. |
| Bottleneck | Benchmark identity configuration, with KDA timing variance still unresolved. |
| Decision | Retain no result from the failed command. Do not change production or profiling code merely to rerun a diagnostic. |
| Redesign | H014-038y supplies the profiler's retained immutable model-wide identity while leaving the already-certified scoped P1 receipts as the assignment-identity proof. |

| Field | H014-038y (preregistered valid-identity performance replay) |
|---|---|
| Hypothesis | The prior H014-038x command failed solely because the profiler was given a directory. Supplying the retained immutable model-wide identity `persistent/h014-026b-model-identity.json` will pass its existing identity contract and execute the unchanged standardized profile; if H014-038w is performance-neutral, KDA/MLA production-mode device p50 will be `<=3.766 / <=3.414 ms` with exact routes, error `<=3e-5`, zero lifecycle deltas and healthy CUDA. |
| Implementation | **None.** Change only the profiler CLI identity path from a directory to the exact previously certified model-wide identity file. The source, binary, weights, routes, warmup and iterations remain fixed. |
| Benchmark | Same H014-038x 20-warmup/100-retained profile. Treat scoped worker identity as independently proven by the immediately preceding P1 receipts; this run tests timing only. |
| Result | **HYPOTHESIS FALSIFIED BEFORE CUDA.** The file was readable and matched the checkpoint, but the hardened production runtime rejected it because it has no assignment SHA-256. No stage loaded and no timing datum was produced. |
| Inspection | H014-027a predates assignment-scoped identity enforcement. Reusing its model-wide identity would weaken the current fail-closed contract and is correctly forbidden. |
| Bottleneck | The performance harness has not been updated to resolve its two profiled workers' separate assignment identities. |
| Decision | Do not weaken runtime identity validation and do not treat either failed command as a performance result. |
| Redesign | H014-038z minimally adapts the profiler to the existing scoped resolver and records the resolved identity in each layer result. |

| Field | H014-038z (preregistered scoped profiler identity) |
|---|---|
| Hypothesis | The resident profiler's only incompatibility with the hardened runtime is its stale one-file identity handling. Resolving `k3-worker-001` and `k3-worker-003` from the supplied directory with the already-tested P1 resolver, setting each runtime worker ID to the resolved owner and recording its assignment hash will allow the unchanged performance workload to execute. The timing hypothesis remains H014-038y's `<=3.766 / <=3.414 ms` production p50 gates. |
| Implementation | Import and call `_resolve_worker_identity_manifest` at the start of `_profile_layer`; use the returned file and worker ID in `PersistentStageRuntime`; include identity path/worker/assignment/manifest hashes in the result. Change no execution, warmup, timing, CUDA or numerical code. |
| Benchmark | Run existing scoped-resolver unit tests and first-party Ruff, then the exact H014-038x/y 20/100 profile with the identity directory. Inspect numerical, route, lifecycle, p50/p95/p99 and GPU health. |
| Result | **IDENTITY/TIMING PREDICTION SUPPORTED; COMPOUND LEGACY PROFILE STATUS FAIL.** Scoped-resolver tests passed `10/10`; first-party Ruff passed. Receipt `performance/h014-038z-extraction-stage-profile.json` SHA `cfa57e0a...fbffc` loaded exact `k3-worker-001/003` assignment identities. Production-mode KDA p50/p95/p99 was `2.9753 / 3.0613 / 3.1196 ms`; MLA was `2.7035 / 2.7740 / 2.8706 ms`, both inside preregistered gates. Errors were `1.619126e-7 / 3.001679e-7`, routes exact and production lifecycle deltas zero. The artifact's overall status remained `FAIL` because its older independent expert-dominance hypothesis was false and layer-1's first `minimal` mode observed one thread. |
| Inspection | The production timings reproduce H014-027a (`3.0126 / 2.7308 ms`) within `-1.24% / -1.00%`, falsifying a source-path performance regression. Detailed attribution measured routed+shared experts at `48.115%`, dense at `30.788%`, router device work at `13.135%` and unattributed device work at about `7.96%`; the legacy `>70%` expert-share prediction is correctly false. The one thread occurred only in the first minimal research mode while its `nvidia-smi dmon` sampler was active; the subsequent production mode was zero and the exact P1 chain independently passed zero warm/delayed lifecycle creation. |
| Bottleneck | **Resolved extraction question:** no KDA/MLA performance regression. The complete layer remains mixed expert+dense+router service, not an expert-only bottleneck. |
| Decision | **RETAIN scoped profiler identity wiring and H014-038w extraction.** Retain the profile's overall FAIL as evidence against its old compound expert-dominance hypothesis; use its passing production mode for this narrowly preregistered regression question. Do not weaken lifecycle gates or optimize arithmetic. |
| Redesign | Run fail-closed promotion preflight over the exact binary, oracle family, P0 graph/component evidence, H014-038u repeat evidence and H014-038w canonical P1 receipts. |

| Field | H014-038aa (preregistered exact-binary promotion) |
|---|---|
| Hypothesis | The fail-closed promotion script will accept only the exact candidate SHA `c3bdb40d...ae326`, canonical `idot0` family, 12 passing component receipts, 93-layer graph receipt, three H014-038u READY processes and the current H014-038w P1 chain; its atomic promotion will preserve the binary SHA and regenerate an 11/11 CUDA operation matrix plus sm_86 SASS/compute_86 PTX certificate. |
| Implementation | Include the resident performance harness and the final package CLI, runtime-qualification and admission paths in the exact source manifest. Make no binary/source-runtime change. Re-run `--check`, then execute the same script without `--check` to atomically copy the validated candidate to its final H014-038 path and regenerate promotion, operation-matrix, sm_86 and integrity receipts. |
| Benchmark | Preflight must report `READY_TO_PROMOTE`, production batch 8, first rejected batch 9, 12 component receipts, 93 layers, 276 router calls and 4,416 expert calls. After promotion, independently hash the final binary, inspect matrix/certificate gates and query GPU health. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** Final source-freeze preflight reported `READY_TO_PROMOTE`; promotion receipt SHA `5a7e9c5e...760d` passed. Candidate and final binary are both `792,576` bytes with SHA `c3bdb40d...ae326`. Regenerated operation matrix SHA `cd9b5bd8...00d1` is PASS with `11/11` CUDA-ready classes. sm_86 certificate SHA `4accd07e...bdc9` is PASS with 14 required-symbol contracts each carrying sm_86 cubin SASS and compute_86 PTX. Source-manifest SHA is `58bfbf27...df24` and source-bundle SHA is `2db68148...1656`; post-copy GPU UUID/free VRAM/P8 matched pre-copy. |
| Inspection | The independent final-file hash exactly equals the candidate hash. Promotion bound 12 component receipts, all 93 layers, 276 router calls, 4,416 routed-expert calls, production batch 8 and first rejection at 9. Physical sm_86 execution remains explicitly false/deferred to the Experiment 015 single-3090 canary; no artifact claims otherwise. |
| Bottleneck | **Resolved locally:** exact evidence/source/package binding. Physical RTX 3090 execution is the preregistered Experiment 015 canary unknown. |
| Decision | **RETAIN AND PROMOTE FOR PRE-CANARY USE.** The quarantined candidate name must no longer be used by the rebuilt deployment package. |
| Redesign | Rebuild Experiment 015 deployment artifacts and final Experiment 014 reports around `coli_cuda-sm86-h014-038-final.dll`, then run package smoke, integrity and full regression gates. |

| Field | H014-038ab (preregistered promoted-placement convergence) |
|---|---|
| Hypothesis | Rebuilding the 93-worker placement with the promoted binary and promotion receipt will preserve exact tensor ownership, memory feasibility and topology while replacing every stale `CANDIDATE_PENDING_FINAL_SAME_BINARY_REGRESSION` marker with a fail-closed `PROMOTED_FOR_PRE_CANARY_USE` identity. |
| Implementation | Add a required promotion receipt to `final-placement`; require PASS, `PROMOTED_FOR_PRE_CANARY_USE`, matching final-binary SHA, sm_86 SASS and compute_86 PTX. Record it in source artifacts, set per-worker pending flags false/promoted flags true, and set the runtime status to promoted. Change no placement arithmetic or ownership. |
| Benchmark | Build from the immutable 96-worker source placement, current capacity receipt, current H014-038w stage-zero/final P1 receipts, retained layer-89/layer-91 B8 receipts, final DLL and H014-038 promotion receipt. Require 93 workers, exact tensor/byte coverage, no duplicates/orphans, all memory gates, unchanged topology class B and zero candidate markers. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** Placement SHA `d0dc1305...3201` is PASS with 93 promoted workers, zero pending workers, exact `497,052` tensors and `1,559,965,606,912` source bytes, no duplicates/orphans and all ten acceptance gates true. Canonical ownership remains SHA `d63e0974...c1e8`; maximum planned VRAM is `24,276,277,658` bytes, minimum total headroom is `4,070,506,496` bytes and minimum unallocated after safety is `1,493,526,118` bytes. |
| Inspection | Ownership and topology remained unchanged while runtime provenance moved to the final DLL SHA `c3bdb40d...ae326` and promotion receipt SHA `5a7e9c5e...760d`. The hash-bound execution plan SHA `f34a2993...3f0e` retained 100 operations and 97 transitions; distribution SHA `4e6d9d84...5c67` retained 93 worker-specific assignments. Rehearsal SHA `45e06dd1...ed4e` passed three generations with 93 READY workers, 576 events, 93 state transitions and zero errors. Identity fixture SHA `902eaba2...6192` and canary fixture manifest SHA `daed0aa8...10a7` passed; compact NPZ SHA remained `e06008bd...6a109`. |
| Bottleneck | **Resolved:** stale deployment provenance. The remaining local blocker is package closure against this promoted chain. |
| Decision | **RETAIN.** Promotion and placement are mechanically linked; all downstream manifests were regenerated because their placement-hash contracts required it. |
| Redesign | H014-038ac builds and attacks a fresh package from this exact chain before installing it at the canonical Experiment 015 path. |

| Field | H014-038ac (preregistered final package convergence) |
|---|---|
| Hypothesis | A package rebuilt from the promoted placement, regenerated dependent manifests/canary fixtures and a wheel of current source will pass all positive and tamper-control gates while reporting local Experiment 014 certification `PASS`, physical 3090 canary `NOT_RUN`, fleet activation false and no stale H014-027w/candidate evidence. |
| Implementation | Require promoted placement/runtime state in package build and validation. Replace old H014-027w graph/matrix/sm_86/batch evidence with H014-038 graph, canonical matrix/certificate, promotion/source manifests and current final/non-final/stage-zero P1 receipts. Change release local certification from `PENDING_FINAL_BINARY_REGRESSION` to `PASS`; retain `READY_FOR_SINGLE_3090_CANARY`, `NOT_RUN` physical canary and `fleet_activation_allowed=false`. |
| Benchmark | Build a fresh wheel; assemble into a new staging directory; validate package lock, 93 requirements, portable manifests, native source/build contract, canary fixture, scripts, network/cost policies and release state; run bounded tamper controls and relevant unit tests. Promote the staging directory to `artifacts/experiment-015-deployment/` only after PASS. |
| Result | **DIRECTORY/CONTROL PASS; CANONICAL ARCHIVE PROMOTION ASSUMPTION FALSIFIED.** Fresh wheel SHA `93ef478f...be48c`; staged package independently passed all 25 positive gates with 151 locked files and 93 exact requirements. All 16 tamper controls passed in `deployment/h014-038ac-deployment-package-controls.json`; focused deployment regressions passed `26/26`. The canonical installed directory then passed the same 25 gates, but its moved ZIP retained the staging root name and failed canonical archive validation with `release archive file set differs`; baseline receipt `deployment/h014-038ad-canonical-archive-baseline.json`. |
| Inspection | Package content and fail-closed state are correct: local certification PASS, physical canary NOT RUN and fleet activation false. The builder makes the archive root name part of its path contract, so moving a staging-built ZIP beside a renamed directory is not identity-preserving even when all member bytes are unchanged. |
| Bottleneck | **Archive container naming, not package contents, CUDA, placement or deployment logic.** |
| Decision | **MODIFY only archive installation.** Retain the passing directory and all controls; do not claim final package convergence until an archive built from the canonical basename passes. |
| Redesign | H014-038ad preserves the invalid moved ZIP, rebuilds only the deterministic archive from the unchanged canonical directory and repeats archive plus tamper validation. |

| Field | H014-038ad (preregistered canonical archive repair) |
|---|---|
| Hypothesis | The sole canonical-package failure is the staged basename embedded in the otherwise valid ZIP. Moving the invalid ZIP aside and calling the existing deterministic archive builder on the unchanged canonical directory will preserve every package-lock/member byte, embed root `experiment-015-deployment/`, pass `_validate_release_archive`, and leave all 25 positive plus 16 negative controls passing. |
| Implementation | Make no package-directory, source, wheel, placement, runtime, policy or script change. Preserve the invalid moved ZIP as `experiment-015-deployment-h014-038-staging-prefix.zip`; generate a new canonical ZIP using the existing archive builder. |
| Benchmark | Baseline receipt must be `EXPECTED_FAIL` with `release archive file set differs`. Hash the package lock before/after, validate the rebuilt archive member set and bytes, rerun canonical package validation and the complete bounded tamper suite. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** Baseline `deployment/h014-038ad-canonical-archive-baseline.json` is `EXPECTED_FAIL` with `release archive file set differs`. Rebuilding only the archive left package-lock SHA `560e0fe6...2a92` unchanged. Canonical ZIP SHA `e04ee6a...b655` passed all `151/151` member paths and hashes. Canonical control receipt SHA `7ce914cb...909d` is PASS with all 25 positive and all 16 negative gates true. |
| Inspection | `RELEASE.json` reports local certification `PASS`, physical RTX 3090 canary `NOT_RUN`, fleet activation false, topology `WHOLE-LAYER`, 93 workers, promotion receipt SHA `5a7e9c5e...760d` and Windows pre-canary binary SHA `c3bdb40d...ae326`. The invalid staging-prefix ZIP is preserved as `artifacts/experiment-015-deployment-h014-038-staging-prefix.zip`; the prior package and archive are recoverable at the H014-037 backup paths. |
| Bottleneck | **Resolved:** staging basename coupling in the archive container. No package member, release wheel or deployment contract changed. |
| Decision | **RETAIN. H014-038ac/ad package convergence PASS.** The canonical directory and archive are the Experiment 015 handoff; full-fleet activation remains correctly locked behind the physical canary certificate. |
| Redesign | Regenerate final Experiment 014 reports, acceptance gates, machine summary, evidence integrity, charts and ledger summary, then execute final full regressions and integrity validation. |

| Field | H014-SUB-011 (preregistered promoted-binary expert regression) |
|---|---|
| Hypothesis | The promoted binary SHA `c3bdb40d...ae326` preserves the measured best four-worker layer-89 expert partition: exact real top-16 ownership/execution/reduction, strict complete-layer equivalence, every worker below one full-layer footprint, critical-worker device p50 `<=0.75 ms` and complete-layer throughput at least `75%` of its same-binary resident reference. |
| Implementation | **None.** Re-run the existing four-worker real expert collective against the final binary and final H014-038 graph receipt. Do not alter ownership, kernels, warmup, retained calls, weights, routes or numerical gates. |
| Benchmark | Real layer 89, modulo-4 disjoint ownership, 224 experts per persistent worker, 10 warmup and 30 retained calls, canonical `idot0` trace/routes, final graph receipt and post-run CUDA/VRAM/`nvidia-smi` health. Stop before batching if this gate fails. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** Receipt `sub-layer/h014-sub-011-promoted-four-worker-real-expert.json` SHA `e5cdac82...efc4` binds final binary SHA `c3bdb40d...ae326`. Selected IDs/weights/ownership and all 16 executions were exact; complete-layer/state error was zero. Every worker held `3,931,060,224` tracked bytes versus `18,270,388,224` complete-layer resident bytes. Critical-worker device p50 was `0.4385 ms`; distributed layer wall p50/p95/p99 was `4.1888 / 4.6593 / 5.0026 ms`, `238.732 layers/s`, and `78.809%` of the same-binary resident reference. |
| Inspection | Mean total transport was `289,550.6` bytes/token, maximum critical-path payload `100,352` bytes, eight root/eight worker messages and two synchronization points. The promoted path slightly improved absolute p50 versus H014-SUB-002/005 while relative throughput remained in the same characterized band. Post-run CUDA and `nvidia-smi` were healthy. |
| Bottleneck | **Parent reduction/residual/shared-expert plus coordination, not critical-worker expert CUDA.** Ideal expert service was `0.2967 ms`, while reduction/residual/other contributed `2.4329 ms` at p50 and exposed loopback/coordination `0.4201 ms`. |
| Decision | **RETAIN promoted-binary logical four-worker proof.** This remains one-GPU logical isolation, not physical multi-GPU efficiency. |
| Redesign | H014-SUB-012 incrementally replays B1/2/4/8 on the same promoted binary. |

| Field | H014-SUB-012 (preregistered promoted-binary incremental batch regression) |
|---|---|
| Hypothesis | After H014-SUB-011 passes, the promoted binary preserves exact four-worker batches 1, 2, 4 and 8 in order, with a known-safe fixture and GPU-health check after each size; B8 retains at least `80%` of same-binary resident B8 throughput, uses at most one worker message and `286,720` tensor payload bytes per row, and has wall p99 no worse than the retained `22.290927 ms` gate. |
| Implementation | **None.** Re-run the retained single-interval worker protocol and parent-owned shared experts with the final binary. Do not attempt B16 or any blind sweep. |
| Benchmark | Real layer 89, four persistent modulo-owned workers, batches `1 -> 2 -> 4 -> 8`, five warmup and 20 retained calls per size, exact rows/routes/state, per-size synchronize/error/safe-fixture/free-VRAM/worker-health checks and final `nvidia-smi`. |
| Result | **SAFE INCREMENTAL BASE PASS; OPTIMIZED-PROTOCOL CONFIGURATION PREDICTION NOT TESTED.** Receipt `sub-layer/h014-sub-012-promoted-four-worker-batch.json` SHA `f5ef0e32...1cc5` passed exact B1/2/4/8, zero numerical/state error, selected-once, per-size safe fixture/CUDA health and all workers below one layer. B8 wall p50/p95/p99 was `18.351 / 19.340 / 19.345 ms`, `435.945` rows/s, `2.294 ms/row` and `80.949%` resident retention. However cycle ID `H014-SUB-012` selected schema v1, so its messaging gates differ from the preregistered retained v2 protocol. |
| Inspection | The promoted binary safely executes every intended size and already satisfies the preregistered B8 p99 and throughput-retention thresholds. Source inspection showed buffered/single-interval selection is intentionally keyed to a cycle ID ending in `006D/006E/006F`; no CLI flag was omitted and no CUDA issue occurred. |
| Bottleneck | **Benchmark configuration identity:** schema v1 rather than the retained v2 single-interval worker protocol. |
| Decision | **RETAIN as final-binary safe incremental evidence, but do not use it to certify v2 messages/payload.** |
| Redesign | H014-SUB-012a-006F repeats the identical safe sequence with the required `006F` suffix so the existing optimized v2 path and its original gates execute. |

| Field | H014-SUB-012a-006F (preregistered promoted-binary single-interval replay) |
|---|---|
| Hypothesis | Selecting the existing buffered single-interval v2 protocol on the promoted binary will preserve exact B1/2/4/8 and all per-size health gates; B8 will retain at least `80%` of same-binary resident throughput, use at most one message and `286,720` tensor payload bytes per row, improve capacity at least `1.20x` versus H014-SUB-006c and keep wall p99 `<=22.290927 ms`. |
| Implementation | **None.** Change only the evidence cycle ID to the required `006F` suffix; all checkpoint, binary, graph, layer, ownership, warmup, iterations and safety checks remain identical to H014-SUB-012. |
| Benchmark | Real layer 89, promoted binary, four persistent 224-expert partitions, parent-owned shared experts, B1 -> B2 -> B4 -> B8, five warmup/20 retained calls, exact state/output and safe CUDA/VRAM/worker checks after every size. |
| Result | **EXECUTION/CORRECTNESS PASS; PERFORMANCE HYPOTHESIS FALSIFIED.** Receipt `sub-layer/h014-sub-012a-006f-promoted-single-interval-batch.json` SHA `db68cf34...448a` is schema v2 on final binary SHA `c3bdb40d...ae326`. B1/2/4/8 all passed exact output/state, selected-once, lifecycle and post-size health. B8 wall p50/p95/p99 was `17.714 / 18.577 / 18.910 ms`, aggregate wall throughput `451.620 rows/s`, per-row service `2.214 ms`, resident retention `84.153%`, one message/row and `286,720` tensor payload bytes/row. Four of five performance gates passed; capacity gain versus fixed H014-SUB-006c was `1.18755x`, below the preregistered `1.20x`. |
| Inspection | The promoted binary is safe and the v2 protocol preserves the economically relevant `>=80%` resident capacity plus message/payload limits. Its p50 is `1.75%` slower than retained H014-SUB-006f (`17.4095 ms`), enough to miss the comparison threshold by `1.04%`; p99 improved and GPU health remained normal. No correctness, CUDA, memory or transport degradation occurred. |
| Bottleneck | **Complete-layer parent/coordination overhead and run-to-run service variance; not expert correctness or unsafe CUDA.** The critical architectural result is unchanged: fine fanout retains useful local capacity but loses to one resident whole layer and incurs stricter network/worker economics. |
| Decision | **RETAIN the exact final-binary measurement; reject the `>=1.20x` claim.** Use `84.153%` as final promoted-binary B8 capacity retention. Keep sub-layer workers out of the initial Experiment 015 fleet. |
| Redesign | Continue to the already-preregistered promoted-binary coarse TCP regression; no further sub-layer optimization is justified before a physical low-latency multi-GPU canary is economically warranted. |

| Field | H014-038ae (preregistered promoted-binary coarse TCP regression) |
|---|---|
| Hypothesis | The final binary and extracted production stage source preserve the real stage-zero -> layer-1 KDA+MoE persistent TCP slice: bit-exact `float32 [1,9,7168]` boundary, exact routes/state, graph-relative error `<=1e-6`, exposed loopback transport p50 `<1.5 ms`, zero warm lifecycle creation and healthy CUDA in both worker processes. |
| Implementation | **None.** Re-run the existing two-process persistent TCP fixture with the final binary and final H014-038 graph receipt; retain FP32 and the existing protocol. |
| Benchmark | Real checkpoint, simultaneous stage-zero/layer-1 residency, 10 warmup and 50 retained calls, canonical trace/routes, `258,048` payload bytes, exact final binary SHA, worker shutdown health and `nvidia-smi`. |
| Result | **PASS; HYPOTHESIS SUPPORTED.** Receipt `coarse/h014-038ae-promoted-stage0-stage1-tcp.json` SHA `4b8806f3...83c3` binds final binary SHA `c3bdb40d...ae326` and final graph SHA `c4a5836a...cb57`. The FP32 boundary was bit exact on all 50 retained calls, routes/state were exact, maximum error was `2.752149e-7`, lifecycle deltas were all zero and both workers synchronized/shut down healthy. Stage-zero/layer-1 device p50 was `2.1668 / 3.0269 ms`; exposed transport p50/p95/p99 was `0.6102 / 1.2023 / 1.3313 ms`; end-to-end p50 was `6.8146 ms`. |
| Inspection | Both processes were simultaneously resident at `3,040,870,400 + 18,270,388,224` bytes. The critical payload remains `258,048` bytes and production-direction frame `258,283` wire bytes. Exposed transport is `0.20045 ms` above H014-032c's earlier p50 but safely below the preregistered gate; stage compute remains dominant. |
| Bottleneck | **Stage compute, with a measurable but bounded loopback/serialization contribution.** No final-binary or extracted-runtime regression affects correctness or lifecycle. |
| Decision | **RETAIN exact promoted-binary coarse proof.** Network thresholds must be replayed from its measured `0.6102 ms` rather than the predecessor receipt. |
| Redesign | H014-038ag reruns the deterministic coarse network matrix and retains the 5 ms/5 Gbps edge class only if it still provides at least 90% capacity. |

| Field | H014-038ag (preregistered promoted-binary coarse network replay) |
|---|---|
| Hypothesis | Replaying the exact promoted-binary coarse payload/base latency will keep the tested `5 ms / 5 Gbps` edge inside the `>=90%` capacity region and remain materially looser than the retained fine-microwork `0.5 ms / 2.5 Gbps` class, so topology and rental admission do not change. |
| Implementation | **None.** Feed H014-038ae into the existing deterministic RTT/bandwidth analysis with the retained real sub-layer payload matrix; regenerate only the coarse network JSON/CSV/chart/edge-class artifacts. |
| Benchmark | Profiles `0/sub-0.5/0.5/1/2/5/10/20 ms` and the existing bandwidth grid; report tested and exact max RTT/min bandwidth, capacity retention at 5 ms/5 Gbps and coarse-versus-fine envelope. |
| Result | **FALSIFIED.** The exact promoted-binary input has `0.6102 ms` exposed loopback/serialization p50 and `258,283` wire bytes. The previously admitted `5 ms / 5 Gbps` pair retains only `85.908%`, below the preregistered `>=90%` gate. Exact bounds are `5.464781 ms` at 100 Gbps and `0.786234 Gbps` at 0.25 ms; at 5 ms the exact minimum is `8.165332 Gbps`, so the next tested viable bandwidth is 10 Gbps. The generated H014-038ag receipt says `PASS` only because its status predicate checked that *some* useful region existed and that coarse edges were looser than fine edges; it did not gate the advertised pair. That receipt is retained as failed-hypothesis and validation-defect evidence, not as admission evidence. |
| Inspection | Coarse transport remains materially looser than fine microwork (`<=0.5 ms / >=2.5 Gbps`), but RTT and bandwidth are a coupled operating point. Selecting the independent headline extrema `5 ms` and `5 Gbps` created an invalid Cartesian pair. At 5 Gbps, the maximum tested viable RTT is 2 ms; at 5 ms, the minimum tested viable bandwidth is 10 Gbps. |
| Bottleneck | **Exposed transport plus wire-transfer time at the jointly selected edge.** The source defect was a hard-coded admission pair and an incomplete receipt status predicate, not CUDA, payload correctness or stage compute. |
| Decision | **REJECT 5 ms / 5 Gbps. MODIFY the admission analysis, capacity model and deployment policy to consume one measured coupled pair and fail closed if that pair misses 90%.** |
| Redesign | H014-038ah tests the minimum coupled-policy correction: preserve the preferred tested 5 ms RTT only with the minimum tested viable bandwidth at that RTT (10 Gbps), enforce the recommendation in the receipt status, and propagate the measured pair rather than duplicate constants. |

| Field | H014-038ah (preregistered coupled coarse-edge admission correction) |
|---|---|
| Hypothesis | Selecting the minimum tested bandwidth that actually passes at the preferred 5 ms RTT will produce a `5 ms / 10 Gbps` coarse admission with `>=90%` modeled capacity retention; making that recommendation an explicit receipt gate and consuming it downstream will prevent the invalid independent-extrema combination without changing Kimi execution or the winning whole-layer topology. |
| Implementation | Add a deterministic coupled-pair selector to coarse network analysis; require the selected row itself to satisfy the 90% gate; emit explicit acceptance gates; replace duplicated 5/5 constants in the capacity model and Experiment 015 package with the resulting 5/10 policy; add a focused regression test. No CUDA, stage-runtime, numerical, payload or topology algorithm change. |
| Benchmark | Replay all 72 exact promoted-binary RTT/bandwidth rows, assert the coupled 5 ms recommendation is the minimum tested viable bandwidth, rerun the capacity/topology/economics model, then rebuild and validate the promoted source/package/placement chain. |
| Result | **PASS for the network-analysis test.** The regenerated v2 receipt selects the coupled `5 ms / 10 Gbps` row, which retains `91.259764%` capacity. All three explicit gates pass. Exact maximum RTT at 100 Gbps is `5.464781 ms`; exact minimum bandwidth at 5 ms is `8.165332 Gbps`; 10 Gbps is the minimum passing tested tier. The measured edge service is `3.316827 ms` (`0.6102` base + `2.5` one-way propagation + `0.206626` wire transfer). Receipt SHA-256: `02a442914db950f7a48e9f01772c191f97ddc91c960ad11eb481e896375336e6`. |
| Inspection | The exact 5/10 row is above the 90% floor and coarse remains materially looser than the four-worker fine class. The matrix/chart numerical bodies are unchanged from H014-038ag because the physical inputs and replay grid did not change; what changed is the fail-closed selection and status semantics. Focused Ruff passed and the new selector plus deployment-seam tests passed `5/5`. |
| Bottleneck | **At the admitted boundary, the 5 ms propagation term dominates the `0.206626 ms` transfer term; aggregate pipeline capacity remains compute-bound because `3.316827 ms` is only modestly above the `3.026928 ms` stage bottleneck.** |
| Decision | **RETAIN the coupled 5 ms / 10 Gbps coarse edge. REJECT independent RTT/bandwidth extrema.** Propagate the receipt, not duplicated constants. |
| Redesign | H014-038ai rebuilds the capacity/topology/economics decision from this receipt and the exact promoted-binary sub-layer batch receipt before placement/package regeneration. |

| Field | H014-038ah2 (preregistered promoted-binary complete-stage batch regression) |
|---|---|
| Hypothesis | The exact promoted DLL will preserve complete layer-89 KDA+MoE and layer-91 Gated-MLA+MoE correctness, state isolation, lifecycle stability and GPU health for the already certified incremental batch prefix 1/2/4/8, while each batch-8 device p50 remains within 15% of its retained predecessor-binary value. |
| Implementation | The first attempt reused the stale pre-assignment identity file and failed closed before model loading with `configured model identity has no valid assignment SHA-256`; GPU health remained measured (`30,248 MiB` free, `46 C`, P8). Update only the harness identity resolver to use the current worker-scoped P1 directory, and add layer-89/layer-91 identities carrying the exact immutable placement assignment hashes. Reuse the promoted DLL/full-graph certificate, stop at batch 8 and do not execute batch 16. |
| Benchmark | For each layer independently: serial oracle, then batch 1, 2, 4 and 8 only after the prior size passes; at every size persist exact routes/state/numerics, synchronize CUDA, check error state, run the known-safe batch-1 fixture, record VRAM and require measured `nvidia-smi`. Compare KDA batch-8 p50 with `13.670944 ms` and MLA batch-8 p50 with `13.169136 ms`. |
| Result | **PASS after two retained fail-closed preflights.** Both corrected runs passed serial oracle and batches 1/2/4/8. Layer 89 batch 8: device p50/p95/p99 `13.305904/15.062227/16.711281 ms`, wall p50 `14.830650 ms`, `601.236868 rows/s`, `-2.670%` p50 versus predecessor. Layer 91 batch 8: device p50/p95/p99 `12.875680/13.169075/13.461638 ms`, wall p50 `14.503900 ms`, `621.326409 rows/s`, `-2.228%` versus predecessor. Both have zero batch-versus-serial error, exact routes/state, every selected expert exactly once, zero warm lifecycle deltas, passing safe fixtures, clean CUDA error state and measured `nvidia-smi` after every size. KDA receipt SHA `71810d0a22d8d80f7664e2d0b28d778837801fe8af7f702d68966d47901dbaaa`; MLA SHA `04bcd176f6dc60e3c1edbe8a397fd42d28f501d26429b4a3c402d11899682339`; both bind DLL SHA `c3bdb40d...ae326` and promoted graph SHA `c4a5836a...fcb57`. |
| Inspection | The failures were identity-contract drift and then a one-line resolved-path plumbing defect in the older benchmark harness, not batch execution; fail-closed checks prevented both from reaching model load/CUDA. After correction, all safety/numerical gates passed and both p50s landed inside the preregistered `[85%,115%]` bands. Tail variance is higher for KDA (`16.711 ms` p99) than MLA (`13.462 ms`), but retained aggregate medians improved. |
| Bottleneck | **Repeated expert-weight traffic with limited cross-row route overlap remains the complete-stage batch limiter; KDA also has the larger observed tail.** No final-binary regression was found. |
| Decision | **RETAIN safe batch 8 and replace both predecessor-binary complete-stage inputs in H014-038ai.** Preserve the two rejected preflights as fail-closed evidence. |
| Redesign | Rebuild the capacity/topology/economics model using only the exact promoted-binary batch, fine, contextual and coarse evidence wherever the model consumes those paths. |

| Field | H014-038ai (preregistered corrected-edge capacity/topology replay) |
|---|---|
| Hypothesis | Consuming the passing coupled 5 ms / 10 Gbps receipt and exact promoted-binary v2 sub-layer batch evidence will keep candidate B as the unique equal-price throughput-per-dollar winner, keep projected 8K aggregate capacity within 90-120 tok/s, keep admitted single-user cadence below 5 tok/s, and change neither the 93-worker count nor the economic conclusion. |
| Implementation | After H014-038ah2 passes, rebuild the source-backed model using the exact promoted-binary complete-stage receipts, H014-SUB-012a-006F for final-binary fine batching, H014-038ae for final-binary coarse transport, and H014-038ah for the coupled edge. The first artifact-generation attempt completed calculations but failed before receipt creation because this module implicitly selected Tk on the headless runner; set only its Matplotlib backend to deterministic `Agg` and rerun unchanged inputs. |
| Benchmark | Recompute all four topology candidates, 3090 operation-class projection, held-out gate, memory fit, serving frontier and price/utilization sensitivity. Require every model gate, candidate B selection, 90-120 aggregate tok/s, and a coarse recommendation exactly equal to 5 ms / 10 Gbps / at least 90% retention. |
| Result | **FALSIFIED after one non-scientific artifact-generation retry.** The valid receipt is `FAIL`: candidate B still wins, edge 5/10 retains `91.259764%`, held-out/memory/economic-rank gates pass, but projected aggregate is only `37.339675 tok/s`, outside 90-120. Per-user is `0.930950 tok/s` at `1,074.171827 ms`; candidate-B cost/M is `$34.5923` at `$0.05/GPU-h` and `$114.1547` at `$0.165/GPU-h`. Receipt SHA `05b99c49b8d08ac83238802c9ba697f901f055e8470283cb5e511774d7a8aa02`. |
| Inspection | The coupled network edge is not the regression. The model now consumes H014-038's three-sample P1 stage-zero correctness timing (`13.765504 ms` device p50), projects it to `26.781165 ms` on 3090 and therefore caps aggregate service at `37.339675 tok/s`. Yet the same exact receipt's seven-call PREPARE fixture measured `2.102624 ms` p50, while prior sustained stage-zero measured `2.702816 ms`. The model mixed a post-3.5-second thread-quiescence/three-call lifecycle probe with a sustained-capacity input. |
| Bottleneck | **The observed model bottleneck is stage zero, but its timing semantic is mismatched: idle-after-quiescence lifecycle latency was treated as sustained service.** This must be measured rather than overwritten with the old value. |
| Decision | **REJECT this capacity receipt for placement. RETAIN it as a failed hypothesis and semantic-validation catch.** Candidate B remains provisionally ranked first, but the 37.34 tok/s estimate is not accepted until steady stage-zero service is isolated. |
| Redesign | H014-038aj measures exact-final stage-zero sustained production service after READY separately from idle first-call behavior, then changes the capacity consumer only if that measurement proves the semantic mismatch. |

| Field | H014-038aj (preregistered exact-final stage-zero timing semantics) |
|---|---|
| Hypothesis | The `13.765504 ms` H014-038 stage-zero p50 is an idle/quiescence lifecycle-probe plateau rather than sustained service: after the unchanged seven-call PREPARE, a continuous production-path window will have device p50 `<=3.5 ms`, zero warm lifecycle deltas, exact state/correctness inherited from P1, and at least 2x lower p50 than the three post-quiescence calls. |
| Implementation | Extend only the stage-zero certification harness with a separate 21-call warmup and 100-call retained production-path timing window using persistent runtime/weights/route and bounded three-token sessions; record p50/p95/p99, wall/device, first-call behavior, lifecycle and health. Keep the existing three-position correctness/lifecycle probe unchanged. Make the capacity model consume `steady_performance` only when its explicit gates pass. |
| Benchmark | Exact promoted DLL, real checkpoint, real token IDs/activations, canonical persistent stage-zero worker, seven-call PREPARE, original P1 correctness probe, then 21 warm and 100 retained calls. Compare retained p50 with `3.5 ms` and with the original post-quiescence p50; inspect CUDA health, VRAM, state closure and lifecycle. |
| Result | **FALSIFIED on lifecycle despite confirming the timing prediction.** Post-quiescence p50 was `27.523712 ms`; continuous retained device p50/p95/p99 was `2.123232/2.247045/2.257762 ms`, wall p50 `3.044450 ms`, a `12.963x` ratio. CUDA error state, health and VRAM recovery passed. However, cycling bounded three-token sessions caused `1,020` persistent-buffer allocation events inside the retained window, so the zero-lifecycle gate failed and the receipt status is `FAIL` (SHA `a09644ef41bf7f8b731a2bde2b372b18724e339e0052a06fd25bfcf1d79cac4c`). |
| Inspection | The numerical timing evidence supports idle-state recovery as the 13-28 ms plateau mechanism, but the first harness design did not model a real persistent decode stream: it opened and closed 34 retained sessions solely to stay under the fixture's three-token context bound. All other gates passed. |
| Bottleneck | **Harness-induced state-buffer recreation, not stage-zero CUDA compute.** Sustained stage-zero compute is approximately 2.12 ms on the RTX 5090; idle first-call latency remains a real tail/TTFT concern. |
| Decision | **REJECT H014-038aj as a capacity source. MODIFY the harness; do not waive lifecycle.** |
| Redesign | H014-038aj2 uses one persistent warm session and one persistent retained session with a sufficiently large bounded context, so no session or state-buffer construction occurs inside the measured window. |

| Field | H014-038aj2 (preregistered persistent-session stage-zero timing) |
|---|---|
| Hypothesis | Repeating the exact stage-zero timing window on one persistent retained decode session will preserve device p50 `<=3.5 ms` and the `>=2x` idle-to-steady separation while reducing all warm lifecycle deltas, including persistent-buffer allocation, to zero. |
| Implementation | Set the stage-zero fixture context bucket to the bounded 121-call window; open one session before each warm/retained phase, execute monotonically increasing positions within it, take lifecycle snapshots only after the retained session is open, then close it and verify memory/session recovery. No CUDA/runtime arithmetic change. |
| Benchmark | Same exact final DLL/checkpoint/identity/P1 path; 21 warm calls then 100 retained calls, p50/p95/p99, first warm call, post-quiescence ratio, CUDA error, `nvidia-smi`, VRAM recovery, active sessions and every lifecycle counter. |
| Result | **PASS.** Post-quiescence device p50/p95/p99 was `13.926880/26.093900/27.175413 ms`; the first warm call was `27.487553 ms`. After 21 warm calls, the 100-call persistent retained window measured device p50/p95/p99 `2.125536/2.152662/2.220488 ms` and wall p50/p95/p99 `2.970350/3.233735/4.000913 ms`, a `6.552x` post-quiescence-to-steady ratio. All nine gates pass: every lifecycle delta is zero, one retained session stays active only during timing and closes afterward, free VRAM returns exactly to `29,386,342,400` bytes, CUDA error state is clean, and pre/post `nvidia-smi` is measured. Receipt SHA `4daf5c34eaf7689556aea95eda617812c6442e960d2bee295bdb6c15c89a97de`; DLL SHA `c3bdb40d...ae326`. |
| Inspection | H014-038aj and H014-038aj2 agree on steady p50 within `0.11%`; eliminating session churn removed all `1,020` allocation events without changing compute time. The 13-27 ms idle plateau is real latency after long quiescence, but it is not sustained stage service and must not cap aggregate throughput. |
| Bottleneck | **GPU idle-state/clock recovery dominates first-call and post-quiescence tails; steady stage-zero service is dense/embedding memory traffic at about 2.13 ms device p50.** |
| Decision | **RETAIN H014-038aj2 as the stage-zero sustained-capacity source. Retain the idle plateau separately for cold/TTFT risk.** |
| Redesign | H014-038ak reruns capacity/topology/economics with the gated steady window and exact promoted-binary sources; it must still independently pass every model gate. |

| Field | H014-038ak (preregistered steady-semantic capacity replay) |
|---|---|
| Hypothesis | Replacing only the invalid three-sample post-quiescence capacity input with H014-038aj2's gated persistent-session metric will restore projected 8K aggregate capacity to 90-120 tok/s, retain candidate B/93 workers as the unique equal-price throughput-per-dollar winner, retain the 5 ms / 10 Gbps coarse admission, and leave the `$15/M` economics conclusion negative at `$0.165/GPU-h`. |
| Implementation | **None beyond the H014-038aj capacity-consumer gate.** Use H014-038aj2 as `stage_zero`; keep the exact-final B8 KDA/MLA, contextual, sub-layer, coarse transport/network, final/head and held-out inputs unchanged. |
| Benchmark | Recompute all candidates, serving latency/frontier, memory, 3090 operation-class projection and price/utilization sensitivity. Require all gates, 90-120 aggregate tok/s, candidate B, 93 workers, coarse 5/10 with >=90% retention, and explicit idle plateau retained outside sustained capacity. |
| Result | **PASS.** All nine gates pass. Projected RTX 3090 aggregate is `97.150284 tok/s` (`38.7089%` device and `39.4338%` wall retention); admitted per-user decode is `0.950295 tok/s` at `1,052.304878 ms`, or `1.215955 tok/s` at the 0.25 ms/25 Gbps fast-domain sensitivity. Stage-zero projects to `4.914216 ms`; the 68 KDA layers total `427.443097 ms`, 23 8K MLA layers `305.255557 ms`, final/head `9.543978 ms`, and each admitted coarse edge `3.316826 ms`. Candidate B remains the unique winner with 93 workers; fine capacity is `75.105171 tok/s`. Held-out median APE remains `0.552550%`. Receipt SHA `38c190a47dc33111550b8d2f7fae410b300068a34fca9ad344ed66f263982c26`. |
| Inspection | H014-038ai's 37.34 tok/s was entirely the invalid stage-zero semantic. With a gated steady input, aggregate capacity returns within `0.32%` of H014-036a despite final-binary batch and network updates. Aggregate service is now limited by the 8K contextual route-mix wall service (`97.150 tok/s`), not stage zero, final/head or network. Sequential dependency depth keeps per-user cadence below 1 tok/s. At `$0.05/GPU-h`, cost/M is `$13.2956` and margin at `$15/M` is `11.363%`; at `$0.165`, cost/M is `$43.8753` and margin is `-192.502%`. |
| Bottleneck | **Aggregate: row-serial 8K Gated-MLA contextual service. Per user: 93-stage sequential compute plus 92 coarse edges. Economics: rental price above roughly the low-price scenario, not GPU utilization alone.** |
| Decision | **RETAIN candidate B: 93-worker WHOLE-LAYER topology, 5 ms / 10 Gbps coupled coarse admission, batch 8, 8K canary workload.** Sub-layer execution remains functional/memory-reducing but uneconomic for the initial fleet. |
| Redesign | H014-038al freezes the expanded source/evidence set, re-promotes the unchanged DLL hash, and regenerates placement/distribution/rehearsal/package identities from H014-038ak. |

| Field | H014-038al (preregistered final evidence-chain promotion) |
|---|---|
| Hypothesis | Because H014-038ah through H014-038ak changed Python harness/analysis/policy but no native CUDA source, re-promotion will preserve DLL SHA `c3bdb40d...ae326` while producing a new exact source manifest; the regenerated 93-worker placement will preserve ownership/memory feasibility, change coarse admission only to 5 ms / 10 Gbps, and all distribution/rehearsal/package validators will pass. |
| Implementation | Extend promotion prerequisites to bind the exact-final complete-stage, steady stage-zero, sub-layer, coarse transport/network and capacity receipts; update the Experiment 015 package to embed H014-038ak/H014-038aj2 evidence and derive its network policy from H014-038ah. Then regenerate—do not hand-edit—promotion, placement, execution, distribution, rehearsal, canary and package artifacts. |
| Benchmark | Promotion `--check` then promotion; exact hash/source/cubin/PTX validation; final placement and all downstream logical fixtures; fresh wheel; staged and canonical package positive validation plus all 16 tamper controls; compare ownership count/bytes, VRAM/headroom and network policy. |
| Result | **PASS.** Promotion preflight and write passed while preserving DLL SHA `c3bdb40d49a1b0e512ddb1e84485e0bdcf9d2e6ac6fa4049d81ff2ecd12ae326`; promotion receipt SHA is `567f881ac0fb6137c34b8588dc68852ea79e616000e816b26306c4465861f974`, source manifest SHA `476b145a73babc16660746db9e921056efd78705ef386b22df2f8519d8a9554c`, source bundle SHA `1f4a9c6154b309bc212615d53aa2fee8d11eb22658629289dd1f3aa874d372c0`, operation matrix SHA `405fd8b49b8036a6ba1bc165a915f3c5c2a9e4a888e6330cfcc8099d0f52e8ec`, and sm_86 certificate SHA `799499fdfa0f7a5f86d71e30f44a451deb7918506982dbabdbe28c3f858eb2c9`. Regenerated placement SHA `943fe0748b055d11f097827bd9e483c3209b6b696bde319e8230083825fda982` passed all 11 gates with 93 workers, 497,052 tensors, 1,559,965,606,912 uniquely owned source bytes, maximum planned VRAM 24,276,277,658 bytes and minimum total headroom 4,070,506,496 bytes. Distribution SHA `aa4e070f5ccf865ebf1a9428d6ff7f91949f8bbb6d41435ec9589759d9efc942` and rehearsal SHA `8c7e7fe72bb4501e2bf912c4f6e64ba06bd784057c4e58af3e4ac16cf9f4ad04` passed with 93 READY workers, 576 events, zero orphan ownership/routes and exact parent hashes. Fresh wheel SHA is `265f1819fe2c33914a783a3c0528a09c7a3365765ab6d6f69b8b7d86783cd5d8`. The promoted canonical package validates 155 locked files; package-lock SHA is `6daa53129d115e8601c36cda3978f2e65a86f61411c69dbed64f2247f4a209b4`, canonical archive SHA `6f9a276d3496871b4359733470b506a6365dce292afe09d08c2cbc10e66dfe99`, and all 16/16 tamper controls pass in `deployment/h014-038al-canonical-package-controls.json`. |
| Inspection | Every host-side downstream artifact points to its immediate parent; the portable package intentionally rewrites internal paths and re-locks their bytes. It consistently carries the 93-node whole-layer plan, 5 ms / 10 Gbps coarse admission, separate optional 0.5 ms / 2.5 Gbps fine-domain class, exact runtime/wheel identities and physical canary status `NOT_RUN`. Fleet activation remains false until a physical certificate, per-node admission and cost guard pass. |
| Bottleneck | **Resolved locally:** evidence/package convergence. The remaining uncertainty is deliberately physical RTX 3090 behavior, not an unfrozen software identity. |
| Decision | **RETAIN.** H014-038al is the canonical Experiment 015 pre-canary package/evidence chain. The former canonical package and archive were retained as the recoverable `h014-038-pre-network-backup`. |
| Redesign | H014-038af runs the full regression and deterministic final-report integrity closure; any failure remains fail-closed. |

| Field | H014-038af (preregistered final report and regression closure) |
|---|---|
| Hypothesis | Once the promoted-binary fine/coarse replays pass, the current source will pass first-party Ruff and the complete pytest suite, the canonical package will remain valid, the Markdown/JSON cycle ledgers will reconcile with no duplicates or incomplete entries, and all 28 final gates can be derived from retained evidence without stale top-level FAIL/PENDING claims. |
| Implementation | Add a deterministic evidence-backed final-report generator and replace the stale original verdict with a current verdict while preserving the original report as explicitly superseded history. Change no runtime, CUDA, topology or benchmark code. |
| Benchmark | Full first-party Ruff; full pytest with JUnit receipt; canonical package/archive validation; ledger reconciliation; machine-summary, manager-summary, acceptance-gate, chart-index and evidence-integrity validation against exact source hashes. |
| Result | **PASS.** First-party Ruff is clean. The exact current repository suite passed `1,091`, skipped `13` explicitly physical/configuration-gated tests, and failed zero in `191.82 s`; retained JUnit SHA is `55f516d35c237f4e070b774c50e175bc549e9a6cf85bbdc08da9ff01e3b65751`. The canonical 155-file package/archive validates and all 16/16 tamper controls reject their mutation. The deterministic compiler generated the current report, manager summary, 28-gate JSON, machine summary, chart index, 177-cycle ledger summary and evidence-integrity manifest; reconciliation found zero duplicate, incomplete or orphaned cycle IDs and the no-write compiler check reported zero drift. |
| Inspection | Every summary metric is read from and hash-bound to its retained source receipt. The stale original FAIL verdict and manager answers remain only beneath an explicit superseded-history heading. All three decision charts passed visual inspection. Physical RTX 3090 execution remains correctly `NOT_RUN`; gate 26 and the deployment package are `READY`, not physical `PASS`, and static fleet activation remains false. |
| Bottleneck | **No locally solvable Experiment 014 blocker remains.** The next uncertainty is physical RTX 3090 execution and fleet behavior. |
| Decision | **RETAIN AND CLOSE EXPERIMENT 014.** All 28 final gates are satisfied: 26 `PASS`, two pre-canary `READY`, zero `FAIL`. Authorize only the fail-closed Experiment 015 single-3090 canary, followed by the full fleet if its certificate, node admission, network and cost gates pass. |
| Redesign | Experiment 015 validates Linux sm_86 and the RTX 5090-to-3090 projection on one physical 3090 before renting/activating the remaining 92 nodes. |

## Manager questions

### 1. Can we run full Kimi K3 on the planned rented cluster?

Not yet. We can run its complete mathematics serially from the real checkpoint, but cannot run the production distributed GPU deployment represented by the manifest.

### 2. What exactly has been proven?

The local checkpoint is completely classified; all 93 real layers connect and execute; routing, experts, KDA/MLA state, final norm, head, logits, sampling, and a subsequent state transition work; an independent mixed-layer reference agrees; official conversation tokenization agrees; 73 nodes fail the stated memory model; a 96-node plan has complete ownership and route coverage; exact local worker-package extraction is deterministic and integrity checked; and the repository regression suites pass.

### 3. What will still only be learned physically?

After the software blockers are fixed, the unavoidable physical unknowns are actual RTX 3090 kernel performance, physical GPU-network performance, real multi-GPU contention/jitter, and resulting full-cluster throughput. At present, additional software/deployment unknowns remain, so those are not yet the only uncertainties.

### 4. What topology should Experiment 015 use?

No topology is approved for Experiment 015 yet. The provisional candidate is a 96-node checkpoint-aligned **coarse persistent stage pipeline**. Hybrid fine-grained collectives should only be introduced inside a measured low-latency domain after real Kimi payload and service measurements show a benefit. Calling this the recommended physical topology now would outrun the evidence.

### 5. How many RTX 3090s should be rented?

Zero now. The 73-node proposal is falsified by the 10%-headroom memory model. Eighty-one is only a modeled absolute lower bound; 96 is the provisional auditable layout, not a rental recommendation.

### 6. What network characteristics should those machines have?

No hard minimum can be certified yet. A coarse boundary carries at least one 7,168-element hidden activation per token: 28,672 bytes in the tested FP32 reference representation or 7,168 bytes in the intended MXFP8 representation, before framing/state/control traffic. Those derived sizes have not been replayed through the production Kimi runtime, so latency/bandwidth rental filters remain unset.

### 7. What are the expected throughput and economics?

Unknown. Reporting a number would be fabricated because there are no production Kimi GPU timings, no RTX 5090-to-3090 bottleneck projection, and no held-out service-model error. The only legitimate economic numbers are the target-line arithmetic above.

### 8. What would cause the Experiment 015 canary to abort?

The current canary aborts unconditionally because certification is blocked. Once implemented, it must abort before fleet activation on any unresolved certification marker; non-`sm_86` or unapproved GPU; insufficient usable VRAM/disk/RAM; driver/CUDA mismatch; missing loadable Kimi `sm_86` kernels; native MXFP4, KDA, MLA, routed-expert, or state fixture mismatch; package/model/manifest/lock hash mismatch; failed shard acquisition; failed secure admission; network below the subsequently certified profile; memory-reserve breach; or performance below a preregistered sanity floor.

**PRE-CLUSTER CERTIFICATION INCOMPLETE. Do not rent the full Kimi K3 cluster. Remaining blocker: no production distributed Kimi layer-stage runtime or RTX 3090 sm_86 execution package exists; the canonical delegated path also still creates per-operation child tasks.**
