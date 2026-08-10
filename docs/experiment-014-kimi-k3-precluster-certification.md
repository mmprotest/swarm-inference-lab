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
