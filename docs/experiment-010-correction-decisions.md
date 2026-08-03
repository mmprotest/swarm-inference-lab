# Experiment 010 correction decisions

## Scope and baseline

This document records the implementation decisions for the Experiment 010
correction pass. The work extends Experiment 010 and the pinned downstream
Colibri integration. It does not create Experiment 011, a second repository, or
a parallel inference implementation.

The correction baseline was captured before code changes:

- swarm-inference-lab commit: `866ea26f04dbd8d12b28c7ca1dee4f15e93b1045`
- tracked tree: `e6ca46e6bd4b39e5333f6154c7261d66e6386faf`
- working tree: clean
- Colibri release and commit: `v1.4.0` at
  `b085b48888a88d9a1c00b151a9979774b72cdbfd`
- dependency policy: `explicit_only`; the pin is preserved
- focused Experiment 010 and Colibri tests: `100 passed`
- complete repository tests: `445 passed, 7 skipped`
- reproduced evidence:
  `artifacts/runs/experiment-010-correction-baseline-20260802/experiment_010`
- reproduced verdict: `PARTIAL`
- reproduced failed gates: 4, 6, 8, 10, 11, 12, 13, and 14

The seven repository-suite skips were existing opt-in real-model or physical
tests. No Experiment 010 unit test failed.

The previous Level B evidence identifies
`Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf`, 48,410,988,384 bytes, SHA-256
`d103b2733ec1012a52d01edda66b7e5c24ae50508c9f99f5297ea459ef3c061a`.
Its historical path under `C:\tmp\swarm-exp008-models` is currently absent.
Historical Experiment 008 measurements will not be reused as current results.

## Audited execution path

The pinned `c/olmoe.c` path is:

1. `main()` reads `ref.json`, constructs `Model` with `model_init()`, and calls
   `generate()` for greedy fixed-ID generation (or `tf_nll()` for teacher
   forcing).
2. `generate()` calls `step()` once for prefill and once per generated token.
3. `step()` performs attention, post-attention RMSNorm, calls `moe()`, adds the
   returned MoE value to the residual stream, and produces logits with the
   original output head.
4. `moe()` computes router logits, softmax, rank-ordered top-k expert IDs and
   routing weights, then calls `expert_get()` for each selected expert.
5. `expert_get()` owns the local LRU and calls `load_expert_merged()` on a miss.
6. `load_expert_merged()` reads the exact merged int8 bytes and FP32 row scales
   named `model.layers.<layer>.mlp.experts.<expert>.merged_weight` and `.qs`.
7. The current expert math is gate `matmul_q`, up `matmul_q`, SiLU-times-up,
   down `matmul_q`, routing-weight multiplication, and rank-order accumulation.

The existing Experiment 010 Level A path instead captures a source
Transformers activation, executes source bfloat16 weights in NumPy workers, and
ends at an operator vector. It is retained as diagnostic evidence only. The
official correction path uses the exact merged Colibri container and feeds the
remote value back into `moe()` before `step()` continues.

The canonical expert protocol is `SWARMEX1` in
`src/swarm_inference/experiments/experiment_010/wire.py`: eight-byte magic,
big-endian JSON-header and length-table sizes, big-endian 64-bit blob lengths,
canonical JSON semantics, binary tensor blobs, and rejection of truncation or
trailing bytes. The Universal Worker ABI remains the authoritative lifecycle
and capability control plane; `MOE_EXPERT` remains its expert job role.

## Downstream Colibri patch plan

The existing patches remain in order and retain their hashes:

1. `0001-swarm-bridge.patch`
   (`8fab70ebcdf47e008e8c80149afff203cfce7bbd2f3d01a1289490ba037eebd1`)
2. `0002-olmoe-routing-telemetry.patch`
   (`600b3870117c138b40ba3e6af994de9cdb52533007f3d895f6127931853ce646`)
3. `0003-aggregate-runtime-telemetry.patch`
   (`b7ad05e7c86caebb360b24c03a23ae5cb8416c3b8ff884c095d5a7328042bea5`)
4. `0004-olmoe-machine-readable-telemetry.patch`
   (`855fa8d3ddf0a1da7ef8d201ae7adbe08d718783b5f8a5e75ef50d73b165afcc`)

The correction adds these ordered patches:

- `0005-olmoe-shared-expert-runtime.patch`: move native merged-int8 expert
  metadata, loading, hashing, gate/up/down execution, activation, whole-expert
  execution, microshard execution, and exact/fast aggregation into
  `c/olmoe_expert_runtime.{h,c}`. Both `olmoe.c` and
  `olmoe_expert_worker.c` call these functions; neither owns a copied matmul.
- `0006-olmoe-external-expert-dispatch.patch`: add the plan parser and
  `SWARMEX1` client, insert dispatch after unchanged router selection and
  before residual addition, support local/rpc/hybrid/planner modes, preserve
  exact router rank, and guard remote-owned experts from `expert_get()`.
- `0007-olmoe-native-microshards.patch`: add matched native gate/up rows and
  down columns, stable `fixed_order_fp32` reduction, and the native worker
  executable with whole, microshard, exact, and fast response modes.
- `0008-olmoe-memory-residency-telemetry.patch`: add process, cache-page
  residency, forbidden-local-load, transport, queue, compute, reduction, and
  memory-hierarchy telemetry without changing routing or sampling.

Every new source file carries an Apache-2.0 downstream-modification notice.
The build exports the immutable upstream commit, applies `series` in order,
builds the engine and worker, runs dedicated tests, and records patch and
binary SHA-256 fingerprints.

## Intentional Colibri execution-path changes

This ledger is normative and will be updated if implementation details change.

| Area | Intentional change | Semantic invariant |
|---|---|---|
| Expert representation | Wrap merged int8 weights and per-row FP32 scales in the shared runtime | Tensor bytes, shapes, scales, and quantization remain unchanged |
| Expert loading | Route local cache misses and native worker loads through the shared loader | Unowned experts/ranges fail; no implicit access to a full container |
| Expert math | Route gate/up/SiLU/down and routing multiplication through the shared runtime | Local mode retains the original operation order |
| Router boundary | Capture the original rank-ordered top-k IDs and weights after routing | No router logits, top-k rule, normalization, or rank is altered |
| Dispatch | Group selected experts per owner and send one activation per destination | Remote results replace only their owned local contributions |
| Exact response | Return per-expert contributions and add them in original router-rank order | Exact mode targets local Colibri accumulation order |
| Fast response | Permit worker-side routing-weighted accumulation | It receives a separate quality and token result |
| Microshards | Split gate/up rows and down input columns on native ranges | Native int8 values and original row scales are copied, never requantized |
| Forbidden loads | Refuse `expert_get()` for remote-owned experts | Official runs fail closed and record every attempted violation |
| Failure policy | Implement fail/local/alternate behavior at the expert boundary | No partial request is reported as complete |
| Telemetry | Emit request-, token-, layer-, worker-, expert-, rank-, codec-, and evidence-scoped events | Observation does not modify model state or sampling |
| Generation | Continue through the existing `step()` and greedy sampler | Same container, tokenizer, input IDs, stop rule, and sampler form the oracle |

## Python integration decisions

- Extend, rather than replace, `ExpertExecutionRequest`,
  `ExpertExecutionResponse`, `StableExpertCoordinator`, `ExpertTransportClient`,
  and `SWARMEX1` framing. Flat routing remains a backward-compatible one-row
  spelling; official requests use per-row expert IDs, weights, and selected
  ranks.
- Add `colibri_expert_bank.py` to copy exact tensor byte ranges from the merged
  container into worker banks and matched microshard banks. Source/destination
  range hashes, dtype, shape, scales, identities, offsets, and ownership are
  verified before a bank is eligible.
- The coordinator receives ownership metadata and endpoints, never remote bank
  paths. Each worker receives only its own bank and ownership manifest.
- Use `ColibriReplayRunner.generate_from_tokens()` for the authoritative
  one-shot greedy oracle and candidate paths. Teacher-forced replay and source
  activation capture cannot satisfy token gates.
- Add a correction runner that records prerequisites and executes all available
  phases. It must never synthesize missing measurements, substitute zeros, or
  convert an artifact's existence into workload completion.
- Add `FULL_COMPLETE`, `INCOMPLETE_FULL_RUN`, `DEVELOPMENT_COMPLETE`, and
  `QUICK_COMPLETE`. Full mode defaults to requiring completeness; only the
  explicit development override permits an incomplete diagnostic run, which
  remains visibly incomplete and ineligible for `PASS_STRONG`/`PASS_CLOSURE`.
- Extend evidence schemas first, then gate on row-level eligibility. Planner and
  simulator inputs must trace to completed real Level A distributed rows.
- Reuse measured route/cache/transport/failure events for simulator behavioral
  replay. Timing calibration uses a 70/30 configuration-level split that holds
  out complete strategy/network/worker/layout combinations.
- Replace the sparse Kimi fixture with deterministic dense, valid E2M1/UE8M0
  groups and a compiled native execution path. It remains
  `SYNTHETIC_FIXTURE` unless checkpoint bytes are present.

## Verification order

Implementation and verification follow the requested phases without skipping
forward over a failed token-identity phase: shared runtime, C wire, native
banks, in-`moe()` dispatch, 50-prompt whole-expert identity, 20-prompt native
microshard identity, capacity isolation, real-model CUDA, mandatory workloads,
failure and corruption matrices, planner, simulator, Level B and dense Kimi,
then final evidence regeneration and audit.

An unavailable mandatory input does not erase completed work. It produces
`INCOMPLETE_FULL_RUN`, lists the exact missing prerequisite, and prevents an
official closure verdict.

## Implemented correction record

### Phases 1 through 4

- Phase 1 reproduced the original `PARTIAL` result at repository commit
  `866ea26f04dbd8d12b28c7ca1dee4f15e93b1045` and tree
  `e6ca46e6bd4b39e5333f6154c7261d66e6386faf`. The pre-correction suite passed
  445 tests with 7 skips. Gates 4, 6, 8, 10, 11, 12, 13, and 14 remained open.
- Patch 0005 introduced the shared runtime and removed the duplicate native
  int8 matmul from `olmoe.c`. The original and refactored engines produced the
  same 32 generated token IDs and byte-identical 608 machine-readable routing
  records on the exact Level A container.
- The canonical adapter now implements strict C encode/decode for `SWARMEX1`
  and `SWARMT01`. Python-to-C, C-to-Python, malformed-length, checksum, and
  trailing-byte tests pass. Per-row rank-preserving routes and exact/fast
  response shapes extend the existing protocol rather than replacing it.
- `colibri_expert_bank.py` streams exact safetensors payload ranges. Whole
  banks contain only authorised merged-weight and scale tensors. Native
  microshards pack original gate/up rows, down input columns, and unchanged
  down row scales. No conversion step dequantizes or requantizes them.
- The real Level A container fingerprint established by the bank converter is
  `sha256:bad0c225e9bc03275cb12c6606dac4358bcdc188ca701f654ea9672a6cecc35e`.
  Real layer 0 expert 59 reconstructs from ranges `0:512` and `512:1024` with
  merged-weight hash
  `273ee0fca6e3da32c31c24b5afaa7b7f7c8d19d8059ef3bf5e02f3826805b810`
  and scale hash
  `9b0fe467fc28baa654e9c1505439f3affb9fbd8265ddc38cdd5f1ce0cd5fcca4`.
  The C runtime accepts owned expert 59 and rejects unowned expert 8.

The worker bank and reconstruction proofs above establish exact native byte
ownership only. They do not satisfy the token-path microshard or capacity
gates; those remain pending until the subsequent in-`moe()` phases consume the
returned contributions.

### Phases 5 through 7

- Patch 0006 inserts the external dispatch call inside the pinned Colibri
  `moe()` function after the unchanged router has produced rank-ordered expert
  IDs and weights, and before the expert contribution is added to the residual
  stream. Exact worker responses are unweighted; the coordinator applies the
  original routing weight and uses Colibri's original rank-order accumulation.
  `expert_get()` fails closed for remote-owned experts.
- Four disjoint whole-expert banks cover all 1,024 layer/expert pairs. The
  authoritative whole-expert exact suite passed 50/50 prompts and 1,600/1,600
  generated tokens with byte-identical router traces, 93,779 consumed remote
  responses, zero forbidden local loads, and zero local fallbacks.
- The first native microshard smoke deliberately remains in the evidence. It
  independently scaled each 512-wide down partial and diverged after 5/12
  tokens. The correction carries the unscaled FP32 down-dot accumulator from
  the `0:512` worker to the `512:1024` worker in canonical `SWARMEX1` chain
  state. The final worker alone applies the unchanged original down-row scale;
  Colibri then applies the unchanged routing weight. This preserves the full
  expert's native 16-wide group and left-to-right accumulation order.
- Patch 0007 has SHA-256
  `fa9fa1d5b6d6ca4ccf384a2945c806e7abd3f8abfdbe560d5997f2db95ca1d7f`.
  Its clean export applies reproducibly to upstream commit
  `b085b48888a88d9a1c00b151a9979774b72cdbfd`. The resulting source-tree hash is
  `50320d4e47283dea523bbcc5b3c68a299d273db4b358874050222d29d47fd6d5`;
  `olmoe.exe` is
  `808b94fde7b6fd0215c9f1095fd240d93bbb23ae10d2efb9de089aaf15dbfde4`
  and `olmoe_expert_worker.exe` is
  `76ab18d4e520cfe3a0221731411fd93ba1a3b247f71954dfdae420507e4cf4c6`.
- The mandatory native-microshard correctness suite passed 20/20 prompts,
  640/640 generated tokens, and 20/20 byte-identical router traces. It consumed
  20,480 real shard responses inside Colibri, with zero forbidden local expert
  loads and zero silent fallback. Each isolated worker bank contains 1,024
  authorised native ranges and 3,233,808,384 bytes. Raw rows are in
  `artifacts/runs/experiment-010-correction-work/phase-7/microshard-exact-correctness/colibri_rpc_token_results.csv`.

The Phase 7 result satisfies the software and measured correctness conditions
for Gate 6. It does not by itself satisfy capacity isolation, performance,
failure, trust, planning, simulation, CUDA, or full-run completeness gates.

### Phases 8 and 9

- Patch 0008 physically removes routed-expert tensors from the coordinator
  container and skips opening its local expert runtime when the plan covers all
  experts remotely. Four disjoint native banks each own 256 of 1,024 experts
  and 25% of the routed bytes. The mandatory 10-prompt, 128-token capacity run
  matched 1,280/1,280 tokens and all router traces, consumed 74,965 remote
  responses, and recorded zero forbidden loads or fallbacks. Every worker
  remained below its 268,435,456-byte cache budget; the coordinator reported
  zero owned routed-expert bytes.
- Patch 0008 uses native Windows process and page-residency APIs and keeps
  unavailable pagefile-read, hard/soft-fault, and memory-compression attribution
  explicitly null. The shared runtime now caches both whole experts and native
  microshards under its measured LRU budget.
- The Phase 9 checkpoint Patch 0008 SHA-256 was
  `4fa82e6c549801502ba297acf3eadb5e1dd6d598c1566a25b3bc3fdb4328e918`.
  A clean pinned export has source-tree hash
  `51ba3660b3f59124d80d211b0e13a7da4f7ced654273ffd01f834186733d942d`,
  `olmoe.exe` hash
  `5b61b1841ef0cfa34f3f3c4b5050936b268b3c636c0c7051439ddc8c82179cf2`,
  worker hash
  `6729b1c506dd7373f8f5fc5e7ac569be378f909b19a41e8d5bd9f45f6ef0512a`,
  and CUDA DLL hash
  `3da4d672faee9a00d3674703b44fdef3d6ab5f80fed3c0d59a2199d991a88d78`.
- The same patch connects the shared runtime to Colibri's CUDA DLL without a
  second expert implementation. The worker uploads the exact merged int8 gate,
  up, and down matrices and original F32 row scales. A configured CUDA target
  fails the request if initialization, upload, residency, or execution fails;
  it never falls back to CPU.
- The real expert proof for layer 0, expert 5 uses weight hash
  `cb4af761168953775bfe7fb38b34aecb865c0a0e8c73302608a7ebe107b0d8bc`.
  Three native tensors occupy 6,307,840 GPU bytes on the RTX 5090. The operator
  relative L2 error is approximately `4.96e-7`, with a recorded GPU kernel event
  and no CPU fallback.
- The end-to-end CUDA run holds `IDOT=0` identical across the local CPU oracle
  and hybrid CUDA path because the CUDA ABI accepts FP32 activations. It matches
  32/32 generated tokens and the router trace byte-for-byte, consumes five RPC
  responses, executes expert 0:5 seven times on CUDA, and records zero fallback.
  The default `IDOT=1` comparison is retained as a negative diagnostic: its
  activation-requantized CPU arithmetic is a different numerical contract and
  diverged at token 5. This result is single-machine process isolation, not
  physical distributed inference.
- Raw Phase 8 evidence is under
  `artifacts/runs/experiment-010-correction-work/phase-8`; reconciled Phase 9
  evidence is
  `artifacts/runs/experiment-010-correction-work/phase-9/real_model_cuda_results.json`.

These results close Gate 8 and the required real-model CUDA participation
subphase. They do not substitute for the mandatory workload, failure, trust,
planner, simulator, Level B, or dense Kimi phases that follow.

### Phase 10 execution-path changes and provisional workload record

- Patch 0008 now includes a native named-shared-memory data plane. It carries
  the exact canonical `SWARMEX1` request and response frames and uses the TCP
  control channel only to declare the two mapping names and bounded sizes.
  Direct TCP, shaped relay TCP, and shared memory therefore share one semantic
  wire implementation. The C shared-memory test and Python relay/shared
  integration tests pass.
- Relayed TCP keeps one framed connection per coordinator/worker pair and
  applies shaping to the actual frame bytes. Relay byte/message counters come
  from the payload path; evidence-file writes are outside the timed relay loop.
- OLMoE attention uses one `max_t`-sized score row per OpenMP thread instead of
  `float sc[4096]`. Scalar score and value accumulation order is unchanged.
  This removes the fixed-replay overflow for the mandatory 8K prefill path; it
  does not expand the model's advertised 4096-position support claim.
- The native worker now reports time waiting for its execution mutex separately
  from compute time. A real two-request concurrent smoke matched 256/256 tokens
  and measured 1,880,772,400 ns of queue wait across 14,874 RPC completions.
- Equal microshards use native ranges `0:512` and `512:1024`. The asymmetric
  layout was derived from measured worker throughput and uses `0:384` plus
  `384:1024`; reconstruction preserves the source weight and scale hashes.
- The completed three-repeat provisional short matrix has 60 measured rows per
  configuration. Exact token identity is 60/60 for local, direct TCP, relayed
  TCP, shared memory, equal microshards, and asymmetric microshards. Median
  throughput is respectively 5.08, 2.99, 2.89, 2.945, 2.82, and 2.84 tok/s.
  Exact distributed candidates are therefore harmful to short-decode speed on
  this host. The separately labelled fast aggregation path reaches 4.44 tok/s
  but matches only 15 of 7,680 generated token positions and is ineligible for
  verified planning.
- The clean instrumentation build applies all eight patches to pinned commit
  `b085b48888a88d9a1c00b151a9979774b72cdbfd`. At this Phase 10 checkpoint its
  Patch 0008 SHA-256 is
  `dcc55f5221ef3e99bdbb8fd53b13546385b8bdba74c156979369c96d5ff3c2c3`, its
  source tree is `fc602ae50afc80df4e5e21cfa6a991c6c57ca3686a7435e6d1653a3b56ce5e0a`,
  `olmoe.exe` is `c9d3a3d895ba093f9bf3873b57cc039b0e70de9b1616b7c59a803735a54961fc`,
  `olmoe_expert_worker.exe` is
  `db36410baf95f4f0655dbe498fefe1cd7475cf10f4a7459e2ce9fa4cdef44136`,
  and `coli_cuda.dll` is
  `aacbb79fd4b41d4a05813c275675d06c275a8e9a5ce384bffd24e088ade488fb`.
  Later failure/trust-only additions must receive a new recorded fingerprint;
  they may not retroactively relabel these measured rows.

### Phase 10 completed measured workload record

- The narrow provisional differences required five repeats. The final
  short-decode table therefore contains 100 real 128-token rows for each of
  local, whole-expert direct TCP, whole-expert shared memory, whole-expert
  relayed TCP, fast aggregation, equal microshards, and asymmetric
  microshards. Exact-token rows are respectively 100, 100, 100, 100, 0, 100,
  and 100. Median throughput is 5.26, 2.945, 2.92, 2.87, 4.385, 2.81, and
  2.86 tok/s. The fast path matched only 25 of 12,800 token positions and is
  rejected by the canonical verified-candidate gate.
- The additional repeat rows were produced after a clean rebuild of the same
  downstream source revision. Per-row current executable hashes are retained;
  the first three repeats retain their earlier suite-level executable
  provenance and are not relabelled with the later hash. Resumed worker
  counters use a monotonicity break as an explicit native-process session
  boundary, so a counter reset is never represented as a negative cache or I/O
  measurement.
- The real concurrency matrix completed at 2, 4, and 8 requests for local,
  four-worker direct TCP, and equal microshards, with exact tokens in every
  group. Local aggregate verified throughput is 5.70, 9.94, and 12.76 tok/s;
  direct TCP is 4.21, 7.22, and 8.87 tok/s; equal microshards is 3.69, 6.68,
  and 7.68 tok/s. Direct worker queue fraction reaches 39.7% at concurrency 8;
  the two-worker equal layout reaches 65.8%.
- The mixed-service matrix completed one interactive plus one background and
  one interactive plus four background requests for the same three
  configurations. All groups are exact and starvation-free. Local wins both:
  its interactive p95 is 46.4/57.8 seconds and combined throughput is
  5.39/10.44 tok/s, versus 60.7/88.0 seconds and 4.16/7.28 tok/s for direct
  TCP, and 66.8/86.1 seconds and 3.83/7.43 tok/s for equal microshards.
- Actual relayed payload shaping ran the real token path for
  `loopback_unshaped`, `fabric_100g`, `lan_10g`, `lan_2_5g`, `lan_1g`,
  `wifi`, `regional_wan`, and `global_wan`. Every 32-token row is exact. The
  measured throughput falls from 2.55 tok/s on unshaped loopback to 0.08 tok/s
  on the reduced global-WAN run; relay forwarded-byte counts reconcile with
  coordinator raw payload bytes.
- The official 8K local suite completed five prompts and 320/320 exact output
  tokens. TTFT values are 828.7, 800.9, 844.3, 831.1, and 892.6 seconds
  (median 831.1 seconds). The same five inputs through four native
  whole-expert workers in exact mode remain 320/320 exact, consume 18,043
  remote responses with zero forbidden loads, and have median TTFT 1,101.7
  seconds. Their real payload total is 195,045,806,553 bytes, so local prefill
  is selected. A harness audit caught and stopped an attempted 32K diagnostic:
  the exact pinned config advertises `max_position_embeddings=4096`, so 32K is
  now recorded as `UNSUPPORTED_BY_MODEL` with null metrics. The explicitly
  validated downstream workspace still permits the mandatory 8K diagnostic;
  a larger allocation is not misreported as a model capability.
- Reuse-distance analysis covers 351,872 real selected-expert accesses and
  selects cache candidates around measured p50/p75/p90/p95 LRU stack-distance
  thresholds. Four real cache runs show zero nonresident hits and zero
  cache-hit page-fault proxies. The 48-slot candidate is fastest in its
  one-prompt sizing pass, but the Amdahl gate rejects a performance claim until
  repeated model-level validation. Native Windows pagefile-read and hard/soft
  fault attribution remain null with the API limitation recorded.
- Phase 10 consolidation streams every raw measurement and produces 700
  short-decode rows, 10 prefill rows, 9 concurrency groups, 6 mixed-service
  groups, 8 network rows, 105 native process-session paging summaries, and
  161,212 one-second peak memory samples. No missing metric is zero-filled.
  Measured decode, prefill, concurrent-decode, and mixed-service plans all
  select local execution. Planner-selected execution therefore reuses the
  measured local candidate rather than running a duplicate configuration.
- The conservative prefetch gate disables both prefill and decode prefetch at
  this checkpoint because timestamped inter-layer idle-gap telemetry is not
  present. It enforces a zero-byte safe lower bound and makes no hidden-latency
  claim. Shared memory, equal microsharding, fast reduction, cache policy, and
  the GPU kernel are all retained in the Amdahl table; no model-level
  optimization is accepted when correctness fails, the end-to-end result does
  not move within the bound, or a real-path counterfactual is absent.
- Consolidated Phase 10 evidence is under
  `artifacts/runs/experiment-010-correction-work/phase-10/analysis`. These rows
  complete the mandatory Level A decode, prefill, concurrency, mixed-service,
  shaping, residency, and phase-plan inputs. They do not satisfy the subsequent
  real-path failure, trust, complete-candidate planner, or simulator gates.

### Phase 11 real-token recovery and trust

- The native worker now accepts deterministic fault and corruption schedules
  keyed by prompt, token, layer, worker, and expert. The shared runtime owns the
  cache-drop operation. Worker termination, fixed and seeded-random delay,
  network outage, cache drop, and storage slowdown therefore execute in the
  native process that serves the actual in-`moe()` request.
- The coordinator implements fail-closed timeout, exact local fallback,
  alternate native replicas, sampled duplicate execution, and concurrent
  hedged duplicates. Replica ownership is explicit in the same plan and must
  be byte-identical to the primary bank. Local fallback temporarily enables
  the coordinator runtime only for the failed request and then restores the
  remote-load guard; it is not a silent retry.
- Result SHA-256 and identity checks, duplicate comparison, hidden challenge
  activations, reputation counters, and quarantine run before a contribution
  is admitted to the residual stream. The canonical `SWARMEX1` request gained
  a validated `challenge` semantic field; the framing and checksum contract is
  unchanged and the Python/C golden-vector suite passes.
- The official failure matrix contains eight real fixed-replay runs. All six
  required fault classes injected. Seven recoverable rows matched 32/32 tokens
  with zero forbidden loads; the explicit-outage row returned nonzero with no
  completed token sequence. Alternate, local, sampled-replication, bounded-wait,
  and hedged strategies are separately measured. Detection latency, reaction
  time, recovery latency, incremental network bytes, worker-execution wall-time
  proxy, p95 RPC impact, and throughput impact are derived from raw events.
- The official trust matrix injected 15 instances of each of eight corruption
  types (120 scheduled corruptions), plus five challenge-only plausible
  perturbations. Every identity, manifest/frame, zero, stale, precision, and
  plausible perturbation was detected; all recovered runs matched 32/32 tokens.
  Across 14,484 clean duplicate/challenge controls there were zero false
  positives. Detection p50/p95, verification bytes and time, p95 RPC impact,
  throughput impact, prevented corruptions, reputation, and quarantine are
  retained per class.
- The Phase 11 instrumentation binaries are
  `olmoe.exe` SHA-256
  `42c9779ef4f5eeabfe701840e6bb4284bf45de324b67cbf75fc18e46c4ee6938`
  and `olmoe_expert_worker.exe` SHA-256
  `29cf2afbc470605421c8eb92d1f193e641b8d4c0d9c448cd275fd82a53f761c3`.
  The unchanged CUDA DLL is
  `aacbb79fd4b41d4a05813c275675d06c275a8e9a5ce384bffd24e088ade488fb`.
  Patch and clean-source hashes will be regenerated after the complete
  downstream correction diff is frozen.
- Raw evidence is under
  `artifacts/runs/experiment-010-correction-work/phase-11/official`. Gate 11
  and Gate 12 are closed by these real-model rows. They do not substitute for
  complete measured planner candidates, held-out simulator validation, a
  current Level B run, or the corrected dense Kimi fixture.

### Phase 12 measured positive-utility planner

- `concurrent_decode` is now a first-class planner phase rather than being
  represented only by a standalone workload artifact. Planner phase/objective
  combinations are explicit: TTFT for prefill, decode throughput for decode,
  aggregate verified throughput for concurrent and mixed service, plus network
  and capacity objectives where meaningful.
- The required 14-candidate catalog maps only to real Level A rows or the
  explicit no-work `idle` control. Exact-response and coalesced-microshard
  entries are labelled measured aliases because response contract and request
  coalescing are orthogonal attributes of the same recorded direct/equal-shard
  runs; they are not counted as additional workload executions.
- Local rows are reused across network profiles only because their measured
  expert-network byte count is zero. Every other unmeasured phase/profile
  cross-product remains `NOT_APPLICABLE_OR_UNMEASURED`, with null metrics and a
  failed admission gate. No simulator or fixture value fills those cells.
- The planner evaluated 168 phase/variant/profile/objective selections and
  2,520 candidate rows. Sampling confidence intervals are reported for repeated
  workloads; single concurrent, mixed, and shaped-network groups are explicitly
  labelled as having no sampling interval.
- Nominal short decode selects `local_monolithic`. Shared memory, direct TCP,
  relayed TCP, equal, asymmetric, and coalesced microshards are all retained and
  rejected as negative marginal utility on this host. The capacity objective
  selects `capacity_isolated` through the measured four-worker exception.
  Conditional plans select alternate-worker recovery after a remote failure,
  verification-only work under nonzero corruption risk, and background-only
  work when a background queue exists.
- Maximum measured regret across eligible selections is `0.0`, below the 5%
  threshold. The complete candidate catalog, harmful-worker rejection, capacity
  selection, human explanation, and machine explanation therefore close Gate
  13. Evidence is under
  `artifacts/runs/experiment-010-correction-work/phase-12/planner`.

### Phase 13 simulator behavioral parity and held-out calibration

- The simulator replays real Level A routing, cache, transport, fallback, and
  failure traces before any timing fit is accepted. Across the eligible input
  set it reconciles 6,170,496 source expert selections, 6,574,976 observed
  worker selections (including verified duplicates), 16,581,626 cache hits,
  877,318 cache misses, 864,960 evictions, 313,461 messages, and
  222,771,682,725 raw payload bytes. Cache and message parity are exact for
  every measured configuration.
- The split holds out complete execution-strategy, network-profile, worker-set,
  and shard-layout combinations rather than repeated rows from the same
  configuration. Held-out median throughput error is 4.0726%, p95 latency error
  is 5.6605%, TTFT error is 3.4572%, ranking agreement is 100%, and measured
  planner regret is 0.0. These measurements close Gate 14. Raw evidence is
  under `artifacts/runs/experiment-010-correction-work/phase-13/simulator`.

### Phase 14 dense Kimi K3-shaped fixture

- The corrected deterministic fixture uses hidden dimension 3,584, routed
  width 3,072, top-16 routing, 92 routed layers, valid E2M1 nibbles, and bounded
  UE8M0 scales. No quantization group is all zero and the official timing path
  does not skip zero groups.
- The compiled Colibri-native MXFP4 path processes 1,519,386,624 groups and
  accounts for 97,240,743,936 operations across whole-expert, equal,
  asymmetric, coalesced, top-16, and 92-layer replay workloads. Relative errors
  are 2.23537e-7 for equal shards, 2.27421e-7 for asymmetric shards, and
  2.23537e-7 for coalesced shards. The evidence remains
  `SYNTHETIC_FIXTURE`; it is not described as full Kimi K3 inference.

### Phase 15 exact numerical closure and final pin audit

- A new observation-only `COLNUM1` trace is emitted inside the real Colibri
  `step()` path. It records native float32 post-MoE hidden states, exact selected
  routing weights, and pre-sampling logits without running local shadow experts
  or changing the sampled path.
- The whole-expert exact suite passed 50/50 prompts and 1,600/1,600 generated
  tokens. All 25,600 post-MoE boundaries, 25,600 routing-weight records, and
  1,600 pre-sampling logit records are byte-identical to the exact-container
  local oracle. It consumed 93,779 remote expert results with zero forbidden
  local loads and zero local fallbacks. Evidence is under
  `artifacts/runs/experiment-010-correction-work/phase-15/numeric-rpc-50`.
- The hybrid exact suite passed 10/10 prompts and 320/320 generated tokens.
  Its plan assigns 512 experts to two remote native banks and 512 experts to
  the coordinator. Real route traces contain 25,636 remote-owned and 25,052
  local-owned selected ranks; 9,439 remote results were consumed. All 5,120
  post-MoE boundaries, 5,120 router-weight records, and 320 logits are
  byte-identical, with zero forbidden loads and fallbacks. Evidence is under
  `artifacts/runs/experiment-010-correction-work/phase-15/hybrid-exact-10`.
- `per_worker_fast` remains a separate quality-bounded measurement: 100 real
  prompt rows ran and none satisfied exact token/canonical admission. Its
  4.385 tok/s median is retained for comparison but never substituted for the
  exact response contract.
- The native-int8 microshard suite passed 20/20 prompts and 640/640 generated
  tokens. All 10,240 post-MoE boundaries, 10,240 routing-weight records, and 640
  pre-sampling logit records are byte-identical. It consumed 20,480 shard
  results with zero forbidden local loads and zero fallbacks. Evidence is under
  `artifacts/runs/experiment-010-correction-work/phase-15/numeric-microshard-20`.
- The frozen downstream series still targets Colibri v1.4.0 commit
  `b085b48888a88d9a1c00b151a9979774b72cdbfd` and applies reproducibly. Patch
  0008 has SHA-256
  `f636436b73f51ea6c78a19c2d6c4cfc85535e63939ad969547287d396790ee2f`.
  The clean final build records source-tree SHA-256
  `ee840fd5d544dd4f5cfc3570425dcd8c7129ed5ccea3ed008484eadfd12ab2ed`;
  current release hashes are
  `8fe79e6315395470efb2d105ae2b4f3a17fb44cd751e40a906574fd28e618862`
  for `olmoe.exe`,
  `0d7be4d1282501f8ec0eac0c2377c96f9289084d7c2f648e05caf2e70dab7162`
  for `olmoe_expert_worker.exe`, and
  `7b2d2ed6d364416b3abb68af06212b4db3dffe6c820ecfb0a0556cdbcbeb972f`
  for `coli_cuda.dll`.

### Final completeness decision

- The exact current Level B model is not present. Historical Experiment 008
  rows are retained only as history and are never substituted for a current
  run. The correction bundle must therefore report `INCOMPLETE_FULL_RUN`, an
  overall `PARTIAL` verdict, and a failed current-Level-B gate even when every
  available Level A gate passes.
- The remaining multi-machine question is deliberately outside this
  single-host virtual-swarm claim: with two independently powered hosts,
  separate memory/storage/failure domains, and physical 10/100 GbE transport,
  do the same exact semantics retain positive utility once real NIC, DMA,
  synchronization, clock, and thermal effects replace loopback shaping?
