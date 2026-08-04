# Experiment 011 design: communication-avoiding exact decode

## Scope and frozen Experiment 010 provenance

Experiment 011 adds a new execution family, `stage_ring_exact`, without replacing or
modifying the Experiment 010 whole-expert, microshard, local Colibri, or evidence-reader
paths. The causal baseline remains the existing Experiment 010 whole-expert RPC workload.

The pre-change audit resolved the following sources of truth:

- Repository commit at design capture: `691fcf72c422b271045f6e213cd76e64e16643a9`.
- The worktree was already dirty with Experiment 008/010 Level-B work. Experiment 011 is
  isolated in new files so those changes remain owned by their author.
- Final Experiment 010 bundle:
  `artifacts/runs/experiment-010-correction-final/experiment_010`.
- Final Experiment 010 ZIP:
  `artifacts/runs/experiment-010-correction-final.zip`.
- Model: `allenai/OLMoE-1B-7B-0125-Instruct`, source weights at
  `artifacts/models/colibri/source-b89a7c4bc24f`, 16 layers, hidden size 2048,
  64 experts, top-8 routing, BF16 model dtype.
- Experiment 010 model fingerprint:
  `sha256:bad0c225e9bc03275cb12c6606dac4358bcdc188ca701f654ea9672a6cecc35e`.
- Experiment 010/Colibri revision:
  `pinned-b085b48888a88d9a1c00b151a9979774b72cdbfd`.
- Tokenizer: the `tokenizer.json` in the same source model directory. Experiment 011
  records its SHA-256 and treats that hash as the tokenizer revision.
- Network workload reference:
  `artifacts/runs/experiment-010-correction-work/phase-6/local-correctness-references/code-01/reference.json`.
  It contains the prompt `Write a Python function that merges two sorted lists.`, 11
  prompt token IDs, and 32 reference generated token IDs.
- Network profile definitions:
  `artifacts/runs/experiment-010-correction-final/experiment_010/transport_profiles.json`.
  The runtime implementation is
  `src/swarm_inference/experiments/experiment_010/transport.py`.
- Experiment 010 network rows:
  `artifacts/runs/experiment-010-correction-work/phase-10/final-binary/network-profiles/network_profile_results.csv`.
  There is one measured `code-01` row per profile, no distinct warm-up row, and medians
  therefore equal the single measured value.
- Existing local oracle capture:
  `build_local_reference_suite` in `experiment_010/colibri_token_path.py`.
- Existing exact whole-expert path:
  `NativeLevelASession`, `measure_reference`, and `run_network_profile_matrix` in
  `experiment_010/colibri_workloads.py`, using `build/colibri/bin/olmoe.exe`,
  `build/colibri/bin/olmoe_expert_worker.exe`, and four banks under
  `artifacts/runs/experiment-010-correction-work/phase-6/banks`.
- Numerical boundary evidence:
  `colibri_rpc_boundary_errors.csv`, `correctness_results.json`, and
  `token_comparisons.json` in the final Experiment 010 bundle, sourced from the Phase 15
  numeric runs listed in `artifact_source_map.json`.

The local Transformers eager-attention path was checked against `code-01` before design
freeze. With the source BF16 weights and greedy decoding, all 32 generated IDs match the
Experiment 010 Colibri reference exactly. This permits one token oracle to cover both the
archived Colibri baseline and the new contiguous-stage implementation.

The pre-change repository suite passed 532 tests with 7 explicitly skipped tests. An
isolated Experiment 010 Quick smoke also completed successfully and correctly reported a
non-closure `PARTIAL` quick verdict.

## Current decode execution graph

The exact whole-expert path keeps the model coordinator between every routed expert
request and response:

```text
client
  |
  v
Colibri coordinator
  | embedding / attention / router / reduction / layer continuation
  |---- request routed expert group ----> relay/shaper ----> expert worker
  |<--- exact expert result ------------- relay/shaper <---- expert worker
  | repeat for routed groups and all 16 layers
  v
final norm / LM head / greedy sample
  |
  v
client
```

The archived network row records 1,850 completed expert RPC operations for 32 generated
tokens, or 57.8125 completed RPC operations per generated token when the row is divided
by its declared generated-token count. The experiment objective cites approximately
58.4 remote operations per token. These are message-operation observations, not yet a
critical-path proof. Experiment 011 reconstructs serial waits from timestamped dependency
edges rather than treating topology or message totals as the answer.

Each old RPC returns to the coordinator before the coordinator can finish the current MoE
layer. Consequently the coordinator is a blocking dependency throughout decode. The
fresh same-run baseline retains this behavior unchanged.

## Proposed contiguous-stage ring

For `N` stages, the 16 transformer layers are divided into `N` disjoint contiguous
ranges. Data-plane connections are persistent and direct:

```text
                asynchronous TOKEN_PUBLICATION
                         +--------> coordinator/client
                         |
stage 0 ----activation--> stage 1 ----activation--> ... ----activation--> stage N-1
   ^                                                                      |
   +----------------------- TOKEN_RESULT ---------------------------------+
```

Stage zero owns token embeddings, the first layer range, its per-layer KV cache, and the
accepted input-token history. Intermediate stages own only their layer ranges and the KV
caches for those ranges. The final stage owns the last layer range, its KV cache, final
RMS normalisation, LM head, and deterministic `argmax` sampler.

After `OPEN_SESSION`, the coordinator performs health monitoring and receives published
tokens over a separate connection. It is not an activation hop and is not awaited before
the next ring step. A bounded publication queue makes client backpressure observable
without silently moving it onto the model critical path.

The expected steady-state serial waits are exactly the number of stage boundaries,
including the final token return:

| Topology | Activation boundaries | Token return | Maximum serial waits/token |
|---|---:|---:|---:|
| 2 stages | 1 | 1 | 2 |
| 4 stages | 3 | 1 | 4 |
| 8 stages | 7 | 1 | 8 |

These are design limits. Gate evidence uses the trace DAG to verify the observed count.

## Layer and state ownership

Every plan lists half-open layer ranges `[start_layer, end_layer)`. Plan validation
requires that ranges start at zero, meet without gaps or overlap, end at the model's
declared layer count, and assign every layer exactly once.

Workers construct only the modules they own and load only matching tensors from the
source safetensor index:

- stage 0: `model.embed_tokens.*` and `model.layers.<owned>.*`;
- intermediate stage: `model.layers.<owned>.*`;
- final stage: `model.layers.<owned>.*`, `model.norm.*`, and `lm_head.*`.

Instantiation uses meta-device modules followed by assignment of the selected source
tensors, avoiding a transient full-model allocation. The ownership record includes every
loaded key, source shard, logical weight bytes, resident CUDA bytes, process ID, device,
and layer range. The coordinator imports configuration and tokenizer metadata but does
not instantiate stage-owned model modules. A full-model load or monolithic fallback emits
`FALLBACK` telemetry and invalidates that performance row.

Each stage keeps a separate cache object per session. Cache slots correspond only to its
owned layers. KV bytes are derived from actual key/value tensor storage after prefill and
after the final accepted token; no complete KV tensor crosses a stage boundary during
normal decode.

## Prefill and decode lifecycle

`OPEN_SESSION` establishes the expected prompt length, generation limit, deterministic
settings, and initial sequence number on every stage. Stage zero accepts prompt token IDs
once. `PREFILL` embeds the complete prompt and executes its range. The resulting BF16
activation traverses the ring once; every stage fills only its local KV slots. The final
stage samples the first generated token and sends it directly to stage zero.

For ordinary decode, stage zero embeds the returned token, each stage consumes exactly
one new position and updates its local KV cache, and the final stage returns the next
token. Position ID, cache position, sequence length, session ID, and token position are
validated at every hop.

Session close zeroes references to cache tensors, synchronises outstanding CUDA work,
records released bytes, and removes sequence-validation state. Cancellation is idempotent
and cannot release another session's cache.

## Wire protocol

The canonical transport is persistent full-duplex TCP. Pickle is forbidden. Each frame
uses a fixed little-endian prefix followed by canonical UTF-8 metadata and a contiguous
payload:

```text
magic[8] | version:u16 | op:u16 | flags:u32 | metadata_len:u32 |
payload_len:u64 | sequence:u64 | token_position:i64 | header_crc32:u32 |
metadata[metadata_len] | payload[payload_len] | payload_sha256[32]
```

Metadata contains the run, request, session, topology, model and tokenizer revisions;
source/destination stage; owned layer range; tensor dtype, shape and byte order;
compression mode; message type; and status. Tensor payloads are C-contiguous and retain
their original dtype and little-endian representation. `send_all` and `recv_exact`
explicitly support partial socket writes and reads. Connections reuse preallocated receive
buffers where capacity permits and use bounded outbound queues for backpressure.

Protocol operations are `HELLO`, `CAPABILITIES`, `LOAD_STAGE`, `OPEN_SESSION`, `PREFILL`,
`DECODE`, `VERIFY_CANDIDATES`, `TOKEN_RESULT`, `SESSION_CHECKPOINT`, `CLOSE_SESSION`,
`CANCEL_SESSION`, `HEALTH`, and `ERROR`. Control operations and data operations have
separate flags and validation state. Receivers reject checksum failure, wrong model or
tokenizer revision, wrong topology/stage/range, unknown session, duplicate or stale
sequence, and a future sequence that skips required work.

Network shaping reuses the exact resolved Experiment 010 `NetworkShapeProfile` objects
and `NetworkShaper` implementation. Shaping occurs on the actual encoded frame before the
socket send completes. No decision branches on a human profile label.

## Lossless activation transport

`none` sends the exact tensor bytes. `byte_shuffle_fast_codec` transposes fixed-width
element bytes into byte lanes and applies the repository-available low-latency codec
selected at runtime. The implementation initially supports the pinned Python zlib
runtime at level 1 when no optional faster codec is installed; the exact Python and zlib
versions are recorded. Decoding reverses the lane transform and verifies SHA-256 over the
restored uncompressed bytes.

The adaptive decision compares predicted uncompressed transfer time with measured encode,
decode, compressed transfer, RTT, bandwidth, and queue-delay costs. It never uses the
profile name. Compression is disabled when measured utility is non-positive.

## Exact speculative verification

`DraftProvider` returns proposals without access to future target tokens. The mandatory
`PromptLookupDraftProvider` searches repeated suffixes in the prompt plus accepted output
history. Candidate depths are 2, 4, and 8.

A candidate block crosses the complete stage pipeline once. Each stage checkpoints only
its local cache before verification. Target logits at every candidate position determine
the longest exact greedy prefix. When all candidates match, the speculative cache state
is committed. On rejection, stages restore the checkpoint and deterministically replay
only the inputs required to commit the accepted prefix and the authoritative mismatch
token. Oracle proposals are permitted only in rows marked `MECHANISM_ONLY`, which the
evidence validator excludes from measured claims.

## Continuous batching

Every stage uses a bounded scheduler queue. It groups ready messages with compatible
operation, tensor shape, position, and cache length for at most the configured batch wait.
Compatible session tensors and per-layer KV tensors are concatenated on the batch axis,
computed once, then split back into independent session caches. Incompatible work remains
independent and cannot block beyond the bounded wait. Cancellation removes queued work
and local cache state by session ID.

## Failure behavior and recovery

Data-plane or process failure closes the affected session and emits `ERROR` and
`WORKER_FAILURE`; no token is published after an unverified boundary. The coordinator may
restart the topology from the latest accepted token history. Recovery opens fresh stage
caches and deterministically replays the prompt and accepted continuation. It compares the
post-recovery continuation with the local reference and records replayed work and latency.
Universal request duplication is not the default.

The smoke matrix injects stage termination, disconnect, duplicate frame, stale token
position, wrong model revision, checksum corruption, final-stage failure before return,
and stage-zero failure after acceptance. Unsupported recovery is an explicit failed or
resource-infeasible row, never a silent local fallback.

## Exactness strategy

The authoritative token oracle is the Experiment 010 `code-01` reference. A local
monolithic Transformers eager-attention reference using the same source weights also
captures per-layer boundaries, final hidden states, and logits. Distributed execution
uses the same module implementations, BF16 layer arithmetic, eager attention, position
construction, cache semantics, and `argmax`. TF32 is disabled and deterministic seeds and
environment settings are fixed.

Boundary tensors cross the socket as their original BF16 bytes. For every boundary that
the local reference exposes, evidence compares SHA-256 and raw bytes. Canonical diagnostic
comparisons cast both sides to FP32 after capture and require max absolute error and
relative L2 error of zero. Any mismatch records the prompt, token, stage, boundary, paths,
errors, token IDs, and reproduction command; failures are never averaged.

## Partition planning

Equal partitioning divides layers contiguously with a maximum one-layer count difference.
Balanced partitioning profiles each layer (or a declared repeated family) for CUDA time,
weight bytes, KV growth, temporary-memory peak, and activation bytes. A dynamic program
selects contiguous cut points that minimise maximum predicted stage time subject to memory
limits, with secondary penalties for imbalance and communication. Plans are derived from
model metadata, never hard-coded for OLMoE, and are written to `stage_plan.json`.

The measured planner compares local monolithic, expert RPC, equal/balanced 2/4/8-stage
rings, compression, and exact speculation. Its objective includes stage compute, memory,
bandwidth, RTT, jitter, loss, queue depth, compression measurements, draft acceptance,
serial boundaries, bytes, imbalance, rejected work, and reliability. Profile names are
provenance labels only.

## Measurement and statistical plan

The canonical network order is loopback, 100G fabric, 10 GbE, 2.5 GbE, 1 GbE, Wi-Fi,
regional WAN, and global WAN. Exact profile objects are loaded from the Experiment 010
manifest and hashed. The Experiment 010 network workload has one measured row per profile;
Experiment 011 preserves that count and records the limitation. With one independent run,
the required run-level bootstrap interval is degenerate and descriptive rather than a
credible estimate of between-run uncertainty. The experiment still applies the fixed
classification rule exactly (`IMPROVED` above zero, `REGRESSED` below zero, otherwise
`INCONCLUSIVE`) and labels the single-repeat statistical limitation in every report.

Strategy order is deterministically rotated by profile. Raw rows retain prompt, run,
exactness, telemetry, system/GPU samples, and provenance. Summaries use medians as the
headline and also record mean, standard deviation, p25, p75, p95, and bootstrap intervals
when estimable. The causal comparison is the fresh same-run expert RPC versus the best
valid exact stage strategy. Archived Experiment 010 values appear only in the continuity
series.

A serial network wait is a socket dependency on the token-critical trace path that must
complete before the next required model-compute node can begin. The analyzer constructs a
per-token DAG from message send/receive identity, queue/compute intervals, stage-output
edges, and the final token return, then counts only network nodes on the longest required
dependency path.

## Expected risks

- Independent CUDA processes on one WDDM GPU may introduce context-switch overhead and
  make loopback slower even when WAN scaling improves.
- Eight CUDA contexts may be resource-infeasible despite constant logical weight bytes.
- The one-run Experiment 010 workload cannot support run-level statistical significance;
  this is preserved rather than silently increasing the baseline repetition count.
- Python zlib may cost more than it saves for 4 KiB BF16 decode activations; correct global
  disablement is an acceptable compression result.
- Prompt lookup may have near-zero acceptance on `code-01`; the planner must disable it
  rather than claim an oracle benefit.
- Loopback network shaping on one host cannot reproduce NIC DMA, physical congestion,
  independent clocks, host failures, or cross-machine GPU/CPU overlap.

## Gate-to-evidence map

| Gate | Primary generated evidence |
|---|---|
| 1 Baseline integrity | `baseline/`, `network_profiles_manifest.json`, `model_identity.json`, fresh baseline raw rows, archived provenance |
| 2 Stage ownership | `stage_plans/*/stage_plan.json`, `exactness/stage_ownership.json`, worker load manifests and process records |
| 3 Exact execution | `exactness/exactness_results.json`, boundary tensor files/hashes, token/logit comparisons |
| 4 Coordinator removal | `traces/*.ndjson`, `critical_path_summary.csv`, dependency-DAG audit |
| 5 Serial waits | `critical_path_summary.csv` reconstructed from trace edges |
| 6 Message reduction | `network_profile_results.csv`, fresh baseline and selected-stage trace summaries |
| 7 Payload reduction | `network_profile_results.csv`, frame payload/wire counters |
| 8 Regional WAN | regional raw rows, bootstrap record, `network_profile_summary.csv` |
| 9 Global WAN | global raw rows, bootstrap record, `network_profile_summary.csv` |
| 10 Curve flattening | `network_profile_summary.csv`, whole-curve metrics, chart families 06/06b/06c/06d |
| 11 Compression | `compression/compression_results.json`, `compression_summary.csv`, byte-identity tests |
| 12 Speculation | `speculation/speculation_results.json`, `speculation_summary.csv`, rollback traces |
| 13 Concurrent goodput | `concurrency/`, `concurrency_summary.csv`, exact per-session sequences and cancellation row |
| 14 Regression safety | `logs/preflight-tests.log`, `logs/postchange-tests.log`, Experiment 010 smoke records |
| 15 Evidence completeness | `manifest.json`, `source_identity.json`, `binary_identity.json`, SHA-256 inventory, artifact validator output |

Failure smoke evidence is in `failures/` and `failure_summary.csv`. Planner decisions are
stored once per profile under `network/planner/`. Chart inspection dimensions and hashes
are part of `manifest.json`. `verdict.json` applies the thresholds declared in the
experiment request without post-result adjustment.
