# Architecture-neutral model artifacts

Nodes do not need complete model snapshots. `swarm run` resolves one exact immutable source,
inspects it through an architecture adapter, and builds content-addressed artifacts for the
selected execution plan. Generic artifact code owns identity, checksums, caching, transfer,
leases, verification, persistence, and eviction. It never interprets a family name or assumes a
tensor spelling.

Architecture adapters own tensor meaning: embeddings, attention and router tensors, layer
groups, routed/shared/always-on experts, final normalization, output heads, tied weights, tensor
slices, and reduction semantics. Execution engines may add engine-specific containers, but they
remain bound to the same immutable model and architecture profile.

## Artifact identity

The identity covers model ID, immutable model and tokenizer revisions, model fingerprint,
architecture adapter, engine, artifact kind/format, source-file hashes, exact stage or expert
ownership, dtype/quantization, total bytes, and complete manifest content. The artifact ID is a
SHA-256 content identity.

A native-stage artifact contains only:

- required configuration and small metadata;
- Safetensors owned by the stage's contiguous layer range;
- a rewritten index containing only owned tensor names;
- the versioned manifest and immutable source evidence;
- tokenizer assets for stages that tokenize or decode; and
- adapter-proved embedding, final-normalization, and output-head ownership.

Tied output weights are derived by the architecture adapter and recorded as an explicit alias;
generic code does not assume `lm_head` or a particular embedding name. Source Safetensors are
read shard by shard on CPU and the complete model is never instantiated merely to construct an
artifact. Publication uses a temporary directory and atomic replacement after verification.

When a checkpoint contains `*.safetensors.index.json`, that index is the authoritative immutable
tensor-to-shard map. Resolution validates every repository-relative target, records the canonical
mapping digest and tensor/shard counts, and includes only referenced shards. A missing shard or an
escaping/malformed target is fatal; an unreferenced Safetensors file is not guessed to be a model
weight. Unindexed local checkpoints are header-inspected and duplicate tensor names fail closed.

## Routed computation inventory

Experts and related routed units use one `ExpertDescriptor` representation containing layer,
unit index/type, tensor groups, parameter/memory size, input/output shapes, routing metadata,
and adapter-proved shard semantics. It represents routed, shared, always-on, latent, grouped,
and fused-expert-axis layouts without teaching caches, transfers, or planners Qwen/Kimi/GLM/
DeepSeek tensor names.

Generic residency retains the promoted bounded LRU, hot-expert cache, content-hash verification,
movement accounting, and storage-backed loading mechanisms. The older exact-byte experiment
container remains an isolated compatibility fixture; new adapters use the descriptor boundary.

## Microshards

`RoutedComputationMicroshardDescriptor` records matched logical tensor slices, native
quantization metadata, shard dimension, required accumulator, and deterministic reduction.
Architecture adapters prove which axes may be sliced and whether reconstruction is concatenate,
gather, sum, or all-reduce-sum. Generic microshard orchestration owns worker assignment,
transport, exact union validation, failure, stable reduction order, and telemetry.

The compatibility ABI for three-projection SiLU experts remains available for Experiment 010
regression evidence. It is not the universal tensor model.

## Acquisition, transfer, and lifecycle

Source order is verified local cache, authenticated cluster peer with exact content, then an
immutable upstream revision when downloads are permitted. Transfers use bounded chunks, a hash
for every chunk, a full content hash, strict relative paths, finite signed leases, bounded RPC
payloads/timeouts, and restart-safe resume documents. Partial artifacts are never loaded.

The transactional deployment phases are:

```text
reserving
preparing-artifacts
transferring-artifacts
verifying-artifacts
loading
verifying-loads
installing-routes
verifying-peers
ready
```

Failure rolls back loaded stages/routes and releases leases. Each node has a bounded,
deterministic LRU cache. Pinned artifacts, active deployment/stage/expert leases, and in-progress
transfers cannot be evicted. Cache and transfer state are strict, versioned, and atomically
replaced.

`model_path` remains a deprecated advanced override on low-level load requests. Normal product
runs use immutable artifact identities.
