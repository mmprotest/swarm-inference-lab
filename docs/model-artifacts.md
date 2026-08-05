# Cluster-owned OLMoE model artifacts

Nodes do not need complete model snapshots. `swarm run` resolves one exact immutable source,
uses existing OLMoE inspection and contiguous partition identity checks, and builds a separate
content-addressed artifact for each planned stage.

## Artifact format version 1

The identity covers model ID, immutable model revision, tokenizer revision, adapter (`olmoe`),
source-file hashes, format version, exact stage assignment, dtype/quantization, and complete
manifest content hash. The artifact ID is the SHA-256 content identity.

An artifact contains only:

- required configuration and small metadata;
- safetensors owned by the stage's contiguous layer range;
- a rewritten safetensors index containing only owned tensor names;
- the versioned artifact manifest and immutable source evidence;
- tokenizer assets for stage 0 and the final stage; and
- final normalization/LM-head ownership on the final stage.

Tied embedding/LM-head ownership is made explicit. Validation fails if an unassigned layer tensor
enters the artifact or an assigned tensor is omitted. Source safetensors are read shard by shard
on CPU; the full model is never instantiated on an accelerator. Publication uses a temporary
directory and atomic replacement after complete verification.

## Acquisition and transfer

Source order is verified local cache, authenticated cluster peer with the exact content, then an
immutable upstream revision when downloads are permitted. Transfers use bounded chunks, a hash
for every chunk, a full content hash, strict relative paths, finite signed transfer leases,
bounded RPC payloads/timeouts, and restart-safe resume documents. Partial artifacts are never
loaded.

The canonical transactional deployment exposes:

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

Artifact preparation is part of the existing deployment transaction. Failure rolls back loaded
stages/routes and releases leases; it does not create a parallel deployment system.

## Cache and eviction

Each node has a configured storage budget and a deterministic LRU document. Pinned artifacts,
loaded-stage/deployment leases, and in-progress transfers cannot be evicted. Eviction and cache
documents are strict, versioned, and atomically replaced. Transfer progress survives agent
restart and exact verified content is reused across plans.

`model_path` remains a deprecated advanced override on low-level load requests. Normal product
runs use artifact identities.
