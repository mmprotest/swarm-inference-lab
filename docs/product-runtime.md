# Product runtime operations

The first OLMoE product slice uses coordinator-managed persistent model stages,
direct worker-to-worker activations, bounded streaming, and restart-and-replay
request recovery. It does not implement transparent KV migration or
coordinator high availability.

## Durable coordinator state

Start the product coordinator with an explicit local state directory:

```powershell
swarm coordinator --config configs/product/olmoe-stage-ring.yaml `
  --state .swarm/coordinator
```

`.swarm/coordinator` is the documented default. It contains the persistent
coordinator Ed25519 key, coordinator metadata, known worker identity records,
plans, deployment status and latest topology generation, per-request recovery
records, accepted-token replay logs, canonical product events, and durable
audit events. Writes that determine recovery state use temporary files plus an
atomic replace; accepted-token and audit logs are flushed to disk.

On restart, the coordinator validates request records against their replay
logs. It preserves deployment evidence for inspection but does not treat old
connections or routes as live. Workers must re-register. A request with a
verified prompt and greedy accepted-token prefix is marked `recoverable`;
inconsistent evidence fails closed. No token is emitted automatically from
restart evidence.

Back up this directory as one unit and restrict access to it: the coordinator
private key establishes route authority.

## Identity bootstrap

The coordinator prints its persistent public-key fingerprint at startup. Every
product stage worker must pin that value explicitly:

```powershell
swarm worker ... --stage-runtime `
  --trusted-coordinator-fingerprint <64-hex-fingerprint>
```

The coordinator can additionally require worker fingerprints from
`trusted_worker_fingerprints` in the product configuration. A known worker ID
cannot silently change its persisted public key. Signed route leases bind all
worker identities, endpoints, revisions, and assignments for one finite-lived
route generation.

## Live inspection and cancellation

The live product does not require experiment evidence bundles for inspection:

```powershell
swarm status --coordinator 127.0.0.1:50051
swarm workers --coordinator 127.0.0.1:50051
swarm topology --coordinator 127.0.0.1:50051
swarm sessions --coordinator 127.0.0.1:50051
swarm cancel --coordinator 127.0.0.1:50051 --request-id <id>
```

Each command supports `--json`. Status covers coordinator identity and uptime,
worker health/endpoints/loaded stages, deployment and route generations,
sessions and token positions, KV memory, queues, throughput and latency,
recoveries, errors, and reservations.

Cancellation is idempotent. It stops token acceptance, sends bounded cancel
requests to every stage in the active generation, releases only that session's
KV state, and leaves shared model stages resident. A client stream disconnect
uses the same cleanup path. Client cancellation is not recorded as a worker
reliability failure.

## Restart-and-replay recovery

For a heartbeat expiry, control-RPC failure, stage-ring closure, execution
error, route-generation mismatch, token-publication timeout, or worker process
termination, the coordinator:

1. disables output acceptance from the failed generation;
2. marks the request recovering and emits a recovery event;
3. selects exact eligible replacement workers and loads missing stages;
4. installs and peer-verifies a higher signed route generation;
5. opens a fresh session on all stages;
6. replays the original prompt and full accepted greedy token history;
7. compares every ring result and stage-zero publication with durable history;
8. suppresses all replay token events; and
9. resumes at the first unaccepted token only after the prefix verifies.

The first divergence fails safely. The runtime does not silently restart from
scratch, emit duplicate token events, transfer distributed KV checkpoints, or
route intermediate activations through the coordinator.

## Network security boundary

Coordinator route leases and direct peer handshakes are authenticated with
Ed25519. Stage-ring frames retain strict sequence and checksum validation.
Payload confidentiality is not provided: stage-ring TCP traffic is not
encrypted, and an unkeyed SHA-256 frame checksum is not malicious tamper
resistance. Deploy only on a trusted LAN or private network. Untrusted Internet
operation remains unsupported.
