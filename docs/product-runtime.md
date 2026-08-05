# Product runtime operations

The OLMoE product uses coordinator-managed persistent stages, direct worker-to-worker
activations, bounded streaming, transactional deployment, and restart-and-replay recovery. The
cluster product configures and owns those canonical components; it does not wrap or duplicate a
second inference runtime.

## Normal lifecycle

```text
swarm cluster create --name <name>
swarm node join <pairing-uri>
swarm cluster status
swarm run <olmoe-model> --revision <immutable-commit> \
  --tokenizer-revision <immutable-identity> --mode speed --prompt <text>
```

The node agent calls `CoordinatorRuntime` and `WorkerRuntime`, which own the existing coordinator
core, RPC server, registration, stage execution, deployment, transport, and recovery paths.
Starts/stops are idempotent, partial startup rolls back, and shutdown has a hard bound.

`swarm run` performs immutable source validation, capability/memory refresh, stale directed-link
collection, common dtype selection, bounded N-stage planning, stage-artifact preparation and
transfer, canonical transactional deployment, route/peer verification, and streaming submission.

## State layout and versions

Cluster persistent documents use schema version 1 and document version 1. Cluster RPC messages
use schema version 1; the product protocol is major 1, minor 0; stage artifact format version is
1. Unknown fields are rejected.

The state root separates `security/`, mutable `runtime/`, `logs/`, content-addressed `artifacts/`,
and partial `downloads/`. Security/recovery-critical documents use temporary files and atomic
replacement. Private identity permissions are restricted where supported. A valid legacy
`.swarm/coordinator` identity is adopted and never rotated automatically.

Default roots are `%LOCALAPPDATA%\SwarmInference\` on Windows,
`${XDG_STATE_HOME:-~/.local/state}/swarm-inference/` on Linux, and
`~/Library/Application Support/SwarmInference/` on macOS. Explicit roots support tests and
repository-local development.

## Planning

Automatic planning includes local monolithic execution and stage counts up to:

```text
min(model layers, healthy eligible workers, configured maximum_stage_count)
```

The deterministic beam search is bounded by candidate-worker, stage-count, and beam-width
configuration. A state tracks next layer, selected workers, previous worker, compute/network
cost, memory feasibility, queue/load, reliability, and objective score. Only measured, fresh
directed links are used for automatic distributed paths. Stable lexicographic worker/stage
tie-breaks make repeated inputs deterministic.

Speed mode compares with the fastest local baseline and prefers fewer equivalent boundaries.
Capacity mode requires collective fit and then headroom/replacement capacity. Balanced mode
reports explicit throughput, headroom, reliability, and participation components. Every healthy
node receives a utility/inclusion or exclusion record; pairing does not require participation.

## Deployment and recovery

Artifact preparation is a phase of the existing deployment manager. It reserves, prepares,
transfers, verifies, loads, verifies ownership, installs routes, verifies peers, and only then
publishes ready. Failure uses the existing rollback path.

For runtime failure, the coordinator disables the failed generation, selects trusted compatible
replacements, installs a higher signed generation, opens new sessions, and replays the prompt
plus accepted deterministic tokens. It rejects the first divergence and suppresses replay
publications. There is no KV checkpoint transfer, transparent migration, or seamless coordinator
failover.

## Advanced diagnostics

Low-level commands remain supported:

```text
swarm coordinator
swarm worker
swarm identity create|show|fingerprint|trust|untrust|list-trusted
swarm model inspect|plan|deploy|unload
swarm submit
swarm status
swarm workers
swarm topology
swarm sessions
swarm cancel
```

Manual trust continues to work. Low-level planning keeps exact-snapshot eligibility by default;
the high-level run path explicitly permits cluster-owned stage-artifact provisioning.
Cancellation remains idempotent and releases only the affected session KV state.

## Observability and security

Strict status includes cluster/node identity, OS/architecture/backend validation, versions,
service state, endpoints/reachability, dtype/memory, directed-link freshness, artifact
cache/transfers, loaded stages/role, inclusion reason, and last categorized error. Structured
events cover all cluster, network, artifact, plan, deployment, and update transitions. Secrets
and prompts are omitted.

Signed membership requests, route leases, artifact transfer leases, and peer handshakes
authenticate authority. Stage-ring activation/token payloads are plaintext TCP. Deploy only on a
trusted LAN or private network. See [security boundary](security-boundary.md).
