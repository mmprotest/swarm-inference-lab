# Node agent

Every cluster machine runs the same persistent node agent. Coordinator machines enable the
`coordinator` and `worker` roles; joined machines normally enable only `worker`. The agent owns
configuration and lifecycle but contains no model execution kernels.

## Canonical lifecycle reuse

The agent starts `CoordinatorRuntime` and `WorkerRuntime`, the same reusable classes used by the
advanced foreground CLI commands. Those classes own the existing coordinator core, RPC server,
worker registration, stage runtime, data transport, and bounded shutdown paths. Startup rollback
stops already-created resources; repeated start/stop calls are idempotent.

## Durable behavior

The agent preserves the node Ed25519 identity across restarts and explicit updates. It reloads
the pinned membership and reconnects without pairing again. Runtime states are:

- `ready`: worker registration and bidirectional reachability passed;
- `degraded`: still operating while refresh or bounded restart work is in progress;
- `blocked`: a correctable permission/connectivity condition prevents readiness;
- `stopped`: deliberately stopped; and
- `failed`: permanent configuration failure or exhausted bounded restart attempts.

Transient worker failures use capped exponential backoff and a finite attempt count. Permanent
identity, compatibility, backend, or configuration failures do not spin. Status is persisted
atomically and published to the coordinator.

## Configuration ownership

The agent selects and periodically re-evaluates:

- an operational CUDA, MPS, or CPU backend and benchmark-approved dtype;
- effective CPU/VRAM/unified-memory budget;
- storage budget and artifact LRU state;
- routed private interface and collision-free control, data, and probe ports;
- private firewall reachability;
- device capability benchmark records; and
- stale outgoing directed network measurements.

Reviewed overrides are persistent:

```text
swarm node configure --backend torch-cpu
swarm node configure --memory-limit 20GB
swarm node configure --memory-percent 60
swarm node configure --storage-limit 100GB
swarm node configure --control-endpoint 192.168.1.20:51001
swarm node configure --data-endpoint 192.168.1.20:51002
swarm node configure --interface Ethernet
```

Wildcard and loopback advertised addresses are rejected for normal physical nodes. Interface
overrides support VPN and multi-NIC hosts.

## Services and updates

Windows uses a current-user Task Scheduler task, Linux uses `systemd --user`, and macOS uses a
LaunchAgent. `--foreground` is the explicit fallback when user service management is unavailable.

Updates are never automatic:

```powershell
swarm node update --source-wheel .\swarm_inference_lab-0.1.1-py3-none-any.whl --yes
```

The updater validates the wheel archive, installs into a staged runtime, validates imports,
points the owned user service at it, and waits for a fresh ready state. It commits the versioned
active-runtime pointer only after startup; otherwise it restores the previous service/runtime.
Identity and membership directories are outside staged runtimes.
