# Swarm Inference Lab

Swarm Inference Lab is a self-configuring, cross-platform cluster runtime for measured
heterogeneous inference. The first product model family is OLMoE. A coordinator manages
persistent workers and deployment transactions, while hidden states travel directly over an
ordered stage ring instead of being relayed through the coordinator.

The supported security boundary is a trusted LAN or private network. Pairing authenticates
onboarding and pins identities, but inference data-plane payloads are not encrypted. Do not
expose the runtime to the public Internet or send sensitive prompts through untrusted nodes.

## Current status — August 2026

Implemented product behavior includes:

- durable single-use X25519/HKDF/AES-GCM pairing backed by Ed25519 node identities;
- a persistent user-scoped node agent using the canonical coordinator and worker runtimes;
- operational CPU, CUDA, and MPS probing with automatic backend, dtype, memory, endpoint, and
  port selection;
- authenticated directed network measurements with freshness evidence;
- deterministic bounded N-stage planning without factorial worker permutations;
- speed, capacity, and balanced objectives where joined nodes may remain idle;
- stage-owned, content-addressed OLMoE artifacts with bounded resumable transfers;
- transactional deployment, direct stage-ring execution, streaming publication, and
  restart-and-replay recovery; and
- strict versioned state, JSON/NDJSON automation, wheel installation, and cross-platform CI.

Implementation support is separate from validation evidence. A clean machine reports software
and physical validation as `not-run`; OS/architecture detection, hardware visibility, imports,
and tensor probes never manufacture validation. Physical multi-machine product performance has
not been proven in this repository state. Loopback, simulation, and CI evidence do not count as
physical validation.

## Install without Git

Build artifacts are installed with the platform installer. A participating node does not need
the repository, Python, or `uv` in advance.

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 `
  -SourceWheel .\swarm_inference_lab-0.1.0-py3-none-any.whl -Json
```

Linux or macOS:

```bash
sh ./install.sh --source-wheel ./swarm_inference_lab-0.1.0-py3-none-any.whl --json
```

The installer locates or installs `uv`, installs a managed supported Python, tests the selected
device through `swarm node doctor`, and installs the wheel as an isolated tool. It does not create
an unpaired service: installer output reports
`deferred-until-cluster-create-or-join`. Cluster creation or joining installs and starts the
persistent user service by default; `--foreground` opts out. See
[platform support](docs/platform-support.md).

## Cluster quick start

Use routable private addresses. The commands select identities, ports, backend, dtype, memory,
and storage automatically.

1. Install the wheel/tool on both machines.
2. Create the cluster on the PC:

```powershell
swarm cluster create --name villani-home --yes --json `
  --pairing-output C:\Protected\villani-home-invitation.uri
```

3. Transfer the protected single-use invitation file through a secure channel. JSON contains
   only the path and redacted metadata, never the URI.
4. Read the URI locally on the laptop and join:

```powershell
swarm node join "<pairing-uri>" --yes --json
```

5. Back on the PC, inspect cluster status:

```powershell
swarm cluster status --json
```

6. Run the physical acceptance command from the runbook. A normal inference command is:

```powershell

swarm run allenai/OLMoE-1B-7B-0125-Instruct `
  --revision b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e `
  --tokenizer-revision sha256:d1e645ebd850d79567e531a3c103ac575d8e9cf45fa941420afc584b293438ea `
  --mode speed `
  --prompt "Explain distributed inference."
```

`speed` compares distributed candidates with the fastest feasible local path and can leave a
slow node idle. `capacity` uses collectively available memory when a faster local path cannot
fit. `balanced` applies explicit throughput, headroom, reliability, and useful-participation
weights. Use `--dry-run --explain-plan` to inspect the decision without deploying.

See [cluster quick start](docs/cluster-quickstart.md), [pairing](docs/pairing.md), and
[troubleshooting](docs/cluster-troubleshooting.md).

## Product architecture

```mermaid
flowchart LR
    A[Persistent node agents] --> C[Coordinator control plane]
    C --> P[Bounded N-stage planner]
    C --> D[Transactional deployment]
    D --> S0[Stage 0]
    D --> S1[Stage 1]
    D --> SN[Stage N]
    S0 -->|plaintext activation on trusted LAN| S1
    S1 -->|plaintext activation| SN
    SN -->|token result / next step| S0
    SN -. ordered publication .-> C
```

The coordinator is absent from steady-state hidden-state forwarding. Workers and loaded stages
persist across requests. Multiple sessions use bounded interleaving; this is not continuous
tensor batching. Recovery retires a failed route generation, installs a higher signed
generation, and replays the prompt plus accepted deterministic token prefix. It is
restart-and-replay, not seamless failover or KV migration.

Stage artifacts contain only tensors assigned to the stage, plus required small metadata.
Stage 0 receives tokenizer assets; the final stage receives final normalization, LM-head, and
decode/publication assets. Nodes do not need complete model snapshots. See
[model artifacts](docs/model-artifacts.md).

## Machine-readable operation

All new commands support `--json`; progress-producing commands support `--ndjson`. `cluster
create --json` and `cluster pair --json` each emit exactly one JSON document. They never include
the pairing URI: automation receives the complete invitation only through an atomically written,
owner-protected `--pairing-output` file (or a default secret path beneath cluster state). Pairing
secrets, private keys, raw authentication proofs, session keys, and prompt contents are never
included in machine-readable output. Machine-readable or non-interactive administrative
mutations require `--yes` and fail with the permission exit category before mutation otherwise.

Useful commands include:

```text
swarm cluster status --json
swarm node status --json
swarm node doctor --json
swarm run ... --dry-run --explain-plan --json
swarm run ... --ndjson
```

## Advanced operations

The low-level product commands remain available for diagnostics and acceptance work:

```text
swarm coordinator
swarm worker
swarm identity ...
swarm model inspect
swarm model plan
swarm model deploy
swarm model unload
swarm submit
swarm status
swarm workers
swarm topology
swarm sessions
swarm cancel
```

Manual identities and trust are compatibility features, not the normal bootstrap path. Their
full contracts are documented in [product runtime operations](docs/product-runtime.md).

## Validation

The supported repository workflow is:

```powershell
uv sync --python 3.11 --extra cpu --extra dev
uv run --python 3.11 ruff format --check src tests scripts
uv run --python 3.11 ruff check src tests scripts
uv run --python 3.11 mypy
uv run --python 3.11 pytest tests/unit -q
uv run --python 3.11 pytest tests/integration tests/failure -m "not gpu" -q
uv build
uv run --python 3.11 python scripts/run_productization_acceptance.py run `
  --run-repeatability --repeatability-timeout-seconds 600 `
  --output artifacts/acceptance
```

Acceptance records `PASS`, `FAIL`, `SKIP`, and `NOT_RUN`. Missing physical configurations are
`NOT_RUN`; a skipped GPU gate never becomes a pass. The physical RTX 5090 plus Windows CPU-node
procedure is in [physical two-machine acceptance](docs/physical-two-machine-acceptance.md).

## Platform status

| Platform | Implementation | Software validation | Physical validation |
|---|---|---|---|
| Windows x86-64 | implemented | retained evidence or `not-run` | RTX 5090 gate currently `NOT_RUN` |
| Linux x86-64 | implemented | retained evidence or `not-run` | retained evidence or `not-run` |
| macOS ARM64 | implemented | retained evidence or `not-run` | retained evidence or `not-run` |
| Linux ARM64 | implemented | retained evidence or `not-run` | retained evidence or `not-run` |
| Windows ARM64, macOS Intel, 32-bit systems | unsupported | `not-run` | `not-run` |

Validation is attached to an OS/architecture/backend combination, not inferred from hardware
presence. See [platform support](docs/platform-support.md) for service and firewall behavior.

## Research boundary

Experiment 011 remains the latest completed experiment. This productization does not begin
Experiment 012 and does not add another model family. Historical experiment evidence remains
archived; canonical product, cluster, protocol, transport, runtime, command, platform, and
acceptance packages do not import `swarm_inference.experiments`.

OLMoE is the only product model family for this milestone. Qwen and Kimi-shaped assets remain
research or analysis inputs and are not product support claims.

## Documentation

- [Cluster quick start](docs/cluster-quickstart.md)
- [Node agent](docs/node-agent.md)
- [Pairing protocol](docs/pairing.md)
- [Platform support](docs/platform-support.md)
- [Model artifacts](docs/model-artifacts.md)
- [Cluster troubleshooting](docs/cluster-troubleshooting.md)
- [Product runtime](docs/product-runtime.md)
- [Security boundary](docs/security-boundary.md)
- [Limitations](docs/limitations.md)
- [Physical nodes](docs/physical_nodes.md)
- [Physical two-machine acceptance](docs/physical-two-machine-acceptance.md)
- [Architecture](docs/architecture.md)
- [Recovery](docs/recovery.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
