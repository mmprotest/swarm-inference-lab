# Physical node procedure

A physical result exists only when the standard artifacts come from actual
networked machines. Never relabel loopback or emulated WAN measurements.

## Supported native targets

The coordinator and worker entry points are operating-system neutral. Initial
backend targets are:

| Host | Backend | Provision command |
|---|---|---|
| Windows 11 x86-64 + NVIDIA | `torch-cuda` | `.\scripts\bootstrap.ps1 -Backend cuda` |
| Windows 11 x86-64 CPU | `torch-cpu` | `.\scripts\bootstrap.ps1 -Backend cpu` |
| Linux x86-64 + NVIDIA | `torch-cuda` | `bash scripts/bootstrap.sh --backend cuda` |
| Linux x86-64 CPU | `torch-cpu` | `bash scripts/bootstrap.sh --backend cpu` |
| Linux ARM64 / Raspberry Pi | `torch-cpu` | `bash scripts/bootstrap.sh --backend cpu` |
| macOS Apple Silicon | `torch-mps` or `torch-cpu` | `bash scripts/bootstrap.sh --backend mps` |

Python 3.11, 3.12, and 3.13 are supported. Use the same supported Python minor
version on all participating machines. Docker, WSL, and a shared filesystem are not required.
Use the same Git commit and lock file on all machines.

Run the backend-specific doctor on every host:

```bash
uv run --no-sync swarm doctor --backend cpu
uv run --no-sync swarm doctor --backend cuda
uv run --no-sync swarm doctor --backend mps
```

The report records the OS, architecture, Python, CPU, GPU, driver, PyTorch,
runtime version, memory, disk, interfaces, ports, Git state, and package-lock
hash. A CUDA or MPS doctor exits non-zero if a tiny operation cannot execute.

## Network preparation

Use a trusted LAN. Permit the coordinator port (default TCP 50051) and each
worker port (default TCP 50052) in the host firewall for the private network
only. Do not expose the initial insecure gRPC transport to the Internet.

Measure round-trip latency and bidirectional bandwidth with OS-appropriate
tools, retain their raw output, and mark copied network-profile values
`measured: true`. Worker registration measures coordinator connect latency;
bandwidth remains an assumption until separately measured.

The worker automatically discovers the source address used to reach the
coordinator. Supply `--advertise <reachable-ip>:<port>` explicitly on machines
with VPNs, multiple NICs, containers, or unusual routing. Wildcard addresses
such as `0.0.0.0` are valid bind addresses but are rejected as advertised
addresses.

## Synthetic physical transport run

On the coordinator:

```bash
uv run --no-sync swarm experiment \
  --config configs/experiments/physical_lan.yaml \
  --listen 0.0.0.0:50051 \
  --workers 3 \
  --startup-timeout-s 300 \
  --duration-s 300
```

The runner prints its run directory and waits for remote workers. On three
other machines:

```bash
uv run --no-sync swarm worker \
  --coordinator 192.168.1.10:50051 \
  --listen 0.0.0.0:50052 \
  --backend synthetic \
  --memory-limit-gb 4 \
  --identity .swarm/worker.pem
```

This measures the physical transport and deterministic synthetic stage work.
It is not evidence of real-model kernel performance.

## Real Qwen3 physical run

Shard once and copy the verified shard directory to worker-local storage. A
worker may store every shard on disk, but it loads only its explicitly assigned
stage and emits a tensor load proof.

Coordinator:

```bash
uv run --no-sync swarm experiment \
  --config configs/experiments/physical_lan.yaml \
  --listen 0.0.0.0:50051 \
  --workers 3 \
  --duration-s 300 \
  --model-manifest artifacts/models/qwen3-0.6b/manifest.json \
  --model-path /local/model-metadata-and-tokenizer \
  --dtype bfloat16 \
  --prompt "Explain why distributed inference is difficult."
```

CUDA worker:

```bash
uv run --no-sync swarm worker \
  --coordinator 192.168.1.10:50051 \
  --listen 0.0.0.0:50052 \
  --backend torch-cuda \
  --memory-limit-gb 0.75 \
  --model-shard-root /local/qwen3-0.6b \
  --identity .swarm/cuda-worker.pem
```

CPU or ARM64 worker:

```bash
uv run --no-sync swarm worker \
  --coordinator 192.168.1.10:50051 \
  --listen 0.0.0.0:50052 \
  --backend torch-cpu \
  --memory-limit-gb 0.75 \
  --model-shard-root /local/qwen3-0.6b \
  --identity .swarm/cpu-worker.pem
```

Apple Silicon uses `--backend torch-mps`. On Windows, use ordinary Windows
paths such as `D:\swarm\qwen3-0.6b`; no source changes or WSL translation are
needed.

The coordinator needs configuration, tokenizer/model metadata, and the
manifest. It does not load a full model during the distributed phase.

## Label enforcement

The physical runner refuses to emit a successful `physical-lan` or
`physical-wan` result unless registered workers include:

1. a hostname different from the coordinator hostname; and
2. a coordinator-reachable advertised address that is not local to the
   coordinator.

This deliberately rejects several logical workers on one computer as physical
evidence. The execution-mode label comes from the resolved experiment config
and is written into the report and environment manifest.

## Failure test

During a sustained run, terminate a worker that owns a replicated stage. Retain
all coordinator and worker logs. A valid recovery artifact records the failed
endpoint, replacement shard hash, ordered replay bytes, replay duration,
additional computation, route change, and terminal correctness.

After the run:

```bash
SWARM_RUN_PHYSICAL=1 \
SWARM_PHYSICAL_RUN=artifacts/runs/<physical-run> \
uv run --no-sync pytest -m physical
```

In PowerShell:

```powershell
$env:SWARM_RUN_PHYSICAL = "1"
$env:SWARM_PHYSICAL_RUN = "artifacts\runs\<physical-run>"
uv run --no-sync pytest -m physical
```

Before claiming capacity, verify that model weight bytes exceed every worker's
enforced logical cap and that every load proof contains only assigned tensors.
