# swarm-inference-lab

`swarm-inference-lab` is an open-source research harness for one falsifiable
question:

> Can heterogeneous consumer devices collectively host a language model larger
> than any participating node and increase aggregate verified output-token
> throughput as useful nodes are added despite latency, uneven hardware, churn,
> and incorrect workers?

The primary metric is **aggregate verified output tokens per second across
concurrent requests**. It is never presented as single-request generation
speed. Every run is labelled `simulation`, `single-host-loopback`,
`physical-lan`, or `physical-wan`.

## Current status

| Capability | Status | Evidence and boundary |
|---|---|---|
| Deterministic event-queue simulation | Implemented and tested | Fixed seeds reproduce event fingerprints; assumed profiles remain marked as assumptions. |
| Static, fastest, replicated, workload-tier, and adversarial scheduling | Implemented for simulation | Counterproductive workers are allowed to remain idle. |
| Standard artifact tree and offline HTML report | Implemented and tested | Includes provenance, machine-readable metrics, acceptance results, hashes, and seven required charts. |
| Process-isolated loopback | Implemented and tested | Real gRPC AsyncIO activation traffic, bounded queues, worker termination, and replay recovery. It is not physical scaling. |
| Native Windows 11 CUDA execution | **PASS for correctness** | RTX 5090, PyTorch 2.13.0+cu130, three process-isolated `torch-cuda` Qwen stages, and a separate full-reference process. This remains `single-host-loopback`. |
| Dense Qwen3 safetensors sharding | Implemented and checkpoint-validated | `Qwen/Qwen3-0.6B@e6de91484c29aa9480d55605af694f39b081c455`; three shards, exact tensor union, and verified shard hashes. |
| Dense Qwen3 split correctness | **PASS on CPU and CUDA loopback** | Four greedy output IDs matched the separate unsplit reference and all recorded boundary errors were zero. |
| Cross-platform worker program | Implemented | The same Python entry point supports native Windows x86-64 CPU/CUDA, Linux x86-64 CPU/CUDA, Linux ARM64 CPU, and macOS Apple Silicon CPU/MPS. Only Windows CUDA has been exercised in this workspace. |
| Physical experiment runner | Implemented, awaiting remote hosts | Waits for remote registrations, rejects false physical labels, runs warm-up and sustained measurement, and writes the standard artifact set. |
| Physical scaling claim | **Not demonstrated** | Requires a completed standard run with at least two actual machines. |
| Over-VRAM Qwen3-Next sparse MoE execution | **PARTIAL** | Experiment 008 generated text with 45.081 GiB of Q4_K_M tensors across an RTX 5090 and system RAM. Capacity passed; correctness and adaptive-performance gates failed. |
| Colibri local expert runtime | **PASS_STRONG** | Experiment 009 ran a real OLMoE model through the universal worker ABI with exact direct/adapter tokens and routes, measured residency/I/O telemetry, and a 5.04% reverse-confirmed routing-placement gain. The exercised native Windows engine was CPU-only. |
| Kimi K3 execution | **Not demonstrated** | Experiment 009 preserves native MXFP4 metadata and detects the Kimi family, but did not execute a Kimi K3 checkpoint or distributed experts. |

The project-wide research status is **FAIL** until every acceptance gate has
measured evidence. An unrun criterion is never converted to `PASS`.

### Milestones

| Milestone | Status | Boundary |
|---|---|---|
| 0 — repository/environment | Green | Native installation, CLI, backend-aware doctor, configuration, lint, typing, and tests. |
| 1 — deterministic simulator | Green | Fixed-seed 4–128-node matrix and explicit PASS/FAIL reports. |
| 2 — loopback services | Green | Independent workers, real gRPC activation transport, chunk streaming, and replica recovery. |
| 3 — real-model sharding | Green | Official Qwen3-0.6B checkpoint passes distributed/reference correctness on native CPU and CUDA. |
| 4 — concurrent pipeline | Partial | Bounded concurrent queues and simulated scaling work; sustained measured hardware scaling is not demonstrated. |
| 5 — churn/integrity | Partial | Replay, signed envelopes, audits, and quarantine work; the physical churn gate is unrun. |
| 6 — physical LAN | Ready for multiple machines | Native coordinator/worker orchestration and artifacts are implemented; a second host has not participated yet. |
| 7 — sparse MoE proxy | Partial | Synthetic routing remains useful for controlled tests; Experiment 008 separately adds real Qwen3-Next execution without routed-expert backend hooks. |
| 8 — single-host adaptive MoE saturation | **PARTIAL** | A completed full run passed capacity, planner, positive-CPU-utility, and architecture gates; correctness and adaptive performance failed. |
| 9 — Colibri adaptive expert runtime | **PASS_STRONG** | Pinned Colibri v1.4.0 executes as a local worker backend with real routes, tier/cache/storage counters, fixed replay, plan translation, and held-out placement evaluation. |

## Native installation

Python 3.11 is required. `uv` selects an operating-system-specific PyTorch
source through the lock file; PyTorch is deliberately not a mandatory base
dependency for synthetic-only nodes.

Windows PowerShell:

```powershell
.\scripts\bootstrap.ps1 -Backend cuda       # NVIDIA
.\scripts\bootstrap.ps1 -Backend cpu        # CPU only
```

Linux, Linux ARM64, or macOS:

```bash
bash scripts/bootstrap.sh --backend cuda    # Linux NVIDIA
bash scripts/bootstrap.sh --backend cpu     # Linux/ARM CPU
bash scripts/bootstrap.sh --backend mps     # Apple Silicon
```

Direct `uv` equivalents:

```bash
uv sync --extra dev --extra cpu
uv sync --extra dev --extra cuda
uv sync --extra dev --extra mps
uv sync --extra dev                         # synthetic-only development
```

Pip fallback:

```bash
python3.11 -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# POSIX:   source .venv/bin/activate
python -m pip install -e ".[cpu,dev]"
pytest
```

For CUDA with pip, install the wheel selected by the current official PyTorch
installer before `pip install -e ".[dev]"`. The `uv` CUDA extra is pinned to an
official PyTorch wheel index in `pyproject.toml` and `uv.lock`.

WSL2 is optional. It is detected and reported, but native Windows is a
first-class runtime.

## One-command native Windows experiment

From the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_experiment.ps1 -Backend cuda
```

The script locates or installs `uv`, synchronises the selected native backend,
runs `swarm doctor`, launches the coordinator and process-isolated workers,
performs clean shutdown, prints the report path, and returns non-zero when the
experiment or its acceptance criteria fail. A reported `FAIL` is a valid
experimental result.

Synthetic loopback without a PyTorch backend:

```powershell
.\scripts\run_loopback.ps1 -Backend synthetic
```

## First measured experiment

The first experiment is a single-host loopback scaling pilot. It compares 2, 4,
and 8 process-isolated workers at 1, 16, and 64 concurrent requests. Each point
uses sustained measured execution, and the default PowerShell script repeats
every point twice.

Run from the repository root on Windows:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_first_experiment.ps1
```

For a quicker harness check:

```powershell
.\scripts\run_first_experiment.ps1 -Repeats 1 -DurationS 10
```

The parent report contains the median scaling curve and links to every child
run. The pilot can validate the scheduler, process isolation, local gRPC
transport, measurement stability, and artifact integrity. It cannot establish
physical multi-machine scaling because every worker shares one host.

## Quick experiments

Deterministic simulation:

```bash
uv run --no-sync swarm simulate --config configs/experiments/smoke.yaml
```

Full 4-to-128-node simulation matrix:

```bash
uv run --no-sync swarm experiment --config configs/experiments/scaling_simulation.yaml
```

Four process-isolated local workers:

```bash
uv run --no-sync swarm experiment \
  --config configs/experiments/scaling_loopback.yaml \
  --workers 4
```

On PowerShell, place the command on one line or use PowerShell backticks instead
of backslashes.

## Real Qwen3 correctness

Model download and sharding are always explicit:

```bash
uv run --no-sync swarm inspect-model --model Qwen/Qwen3-0.6B
uv run --no-sync swarm shard-model \
  --model Qwen/Qwen3-0.6B \
  --output artifacts/models/qwen3-0.6b \
  --target-stage-bytes 536870912 \
  --max-stage-bytes 536870912
```

Native Windows CUDA validation:

```powershell
.\scripts\run_native_model.ps1 `
  -Shards .\artifacts\models\qwen3-0.6b `
  -ModelPath C:\path\to\resolved\huggingface\snapshot `
  -Backend cuda `
  -Workers 3
```

Portable CLI equivalent:

```bash
uv run --no-sync swarm validate-model \
  --shards artifacts/models/qwen3-0.6b \
  --model-path /path/to/resolved/huggingface/snapshot \
  --output artifacts/validation/qwen3-correctness \
  --max-new-tokens 4 \
  --device cuda \
  --dtype bfloat16 \
  --distributed-loopback-workers 3 \
  --distributed-backend torch-cuda
```

Use `cpu`, `float32`, and `torch-cpu` on CPU nodes; use `mps`, `float16`, and
`torch-mps` on Apple Silicon. The distributed phase runs first. No worker or
coordinator loads the full model. The unsplit reference runs later in a
separate disclosed validation process and its memory is excluded from swarm
capacity.

The final native Windows CUDA result from this workspace is at
[`artifacts/validation/qwen3-0.6b-native-cuda-final/correctness.json`](artifacts/validation/qwen3-0.6b-native-cuda-final/correctness.json).
It proves split correctness on one host, not throughput scaling.

## Physical machines

Start the artifact-producing runner on the coordinator:

```bash
uv run --no-sync swarm experiment \
  --config configs/experiments/physical_lan.yaml \
  --listen 0.0.0.0:50051 \
  --workers 3 \
  --duration-s 300
```

Then start one worker per remote host:

```bash
uv run --no-sync swarm worker \
  --coordinator 192.168.1.10:50051 \
  --listen 0.0.0.0:50052 \
  --backend synthetic \
  --memory-limit-gb 4
```

The worker derives a coordinator-reachable advertised address. Use
`--advertise <worker-ip>:50052` when routing is ambiguous. For real Qwen,
provide `--model-manifest` and `--model-path` to the experiment command, then
run workers with a `torch-*` backend and `--model-shard-root`.

The runner refuses a `physical-*` result unless registration contains both a
different hostname and a non-local advertised address. Read
[the physical-node procedure](docs/physical_nodes.md) before making a physical
claim.

## Metrics

- **Aggregate verified output tokens/s:** committed tokens from successfully
  completed and verified requests divided by experiment elapsed time.
- **Per-request tokens/s:** decode throughput for every request, reported
  separately.
- **Time to first token:** request acceptance to first committed output token.
- **End-to-end latency:** request acceptance to terminal completion.
- **Stage utilisation:** busy service time divided by available replica time.
- **Network traffic:** payload bytes and measured or emulated time by directed
  link.
- **Scaling efficiency:** homogeneous throughput gain divided by node-count
  gain, or observed throughput divided by the measured/configured ideal.
- **Failure/recovery:** failures, retries, replay bytes/time, additional
  computation, route changes, quarantine, and completion status.

## Security warning

Never expose the initial insecure gRPC channel to the public Internet and do
not send sensitive prompts to untrusted workers. Workers can inspect their
input activations and caches. The project does not provide prompt privacy,
activation privacy, collusion resistance, Byzantine fault tolerance, or
cryptographic proof of neural computation.

See [architecture](docs/architecture.md), [protocol](docs/protocol.md),
[experiments](docs/experiments.md), and [limitations](docs/limitations.md).
