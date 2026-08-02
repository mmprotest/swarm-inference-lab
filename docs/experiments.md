# Experiments

## Hypothesis

A heterogeneous consumer network can host a model larger than any one node and
increase **aggregate verified output tokens per second across concurrent
requests** as useful nodes are added.

Single-request speed is a separate dependent variable and may decline while
aggregate throughput rises.

## Arms

| Arm | Scheduler | Behavior |
|---|---|---|
| A | static | One fixed replica per stage; failure needs a configured backup. |
| B | fastest route | Route selected by queue, execution, transfer, and reliability cost. |
| C | replicated stage | Bottleneck-aware balanced replica pools and per-request selection. |
| D | workload tier | Interactive, standard, and background cost weights; slow nodes only with positive utility. |
| E | adversarial | Reputation, probabilistic duplicate audits, quarantine, joins/leaves, and replay. |

The full simulation matrix supports node counts `4, 8, 16, 32, 64, 128`;
concurrency `1, 4, 16, 64, 256`; localhost, home-LAN, residential-WAN, and
global-WAN links; hourly churn `0%, 1%, 5%, 10%` plus a 20% burst; and corrupt
fractions `0%, 1%, 5%, 10%`. Fixed prompt and random seeds must be held constant
between scheduler arms.

## Variables and controls

Independent variables are node profile/count, measured stage service rate,
stage placement/replication, concurrency, workload class, network profile,
churn, corruption, audit fraction, microbatch settings, and scheduler arm.

Dependent variables are:

1. aggregate verified output tokens/s;
2. each request's decode tokens/s;
3. time to first token;
4. end-to-end latency;
5. stage utilisation and queue depth;
6. bytes and time by directed link;
7. homogeneous and capacity-normalised scaling efficiency;
8. completion, replay, retry, recovery, disagreement, and quarantine behavior.

Controls include exact model revision/files, tokenizer, precision, sampling,
prompt set, seeds, warm-up interval, measurement interval, queue limits, and
acceptance thresholds.

## Interpretation

- Simulation demonstrates properties of the configured model, not hardware.
- Loopback demonstrates processes, transport, queueing, and execution on one
  host, not physical scaling.
- A physical claim requires `physical-lan` or `physical-wan` artifacts.
- Only committed tokens from successfully completed and verified requests count
  in the primary metric.
- An unused slow node is a valid scheduler decision.
- Any missed threshold is `FAIL`; uncertainty is not rounded into a pass.

Acceptance evaluation uses the criteria in the root build specification:
two consecutive capacity doublings of at least 1.6x at concurrency 16 or above,
non-negative useful-node marginal throughput, at least 70% stage utilisation,
at most 20% capacity imbalance, the 20-token/s five-minute milestone, and the
specified churn/correctness/capacity gates.

Every run writes resolved config, environment provenance, manifests, JSONL
events/metrics, scaling/latency/failure CSVs, summary JSON, offline HTML, PNG
charts, and an artifact hash manifest.

## Physical runner

For `physical-lan` and `physical-wan`, `swarm experiment` starts the coordinator,
prints the pending run directory, waits a bounded time for the configured
number of remote registrations, runs warm-up and steady-state windows, stops
cleanly, and writes the same artifact contract as simulation and loopback.
`--duration-s` can shorten a smoke test but the five-minute throughput
milestone cannot pass with a shorter measured interval.

The runner accepts an optional verified Qwen manifest/model path pair. Without
them it exercises synthetic stages over the actual network and labels model
compute accordingly. Physical classification additionally requires a remote
hostname and non-local advertised endpoint; worker count alone is insufficient.

## Experiment 1: single-host replicated-stage calibration

Configuration: `configs/experiments/first_loopback_scaling.yaml`

Command:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_first_experiment.ps1
```

Question:

> Does the measured process-isolated loopback runtime produce a complete and reproducible scaling curve when replicated workers are added under sustained concurrent load?

The matrix uses worker counts 2, 4, and 8, with concurrency 1, 16, and 64. A single request is included as a negative control because stage replicas should not accelerate one autoregressive request. The higher concurrency points test whether the scheduler distributes independent requests across replicas.

The experiment passes when every matrix point completes sustained verified work, retains valid evidence artifacts, covers every stage, and reports at least two worker counts and two concurrency levels. Positive throughput scaling is an observation, not a pilot pass condition. All workers share one physical host, so the result is not physical scaling evidence.

## Experiment 008: single-host adaptive MoE saturation

Configuration: `configs/experiments/experiment_008_adaptive_moe.yaml`

Quick software and hardware validation:

```powershell
.\experiments\008_single_host_adaptive_moe_saturation\reproduce.ps1 -Quick
```

Official over-VRAM model run:

```powershell
.\experiments\008_single_host_adaptive_moe_saturation\reproduce.ps1 -Full
```

Experiment 008 uses a real sparse-MoE GGUF larger than physical VRAM, a bounded and
workload-specific stock llama.cpp search, logical tensor metadata, capability-gated A-G
ablations, deterministic token comparison, machine profiling, resource telemetry, and a
resumable evidence bundle. Unsupported target-backend hooks remain null and cannot be replaced
by fixture measurements. Only `-Full` is eligible for the official verdict.

## Experiment 009: Colibri-backed adaptive expert runtime

Configuration: `configs/experiments/experiment_009_colibri.yaml`

Fast build, bridge, ABI, and fixture validation:

```powershell
.\experiments\009_colibri_adaptive_expert_runtime\reproduce.ps1 -Quick -ApplyBridgePatches
```

Official practical-model run:

```powershell
.\experiments\009_colibri_adaptive_expert_runtime\reproduce.ps1 -Full -ApplyBridgePatches
```

Experiment 009 pins `JustVugg/colibri` v1.4.0 at commit
`b085b48888a88d9a1c00b151a9979774b72cdbfd` and makes it a first-class local
worker backend. The swarm layer owns capability negotiation, inventory, plan
translation, fixed replay, held-out policy evaluation, and evidence; Colibri
retains model-family math, tokenization, routing, kernels, and expert loading.

The 2026-08-02 full reference run used
`allenai/OLMoE-1B-7B-0125-Instruct@b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`.
All ten gates passed: exact direct/adapter output and routing parity, 0.37%
median decode regression, measured route/cache/storage/tier telemetry, and a
5.04% reverse-confirmed held-out gain for the routing-aware configuration. The
bounded scheduling tuner itself retained baseline; the gain belongs to the
separate routing-placement evaluation. The exercised build was CPU-only, real
microshard execution remains unsupported, and no distributed Kimi K3 claim is
made.
