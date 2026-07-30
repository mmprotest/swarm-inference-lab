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
