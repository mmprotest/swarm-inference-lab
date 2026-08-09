# Experiment 012: evidence-driven hierarchical microworker scaling

Status: in progress  
Evidence root: `artifacts/runs/experiment-012-20260809T004356Z`  
Physical boundary: one Windows host, independent local processes, loopback endpoints, and explicitly labelled network shaping.

## Decision being tested

The experiment separates three decisions:

1. **Root scalability:** whether worker-to-worker delegation bounds the stage owner's direct degree, RPCs, bytes, and synchronous joins as worker count grows.
2. **Single-operation latency:** whether a hierarchy improves or worsens end-to-end latency.
3. **System efficiency:** what CPU, traffic, queueing, forwarding, and failure-recovery work moves to intermediate workers.

The scientific thesis may pass while runtime promotion remains rejected. Promotion requires a measured useful operating region in addition to structural scaling and correctness.

## Immutable operating method

Every candidate mechanism follows this sequence:

1. write a falsifiable hypothesis and fixed PASS/FAIL criteria;
2. implement only the smallest discriminating mechanism;
3. run warmup and repeated measured trials immediately;
4. retain raw rows, errors, failed runs, traces, topology, environment, and source identity;
5. reconstruct runtime behavior from traces and explain all requested root/system metrics;
6. declare PASS or FAIL without changing the criteria;
7. derive the next hypothesis only from the measured bottleneck.

A new question receives a new hypothesis ID. Earlier records are append-only after their first measured trial. Synthetic protocol evidence is always labelled synthetic; shaped links are never called physical LAN or WAN evidence.

## Frozen baselines

Baseline A is concurrent flat root-to-leaf TCP fanout. Baseline B reproduces the existing local scheduler-tree semantics: local scheduler nodes dispatch work, but every network edge still originates at the stage owner and every leaf result returns there. Both use the same persistent process workers, payload, transport framing, warmup, trial counts, concurrency cap, and same-host link profile.

The immutable scale sequence is `2, 8, 32, 128, 512, 1000`. Each scale is attempted. Process startup failures and operating-system limits are evidence and are not replaced silently. The baseline source snapshot and its hashes are retained before delegated behavior is introduced.

## Metric definitions

- `root_rpc_count`: request/response exchanges whose sender is the stage owner.
- `root_leaf_rpc_count`: root RPCs whose receiver has no delegated children for that operation.
- `root_direct_degree`: distinct worker endpoints contacted by the stage owner during the measured operation; topology installation is recorded separately.
- `root_serial_waits`: synchronous response joins performed by root-owned execution contexts. Concurrent joins remain distinct synchronization obligations; `root_coordinator_waits` separately counts outer barriers.
- `critical_path_sync_points`: request/response levels on the longest observed operation path.
- `total_messages`: application request and response messages over every observed edge; connection packets are reported separately when available.
- `root/system bytes`: length-prefixed application bytes observed on the corresponding edges.
- latency percentiles: nearest-rank percentiles over raw successful trial or leaf observations, never percentiles of aggregates.
- throughput: completed operations divided by measured wall time; startup and topology installation are reported separately.
- root CPU: process CPU consumed during the measured operation window.

Every metric is derived from retained trace events and cross-checked against summary counters. A topology object alone is not evidence of delegation.

## Correctness contract

The controlled workload assigns each worker a stable ordering key and exact signed integer contribution. Intermediate nodes return one fixed-width aggregate, reject stale generations and duplicates, and preserve a deterministic contribution digest. The flat reference uses the identical ordered contribution set. Later tensor/model checks use explicit dtype-appropriate tolerances and token identity where exactness is expected.

Incorrect output invalidates the associated performance row. Missing intermediate children, exhausted retries, or an intermediate failure must return an explicit error; partial aggregates are never accepted silently.

## Scaling analysis

For each important root metric, ordinary least squares fits compare `constant`, `log2(N)`, and `N`; `log_B(N)` and `N log2(N)` are added where useful. The evidence records coefficients, residual sum of squares, RMSE, R-squared where defined, AICc when sample size permits, leave-one-out error, and largest-N behavior. A scaling label requires both the best quantitative fit and matching structural traces.

## Network and topology scope

The reusable shaping layer supports same-host, fast LAN, slower LAN, metro/intercity, moderate WAN, and intercontinental profiles through RTT, asymmetric bandwidth, deterministic jitter, request loss, and temporary disconnect controls. Fine-grained hierarchy remains stage-internal and low-latency by default. Experiment 011 persistent coarse stage boundaries are an invariant, not a candidate in this experiment.

## Closure rules

The thesis passes only if traces prove genuine delegation, bounded root degree, zero root-to-leaf RPCs after setup, constant/logarithmic rather than linear root synchronization, hierarchical reduction, and deterministic correctness. Runtime promotion additionally requires measured utility. If credible redesigns are exhausted without those gates, the experiment closes as a documented failure and no candidate enters canonical planning.

