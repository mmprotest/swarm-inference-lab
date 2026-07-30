# Experiment 001: replicated-stage scaling

## Scope

This experiment asks whether replicated stage capacity increases aggregate
verified throughput under sufficient concurrent load after coordinator relay,
stale load information, per-operation stream creation, and unrealistically
cheap stage work are removed.

The execution mode is **single-host-loopback**. Results describe isolated
worker processes on one host; they are not evidence of physical distributed
scaling.

## Preserved baseline

The original artifact is
`artifacts/runs/20260730T032044Z-loopback-matrix-7cf6e3a7`. The values below
also remain the supplied historical baseline if that artifact is unavailable.

At 64 concurrent requests, median verified throughput was:

| Workers | Replicas per stage | Verified tokens/s |
| ------: | -----------------: | ----------------: |
| 2       | 1                  | 330.186           |
| 4       | 2                  | 330.579           |
| 8       | 4                  | 322.430           |

The 2-to-4 ratio was 1.0012, the 4-to-8 ratio was 0.9753, and the 2-to-8
ratio was 0.9765. The old parent report nevertheless returned `PASS` because
it evaluated artifact completeness rather than the scaling hypothesis.

The baseline also showed approximately 96% of request time in transport and
protocol handling, approximately 0.17% in stage execution, mean stage
utilisation falling from about 5.3% at two workers to 1.3% at eight workers,
unused assigned replicas, and an implausible 25,884 tokens/s capacity
prediction at the eight-worker primary point.

## Existing code paths and causes

The pre-correction paths were inspected before changing runtime behaviour:

* Route selection was `CoordinatorCore._choose_route()` in
  `coordinator/service.py`. It sorted copied heartbeat queue depths and chose
  the first equal worker. There was no admission-time reservation.
* Queue depth originated in `worker/service.py`'s heartbeat loop, was copied
  through `WorkerRegistry.heartbeat()` in `coordinator/registry.py`, and was
  the sole dynamic load signal used by route selection.
* A route was a local `dict[int, StageReplica]` created once in
  `CoordinatorCore.submit()`. The worker IDs were copied into a transient
  `RequestState`; there was no typed, leased, signed route plan or route
  generation.
* Activation transmission was
  `CoordinatorCore._call_replica()` -> `GrpcTransport.execute()` ->
  `WorkerRpcServer._execute_stream()`. Each stage result returned to the
  coordinator before the coordinator encoded and sent the next stage input.
* Coordinator forwarding was the nested stage loop in
  `CoordinatorCore.submit()`, which decoded every intermediate activation and
  retransmitted it.
* `GrpcTransport.execute()` created a new bidirectional `ExecuteStream` RPC
  for every stage operation. Channels were cached, but streams were not.
* Synthetic execution was the inexpensive NumPy affine/roll loop in
  `SyntheticStageModule._layer_transform()`. It was not calibrated and did
  not represent material per-stage CPU work.
* Capacity prediction in `experiments/loopback_matrix.py` used the minimum
  aggregate of placement-time benchmark service rates. It excluded queueing,
  serialisation, integrity, transport, admission, and final-result costs.
* Parent status in `experiments/loopback_matrix.py` was derived only from six
  pilot-integrity criteria. Flat or negative scaling could therefore produce
  `status=PASS`.
* `scripts/run_experiment.ps1` ran `uv sync --extra dev` by default. For the
  synthetic backend it omitted a torch extra, allowing uv to remove an
  existing optional CUDA torch installation. The first-experiment wrapper
  emitted no per-point warm-up or measurement progress.

These paths explain the baseline: concurrent admissions observed the same
stale queue state, selected the same replicas, relayed every activation
through the coordinator, opened thousands of short-lived streams, and spent
almost no time in work that additional stage processes could parallelise.

## Corrections

Experiment 001 introduces:

1. Coordinator-owned atomic route reservations and idempotent route leases.
2. Typed, generation-bound route plans installed on every assigned worker.
3. Direct worker-to-worker activation envelopes in `direct` mode, with an
   explicit `coordinator-relay` regression mode and `emulated` transport mode.
4. A bounded, reconnecting peer connection pool with one multiplexed stream
   per active ordered worker pair.
5. Deterministic calibrated CPU work targeting an 8 ms median stage operation,
   frozen across the matrix, with single-thread settings and recorded affinity.
6. Full-path transport, queue, integrity, execution, reservation, replica, and
   capacity diagnostics.
7. Separate integrity, correctness, direct-data-plane, utilisation, capacity,
   scaling-hypothesis, and overall statuses.
8. A dependency-preserving launcher that uses `uv run --no-sync` when the
   current environment is ready and bootstraps only on explicit request or
   missing required packages.

The initial implementation keeps a request route fixed for its lifetime.
Atomic admission reservations balance concurrent requests, and generation
changes plus replay are used only for failure recovery. This avoids unsafe
movement of stage-local cache state.

## Acceptance criteria

The full run contains 27 points: workers `[2, 4, 8]`, concurrency
`[1, 16, 64]`, three repeats, ten seconds of warm-up, and thirty seconds of
measurement per point.

At concurrency 64, median verified throughput must satisfy all three ratios:

* 4 workers / 2 workers >= 1.50
* 8 workers / 4 workers >= 1.50
* 8 workers / 2 workers >= 2.25

It must also have 100% correctness and completion, coefficient of variation no
greater than 10% at every primary point, at least 75% meaningful replica use,
replica imbalance no greater than 1.5, zero coordinator-relayed activation
bytes in direct mode, stream creation proportional to active peer pairs,
unchanged optional dependencies, and median capacity-prediction error no
greater than 25%.

`overall_status` is `PASS` only when every mandatory component status passes.
The 20 verified tokens/s milestone remains visible but cannot substitute for
replica scaling.
