# Experiment 015: Full Physical Kimi K3 on Rented RTX 3090 Cluster

## Status: BLOCKED — DO NOT EXECUTE

This file is the fail-closed handoff from Experiment 014, not an approved preregistration. Experiment 014 failed 12 of 20 hard gates. No GPU fleet should be rented until `artifacts/experiment-014/acceptance-gates.json` reports all required pre-cluster gates passing and the deployment directory no longer contains `DEPLOYMENT-BLOCKED.json`.

The next numbered experiment remains Experiment 015. Resolving the blockers is continuing Experiment 014, not creating an intervening synthetic experiment.

## Provisional cluster candidate

This candidate records the best current placement result without authorizing procurement.

* Node count: 96 provisional; 73 and 80 fail the 10%-headroom memory model.
* GPU: RTX 3090 24 GB or explicitly certified `sm_86` equivalent.
* Topology: checkpoint-aligned coarse persistent pipeline; embeddings, 93 layer stages, final norm, and LM head.
* Placement: `artifacts/experiment-014/k3-3090-placement-manifest.json`.
* Execution DAG: `artifacts/experiment-014/k3-full-execution-plan.json`.
* Host RAM, disk performance, and network minimum: not certified.
* Security ports and provider networking: not certified.

## Model lock candidate

* Model: official local Kimi K3 checkpoint.
* Revision: `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.
* Census semantic fingerprint: `f320c7e1483f06130ca93afc7ac501754498575bb1efd6c62c357a39f8c84737`.
* Required physical tensor bytes: 1,559,965,606,912.
* Required tensor coverage: 497,052/497,052.
* Runtime/package/native dependency lock: missing.

## Provisional serving configuration

The memory solver used context 8,192, four active streams, 3.5 GiB fixed runtime/state reserve, and 10% device headroom. Batch, continuous-scheduler settings, prefill/decode mode, EOS lifecycle, cancellation, queue bounds, and production MXFP8 state sizes have not been certified. This configuration must not be treated as passed merely because it was used as a packing input.

## Required canary order

Experiment 015 may begin only after Experiment 014 produces a real canary that performs, in order:

1. certification/lock check;
2. exact GPU, compute capability, usable VRAM, host RAM, disk, OS, driver, and CUDA checks;
3. clean release installation with no repository checkout;
4. authenticated coordinator admission;
5. explicit `sm_86` Kimi kernel load;
6. native MXFP4 and production activation operation;
7. representative KDA and Gated MLA operations;
8. router, 16 routed experts, shared experts, and reduction fixture;
9. state advancement and isolation fixture;
10. assigned-shard download, hash validation, atomic load, and READY report;
11. memory-reserve check;
12. preregistered performance and network sanity checks.

Any failure must prevent activation of the remaining fleet. The current `artifacts/experiment-015-deployment/canary.sh` intentionally exits with failure because these prerequisites do not exist.

## Metrics reserved for the physical run

When unblocked, collect TTFT; prefill throughput; decode tok/s/user; aggregate output tok/s; GPU, network, stage, and VRAM utilization; p50/p95/p99 latency; scheduler queueing; recovery behavior; hourly cost; and cost per million output tokens. Production telemetry must first have a measured overhead budget.

## Gates that cannot yet be preregistered

The following thresholds are intentionally unset rather than guessed:

* minimum inter-stage bandwidth, RTT, jitter, and loss;
* canary kernel performance floor;
* full-fleet TTFT and decode thresholds;
* aggregate throughput PASS line;
* context/batch/concurrency product envelope;
* maximum cold-start time;
* recovery-time objective;
* capacity-model prediction interval.

Economic target lines, not predicted performance:

* 96 GPUs at $0.165/GPU-hour cost $15.84/hour;
* break-even at $15/M output requires 293.33 aggregate output tok/s;
* a 50% GPU-infrastructure margin requires 586.67 aggregate output tok/s.

## Conditions to unblock this plan

1. Wire and validate the complete Kimi graph to production CUDA and emit inspected `sm_86` binaries/packages.
2. Bind the placement/DAG artifacts to canonical persistent workers using real Kimi tensors and no per-token topology, process, connection, or child-task construction.
3. Measure all distinct Kimi components, prefill/decode, boundary traffic, telemetry overhead, and shaped-network behavior.
4. Validate the service/capacity model on held-out real Kimi slices with reported error.
5. Certify physical memory allocation, context/concurrency envelope, recovery, scheduler, API lifecycle, and state isolation.
6. Complete remote targeted distribution, clean Linux bootstrap, admission, security trust, immutable version lock, health/readiness, cost guard, shutdown, cleanup, and artifact collection.
7. Re-run the exact logical plan through the production runtime and change every required hard gate to PASS.

Until then, Experiment 015 has no approved node count, network requirement, performance threshold, or launch command.
