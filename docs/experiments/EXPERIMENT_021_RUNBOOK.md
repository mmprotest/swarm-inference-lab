# Experiment 021 Runbook: First Full Physical Kimi K3 Swarm

This is the frozen operating procedure for the first paid physical swarm. Experiment 020 itself is hard-locked read-only. Do not launch until every preflight condition below is green and the user has explicitly approved a dollar cap.

## E020 blocker notice

This runbook preregisters the intended E021 procedure, but it is **not executable for rental yet**. E020 closed with `E021_NOT_READY`. Complete and rerun these pre-rental items first:

- physically execute the exact ordered single-resource shard workload on RTX 5090 and pass the primitive/layer/span replay error gates
- implement and loopback-test the controller-host-agent entrypoint rendered for pod 0
- wire authenticated EXECUTE_SHARD frames to the real KDA/MLA/expert/projection/endpoint worker primitives and rerun the 93-layer graph through worker processes
- implement the real Vast backend adapter for the already-tested provisioning state machine, keep it E020-disabled, and test every call at the mocked subprocess boundary
- integrate the proven pod-bundle/ShardCache acquisition and hash verification into SwarmHostAgent before worker registration
- make preflight accept and verify a user-supplied immutable registry image digest instead of hardcoding published=false

After those pass, rebuild and publish the image by digest, rerun this full zero-rental preflight, and require a complete 12-host market plan. Until then, `--apply` must remain disabled.

## Frozen experiment

- Architecture: 96 bounded GPU workers in 12 pods, 8 workers/GPUs per pod; stripe degree 8, depth span 8.
- Primary GPU policy: homogeneous RTX 3090 24 GB, exactly eight GPUs on one physical host per pod. Re-profiled homogeneous RTX 3090 Ti, A5000, A6000, or RTX 4090 fleets are fallbacks; never mix classes in the primary run.
- Maximum manifest peak: 16.979 GiB/worker.
- Primary physical gate: at least 5 real Kimi K3 output/accepted target tokens/s/user under the bounded-worker architecture.
- Frozen benchmark includes target-only first, then speculative decoding only if target-only passes.

## Prerequisites

1. Checkout the reviewed E021 release. Its reviewed policy change must remove the E020 compile-time read-only lock; the E020 commit must remain unable to rent.
2. Install the same Vast CLI family validated in E020 and authenticate via the local credential store. Never place the API key in this repository, image, command log, or artifact.
3. Publish `swarm-inference-lab:e021-sm86-e020` to an approved registry and replace the placeholder with a pinned digest. The image must pass the local health and eight-worker lifecycle checks.
4. Supply the Hugging Face/model-source credential at runtime only if the authoritative source requires it. Artifacts record presence, never value.
5. Ensure the controller has a per-run TLS credential generator and a secure artifact destination.
6. Rerun the live offer search. Require 12 distinct homogeneous hosts, exactly 8 GPUs/pod, reliability >= 0.95, disk >= 200.0 GB, downlink >= 500.0 Mbps, CUDA >= 13.0, Linux driver >= 580.65.06, direct ports >= 1, verified status permitted by policy, and price <= $8.00/pod-hour.
7. Review the rendered create commands and generated plan digest. Offer IDs are ephemeral; never reuse the E020 snapshot IDs.

## Budget approval

The E020 estimate is $158.95 expected, $632.06 conservative, with an unapproved worst-case cap suggestion of $900.00. The user must explicitly choose a maximum budget. The runtime tracks instance rates x elapsed time plus known disk/transfer costs and tears down before the approved cap.

Generate an immutable approved plan containing `experiment_id=experiment-021`, `approved=true`, and its `plan_sha256`. Approval is for that exact snapshot, image digest, run ID, and maximum budget only.

## One-command preflight (zero rental)

```powershell
python scripts/run_experiment_021.py --preflight --run-id <run_id>
```

Expected output: repository/model validation, redacted Vast authentication, a fresh offer snapshot, exact 12-pod plan or `NO_GO`, cost estimate, intended launch commands with `EXECUTED=false`, bootstrap render, and ledger-driven teardown render. A missing pinned registry digest or fewer than 12 policy-compliant P8 hosts is `NO_GO` and must spend $0.

## One-command launch

Only from the reviewed E021 release, after explicit budget approval:

```powershell
$env:SWARM_ALLOW_RENTAL='EXPERIMENT_021'; python scripts/run_experiment_021.py --apply --experiment-id experiment-021 --approved-plan artifacts/experiment-021/<run_id>/approved-fleet-plan.json --max-budget-usd <APPROVED_USD> --run-id <run_id>
```

All arms are mandatory: the environment variable, `--apply`, exact experiment identifier, approved plan digest, and positive maximum-dollar budget. The state machine—not hand-written commands—owns launch and rollback.

## Provisioning sequence and expected signals

The state log must advance through:

`DISCOVER_OFFERS -> PLAN_FLEET -> USER_BUDGET_GATE -> RENT_CONTROLLER_POD -> WAIT_CONTROLLER -> DISCOVER_CONTROLLER_ADDRESS -> RENT_WORKER_PODS -> WAIT_INSTANCES -> BOOTSTRAP -> DOWNLOAD_SHARDS -> VERIFY_MODEL -> REGISTER_WORKERS -> MEASURE_NETWORK -> VALIDATE_TOPOLOGY -> READY -> RUN_EXPERIMENT -> COLLECT_RESULTS -> DESTROY_ALL -> VERIFY_DESTROYED`.

Every created instance immediately enters the append-only rental ledger with instance ID, offer ID, machine ID, creation time, hourly rate, label `swarm-e021-<run_id>-pod-NNN`, and pod assignment. Stop/destroy may target only ledger IDs.

## Bootstrap and model download

Each multi-GPU host starts one `SwarmHostAgent`, which manages exactly eight GPU-bound worker processes and performs no model compute. It creates one content-addressed pod cache, downloads only the bundle described in `deployment/pod-bundles.json`, resumes partial objects, verifies SHA-256, writes the atomic completion marker, and exposes the cache read-only to its workers. Never download the 1.56 TB checkpoint to every worker and never load a whole layer/expert into GPU memory before slicing.

A bundle hash mismatch, insufficient disk, failed resume, or missing tensor range aborts before registration and enters full ledger teardown.

## Worker registration and health

Expect 96 unique worker IDs with 12 pod memberships and GPU indices 0-7. Each worker reports manifest hash, image digest, GPU identity, SM capability, native-binary hashes, model-bundle hash, and a health nonce over the authenticated TLS channel. Reject duplicate IDs, wrong GPU count, whole-layer fallback capability, missing state ownership, or an unrecognized image/manifest.

## Network qualification

Measure sustained RTT, upload, download, and jitter on every required controller/pod and collective path; marketplace metadata is only a discovery hint. The current candidate thresholds below are **not accepted E021 gates** because the single-resource replay methodology failed. Re-derive and freeze them after that replay passes:

- Intra-pod: RTT <= 1.000 ms and bandwidth >= 5.000 Gbps at that RTT.
- Inter-pod: RTT <= 10.000 ms and bandwidth >= 1.000 Gbps at that RTT.
- Jitter: <= 10% in the gate model.

After corrected validation, every path must satisfy the re-frozen relevant gate. Otherwise emit `TOPOLOGY_REJECTED`, collect diagnostics, and tear down without benchmarking. Never substitute unrelated single-GPU WAN hosts for a local P8 pod.

## Go/no-go gates

Proceed to `READY` only when image/manifests/hashes match, 96 workers are healthy, every pod has eight correct GPUs, model caches are complete, the topology is accepted, estimated spend remains below the approved trajectory, and no ledger inconsistency exists. Any failure is `NO_GO` and triggers teardown.

## Frozen benchmark plan

1. Warm up every worker primitive and record per-worker service, CUDA utilization, memory, power (where exposed), and transport counters.
2. Run the single-user target-only oracle over block/chunk sweep: blocks 7, 12, 16 and chunks 1, 2, 4 where valid. Block 16 is mandatory.
3. Measure physical target tok/s/user, aggregate tok/s, full critical path, network traffic, worker/GPU utilization, power, startup time, cost/token, and failure behavior.
4. Apply the unchanged primary gate: >=5 real Kimi K3 output/accepted target tok/s/user.
5. Only after target-only passes, measure the frozen DSpark proposal path. Record exact accepted-length histograms for coding, math/reasoning, chat, and creative prompts at blocks 7/12/16, plus draft, commit, and rollback latency. Report real speculative output tok/s/user separately.
6. Exercise 1, 2, 4, 8, and 16 requests as an economic secondary analysis. This never replaces the single-user gate.
7. Inject/observe recoverable worker and network degradation only within the approved budget and safety plan; do not change the primary gate.

## Artifact collection

Collect controller trace, worker traces, topology matrices, service samples, GPU telemetry, manifest/image/bundle hashes, acceptance traces, costs, ledger, and failure logs before teardown. Secrets and full account payloads are forbidden in artifacts.

## Normal teardown

The state machine enters `DESTROY_ALL`, destroys every and only ledger-owned instance, records a hash-chained `DESTROYED` event, then repeatedly calls read-only `show instances` until none of the ledger IDs remain. Finish only at `VERIFY_DESTROYED`.

## Emergency teardown

```powershell
$env:SWARM_ALLOW_RENTAL='EXPERIMENT_021'; python scripts/run_experiment_021.py --destroy-ledger artifacts/experiment-021/<run_id>/rental-ledger.json --apply --experiment-id experiment-021 --max-budget-usd <APPROVED_USD>
```

The emergency path rejects IDs absent from the immutable ledger. If the controller is lost, run it from the operator machine against the last fsynced ledger. Preserve the ledger and all Vast responses.

## Failure recovery

- Offer disappears: replan before creating anything else; destroy any controller already ledgered.
- Pod provision/ready/GPU/disk failure: stop the run and destroy all ledgered instances.
- Bootstrap/download/hash/health failure: retain diagnostic hashes, then full teardown.
- Network too slow: `TOPOLOGY_REJECTED`, no benchmark, full teardown.
- Controller crash: invoke emergency ledger teardown from the operator machine.
- Budget trajectory exceeded: kill switch immediately initiates ledger teardown.

Retries require a new run ID, fresh offer snapshot, new approved plan digest, and renewed budget approval. Never silently substitute hardware or reuse an old approval.

## Verify every rental is gone

Run the read-only `vastai show instances --raw`, compare it with the ledger, and require zero surviving ledger IDs. Unrelated user instances must remain untouched. Store a redacted verification receipt and the final valid ledger hash.
