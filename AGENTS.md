# AGENTS.md — Swarm Inference Lab

## 1. Project mission

Swarm Inference Lab exists to test whether very large LLM inference can be assembled from **fragmented commodity compute and memory** rather than requiring any participating machine to hold large model units.

The north-star model is **Kimi K3**. The current technical target is **>= 5 exact output/accepted target tokens per second per user**, eventually with economics competitive with conventional hosted inference.

The project is experimental. Every material claim must be tied to a reproducible experiment and an explicit evidence class.

## 2. Canonical definition of a Swarm result

> **A Swarm result is only a Swarm result if the model cannot be executed by assigning whole layers to the participating workers, and the reported performance emerges from sub-layer fragments distributed across independent machines.**

This definition is non-negotiable.

If an experiment can be implemented by assigning complete transformer layers to the same participating machines, it may still be useful distributed-inference research, but **it is not evidence for the core Swarm thesis**.

## 3. What a Swarm worker means

For core-thesis experiments:

- A **worker is one independent machine**, not a logical GPU rank hidden inside a multi-GPU server.
- A worker owns only **sub-layer model fragments**.
- A worker must have an explicit total peak memory cap.
- The primary pre-physical proof tier is **<= 8 GiB total peak worker memory**.
- Also test 4 GiB, 2 GiB, and 1 GiB tiers where feasible.
- Worker peak memory includes weights, quantization scales, recurrent/KV state, activations, scratch, reduction buffers, transport buffers, CUDA workspace, queued chunks, and measurable allocator overhead.
- No worker may own an entire ordinary K3 transformer layer.
- No worker may own an entire routed expert or entire shared expert in a core-thesis headline configuration.
- A compute event must always have a concrete `worker_id`.
- Every resident model byte and mutable state object must have a concrete worker owner.

A host/pod/locality grouping may exist for orchestration or reporting, but it must not be an aggregate compute resource and must not hide memory, compute, or communication.

## 4. Whole-layer infeasibility must be proven, not asserted

Every core-thesis experiment must run an explicit **whole-layer placement feasibility test** using the same participating worker capacities.

The test asks whether the complete K3 graph could be executed by assigning every transformer layer whole to one of the participating machines without sub-layer sharding.

For the current checkpoint, measured physical checkpoint payload by transformer layer is approximately:

- layer 0: ~2.18 GiB, a small special layer;
- 24 layers: ~15.43 GiB each;
- 68 layers: ~15.82 GiB each.

Therefore an 8 GiB worker tier makes whole-layer placement of the complete model impossible even though the small first layer can fit. The experiment must emit a machine-readable infeasibility receipt rather than relying on these approximate numbers.

Do not weaken the worker cap merely to obtain a favorable throughput result.

## 5. Independent-machine rule

The core thesis is not proven by putting many shard workers inside one conventional multi-GPU server.

For a core Swarm result:

- Each worker is modeled or physically instantiated as an **independent machine**.
- There is no free shared VRAM.
- There is no free PCIe/NVLink/NCCL path between workers.
- Every inter-worker dependency crosses an explicit transport edge with latency, bandwidth, serialization/protocol overhead, queuing, and synchronization accounted for.
- One machine must not host multiple headline workers whose aggregate memory would quietly reconstruct a conventional layer-sized executor.

A multi-GPU server may be used later as a control or implementation comparison, but it must be labeled **clustered distributed inference**, not Swarm proof.

## 6. No monolithic compute abstractions

The following are forbidden as headline compute resources in core Swarm experiments:

- 8-layer `microcell` service times;
- whole-layer service times;
- aggregate GPU pools;
- aggregate host compute;
- idealized stages whose service is supplied externally;
- `sum(layer_time)/workers` or similar divide-by-N speedups;
- hidden monolithic fallbacks underneath a shard API.

A logical stage may be derived **after** worker events complete for reporting, but stage latency must emerge from explicit worker tasks and communication.

The term `microcell` should not be used for a compute resource in new core-thesis experiments. Historical references to Experiments 012–020 are allowed.

## 7. Performance must emerge bottom-up

Every headline throughput result must emerge from:

- physically measured shard primitives where available;
- explicit worker queues;
- explicit shard ownership;
- explicit fanout;
- explicit reductions/collectives;
- explicit state dependencies;
- explicit transport;
- explicit wavefront scheduling;
- explicit scheduler/software overhead.

Do not normalize bottom-up timings to historical target-pass numbers or to the desired result.

No global correction multiplier may be applied merely to force agreement.

## 8. Model validation rule

The model-validity question is:

> Does the event/runtime model predict the physically executed **sharded algorithm**?

It is **not**:

> Does serial execution of a sharded algorithm equal the optimized monolithic algorithm?

Sharding may legitimately increase total compute work.

Validation should use ordered physical shard execution on the local GPU, then constrain the event model to the same resource/order and compare predicted wall time with measured wall time.

Recommended gates unless a later preregistered experiment justifies stricter ones:

- median absolute error <= 5%;
- p90 <= 10%;
- maximum <= 15%.

No post-hoc normalization.

## 9. Evidence classes

Every important performance claim must be labeled as one of:

- **PHYSICAL_SINGLE_MACHINE** — actually timed on one physical machine/GPU.
- **PHYSICAL_SHARD_EXECUTION** — real shard code/weights executed physically, possibly serialized on one device.
- **VALIDATED_INDEPENDENT_MACHINE_MODEL** — explicit independent-worker model whose shard timing/accounting has passed the declared validation gates.
- **SHAPED_NETWORK** — network behavior imposed/modelled, not physically measured between independent machines.
- **PHYSICAL_SWARM** — multiple independent physical machines actually execute the distributed inference path.

Never call a shaped network physical WAN.
Never call a one-GPU logical-worker test a physical swarm.
Never call a multi-GPU single-host run proof of independent-machine Swarm.

## 10. Kimi K3 canonical facts for this repo

Current authoritative local checkpoint path:

`F:\\models\\Kimi-K3`

Checkpoint facts established by prior experiments:

- ~1.56 TB physical checkpoint payload;
- 497,220 tensors in the current census;
- 93 transformer layers;
- 69 KDA layers;
- 24 Gated MLA layers;
- 896 routed experts;
- 16 selected routed experts per applicable token/layer;
- 2 shared experts;
- hidden dimension 7168;
- latent dimension 3584.

Use current repo artifacts/checkpoint metadata as the source of truth and revalidate hashes/identities for new decisive experiments.

## 11. Exactness

Core experiments are exact unless explicitly classified otherwise.

Do not use in a headline exact result:

- expert dropping;
- approximate routing;
- route prediction;
- lossy activation compression;
- lossy state compression;
- state quantization that fails qualification;
- approximate reductions;
- altered K3 weights/model architecture.

Preserve routes, recurrent state, MLA/KV state, AttnRes state, hidden outputs, logits, and greedy token under the established numerical tolerance policy.

Approximate research may be performed as a separately labeled arm only if an exact control is retained.

## 12. The scientific loop

Every experiment follows:

**hypothesis -> implementation -> benchmark -> inspect result -> redesign**

Do not optimize for PASS.
A strong falsification is a successful experiment.

Never change gates after seeing results.
Never promote a microbenchmark to a system claim without passing through the actual worker-level critical path.

## 13. System target

Current primary technical target:

**>= 5 exact Kimi K3 tok/s/user**

Always report:

- target pass ms;
- tok/s/user;
- worker count;
- max peak memory/worker;
- total resident bytes;
- active worker-seconds/token;
- network bytes/token;
- compute-work inflation;
- worker utilization;
- critical path;
- evidence class.

A throughput number without worker size and network assumptions is incomplete.

## 14. Memory-fragmentation curve is a core result

New Swarm experiments should, where possible, measure the tradeoff:

- 8 GiB workers;
- 4 GiB workers;
- 2 GiB workers;
- 1 GiB workers.

For each tier report the best exact throughput, worker count, communication, and utilization.

The core research question is:

> How small can independent worker memory become before useful inference collapses?

Do not substitute a larger-memory tier merely because it benchmarks better.

## 15. Network regimes must be separated

Independent-machine Swarm performance must be reported across explicit network regimes rather than hidden behind one favorable link assumption.

At minimum maintain comparable sweeps such as:

- very fast independent-host: ~0.25 ms / 25 Gb/s;
- fast LAN: ~1 ms / 10 Gb/s;
- regional: ~5 ms / 1 Gb/s;
- WAN/consumer-like: ~20 ms / 100 Mb/s;
- wider WAN sensitivity where useful.

Exact values may be changed only when preregistered and justified.

Report the **break-even network envelope** for the 5 tok/s goal.

A result that works only at datacenter-class network latency must say so clearly.

## 16. Hierarchy is allowed only if it does not redefine the worker

Hierarchical scheduling, reduction trees, caching, and wavefront execution are encouraged.

However, hierarchy must not turn several independent workers into an assumed aggregate compute unit.

All lower-level network costs remain explicit.

## 17. Wavefront lesson from Experiment 018

Experiment 018 demonstrated a useful scheduling property: exact verification chunks can overlap across model depth and significantly shorten the critical path.

Its 6.7022 tok/s result was **not proof of the core Swarm thesis** because the compute model used logical 8-layer stages of roughly 122–136 GiB aggregate model state.

Retain the wavefront scheduling idea.
Discard the giant-stage abstraction for core Swarm claims.

## 18. Experiment 019 lesson

Experiment 019 materially advanced the real thesis:

- complete K3 shard-only placement;
- 376 bounded workers in its winning provisional layout;
- max worker peak ~4.495 GiB;
- no whole layers/experts;
- full 93-layer sharded correctness;
- exact routes and greedy token.

Its `MODEL_INVALID` result followed a flawed serial-equality validation gate. Do not repeat that gate.

E019 also showed that naïve per-expert network microsharding is poor and that **expert-stripe workers** are much better: a stripe worker owns the same sub-expert slice across all experts in its assigned depth, computes the top-16 local fragments, applies route weights locally, and emits one partial output for reduction.

Retain that architecture unless newer evidence falsifies it.

## 19. Experiment 020 lesson

Experiment 020 was correctly zero-spend and found deployment blockers, but its frozen 96-worker plan used 12 conventional 8-GPU P8 hosts.

That plan is **not the core Swarm proof** because:

- each 24 GB RTX 3090 can already hold an ordinary K3 layer;
- an 8x3090 host is a conventional multi-GPU cluster;
- intra-host P8 sharding is optional tensor parallelism, not memory-forced sub-layer Swarm.

The Vast provisioning, safety, model acquisition, hashing, Linux/SM86 build, controller work, and other reusable infrastructure remain valuable.

Do not treat the 12xP8-host deployment topology as canonical for the core thesis.

## 20. Whole-layer control

When hardware/memory permits, compare a sub-layer Swarm architecture against whole-layer placement as a control.

But the **headline Swarm result must use a worker tier where complete-model whole-layer placement is infeasible**.

A whole-layer control is there to quantify fragmentation tax, not to define the Swarm architecture.

## 21. Repeated-work/caching principles

Retain exact optimizations that reduce repeated work without changing the thesis, including:

- immutable AttnRes object caching;
- device-resident persistent state;
- grouped top-16 expert-stripe execution;
- local route-weighted accumulation before reduction;
- persistent connections;
- cached tensor descriptors/pointer maps;
- direct shard loading from checkpoint ranges;
- no repeated whole-weight repacking on the hot path.

Do not resurrect optimizations already falsified as primary latency paths unless new evidence changes the bottleneck.

## 22. Control plane

There must not be one central RPC per microshard.

Use hierarchical/task-batched scheduling while retaining explicit independent worker ownership.

Measure scaling to hundreds/thousands of logical workers/tasks.

Logical task count and physical kernel count are different. Coalesce compatible local shard work into efficient launches where exact dependencies permit.

## 23. Vast.ai policy

No GPU rental without explicit user approval in the current conversation.

Pre-rental experiments may use the installed Vast.ai CLI only for read-only operations such as account/auth verification and live offer searches.

Rental/mutation commands must fail closed unless an explicitly approved physical-swarm experiment arms them with a budget.

For core Swarm planning, search **independent single-GPU/small-memory machines**, not giant multi-GPU hosts, unless those hosts are being used only as controls.

A market/fleet plan must not redefine independent workers into one host-level resource.

## 24. Physical swarm threshold

Do not claim the core Swarm thesis physically proven until multiple independent machines actually execute sub-layer fragments and complete Kimi K3 inference through the production transport.

The eventual physical test should use machines whose per-machine capacity makes whole-layer placement of the complete model impossible.

Until then use `VALIDATED_INDEPENDENT_MACHINE_MODEL`, not `PHYSICAL_SWARM`.

## 25. No OLMoE

Do not add OLMoE-specific code, fixtures, examples, compatibility work, or documentation.

General-purpose open-weight model support remains a product goal, but Kimi K3 is the current north-star proof target.

## 26. Before declaring any experiment complete

Answer these questions explicitly:

1. Could the participating machines have run the complete model by assigning whole layers? If yes, this is not a core Swarm result.
2. Does any headline worker secretly aggregate multiple GPUs/machines? If yes, fix the abstraction.
3. Is every compute event tied to an independent worker?
4. Are all worker memory caps respected including runtime state/buffers?
5. Is the entire checkpoint covered?
6. Are arbitrary real expert routes supported without hot-path weight movement?
7. Are all fanout/reduction/network costs explicit?
8. Is the sharded timing model validated against actual ordered shard execution?
9. Is the result exact?
10. Is the evidence class stated correctly?
11. Does the result move the 5 tok/s goal?
12. What memory tier and network regime does it actually support?

If the answer to #1 is yes, do not use the word `Swarm` in the headline claim.

