# AGENTS.md

# Swarm Inference Lab

This file defines the permanent product thesis, engineering principles, experimental method, and decision rules for Swarm Inference Lab.

These principles override temporary experiment-specific assumptions. Individual experiments may impose artificial constraints to isolate a hypothesis, but those constraints must never silently become the product architecture.

---

## 1. Core Product Thesis

Swarm Inference is a heterogeneous distributed inference runtime for large AI models.

The system should accept a changing pool of compute nodes with different capabilities, understand what each node is good at, and dynamically partition model execution across those nodes in the way that best satisfies the current objective.

A node may differ from another node in:

- accelerator type;
- accelerator memory;
- system RAM;
- compute throughput;
- memory bandwidth;
- supported numerical formats;
- CPU capability;
- network latency;
- network bandwidth;
- locality relative to other nodes;
- reliability;
- availability;
- power or monetary cost;
- model data already cached locally.

The runtime must treat this heterogeneity as a first-class scheduling input.

The long-term objective is not to force one fixed sharding topology onto every deployment. The objective is to build a runtime capable of choosing the right granularity and placement for the resources that actually exist.

---

## 2. Ultimate Product Goal

The eventual product should behave approximately like this:

1. Nodes join the swarm.
2. The runtime profiles or learns their capabilities.
3. The runtime discovers network relationships between nodes.
4. The runtime determines which parts of a model each node can execute efficiently.
5. The runtime creates a placement and execution plan.
6. Model state is distributed or reused according to that plan.
7. Inference executes across the swarm.
8. The runtime measures actual performance.
9. Placement and scheduling can adapt as nodes, workloads, or network conditions change.

The user should not need to manually decide which node runs which layer, expert, projection, attention head, shard, or reduction.

The runtime should make those decisions.

---

## 3. Model Partitioning Is Multi-Granularity

Swarm must support multiple useful partitioning granularities.

Possible units include:

- groups of layers;
- individual layers;
- experts;
- groups of experts;
- expert stripes;
- attention heads;
- projection rows or columns;
- tensor shards;
- sub-layer fragments;
- recurrent or cache state ownership;
- other architecture-specific units where mathematically valid.

No single granularity is the permanent architecture.

Whole-layer placement is valid when it is efficient.

Sub-layer placement is valid when it improves the system.

Mixed placement is expected to be important.

A strong Swarm planner may choose different granularities for different parts of the same model.

---

## 4. Why Sub-Layer Execution Matters

Sub-layer execution is a critical capability, but it is not a requirement that every layer be split.

Its value must be demonstrated empirically.

Sub-layer execution may be useful because it can:

- allow nodes that cannot efficiently take a larger model unit to contribute;
- make fragmented memory usable;
- improve utilization of heterogeneous hardware;
- increase parallelism;
- reduce bottlenecks caused by one slow or oversized stage;
- allow expert, projection, attention, or other model work to be distributed independently;
- enable more flexible placement across a changing node pool;
- improve cost efficiency;
- increase aggregate capacity available to the inference system.

The important experimental question is therefore not:

> Can we split a layer?

The important question is:

> Does having access to sub-layer partitioning improve the best achievable inference system compared with a planner restricted to coarser placement?

Sub-layer execution must earn its complexity through measured system benefit.

---

## 5. Swarm Is Not a Fixed Cluster Topology

Do not hard-code the product around a topology discovered in one experiment.

Concepts such as:

- stages;
- pods;
- cells;
- worker groups;
- locality domains;
- pipelines;

may be useful implementation abstractions.

They must not become permanent compute assumptions unless the evidence justifies them.

In particular, never create an abstract aggregate compute resource and assume it has a service time without deriving that service from the actual resources underneath it.

Any aggregate structure used in a performance model must be explainable from:

- concrete workers;
- concrete compute;
- concrete memory ownership;
- concrete communication;
- concrete synchronization.

Topology is a consequence of placement, not the product thesis.

---

## 6. Nodes Are Capability Descriptions

A node should be represented by measured capabilities rather than a hard-coded hardware class.

A useful node capability record should eventually include fields such as:

- node identifier;
- accelerator type;
- accelerator architecture;
- available accelerator memory;
- available system RAM;
- measured matrix/vector performance for relevant shapes;
- supported dtypes and quantization formats;
- memory bandwidth;
- CPU capability;
- network latency to relevant peers;
- network bandwidth to relevant peers;
- reliability;
- current utilization;
- monetary cost if applicable;
- power cost if applicable;
- local model shards or cached objects;
- software/runtime capabilities.

Prefer measured capability over hardware-name heuristics.

Two devices with the same product name may behave differently because of clocks, topology, drivers, contention, or network.

---

## 7. The Placement Problem

The long-term planner should solve a constrained optimization problem.

Inputs include:

- model graph;
- tensor sizes;
- operator dependencies;
- state dependencies;
- available nodes;
- node capabilities;
- network topology;
- cached model state;
- workload;
- latency target;
- throughput target;
- cost objective;
- reliability constraints.

Outputs include:

- model partition;
- tensor ownership;
- state ownership;
- worker assignment;
- replication decisions;
- communication plan;
- collective plan;
- execution schedule;
- wavefront or pipeline schedule where useful.

Potential objectives include:

- minimize single-user latency;
- maximize throughput;
- minimize cost per token;
- maximize useful hardware utilization;
- minimize network communication;
- minimize expensive synchronization;
- satisfy a latency target at minimum cost.

Different product modes may optimize different objectives.

---

## 8. Performance Must Be End-to-End

Do not confuse a local kernel improvement with a Swarm improvement.

Every optimization must eventually answer:

- Did target-pass latency improve?
- Did tokens per second improve?
- Did cost per token improve?
- Did usable node capacity improve?
- Did the critical path shrink?
- Did communication or synchronization increase elsewhere?
- Did the optimization still help after composing the full system?

Local benchmarks are diagnostic evidence.

System-level performance is the decision metric.

---

## 9. Communication Is Part of Compute Architecture

Distributed execution is useful only if communication does not erase the benefit.

Every partitioning design must account for:

- bytes transferred;
- message count;
- serial waits;
- collective steps;
- fanout;
- reduction;
- software transport overhead;
- network latency;
- network bandwidth;
- state movement;
- synchronization frequency.

Prefer architectures that:

- keep nonlinear intermediate state local;
- send compact inputs and outputs;
- coalesce many logical operations into fewer physical messages;
- reduce locally before communicating;
- cache immutable state;
- avoid repeatedly transmitting unchanged data;
- overlap communication with useful compute where dependencies permit;
- keep fine-grained synchronization on sufficiently fast links;
- use coarse communication across slower links.

Many messages are acceptable if they do not become serial critical-path waits.

---

## 10. Logical Granularity and Physical Execution Are Different

A logical task may be tiny without requiring one kernel, one process, or one network message per task.

This distinction is fundamental.

The runtime should be able to represent thousands of fine-grained logical tasks while physically coalescing compatible work.

Examples:

- many expert fragments may execute in one grouped kernel;
- many route assignments may be reduced locally before one network transfer;
- multiple logical shard tasks may share persistent state;
- a persistent worker may execute an internal task graph without returning to the central coordinator.

Fine-grained ownership should not automatically imply fine-grained overhead.

---

## 11. Persistent State Is Important

Where possible, workers should retain the state they repeatedly need.

Examples include:

- model weights;
- quantization metadata;
- recurrent state;
- KV or compressed attention state;
- immutable depth-state objects;
- static pointer maps;
- tensor descriptors;
- routing structures;
- compiled kernels;
- reusable buffers.

Performance experiments must distinguish:

- startup cost;
- model acquisition cost;
- steady-state inference cost.

Do not accidentally include repeated model loading in steady-state execution unless the intended product truly requires it.

Likewise, do not exclude loading or transfer costs if the proposed deployment would actually pay them repeatedly.

The validation environment and modeled environment must use the same residency assumptions.

---

## 12. Dynamic Adaptation Is a Product Requirement

The final system should not assume a static fleet forever.

Nodes may:

- join;
- leave;
- slow down;
- become unavailable;
- change network conditions;
- change utilization;
- change cost.

The planner should eventually be capable of:

- profiling new nodes;
- deciding whether a node is useful;
- assigning useful work;
- rebalancing bottlenecks;
- avoiding or removing harmful nodes;
- recomputing placement when necessary.

More nodes are not automatically better.

A node should participate only when its contribution improves the selected system objective or provides required capacity/reliability.

---

## 13. Scientific Experimental Process

All major Swarm development must follow:

> hypothesis -> implementation -> benchmark -> inspect result -> redesign

Every experiment must begin with a falsifiable hypothesis.

Every experiment must define success and failure criteria before seeing the result.

Every experiment must preserve enough artifacts for an independent reader to reconstruct:

- what was tested;
- what code ran;
- what model/checkpoint ran;
- what hardware ran;
- what assumptions were modeled;
- what was physically measured;
- how metrics were calculated;
- why the conclusion follows from the evidence.

A failed hypothesis is a useful result.

Do not redesign acceptance thresholds after seeing the data.

---

## 14. Experimental Evidence Classes

Every result must clearly identify its evidence class.

### PHYSICAL

Actually executed on the stated independent hardware and network.

### PHYSICALLY GROUNDED MODEL

Uses physical measurements of lower-level operations but composes them into a larger unmeasured system model.

### SHAPED NETWORK

Uses explicit simulated or shaped network conditions.

### PROJECTION

Derived from measurements and assumptions but not directly physically instantiated.

### SYNTHETIC

Uses artificial workload or timing inputs.

Never present one class as another.

A modeled full swarm is not a physical swarm.

A single-device sequential replay is not a physical many-device swarm.

---

## 15. Model Validation

A performance model must predict the algorithm it is actually modeling.

Validation should compare:

> predicted execution of implementation X

with:

> physically measured execution of implementation X.

Do not require a distributed/sharded algorithm to have the same serial cost as a different monolithic implementation.

Sharding may introduce additional work.

That work must be measured and charged.

The important checks are:

- are all compute operations included?
- are all communication operations included?
- are all waits included?
- are all state operations included?
- does the model predict a physically executed version of the same task graph?

Never normalize a model to force agreement with the desired result.

If a correction factor is required, its origin must be independently justified and validated on held-out evidence.

---

## 16. Baselines Must Be Strong

When testing whether a new capability is valuable, compare it against the strongest reasonable alternative using the same resource pool.

Examples:

- whole-layer placement versus mixed/sub-layer placement;
- whole experts versus expert stripes;
- serial stage execution versus wavefront execution;
- fixed placement versus adaptive placement;
- current transport versus improved transport.

Do not compare a sophisticated new method against an intentionally weak baseline.

The goal is to learn whether the capability adds value to the best system we could otherwise build.

---

## 17. Whole-Layer vs Sub-Layer Value Test

This should become a recurring Swarm benchmark.

Given the same heterogeneous node inventory, evaluate increasingly capable planners:

### Planner A
Coarse placement only.

### Planner B
Whole layers plus architecture-specific coarse units such as whole experts.

### Planner C
Adds selective sub-layer partitioning.

### Planner D
Fully adaptive mixed-granularity placement.

Measure:

- feasible/not feasible;
- tokens per second;
- latency;
- aggregate throughput;
- critical path;
- utilized memory;
- stranded memory;
- worker utilization;
- communication;
- cost per token;
- number of useful nodes.

The value of sub-layer execution is the improvement from allowing the planner to use it when advantageous.

---

## 18. Heterogeneous Resource Experiments

Experiments should increasingly use realistic mixed inventories rather than only homogeneous fleets.

Useful variations include:

- different memory capacities;
- different compute speeds;
- different accelerator architectures;
- CPU-only nodes;
- different network links;
- changing reliability;
- different monetary costs.

The planner should learn or measure which nodes are useful.

Do not assume every available node should be used.

---

## 19. Critical-Path Thinking

The main performance question is:

> What is on the critical path of one output token or verification block?

Track:

- total compute work;
- parallel compute work;
- serial compute work;
- communication critical path;
- synchronization critical path;
- pipeline fill/drain;
- load imbalance;
- straggler amplification.

Prefer designs that transform:

`sum(all work latency)`

toward:

`critical path through overlapping work`.

Do not claim parallelism merely because work has been divided into many tasks.

---

## 20. Standard Metrics

Where applicable, Swarm experiments should report:

- output tokens/second/user;
- target-only oracle tokens/second/user;
- aggregate tokens/second;
- latency per accepted/output token;
- critical-path latency;
- useful parallelism;
- critical-path fraction;
- total worker compute;
- compute-work inflation;
- bytes transferred/token;
- serial waits/token;
- messages/token;
- worker utilization;
- memory utilization;
- maximum worker memory;
- total resident model memory;
- replication factor;
- active worker-seconds/token;
- cost per million output tokens;
- prediction error when a model is used.

Metrics should be mechanically derived from saved artifacts where possible.

---

## 21. Correctness Comes Before Performance

Optimization must preserve the intended model semantics unless an experiment explicitly studies approximation.

Exact-mode checks may include:

- tensor coverage;
- output equivalence;
- route identity;
- expert identity/order;
- recurrent-state identity;
- cache/state fingerprints;
- hidden-state error;
- logit error;
- greedy-token identity.

Approximate execution, if ever tested, must be clearly labeled and compared with an exact control.

Never silently trade correctness for speed.

---

## 22. Generality

The runtime is intended to become a general distributed inference system, not a one-model benchmark harness.

Specific models may serve as demanding proving grounds.

Model-specific kernels and execution strategies are acceptable when architecture-specific behavior genuinely requires them.

However:

- model names must not be used as hidden benchmark shortcuts;
- scheduling abstractions should remain general where possible;
- architecture-specific capabilities should be exposed cleanly;
- the planner should be able to reason about different model structures.

A successful experiment should be incorporated into the canonical runtime when the result is general enough to justify it.

---

## 23. No Benchmark-Specific Tricks

Never introduce:

- hard-coded benchmark routes;
- task-name heuristics;
- test-fixture-specific branches;
- synthetic shortcuts in production paths;
- hidden precomputed outputs;
- assumptions chosen only because they make one experiment pass.

Experiments must test architecture, not exploit the benchmark.

---

## 24. Do Not Confuse Capacity With Performance

A partition may prove that a model fits across a set of nodes.

That does not prove it runs efficiently.

Always separate:

### Capacity result
The model can be represented and executed correctly using the available memory.

### Performance result
The resulting execution has useful latency/throughput.

### Economic result
The resulting execution is cost-effective enough for the intended product.

These are separate gates.

---

## 25. Do Not Confuse Simulation With Deployment

Single-machine logical workers are useful for:

- correctness;
- task-graph validation;
- scheduler testing;
- control-plane scaling;
- deterministic network models;
- placement experiments.

They cannot prove:

- actual inter-machine transport performance;
- real distributed contention;
- actual straggler behavior;
- real multi-device collectives;
- real full-swarm throughput.

Before spending money on large physical experiments, simulations should eliminate every uncertainty they reasonably can.

Once those uncertainties are exhausted, physical experiments should adjudicate the remaining ones.

---

## 26. Experimental Scope Discipline

Each experiment should answer one primary scientific question.

Do not allow an experiment to expand indefinitely into unrelated optimizations.

Secondary arms are appropriate when they:

- directly explain the primary result;
- remove a discovered bottleneck;
- test an immediate redesign suggested by evidence.

If the central hypothesis is falsified, say so.

Do not rescue it by silently changing the question.

---

## 27. Historical Results Do Not Become Assumptions

Previous experiment results are evidence.

They should inform later hypotheses.

They must not become permanent architectural constraints merely because they once won a benchmark.

Every inherited choice should be revisited when:

- the available resource pool changes;
- the optimization objective changes;
- the bottleneck moves;
- a more general planner becomes available.

---

## 28. Product Economics

The eventual runtime must be economically meaningful.

Performance alone is insufficient.

Relevant economic questions include:

- What hardware capacity must remain reserved?
- How much of it is actively computing?
- How much network traffic is generated?
- How many concurrent users can the placement serve?
- What is the cost per output token?
- What is the value of otherwise stranded hardware?
- Does using a weak node help or hurt the system?
- Is replication worth its cost?
- Is a different partition cheaper for the same service target?

The planner should eventually be able to optimize for cost as well as latency.

---

## 29. The Long-Term Product Test

The ultimate demonstration should look like this:

1. Provide Swarm with a heterogeneous set of nodes.
2. Do not manually prescribe the model split.
3. Swarm profiles the resources.
4. Swarm discovers network topology.
5. Swarm creates a mixed-granularity placement.
6. The model is distributed.
7. Inference runs correctly.
8. Swarm measures performance.
9. A node joins, leaves, slows down, or changes.
10. Swarm adapts its placement or schedule.
11. Performance remains useful.

The important achievement is not one clever partition.

It is a runtime that can repeatedly find a good partition for the resources it has.

---

## 30. North-Star Research Question

All experiments should ultimately contribute evidence toward:

> Can a heterogeneous pool of otherwise fragmented compute and memory be turned into a useful, adaptive virtual inference accelerator for models whose efficient execution would normally require much more rigid infrastructure?

Sub-layer partitioning is a central capability in answering this question.

It is not the entire answer.

---

## 31. Permanent Experimental Loop

When deciding what to do next, always return to:

**Hypothesis**

What specific claim are we testing?

**Implementation**

What is the smallest honest implementation capable of testing it?

**Benchmark**

What measurement would distinguish success from failure?

**Inspect**

Where did time, memory, network traffic, and synchronization actually go?

**Redesign**

What does the evidence imply should change next?

Then repeat:

> hypothesis -> implementation -> benchmark -> inspect result -> redesign

This loop is the operating method of Swarm Inference Lab.

---

## 32. Final Guardrail

Before proposing or implementing a major architectural change, ask:

1. Does this serve the heterogeneous adaptive Swarm thesis?
2. Is this a temporary experimental constraint or a product requirement?
3. Am I accidentally hard-coding a topology because it performed well once?
4. Am I allowing the planner to choose whole-layer and sub-layer placement when appropriate?
5. Does the claimed performance emerge from the actual workers and communication underneath it?
6. Is the evidence physical, modeled, or projected, and is it labeled correctly?
7. What strong baseline should this be compared against?
8. What would falsify the idea?

If these questions cannot be answered clearly, do not proceed until the experiment is reframed.
