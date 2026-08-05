# Experiment 010 runtime mapping

Status: frozen after productization. This document classifies the retained Experiment 010
Python modules without changing archived evidence or historical results.

The classifications are:

- `CANONICAL_REEXPORT`: a deprecated import path that resolves to the canonical product object;
- `EXPERIMENT_ADAPTER`: experiment-specific glue around canonical code;
- `EXPERIMENT_ONLY`: evidence, comparison, shaping, analysis, or verdict code with no product role;
- `LEGACY_FROZEN`: historical runtime code isolated under `legacy_runtime`; no new features permitted;
- `REMOVE`: safe to delete after the stated compatibility point (none are safe to remove today).

All archived reproduction commands may continue to use the public
`swarm_inference.experiments.experiment_010.*` paths. The coordinator, worker, dispatcher,
planner, process entry point, and shaped transport paths are thin named shims; their retained
implementations live under `legacy_runtime`. Canonical product code must never import either
location.

| Module | Class | Current purpose / canonical replacement | Active imports and archived path | Future features | Intended deletion point |
|---|---|---|---|---|---|
| `__init__.py` | EXPERIMENT_ADAPTER | Exposes Experiment 010 modes, verdicts, and canonical expert schema aliases. | Active experiment imports; archived package path required. | Experiment metadata fixes only. | Delete with all Experiment 010 reproduction support. |
| `__main__.py` | EXPERIMENT_ADAPTER | Preserves `python -m ...experiment_010`; product commands live in `swarm` CLI. | Active CLI entry; archived command required. | No product commands. | Delete with the archived runner. |
| `cli.py` | EXPERIMENT_ADAPTER | Parses historical experiment arguments and invokes `runner.py`. | Active experiment entry; archived command required. | Reproduction fixes only. | Delete with the archived runner. |
| `batching.py` | EXPERIMENT_ONLY | Historical batching comparison and metrics; not product session interleaving. | Imported by the Experiment 010 runner. | Experiment corrections only. | Delete when Experiment 010 executable reproduction is retired. |
| `bundle.py` | EXPERIMENT_ONLY | Versioned evidence-bundle writing and validation. | Imported by runner/correction tools; archived bundles retain their files, not a runtime dependency. | Evidence compatibility fixes only. | Delete after archived bundle tooling is retired. |
| `codecs.py` | CANONICAL_REEXPORT | Named aliases to `swarm_inference.transport.expert` codecs. | Public historical imports and tests require the path. | None; extend canonical transport only. | Delete after archived import compatibility is retired. |
| `colibri_expert_bank.py` | CANONICAL_REEXPORT | Named aliases to `swarm_inference.backends.colibri.expert_bank`. | Used by Colibri evidence builders and archived commands. | None; extend canonical backend only. | Delete after archived import compatibility is retired. |
| `colibri_native.py` | EXPERIMENT_ADAPTER | Colibri worker launching, ownership evidence, and comparison orchestration. | Used by token-path/workload evidence. | Evidence corrections only. | Delete when Colibri Experiment 010 reproduction is retired. |
| `colibri_token_path.py` | EXPERIMENT_ONLY | Correctness/capacity token-path evidence and audit logic. | Used by evidence and unit tests. | Evidence corrections only. | Delete after the evidence workflow is retired. |
| `colibri_workloads.py` | EXPERIMENT_ONLY | Experiment-specific workload suites and verdict inputs. | Used by archived workload commands. | Evidence corrections only. | Delete after the workload workflow is retired. |
| `coordinator.py` | LEGACY_FROZEN | Compatibility shim to `legacy_runtime.coordinator`; product coordination is `swarm_inference.coordinator`. | Used by runner/Level A; archived import path required. | Prohibited. | Delete after all historical coordinator reproduction paths are retired. |
| `correction_bundle.py` | EXPERIMENT_ONLY | Corrected evidence validation and bundling. | Used by correction commands; archived evidence itself is unchanged. | Evidence corrections only. | Delete after correction reproduction is retired. |
| `dispatch.py` | LEGACY_FROZEN | Compatibility shim to the historical expert dispatcher; canonical deployment/recovery is coordinator-managed product code. | Used by runner/Level A; archived import path required. | Prohibited. | Delete after historical dispatcher reproduction is retired. |
| `expert.py` | CANONICAL_REEXPORT | Named aliases to `swarm_inference.execution.expert`. | Public historical imports and tests require the path. | None; extend canonical execution only. | Delete after archived import compatibility is retired. |
| `kimi.py` | EXPERIMENT_ONLY | Kimi-specific reference kernels and comparison evidence. | Used by runner and historical tests. | Experiment corrections only. | Delete when Kimi comparison reproduction is retired. |
| `level_a.py` | EXPERIMENT_ONLY | Historical protocol/runtime level-A gate. | Used by runner and tests. | Gate corrections only. | Delete with the archived gate runner. |
| `level_b.py` | EXPERIMENT_ONLY | Historical process/network level-B gate. | Used by runner and tests. | Gate corrections only. | Delete with the archived gate runner. |
| `memory_analysis.py` | EXPERIMENT_ONLY | Evidence memory accounting. | Used by phase-10 analysis. | Analysis corrections only. | Delete with archived analysis tooling. |
| `phase10_analysis.py` | EXPERIMENT_ONLY | Phase-10 report derivation and verdict support. | Used by archived analysis commands. | Analysis corrections only. | Delete with archived analysis tooling. |
| `planner.py` | LEGACY_FROZEN | Compatibility shim to historical marginal-utility experiment planning; product planning lives under `coordinator`. | Used by runner/real-path planner; archived import required. | Prohibited. | Delete after historical planning reproduction is retired. |
| `process_main.py` | LEGACY_FROZEN | Compatibility entry point for the frozen expert worker. | Historical subprocess commands require it. | Prohibited. | Delete after worker-process reproduction is retired. |
| `real_path_planner.py` | EXPERIMENT_ONLY | Experiment-specific real-path feasibility and comparison planning. | Used by correction evidence/tests. | Evidence corrections only. | Delete with real-path experiment tooling. |
| `real_path_resilience.py` | EXPERIMENT_ONLY | Experiment fault and resilience evidence. | Used by correction evidence/tests. | Evidence corrections only. | Delete with resilience evidence tooling. |
| `real_path_simulator.py` | EXPERIMENT_ONLY | Historical simulation baseline, not a product runtime. | Used by experiment tests and reports. | Baseline corrections only. | Delete with simulation reproduction support. |
| `relay.py` | EXPERIMENT_ONLY | Network-shaped relay baseline and fault injection. | Used by Colibri comparison paths; no product import. | Experiment-only shaping changes. | Delete when relay comparison evidence is retired. |
| `relay_main.py` | EXPERIMENT_ONLY | Subprocess entry point for the relay baseline. | Historical commands require it. | Reproduction fixes only. | Delete with `relay.py`. |
| `reporting.py` | EXPERIMENT_ONLY | Charts, tables, and Experiment 010 verdict rendering. | Used by runner and archived report commands. | Report corrections only. | Delete after report reproduction is retired. |
| `runner.py` | EXPERIMENT_ONLY | Top-level historical experiment and evidence orchestrator. | Active only through Experiment 010 CLI/tests; archived command required. | Reproduction and evidence fixes only. | Delete when Experiment 010 execution is formally archived. |
| `schemas.py` | EXPERIMENT_ADAPTER | Experiment enums plus aliases to canonical expert request/response protocols. | Broad experiment use and archived import compatibility. | Experiment metadata only; protocol changes canonical-only. | Delete after archived imports are retired. |
| `transport.py` | LEGACY_FROZEN | Compatibility shim to historical shaped expert client; product transports live under `transport`. | Used by runner, relay, and comparisons; archived import required. | Prohibited except reproducibility fixes in frozen code. | Delete after shaped-transport reproduction is retired. |
| `verification.py` | EXPERIMENT_ONLY | Experiment signatures, numerical comparisons, and verdict evidence. | Used by dispatcher/runner/Level A. | Evidence corrections only. | Delete with verification evidence tooling. |
| `wire.py` | CANONICAL_REEXPORT | Named aliases to `swarm_inference.transport.expert` framing. | Public historical imports and tests require the path. | None; extend canonical transport only. | Delete after archived import compatibility is retired. |
| `worker.py` | LEGACY_FROZEN | Compatibility shim to historical process-isolated expert worker; product workers live under `worker`. | Used by runner, verification, and process entry; archived import required. | Prohibited. | Delete after historical worker reproduction is retired. |
| `legacy_runtime/__init__.py` | LEGACY_FROZEN | Declares the isolated historical-runtime boundary. | Imported only by Experiment 010 compatibility shims. | Prohibited. | Delete with all frozen modules below. |
| `legacy_runtime/{coordinator,dispatch,planner,process_main,transport,worker}.py` | LEGACY_FROZEN | Unmodified historical implementations relocated behind named compatibility shims. | Only Experiment 010 code may import them; archived public paths remain the shims above. | No new functionality; narrowly scoped reproduction fixes must be documented. | Delete after every retained Experiment 010 path has been archived or replaced by an explicit compatibility decision. |

No module is currently classified `REMOVE`: archived reproduction still depends on public import
paths. This is an explicit compatibility decision, not permission to evolve the frozen runtime.
