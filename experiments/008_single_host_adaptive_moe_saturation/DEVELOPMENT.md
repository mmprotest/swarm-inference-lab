# Experiment 008 development log

## 2026-08-01: repository inspection and implementation plan

Experiment 008 extends the existing experiment system; it is not a separate runtime.

Reusable components identified before implementation:

- `swarm_inference.microsharding` already provides logical tensor slices, matched MoE
  projection ranges, real safetensors inspection, correctness checks, and measured versus
  projected evidence conventions.
- Experiment 004 provides isolated backend processes, deterministic token fixtures, resumable
  job artifacts, resource metrics, plotting, and report delivery patterns.
- The corrected Experiment 007 provides a canonical matched CPU/GPU MoE executor, real routing
  corpora, fixed-window mixed-service accounting, planner-regret evaluation, and evidence that
  CPU expert execution may be harmful even when it saves VRAM.
- `swarm_inference.planner` already establishes the positive-utility and explicit-idle policy.
- The vendored llama.cpp revision supports conventional layer offload, `--cpu-moe`,
  `--n-cpu-moe`, and regex-based `--override-tensor`. It does not expose per-expert routing
  traces, dynamic expert residency, or an end-to-end expert-prefetch API. Experiment 008 will
  capability-gate these features and record `UNSUPPORTED` rather than emulate them in an
  official run.

Implementation sequence:

1. Add strict Experiment 008 configuration and evidence schemas, including the required
   `MEASURED`, `EMULATED`, and `PROJECTED` result classes and null-with-reason semantics.
2. Add a GGUF tensor inventory reader and tensor-tile IR. Expert microtiles will use matched
   logical projection slices (`up`, `gate`, and `down`), never arbitrary file byte ranges.
3. Add fingerprinted CPU, CUDA, PCIe, and storage profiling based on actual timings, plus a
   resource sampler and profiler-trace contract.
4. Add a measured cost model with critical-path overlap, bounded candidate generation,
   workload-specific objectives, readable decisions, prediction error, and regret evaluation.
5. Add activation statistics, hot/warm/cold cache policy, bounded predictors, reusable
   asynchronous prefetch/cache machinery, and correctness tests. Official evidence is emitted
   only when the selected backend supplies real routing and dynamic-residency hooks.
6. Add a narrowly scoped llama.cpp process adapter for real deterministic generation,
   conventional baseline tuning, tensor overrides, CPU-MoE placement, streaming token timing,
   failure capture, resource sampling, and clean process/GPU release.
7. Add the resumable A-to-G runner. Each configuration is checkpointed atomically; unsupported,
   failed, and incomplete states remain distinct and missing metrics remain null.
8. Generate the required raw tables, ten plots, answer-first Markdown report, manifest, verdict,
   and PowerShell reproduction script from saved evidence.
9. Run focused unit/integration tests, then the existing suite. Run `-Quick` for software and
   real-kernel validation, followed by the largest valid real-model execution available. Only a
   completed `-Full` run with an over-VRAM model may issue an official non-`PARTIAL` verdict.

The implementation will not claim that resource utilisation proves useful work, that process
co-residency proves overlap, or that a fixture-level cache/prefetch result applies to the target
model. Those distinctions are acceptance conditions, not report caveats added later.

## 2026-08-02: completed full execution and final verification

The official run used the pinned `Qwen/Qwen3-Next-80B-A3B-Instruct` artifact revision
`4c8630cf7af926a9c5095cb4bbbbc65d36e20f77`, local GGUF SHA-256
`d103b2733ec1012a52d01edda66b7e5c24ae50508c9f99f5297ea459ef3c061a`, Q4_K_M
quantization, and llama.cpp b9637. The 45.081 GiB tensor inventory genuinely exceeded the
31.843 GiB physical RTX 5090 VRAM. The resumable run completed every configuration checkpoint
and final analysis with no terminal error.

Measured outcome: `PARTIAL`. Capacity, planner quality, positive CPU utility through otherwise
impossible model residency, and reusable architecture passed. Correctness and adaptive
performance failed. G measured 44.288 decode tok/s versus 44.101 stock (+0.42%) and 43,488 ms
32K TTFT versus 43,509 ms stock (-0.05%), both below acceptance thresholds. Its raw mixed
throughput was 40.177 tok/s, but both mixed outputs differed from the reference, so verified
mixed throughput was correctly recorded as 0. B was substantially slower and matched only
5/32 candidate outputs. C-F remained `UNSUPPORTED` because the chosen server does not export
routing IDs, dynamic expert residency, bounded expert prefetch, or operation-level target-model
overlap traces.

The checkpoint preserves two early serialization/schema implementation failures and one
connection reset during B's 32K workload. The implementation defects were fixed; resume then
completed the missing B 32K and mixed measurements without discarding the failed attempts.
Configuration B's final measurements peaked at 95.049 GiB system RAM and had 585,911 ms median
32K TTFT, providing evidence of severe pressure for that plan. No single dominant bottleneck is
claimed for G.

Final validation:

- Experiment 008 focused tests: 34 passed.
- Full repository suite: 345 passed, 7 explicitly skipped hardware/manual tests.
- Ruff: all changed Python files passed.
- `compileall`: all `swarm_inference` sources compiled.
- Evidence audit: all 24 required files and all 10 required raw-data plots are present; plots
  received a visual readability check.

Evidence bundle:

`artifacts/runs/experiment-008-20260801T233500-sydney/experiment_008`
