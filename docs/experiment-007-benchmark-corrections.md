# Experiment 007 benchmark corrections

## Scope

This note preserves the benchmark-validity findings for the immutable Experiment 007 run:

`20260801T013144Z-heterogeneous-node-utility-6a2b51ce`

The original files under that run directory remain historical evidence. They are not overwritten
or reinterpreted by the correction experiment.

## CPU expert result: superseded

Label: `superseded_unmatched_cpu_expert_result`

The original headline reported that one BF16 CPU-owned expert saved 9,437,184 bytes of GPU
memory while retaining 189.387861% of baseline layer throughput. Its reported timings were:

- full-GPU reference layer latency: 364.736400 ms
- hybrid layer latency: 192.587000 ms
- selected CPU expert calls: 13
- CPU dispatch fraction: 0.025390625
- placement policy: `predicted_next_experts`

That throughput comparison is invalid. The full-GPU reference was the Hugging Face decoder-layer
implementation, while the hybrid path used a separate custom expert executor and reused a
separately measured common-component time. Consequently, executor graph, kernel sequence,
synchronisation and timing boundaries were not matched. The same defect is visible in rows that
reported approximately 1.9x retained throughput while `selected_cpu_expert_calls == 0`.

The correction benchmark therefore uses one canonical executor for all-GPU and hybrid placement,
a frozen real routing corpus, matched BF16 weights, matched timing boundaries, active-dispatch
gates and separate quantised diagnostics. No original MoE speed or retained-throughput value is
valid for a product conclusion.

## Background result: superseded

Label: `superseded_fixed_job_background_result`

The original headline at GPU concurrency 1 and CPU concurrency 1 reported:

- GPU-only throughput: 34.558816 output tokens/s
- paired GPU throughput: 34.504872 output tokens/s
- independently measured CPU lane throughput: 58.121600 output tokens/s
- reported combined throughput: 69.009250 output tokens/s
- reported combined gain: 99.686385%
- reported GPU p95 change: 0.028602%
- paired makespan: 7.419295 s

The combined value was computed from a fixed number of GPU and CPU requests divided by the paired
completion makespan. The lanes could finish at different times, and request count changed with
concurrency. That denominator does not measure capacity added during a common serving interval.
The original combined-throughput and gain values are therefore invalid for a product conclusion.

The correction benchmark uses one monotonic warm-up/measurement/drain schedule, continuously
supplied lanes, token-level completion timestamps, identical GPU fixtures between baseline and
combined arms, equal-duration repeats, and rate-based accounting:

`combined_verified_tps = (in_window_gpu_tokens + in_window_cpu_tokens) / window_seconds`

## Original evidence hashes

| File | SHA-256 |
|---|---|
| `cpu_expert_results.csv` | `b21d3426a9424068fd76ff11d445eb6f5c37dc617857999d80ff8a7519cccc1d` |
| `expert_placement_results.csv` | `b21d3426a9424068fd76ff11d445eb6f5c37dc617857999d80ff8a7519cccc1d` |
| `expert_cache_results.csv` | `78d57b42aaf928596fc095f4f11156c9748e5be4d08c49907d28d8f973760f41` |
| `background_results.csv` | `f6a0c6387213990a7ce674f70ed9cd9b16de5877daa37eaec71261475839846a` |
| `summary.json` | `0c636660192af0a346aa6c3e1568f35568b0af9c1d9a8828fadb0b54de2d06da` |
| `config.resolved.yaml` | `f526cb50374e499af49ea35c15001ec92ee415f734af442f7cdc2bbc90646a35` |

Corrected results are written to a new
`artifacts/runs/<timestamp>-experiment-007-corrections-<id>/` directory and explicitly compare
the superseded and corrected measurements.
