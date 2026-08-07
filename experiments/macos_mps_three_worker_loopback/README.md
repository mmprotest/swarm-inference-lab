# Apple Silicon Three-Worker Stage Loopback

This development reproduction runs the pinned OLMoE model through three process-isolated workers
using the product coordinator, worker, planning, deployment, and streaming request paths. All
workers share one Apple Silicon MPS device and unified-memory pool, so the result is single-host
software evidence rather than physical distributed or scaling evidence.

## Prerequisites

- Apple Silicon macOS with an operational PyTorch MPS backend.
- The repository environment installed with `uv sync`.
- A running persistent cluster from this checkout with exactly one healthy, unloaded MPS worker.
- The pinned OLMoE snapshot materialized beneath the cluster state root.

## Run

```bash
./experiments/macos_mps_three_worker_loopback/reproduce.sh
```

The default state root is `.local-state`. Use `--state-root` to select another cluster or
`--output-directory` to select a new evidence directory. Existing output directories are never
overwritten.

A pass requires three healthy workers, complete non-overlapping layer ownership, successful
deployment gates, the configured exact token sequence, no remaining session, a clean topology
unload, and restoration of the original persistent worker. The gitignored output under
`artifacts/runs` records the resolved config, provenance, worker states, plan, deployment, streamed
events, unload result, summary, logs, and SHA-256 manifest.

## Development validation

The workflow passed on 2026-08-06 on a 64 GiB Apple M5 Pro. Three workers owned layer ranges
`[0, 6)`, `[6, 11)`, and `[11, 16)`, and generated all 32 expected token IDs. Time to first token
was 1.20 seconds and end-to-end time was 18.50 seconds. These measurements characterize that local
development run only.
