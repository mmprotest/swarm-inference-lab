# Changelog

## 0.1.0 - unreleased

- Establish the deterministic simulation, loopback runtime, experiment artifact format,
  Qwen3 shard tooling, replay recovery, and integrity-audit research harness.
- Validate the immutable Qwen3-0.6B smoke checkpoint through three process-isolated
  stage workers, bidirectional chunked gRPC activations, replay, and a separate
  full-reference phase.
- Make host discovery, endpoint parsing, process shutdown, backend selection,
  environment diagnosis, and launch scripts native across Windows, Linux,
  Linux ARM64, and macOS.
- Validate the Qwen3 split path on native Windows 11 with three CUDA stage
  workers on an RTX 5090 using PyTorch 2.13.0+cu130.
- Add a standard artifact-producing physical LAN/WAN runner that waits for
  remote workers and rejects single-host registrations as physical evidence.

## Unreleased

- Productize the canonical OLMoE runtime as a self-configuring cross-platform cluster with
  `swarm cluster`, `swarm node`, and one-shot `swarm run` commands.
- Add reusable idempotent coordinator/worker lifecycle classes and a persistent bounded node
  agent using Task Scheduler, `systemd --user`, or LaunchAgent service adapters.
- Add strict versioned cluster state, single-use transcript-bound X25519/Ed25519/AES-GCM pairing,
  authenticated membership status, leave/revocation, and backwards-compatible manual trust.
- Add operational backend/dtype benchmarks, automatic memory/endpoints/ports/firewall selection,
  and authenticated directed peer-network measurements with TTL evidence.
- Replace factorial two-stage planning with deterministic bounded N-stage beam search and speed,
  capacity, and balanced node-utility reports.
- Add stage-owned content-addressed OLMoE artifacts, resumable verified transfers, leases/LRU,
  and canonical transactional deployment phases.
- Add wheel-first Windows/POSIX installers, explicit update/rollback, multi-platform CI, expanded
  software/physical acceptance gates, and trusted-LAN documentation.

- Added a sustained single-host loopback scaling matrix with repeated 2, 4, and 8 worker measurements.
- Added a parent scaling report that aggregates child evidence bundles without claiming physical scaling.
- Added `scripts/run_first_experiment.ps1` and `configs/experiments/first_loopback_scaling.yaml`.
- Added sustained loopback duration and matrix repeat options to `swarm experiment`.
