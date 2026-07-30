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

- Added a sustained single-host loopback scaling matrix with repeated 2, 4, and 8 worker measurements.
- Added a parent scaling report that aggregates child evidence bundles without claiming physical scaling.
- Added `scripts/run_first_experiment.ps1` and `configs/experiments/first_loopback_scaling.yaml`.
- Added sustained loopback duration and matrix repeat options to `swarm experiment`.
