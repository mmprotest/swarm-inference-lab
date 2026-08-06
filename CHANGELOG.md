# Changelog

## 0.1.0rc3 - 2026-08-06

- Pin the GitHub Actions orchestration interpreter to Python 3.11.9, the exact newest 3.11 build
  published for the Windows runner, while retaining uv-managed Python 3.11.15 in the installed
  runtime.

## 0.1.0rc2 - 2026-08-06

- Add a self-contained native Windows x86-64 setup executable with verified embedded wheel,
  pinned uv and Python tooling, exact CPU/CUDA dependency profiles, per-user Apps & Features and
  PATH registration, transactional repair/upgrade/rollback, and state-preserving uninstall.
- Add strict release manifests, checksums, CycloneDX SBOM generation, optional Authenticode
  signing, draft-first GitHub prerelease publication, clean-install/lifecycle acceptance, and
  tag/version/commit validation.
- Print one complete opaque `swarm node join "swarm://..."` command for human pairing while
  retaining secret-free JSON and protected invitation files for automation.
- Add release-aware `swarm update` behavior for native Windows installations without introducing
  a competing runtime updater.

## 0.1.0 - historical unreleased development

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

- Correct release-candidate pairing output so `cluster create --json` and `cluster pair --json`
  each emit one non-secret document, with atomic protected invitation-file delivery for
  automation and one-time URI display for humans.
- Split platform implementation support from exact backend-scoped retained software/physical
  validation evidence, including conservative node-registry schema-2 migration.
- Isolate Linux nftables tables, macOS PF anchors, and Windows firewall reconciliation by a
  bounded owner hash so one cluster cannot remove another cluster's resources.
- Defer service creation during clean wheel installation; cluster creation and node join now own
  default install/start behavior, while foreground mode and post-pair administration remain.
- Enforce shared interactive/`--yes` confirmation semantics before administrative mutation and
  permission-category failure for non-interactive machine calls without `--yes`.
- Initialize/verify the pinned source-only Colibri submodule in source CI while keeping wheels and
  clean node operation independent of `third_party/colibri` and experiment packages.
- Extend productization acceptance with explicit pairing, secrecy, platform evidence, firewall
  isolation, installer, confirmation, source dependency, clean-wheel, and physical-readiness
  gates, a shared repeatability schema, and fail-closed handling for skipped product tests. Opt-in
  Experiment 007 artifact audits are explicitly outside the product selection. The
  two-physical-machine gate remains `NOT_RUN` until executed on distinct hosts.

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
