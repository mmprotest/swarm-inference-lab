# Physical-node runbooks

Physical validation is distinct from CI, process isolation, simulation, and loopback. Every gate
uses wheel installation without a repository clone, high-level pairing, persistent services,
non-loopback direct traffic, exact token IDs, artifact hashes, and checksummed evidence schema 2.

## Gate A - Windows RTX 5090 plus Windows CPU laptop

Use [the complete two-machine procedure](physical-two-machine-acceptance.md). The RTX PC is the
coordinator/local CUDA worker; the laptop is the joined CPU node. It must demonstrate speed-mode
exclusion and capacity-mode participation in separate runs.

Acceptance option: `--physical-config <configuration.json>`.

## Linux x86-64 CPU

Use two distinct Linux x86-64 machines with reachable authenticated cluster endpoints. A LAN is
a useful first physical topology, but is not the product boundary.

1. Build the wheel elsewhere; copy the wheel and `install.sh`, not the repository.
2. Install with `sh install.sh --source-wheel <wheel> --json`.
3. Create on one host and join the second using the URI.
4. Verify both `systemd --user` services persist after terminal closure and reconnect.
5. Capture CPU backend/dtype/memory, authenticated reachable endpoints, both directed measurements, artifact
   transfer, exact tokens, and speed/capacity decisions.
6. Produce evidence schema 2 and validate with
   `--linux-x86-physical-config <configuration.json>`.

If two Linux x86-64 machines are unavailable, status is `NOT_RUN`.

## macOS ARM64 MPS

Use an Apple Silicon Mac and a distinct authenticated peer. The Mac must pass an operational MPS
tensor and benchmark probe; hardware/OS identity alone is insufficient. Install with the POSIX
wheel installer, verify LaunchAgent persistence, capture MPS memory budgeting and direct link
evidence, then validate with `--macos-arm64-physical-config <configuration.json>`.

Hosted macOS CI can provide build/software status, but only real MPS execution produces physical
validation. If unavailable, status is `NOT_RUN`.

## Linux ARM64 CPU

Use two distinct ARM64 Linux machines (or one ARM64 coordinator and a distinct compatible peer).
An x86 emulator/build is not physical ARM64 validation. Install the wheel without Git, verify
`systemd --user`, CPU benchmark/memory selection, direct links, artifacts, and exact tokens. Use
`--linux-arm64-physical-config <configuration.json>`.

If real ARM64 hardware is unavailable, a build/emulation check may be reported only as
implemented-unvalidated and the physical gate remains `NOT_RUN`.

## Configuration contract

Each configuration names coordinator/worker hosts, non-loopback endpoint, identity evidence,
immutable model revision, and evidence directory. The validator resolves hosts, rejects shared
addresses/process namespaces, validates exact token and route evidence, rejects manual low-level
bootstrap commands, and verifies every source-file SHA-256.

Never include pairing secrets, private keys, raw proofs, session/AES keys, or prompt contents in
evidence. Redact the URI and retain only its public session ID and consumed result.
