# Platform implementation and validation

Platform implementation describes whether an adapter and product path exist. Validation is a
separate, backend-specific claim that requires retained acceptance evidence. Detection of an OS,
architecture, accelerator, or successful tensor probe never changes software or physical
validation from `not-run`.

The following table is checked against `PLATFORM_IMPLEMENTATION_CONTRACT` in the runtime:

| Runtime system | Accepted normalized architectures | Implementation status |
|---|---|---|
| `windows` | `amd64,x86_64` | implemented |
| `linux` | `aarch64,amd64,arm64,x86_64` | implemented |
| `macos` | `aarch64,arm64` | implemented |

Other architecture combinations are `unsupported`. These rows make no validation claim.

## Retained validation evidence

Each backend record carries the exact OS family, OS release, architecture, backend, independent
software and physical statuses, evidence identity/path, validation timestamp, and detail. The
only statuses are `validated`, `failed`, and `not-run`. A clean machine with no retained evidence
reports `not-run`.

| Example scope | Implementation | Software validation | Physical validation |
|---|---|---|---|
| Windows x86-64 / CPU | implemented | evidence-specific or `not-run` | evidence-specific or `not-run` |
| Windows x86-64 / CUDA | implemented | evidence-specific or `not-run` | RTX 5090 gate currently `NOT_RUN` |
| Linux x86-64 / CPU | implemented | evidence-specific or `not-run` | evidence-specific or `not-run` |
| macOS ARM64 / MPS or CPU | implemented | evidence-specific or `not-run` | evidence-specific or `not-run` |
| Linux ARM64 / CPU | implemented | evidence-specific or `not-run` | evidence-specific or `not-run` |

A passed CPU software gate applies only to CPU software in that exact platform scope. It does not
validate CUDA or physical hardware. CI is also distinct from physical validation. Hosted jobs can
prove software contracts but cannot stand in for the RTX 5090 plus laptop gate.

Legacy node-registry schema 1 is migrated to schema 2 without replacing cluster identity,
membership, or trust. Legacy `unsupported` becomes implementation `unsupported`; pending or
implemented-unvalidated values become implementation `implemented` with validation `not-run`.
Legacy `validated` is not promoted without verifiable retained acceptance evidence and receives
an explicit migration note.

## Backend selection

Priority is operational CUDA on Windows/Linux, operational MPS on Apple Silicon, then operational
CPU. Each probe records detection, actual tensor result, supported dtypes, available memory, and
rejection reason. These facts select a usable backend; they are not validation evidence.

Default memory budgets are bounded:

- CPU: no more than 75% of currently available RAM and the configured total-RAM fraction;
- CUDA: no more than 85% of free VRAM after a reserve; and
- MPS: no more than 70% of available unified memory after an OS reserve.

## Service lifecycle

Windows uses a current-user Task Scheduler task, Linux uses `systemd --user`, and macOS uses a
LaunchAgent. Wheel installation is deliberately unpaired and reports
`deferred-until-cluster-create-or-join`. `cluster create` installs/starts the coordinator plus
local-worker service, and `node join` installs/starts the worker service. `--foreground` skips
service installation. `node install-service` remains an idempotent post-pair administrative
command.

## Firewall ownership

Only selected TCP control/data/probe ports and RFC1918 sources are allowed by owned rules. The
resource name is a bounded SHA-256-derived name, so arbitrary labels never reach privileged
commands:

- Linux uses one `swarm_<owner-hash>` nftables table per cluster/node owner.
- macOS uses one `swarm-inference/<owner-hash>` PF anchor per owner.
- Windows uses `SwarmInference-<owner-hash>-control` and `-data` display names.

Status and cleanup target only that resource. Repeated configuration reconciles stale owned
ports/profile/scope instead of duplicating rules. Broader unrelated rules are reported but never
adopted or removed. The agent never silently elevates; Linux and macOS provide exact bounded
administrator remediation, and Windows requires an already elevated process. Until advertised
private endpoints are reachable, the node remains `blocked`.

## State locations

- Windows: `%LOCALAPPDATA%\SwarmInference\`
- Linux: `${XDG_STATE_HOME:-~/.local/state}/swarm-inference/`
- macOS: `~/Library/Application Support/SwarmInference/`

Explicit state roots remain available for tests and repository-local development.
