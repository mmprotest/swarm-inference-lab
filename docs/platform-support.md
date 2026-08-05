# Platform support

Support status is recorded per OS, architecture, and selected backend. Hardware presence alone
is not validation; a real tensor correctness and benchmark probe must pass.

| OS / architecture | Backend | Product state | Service manager | Physical status |
|---|---|---|---|---|
| Windows x86-64 | CPU | validated | Task Scheduler, current user | software validated |
| Windows x86-64 | CUDA | implemented-unvalidated | Task Scheduler | RTX 5090 gate `NOT_RUN` unless supplied |
| Linux x86-64 | CPU | implemented-unvalidated | `systemd --user` | physical gate required |
| macOS ARM64 | MPS, CPU fallback | implemented-unvalidated | LaunchAgent | physical MPS gate required |
| Linux ARM64 | CPU | implemented-unvalidated | `systemd --user` | ARM64 physical gate required |
| Other/32-bit combinations | — | unsupported | foreground diagnostics only | unsupported |

CI is distinct from physical validation. The workflow declares Windows x86-64, Linux x86-64,
macOS ARM64, and Linux ARM64 jobs, builds a wheel, installs it outside the checkout, runs
installer smoke tests, static checks, unit tests, and non-GPU integrations. Hosted CPU jobs do
not imply CUDA validation.

## Backend selection

Priority is operational CUDA on Windows/Linux, operational MPS on Apple Silicon, then operational
CPU. Each candidate record contains detection, actual tensor result, supported dtypes, available
memory, rejection reason, and timestamp. A dtype is advertised only after correctness and the
selected-device benchmark pass.

Default memory budgets are bounded:

- CPU: no more than 75% of currently available RAM and the configured total-RAM fraction;
- CUDA: no more than 85% of free VRAM after a reserve; and
- MPS: no more than 70% of available unified memory after an OS reserve.

## Firewall policy

Only selected TCP control/data/probe ports are considered. Owned rules are labeled with cluster
and node IDs and restricted to private subnets. Existing broader rules are reported, not adopted.
Leave/uninstall removes only owned rules.

The agent never silently elevates. Windows can apply rules only when already elevated; Linux and
macOS return one exact administrator remediation command. Until the coordinator can reach the
advertised endpoints, the node remains `blocked`. No public-profile or public-Internet rule is
created by default.

## State locations

- Windows: `%LOCALAPPDATA%\SwarmInference\`
- Linux: `${XDG_STATE_HOME:-~/.local/state}/swarm-inference/`
- macOS: `~/Library/Application Support/SwarmInference/`

Explicit state roots remain available for tests and repository-local `.swarm` development.
