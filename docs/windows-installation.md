# Install Swarm Inference on Windows

Swarm Inference supports a normal per-user setup on Windows 11 x86-64. You do not need an
administrator account, Git, Python, `uv`, a wheel, PowerShell installation commands, or a source
checkout.

## Install

1. Open [Swarm Inference releases](https://github.com/mmprotest/swarm-inference-lab/releases).
2. Select the intended release and download `SwarmInferenceSetup-x64.exe`.
3. Double-click the downloaded file.
4. Review the licence and complete the setup wizard.
5. Open a new PowerShell or Command Prompt window.
6. Run:

```powershell
swarm --version
swarm node doctor
```

Already-open terminals do not see the new user PATH until they are reopened. A release candidate
may be explicitly labelled `unsigned prerelease`; Windows will show the corresponding publisher
warning. Do not treat an unsigned prerelease as a signed public release. Stable releases cannot be
published by the release tooling without valid timestamped Authenticode signatures.

## CPU and CUDA selection

Automatic setup probes `nvidia-smi` with a timeout. A successful probe makes CUDA a candidate,
not a conclusion. Setup installs the exact CUDA lock into a staging runtime and runs the installed
`swarm node doctor --json`. CUDA is selected only when doctor reports `torch-cuda` and a real CUDA
tensor operation passes. If an automatic CUDA candidate fails, setup discards it and installs the
locked CPU profile. `/BACKEND=cuda` is strict and fails instead of falling back;
`/BACKEND=cpu` is the troubleshooting and CPU-acceptance override.

The online installer downloads only artifacts named and hashed by its release-generated lock.
The Swarm application wheel, pinned `uv.exe`, bootstrapper, profiles, and release manifest are
inside setup and verified before use.

## Files and state

Application files are installed under:

```text
%LOCALAPPDATA%\Programs\SwarmInference\
```

Durable cluster identity, membership, trust, model cache, and evidence remain separate under:

```text
%LOCALAPPDATA%\SwarmInference\
```

Setup does not create a node service. `swarm cluster create` or `swarm node join` creates the
cluster-specific current-user service only after valid membership exists.

## Create and join a cluster

On the RTX PC:

```powershell
swarm cluster create --name villani-home
```

Paste the single command it prints into the independently installed CPU laptop:

```powershell
swarm node join "swarm://<private-address>:<port>/join/<single-use-data>"
```

Then, on the PC:

```powershell
swarm cluster status
swarm run <model> ...
```

The invitation is single-use and expires. Human setup needs no invitation file. JSON automation
remains secret-free and uses an owner-protected file.

## Repair, update, and uninstall

Run the same setup version again to repair it. Install a newer setup to perform an in-place,
transactional upgrade. The previous runtime and installation record are restored if the new
runtime, doctor, state compatibility, or service readiness fails. An older version is refused
unless the documented recovery-only `/ALLOWDOWNGRADE=1` setup flag is explicit.

For a native installation, `swarm update` checks the fixed GitHub repository and launches a
verified setup. Update checks are manual; there is no background updater or telemetry.

Use Windows **Installed apps** to uninstall. Normal uninstall removes application files, the
owned user PATH entry, shortcuts, registration, and Swarm-owned scheduled-task definitions while
preserving cluster state. The wizard offers an explicit purge-state checkbox; silent removal uses
`/PURGESTATE=1`. Purging cannot be undone.

## Silent setup for managed acceptance

These options are intended for administrators and CI, not the ordinary user path:

```powershell
SwarmInferenceSetup-x64.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /BACKEND=cpu /PURGESTATE=0 /LOG=C:\Path\setup.log
```

Setup returns failure when the native bootstrapper fails. The failure dialog identifies the
bootstrapper log; setup never masks an unsuccessful dependency or doctor operation as success.
