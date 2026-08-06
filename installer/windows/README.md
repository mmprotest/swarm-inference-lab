# Native Windows installer

`SwarmInferenceSetup-x64.exe` is a per-user Inno Setup package. It embeds the
self-contained `SwarmBootstrap.exe`, the exact application wheel, pinned `uv.exe`,
two hash-locked runtime profiles, and the release manifest. The normal installation
path never invokes `scripts/install.ps1` and does not require Python, Git, `uv`, or
.NET on the target machine.

The application lives under `%LOCALAPPDATA%\Programs\SwarmInference`; durable
cluster identity and state remain separate under `%LOCALAPPDATA%\SwarmInference`.
Inno Setup owns Apps & Features, shortcuts, and uninstall registration. The compiled
bootstrapper owns verified dependency installation, backend validation, PATH, runtime
transactions, service restoration, rollback, and state-preserving removal.

Build from the repository root:

```powershell
uv run --python 3.11 python scripts/build_windows_installer.py
```

Large generated payloads are written below `build/windows-installer` and
`release/generated` and are intentionally ignored. Tool versions and upstream hashes
are controlled by `installer/windows/toolchain.json`.

An unsigned setup is valid only for a clearly labelled prerelease. A stable build is
rejected unless both the bootstrapper and final setup have valid timestamped
Authenticode signatures.

Acceptance-only setup flags are `/BACKEND=auto|cpu|cuda`, `/PURGESTATE=0|1`,
`/ALLOWDOWNGRADE=0|1`, and `/LOG=<path>`. Standard Inno silent flags work. The
compiled bootstrapper has stable machine-readable operations and exit codes; see
[`docs/windows-installation.md`](../../docs/windows-installation.md) for the user
path and [`docs/releasing.md`](../../docs/releasing.md) for release maintenance.
