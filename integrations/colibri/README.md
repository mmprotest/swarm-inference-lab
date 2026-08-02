# Colibri integration

This directory contains the auditable integration layer between
`swarm-inference-lab` (the control plane) and Colibri (the local inference
engine). The upstream source is a Git submodule at `third_party/colibri` and is
pinned to the immutable revision below.

| Field | Value |
|---|---|
| Repository | `https://github.com/JustVugg/colibri` |
| Release | `v1.4.0` |
| Commit | `b085b48888a88d9a1c00b151a9979774b72cdbfd` |
| License | Apache-2.0 (`third_party/colibri/LICENSE`) |

The build never edits the submodule. It exports the pinned source to a build
directory, verifies the revision, applies every patch listed in
`patches/series`, fails on a rejected patch, builds the four model-family
engines, runs bridge tests, and writes source and binary fingerprints.

Stock mode uses the unmodified export. Bridge mode adds only dedicated NDJSON
event channels, token-ID observations, route/history export, and aggregate
OLMoE cache, storage, prefetch, residency, resource, and phase counters. It does
not change weights, tokenization, routing, sampling, expert selection,
quantization, or model math.

## Build

```powershell
.\integrations\colibri\build.ps1 -ApplyBridgePatches
```

On Linux/macOS:

```bash
./integrations/colibri/build.sh --apply-bridge-patches
```

The Windows build accepts `-MakePath` when MinGW `make` is not on `PATH`.
Artifacts land under `build/colibri` by default and include the upstream
license, the applied patch manifest, and `colibri_build.json`.

The fast C fixture is generated deterministically from Colibri's upstream GLM
oracle script. It uses an isolated Transformers 5.14.1 target because the
project environment intentionally remains on Transformers 4.x:

```powershell
.\integrations\colibri\tests\prepare_fixture.ps1
```

The experiment reproduction script invokes this command when the fixture is
absent. Fixture hashes and the upstream teacher-forced oracle floor are recorded
in `fixture_manifest.json`; fixture evidence is never accepted as an official
performance result. Exact generation correctness is checked by comparing the
direct binary with the universal-worker adapter over identical input IDs.

## Updating the pin

Updating is deliberately explicit. Review the upstream release and license,
then run:

```powershell
.\integrations\colibri\update-pin.ps1 -Release <tag> -Commit <40-hex-commit>
```

That command fetches and checks out only the requested revision. It does not
edit the recorded constants; update those in the same reviewed change, refresh
the patch series, and rerun all build and bridge tests.

## Runtime

The Python adapter lives in `src/swarm_inference/backends/colibri`. Schemas in
`schemas/` are the stable file/wire contracts used by both stock telemetry
translation and bridge mode. `COLI_SWARM_BRIDGE=1` activates bridge emission;
`COLI_SWARM_BRIDGE_PATH` selects the dedicated NDJSON file and
`COLI_SWARM_TELEMETRY` selects `off`, `summary`, `detailed`, or `trace`.
Bridge-mode OLMoE runs may additionally set `COLI_USAGE_PATH` for an isolated
usage-history file and `COLI_HOT_PIN_PATH` for a validated per-layer hot-expert
bitmap. Those controls are ignored in stock mode and never change router output.
