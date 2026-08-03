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
engines plus the native OLMoE expert worker, runs bridge tests, and writes
source and binary fingerprints.

Stock mode uses the unmodified export. Bridge mode adds dedicated NDJSON event
channels, token-ID observations, route/history export, aggregate OLMoE counters,
and the Experiment 010 shared native expert runtime. The runtime is the single
implementation used by both `olmoe` and `olmoe_expert_worker`; it preserves the
merged int8 bytes and F32 row scales. Patch 0006 inserts external dispatch after
the native router and before its contribution reaches the residual stream. The
canonical byte-compatible `SWARMEX1` adapter is copied from `adapter/` into the
exported source tree at build time. The downstream patches do not alter model
weights, tokenization, routing, sampling, expert selection, or quantization.

Patch 0007 adds native int8 OLMoE microshards. Gate and up rows and the matching
down columns retain the source bytes and row scales. In exact mode, workers run
in the plan's recorded `reduction_order`: each non-final shard returns the
unscaled F32 down-dot accumulator, the next shard resumes that accumulator, and
only the final shard applies the original down row scale. This preserves the
same group and accumulation order as the local merged expert. Fast mode keeps
independently scaled worker partials as a separately measured quality-bounded
path.

Patch 0008 makes capacity isolation physical rather than a coordinator
allowlist convention: when the plan remotely covers every routed expert,
`olmoe` does not open the local expert runtime and therefore accepts a
dense-only coordinator container containing zero routed-expert tensors. It also
adds a shared read-only memory telemetry module. On Windows this uses
`GetProcessMemoryInfo`, `GetPerformanceInfo`, `GlobalMemoryStatusEx`, and sampled
`QueryWorkingSetEx` queries to report working set, private/commit bytes, total
page faults, system commit pressure, and resident versus nonresident native
expert-cache hits. Windows does not attribute pagefile reads, hard versus soft
faults, or compressed-store ownership through these APIs; those fields remain
explicitly unavailable instead of being filled with zero. Both whole experts
and native microshards use the runtime's bounded LRU cache and ownership checks.

Patch 0008 also connects whole native-int8 OLMoE experts to Colibri's existing
CUDA DLL ABI. `-BuildCuda` builds both `olmoe` and `olmoe_expert_worker` with the
runtime loader. Setting `COLI_SWARM_EXPERT_CUDA_TARGET=all|<layer>:<expert>` and
`COLI_SWARM_EXPERT_CUDA_DEVICE=<ordinal>` makes CUDA mandatory for the selected
worker expert: DLL initialization, exact int8/F32-scale upload, residency, or
kernel failure fails the request, with no CPU fallback. Telemetry records the
target, tensor residency, upload time, PCIe/kernel timing, execution count, and
the fallback count. The environment is opt-in and does not affect local mode.

Patch 0008 additionally makes `COLI_SWARM_EXPERT_DATA_PLANE` select
`direct_tcp`, `relayed_tcp`, or `shared_memory`. All three carry the same
canonical `SWARMEX1` request and response bytes. Shared memory uses named
file mappings only after a `SWARMEX1` control handshake declares the exact
mapping names and sizes; relay mode shapes the real framed socket payload.
The worker reports mutex wait separately from native compute time so
concurrent decode queueing is measurable. OLMoE attention scratch is sized
from the loaded container's `max_position_embeddings` rather than a fixed
4096-element stack array, without changing scalar attention order.

Native expert modes are selected with `COLI_SWARM_EXPERT_MODE=local|rpc|hybrid|planner`.
Non-local modes require absolute `COLI_SWARM_EXPERT_PLAN` and
`COLI_SWARM_EXPERT_TELEMETRY` paths. Exact mode returns an unweighted contribution
for each selected router rank in one response per worker, then applies the
original routing weight and accumulates ranks with the same shared scalar
primitive as local Colibri. This avoids an extra float rounding point while
retaining coalesced transport.

For correctness audits, `COLI_SWARM_NUMERIC_TRACE=<absolute path>` enables an
observation-only binary trace in the real OLMoE step path. Records use the
`COLNUM1` format and capture post-MoE hidden states, exact selected routing
weights, and pre-sampling logits as native float32 bytes. The trace does not
execute shadow experts, change routing, or participate in sampling; it exists
only to locate the first numerical divergence between an exact-container local
run and a distributed run.

The CUDA build also emits `coli_kimi_mxfp4.dll`. The dense Kimi K3-shaped
fixture calls the shared native `quant.h` MXFP4 arithmetic through that DLL,
processes every quantization group, and disables zero-group skipping. This is
labelled `SYNTHETIC_FIXTURE` unless official checkpoint bytes are supplied; it
is not full Kimi K3 inference.

## Build

```powershell
.\integrations\colibri\build.ps1 -ApplyBridgePatches
```

For the native Windows CUDA DLL plus CUDA-enabled OLMoE engine and worker:

```powershell
.\integrations\colibri\build.ps1 -ApplyBridgePatches -BuildCuda
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
