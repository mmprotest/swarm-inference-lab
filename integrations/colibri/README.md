# Colibri integration

Swarm pins [JustVugg/colibri](https://github.com/JustVugg/colibri) release
`v1.4.0` at commit
`b085b48888a88d9a1c00b151a9979774b72cdbfd`. Updates are explicit; builds
never follow a moving branch.

The versioned patch series contains only architecture-neutral integration work:

- `0001-swarm-bridge.patch`: structured bridge events and capability exchange.
- `0003-aggregate-runtime-telemetry.patch`: aggregate runtime observations.
- `0009-generic-sparse-moe-component.patch`: a family-neutral routed-expert ABI
  whose tensor semantics come from Swarm architecture adapters.

The canonical hybrid path does not require a second CUDA context. A healthy
native stage may advertise the embedded, pin-bound Colibri component, which
uses the same selected PyTorch device, streams adapter-described expert tensors
from the immutable stage artifact, and maintains a bounded persistent expert
LRU. It claims routed-expert execution and storage tiering only; attention,
KV state, embeddings, normalization, the language-model head, sampling, and
token publication remain explicit components of the complete Swarm plan.

An installer-owned external Colibri runtime is still supported through a
hash-verified runtime manifest. An explicitly configured invalid runtime fails
closed and is never silently replaced by the embedded component.

## Reproducible source build

Initialize the exact checkout first:

```powershell
git submodule update --init --recursive third_party/colibri
```

Windows:

```powershell
.\integrations\colibri\build.ps1 -ApplyBridgePatches
```

POSIX:

```bash
./integrations/colibri/build.sh --apply-bridge-patches
```

Both build scripts verify the pinned commit, apply only the ordered patch
series, copy the architecture-neutral adapter sources, run retained native
tests, and emit hashed build and patch manifests.

## Component contract

The generic sparse-MoE component accepts exact router-selected activations and
route weights and returns the model-required expert contribution. Contracts
carry shape, dtype, device, batch and sequence dimensions, model revision,
token position, and route identity. Packed representations are admitted only
when a matching adapter and kernel describe their semantics; the PyTorch
component fails closed on unrecognized packed tensors.

Steady-state activations use an in-process or direct worker data edge. The
coordinator owns membership, planning, deployment, recovery, telemetry, and
publication coordination but is not an activation relay.
