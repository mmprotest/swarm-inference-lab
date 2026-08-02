# Experiment 009: Colibri-Backed Adaptive Expert Runtime

Experiment 009 integrates pinned Colibri v1.4.0 as the local MoE engine while
`swarm-inference-lab` remains the worker, planning, tuning, and evidence control
plane. The bridge emits isolated machine-readable telemetry and exact token IDs;
it does not alter routing, weights, quantization, tokenization, sampling, or model
mathematics.

Build the exact patched revision and run the generated C fixture:

```powershell
.\integrations\colibri\build.ps1 -ApplyBridgePatches
.\experiments\009_colibri_adaptive_expert_runtime\reproduce.ps1 -Quick
```

Run the official practical-model matrix (downloads the immutable configured
OLMoE revision when it is not already converted):

```powershell
.\experiments\009_colibri_adaptive_expert_runtime\reproduce.ps1 -Full
```

Use `-ModelPath` for a pre-converted model, `-ColibriPath` for a pinned checkout
or built binary directory, `-SkipModelDownload` for an offline run, and
`-OutputDirectory` plus `-Resume` for a stable resumable bundle. A completed
bundle is returned without rerunning its matrix. `-Configuration` is accepted
for diagnostic selection and is recorded in the manifest; omit it for the
complete A-E matrix required by an official verdict.

Quick results are fixture evidence and can only be `PARTIAL`. A full verdict is
computed solely from saved raw evidence and fails closed when real-model tokens,
routing, residency, storage, timing, correctness, or required overhead metrics are
missing. Real tensor microshard execution and distributed Kimi K3 inference remain
explicitly unsupported.

## Measured reference run

The 2026-08-02 native Windows reference run used
`allenai/OLMoE-1B-7B-0125-Instruct` at revision
`b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`. All ten gates passed. Direct and
adapter tokens, stop reasons, router selections, and router weights matched;
median decode regression was 0.37%. The bounded scheduling tuner retained its
baseline, while the separately evaluated routing-aware policy produced a 5.04%
reverse-confirmed held-out gain. This is a single-host CPU result, not a CUDA,
distributed, or Kimi K3 execution claim.
