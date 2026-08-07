# Adapter location

The Python architecture adapters are kept with the first-class backend at
`src/swarm_inference/backends/colibri`. This directory owns only the small,
versioned native ABIs compiled alongside the exact Colibri pin:

- `swarm_expert_wire.*` is the distributed expert transport ABI promoted from
  Experiment 010;
- `swarm_moe_runtime.*` is the architecture-neutral FP32/BF16 and symmetric
  packed-INT4-G32 SwiGLU component ABI used by Swarm adapters for standard
  routed-expert layouts;
- `kimi_mxfp4_runtime.c` is the Kimi MXFP4 validation bridge over Colibri's
  native quantization primitive.

Tensor names, router rules, and family identities do not enter the generic C
runtime. They are supplied by the matching architecture adapter.
