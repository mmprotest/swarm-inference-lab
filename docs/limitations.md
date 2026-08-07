# Limitations

- Single-host loopback does not prove physical scaling.
- Simulation results depend on assumed profiles and deterministic abstractions.
- The WAN model cannot capture every transport, congestion, NAT, routing, and
  residential availability effect.
- Stage-input replay has high storage, bandwidth, latency, and compute overhead.
- Untrusted workers can inspect activations and local cache state.
- Integrity audits are probabilistic and do not prevent collusion.
- A central coordinator is a control-plane and token-commit dependency.
- Slow devices may have zero or negligible useful contribution and can remain
  idle.
- Single-request speed can decline while aggregate throughput rises.
- Canonical native distributed execution uses persistent direct stage-to-stage
  transport; the coordinator remains a control-plane and token-commit
  dependency and never relays steady-state hidden activations.
- Canonical remote product transports use authenticated TLS 1.3, but a legitimate worker can
  inspect the weights, activations, and KV/cache state assigned to it.
- TLS does not provide NAT traversal, denial-of-service resistance, computation verification,
  or privacy from an admitted node. Routing and least-privilege firewall exposure remain
  operator responsibilities.
- Microbatch grouping is intentionally conservative and does not yet implement
  every backend-specific cache layout.
- `Qwen/Qwen3-0.6B` correctness has been validated in native Windows CPU and
  RTX 5090 CUDA single-host loopback. Those results prove split correctness,
  not hardware throughput scaling.
- The real-model loopback validator is a correctness gate, not a sustained
  throughput experiment with the complete standard report matrix.
- CUDA compatibility depends on the installed driver, PyTorch build, and GPU
  architecture.
- Windows x86-64, Linux x86-64, Linux ARM64, and macOS ARM64 adapters are implemented.
  Implementation is not validation: every backend's software and physical status remains
  `not-run` on a clean machine until exact retained evidence is attached.
- Physical LAN/WAN artifacts require another actual machine and have not yet
  been produced in this repository build.
- Pairing is only onboarding; durable cluster certificates protect subsequent traffic. It does
  not add NAT traversal or anonymous public participation.
- JSON/NDJSON never contains the pairing URI. Automation must protect and later retire the
  invitation file; a user-scoped ACL fallback can be weaker than an explicit Windows SID ACL and
  is reported as a limitation in the delivery receipt.
- User-scoped firewall automation cannot silently elevate. A node remains
  blocked until the exact remediation is reviewed and reachability passes.
- Firewall resources are isolated per cluster/node, but broader rules created by other software
  remain an operator responsibility; Swarm Inference reports and does not delete them.
- Stage-artifact source resolution may download one complete immutable snapshot
  on the source node; participating stage nodes receive only owned artifacts.
- Stale or unmeasured links are excluded from automatic distributed plans and
  are never labeled measured throughput.
- Native Qwen3 MoE execution supports the Transformers `qwen3_moe` Safetensors representation.
  Qwen3.5/Qwen3.6 hybrid Safetensors have an architecture profile and Colibri routed-expert
  component but no complete hybrid attention/KV execution plan yet. Compatible GGUF artifacts
  require an installed llama.cpp build that proves the exact loader and quantization.
- Kimi K3 and current GLM MoE have complete pinned-Colibri adapter paths, but heavyweight
  real-model and physical-distributed validation remains `NOT_TESTED` on a clean checkout.
- Kimi K2, DeepSeek V3/V3.x/R1, MiniMax MoE, Mixtral, Llama 4 MoE, and newer Mistral MoE layouts
  have software-validated architecture/routing/expert descriptions and generic Colibri expert
  components. Their complete Safetensors hybrid plans are still integration gaps.
- Dense Llama, Mistral, and Gemma Safetensors are profileable but do not yet have a generic
  native-stage executor. Runtime-probed GGUF through llama.cpp is the current complete path where
  the installed build and hardware qualify.
- The generic Colibri SwiGLU component supports BF16/F16, adapter-described FP8, and exact
  symmetric INT4-G32 projection decoding when the adapter proves compressed-tensors packing and
  scales. Other packed int4/int8 layouts are rejected until an exact quantization-specific
  kernel is installed; they are not silently dequantized through an unreported fallback.
- Architecture/profile fixtures and numerical expert tests are software evidence, not real-model
  validation. Opt-in gates cannot pass without an executed, immutable, runtime-hash-bound
  evidence bundle.
- The system does not provide prompt privacy, activation privacy, protection
  against colluding malicious workers, Byzantine fault tolerance, or
  cryptographic proof of correct neural computation.
