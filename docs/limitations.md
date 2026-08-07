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
- Channel confidentiality is not implemented by the initial insecure gRPC
  deployment; envelope signatures do not provide privacy.
- Product route leases and stage-ring peer handshakes authenticate
  participation, but product stage-ring TCP payloads are not encrypted. The
  supported deployment boundary is a trusted LAN or private network; untrusted
  Internet participation remains unsupported.
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
- Pairing protects onboarding, not inference payload confidentiality. It does
  not add NAT traversal or public-Internet participation.
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
- Qwen3 MoE execution is not yet implemented. Synthetic MoE-like routing is a
  scheduler/fan-out proxy only.
- Kimi K3 support is absent until independently validated. Index analysis is
  not model execution support.
- The system does not provide prompt privacy, activation privacy, protection
  against colluding malicious workers, Byzantine fault tolerance, or
  cryptographic proof of correct neural computation.
