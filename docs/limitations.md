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
- Legacy execution paths still relay activations through the coordinator. The
  OLMoE product stage ring sends intermediate activations directly between
  workers.
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
- Windows is the exercised native development platform. Linux x86-64,
  Linux ARM64, and macOS code paths are implemented and tested where they can
  be tested without those kernels, but still require execution on each target.
- Physical LAN/WAN artifacts require another actual machine and have not yet
  been produced in this repository build.
- Qwen3 MoE execution is not yet implemented. Synthetic MoE-like routing is a
  scheduler/fan-out proxy only.
- Kimi K3 support is absent until independently validated. Index analysis is
  not model execution support.
- The system does not provide prompt privacy, activation privacy, protection
  against colluding malicious workers, Byzantine fault tolerance, or
  cryptographic proof of correct neural computation.
