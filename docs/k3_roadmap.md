# Kimi K3 feasibility gates

Do not load or claim K3 support before all gates are backed by measurements.
`swarm analyse-large-model` reads only configuration and safetensors headers and
sets `claim_of_model_support: false`.

1. **Maximum shard:** the largest executable contiguous stage plus cache and
   runtime margin must remain below a 32 GiB worker cap.
2. **State semantics:** attention, recurrent/linear state, rotary position, and
   cache update behavior must have a tested stage adapter.
3. **Quantised kernels:** the selected weight/activation format must have
   correct, measured CPU/CUDA/MPS kernels on target nodes.
4. **Service rate:** proxy-model physical benchmarks must predict adequate
   capacity for every stage; the slowest stage controls pipeline capacity.
5. **Activation transport:** measured boundary bytes, link serialization,
   latency, jitter, and fan-out must permit positive aggregate throughput.
6. **Recovery:** cache replay or snapshots must have practical storage,
   transfer, and restoration cost.
7. **Node count:** total and largest-layer bytes, redundancy, availability, and
   safety margin must yield a feasible physical node count.
8. **Economics:** measured energy, bandwidth, storage, utilisation, and
   operational complexity must be positive relative to alternatives.

The feasibility report includes total/largest-layer bytes, expert/non-expert
bytes, minimum and safety-margin 32 GiB node counts, activation/cache estimates,
proxy throughput only when measured proxy data is supplied, unsupported
components, and required backend work.

Worldwide per-token remote-expert routing should not be implemented unless a
proxy demonstrates that network fan-out remains competitive with whole-layer
placement.

