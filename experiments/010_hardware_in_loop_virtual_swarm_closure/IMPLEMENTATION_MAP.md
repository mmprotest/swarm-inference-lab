# Experiment 010 implementation map

## Extension points

1. `worker.abi` remains the process-control ABI. Experiment 010 adds backend
   neutral `ExpertExecutionRequest` and `ExpertExecutionResponse` payloads and
   uses the `MOE_EXPERT` job role; it does not add a competing worker service.
2. `backends.colibri` remains the source of truth for dependency, model,
   quantisation, route, cache, and storage metadata. CUDA capability is tightened
   to require a saved kernel proof.
3. Experiment 006 `ExpertMicroshardDescriptor` remains the ownership unit. An
   execution status is earned only after a worker loads the described native
   slice and returns a validated partial.
4. The existing data-plane and TCP framing conventions are reused. Direct TCP,
   relay TCP, and shared-memory payload buffers differ only below the semantic
   expert request.
5. Experiment 007's `HeterogeneousPlanner` executes as an additional role and
   non-degradation gate. Experiment 008's hardware identity collector feeds the
   environment record. Calibration and held-out validation IDs are stored
   separately.
6. Experiment 009 atomic artifact writes, checkpoints, failure retention, and
   null handling are retained by the 010 evidence bundle.

## Deliberate extension boundaries

- Experiment 009's fixed-replay tuner requires genuine input and output token
  IDs. The current expert RPC probe ends at an operator vector, so invoking the
  tuner with invented token results would be invalid. The tuner remains
  deferred until the Colibri redirection hook continues generation; Gate 4
  therefore fails.
- The existing `transport.FaultProxy` wraps stage-level
  `ActivationRequest`/`ActivationResult` messages. It cannot select an expert,
  terminate an expert worker, or corrupt the expert response schema. Experiment
  010 adds those controls at the expert semantic boundary and records the
  incompatibility rather than claiming direct reuse.

## Colibri bridge changes

- Add a fifth auditable patch only if the pinned OLMoE engine can redirect a
  coalesced layer request without altering routing or model math.
- The hook boundary is after router top-k selection and before expert weight
  lookup. One request carries all experts owned by one destination.
- Absence of this hook is an explicit failed gate. A Python expert fixture or a
  low-level CUDA kernel proof cannot satisfy end-to-end Colibri RPC.
- The upstream pin remains v1.4.0 at
  `b085b48888a88d9a1c00b151a9979774b72cdbfd`.

## Protocol invariants

- Canonical JSON metadata plus length-delimited tensor blobs; never pickle.
- Request identity, model revision, quantisation fingerprint, deadline, and
  deterministic mode are validated before execution.
- Responses bind result bytes to worker and model fingerprints with SHA-256 and
  a worker signature.
- Multiple selected experts sharing a worker are coalesced into one request.
- Exact mode uses raw FP32 transport and fixed-order FP32 reduction.
- Network shaping delays or rejects the actual bytes before socket writes.

## Evidence boundary

Every row contains exactly one Experiment 010 evidence category. Native compute,
single-host process execution, shaped real messages, fixtures, calibrated
simulation, uncalibrated simulation, and manual projections are never merged in
one headline value.
