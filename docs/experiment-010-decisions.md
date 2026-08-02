# Experiment 010 architecture decisions

## ADR-010-1: one worker ABI

Status: accepted.

Expert work is an additive Universal Worker ABI role. Process discovery,
identity, capability negotiation, cancellation, heartbeat, and shutdown stay on
the existing service. The expert wire schema is independently versioned so
binary tensor framing can evolve without changing process control.

## ADR-010-2: stable coordinator owns routing

Status: accepted.

The coordinator owns tokenization, dense state, routers, request state, output
head, dispatch, deterministic reduction, verification, and fallback. Expert
workers own only explicitly inventoried whole experts or matched intermediate
slices. A worker never decides routing.

## ADR-010-3: capability proof is executable

Status: accepted.

A CUDA DLL, build flag, or visible GPU is necessary but insufficient. Colibri
advertises CUDA only when a proof record binds the DLL hash and GPU identity to
a successful real expert kernel with a CPU numerical comparison. Requested CUDA
fails closed when the proof is missing or stale.

## ADR-010-4: data paths share semantics

Status: accepted.

Direct TCP, relayed TCP, and shared-memory buffers transport the same canonical
expert request. This makes relay and serialization taxes comparable and prevents
data-plane selection from changing model semantics.

## ADR-010-5: native microshards

Status: accepted.

Microshards slice matched up, gate, and down projections along the intermediate
dimension. Native packed bytes and scale groups remain authoritative. Temporary
dequantisation of only the executing slice is allowed; persistent full-expert
re-encoding is not.

## ADR-010-6: fail-closed evidence

Status: accepted.

Quick and development runs can validate software but cannot award an official
verdict. A failed physical capability, missing model, absent measurement, or
uncalibrated simulator remains failed or null. Projections cannot backfill a
measured gate.
