# Stage-ring protocol

The stage-ring protocol carries control messages, token IDs, and stage-boundary
tensor payloads over a persistent byte stream. Version 1 uses the product magic
`SWRING01`; it is independent of the archived experiment protocols and is not
the Experiment 010 `SWARMEX1` format.

## Frame layout

Every frame is exactly:

```text
76-byte fixed header | canonical UTF-8 JSON metadata | opaque payload bytes
```

The fixed header is the little-endian struct
`<8sHHIIQQihh32s`:

| Offset | Bytes | Field | Type and meaning |
| ---: | ---: | --- | --- |
| 0 | 8 | magic | ASCII `SWRING01` |
| 8 | 2 | version | unsigned protocol version; currently `1` |
| 10 | 2 | operation | unsigned operation code from the table below |
| 12 | 4 | flags | unsigned operation flags; currently caller-defined |
| 16 | 4 | metadata length | unsigned canonical-JSON byte count |
| 20 | 8 | payload length | unsigned opaque-payload byte count |
| 28 | 8 | sequence number | unsigned directed-session-edge sequence |
| 36 | 4 | token position | signed position; `-1` is available to control messages |
| 40 | 2 | source stage | signed stage ID; `-1` denotes the coordinator |
| 42 | 2 | destination stage | signed stage ID; `-1` denotes the coordinator |
| 44 | 32 | checksum | SHA-256 of `metadata || payload` |

The metadata is serialized with sorted keys, compact separators, UTF-8, and no
NaN or infinity values. It contains the protocol version, message type, model
and tokenizer revisions, topology ID, addressed stage and layer interval,
session and request IDs, sequence and token positions, source and destination,
tensor shape and dtype, compression mode, payload length, status, and a JSON
attributes object. Header fields duplicated in metadata must agree exactly.
Unknown, missing, duplicated, or malformed metadata fields are rejected.

Raw tensor bytes are never embedded in the JSON metadata.

## Operations

| Code | Operation | Purpose |
| ---: | --- | --- |
| 1 | `HELLO` | establish protocol intent on a connection |
| 2 | `CAPABILITIES` | describe supported operations and runtime properties |
| 3 | `LOAD_STAGE` | request or acknowledge stage materialization |
| 4 | `OPEN_SESSION` | create per-session execution state |
| 5 | `PREFILL` | execute a prompt or forward its boundary activation |
| 6 | `DECODE` | execute one decode step or forward its boundary activation |
| 7 | `VERIFY_CANDIDATES` | verify speculative candidates exactly |
| 8 | `TOKEN_RESULT` | return a selected token |
| 9 | `SESSION_CHECKPOINT` | describe a session checkpoint |
| 10 | `CLOSE_SESSION` | close a completed session and release state |
| 11 | `CANCEL_SESSION` | cancel a session and release state |
| 12 | `HEALTH` | request or return health state |
| 13 | `ERROR` | report an operation failure |

Unknown operation codes are rejected. The operation code and metadata name must
identify the same operation.

## Integrity and limits

The checksum covers the exact metadata bytes followed by the exact payload
bytes. Receivers compare it with SHA-256 using a constant-time digest
comparison. A checksum failure, declared/actual length mismatch, unsupported
version, invalid magic, malformed field, or oversized component rejects the
whole frame.

The version-1 limits are:

- metadata: 4 MiB;
- payload: 1 GiB;
- source and destination stage IDs: `-1` through `32767`;
- stage IDs in semantic metadata: `0` through `32767`;
- sequence numbers: unsigned 64-bit;
- token positions: `-1` through signed 32-bit maximum.

Socket helpers handle partial reads and partial writes. Receive buffers may be
drawn from a bounded reusable pool; pool capacity bounds concurrent buffer
ownership, while a retained buffer can grow to the largest accepted frame
component.

## Sequence semantics

Sequence numbers are scoped to `(session ID, source stage, destination stage)`.
The first observed number establishes the edge state. Every subsequent frame
on that edge must be exactly the previous value plus one. Equal numbers are
duplicates, lower numbers are stale, and higher non-consecutive numbers are
gaps; all three cases are rejected. Closing a session permits its allocator and
validator state to be reset explicitly.

Persistent connections can carry any number of consecutive frames. Frame
boundaries come only from the fixed header lengths, not from socket read
boundaries.

## Session semantics

`HELLO`, `CAPABILITIES`, and `LOAD_STAGE` occur before ordinary session state is
required. A non-empty session ID is opened exactly once. Prefill, decode,
verification, token result, checkpoint, close, cancellation, health, and error
messages validated in a session context must refer to an active session.
Closing or cancelling removes that active identity; later data-plane messages
for it are rejected.

Session validation also pins the model revision and topology identity, and can
pin the tokenizer revision, destination stage, and owned layer interval. This
prevents a valid frame from being applied to the wrong loaded stage or topology.

## Tensor payload codec

Stage-ring tensors use `swarm_inference.transport.stage_tensor`. It produces a
canonical row-major, little-endian byte representation even for non-contiguous
or strided PyTorch views. Dtype, shape, byte order, raw and encoded lengths,
SHA-256 checksum, codec identity, and measured codec timings remain semantic
message attributes. The byte payload stays separate. Compression is opt-in;
adaptive mode selects the lossless byte-shuffle/fast-codec path only when the
measured expected latency is lower.

The older `swarm_inference.protocol.tensor_codec` serves a different contract:
it creates a self-contained `SWARMT01` activation envelope for the existing
activation transport, uses a big-endian header-length prefix, and carries NumPy
array metadata inside that envelope. Neither codec is a fallback for the other,
and their magic values, headers, and metadata contracts must not be mixed.

## Route and peer authentication

Before a product worker installs a route, it verifies a finite-lived route
lease signed by the coordinator's persistent Ed25519 identity. The worker must
be configured with that coordinator key's SHA-256 fingerprint. The signed
lease binds the topology and route generation, exact model and tokenizer
revisions, adapter and dtype, ordered stage assignments, worker IDs and public
key fingerprints, control and data endpoints, issue and expiry times, and a
random nonce. Workers reject missing or invalid signatures, unknown
coordinators, expired or excessively future-dated leases, reused nonces, stale
generations, and any local identity, endpoint, model, tokenizer, dtype, or
assignment mismatch.

Each direct worker connection begins with a signed `HELLO` handshake. It binds
both stage identities, worker identity and public-key fingerprint, topology,
route generation, model revision, timestamp, one-time nonce, and the hash of
the installed coordinator-signed lease. Both peers check the other against the
installed route before accepting execution frames. A bounded nonce cache,
strict per-session sequence numbers, request generations, route generations,
and lease expiry reject replayed or stale participation. Frames from an older
route generation are rejected after recovery.

## Security boundary

The frame format is transport-independent. Canonical non-loopback stage-ring connections run
inside mutually authenticated TLS 1.3 with cluster-CA validation and a durable node-fingerprint
pin. Remote plaintext is rejected before a `HELLO` or tensor frame is accepted. Address changes
do not alter the stable identity name or fingerprint pin.

The frame SHA-256 value remains an unkeyed accidental-corruption check, not malicious tamper
resistance. TLS supplies in-transit confidentiality and integrity; signed route leases and peer
handshakes bind authorization to the deployed topology. Neither proves that a participating
worker computed the requested neural operation correctly, and a legitimate stage owner can
inspect the state assigned to it.
