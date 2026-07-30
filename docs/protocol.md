# Protocol

Control and activation RPCs use gRPC AsyncIO. The checked-in
[`proto/swarm.proto`](../proto/swarm.proto) defines coordinator and worker
services. Validated control models are carried in protobuf `Any` envelopes; no
pickle or arbitrary Python object is transmitted.

## Tensor envelope

Activation tensors use the versioned binary format:

```text
SWARMT01
uint32 big-endian header length
canonical UTF-8 JSON header
raw C-order tensor bytes
```

The header contains:

- dtype and byte order;
- shape;
- tensor, request, and stage identifiers;
- token position and sequence length;
- payload length;
- SHA-256 payload checksum.

The decoder rejects bad magic, malformed metadata, length mismatches, invalid
shape/dtype, and checksum mismatches before worker execution.

Payloads larger than `maximum_message_bytes` use bidirectional streaming.
Individual chunks contain message ID, zero-based index, total chunk count, total
serialized length, bytes, and SHA-256 checksum. Missing, duplicated,
inconsistent, or corrupt chunks are rejected. Both request activations and
result activations are chunked.

## RPC surface

- `Coordinator.Register`: signed capability and measured benchmark.
- `Coordinator.Heartbeat`: signed liveness, queue depth, and assignments.
- `Coordinator.Submit`: prompt/token request and terminal response.
- `Worker.Assign`: exact stage definition, revision, shard hash, and model
  metadata.
- `Worker.Execute`: bounded unary activation operation.
- `Worker.ExecuteStream`: chunked bidirectional operation.
- `Worker.Cancel`: remove request-local state.
- `Worker.Health`: loaded stages, health, and queue depth.

The initial transport is insecure gRPC at the channel layer. Ed25519 protects
envelope authenticity, not confidentiality. Physical experiments should use a
trusted LAN or add TLS termination; Internet exposure is unsafe.

