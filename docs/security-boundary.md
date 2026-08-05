# Product security boundary

The current product is intended for a trusted LAN or private network. Identity and signatures
authenticate control decisions and peers; they do not make plaintext model traffic safe for the
public Internet.

## Identity generation and storage

`swarm identity create` generates an Ed25519 keypair using the canonical security module and
writes a versioned JSON identity document. The document records its role (`coordinator` or
`worker`), public key, SHA-256 public-key fingerprint, UTC creation time, and format version. The
private key is stored in the same identity document with owner-only permissions where the host
filesystem supports them.

The CLI refuses to overwrite an identity unless `--force` is explicit. `identity show` and
`identity fingerprint` validate the complete document but expose only public metadata. Private
key bytes must never be copied into logs, tickets, trust stores, or acceptance summaries.

Legacy unencrypted PEM identities remain readable for compatibility. New product bootstrap uses
the versioned JSON format.

## Worker trust

With `require_trusted_workers: true`, a signed registration is accepted only when its public-key
fingerprint is present in either:

1. the static `trusted_worker_fingerprints` configuration list; or
2. the versioned coordinator trust store at `trust_store_path`.

The sources are normalized, deduplicated, and combined deterministically. The trust store is
atomically replaced, audited, and reloaded for each decision, so a malformed update cannot
replace the last valid in-memory registration state. Use:

```powershell
swarm identity trust --coordinator-state .swarm/coordinator `
  --identity .swarm/identities/worker-1.json --label worker-1
swarm identity untrust --coordinator-state .swarm/coordinator `
  --identity .swarm/identities/worker-1.json
swarm identity list-trusted --coordinator-state .swarm/coordinator --json
```

An unknown registration is rejected with the public fingerprint and a remediation command. The
private key is never logged.

Revocation policy is deliberately bounded: an already admitted session may finish on its
installed route, because abruptly invalidating in-flight authenticated peers would turn a trust
administration action into partial output. Trust removal blocks fresh registration, new session
admission, route deployment, and replacement-route installation. Cancel active sessions first if
immediate operational removal is required.

## Coordinator trust

Workers pin the coordinator public-key fingerprint with
`--trusted-coordinator-fingerprint`. Create and inspect the coordinator identity before startup,
then provision the fingerprint to each worker over an authenticated administrative channel. A
worker must not learn this fingerprint from the unauthenticated coordinator connection it is
about to trust.

## Signed route leases

The coordinator signs each route lease with Ed25519. The signed content binds:

- coordinator and worker identities;
- topology and monotonically increasing route generation;
- model, tokenizer, dtype, and exact stage assignment;
- control and data endpoints;
- lease expiry; and
- a bounded-use nonce.

Workers verify the signature, pinned coordinator, time bounds, assignment, and nonce before
installing a route. A changed endpoint or assignment invalidates the signature.

## Peer handshakes and replay protection

After a signed route is installed, direct stage peers exchange signed handshake material and
verify that the presented worker key and endpoint match the lease. Per-edge sequence validation,
session identity, request generation, token position, and route generation reject duplicate,
out-of-order, cross-session, and stale-route traffic. Bounded nonce caches reject registration,
lease, and handshake replay in their respective scopes.

Replay protection is process- and durable-state scoped as documented by each protocol. It is not
a global consensus ledger and does not prevent a compromised endpoint from withholding service.

## What checksums and authentication protect

Stage-ring SHA-256 checksums detect truncation, accidental corruption, and a frame whose bytes do
not match its declared checksum. They are unkeyed and therefore do not authenticate a malicious
sender.

Ed25519 signatures authenticate possession of a provisioned identity key and bind signed
metadata. They do not prove that a neural computation was performed correctly, that the host is
uncompromised, or that multiple identities are independent physical operators.

Neither mechanism supplies payload confidentiality. Stage-ring activation, token, expert, and
control traffic is not protected by validated TLS in the current product. Network observers and
participating workers may inspect data available to them. Sensitive prompts and proprietary
weights require an independently secured network boundary.

## Rotation and revocation

There is no live key-rotation protocol. Rotate a worker safely by draining/cancelling its active
sessions, creating a new identity at a new path, adding the new fingerprint, restarting the
worker with that identity, verifying registration, and then removing the old fingerprint. A
persisted worker ID cannot silently switch keys; use an explicit new worker identity/ID or the
documented administrative migration.

Coordinator rotation requires unloading routes and stopping workers, creating a new coordinator
identity, distributing its fingerprint through an authenticated channel, and restarting every
worker with the new pin. Do not use `--force` as an unattended rotation mechanism.

## Audit logs

Coordinator durable state records trust rejections, registrations, deployments, recoveries, and
request lifecycle events. The trust store has a separate append-only JSONL audit containing the
event, public fingerprint, label/notes, timestamp, and path. Audit files aid investigation but
are local files, not tamper-evident remote attestation. Restrict access to the entire coordinator
state directory and back it up as one unit.

## Explicit non-goals

The current product does not claim public-Internet safety, payload encryption, anonymous worker
admission, Byzantine fault tolerance, collusion resistance, confidential computing, remote
attestation, or cryptographic proof of inference. Deploy only where the trusted-network and
operator assumptions are acceptable.
