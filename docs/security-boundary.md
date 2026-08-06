# Product security boundary

The current product is intended for a trusted LAN or private network. Identity and signatures
authenticate control decisions and peers; they do not make plaintext model traffic safe for the
public Internet.

## Pairing and identity generation

The normal bootstrap path is `swarm cluster create` followed by `swarm node join`. Pairing uses a
random single-use secret, ephemeral X25519, HKDF-SHA256 transcript binding, reciprocal Ed25519
possession proofs, and AES-GCM encrypted completion documents. Invitations expire after ten
minutes by default and have bounded attempts/source rate limits. Consumed, expired, rejected, or
restart-invalidated sessions cannot be revived.

Only public keys and fingerprints cross the network. Successful completion atomically writes
membership/trust on the coordinator and pins cluster/coordinator metadata on the node. Pairing
secrets, raw proofs, ephemeral/session/AES keys, and private keys are excluded from logs, machine
status, and audit documents. See [pairing](pairing.md).

Human cluster creation/pairing may reveal the complete URI once. Machine-readable creation and
pairing each return one non-secret JSON document. The secret is delivered only through an
explicit/default invitation file created from a complete temporary file with atomic publication,
POSIX `0600` or a user-only Windows SID ACL where available, and explicit overwrite refusal.
Secret-bearing URIs are redacted recursively in nested command payloads and scrubbed from error
text. Default files for expired, consumed, or invalidated sessions are retired without deleting
active-session invitations. Invitation contents are never ordinary cluster metadata or service
state.

## Identity generation and storage

Cluster creation/join automatically generates an Ed25519 keypair using the canonical security module and
writes a versioned JSON identity document. The document records its role (`coordinator` or
`worker`), public key, SHA-256 public-key fingerprint, UTC creation time, and format version. The
private key is stored in the same identity document with owner-only permissions where the host
filesystem supports them.

The CLI refuses to overwrite an identity unless `--force` is explicit. `identity show` and
`identity fingerprint` validate the complete document but expose only public metadata. Private
key bytes must never be copied into logs, tickets, trust stores, or acceptance summaries.

Legacy unencrypted PEM identities and manual `swarm identity` commands remain readable for
advanced compatibility. New product bootstrap uses strict versioned JSON state automatically.

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

Secure pairing pins the coordinator public-key fingerprint after the coordinator proves its
Ed25519 key and pairing secret over the complete transcript. The low-level
`--trusted-coordinator-fingerprint` option remains available for manually provisioned workers.

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

Neither mechanism supplies inference payload confidentiality. Stage-ring activation, token, and
expert traffic is not protected by validated TLS in the current product. Pairing completion
encryption must not be generalized into a data-plane encryption claim. Network observers and
participating workers may inspect data available to them. Sensitive prompts and proprietary
weights require an independently secured network boundary.

## Owned firewall and confirmation boundary

Privileged firewall identifiers are deterministic hashes of validated cluster/node ownership;
raw labels never reach a command. Linux allocates one nftables table per owner, macOS one PF
anchor per owner, and Windows two exact owned rule names. Status, reconciliation, and removal
address only those resources. Broader unrelated rules are reported but not adopted or removed.
All owned rules remain limited to selected ports and RFC1918 source networks.

Administrative mutations prompt once only for an interactive human. `--yes` preauthorizes the
operation. JSON/NDJSON or non-interactive input without `--yes` fails with permission exit code 10
before service, firewall, trust, update, deletion, leave, or revocation mutation. Confirmation
text is command-owned and contains no secret material.

## Rotation and revocation

There is no live key-rotation protocol. Revoke or leave, drain/cancel active sessions, preserve
the historical identity record, and pair the replacement identity explicitly. A persisted worker
ID cannot silently switch keys.

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
