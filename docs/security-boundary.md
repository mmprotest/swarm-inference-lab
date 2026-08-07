# Product security boundary

Swarm is designed to carry inference across real WAN links. Canonical non-loopback control and
data transports use TLS 1.3, validate cluster credentials, and bind certificates to the durable
Ed25519 node identities established by pairing. A remote product endpoint without TLS material
fails closed. Plaintext loopback is retained only as an explicit development/test transport.

## Pairing and durable identity

The normal bootstrap path is \`swarm cluster create\` followed by \`swarm node join\`. Pairing uses
a random single-use secret, ephemeral X25519, HKDF-SHA256 transcript binding, reciprocal
Ed25519 possession proofs, and AES-GCM encrypted completion documents. Invitations expire,
have bounded attempts and source rate limits, and cannot be revived after consumption,
rejection, expiry, or coordinator restart.

Each node creates and retains its own durable Ed25519 identity. Only public keys, fingerprints,
TLS public keys, certificates, and signed/encrypted protocol material cross the network. Private
identity and TLS keys are never transmitted. Pairing atomically records trust and pins the
cluster ID, coordinator fingerprint, and cluster CA.

The invitation is an onboarding credential, not a long-lived bearer token. Secret-bearing URIs
are omitted from machine output, logs, status, and audits. Invitation files use owner-only
permissions where the platform supports them.

## Identity-bound TLS

The coordinator creates a cluster-scoped P-256 CA certificate cryptographically bound to its
durable Ed25519 fingerprint. A joining worker contributes a P-256 TLS public key inside its
signed pairing transcript. The coordinator issues a short-lived worker certificate containing:

- the cluster ID;
- the worker's durable Ed25519 fingerprint;
- a stable identity DNS name independent of its current IP address; and
- the exact TLS public key authorized by the signed transcript.

Workers validate the coordinator CA's Ed25519 binding before trusting it. Certificates are
checked for issuer, time bounds, role, cluster, public-key binding, expected peer fingerprint,
and revocation state where the transport has a revocation set. A changed hostname or IP address
does not change the pinned durable identity. Certificate renewal can reissue credentials without
creating a new cluster; a TLS-key change must be authorized by the same durable node identity.

TLS private-key files are written atomically with owner-only permissions. Certificate
replacement is separate from the short-lived pairing secret and does not transmit a private
key.

## Protected product paths

TLS 1.3 protects:

- coordinator-to-worker control and artifact-routing traffic;
- direct worker-to-worker stage-ring traffic;
- whole-expert/microshard data traffic and directed network probes;
- route/session control, activations, and token/result traffic;
- peer and engine-control RPCs; and
- llama.cpp RPC streams exposed through Swarm-managed metering proxies.

Stage and worker paths use mutually authenticated TLS. The coordinator listener also serves
initial onboarding, so its TLS handshake is server-authenticated during pairing; every
long-lived coordinator request additionally carries an Ed25519 signature from an active,
trusted cluster member. This gives mutual durable-identity authentication without allowing an
unpaired client certificate to bypass onboarding.

The coordinator remains a control-plane and token-commit participant. Hidden states flow
directly between persistent contiguous stages, not through the coordinator.

## Authorization and replay protection

The coordinator signs finite-lived route leases that bind topology generation, model and
tokenizer revisions, stage ownership, worker identities, endpoints, and nonce. Direct peers
exchange signed \`HELLO\` material tied to the installed lease. Per-edge sequence numbers,
request/session identity, route generation, time bounds, and bounded nonce caches reject
duplicate, stale, cross-session, and replayed participation.

Removing trust blocks fresh registration, new sessions, deployments, and replacement routes.
Operators should cancel active sessions before revocation when immediate removal is required.
Revocation is an authorization action, not a claim that an already compromised host can be
remotely cleaned.

## Checksums versus cryptographic transport

Stage-frame SHA-256 checksums detect accidental corruption and truncation. They are unkeyed and
do not authenticate a malicious sender. TLS supplies in-transit confidentiality, integrity, and
peer certificate validation; signed leases and handshakes supply topology authorization.

These mechanisms do not prove that a worker ran a neural operation correctly, that its host is
uncompromised, or that multiple identities represent independent operators.

## Firewall and confirmation boundary

Privileged firewall identifiers are deterministic hashes of validated cluster/node ownership,
and reconciliation only touches rules owned by that cluster. Administrative mutations prompt
once for an interactive human; \`--yes\` preauthorizes the operation. JSON/NDJSON or
non-interactive mutation without \`--yes\` fails before service, firewall, trust, update,
deletion, leave, or revocation changes.

TLS does not make an unreachable endpoint reachable. Operators must still configure routing,
NAT/overlay networking, firewall policy, and least-privilege exposure appropriate to their
environment.

## Audit and local storage

Coordinator state records trust rejections, registrations, deployments, recoveries, and request
lifecycle events. Trust changes have a separate append-only JSONL audit. Secrets and prompts
are omitted. These local files are not remote attestation or a tamper-evident consensus log;
protect and back up the state directory.

## Explicit non-goals

The current product does not claim:

- verification of computation returned by a malicious worker;
- Byzantine consensus or collusion resistance;
- privacy from a legitimate node that owns a model stage or expert;
- anonymous or permissionless compute;
- confidential computing or remote attestation; or
- anonymity for operators or traffic.

Swarm authenticates admitted nodes and encrypts traffic in transit. It assumes those admitted
nodes are trusted to see and correctly process the model state assigned to them.
