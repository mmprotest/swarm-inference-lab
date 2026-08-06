# Secure single-use pairing

Pairing replaces manual fingerprint copying and trust-store editing. It secures onboarding on a
trusted LAN/private network; it does not encrypt subsequent inference payloads.

## Invitation

`swarm cluster pair` creates a cryptographically random secret with at least 128 bits of entropy,
a session ID, and an ephemeral X25519 keypair. The default expiry is ten minutes. The URI contains
the private coordinator endpoint and opaque single-use data. Human `cluster create` and `cluster
pair` output prints it exactly once as a complete quoted command:

```text
swarm node join "swarm://<private-address>:<port>/join/<single-use-data>"
```

The URI is never repeated in status, logs, errors, audits, or service state. The ordinary human
workflow pastes this text on an independently installed node and transfers no invitation file.

`cluster create --json` and `cluster pair --json` each emit exactly one JSON document and never
contain the URI. With `--pairing-output <path>`, the complete URI is written using a complete
temporary file followed by atomic publication. JSON without an explicit path uses a secret
location under the cluster state directory and returns only that path. Existing files are
rejected unless `--force-pairing-output` is explicit, and `--pairing-output -` is forbidden in
machine-readable mode. Parent directories are created safely.

Invitation files use POSIX mode `0600`. Windows grants the actual process-user SID exclusive
access when ACL tooling is available; otherwise the result explicitly reports the strongest
user-scoped fallback and its limitation. Default expired, consumed, or invalidated invitation
files are retired during session cleanup while active sessions remain untouched. The URI is not
stored in ordinary cluster metadata.

Unused in-memory ephemeral sessions are invalidated by coordinator restart. Consumed, expired,
or rejected sessions cannot be revived.

## Handshake

1. The joining node creates or reuses its durable Ed25519 identity and creates ephemeral X25519
   material.
2. Hello and challenge messages bind node/coordinator keys, fingerprints, nonces, versions,
   endpoints, session identity, and ephemeral keys into a canonical transcript.
3. Both sides derive a session key using X25519 and HKDF-SHA256 with the pairing secret and
   complete transcript as authenticated context.
4. The node signs and proves possession of its Ed25519 key and pairing secret over the complete
   transcript.
5. The coordinator returns the reciprocal proof and signature.
6. Sensitive completion documents use AES-GCM with unique nonces and phase-specific associated
   data.

Secret-derived proof comparison is constant-time. Completion is bounded by expiry, maximum
attempts, per-session and source-address rate limits, nonce replay caches, and explicit RPC
timeouts.

## Commit and revocation

Only after all checks pass does the coordinator atomically persist membership, add the node
fingerprint to the existing `WorkerTrustStore`, consume the session, and write a non-secret audit
event. The node atomically pins cluster metadata, coordinator fingerprint, and membership. No
private identity crosses the network.

```text
swarm cluster revoke <node-id> --reason "retired node"
swarm node leave
```

Revocation preserves history but removes active membership/trust. Fresh registration,
deployment selection, and recovery replacement are blocked. Manually trusted workers remain a
supported advanced compatibility path.

## Secret handling

Never paste a pairing URI into public issue trackers or logs. If disclosure is suspected, let it
expire or create a new session; the old session is not reusable after a successful join. Audit
records include session IDs and decisions, not secrets, raw proofs, private keys, AES/session
keys, or prompt contents.

Protected invitation files are an automation feature, not an interactive prerequisite:

```powershell
swarm cluster pair --json --pairing-output C:\Protected\one-time.uri
```

The caller reads that protected file locally, transfers it through an authenticated secure
channel, joins, and deletes the consumed transfer copy. Do not redirect JSON stdout expecting a
secret; JSON is intentionally non-secret.
