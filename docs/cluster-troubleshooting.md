# Cluster troubleshooting

Start with:

```text
swarm node doctor --json
swarm node status --json
swarm cluster status --json
```

Known failures retain their stage, node, category, corrective action, and retry safety. New CLI
exit categories are permission (10), connectivity (11), compatibility (12), capacity (13),
artifact integrity (14), and execution (15).

## Permission error requires `--yes`

Administrative mutations prompt once only in an interactive human terminal. JSON/NDJSON and
non-interactive invocations never prompt; add `--yes` after reviewing the operation. A refusal or
missing `--yes` exits with category `permission` before service, firewall, trust, configuration,
update, deletion, leave, or revocation state changes. Prompts contain fixed action text and never
contain a pairing URI.

## Installer creates no cluster service

This is the successful clean-install state, not an error. Create the coordinator with
`swarm cluster create` or join with `swarm node join`; either installs and starts the correct
cluster-specific service unless `--foreground` is used. Running `swarm node install-service`
while unpaired correctly fails with `create or join a cluster first`.

The setup log path is displayed on failure. Native bootstrapper logs are under the installation
`logs` directory. A failed upgrade returns non-zero and restores the previous runtime before
reporting failure.

## Node is blocked

Read `runtime.reason` and `runtime.error_category`. A firewall block includes one exact remediation
command; review and run it in an administrator shell, then restart the owned user service. The
agent does not request elevation itself. Confirm the selected advertised addresses are private,
routable, and not wildcard/loopback.

For VPN or multi-NIC hosts, persist an interface or explicit endpoints:

```text
swarm node configure --interface <name>
swarm node configure --control-endpoint <private-ip:port>
swarm node configure --data-endpoint <private-ip:port>
```

## Pairing rejected

Create a new invitation if the URI expired, was consumed, exceeded attempts, or the coordinator
restarted. Do not retry a known-disclosed secret. Version errors identify supported protocol
minor/artifact ranges. Major mismatches require an explicit wheel update.

Human pairing output prints one quoted `swarm node join "swarm://..."` command. JSON pairing
output is exactly one non-secret document. Automation reads `pairing.invitation_file`, not stdout,
to obtain the complete URI. The file is owner-protected and atomically published. An existing
destination is refused unless `--force-pairing-output` is explicit. If a default invitation
expires, cleanup removes that expired file without touching active session files.

## Validation remains `not-run`

This is expected on a clean machine. `implementation_status=implemented` says the adapter path
exists; it does not imply software or physical validation. A visible GPU and a passing tensor
probe select CUDA but do not validate the RTX 5090 gate. Only a retained, exact
platform/backend-scoped acceptance record can report `validated`.

## Firewall ownership and stale rules

Linux status/removal targets only `swarm_<owner-hash>`; macOS targets only
`swarm-inference/<owner-hash>`; Windows targets the two hashed owned display names. Re-run the
reported remediation when ports change to reconcile stale owned rules. Broader unrelated rules
are reported for operator review but are never adopted or deleted by Swarm Inference.

## No distributed speed plan

This can be correct. Speed mode excludes a node whose measured compute, queue load, reliability,
memory, or directed link reduces predicted single-request throughput. Use
`--dry-run --explain-plan` and inspect every node's utility report. Do not force a slow node unless
the workload requires capacity or the diagnostic run intentionally uses `--require-node`.

Stale/unmeasured links are never labeled measured throughput. Agents refresh links on a bounded
interval; a run waits briefly for refresh and then lets the planner exclude unsupported paths.

## Artifact failure

An integrity failure is not safely bypassed. Check storage budget, source revision, tokenizer
identity, chunk hash, and final content hash. Partial directories are isolated under downloads
and never loaded. Retrying is safe after the exact source/space problem is corrected; do not edit
published artifact contents.

## Service fails after update

For a native Windows installation, use `swarm update`. It selects a GitHub Release, validates the
fixed repository identity and manifest, downloads with bounded HTTPS requests, verifies SHA-256,
requires Authenticode for stable releases, and launches setup in upgrade mode. It never replaces
running application files itself. A setup failure restores the previous runtime and service.

`swarm node update --source-wheel ...` remains an explicit developer/offline recovery path for
non-native installations. Inspect the platform service log from `swarm node status`; identity and
membership remain outside application runtime slots.

## Public networking

Swarm accepts mutually authenticated TLS traffic over any reachable routed address, including a
public address or private overlay. It does not yet provide NAT traversal, relay discovery, or
automatic port forwarding. The owned firewall automation deliberately stays scoped to RFC1918
sources; a directly public endpoint therefore needs an explicit least-privilege operator rule.
Pairing provisions identity and credentials but does not make an unreachable address reachable.
