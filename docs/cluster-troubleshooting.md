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

`swarm node update` stages a separate runtime and commits only after a fresh ready state. A
startup failure restores the previous service executable. Inspect the platform service log from
`swarm node status`; identity and membership remain outside runtime slots.

## Public networking

NAT traversal and public-Internet participation are unsupported. Use a trusted routed LAN or
private VPN. Pairing is not a substitute for inference-payload encryption.
