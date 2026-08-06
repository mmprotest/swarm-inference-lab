# Cluster quick start

This is the normal OLMoE product workflow. It assumes a trusted LAN or private network. Pairing
traffic is authenticated and completion payloads are encrypted; inference data-plane payloads
are not encrypted.

## 1. Install the wheel/tool on each machine

Give each machine the built wheel and the matching installer. Git, Python, and `uv` are not
prerequisites.

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 `
  -SourceWheel .\swarm_inference_lab-0.1.0-py3-none-any.whl -Json
```

The clean installer runs `swarm node doctor` but does not create a service before membership
exists. Its successful service state is `deferred-until-cluster-create-or-join`; the legacy
installer `--install-service` preference is accepted but remains deferred.

## 2. Create the cluster on the PC

On the coordinator machine:

```powershell
swarm cluster create --name villani-home --yes --json `
  --pairing-output C:\Protected\villani-home-invitation.uri
```

The command creates or adopts the durable identity, selects a private endpoint and available
ports, probes the backend, establishes memory/storage budgets, and installs/starts the user
service. JSON is one document and contains only redacted invitation metadata and the protected
file path, never the URI.

## 3. Transfer the invitation securely

Transfer the owner-protected invitation file to the laptop through a secure channel, read the URI
only at the destination, and delete the transfer copy after successful consumption. Human mode
without `--json` may instead display the URI once.

## 4. Join from the laptop

On another machine:

```powershell
swarm node join "<pairing-uri>" --yes --json
```

Joining creates or reuses the node identity, verifies both transcript-bound identity proofs,
pins the coordinator, installs the worker service, validates bidirectional reachability,
benchmarks the selected device/dtype, and measures directed links. A node that cannot be reached
is `blocked`, not `ready`.

## 5. Run cluster status on the PC

```powershell
swarm cluster status --json
```

Implementation, software validation, and physical validation are independent fields. A tensor
probe selects an operational backend but does not turn a `not-run` validation record into
`validated`.

## 6. Run physical acceptance, then infer

Follow [physical two-machine acceptance](physical-two-machine-acceptance.md). A normal run is:

```powershell

swarm run allenai/OLMoE-1B-7B-0125-Instruct `
  --revision b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e `
  --tokenizer-revision sha256:d1e645ebd850d79567e531a3c103ac575d8e9cf45fa941420afc584b293438ea `
  --mode speed `
  --prompt "Explain distributed inference."
```

The revision is mandatory and immutable. `swarm run` refreshes capabilities and network
evidence, selects a benchmark-approved common dtype, performs bounded planning, builds and
transfers stage-owned artifacts, deploys transactionally, verifies routes/peers, and streams the
request through the canonical runtime.

Useful plan controls:

```powershell
swarm run <model> --revision <commit> --tokenizer-revision <identity> `
  --prompt "..." --mode balanced --dry-run --explain-plan

swarm run <model> --revision <commit> --tokenizer-revision <identity> `
  --prompt "..." --require-node <node-id> --exclude-node <node-id>
```

Speed mode can leave a paired slow node idle. Capacity mode can include it when collective fit
requires it. Balanced mode reports weighted throughput, memory-headroom, reliability, and
participation components.

## Automation

Use `--json` for a final document and `--ndjson` for progress/token streams. Machine output does
not contain pairing secrets or prompts. Administrative mutations in JSON/NDJSON or with
non-interactive stdin require `--yes`; they never prompt. Exit categories are documented in
[cluster troubleshooting](cluster-troubleshooting.md).
