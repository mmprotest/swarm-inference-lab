# Cluster quick start

This is the normal OLMoE product workflow. It assumes a trusted LAN or private network. Pairing
traffic is authenticated and completion payloads are encrypted; inference data-plane payloads
are not encrypted.

## 1. Install the GitHub Release on each machine

On each machine, independently open the same [GitHub Release](https://github.com/mmprotest/swarm-inference-lab/releases),
download `SwarmInferenceSetup-x64.exe`, and double-click it. Do not transfer installation files
between the machines. Git, Python, `uv`, a wheel, PowerShell installation, and a repository clone
are not prerequisites.

Open a new terminal on each machine and run:

```powershell
swarm --version
swarm node doctor
```

The installer verifies the embedded application and locked dependency profile, runs the real
installed doctor, and does not create a service before cluster membership exists.

## 2. Create the cluster on the PC

On the coordinator machine:

```powershell
swarm cluster create --name villani-home
```

The command creates or adopts the durable identity, selects a private endpoint and available
ports, probes the backend, establishes memory/storage budgets, and installs/starts the user
service. Human output prints exactly one complete command:

```text
Cluster ready: villani-home

Run this command on the machine joining the cluster:

swarm node join "swarm://<private-address>:<port>/join/<single-use-data>"
```

## 3. Paste the join command

Paste the command printed by the PC into the laptop terminal. The single-use invitation is text;
no installer, wheel, repository, or invitation file is transferred. Do not save the command in
logs or screenshots.

## 4. Join from the laptop

On another machine:

```powershell
swarm node join "swarm://<private-address>:<port>/join/<single-use-data>"
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

For pairing automation, `--json` keeps the URI out of JSON and writes it to an owner-protected
file. An explicit `--pairing-output` remains available for automation; it is not needed in the
normal human workflow.
