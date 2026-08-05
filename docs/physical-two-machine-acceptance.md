# Physical two-machine product acceptance

This runbook produces physical evidence only when the normal product commands use workers on two
distinct machines. A loopback process, two processes on one host, containers sharing one machine
identity, or shaped network simulation cannot satisfy this gate.

The examples use:

- machine A / coordinator: `192.0.2.10`;
- machine A / worker A control and data: `192.0.2.10:50052` and `:50053`;
- machine B / worker B control and data: `192.0.2.20:50052` and `:50053`; and
- optional replacement worker B2: `192.0.2.20:50062` and `:50063`.

Replace documentation addresses with routable private addresses. Public Internet exposure is
unsupported because payload confidentiality is absent.

## Evidence directory and facts to record

Create `artifacts/acceptance/physical-<UTC timestamp>/` and retain, for every machine:

- manufacturer/model and stable machine-identity JSON;
- CPU model, physical/logical core count, installed and available memory;
- every GPU model, VRAM, driver, CUDA runtime, and PyTorch version;
- operating system edition/build, architecture, Python, `uv`, and package-lock identity;
- git commit and complete dirty-tree status;
- NIC model, link speed/duplex, MTU, switch path, and wired/wireless medium;
- hostname hash, private address, and process-namespace identity;
- clock source, synchronization status, measured offset, and timezone;
- model ID, exact model revision, tokenizer revision, snapshot hashes, and local cache path;
- worker public fingerprints, roles, endpoints, PIDs, and advertised capabilities; and
- every command, stdout/stderr log, topology, token event, timing, recovery event, and checksum.

Generate non-secret machine records on each host:

```powershell
uv run python scripts/run_productization_acceptance.py machine-identity `
  --output .swarm/evidence/machine-identity.json
git rev-parse HEAD | Tee-Object .swarm/evidence/git-commit.txt
git status --porcelain=v1 | Tee-Object .swarm/evidence/git-status.txt
python --version 2>&1 | Tee-Object .swarm/evidence/python-version.txt
uv --version 2>&1 | Tee-Object .swarm/evidence/uv-version.txt
nvidia-smi -q | Out-File .swarm/evidence/nvidia-smi.txt
Get-NetAdapter | Format-List * | Out-File .swarm/evidence/network-adapters.txt
Get-NetIPConfiguration | Format-List * | Out-File .swarm/evidence/ip-configuration.txt
```

Record “not installed” when NVIDIA tooling is inapplicable. Do not delete or edit a dirty-tree
record to make provenance appear clean.

## Clock synchronization

Use the same trusted NTP source. On Windows, run and save:

```powershell
w32tm /query /status | Tee-Object .swarm/evidence/clock-status.txt
w32tm /stripchart /computer:<ntp-server> /samples:10 /dataonly `
  | Tee-Object .swarm/evidence/clock-offset.txt
```

On Linux, use `chronyc tracking` and `chronyc sources -v`. Do not compare wall-clock intervals
across hosts without recording the observed offset. Product traces use local monotonic clocks;
cross-host evidence must preserve the synchronization capture.

## Firewall and reachability

Permit only the trusted private subnet:

- machine A inbound TCP 50051 (coordinator), 50052 (worker control), 50053 (worker data);
- machine B inbound TCP 50052 (worker control), 50053 (worker data); and
- for recovery, machine B inbound TCP 50062 and 50063 for replacement worker B2.

Test each destination from the other host with `Test-NetConnection <host> -Port <port>`. Save
results. Do not use an SSH tunnel or port-forward resolving every endpoint to loopback; it defeats
the physical-address gate.

## Model provisioning

Both hosts need the exact pinned snapshot in a host-local cache:

```powershell
$ModelId = "allenai/OLMoE-1B-7B-0125-Instruct"
$ModelRevision = "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"
$TokenizerRevision = "sha256:d1e645ebd850d79567e531a3c103ac575d8e9cf45fa941420afc584b293438ea"
$ModelSnapshot = "D:\models\OLMoE-1B-7B-0125-Instruct-b89a7c4"
```

Verify `config.json`, tokenizer files, safetensors index/files, revision provenance, and hashes on
both machines. Do not enable download during the measured run. A synthetic fixture is not a
real-model gate.

## Identity provisioning

On machine A:

```powershell
uv run swarm identity create --path .swarm/coordinator/coordinator-identity.json --kind coordinator
uv run swarm identity create --path .swarm/identities/worker-a.json --kind worker
$CoordinatorFingerprint = (uv run swarm identity fingerprint `
  --path .swarm/coordinator/coordinator-identity.json).Trim()
$WorkerAFingerprint = (uv run swarm identity fingerprint `
  --path .swarm/identities/worker-a.json).Trim()
```

On machine B:

```powershell
uv run swarm identity create --path .swarm/identities/worker-b.json --kind worker
uv run swarm identity create --path .swarm/identities/worker-b2.json --kind worker
$WorkerBFingerprint = (uv run swarm identity fingerprint `
  --path .swarm/identities/worker-b.json).Trim()
$WorkerB2Fingerprint = (uv run swarm identity fingerprint `
  --path .swarm/identities/worker-b2.json).Trim()
```

Exchange only public fingerprints over an authenticated administrative channel. Never copy a
worker identity document off its owner merely to establish trust: it contains the private key.
On A, trust the received values:

```powershell
uv run swarm identity trust --coordinator-state .swarm/coordinator `
  --identity .swarm/identities/worker-a.json --label machine-a-worker
uv run swarm identity trust --coordinator-state .swarm/coordinator `
  --fingerprint $WorkerBFingerprint --label machine-b-worker
uv run swarm identity trust --coordinator-state .swarm/coordinator `
  --fingerprint $WorkerB2Fingerprint --label machine-b-replacement
```

Provision `$CoordinatorFingerprint` to B over the same authenticated channel.

## Start the product

Machine A coordinator:

```powershell
uv run swarm coordinator --config configs/product/olmoe-stage-ring.yaml `
  --state .swarm/coordinator --listen 0.0.0.0:50051 `
  --advertise 192.0.2.10:50051 2>&1 `
  | Tee-Object artifacts/acceptance/physical-current/coordinator.log
```

Machine A worker:

```powershell
uv run swarm worker --coordinator 192.0.2.10:50051 --worker-id worker-a `
  --identity .swarm/identities/worker-a.json --backend torch-cuda `
  --memory-limit-gb 24 --listen 0.0.0.0:50052 --advertise 192.0.2.10:50052 `
  --stage-runtime --data-listen 0.0.0.0:50053 --data-advertise 192.0.2.10:50053 `
  --device cuda --dtype bfloat16 --model-snapshot $ModelSnapshot `
  --trusted-coordinator-fingerprint $CoordinatorFingerprint 2>&1 `
  | Tee-Object artifacts/acceptance/physical-current/worker-a.log
```

Machine B primary worker (and, for recovery, repeat with worker B2, ports 50062/50063, and its
identity):

```powershell
uv run swarm worker --coordinator 192.0.2.10:50051 --worker-id worker-b `
  --identity .swarm/identities/worker-b.json --backend torch-cuda `
  --memory-limit-gb 24 --listen 0.0.0.0:50052 --advertise 192.0.2.20:50052 `
  --stage-runtime --data-listen 0.0.0.0:50053 --data-advertise 192.0.2.20:50053 `
  --device cuda --dtype bfloat16 --model-snapshot $ModelSnapshot `
  --trusted-coordinator-fingerprint $CoordinatorFingerprint 2>&1 `
  | Tee-Object artifacts/acceptance/physical-current/worker-b.log
```

Expected `swarm workers --json` output has distinct worker IDs, public fingerprints, process IDs,
non-loopback advertised endpoints, stage-ring protocol support, and healthy registrations.

## Plan, deploy, and run the reference request

On A:

```powershell
uv run swarm workers --coordinator 192.0.2.10:50051 --json `
  | Tee-Object artifacts/acceptance/physical-current/workers.json
uv run swarm model inspect --coordinator 192.0.2.10:50051 `
  --model-id $ModelId --revision $ModelRevision `
  --tokenizer-revision $TokenizerRevision --dtype bfloat16 `
  | Tee-Object artifacts/acceptance/physical-current/model-inspect.json
uv run swarm model plan --coordinator 192.0.2.10:50051 `
  --model-id $ModelId --revision $ModelRevision `
  --tokenizer-revision $TokenizerRevision --dtype bfloat16 `
  --stage-count 2 --partition balanced --require-distributed `
  --output artifacts/acceptance/physical-current/stage-plan.json
uv run swarm model deploy --coordinator 192.0.2.10:50051 `
  --plan artifacts/acceptance/physical-current/stage-plan.json `
  | Tee-Object artifacts/acceptance/physical-current/deployment.json
uv run swarm topology --coordinator 192.0.2.10:50051 --json `
  | Tee-Object artifacts/acceptance/physical-current/topology.json
uv run swarm submit --coordinator 192.0.2.10:50051 `
  --model-id $ModelId --model-revision $ModelRevision `
  --prompt "Write a Python function that merges two sorted lists." `
  --max-new-tokens 32 --temperature 0 --stream --ndjson `
  | Tee-Object artifacts/acceptance/physical-current/tokens.ndjson
```

Compare exact token IDs with a separately captured, pinned full-model reference. Record a mismatch
as `FAIL`; do not compare only decoded text.

## Physical failure injection

1. Start trusted replacement worker B2 before deployment but leave it unassigned.
2. Submit the same prompt with enough tokens to observe at least one accepted token.
3. From `topology.json`, identify the PID and host owning stage 1. Confirm it is machine B.
4. On machine B, run `Stop-Process -Id <stage-1-worker-pid>` after the accepted token. Save the
   command and timestamp. This is a real process failure; do not close only a listener and do not
   run the replacement on loopback.
5. Save the streaming events, coordinator/worker logs, sessions, workers, and new topology.
6. Require exactly one recovery, a higher route generation, replacement selection, verified
   replay of the accepted prefix, no replay-token client event, no duplicate position, and exact
   final token IDs.

If the request finishes before the process termination, the injection did not occur and the gate
is `FAIL`, not a retry relabelled as evidence. If no compatible replacement exists, record `FAIL`.

## Physical configuration and bundle

Copy the two public machine-identity JSON files into the evidence directory and prepare a config:

```json
{
  "coordinator_host": "192.0.2.10",
  "coordinator_endpoint": "192.0.2.10:50051",
  "worker_a_host": "192.0.2.10",
  "worker_b_host": "192.0.2.20",
  "worker_a_identity_path": ".swarm/identities/worker-a.json",
  "worker_b_identity_path": ".swarm/identities/worker-b.json",
  "worker_a_machine_identity": "artifacts/acceptance/physical-current/machine-a.json",
  "worker_b_machine_identity": "artifacts/acceptance/physical-current/machine-b.json",
  "model_revision": "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e",
  "evidence_output_directory": "artifacts/acceptance/physical-current"
}
```

Validate it with:

```powershell
uv run python scripts/run_productization_acceptance.py run --real-model `
  --physical-config artifacts/acceptance/physical-current/config.json `
  --output artifacts/acceptance
```

The validator rejects loopback, shared resolved addresses, matching host/machine identities, and
matching process namespaces. A validated config alone is not a physical pass.

Copy [the physical evidence template](physical-evidence.example.json) to
`physical-evidence.json` inside `evidence_output_directory`, then replace every placeholder with
captured product output. In particular:

- `machine_identities` must reproduce the two independently captured machine records;
- `worker_identities` and each worker status record must contain matching, distinct public
  fingerprints but never private key bytes;
- `workers` must use non-loopback control and data endpoints and distinct machine identities;
- `topology.assignments` must place stages on both reported worker machines;
- `normal_run` must contain the exact independently pinned reference token IDs;
- `recovery_run` must contain one `RECOVERY_COMPLETED` event, strictly increasing distinct route
  generations, consecutive token positions, no duplicate positions, and the exact tokens;
- `commands` must preserve the coordinator, two worker, deploy, and two submit commands; and
- `source_files` must map every retained log or command-output path, relative to the evidence
  directory, to `sha256:<lowercase hex>`.

Generate a checksum on PowerShell with `Get-FileHash -Algorithm SHA256 <path>` and lowercase the
reported hex when adding it to the JSON. The validator recomputes every declared checksum and
fails path traversal, missing files, or mismatches. Missing `physical-evidence.json` remains
`NOT_RUN`; a present but invalid attempted bundle is `FAIL`. Only a fully validated summary can
produce the physical gate's `PASS`, and overall `PHYSICAL_ACCEPTANCE_PASS` additionally requires
all software and real-model gates to pass in the same acceptance invocation.

## Cleanup

Unload the topology, stop workers gracefully where still running, then stop the coordinator:

```powershell
uv run swarm model unload --coordinator 192.0.2.10:50051 --topology-id <topology-id>
```

Press Ctrl+C in worker B2, worker B (if restarted), worker A, then coordinator terminals. Confirm
no listed PID or listening test port remains. Restore firewall rules if they were temporary.

## Troubleshooting

- Registration rejection: compare `identity fingerprint` with `identity list-trusted --json`;
  never disable secure mode to bypass it.
- Peer handshake failure: verify the coordinator pin, worker public keys, advertised endpoint, and
  route generation.
- Wildcard/loopback endpoint in topology: set explicit routable `--advertise` and
  `--data-advertise` values.
- Model inspection mismatch: compare exact revision, tokenizer hash, dtype, config, and local
  snapshot hashes on both hosts.
- Firewall timeout: test each control/data port in both required directions and record the result.
- Clock anomaly: stop, resynchronize, and rerun; do not hand-edit timestamps.
- Recovery without replacement: start and trust B2 before the request, then redeploy.
- Token mismatch or replay divergence: preserve all artifacts and mark the gate `FAIL`.

Loopback results may accompany a physical bundle as a software control, but they can never be
used as physical evidence.
