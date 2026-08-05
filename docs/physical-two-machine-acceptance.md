# Physical Gate A: Windows RTX 5090 plus CPU-only laptop

This gate requires two distinct physical Windows machines on the same trusted private network:

- Machine A: RTX 5090 PC, coordinator and CUDA worker;
- Machine B: CPU-only laptop, joined worker.

Loopback aliases, VMs on one host, simulations, shaped traffic, or two processes on one machine
can never be used as physical evidence. If both machines are unavailable, record `NOT_RUN` with
the exact reason.

## 1. Build and transfer the product artifact

On the development machine:

```powershell
uv build --wheel
Get-FileHash .\dist\swarm_inference_lab-0.1.0-py3-none-any.whl -Algorithm SHA256
```

Copy only the wheel and `scripts/install.ps1` to each machine. Do not clone the repository. Save
the copied wheel hash from both machines.

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 `
  -SourceWheel .\swarm_inference_lab-0.1.0-py3-none-any.whl -Json `
  | Tee-Object .\install-evidence.json
```

The RTX machine must report `torch-cuda`; the laptop must report `torch-cpu`. A hardware-present
but failed tensor probe is not CUDA validation.

## 2. Create and pair using only product commands

Machine A:

```powershell
swarm cluster create --name physical-gate-a
```

Machine B, using the single-use URI shown once by Machine A:

```powershell
swarm node join "<pairing-uri>"
```

Do not run identity, trust, coordinator, worker, inspect, plan, deploy, or submit commands. Save
redacted command transcripts; never save the secret-bearing URI. Capture its public session ID,
expiry, and consumed audit event from status/audit evidence.

## 3. Prove automatic configuration and persistence

On Machine A:

```powershell
swarm cluster status --json | Tee-Object .\cluster-status-before-restart.json
```

The document must prove both distinct machine/node identities, OS/architecture/backend, selected
dtype and memory, non-loopback control/data/probe endpoints, service state, bidirectional
reachability, and fresh directed measurements in both directions.

Close both install/join terminals. Restart each owned user service or sign out/in. Then capture:

```powershell
swarm cluster status --json | Tee-Object .\cluster-status-after-restart.json
```

Both nodes must reconnect with unchanged identities and without pairing again.

## 4. Run speed and capacity objectives

Machine A:

```powershell
swarm run allenai/OLMoE-1B-7B-0125-Instruct `
  --revision b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e `
  --tokenizer-revision sha256:d1e645ebd850d79567e531a3c103ac575d8e9cf45fa941420afc584b293438ea `
  --mode speed --prompt "Explain distributed inference." --ndjson `
  | Tee-Object .\speed.ndjson

swarm run allenai/OLMoE-1B-7B-0125-Instruct `
  --revision b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e `
  --tokenizer-revision sha256:d1e645ebd850d79567e531a3c103ac575d8e9cf45fa941420afc584b293438ea `
  --mode capacity --require-node <laptop-node-id> `
  --prompt "Explain distributed inference." --ndjson `
  | Tee-Object .\capacity.ndjson
```

Speed mode must exclude the laptop when its measured utility lowers expected single-request
throughput. Capacity mode must include it in a separate feasible run. Both runs must match a
separately recorded exact deterministic token-ID expectation. Artifact events must show automatic
preparation/transfer/verification. Selected stage traffic must use direct non-loopback data
endpoints; coordinator latency is not link evidence.

## 5. Revocation check

After inference evidence is complete:

```powershell
swarm cluster revoke <laptop-node-id> --reason "physical gate revocation" --yes --json
```

Restart the laptop agent and prove registration is rejected. New deployment/recovery selection
must not trust the revoked node. Preserve the historical membership/revocation record.

## 6. Evidence schema version 2

Create `physical-evidence.json` with:

- `document_type: swarm-physical-two-machine-evidence` and `format_version: 2`;
- exact model/tokenizer revisions and distinct machine/node/worker identities;
- `installations` for both nodes with source-wheel SHA-256 and `repository_cloned: false`;
- consumed single-use `pairing` with `fingerprint_copied_manually: false`;
- `automatic_configuration` booleans for backend, memory, endpoints, and ports;
- persistent/reconnected `services` records;
- verified stage `artifacts` and content identities;
- authenticated, measured, non-loopback `directed_network_links` in both directions;
- topology, normal/recovery exact-token evidence;
- `speed_run.excluded_slow_node_id` and `capacity_run.included_slow_node_id`;
- only the high-level product commands above; and
- SHA-256 `source_files` for every retained log/output used by the summary.

Start from [physical-evidence.example.json](physical-evidence.example.json), replace every
placeholder with captured data, and validate:

```powershell
uv run python scripts/run_productization_acceptance.py run `
  --repeatability-evidence <repeatability-directory> `
  --physical-config <gate-a-configuration.json> `
  --output artifacts/acceptance
```

The gate is `PASS` only when the acceptance validator verifies the checksummed evidence from two
distinct physical machines. Missing evidence is `NOT_RUN`; malformed or contradictory executed
evidence is `FAIL`.
