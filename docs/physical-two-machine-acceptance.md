# Physical Gate A: Windows RTX 5090 plus CPU-only laptop

This gate requires two distinct physical Windows machines on the same trusted private network:

- Machine A: RTX 5090 PC, coordinator and CUDA worker;
- Machine B: CPU-only laptop, joined worker.

Loopback aliases, VMs on one host, simulations, shaped traffic, or two processes on one machine
can never be used as physical evidence. If both machines are unavailable, record `NOT_RUN` with
the exact reason.

## 1. Independently install the same GitHub prerelease

On both machines, open the same GitHub prerelease page and independently download
`SwarmInferenceSetup-x64.exe`. Do not transfer the setup executable between machines. Do not
clone the repository or install Git, Python, `uv`, a wheel, or a PowerShell installer.

Double-click setup on each machine, complete the wizard, open a new terminal, and run:

```powershell
swarm --version
swarm node doctor --json | Tee-Object .\doctor.json
```

Record the GitHub release tag/URL, installer filename, installer SHA-256 on each machine,
Authenticode status, exact release-manifest SHA-256, selected profile, product version, and the
complete non-secret installation record. The setup hashes on both machines must be identical.

The RTX machine must report the CUDA profile and operational `torch-cuda`; the laptop must report
the CPU profile and `torch-cpu`. A hardware-present but failed tensor probe is not CUDA validation.
Installation alone creates no cluster service and no validation record. An unsigned RC must be
recorded as `unsigned-prerelease`; it must never be described as signed.

## 2. Create and pair using only product commands

Machine A:

```powershell
swarm cluster create --name villani-home
```

The human output must contain one complete quoted `swarm node join "swarm://..."` command. Do not
write the secret URI into retained evidence.

Machine B, paste the command printed by Machine A:

```powershell
swarm node join "swarm://<private-address>:<port>/join/<single-use-data>"
```

Do not run identity, trust, coordinator, worker, inspect, plan, deploy, or submit commands. Save
redacted command transcripts; never copy the secret-bearing URI into a transcript, JSON, log, or
acceptance summary. No invitation file is transferred. Capture its public session ID, expiry, and
consumed audit event from status/audit evidence.

## 3. Prove automatic configuration and persistence

On Machine A:

```powershell
swarm cluster status --json | Tee-Object .\cluster-status-before-restart.json
```

The document must prove both distinct machine/node identities, OS/architecture/backend, selected
dtype and memory, non-loopback control/data/probe endpoints, service state, bidirectional
reachability, and fresh directed measurements in both directions. It must report implementation,
software validation, and physical validation separately. Tensor probing cannot pre-populate the
physical validation status.

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
  --mode speed --prompt "Write a Python function that merges two sorted lists." --ndjson `
  | Tee-Object .\speed.ndjson

swarm run allenai/OLMoE-1B-7B-0125-Instruct `
  --revision b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e `
  --tokenizer-revision sha256:d1e645ebd850d79567e531a3c103ac575d8e9cf45fa941420afc584b293438ea `
  --mode capacity --require-node <laptop-node-id> `
  --prompt "Write a Python function that merges two sorted lists." --ndjson `
  | Tee-Object .\capacity.ndjson
```

Speed mode may exclude the laptop when its measured utility lowers expected single-request
throughput. Capacity mode must assign work to both machines when the model/partition requires and
permits it. Neither result counts as a two-machine pass unless the deployed topology names both
distinct machine identities. Both runs must match a separately recorded exact deterministic
token-ID expectation. Artifact events must show automatic preparation/transfer/verification.
Selected stage traffic must use direct non-loopback data endpoints and retain direct-stage traffic
evidence; coordinator latency is not link evidence.

## 5. Revocation check

After inference evidence is complete:

```powershell
swarm cluster revoke <laptop-node-id> --reason "physical gate revocation" --yes --json
```

Restart the laptop agent and prove registration is rejected. New deployment/recovery selection
must not trust the revoked node. Preserve the historical membership/revocation record.

## 6. Evidence schema version 4

Create `physical-evidence.json` with:

- `document_type: swarm-physical-two-machine-evidence` and `format_version: 4`;
- one `release` record with GitHub tag/URL, `SwarmInferenceSetup-x64.exe`, release-manifest
  SHA-256, and explicit Authenticode status;
- exact model/tokenizer revisions and distinct machine/node/worker identities;
- `installations` for both nodes with the identical setup SHA-256, complete installation record,
  selected CPU/CUDA profile, product version, manifest hash, `source:
  github-release-installer`, and `repository_cloned: false`;
- consumed single-use `pairing` with `fingerprint_copied_manually: false`;
- `automatic_configuration` booleans for backend, memory, endpoints, and ports;
- persistent/reconnected `services` records;
- verified stage `artifacts` and content identities;
- authenticated, measured, non-loopback `directed_network_links` in both directions;
- per-run deployed topology, normal/recovery exact-token evidence, and retained direct-stage
  traffic observations;
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
distinct physical machines, including one operational CUDA installation and one CPU installation
from the exact same GitHub setup. Missing evidence is `NOT_RUN`; malformed or contradictory
executed evidence is `FAIL`. During release-candidate software CI the physical gate is honestly
`NOT_RUN`; only the later two-machine execution may create physical validation records.

Software repeatability runs the complete non-GPU cluster product suite three times and the product
stage-ring module five times. The retained command explicitly excludes the two opt-in Experiment
007 artifact-audit files because starting or supplying that research experiment is outside this
product gate. Any skipped test that remains in the product selection makes the evidence incomplete;
the producer and consumer share the same schema and test-command version constants.
