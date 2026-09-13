# E025 RunPod paid-run handoff

Preparation status: **`BLOCKED_BEFORE_PAID_RUNPOD_CANARIES`**

## Frozen facts

- Model: `moonshotai/Kimi-K3` at `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.
- Checkpoint fingerprint: `25162130a11904bac1220a7d654a3f7dfd616ea5f4035d488e40ac74ddea8f94`.
- Worker image: `ghcr.io/mmprotest/swarm-inference-lab@sha256:46cad5a031b98aeecdee9ba471e2e8338eb8e9e485d1d2e90a1505f8890ee8e4` (unchanged).
- Physical placement SHA-256: `015c7bc6ce4f2724123df2315607e7c261ad4aba2dd1b9def5266bfa2ac334f5`.
- Frozen topology: 93 backbone GPU roles plus four Layer 89 fragment roles; 497,052 tensors and 1,559,965,606,912 source bytes, with zero duplicate/unassigned ownership.
- Complete Layer 89 runtime peak: 20,556,349,440 bytes (19.14459228515625 GiB). Its four real expert fragments already fit and ran on independent 8-12 GB consumer Ampere machines.
- Existing E025 Stage 1/Stage 2 physical evidence remains authoritative; this preparation did not rerun it.

## RunPod plan

- Backbone: `NVIDIA GeForce RTX 3090`, Secure Cloud, one GPU per backbone Pod (current negative-control-only stock).
- Provider objects: 97 Pods for 97 GPUs (93 backbone/parent plus four single-GPU fragments), a reduction of 0 objects from the Vast layout.
- Physical-host claim: four fragment `machineId` values must be pairwise distinct. Total distinct `machineId` values are measured after allocation; Pod IDs never prove host identity.
- Layer 89 parent: isolated on one Secure RTX 3090 Pod so endpoint-generation changes recycle only that Pod.
- Preferred backbone datacenter: `EU-CZ-1`; alternates: `[]`.
- Networking: `RUNPOD_PUBLIC_TCP` and `RUNPOD_GLOBAL_PRIVATE` are implemented as endpoint modes; selection remains `REQUIRES_PAID_TWO_POD_CANARY`.
- Distribution: direct selective checkpoint download with unchanged worker-scoped caches; no network volume.
- Container disk: exact per-Pod values are in `runpod-pod-storage-plan.json`; total requested is 5887 GB.

## Live inventory snapshot

- Captured: `2026-08-28T04:37:18.437806+00:00`.
- Secure RTX 3090 stock: Low; count-specific GraphQL scheduling visible for 1-GPU Pods, at $0.50/GPU-hour.
- Validated small Ampere fragment inventory: GraphQL Low-price records exist, but the authenticated CLI snapshot has no located available datacenter.
- Account credit: $0.0; hourly spend limit: $80.0.

## Current blockers

- `ACCOUNT_CREDIT_ZERO` -- RunPod requires funded credit before any on-demand Pod can be created.
- `NO_CLI_LOCATED_VALIDATED_AMPERE_FRAGMENT_STOCK` -- GraphQL count-specific lowestPrice says Low for validated small Ampere SKUs, but authenticated CLI inventory exposes no available datacenter.
- `NO_MATERIAL_MULTI_GPU_BACKBONE_CONFIGURATION` -- Current inventory does not reduce provider objects materially.

## Paid gates remaining

1. P1 single Pod.
2. P2 network.
3. P3 multi-GPU.
4. P4 sub-layer.
5. P5 full acquisition rehearsal.
6. Headline run.

## Cost

- P1-P5 projected base: $14.93-$15.18; safety budget: $22.40-$22.77.
- Headline projected hourly: $47.84-$48.04.
- Headline 2.25-hour hard window with safety: $134.54-$135.11.
- Recommended headline account credit: $136. Current balance is $0.0; the hourly limit is $80.0.

## Exact future commands

From the repository root in PowerShell:

```powershell
# Refresh inventory and regenerate plans; guaranteed read-only.
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod_inventory.py
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode prepare

# Review all P1-P5 requests without creating anything.
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode paid-canaries

# After funding, stock refresh, and explicit authorization, run one gate at a time.
$runpodSecret = Read-Host 'RunPod API key for this PowerShell process' -AsSecureString
$env:RUNPOD_API_KEY = [System.Net.NetworkCredential]::new('', $runpodSecret).Password
$env:E025_RUNPOD_ALLOW_RENTAL = 'YES'
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode p1 --allow-paid-run
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode p2 --allow-paid-run
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode p3 --allow-paid-run
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode p4 --allow-paid-run
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode p5 --allow-paid-run

# Only after P1-P5 receipts pass:
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode headline --allow-paid-run

# Independent emergency permanent deletion for this run only:
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod_cleanup.py --run-id 20260819T013016Z --ledger artifacts/runs/experiment-025-20260819T013016Z/rental/runpod/pod-ledger.jsonl --allow-paid-run

# Read-only final verification:
& ./.venv/Scripts/python.exe scripts/experiment_025_runpod.py --mode verify-zero

# Remove ephemeral authorization and API-key material from this shell.
Remove-Item Env:E025_RUNPOD_ALLOW_RENTAL -ErrorAction SilentlyContinue
Remove-Item Env:RUNPOD_API_KEY -ErrorAction SilentlyContinue
```

Do not place the API key in a command, artifact, source file, or Git commit. Use the existing `runpodctl` configuration; the Python paid path accepts `RUNPOD_API_KEY` only from the process environment.
