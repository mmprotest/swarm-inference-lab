param(
    [switch]$AllowPaidRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $AllowPaidRun) {
    throw "Refusing cleanup mutation without -AllowPaidRun."
}
if ($env:E025_RUNPOD_ALLOW_RENTAL -ne "YES") {
    throw "Refusing cleanup mutation without E025_RUNPOD_ALLOW_RENTAL=YES."
}
if (-not $env:RUNPOD_API_KEY) {
    throw "RUNPOD_API_KEY must be present only in the process environment."
}

$RunId = "20260819T013016Z"
$Ledger = "artifacts/runs/experiment-025-$RunId/rental/runpod/pod-ledger.jsonl"
$Receipt = "artifacts/runs/experiment-025-$RunId/rental/runpod/emergency-cleanup-receipt.json"

& ./.venv/Scripts/python.exe scripts/experiment_025_runpod_cleanup.py `
    --run-id $RunId `
    --ledger $Ledger `
    --receipt $Receipt `
    --allow-paid-run
exit $LASTEXITCODE
