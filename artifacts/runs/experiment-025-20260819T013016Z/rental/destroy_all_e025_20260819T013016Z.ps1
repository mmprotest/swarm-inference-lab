param(
    [string]$Reason = 'manual emergency cleanup'
)
$ErrorActionPreference = 'Stop'
$E025Repo = 'C:\Users\Simon\OneDrive\Documents\Python Scripts\swarm-inference-lab'
$env:PYTHONPATH = Join-Path $E025Repo 'src'
& (Join-Path $E025Repo '.venv\Scripts\python.exe') `
    (Join-Path $E025Repo 'scripts\experiment_025_cleanup.py') `
    --run-id '20260819T013016Z' `
    --ledger 'C:\Users\Simon\OneDrive\Documents\Python Scripts\swarm-inference-lab\artifacts\runs\experiment-025-20260819T013016Z\rental\instance-ledger.jsonl' `
    --output 'C:\Users\Simon\OneDrive\Documents\Python Scripts\swarm-inference-lab\artifacts\runs\experiment-025-20260819T013016Z\rental\emergency-cleanup.json' `
    --reason $Reason
exit $LASTEXITCODE
