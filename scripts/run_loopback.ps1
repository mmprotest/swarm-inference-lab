param(
    [string]$Config = "configs/experiments/scaling_loopback.yaml",
    [ValidateSet("synthetic", "cpu", "cuda", "mps")]
    [string]$Backend = "synthetic"
)

$runner = Join-Path $PSScriptRoot "run_experiment.ps1"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runner `
    -Config $Config -Backend $Backend -AllowCpuDoctor
exit $LASTEXITCODE
