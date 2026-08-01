[CmdletBinding()]
param(
    [string]$Config = "configs\experiments\experiment_007_heterogeneous_node_utility.yaml",
    [string]$Experiment004Run = "",
    [string]$Experiment006Run = "",
    [string]$Output = "",
    [switch]$SkipSpeculative,
    [switch]$SkipMoe,
    [switch]$SkipBackground,
    [switch]$SkipArm64,
    [switch]$Smoke,
    [switch]$Resume,
    [switch]$Profile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepositoryRoot
$ConfigPath = (Resolve-Path (Join-Path $RepositoryRoot $Config)).Path

$UvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($null -ne $UvCommand) {
    $UvExe = $UvCommand.Source
}
else {
    $UvCandidate = Join-Path $env:APPDATA "Python\Python311\Scripts\uv.exe"
    if (-not (Test-Path -LiteralPath $UvCandidate)) {
        throw "uv is required but was not found on PATH or at $UvCandidate"
    }
    $UvExe = $UvCandidate
}

$Arguments = @(
    "run",
    "--no-sync",
    "swarm",
    "experiment",
    "heterogeneous-node-utility",
    "--config",
    $ConfigPath
)
if ($Experiment004Run) { $Arguments += @("--experiment-004-run", $Experiment004Run) }
if ($Experiment006Run) { $Arguments += @("--experiment-006-run", $Experiment006Run) }
if ($Output) { $Arguments += @("--output", $Output) }
if ($SkipSpeculative) { $Arguments += "--skip-speculative" }
if ($SkipMoe) { $Arguments += "--skip-moe" }
if ($SkipBackground) { $Arguments += "--skip-background" }
if ($SkipArm64) { $Arguments += "--skip-arm64" }
if ($Smoke) { $Arguments += "--smoke" }
if ($Resume) { $Arguments += "--resume" }
if ($Profile) { $Arguments += "--profile" }

$ExitCode = 1
try {
    Write-Host "[Experiment 007] Starting heterogeneous real-model execution"
    & $UvExe @Arguments
    $ExitCode = $LASTEXITCODE
}
finally {
    $Containers = & docker ps -a --filter "name=swarm-exp007-" --format "{{.Names}}" 2>$null
    foreach ($Container in $Containers) {
        if ($Container -like "swarm-exp007-*") {
            & docker stop --time 15 $Container 2>$null | Out-Null
        }
    }
}

if ($ExitCode -ne 0) {
    Write-Error (
        "Experiment 007 mandatory infrastructure failed with exit code {0}. " +
        "Partial evidence remains under artifacts\runs or the requested output path."
    ) -f $ExitCode
}
exit $ExitCode
