[CmdletBinding()]
param(
    [string]$Config = "configs\experiments\experiment_007_corrections.yaml",
    [string]$OriginalRun = "",
    [switch]$SkipExpertFix,
    [switch]$SkipBackgroundFix,
    [switch]$Smoke,
    [switch]$Resume,
    [switch]$Profile,
    [string]$OutputRoot = "",
    [switch]$KeepServers
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

# The project environment is deliberately preserved: uv run --no-sync never prunes or syncs it.
$Arguments = @(
    "run",
    "--no-sync",
    "swarm",
    "experiment",
    "experiment-007-corrections",
    "--config",
    $ConfigPath
)
if ($OriginalRun) { $Arguments += @("--original-run", $OriginalRun) }
if ($SkipExpertFix) { $Arguments += "--skip-expert-fix" }
if ($SkipBackgroundFix) { $Arguments += "--skip-background-fix" }
if ($Smoke) { $Arguments += "--smoke" }
if ($Resume) { $Arguments += "--resume" }
if ($Profile) { $Arguments += "--profile" }
if ($OutputRoot) { $Arguments += @("--output-root", $OutputRoot) }
if ($KeepServers) { $Arguments += "--keep-servers" }

$ExitCode = 1
try {
    Write-Host "[Experiment 007 corrections] Starting matched MoE and fixed-window measurements"
    & $UvExe @Arguments
    $ExitCode = $LASTEXITCODE
}
finally {
    if (-not $KeepServers) {
        $Containers = & docker ps -a --filter "name=swarm-exp007-" --format "{{.Names}}" 2>$null
        foreach ($Container in $Containers) {
            if ($Container -like "swarm-exp007-*-correction") {
                & docker stop --time 15 $Container 2>$null | Out-Null
            }
        }
    }
}

if ($ExitCode -ne 0) {
    Write-Error (
        "Experiment 007 correction benchmark integrity failed with exit code {0}. " +
        "Partial evidence remains under artifacts\runs or the requested output root."
    ) -f $ExitCode
}
exit $ExitCode
