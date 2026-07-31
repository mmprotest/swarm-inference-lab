[CmdletBinding()]
param(
    [string]$PrimaryModel = "Qwen/Qwen3-0.6B",
    [string]$SecondaryModel = "Qwen/Qwen3-4B",
    [switch]$SkipSecondary,
    [switch]$SkipOptionalEngines,
    [string]$OutputRoot = "artifacts/runs",
    [switch]$Resume,
    [switch]$Smoke,
    [switch]$Profile,
    [switch]$KeepServers
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepositoryRoot

$ConfigPath = Join-Path $RepositoryRoot `
    "configs\experiments\experiment_004_engine_performance.yaml"
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Experiment 004 configuration is missing: $ConfigPath"
}
if ($PrimaryModel -ne "Qwen/Qwen3-0.6B") {
    throw "Experiment 004 requires Qwen/Qwen3-0.6B as the primary model."
}
if ($SecondaryModel -ne "Qwen/Qwen3-4B") {
    throw "Experiment 004 requires Qwen/Qwen3-4B as the secondary model."
}

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

$OutputDirectory = Join-Path $RepositoryRoot $OutputRoot
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$ExternalEnvironmentDirectory = Join-Path $RepositoryRoot `
    "artifacts\engine-environments"
New-Item -ItemType Directory -Force `
    -Path $ExternalEnvironmentDirectory | Out-Null

$Arguments = @(
    "run",
    "--no-sync",
    "swarm",
    "experiment",
    "engine-performance",
    "--config",
    $ConfigPath,
    "--primary-model",
    $PrimaryModel,
    "--secondary-model",
    $SecondaryModel,
    "--output-root",
    $OutputRoot
)
if ($SkipSecondary) {
    $Arguments += "--skip-secondary"
}
if ($SkipOptionalEngines) {
    $Arguments += "--skip-optional-engines"
}
if ($Resume) {
    $Arguments += "--resume"
}
if ($Smoke) {
    $Arguments += "--smoke"
}
if ($Profile) {
    $Arguments += "--profile"
}
if ($KeepServers) {
    $Arguments += "--keep-servers"
}

$ExitCode = 1
try {
    Write-Host "[Experiment 004] Starting isolated production-engine benchmark"
    & $UvExe @Arguments
    $ExitCode = $LASTEXITCODE
}
finally {
    if (-not $KeepServers) {
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                $_.CommandLine -like "*sglang.launch_server*" -or
                $_.CommandLine -like "*vllm.entrypoints.openai.api_server*"
            } |
            ForEach-Object {
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            }
    }
}

if ($ExitCode -ne 0) {
    $FailureMessage = (
        "Experiment 004 did not pass (exit code {0}). Partial evidence was " +
        "preserved under {1}."
    ) -f $ExitCode, $OutputDirectory
    Write-Error $FailureMessage
}
exit $ExitCode
