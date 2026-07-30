param(
    [string]$Config = "configs/experiments/scaling_loopback.yaml",
    [ValidateSet("auto", "synthetic", "cpu", "cuda", "mps")]
    [string]$Backend = "auto",
    [switch]$AllowCpuDoctor,
    [switch]$SkipSync,
    [int]$Repeats = 1,
    [double]$DurationS = 0
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot "common.ps1")

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$configPath = if ([System.IO.Path]::IsPathRooted($Config)) {
    (Resolve-Path $Config).Path
} else {
    (Resolve-Path (Join-Path $repo $Config)).Path
}
Set-Location $repo
$env:UV_CACHE_DIR = Join-Path $repo ".uv-cache"

function Resolve-SwarmBackend {
    param([string]$Requested)
    if ($Requested -ne "auto") {
        return $Requested
    }
    if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
        return "cuda"
    }
    return "cpu"
}

$resolvedBackend = Resolve-SwarmBackend $Backend
$uv = Get-SwarmUv
if ($null -eq $uv) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $PSScriptRoot "bootstrap.ps1") -Backend $resolvedBackend
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $uv = Get-SwarmUv
}
if ($null -eq $uv) {
    throw "uv is not available after bootstrap."
}

if (-not $SkipSync) {
    $syncArguments = @("sync", "--extra", "dev")
    if ($resolvedBackend -ne "synthetic") {
        $syncArguments += @("--extra", $resolvedBackend)
    }
    & $uv @syncArguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$doctorArguments = @(
    "run", "--no-sync", "swarm", "doctor", "--backend", $resolvedBackend
)
if ($AllowCpuDoctor) {
    $doctorArguments += "--allow-cpu"
}
& $uv @doctorArguments
if ($LASTEXITCODE -ne 0) {
    $doctorExitCode = $LASTEXITCODE
    Write-Error "Environment doctor failed; the experiment was not started."
    exit $doctorExitCode
}

Write-Host "Starting native coordinator and process-isolated workers..."
$experimentArguments = @(
    "run", "--no-sync", "swarm", "experiment",
    "--config", $configPath,
    "--repeats", $Repeats
)
if ($DurationS -gt 0) {
    $experimentArguments += @("--duration-s", $DurationS)
}
& $uv @experimentArguments
$status = $LASTEXITCODE
Write-Host "Experiment process cleanup completed."
if ($status -ne 0) {
    Write-Host "Experiment/report status is FAIL (exit $status)." -ForegroundColor Yellow
}
exit $status
