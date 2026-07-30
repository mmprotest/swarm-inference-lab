param(
    [ValidateSet("auto", "synthetic", "cpu", "cuda", "mps")]
    [string]$Backend = "auto",
    [switch]$NoDev
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot "common.ps1")

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
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

$uv = Get-SwarmUv
if ($null -eq $uv) {
    Write-Host "Installing uv for the current Windows user..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $uvBin = Join-Path $env:USERPROFILE ".local\bin"
    if (Test-Path $uvBin) {
        $env:PATH = "$uvBin;$env:PATH"
    }
    $uv = Get-SwarmUv
}
if ($null -eq $uv) {
    throw "uv installation completed but uv.exe could not be located."
}

$resolvedBackend = Resolve-SwarmBackend $Backend
$arguments = @("sync")
if (-not $NoDev) {
    $arguments += @("--extra", "dev")
}
if ($resolvedBackend -ne "synthetic") {
    $arguments += @("--extra", $resolvedBackend)
}

Write-Host "Synchronising native Windows environment for backend '$resolvedBackend'..."
& $uv @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $uv run --no-sync swarm doctor --backend $resolvedBackend
$status = $LASTEXITCODE
if ($status -ne 0) {
    Write-Host "Environment installed, but the selected backend is not operational." -ForegroundColor Yellow
}
exit $status
