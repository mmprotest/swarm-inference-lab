param(
    [ValidateSet("auto", "synthetic", "cpu", "cuda", "mps")]
    [string]$Backend = "auto",
    [switch]$AllowCpu
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot "common.ps1")
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
$env:UV_CACHE_DIR = Join-Path $repo ".uv-cache"

$uv = Get-SwarmUv
if ($null -eq $uv) {
    Write-Error "uv is not installed. Run: .\scripts\bootstrap.ps1 -Backend $Backend"
    exit 2
}

$arguments = @("run", "--no-sync", "swarm", "doctor", "--backend", $Backend)
if ($AllowCpu) {
    $arguments += "--allow-cpu"
}
& $uv @arguments
exit $LASTEXITCODE
