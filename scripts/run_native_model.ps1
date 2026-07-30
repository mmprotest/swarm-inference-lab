param(
    [Parameter(Mandatory = $true)]
    [string]$Shards,
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [string]$Output = "artifacts/validation/native-hardware",
    [ValidateSet("auto", "cpu", "cuda", "mps")]
    [string]$Backend = "auto",
    [int]$Workers = 3,
    [int]$MaxNewTokens = 4,
    [switch]$SkipSync
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot "common.ps1")
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
$env:UV_CACHE_DIR = Join-Path $repo ".uv-cache"

if ($Backend -eq "auto") {
    $Backend = if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
        "cuda"
    } else {
        "cpu"
    }
}
if (-not $SkipSync) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $PSScriptRoot "bootstrap.ps1") -Backend $Backend
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
$uv = Get-SwarmUv
if ($null -eq $uv) {
    throw "uv is not available after bootstrap."
}

$workerBackend = switch ($Backend) {
    "cuda" { "torch-cuda" }
    "mps" { "torch-mps" }
    default { "torch-cpu" }
}
$device = switch ($Backend) {
    "cuda" { "cuda" }
    "mps" { "mps" }
    default { "cpu" }
}
$dtype = switch ($Backend) {
    "cuda" { "bfloat16" }
    "mps" { "float16" }
    default { "float32" }
}

& $uv run --no-sync swarm validate-model `
    --shards (Resolve-Path $Shards).Path `
    --model-path (Resolve-Path $ModelPath).Path `
    --output $Output `
    --device $device `
    --dtype $dtype `
    --max-new-tokens $MaxNewTokens `
    --distributed-loopback-workers $Workers `
    --distributed-backend $workerBackend
exit $LASTEXITCODE
