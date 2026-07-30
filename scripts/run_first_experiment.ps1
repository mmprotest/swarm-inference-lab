param(
    [ValidateSet("synthetic", "cpu", "cuda")]
    [string]$Backend = "synthetic",
    [string]$Config = "configs/experiments/experiment_001_replica_scaling.yaml",
    [switch]$Bootstrap,
    [switch]$Profile
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot "common.ps1")

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
$env:UV_CACHE_DIR = Join-Path $repo ".uv-cache"
$venv = Join-Path $repo ".venv"
$python = Join-Path $venv "Scripts\python.exe"
$configPath = if ([System.IO.Path]::IsPathRooted($Config)) {
    (Resolve-Path -LiteralPath $Config).Path
} else {
    (Resolve-Path -LiteralPath (Join-Path $repo $Config)).Path
}

function Get-InstalledPackageMap {
    param([string]$Interpreter)

    if (-not (Test-Path -LiteralPath $Interpreter)) {
        return @{}
    }
    $packageJson = & $Interpreter -c @"
import importlib.metadata, json
print(json.dumps({
    (dist.metadata.get('Name') or '').lower(): dist.version
    for dist in importlib.metadata.distributions()
    if dist.metadata.get('Name')
}, sort_keys=True))
"@
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inventory packages with $Interpreter."
    }
    $result = @{}
    $parsed = $packageJson | ConvertFrom-Json
    foreach ($property in $parsed.PSObject.Properties) {
        $result[$property.Name] = [string]$property.Value
    }
    return $result
}

function Test-SyntheticImports {
    param([string]$Interpreter)

    if (-not (Test-Path -LiteralPath $Interpreter)) {
        return $false
    }
    & $Interpreter -c @"
import cryptography, grpc, matplotlib, numpy, pandas, psutil, pydantic, typer, yaml
from google import protobuf
import swarm_inference
print('Synthetic experiment dependencies: PASS')
"@
    return $LASTEXITCODE -eq 0
}

function Assert-PackagePreservation {
    param(
        [hashtable]$Before,
        [hashtable]$After
    )

    $violations = @()
    foreach ($name in $Before.Keys) {
        if (-not $After.ContainsKey($name)) {
            $violations += "$name removed (was $($Before[$name]))"
        } elseif ($After[$name] -ne $Before[$name]) {
            $violations += "$name changed $($Before[$name]) -> $($After[$name])"
        }
    }
    if ($violations.Count -gt 0) {
        throw (
            "Dependency preservation check failed: " +
            ($violations -join "; ")
        )
    }
    Write-Host "Dependency preservation check: PASS ($($Before.Count) pre-existing packages retained)."
}

$uv = Get-SwarmUv
if ($null -eq $uv) {
    throw "uv is unavailable. Install uv, then rerun with -Bootstrap if the environment needs creation."
}
$uvVersion = (& $uv --version)
$uvRunHelp = (& $uv help run | Out-String)
if ($LASTEXITCODE -ne 0 -or $uvRunHelp -notmatch "--no-sync") {
    throw "Installed uv does not support the required non-synchronising run mode."
}
Write-Host "Using $uvVersion; verified that uv run --no-sync is supported."

$environmentExists = Test-Path -LiteralPath $python
$beforePackages = if ($environmentExists) {
    Get-InstalledPackageMap -Interpreter $python
} else {
    @{}
}
$dependenciesReady = $false
if ($environmentExists) {
    $dependenciesReady = Test-SyntheticImports -Interpreter $python
}

if (-not $dependenciesReady) {
    if (-not $Bootstrap) {
        throw (
            "The existing .venv is missing or cannot import required synthetic " +
            "experiment packages. No dependency mutation was attempted. " +
            "Rerun with -Bootstrap to install only missing requirements."
        )
    }
    if (-not $environmentExists) {
        Write-Host "Dependency action: creating .venv with Python 3.11."
        & $uv venv --python 3.11 $venv
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        $environmentExists = $true
    }
    $extra = if ($Backend -eq "synthetic") {
        "."
    } else {
        ".[$Backend]"
    }
    Write-Host (
        "Dependency action: non-pruning install into the existing environment " +
        "using 'uv pip install --editable $extra'. No unrelated optional " +
        "dependency will be uninstalled."
    )
    & $uv pip install --python $python --editable $extra
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    if (-not (Test-SyntheticImports -Interpreter $python)) {
        throw "Required synthetic experiment imports still fail after bootstrap."
    }
} else {
    Write-Host "Dependency action: none; using the existing environment with uv run --no-sync."
}

$env:VIRTUAL_ENV = $venv
$exitCode = 1
try {
    Write-Host "Running environment doctor with the experiment interpreter (backend=$Backend)..."
    & $uv run --no-sync --python $python swarm doctor --backend $Backend
    if ($LASTEXITCODE -ne 0) {
        $exitCode = $LASTEXITCODE
        throw "Environment doctor failed; the experiment was not started."
    }

    Write-Host "Starting Experiment 001 (single-host-loopback, direct data plane)..."
    $experimentArguments = @(
        "run", "--no-sync", "--python", $python,
        "swarm", "experiment", "--config", $configPath
    )
    if ($Profile) {
        $experimentArguments += "--profile"
    }
    & $uv @experimentArguments
    $exitCode = $LASTEXITCODE
} catch {
    if ($exitCode -eq 0) {
        $exitCode = 1
    }
    Write-Host "Experiment failed: $_" -ForegroundColor Red
} finally {
    Write-Host "Stopping child workers, coordinator, and peer streams..."
    $afterPackages = Get-InstalledPackageMap -Interpreter $python
    try {
        Assert-PackagePreservation -Before $beforePackages -After $afterPackages
    } catch {
        $exitCode = 1
        Write-Host "Dependency validation failed: $_" -ForegroundColor Red
    }
    Write-Host "Experiment process cleanup completed."
}

if ($exitCode -ne 0) {
    Write-Host "Experiment/report status is FAIL (exit $exitCode)." -ForegroundColor Yellow
}
exit $exitCode
