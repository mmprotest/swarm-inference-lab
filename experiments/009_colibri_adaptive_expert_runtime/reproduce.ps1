[CmdletBinding()]
param(
    [string]$ColibriPath,
    [string]$ModelPath,
    [string]$ModelFamily,
    [string]$OutputDirectory,
    [switch]$Quick,
    [switch]$Full,
    [switch]$Resume,
    [switch]$RebuildColibri,
    [switch]$ApplyBridgePatches,
    [ValidateSet("off", "summary", "detailed", "trace")]
    [string]$TelemetryLevel,
    [ValidateSet("A", "B", "C", "D", "E")]
    [string]$Configuration,
    [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"

if ($Quick -and $Full) {
    throw "Choose only one of -Quick or -Full."
}
if (-not $Quick -and -not $Full) {
    $Quick = $true
    Write-Host "No mode selected; running -Quick. A fixture-only run cannot receive an official pass."
}

$experimentDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = $null
$searchDirectory = (Resolve-Path $experimentDirectory).Path
while ($searchDirectory) {
    $configProbe = Join-Path $searchDirectory "configs\experiments\experiment_009_colibri.yaml"
    if ((Test-Path -LiteralPath (Join-Path $searchDirectory "pyproject.toml") -PathType Leaf) -and
        (Test-Path -LiteralPath $configProbe -PathType Leaf)) {
        $repositoryRoot = $searchDirectory
        break
    }
    $parent = Split-Path -Parent $searchDirectory
    if (-not $parent -or $parent -eq $searchDirectory) { break }
    $searchDirectory = $parent
}
if (-not $repositoryRoot) {
    throw "Could not locate the swarm-inference-lab repository above $experimentDirectory."
}

$python = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = (Get-Command python -ErrorAction Stop).Source
}
$configurationPath = Join-Path $repositoryRoot "configs\experiments\experiment_009_colibri.yaml"
$fixtureConfig = Join-Path $repositoryRoot "build\fixtures\glm_tiny\config.json"
$fixtureWeights = Join-Path $repositoryRoot "build\fixtures\glm_tiny\model.safetensors"
if (-not (Test-Path -LiteralPath $fixtureConfig -PathType Leaf) -or
    -not (Test-Path -LiteralPath $fixtureWeights -PathType Leaf)) {
    & (Join-Path $repositoryRoot "integrations\colibri\tests\prepare_fixture.ps1") -PythonPath $python
    if ($LASTEXITCODE -ne 0) { throw "Colibri fixture preparation failed." }
}
$arguments = @(
    "-m", "swarm_inference.cli",
    "experiment", "colibri-adaptive-runtime",
    "--config", $configurationPath
)
if ($Quick) { $arguments += "--quick" }
if ($Full) { $arguments += "--full" }
if ($Resume) { $arguments += "--resume" }
if ($RebuildColibri) { $arguments += "--rebuild-colibri" }
if ($ApplyBridgePatches) { $arguments += "--apply-bridge-patches" }
if ($SkipModelDownload) { $arguments += "--skip-model-download" }
if ($ColibriPath) { $arguments += @("--colibri-path", (Resolve-Path $ColibriPath).Path) }
if ($ModelPath) { $arguments += @("--model-path", (Resolve-Path $ModelPath).Path) }
if ($ModelFamily) { $arguments += @("--model-family", $ModelFamily) }
if ($OutputDirectory) { $arguments += @("--output-directory", $OutputDirectory) }
if ($TelemetryLevel) { $arguments += @("--telemetry-level", $TelemetryLevel) }
if ($Configuration) { $arguments += @("--configuration", $Configuration) }

$existingPythonPath = $env:PYTHONPATH
$sourceRoot = Join-Path $repositoryRoot "src"
$env:PYTHONPATH = if ($existingPythonPath) { "$sourceRoot;$existingPythonPath" } else { $sourceRoot }
try {
    Push-Location $repositoryRoot
    & $python @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
    $env:PYTHONPATH = $existingPythonPath
}
