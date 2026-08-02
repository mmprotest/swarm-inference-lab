[CmdletBinding()]
param(
    [ValidateSet("quick", "development", "full", "frontier")]
    [string]$Mode,
    [string]$ModelPathLevelA,
    [string]$ModelPathLevelB,
    [string]$KimiFixturePath,
    [string]$ColibriPath,
    [string]$OutputDirectory,
    [switch]$Resume,
    [switch]$RebuildColibri,
    [switch]$RebuildCuda,
    [switch]$ApplyBridgePatches,
    [string]$Topology,
    [string]$NetworkProfile,
    [string]$Configuration,
    [ValidateRange(1, 100)]
    [int]$Repeats = 1,
    [ValidateSet("off", "summary", "detailed", "trace")]
    [string]$TelemetryLevel = "detailed",
    [switch]$SkipModelDownload,
    [switch]$SkipLevelB,
    [switch]$SkipKimiFixture,
    [switch]$Quick,
    [switch]$Full,
    [switch]$Frontier,
    [string]$ModelPathFrontier
)

$ErrorActionPreference = "Stop"
$modeSwitches = @($Quick.IsPresent, $Full.IsPresent, $Frontier.IsPresent) |
    Where-Object { $_ }
if ($modeSwitches.Count -gt 1) {
    throw "Choose only one of -Quick, -Full, or -Frontier."
}
if ($Mode -and $modeSwitches.Count -gt 0) {
    throw "Use either -Mode or a mode switch, not both."
}
if ($Quick) { $Mode = "quick" }
elseif ($Full) { $Mode = "full" }
elseif ($Frontier) { $Mode = "frontier" }
elseif (-not $Mode) {
    $Mode = "quick"
    Write-Host "No mode selected; running quick. Quick evidence cannot produce an official verdict."
}
if ($Mode -eq "full" -and $Repeats -lt 3) {
    $Repeats = 3
    Write-Host "Full mode requires at least three repeats; using 3."
}

$experimentDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $experimentDirectory "..\..")).Path
if (-not (Test-Path -LiteralPath (Join-Path $repositoryRoot "pyproject.toml") -PathType Leaf)) {
    throw "Could not locate the swarm-inference-lab repository."
}
$python = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = (Get-Command python -ErrorAction Stop).Source
}

$runtimeColibriPath = $ColibriPath
if ($RebuildColibri -or $RebuildCuda -or $ApplyBridgePatches) {
    $buildScript = Join-Path $repositoryRoot "integrations\colibri\build.ps1"
    $buildArguments = @("-PythonPath", $python)
    if ($ColibriPath) { $buildArguments += @("-ColibriPath", $ColibriPath) }
    if ($ApplyBridgePatches) { $buildArguments += "-ApplyBridgePatches" }
    if ($RebuildCuda) { $buildArguments += "-BuildCuda" }
    & $buildScript @buildArguments
    if ($LASTEXITCODE -ne 0) { throw "Colibri build failed with exit code $LASTEXITCODE." }
    $runtimeColibriPath = Join-Path $repositoryRoot "build\colibri"
}

$arguments = @(
    "-m", "swarm_inference.experiments.experiment_010.cli",
    "--repository-root", $repositoryRoot,
    "--mode", $Mode,
    "--repeats", $Repeats,
    "--telemetry-level", $TelemetryLevel
)
if ($ModelPathLevelA) { $arguments += @("--model-path-level-a", $ModelPathLevelA) }
if ($ModelPathLevelB) { $arguments += @("--model-path-level-b", $ModelPathLevelB) }
if ($KimiFixturePath) { $arguments += @("--kimi-fixture-path", $KimiFixturePath) }
if ($runtimeColibriPath) { $arguments += @("--colibri-path", $runtimeColibriPath) }
if ($OutputDirectory) { $arguments += @("--output-directory", $OutputDirectory) }
if ($Resume) { $arguments += "--resume" }
if ($RebuildColibri) { $arguments += "--rebuild-colibri" }
if ($RebuildCuda) { $arguments += "--rebuild-cuda" }
if ($ApplyBridgePatches) { $arguments += "--apply-bridge-patches" }
if ($Topology) { $arguments += @("--topology", $Topology) }
if ($NetworkProfile) { $arguments += @("--network-profile", $NetworkProfile) }
if ($Configuration) { $arguments += @("--configuration", $Configuration) }
if ($SkipModelDownload) { $arguments += "--skip-model-download" }
if ($SkipLevelB) { $arguments += "--skip-level-b" }
if ($SkipKimiFixture) { $arguments += "--skip-kimi-fixture" }
if ($ModelPathFrontier) { $arguments += @("--model-path-frontier", $ModelPathFrontier) }

$previousPythonPath = $env:PYTHONPATH
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
$sourceRoot = Join-Path $repositoryRoot "src"
$runTemp = Join-Path $repositoryRoot ".experiment-010-temp"
New-Item -ItemType Directory -Force $runTemp | Out-Null
$env:PYTHONPATH = if ($previousPythonPath) { "$sourceRoot;$previousPythonPath" } else { $sourceRoot }
$env:TEMP = $runTemp
$env:TMP = $runTemp
try {
    Push-Location $repositoryRoot
    & $python @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
}
