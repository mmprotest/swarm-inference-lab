[CmdletBinding()]
param(
    [ValidateSet("quick", "development", "full", "frontier")]
    [string]$Mode,
    [string]$ModelPathLevelA,
    [string]$ModelPathLevelASource,
    [string]$ModelPathLevelB,
    [string]$ExpertBankRoot,
    [string]$KimiFixturePath,
    [string]$ColibriPath,
    [string]$OutputDirectory,
    [switch]$Resume,
    [switch]$CorrectionPass,
    [switch]$LevelBOnly,
    [switch]$RebuildColibri,
    [switch]$RebuildExpertWorkers,
    [switch]$RebuildCuda,
    [switch]$ApplyBridgePatches,
    [string]$Topology,
    [string]$NetworkProfile,
    [string]$Configuration,
    [ValidateSet("per_expert_exact", "per_worker_fast")]
    [string]$ResponseMode = "per_expert_exact",
    [switch]$FailureMatrix,
    [switch]$CorruptionMatrix,
    [switch]$RequireCompleteFullRun,
    [switch]$AllowIncomplete,
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
if ($RequireCompleteFullRun -and $AllowIncomplete) {
    throw "-RequireCompleteFullRun and -AllowIncomplete are mutually exclusive."
}
if ($Mode -eq "full" -and -not $AllowIncomplete) {
    $RequireCompleteFullRun = $true
}
if ($LevelBOnly -and -not $CorrectionPass) {
    throw "-LevelBOnly requires -CorrectionPass."
}
if ($LevelBOnly -and $Mode -ne "full") {
    throw "-LevelBOnly requires full mode; -Quick and -Frontier are incompatible."
}
if ($LevelBOnly -and $SkipLevelB) {
    throw "-LevelBOnly and -SkipLevelB are mutually exclusive."
}
if ($LevelBOnly -and ($RebuildColibri -or $RebuildExpertWorkers -or $RebuildCuda -or $ApplyBridgePatches)) {
    throw "-LevelBOnly cannot rebuild Colibri, expert workers, or CUDA; it reuses existing correction evidence."
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

$previousPythonPath = $env:PYTHONPATH
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
$sourceRoot = Join-Path $repositoryRoot "src"
$runTemp = Join-Path $repositoryRoot ".experiment-010-temp"
New-Item -ItemType Directory -Force $runTemp | Out-Null
$env:PYTHONPATH = if ($previousPythonPath) { "$sourceRoot;$previousPythonPath" } else { $sourceRoot }
$env:TEMP = $runTemp
$env:TMP = $runTemp
$finalExitCode = 1
try {
$correctionWork = Join-Path $repositoryRoot "artifacts\runs\experiment-010-correction-work"
$correctionOutput = if ($OutputDirectory) {
    if ([System.IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else { Join-Path $repositoryRoot $OutputDirectory }
} else {
    Join-Path $repositoryRoot "artifacts\runs\experiment-010-correction-final"
}
if ($LevelBOnly) {
    if (-not (Test-Path -LiteralPath $correctionWork -PathType Container)) {
        throw "Existing correction work directory is missing: $correctionWork"
    }
    Write-Host "Validating reusable Experiment 010 correction evidence before Level B acquisition."
    & $python -m swarm_inference.experiments.experiment_010.level_b preflight `
        --repository-root $repositoryRoot `
        --work-directory $correctionWork `
        --final-bundle $correctionOutput
    if ($LASTEXITCODE -ne 0) {
        throw "Existing correction evidence is absent or incomplete; Level B was not started."
    }
}

$runCurrentLevelB = (
    $CorrectionPass -and
    $Mode -eq "full" -and
    -not $SkipLevelB -and
    ($LevelBOnly -or [bool]$ModelPathLevelB)
)
if ($runCurrentLevelB) {
    $levelBRunner = Join-Path $repositoryRoot "experiments\008_single_host_adaptive_moe_saturation\reproduce.ps1"
    $levelBOutput = Join-Path $repositoryRoot "artifacts\runs\experiment-010-correction-work\phase-14\level-b-current"
    $levelBArguments = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $levelBRunner,
        "-Full",
        "-OutputDirectory", $levelBOutput,
        "-Configuration", "A"
    )
    if ($ModelPathLevelB) {
        $resolvedLevelB = (Resolve-Path -LiteralPath $ModelPathLevelB).Path
        $levelBArguments += @("-ModelPath", $resolvedLevelB)
    }
    if ($Resume) { $levelBArguments += "-Resume" }
    if ($SkipModelDownload) { $levelBArguments += "-SkipDownload" }
    if ($LevelBOnly) { $levelBArguments += "-Gate17Only" }
    $windowsPowerShell = (Get-Command powershell.exe -ErrorAction Stop).Source
    Write-Host "Running the current Level B over-VRAM workload through Experiment 008 configuration A."
    & $windowsPowerShell @levelBArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Current Level B workload failed with exit code $LASTEXITCODE. Completed Level A evidence remains resumable."
    }
    if ($LevelBOnly) {
        $levelBConfig = Join-Path $repositoryRoot "configs\experiments\experiment_008_adaptive_moe.yaml"
        Write-Host "Validating current measured Level B rows for Gate 17."
        & $python -m swarm_inference.experiments.experiment_010.level_b validate `
            --level-b-root $levelBOutput `
            --config $levelBConfig
        if ($LASTEXITCODE -ne 0) {
            throw "Gate 17 validation failed; the existing Experiment 010 correction bundle was not rebuilt."
        }
    }
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
if ($ModelPathLevelASource) { $arguments += @("--model-path-level-a-source", $ModelPathLevelASource) }
if ($ModelPathLevelB) { $arguments += @("--model-path-level-b", $ModelPathLevelB) }
if ($ExpertBankRoot) { $arguments += @("--expert-bank-root", $ExpertBankRoot) }
if ($KimiFixturePath) { $arguments += @("--kimi-fixture-path", $KimiFixturePath) }
if ($runtimeColibriPath) { $arguments += @("--colibri-path", $runtimeColibriPath) }
if ($OutputDirectory) { $arguments += @("--output-directory", $OutputDirectory) }
if ($Resume) { $arguments += "--resume" }
if ($CorrectionPass) { $arguments += "--correction-pass" }
if ($RebuildColibri) { $arguments += "--rebuild-colibri" }
if ($RebuildExpertWorkers) { $arguments += "--rebuild-expert-workers" }
if ($RebuildCuda) { $arguments += "--rebuild-cuda" }
if ($ApplyBridgePatches) { $arguments += "--apply-bridge-patches" }
if ($Topology) { $arguments += @("--topology", $Topology) }
if ($NetworkProfile) { $arguments += @("--network-profile", $NetworkProfile) }
if ($Configuration) { $arguments += @("--configuration", $Configuration) }
$arguments += @("--response-mode", $ResponseMode)
if ($FailureMatrix) { $arguments += "--failure-matrix" }
if ($CorruptionMatrix) { $arguments += "--corruption-matrix" }
if ($RequireCompleteFullRun) { $arguments += "--require-complete-full-run" }
if ($AllowIncomplete) { $arguments += "--allow-incomplete" }
if ($SkipModelDownload) { $arguments += "--skip-model-download" }
if ($SkipLevelB) { $arguments += "--skip-level-b" }
if ($SkipKimiFixture) { $arguments += "--skip-kimi-fixture" }
if ($ModelPathFrontier) { $arguments += @("--model-path-frontier", $ModelPathFrontier) }
if ($LevelBOnly) { $arguments += "--level-b-only" }

try {
    Push-Location $repositoryRoot
    & $python @arguments
    $finalExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
}
finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
}
exit $finalExitCode
