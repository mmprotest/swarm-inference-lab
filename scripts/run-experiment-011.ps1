[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$Full,
    [switch]$SkipSpeculation,
    [switch]$NetworkOnly,
    [switch]$ExactnessOnly,
    [switch]$Resume,
    [string]$RunId,
    [string]$ModelPath,
    [string]$DraftModelPath,
    [ValidateSet(2, 4, 8)]
    [int[]]$StageCounts = @(2, 4, 8),
    [ValidateSet(
        'loopback_unshaped',
        'fabric_100g',
        'lan_10g',
        'lan_2_5g',
        'lan_1g',
        'wifi',
        'regional_wan',
        'global_wan'
    )]
    [string[]]$ProfileNames = @(
        'loopback_unshaped',
        'fabric_100g',
        'lan_10g',
        'lan_2_5g',
        'lan_1g',
        'wifi',
        'regional_wan',
        'global_wan'
    )
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($Quick -and $Full) {
    throw '-Quick and -Full are mutually exclusive.'
}
if ($NetworkOnly -and $ExactnessOnly) {
    throw '-NetworkOnly and -ExactnessOnly are mutually exclusive.'
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Repository Python environment was not found: $python"
}

$mode = if ($Quick) { 'quick' } else { 'full' }
$cliArgs = @(
    '-m',
    'swarm_inference.experiments.experiment_011',
    'run',
    '--mode',
    $mode,
    '--stage-counts'
)
$cliArgs += $StageCounts | ForEach-Object { [string]$_ }
$cliArgs += '--profile-names'
$cliArgs += $ProfileNames

if ($SkipSpeculation) { $cliArgs += '--skip-speculation' }
if ($NetworkOnly) { $cliArgs += '--network-only' }
if ($ExactnessOnly) { $cliArgs += '--exactness-only' }
if ($Resume) { $cliArgs += '--resume' }
if ($RunId) { $cliArgs += @('--run-id', $RunId) }
if ($ModelPath) { $cliArgs += @('--model-path', $ModelPath) }
if ($DraftModelPath) { $cliArgs += @('--draft-model-path', $DraftModelPath) }

Write-Host "Experiment 011 mode: $mode"
Write-Host "Repository: $repositoryRoot"
Push-Location $repositoryRoot
try {
    & $python @cliArgs
    $experimentExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $experimentExitCode
