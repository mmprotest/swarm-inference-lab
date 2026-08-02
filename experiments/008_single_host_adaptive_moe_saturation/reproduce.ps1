[CmdletBinding()]
param(
    [string]$ModelPath,
    [string]$OutputDirectory,
    [switch]$Resume,
    [switch]$Quick,
    [switch]$Full,
    [switch]$SkipDownload,
    [ValidateSet("A", "B", "C", "D", "E", "F", "G")]
    [string]$Configuration,
    [string]$ServerPath
)

$ErrorActionPreference = "Stop"

if ($Quick -and $Full) {
    throw "Choose only one of -Quick or -Full."
}
if (-not $Quick -and -not $Full) {
    $Quick = $true
    Write-Host "No mode selected; running -Quick. Only -Full can produce an official verdict."
}

$experimentDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = $null
$searchDirectory = (Resolve-Path $experimentDirectory).Path
while ($searchDirectory) {
    $configProbe = Join-Path $searchDirectory "configs\experiments\experiment_008_adaptive_moe.yaml"
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

$configurationPath = Join-Path $repositoryRoot "configs\experiments\experiment_008_adaptive_moe.yaml"
$arguments = @(
    "-m", "swarm_inference.cli",
    "experiment", "adaptive-moe-saturation",
    "--config", $configurationPath
)
if ($Quick) { $arguments += "--quick" }
if ($Full) { $arguments += "--full" }
if ($Resume) { $arguments += "--resume" }
if ($SkipDownload) { $arguments += "--skip-download" }
if ($ModelPath) { $arguments += @("--model-path", (Resolve-Path $ModelPath).Path) }
if ($OutputDirectory) { $arguments += @("--output-directory", $OutputDirectory) }
if ($Configuration) { $arguments += @("--configuration", $Configuration) }
if ($ServerPath) { $arguments += @("--server-path", (Resolve-Path $ServerPath).Path) }

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
