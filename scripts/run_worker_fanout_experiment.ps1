[CmdletBinding()]
param(
    [string]$ModelId = "Qwen/Qwen3-0.6B",
    [string]$Revision = "c1899de289a04d12100db370d81485cdf75e47ca",
    [int[]]$WorkerCounts = @(1, 2, 4, 7, 14, 21, 28),
    [ValidateRange(1, 100)]
    [int]$Repeats = 3,
    [ValidateRange(1, 28)]
    [int]$MaxWorkerCount = 28,
    [string]$OutputRoot = "artifacts/runs",
    [switch]$Smoke,
    [switch]$SkipAcquisitionTests,
    [switch]$SkipRejoinTest,
    [switch]$Resume,
    [switch]$KeepWorkers,
    [switch]$Profile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepositoryRoot

if ($KeepWorkers) {
    throw "-KeepWorkers is incompatible with mandatory Experiment 003 cleanup evidence."
}
if ($Revision -ne "c1899de289a04d12100db370d81485cdf75e47ca") {
    throw "Experiment 003 requires immutable revision c1899de289a04d12100db370d81485cdf75e47ca."
}
if (-not (Test-Path -LiteralPath "pyproject.toml")) {
    throw "Run this launcher from the swarm-inference-lab repository."
}

$UvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($null -ne $UvCommand) {
    $UvExe = $UvCommand.Source
}
else {
    $UvCandidate = Join-Path $env:APPDATA "Python\Python311\Scripts\uv.exe"
    if (-not (Test-Path -LiteralPath $UvCandidate)) {
        throw "uv is required but was not found on PATH or at $UvCandidate"
    }
    $UvExe = $UvCandidate
}

$ConfigPath = Join-Path $RepositoryRoot "configs\experiments\experiment_003_worker_fanout.yaml"
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Experiment 003 configuration is missing: $ConfigPath"
}
$TestTemp = Join-Path $RepositoryRoot "artifacts\test-temp"
New-Item -ItemType Directory -Force -Path $TestTemp | Out-Null
$env:TEMP = $TestTemp
$env:TMP = $TestTemp

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )
    Write-Host $Label
    & $UvExe run --no-sync @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

$BaselineWorkerPids = @(
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -like "*swarm_inference.worker.process_main*"
        } |
        Select-Object -ExpandProperty ProcessId
)
$ExitCode = 1
try {
    Write-Host "[1/9] Inspecting environment"
    Invoke-Checked -Label "Checking CUDA imports" -Arguments @(
        "python",
        "-c",
        "import torch,transformers,safetensors; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
    )
    Invoke-Checked -Label "Running CUDA doctor" -Arguments @(
        "swarm", "doctor", "--backend", "cuda"
    )

    Write-Host "[quality] Running required Ruff, formatting, mypy, and pytest gates"
    Invoke-Checked -Label "ruff check ." -Arguments @("ruff", "check", ".")
    Invoke-Checked -Label "ruff format --check ." -Arguments @(
        "ruff", "format", "--check", "."
    )
    Invoke-Checked -Label "mypy src" -Arguments @("mypy", "src")
    Invoke-Checked -Label "pytest" -Arguments @("pytest", "-q")

    Write-Host "[2/9] Preparing stage layouts"
    $RunArguments = @(
        "run", "--no-sync", "swarm", "experiment", "worker-fanout",
        "--config", $ConfigPath,
        "--model-id", $ModelId,
        "--revision", $Revision,
        "--worker-counts", ($WorkerCounts -join ","),
        "--repeats", [string]$Repeats,
        "--max-worker-count", [string]$MaxWorkerCount,
        "--output", $OutputRoot
    )
    if ($Smoke) {
        $RunArguments += "--smoke"
    }
    if ($SkipAcquisitionTests) {
        $RunArguments += "--skip-acquisition-tests"
    }
    if ($SkipRejoinTest) {
        $RunArguments += "--skip-rejoin-test"
    }
    if ($Resume) {
        $RunArguments += "--resume"
    }
    if ($Profile) {
        $RunArguments += "--profile"
    }

    & $UvExe @RunArguments
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "Experiment 003 returned exit code $ExitCode"
    }
}
finally {
    Write-Host "[cleanup] Checking for experiment worker processes"
    $CurrentWorkers = @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                $_.CommandLine -like "*swarm_inference.worker.process_main*"
            }
    )
    foreach ($Worker in $CurrentWorkers) {
        if ($BaselineWorkerPids -notcontains $Worker.ProcessId) {
            Stop-Process -Id $Worker.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
}

exit $ExitCode
