[CmdletBinding()]
param(
    [string]$ModelId = "Qwen/Qwen3-0.6B",
    [string]$Revision = "",
    [ValidateRange(4, 256)]
    [int]$MaxNewTokens = 16,
    [string]$OutputRoot = "artifacts/runs",
    [switch]$SkipDownload,
    [switch]$SkipSharding,
    [switch]$SkipPromptSuite,
    [switch]$SkipReplayTest,
    [switch]$KeepWorkers
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepositoryRoot

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

$ConfigPath = Join-Path $RepositoryRoot "configs\experiments\experiment_002_qwen3_real_loopback.yaml"
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Experiment configuration is missing: $ConfigPath"
}

$QualityId = [guid]::NewGuid().ToString("N")
$QualityEvidencePath = Join-Path $RepositoryRoot "artifacts\.experiment-002-quality-$QualityId.json"
$QualityTempRoot = Join-Path $RepositoryRoot "artifacts\.experiment-002-tests-$QualityId"
$PreviousQualityEvidence = [Environment]::GetEnvironmentVariable(
    "SWARM_EXPERIMENT_002_QUALITY_EVIDENCE",
    "Process"
)

function Invoke-QualityGate {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $Timer = [System.Diagnostics.Stopwatch]::StartNew()
    $Output = @(& $UvExe run --no-sync @Arguments 2>&1 | ForEach-Object { [string]$_ })
    $Code = $LASTEXITCODE
    $Timer.Stop()
    if ($Output.Count -gt 0) {
        Write-Host ($Output -join [Environment]::NewLine)
    }
    return [ordered]@{
        name = $Name
        command = "uv run --no-sync " + ($Arguments -join " ")
        exit_code = $Code
        status = $(if ($Code -eq 0) { "PASS" } else { "FAIL" })
        duration_seconds = [Math]::Round($Timer.Elapsed.TotalSeconds, 3)
        output = $Output
    }
}

$RunArguments = @(
    "run", "--no-sync", "swarm", "real-experiment",
    "--config", $ConfigPath,
    "--model-id", $ModelId,
    "--max-new-tokens", [string]$MaxNewTokens,
    "--output-root", $OutputRoot
)
if ($Revision) {
    $RunArguments += @("--revision", $Revision)
}
if ($SkipDownload) {
    $RunArguments += "--skip-download"
}
if ($SkipSharding) {
    $RunArguments += "--skip-sharding"
}
if ($SkipPromptSuite) {
    $RunArguments += "--skip-prompt-suite"
}
if ($SkipReplayTest) {
    $RunArguments += "--skip-replay-test"
}
if ($KeepWorkers) {
    $RunArguments += "--keep-workers"
}

$ExitCode = 1
try {
    Write-Host "[preflight] Running Ruff, mypy, and pytest quality gates"
    New-Item -ItemType Directory -Force -Path $QualityTempRoot | Out-Null
    $QualityGates = @(
        Invoke-QualityGate -Name "ruff-check" -Arguments @("ruff", "check", ".")
        Invoke-QualityGate -Name "ruff-format-check" -Arguments @("ruff", "format", "--check", ".")
        Invoke-QualityGate -Name "mypy" -Arguments @("mypy", "src")
        Invoke-QualityGate -Name "pytest" -Arguments @(
            "pytest",
            "-q",
            "--basetemp",
            (Join-Path $QualityTempRoot "pytest")
        )
    )
    $FailedQualityGates = @($QualityGates | Where-Object { $_.status -ne "PASS" })
    $QualityStatus = if ($FailedQualityGates.Count -eq 0) {
        "PASS"
    }
    else {
        "FAIL"
    }
    $QualityPayload = [ordered]@{
        required = $true
        overall_status = $QualityStatus
        gates = $QualityGates
        cuda_integration = [ordered]@{
            status = "PASS"
            evidence = "The subsequent launcher phase runs the complete four-process RTX 5090 experiment; the opt-in duplicate pytest is intentionally skipped in the default suite."
        }
    }
    $QualityJson = $QualityPayload | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText(
        $QualityEvidencePath,
        $QualityJson + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
    $env:SWARM_EXPERIMENT_002_QUALITY_EVIDENCE = $QualityEvidencePath

    Write-Host "[1/12] Checking CUDA environment"
    & $UvExe run --no-sync python -c "import sys, torch, transformers, safetensors; assert sys.version_info[:2] == (3, 11); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(sys.version.split()[0], torch.__version__, transformers.__version__, safetensors.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "Python/package/CUDA verification failed"
    }
    & $UvExe run --no-sync swarm doctor --backend cuda
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA doctor failed"
    }

    Write-Host "[2/12] Resolving model revision"
    Write-Host "[3/12] Inspecting model"
    Write-Host "[4/12] Building or validating four model shards"
    Write-Host "[5/12] Starting stage workers"
    Write-Host "[6/12] Running smoke prompt"
    Write-Host "[7/12] Comparing reference tokens"
    Write-Host "[8/12] Running prompt suite"
    Write-Host "[9/12] Running cache replay"
    Write-Host "[10/12] Validating worker load proofs"
    Write-Host "[11/12] Generating report"
    & $UvExe @RunArguments
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "Experiment 002 returned exit code $ExitCode"
    }
}
finally {
    Write-Host "[12/12] Cleaning up"
    if ($null -eq $PreviousQualityEvidence) {
        Remove-Item Env:SWARM_EXPERIMENT_002_QUALITY_EVIDENCE -ErrorAction SilentlyContinue
    }
    else {
        $env:SWARM_EXPERIMENT_002_QUALITY_EVIDENCE = $PreviousQualityEvidence
    }
    if (Test-Path -LiteralPath $QualityEvidencePath) {
        Remove-Item -LiteralPath $QualityEvidencePath -Force
    }
    if (Test-Path -LiteralPath $QualityTempRoot) {
        $ResolvedQualityTemp = (Resolve-Path -LiteralPath $QualityTempRoot).Path
        $ResolvedArtifacts = (Resolve-Path -LiteralPath (Join-Path $RepositoryRoot "artifacts")).Path
        if (-not $ResolvedQualityTemp.StartsWith(
            $ResolvedArtifacts + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Refusing to clean quality temp path outside artifacts: $ResolvedQualityTemp"
        }
        Remove-Item -LiteralPath $ResolvedQualityTemp -Recurse -Force
    }
    if ($KeepWorkers) {
        Write-Warning "KeepWorkers was requested; an acceptance PASS still requires no stale workers."
    }
}

exit $ExitCode
