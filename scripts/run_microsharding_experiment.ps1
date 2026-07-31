[CmdletBinding()]
param(
    [string]$Config = "configs\experiments\experiment_006_microsharding.yaml",
    [string]$DenseModel = "Qwen/Qwen3-0.6B",
    [string]$DenseRevision = "c1899de289a04d12100db370d81485cdf75e47ca",
    [string]$PipelineStages = "1,4",
    [string]$TensorParallelDegrees = "1,2,4,8",
    [switch]$SkipSecondaryModel,
    [switch]$SkipRealMoe,
    [double]$RealMoeDownloadBudgetGiB = 25,
    [switch]$SkipK3Projection,
    [switch]$Resume,
    [switch]$Smoke,
    [switch]$Profile,
    [string]$Output = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$OriginalLocation = Get-Location
$OriginalProgressPreference = $ProgressPreference
$ProgressPreference = "SilentlyContinue"
$ExitCode = 1

try {
    $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    Set-Location $RepositoryRoot
    $ConfigPath = (Resolve-Path (Join-Path $RepositoryRoot $Config)).Path

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

    Write-Host "[Experiment 006] CUDA doctor"
    & $UvExe run --no-sync swarm doctor --backend cuda --json
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA doctor failed with exit code $LASTEXITCODE"
    }

    $Arguments = @(
        "run",
        "--no-sync",
        "swarm",
        "experiment",
        "microsharding",
        "--config",
        $ConfigPath,
        "--dense-model",
        $DenseModel,
        "--dense-revision",
        $DenseRevision,
        "--pipeline-stages",
        $PipelineStages,
        "--tensor-parallel-degrees",
        $TensorParallelDegrees,
        "--real-moe-download-budget-gib",
        $RealMoeDownloadBudgetGiB.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
    if ($SkipSecondaryModel) { $Arguments += "--skip-secondary-model" }
    if ($SkipRealMoe) { $Arguments += "--skip-real-moe" }
    if ($SkipK3Projection) { $Arguments += "--skip-k3-projection" }
    if ($Resume) { $Arguments += "--resume" }
    if ($Smoke) { $Arguments += "--smoke" }
    if ($Profile) { $Arguments += "--profile" }
    if ($Output) { $Arguments += @("--output", $Output) }

    Write-Host "[Experiment 006] Running dense correctness, projections, MoE, and K3 phases"
    & $UvExe @Arguments
    $ExitCode = $LASTEXITCODE
}
finally {
    # The runner unloads rank modules and empties CUDA allocations between
    # configurations.  No background CUDA or network worker is created.
    $ProgressPreference = $OriginalProgressPreference
    Set-Location $OriginalLocation
    Write-Host "[Experiment 006] Cleanup complete; partial evidence is preserved on failure."
}

if ($ExitCode -ne 0) {
    Write-Error "Experiment 006 failed a mandatory gate (exit code $ExitCode)."
}
exit $ExitCode
