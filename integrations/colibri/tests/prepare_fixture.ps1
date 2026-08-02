[CmdletBinding()]
param(
    [string]$PythonPath,
    [string]$OutputDirectory,
    [string]$DependencyDirectory,
    [switch]$Force,
    [switch]$SkipDependencyInstall
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
if (-not $PythonPath) {
    $venvPython = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
    $PythonPath = if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $venvPython
    } else {
        (Get-Command python -ErrorAction Stop).Source
    }
}
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $repositoryRoot 'build\fixtures\glm_tiny'
}
if (-not $DependencyDirectory) {
    $DependencyDirectory = Join-Path $repositoryRoot 'build\fixture-python'
}
$output = [IO.Path]::GetFullPath($OutputDirectory)
$dependencies = [IO.Path]::GetFullPath($DependencyDirectory)
$config = Join-Path $output 'config.json'
$weights = Join-Path $output 'model.safetensors'
$tokenizer = Join-Path $output 'tokenizer.json'
if (-not $Force -and (Test-Path $config) -and (Test-Path $weights) -and (Test-Path $tokenizer)) {
    Write-Host "Colibri fixture already present: $output"
    exit 0
}
if ((Test-Path -LiteralPath $output) -and -not $Force) {
    throw "Incomplete fixture directory exists; use -Force after reviewing it: $output"
}

New-Item -ItemType Directory -Force $dependencies | Out-Null
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ($previousPythonPath) { "$dependencies;$previousPythonPath" } else { $dependencies }
try {
    & $PythonPath -c "import transformers; assert tuple(map(int, transformers.__version__.split('.')[:2])) >= (5, 11)"
    $dependenciesReady = $LASTEXITCODE -eq 0
    if (-not $dependenciesReady) {
        if ($SkipDependencyInstall) {
            throw "Transformers >=5.11 is absent and -SkipDependencyInstall was selected."
        }
        & $PythonPath -m pip install --upgrade --target $dependencies 'transformers==5.14.1' 'safetensors>=0.5'
        if ($LASTEXITCODE -ne 0) { throw "fixture dependency installation failed" }
    }

    $workParent = Join-Path $repositoryRoot 'build\fixture-work'
    New-Item -ItemType Directory -Force $workParent | Out-Null
    $work = Join-Path $workParent ([guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory $work | Out-Null
    try {
        Push-Location $work
        try {
            & $PythonPath (Join-Path $repositoryRoot 'third_party\colibri\c\tools\make_glm_oracle.py')
            if ($LASTEXITCODE -ne 0) { throw "upstream GLM fixture generation failed" }
        }
        finally { Pop-Location }

        New-Item -ItemType Directory -Force $output | Out-Null
        Copy-Item -LiteralPath (Join-Path $work 'glm_tiny\config.json') -Destination $config -Force
        Copy-Item -LiteralPath (Join-Path $work 'glm_tiny\model.safetensors') -Destination $weights -Force
        Copy-Item -LiteralPath (Join-Path $work 'ref_glm.json') -Destination (Join-Path $output 'ref_glm.json') -Force
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'fixtures\glm_tokenizer.json') -Destination $tokenizer -Force
    }
    finally {
        $resolvedWork = (Resolve-Path -LiteralPath $work).Path
        $resolvedParent = (Resolve-Path -LiteralPath $workParent).Path
        if (-not $resolvedWork.StartsWith($resolvedParent, [StringComparison]::OrdinalIgnoreCase)) {
            throw "refusing to remove fixture work directory outside $resolvedParent"
        }
        Remove-Item -LiteralPath $resolvedWork -Recurse -Force
    }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}

$engine = Join-Path $repositoryRoot 'build\colibri\bin\colibri.exe'
if (Test-Path -LiteralPath $engine -PathType Leaf) {
    $saved = @{
        SNAP = $env:SNAP; REF = $env:REF; TF = $env:TF; PROMPT = $env:PROMPT
        CAP_RAISE = $env:CAP_RAISE; AUTOPIN = $env:AUTOPIN
        COLI_SWARM_BRIDGE = $env:COLI_SWARM_BRIDGE
    }
    try {
        $env:SNAP = $output
        $env:REF = Join-Path $output 'ref_glm.json'
        $env:TF = '1'
        $env:PROMPT = $null
        $env:CAP_RAISE = '0'
        $env:AUTOPIN = '0'
        $env:COLI_SWARM_BRIDGE = $null
        $savedErrorPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $validation = (& $engine 8 2>&1) -join "`n"
            $validationExit = $LASTEXITCODE
        }
        finally { $ErrorActionPreference = $savedErrorPreference }
        if ($validationExit -ne 0) { throw "fixture engine validation failed with exit $validationExit" }
        $oracle = [regex]::Match($validation, 'PREFILL \(teacher-forcing\) C vs oracle:\s+(\d+)/32')
        if (-not $oracle.Success -or [int]$oracle.Groups[1].Value -lt 30) {
            throw "fixture teacher-forced oracle score fell below the reviewed 30/32 floor"
        }
    }
    finally {
        foreach ($name in $saved.Keys) {
            if ($null -eq $saved[$name]) { Remove-Item -Path "Env:$name" -ErrorAction SilentlyContinue }
            else { Set-Item -Path "Env:$name" -Value $saved[$name] }
        }
    }
}

foreach ($mutableName in @('.coli_usage', '.coli_kv', 'hot_pinned.bin', '.coli_swarm_telemetry.ndjson')) {
    $mutablePath = Join-Path $output $mutableName
    if (Test-Path -LiteralPath $mutablePath -PathType Leaf) {
        Remove-Item -LiteralPath $mutablePath -Force
    }
}

$manifest = [ordered]@{
    schema_version = 'experiment-009-fixture-v1'
    generator = 'third_party/colibri/c/tools/make_glm_oracle.py'
    transformers = '5.14.1'
    torch_seed = 1234
    expected_prompt = '?'
    expected_input_token_ids = @(63)
    output_token_ids_pinned = $false
    output_identity_rule = 'direct Colibri must equal swarm adapter for the same deterministic process configuration'
    teacher_forced_oracle_minimum = '30/32'
    config_sha256 = (Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash.ToLowerInvariant()
    weights_sha256 = (Get-FileHash -LiteralPath $weights -Algorithm SHA256).Hash.ToLowerInvariant()
    tokenizer_sha256 = (Get-FileHash -LiteralPath $tokenizer -Algorithm SHA256).Hash.ToLowerInvariant()
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'fixture_manifest.json') -Encoding utf8
Write-Host "Colibri fixture ready: $output"
