[CmdletBinding()]
param(
    [string]$ColibriPath = (Join-Path $PSScriptRoot '..\..\third_party\colibri'),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\..\build\colibri'),
    [string]$MakePath = 'make.exe',
    [string]$PythonPath = 'python.exe',
    [switch]$ApplyBridgePatches,
    [switch]$SkipBridgeTests
)

$ErrorActionPreference = 'Stop'
$expectedCommit = 'b085b48888a88d9a1c00b151a9979774b72cdbfd'
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$colibri = (Resolve-Path -LiteralPath $ColibriPath).Path
$output = [IO.Path]::GetFullPath($OutputDirectory)
$source = Join-Path $output 'source'
$binaryDirectory = Join-Path $output 'bin'
$archive = Join-Path $output 'colibri-source.tar'

$safeDirectory = $colibri.Replace('\', '/')
$actualCommit = (& git -c "safe.directory=$safeDirectory" -C $colibri rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw "unable to read Colibri revision" }
if ($actualCommit -ne $expectedCommit) {
    throw "Colibri revision mismatch: expected $expectedCommit, found $actualCommit"
}
if (-not (Test-Path -LiteralPath (Join-Path $colibri 'LICENSE') -PathType Leaf)) {
    throw 'pinned Colibri checkout does not contain LICENSE'
}

New-Item -ItemType Directory -Force $output, $binaryDirectory | Out-Null
if (Test-Path -LiteralPath $source) {
    $resolvedSource = (Resolve-Path -LiteralPath $source).Path
    if (-not $resolvedSource.StartsWith($output, [StringComparison]::OrdinalIgnoreCase)) {
        throw "refusing to replace source directory outside build output: $resolvedSource"
    }
    Remove-Item -LiteralPath $resolvedSource -Recurse -Force
}
New-Item -ItemType Directory -Path $source | Out-Null

& git -c "safe.directory=$safeDirectory" -C $colibri archive --format=tar $expectedCommit -o $archive
if ($LASTEXITCODE -ne 0) { throw "git archive failed with exit code $LASTEXITCODE" }
& tar.exe -xf $archive -C $source
if ($LASTEXITCODE -ne 0) { throw "source extraction failed with exit code $LASTEXITCODE" }

$patchRows = @()
if ($ApplyBridgePatches) {
    $series = Join-Path $PSScriptRoot 'patches\series'
    # The build tree normally lives below this repository and is ignored by
    # its parent Git worktree.  Without a ceiling, `git apply` discovers the
    # parent repository and silently skips every ignored build-tree path.
    $previousGitCeiling = $env:GIT_CEILING_DIRECTORIES
    $env:GIT_CEILING_DIRECTORIES = (Split-Path -Parent $source)
    try {
        foreach ($entry in Get-Content -LiteralPath $series) {
            $name = $entry.Trim()
            if (-not $name -or $name.StartsWith('#')) { continue }
            $patch = Join-Path $PSScriptRoot "patches\$name"
            if (-not (Test-Path -LiteralPath $patch -PathType Leaf)) {
                throw "patch listed in series is missing: $patch"
            }
            Push-Location $source
            try {
                & git apply --check $patch
                if ($LASTEXITCODE -ne 0) { throw "patch check failed for $name" }
                & git apply $patch
                if ($LASTEXITCODE -ne 0) { throw "patch application failed for $name" }
            }
            finally { Pop-Location }
            $patchRows += [ordered]@{
                name = $name
                sha256 = (Get-FileHash -LiteralPath $patch -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        }
    }
    finally { $env:GIT_CEILING_DIRECTORIES = $previousGitCeiling }
}

$make = (Get-Command $MakePath -ErrorAction Stop).Source
$toolchainDirectory = Split-Path -Parent $make
# Portable MinGW distributions (for example w64devkit) resolve gcc, sed, and
# related tools relative to make.exe.  Put that directory first for this build
# process rather than requiring callers to mutate their persistent PATH.
$previousPath = $env:PATH
$env:PATH = "$toolchainDirectory;$previousPath"
$targetNames = @('colibri.exe', 'olmoe.exe', 'inkling.exe', 'kimi_k3.exe')
try {
    & $make -C (Join-Path $source 'c') -j4 @targetNames 'ARCH=native'
    $makeExit = $LASTEXITCODE
}
finally { $env:PATH = $previousPath }
if ($makeExit -ne 0) { throw "Colibri build failed with exit code $makeExit" }

foreach ($name in $targetNames) {
    $built = Join-Path $source "c\$name"
    if (-not (Test-Path -LiteralPath $built -PathType Leaf)) { throw "missing built binary: $built" }
    Copy-Item -LiteralPath $built -Destination (Join-Path $binaryDirectory $name) -Force
}
Copy-Item -LiteralPath (Join-Path $source 'c\openai_server.py') -Destination $binaryDirectory -Force
if (Test-Path -LiteralPath (Join-Path $source 'c\swarm_bridge.py')) {
    Copy-Item -LiteralPath (Join-Path $source 'c\swarm_bridge.py') -Destination $binaryDirectory -Force
}
Copy-Item -LiteralPath (Join-Path $source 'c\coli') -Destination $binaryDirectory -Force
Copy-Item -LiteralPath (Join-Path $source 'c\resource_plan.py') -Destination $binaryDirectory -Force
Copy-Item -LiteralPath (Join-Path $source 'c\doctor.py') -Destination $binaryDirectory -Force
Copy-Item -LiteralPath (Join-Path $source 'c\autotune.py') -Destination $binaryDirectory -Force
Copy-Item -LiteralPath (Join-Path $source 'LICENSE') -Destination (Join-Path $output 'LICENSE.colibri') -Force

$patchManifest = [ordered]@{
    schema_version = 'experiment-009-colibri-patches-v1'
    upstream_commit = $actualCommit
    bridge_enabled = [bool]$ApplyBridgePatches
    patches = $patchRows
}
$patchManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'colibri_patch_manifest.json') -Encoding utf8

$manifestArguments = @(
    (Join-Path $PSScriptRoot 'build_manifest.py'),
    '--source', $source,
    '--bin', $binaryDirectory,
    '--output', (Join-Path $output 'colibri_build.json'),
    '--commit', $actualCommit
)
if ($patchRows.Count) {
    $manifestArguments += '--patches'
    $manifestArguments += @($patchRows | ForEach-Object { $_.name })
}
$previousManifestPath = $env:PATH
$previousCompiler = $env:CC
$env:PATH = "$toolchainDirectory;$previousManifestPath"
$env:CC = Join-Path $toolchainDirectory 'gcc.exe'
try {
    & $PythonPath @manifestArguments
    $manifestExit = $LASTEXITCODE
}
finally {
    $env:PATH = $previousManifestPath
    $env:CC = $previousCompiler
}
if ($manifestExit -ne 0) { throw "build fingerprint generation failed with exit code $manifestExit" }

if (-not $SkipBridgeTests) {
    $previousPythonPath = $env:PYTHONPATH
    $previousTemp = $env:TEMP
    $previousTmp = $env:TMP
    $testTemp = Join-Path $output 'test-temp'
    $baseTemp = Join-Path $testTemp 'pytest'
    New-Item -ItemType Directory -Force $testTemp, $baseTemp | Out-Null
    $env:PYTHONPATH = if ($previousPythonPath) { "$(Join-Path $repositoryRoot 'src');$previousPythonPath" } else { Join-Path $repositoryRoot 'src' }
    $env:TEMP = $testTemp
    $env:TMP = $testTemp
    try {
        & $PythonPath -m pytest -q (Join-Path $repositoryRoot 'tests\unit\test_colibri_integration.py') --basetemp $baseTemp
        if ($LASTEXITCODE -ne 0) { throw "bridge-specific tests failed with exit code $LASTEXITCODE" }
    }
    finally {
        $env:PYTHONPATH = $previousPythonPath
        $env:TEMP = $previousTemp
        $env:TMP = $previousTmp
    }
}

Write-Host "Colibri build complete: $output"
Get-Content -LiteralPath (Join-Path $output 'colibri_build.json')
