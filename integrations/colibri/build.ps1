[CmdletBinding()]
param(
    [string]$ColibriPath = (Join-Path $PSScriptRoot '..\..\third_party\colibri'),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\..\build\colibri'),
    [string]$MakePath = 'make.exe',
    [string]$PythonPath = 'python.exe',
    [switch]$ApplyBridgePatches,
    [switch]$BuildCuda,
    [string]$CudaArchitecture = 'auto',
    [string]$VcVarsPath = '',
    [string]$MsvcToolsetVersion = '',
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

    # The canonical SWARMEX1 implementation belongs to this repository rather
    # than the downstream Colibri patch series. Copy the exact audited adapter
    # into the exported tree after every patch has applied and before compile.
    foreach ($adapterName in @('swarm_expert_wire.h', 'swarm_expert_wire.c')) {
        $adapterSource = Join-Path $PSScriptRoot "adapter\$adapterName"
        if (-not (Test-Path -LiteralPath $adapterSource -PathType Leaf)) {
            throw "missing canonical C wire adapter: $adapterSource"
        }
        Copy-Item -LiteralPath $adapterSource -Destination (Join-Path $source "c\$adapterName") -Force
    }
}

$make = (Get-Command $MakePath -ErrorAction Stop).Source
$toolchainDirectory = Split-Path -Parent $make
# Portable MinGW distributions (for example w64devkit) resolve gcc, sed, and
# related tools relative to make.exe.  Put that directory first for this build
# process rather than requiring callers to mutate their persistent PATH.
$previousPath = $env:PATH
$gitUnixTools = 'C:\Program Files\Git\usr\bin'
$buildPathParts = @($toolchainDirectory)
if (Test-Path -LiteralPath (Join-Path $gitUnixTools 'sed.exe') -PathType Leaf) {
    $buildPathParts += $gitUnixTools
}
$buildPathParts += $previousPath
$env:PATH = $buildPathParts -join ';'
$targetNames = @('colibri.exe', 'olmoe.exe', 'olmoe_expert_worker.exe', 'inkling.exe', 'kimi_k3.exe')
$adapterBinaryNames = @('coli_kimi_mxfp4.dll')
$nativeTestTargets = if ($ApplyBridgePatches) {
    @(
        'tests/test_olmoe_expert_runtime.exe',
          'tests/test_olmoe_external_dispatch.exe',
          'tests/test_olmoe_memory_residency.exe',
          'tests/test_olmoe_expert_shm.exe'
    )
} else { @() }
$allBuildTargets = @($targetNames) + @($nativeTestTargets)
$cudaBuild = $null
if ($BuildCuda) {
    if (-not $IsWindows -and $env:OS -ne 'Windows_NT') {
        throw '-BuildCuda currently implements the required native Windows DLL path only'
    }
    $computeCapability = (& nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader,nounits).Trim()
    if ($LASTEXITCODE -ne 0 -or $computeCapability -notmatch '^\d+\.\d+$') {
        throw 'unable to detect NVIDIA GPU compute capability; refusing to assume an architecture'
    }
    $detectedArchitecture = 'sm_' + $computeCapability.Replace('.', '')
    $selectedArchitecture = if ($CudaArchitecture -eq 'auto') { $detectedArchitecture } else { $CudaArchitecture }
    if ($selectedArchitecture -notmatch '^sm_\d+$') {
        throw "invalid CUDA architecture: $selectedArchitecture"
    }
    if ($selectedArchitecture -ne $detectedArchitecture) {
        throw "requested CUDA architecture $selectedArchitecture does not match detected $detectedArchitecture"
    }
    $nvcc = (Get-Command nvcc.exe -ErrorAction Stop).Source
    $selectedVcVars = $VcVarsPath
    if (-not $selectedVcVars -and -not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        $vcVarsCandidates = @(
            'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
            'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat',
            'C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat'
        )
        $selectedVcVars = $vcVarsCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    }
    if ($selectedVcVars) {
        $selectedVcVars = (Resolve-Path -LiteralPath $selectedVcVars).Path
    }
    if (-not $MsvcToolsetVersion -and $selectedVcVars -like '*\18\Insiders\*') {
        # CUDA 13.0 accepts the 19.44 compiler bundled side-by-side in VS 18,
        # while the newer preview compiler intentionally trips nvcc's version gate.
        $MsvcToolsetVersion = '14.44'
    }
    $previousCudaArchitecture = $env:CUDA_ARCH
    $previousCudaPath = $env:CUDA_PATH
    $env:CUDA_ARCH = $selectedArchitecture
    if (-not $env:CUDA_PATH) {
        $env:CUDA_PATH = Split-Path -Parent (Split-Path -Parent $nvcc)
    }
    $cudaStarted = Get-Date
    try {
        $cudaBatch = Join-Path $source 'c\build_cuda.bat'
        if ($selectedVcVars) {
            $toolsetArgument = if ($MsvcToolsetVersion) { " -vcvars_ver=$MsvcToolsetVersion" } else { '' }
            $cudaCommand = "call `"$selectedVcVars`"$toolsetArgument >nul && set `"CUDA_ARCH=$selectedArchitecture`" && set `"CUDA_PATH=$env:CUDA_PATH`" && call `"$cudaBatch`""
            & cmd.exe /d /s /c $cudaCommand
        }
        else {
            & $cudaBatch
        }
        $cudaExit = $LASTEXITCODE
    }
    finally {
        $env:CUDA_ARCH = $previousCudaArchitecture
        $env:CUDA_PATH = $previousCudaPath
    }
    if ($cudaExit -ne 0) { throw "Colibri CUDA DLL build failed with exit code $cudaExit" }
    $cudaBuild = [ordered]@{
        schema_version = 'experiment-010-colibri-cuda-build-v1'
        requested = $true
        detected_compute_capability = $computeCapability
        detected_architecture = $detectedArchitecture
        selected_architecture = $selectedArchitecture
        nvcc = $nvcc
        vcvars = $selectedVcVars
        msvc_toolset_version = if ($MsvcToolsetVersion) { $MsvcToolsetVersion } else { $null }
        started_at = $cudaStarted.ToUniversalTime().ToString('o')
        completed_at = (Get-Date).ToUniversalTime().ToString('o')
        exit_code = $cudaExit
    }
}
try {
    if ($BuildCuda) {
        # CUDA_DLL defines COLI_CUDA for hosts using backend_loader.c. Inkling
        # has a distinct CUDA ABI, so applying that flag to every family creates
        # unresolved ink_cuda_* references. Build unrelated family hosts and
        # the portable native tests with their truthful CPU configuration, then
        # relink the generic and OLMoE hosts against the runtime loader. The
        # latter is required for real native-int8 expert execution in both the
        # coordinator and isolated C worker.
        $cpuTargets = @('inkling.exe', 'kimi_k3.exe') + @($nativeTestTargets)
        & $make -C (Join-Path $source 'c') -j4 @cpuTargets 'ARCH=native' 'CUDA_DLL=0'
        $makeExit = $LASTEXITCODE
        if ($makeExit -eq 0) {
            & $make -C (Join-Path $source 'c') -j4 @('colibri.exe', 'olmoe.exe', 'olmoe_expert_worker.exe') 'ARCH=native' 'CUDA_DLL=1'
            $makeExit = $LASTEXITCODE
        }
    }
    else {
        & $make -C (Join-Path $source 'c') -j4 @allBuildTargets 'ARCH=native'
        $makeExit = $LASTEXITCODE
    }
    if ($makeExit -eq 0) {
        $gcc = (Get-Command gcc.exe -ErrorAction Stop).Source
        $kimiAdapter = Join-Path $PSScriptRoot 'adapter\kimi_mxfp4_runtime.c'
        if (-not (Test-Path -LiteralPath $kimiAdapter -PathType Leaf)) {
            throw "missing native Kimi MXFP4 adapter: $kimiAdapter"
        }
        $kimiCompileArguments = @(
            '-D_FILE_OFFSET_BITS=64', '-O3', '-march=native', '-fopenmp',
            '-Wall', '-Wextra', '-Wno-unused-parameter',
            '-Wno-misleading-indentation', '-Wno-unused-function', '-shared',
            '-static', '-I', (Join-Path $source 'c'), $kimiAdapter,
            '-o', (Join-Path $source 'c\coli_kimi_mxfp4.dll'),
            '-lm', '-fopenmp', '-static', '-lpsapi'
        )
        & $gcc @kimiCompileArguments
        $makeExit = $LASTEXITCODE
    }
}
finally { $env:PATH = $previousPath }
if ($makeExit -ne 0) { throw "Colibri build failed with exit code $makeExit" }

foreach ($name in @($targetNames) + @($adapterBinaryNames)) {
    $built = Join-Path $source "c\$name"
    if (-not (Test-Path -LiteralPath $built -PathType Leaf)) { throw "missing built binary: $built" }
    Copy-Item -LiteralPath $built -Destination (Join-Path $binaryDirectory $name) -Force
}
if ($BuildCuda) {
    $builtCudaDll = Join-Path $source 'c\coli_cuda.dll'
    if (-not (Test-Path -LiteralPath $builtCudaDll -PathType Leaf)) {
        throw "missing built CUDA runtime: $builtCudaDll"
    }
    $installedCudaDll = Join-Path $binaryDirectory 'coli_cuda.dll'
    Copy-Item -LiteralPath $builtCudaDll -Destination $installedCudaDll -Force
    Copy-Item -LiteralPath (Join-Path $source 'c\.build-config') -Destination (Join-Path $binaryDirectory '.build-config') -Force
    $cudaBuild.cuda_dll = $installedCudaDll
    $cudaBuild.cuda_dll_sha256 = (Get-FileHash -LiteralPath $installedCudaDll -Algorithm SHA256).Hash.ToLowerInvariant()
    $cudaBuild | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'colibri_cuda_build.json') -Encoding utf8
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
    schema_version = 'experiment-010-correction-colibri-patches-v1'
    upstream_commit = $actualCommit
    bridge_enabled = [bool]$ApplyBridgePatches
    patches = $patchRows
    wire_adapter = if ($ApplyBridgePatches) {
        @('swarm_expert_wire.h', 'swarm_expert_wire.c', 'kimi_mxfp4_runtime.c') | ForEach-Object {
            $adapter = Join-Path $PSScriptRoot "adapter\$_"
            [ordered]@{
                name = $_
                sha256 = (Get-FileHash -LiteralPath $adapter -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        }
    } else { @() }
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
$env:CC = (Get-Command gcc.exe -ErrorAction Stop).Source
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
        foreach ($target in $nativeTestTargets) {
            $nativeTest = Join-Path (Join-Path $source 'c') $target
            & $nativeTest
            if ($LASTEXITCODE -ne 0) {
                throw "native Colibri test failed: $target (exit code $LASTEXITCODE)"
            }
        }
        & $PythonPath -m pytest -q (Join-Path $repositoryRoot 'tests\unit\test_colibri_integration.py') --basetemp $baseTemp
        if ($LASTEXITCODE -ne 0) { throw "bridge-specific tests failed with exit code $LASTEXITCODE" }
    }
    finally {
        $env:PYTHONPATH = $previousPythonPath
        $env:TEMP = $previousTemp
        $env:TMP = $previousTmp
    }
}

if ($BuildCuda) {
    $proofPath = Join-Path $output 'colibri_cuda_kernel_proof.json'
    $previousPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = if ($previousPythonPath) { "$(Join-Path $repositoryRoot 'src');$previousPythonPath" } else { Join-Path $repositoryRoot 'src' }
    try {
        & $PythonPath -m swarm_inference.backends.colibri.cuda --dll (Join-Path $binaryDirectory 'coli_cuda.dll') --output $proofPath
        $proofExit = $LASTEXITCODE
    }
    finally { $env:PYTHONPATH = $previousPythonPath }
    if ($proofExit -ne 0) { throw "Colibri CUDA kernel proof failed with exit code $proofExit" }
}

Write-Host "Colibri build complete: $output"
Get-Content -LiteralPath (Join-Path $output 'colibri_build.json')
