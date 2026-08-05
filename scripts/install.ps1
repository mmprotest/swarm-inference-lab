[CmdletBinding()]
param(
    [Alias("source-wheel")]
    [string]$SourceWheel,
    [Alias("install-service")]
    [switch]$InstallService,
    [switch]$Json,
    [Alias("uv-path")]
    [string]$UvPath,
    [Alias("python-version")]
    [ValidateSet("3.11", "3.12", "3.13")]
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:UV_HTTP_TIMEOUT = "120"
$env:UV_CONCURRENT_DOWNLOADS = "8"

function Normalize-ProcessPath {
    # Windows environment blocks can legally contain both PATH and Path. Windows
    # PowerShell's Start-Process treats them as a duplicate dictionary key and
    # refuses to start. Preserve both value sets under one canonical key.
    $environment = [Environment]::GetEnvironmentVariables([EnvironmentVariableTarget]::Process)
    $parts = @()
    foreach ($key in @("Path", "PATH")) {
        $value = [string]$environment[$key]
        if (-not [string]::IsNullOrWhiteSpace($value)) { $parts += $value }
    }
    $combined = ($parts -join ";")
    [Environment]::SetEnvironmentVariable("PATH", $null, [EnvironmentVariableTarget]::Process)
    [Environment]::SetEnvironmentVariable("Path", $null, [EnvironmentVariableTarget]::Process)
    [Environment]::SetEnvironmentVariable("Path", $combined, [EnvironmentVariableTarget]::Process)
}

Normalize-ProcessPath

function ConvertTo-NativeArgument {
    param([AllowEmptyString()][string]$Argument)
    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') { return $Argument }
    $builder = [System.Text.StringBuilder]::new()
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes += 1
            continue
        }
        if ($character -eq '"') {
            for ($index = 0; $index -lt (2 * $backslashes + 1); $index++) {
                [void]$builder.Append('\')
            }
            [void]$builder.Append('"')
        }
        else {
            for ($index = 0; $index -lt $backslashes; $index++) {
                [void]$builder.Append('\')
            }
            [void]$builder.Append($character)
        }
        $backslashes = 0
    }
    for ($index = 0; $index -lt (2 * $backslashes); $index++) {
        [void]$builder.Append('\')
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Invoke-BoundedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [int]$TimeoutSeconds = 1800,
        [switch]$AllowFailure
    )
    $temporary = [System.IO.Path]::GetTempPath()
    $identifier = [Guid]::NewGuid().ToString("N")
    $stdoutPath = Join-Path $temporary "swarm-install-$identifier.out"
    $stderrPath = Join-Path $temporary "swarm-install-$identifier.err"
    try {
        $argumentLine = (($Arguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join " ")
        $process = Start-Process -FilePath $FilePath -ArgumentList $argumentLine `
            -NoNewWindow -PassThru -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath
        # Materialize the native handle before a very short process can exit;
        # otherwise Windows PowerShell 5 can expose a null ExitCode.
        [void]$process.Handle
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            try { $process.Kill() } catch { }
            throw "bounded command timed out after $TimeoutSeconds seconds: $FilePath"
        }
        # Complete redirected-stream bookkeeping before reading ExitCode. The
        # timeout overload alone can leave ExitCode unset in Windows PowerShell 5.
        $process.WaitForExit()
        $process.Refresh()
        $exitCode = $process.ExitCode
        $stdout = if (Test-Path $stdoutPath) { Get-Content $stdoutPath -Raw } else { "" }
        $stderr = if (Test-Path $stderrPath) { Get-Content $stderrPath -Raw } else { "" }
        if ($exitCode -ne 0 -and -not $AllowFailure) {
            throw "command failed ($exitCode): $FilePath $stderr"
        }
        return [pscustomobject]@{
            ExitCode = $exitCode
            Stdout = $stdout
            Stderr = $stderr
        }
    }
    finally {
        Remove-Item -LiteralPath $stdoutPath -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue
    }
}

function Resolve-Uv {
    if ($UvPath) {
        $resolved = Resolve-Path -LiteralPath $UvPath -ErrorAction Stop
        return $resolved.Path
    }
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $installer = Join-Path ([System.IO.Path]::GetTempPath()) "swarm-uv-install-$([Guid]::NewGuid().ToString('N')).ps1"
    try {
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 120 `
            -Uri "https://astral.sh/uv/install.ps1" -OutFile $installer
        $result = Invoke-BoundedProcess -FilePath "powershell.exe" -Arguments @(
            "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", $installer
        ) -TimeoutSeconds 300
        $candidates = @(
            (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
            (Join-Path $env:LOCALAPPDATA "Programs\uv\uv.exe")
        )
        foreach ($candidate in $candidates) {
            if (Test-Path -LiteralPath $candidate) { return (Resolve-Path $candidate).Path }
        }
        throw "uv installer completed but uv.exe was not found: $($result.Stderr)"
    }
    finally {
        Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
    }
}

function Test-NvidiaCandidate {
    $command = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $command) { return $false }
    $probe = Invoke-BoundedProcess -FilePath $command.Source -Arguments @("--query-gpu=name", "--format=csv,noheader") -TimeoutSeconds 20 -AllowFailure
    return $probe.ExitCode -eq 0 -and -not [string]::IsNullOrWhiteSpace($probe.Stdout)
}

function Install-Tool {
    param([string]$Uv, [string]$Package, [string]$Extra)
    $arguments = @("tool", "install", "--force", "--python", $PythonVersion)
    if ($Extra -eq "cpu") { $arguments += @("--torch-backend", "cpu") }
    elseif ($Extra -eq "cuda") { $arguments += @("--torch-backend", "cu130") }
    else { $arguments += @("--torch-backend", "auto") }
    $arguments += "$Package[$Extra]"
    Invoke-BoundedProcess -FilePath $Uv -Arguments $arguments -TimeoutSeconds 1800 | Out-Null
}

try {
    if (-not [Environment]::Is64BitOperatingSystem) {
        throw "Swarm Inference requires a supported 64-bit operating system"
    }
    $architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    if ($architecture -ne "X64") {
        throw "unsupported Windows architecture: $architecture"
    }
    $uv = Resolve-Uv
    Invoke-BoundedProcess -FilePath $uv -Arguments @("python", "install", $PythonVersion) -TimeoutSeconds 900 | Out-Null
    if ($SourceWheel) {
        $wheel = (Resolve-Path -LiteralPath $SourceWheel -ErrorAction Stop).Path
        if (-not $wheel.EndsWith(".whl", [StringComparison]::OrdinalIgnoreCase)) {
            throw "--source-wheel must name a built wheel"
        }
        $package = $wheel
    }
    else {
        $package = "swarm-inference-lab"
    }
    $candidateExtra = if (Test-NvidiaCandidate) { "cuda" } else { "cpu" }
    Install-Tool -Uv $uv -Package $package -Extra $candidateExtra
    $binResult = Invoke-BoundedProcess -FilePath $uv -Arguments @("tool", "dir", "--bin") -TimeoutSeconds 30
    $binDirectory = $binResult.Stdout.Trim()
    $swarm = Join-Path $binDirectory "swarm.exe"
    if (-not (Test-Path -LiteralPath $swarm)) { throw "installed swarm executable was not found" }
    $doctor = Invoke-BoundedProcess -FilePath $swarm -Arguments @("node", "doctor", "--json") -TimeoutSeconds 180
    $doctorDocument = $doctor.Stdout | ConvertFrom-Json
    $selected = [string]$doctorDocument.backend_selection.selected_backend
    $selectedExtra = if ($selected -eq "torch-cuda") { "cuda" } else { "cpu" }
    if ($selectedExtra -ne $candidateExtra) {
        Install-Tool -Uv $uv -Package $package -Extra $selectedExtra
        $doctor = Invoke-BoundedProcess -FilePath $swarm -Arguments @("node", "doctor", "--json") -TimeoutSeconds 180
        $doctorDocument = $doctor.Stdout | ConvertFrom-Json
        $selected = [string]$doctorDocument.backend_selection.selected_backend
    }
    $serviceStatus = "deferred-until-cluster-join"
    if ($InstallService) {
        $service = Invoke-BoundedProcess -FilePath $swarm -Arguments @("node", "install-service", "--yes", "--json") -TimeoutSeconds 180 -AllowFailure
        if ($service.ExitCode -ne 0) { throw "service installation failed: $($service.Stderr)" }
        $serviceStatus = "installed"
    }
    $diagnostics = [ordered]@{
        schema_version = 1
        status = "PASS"
        operating_system = "windows"
        architecture = $architecture
        python_version = $PythonVersion
        uv = $uv
        source = if ($SourceWheel) { "source-wheel" } else { "package-index" }
        package_extra = $selectedExtra
        selected_backend = $selected
        swarm_executable = $swarm
        service = $serviceStatus
        doctor = $doctorDocument
    }
    if ($Json) { $diagnostics | ConvertTo-Json -Depth 20 -Compress }
    else {
        Write-Output "status=PASS"
        Write-Output "swarm_executable=$swarm"
        Write-Output "selected_backend=$selected"
        Write-Output "service=$serviceStatus"
    }
    exit 0
}
catch {
    $failure = [ordered]@{
        schema_version = 1
        status = "FAIL"
        stage = "installation"
        category = "execution"
        detail = $_.Exception.Message
        retry_safe = $true
    }
    if ($Json) { $failure | ConvertTo-Json -Compress }
    else { Write-Error $_.Exception.Message }
    exit 1
}
