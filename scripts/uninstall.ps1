[CmdletBinding()]
param(
    [Alias("purge-state")][switch]$PurgeState,
    [switch]$Yes,
    [switch]$Json,
    [Alias("uv-path")][string]$UvPath
)

$ErrorActionPreference = "Stop"

function Normalize-ProcessPath {
    $environment = [Environment]::GetEnvironmentVariables([EnvironmentVariableTarget]::Process)
    $parts = @()
    foreach ($key in @("Path", "PATH")) {
        $value = [string]$environment[$key]
        if (-not [string]::IsNullOrWhiteSpace($value)) { $parts += $value }
    }
    [Environment]::SetEnvironmentVariable("PATH", $null, [EnvironmentVariableTarget]::Process)
    [Environment]::SetEnvironmentVariable("Path", $null, [EnvironmentVariableTarget]::Process)
    [Environment]::SetEnvironmentVariable("Path", ($parts -join ";"), [EnvironmentVariableTarget]::Process)
}

function ConvertTo-NativeArgument {
    param([AllowEmptyString()][string]$Argument)
    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') { return $Argument }
    $builder = [System.Text.StringBuilder]::new()
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ($character -eq '\') { $backslashes += 1; continue }
        if ($character -eq '"') {
            for ($index = 0; $index -lt (2 * $backslashes + 1); $index++) { [void]$builder.Append('\') }
            [void]$builder.Append('"')
        }
        else {
            for ($index = 0; $index -lt $backslashes; $index++) { [void]$builder.Append('\') }
            [void]$builder.Append($character)
        }
        $backslashes = 0
    }
    for ($index = 0; $index -lt (2 * $backslashes); $index++) { [void]$builder.Append('\') }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Invoke-BoundedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [int]$TimeoutSeconds = 300,
        [switch]$AllowFailure
    )
    $identifier = [Guid]::NewGuid().ToString("N")
    $stdoutPath = Join-Path ([System.IO.Path]::GetTempPath()) "swarm-uninstall-$identifier.out"
    $stderrPath = Join-Path ([System.IO.Path]::GetTempPath()) "swarm-uninstall-$identifier.err"
    try {
        $argumentLine = (($Arguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join " ")
        $process = Start-Process -FilePath $FilePath -ArgumentList $argumentLine -NoNewWindow `
            -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        [void]$process.Handle
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            try { $process.Kill() } catch { }
            throw "bounded command timed out after $TimeoutSeconds seconds: $FilePath"
        }
        $process.WaitForExit()
        $process.Refresh()
        $exitCode = $process.ExitCode
        $stderr = if (Test-Path $stderrPath) { Get-Content $stderrPath -Raw } else { "" }
        if ($exitCode -ne 0 -and -not $AllowFailure) {
            throw "command failed ($exitCode): $FilePath $stderr"
        }
        return $exitCode
    }
    finally {
        Remove-Item -LiteralPath $stdoutPath -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue
    }
}

Normalize-ProcessPath

try {
    $swarm = Get-Command swarm -ErrorAction SilentlyContinue
    if ($swarm) {
        Invoke-BoundedProcess -FilePath $swarm.Source -Arguments @(
            "node", "uninstall-service", "--yes", "--json"
        ) -TimeoutSeconds 180 -AllowFailure | Out-Null
    }
    $uv = if ($UvPath) { (Resolve-Path -LiteralPath $UvPath).Path } else { (Get-Command uv -ErrorAction Stop).Source }
    Invoke-BoundedProcess -FilePath $uv -Arguments @(
        "tool", "uninstall", "swarm-inference-lab"
    ) -TimeoutSeconds 300 | Out-Null
    $state = Join-Path $env:LOCALAPPDATA "SwarmInference"
    $purged = $false
    if ($PurgeState) {
        if (-not $Yes) { throw "--purge-state requires --yes because identities and membership are removed" }
        $resolved = [System.IO.Path]::GetFullPath($state)
        $expected = [System.IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "SwarmInference"))
        if ($resolved -ne $expected) { throw "refusing to purge an unexpected state path" }
        if (Test-Path -LiteralPath $resolved) { Remove-Item -LiteralPath $resolved -Recurse -Force }
        $purged = $true
    }
    $result = [ordered]@{ schema_version = 1; status = "PASS"; state_purged = $purged }
    if ($Json) { $result | ConvertTo-Json -Compress } else { Write-Output "status=PASS`nstate_purged=$purged" }
    exit 0
}
catch {
    $result = [ordered]@{ schema_version = 1; status = "FAIL"; detail = $_.Exception.Message }
    if ($Json) { $result | ConvertTo-Json -Compress } else { Write-Error $_.Exception.Message }
    exit 1
}
