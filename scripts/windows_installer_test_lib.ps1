Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-Condition {
    param([Parameter(Mandatory)][bool]$Condition, [Parameter(Mandatory)][string]$Message)
    if (-not $Condition) { throw "ACCEPTANCE ASSERTION FAILED: $Message" }
}

function ConvertTo-NativeArgument {
    param([Parameter(Mandatory)][AllowEmptyString()][string]$Value)
    if ($Value -notmatch '[\s"]') { return $Value }
    $builder = [Text.StringBuilder]::new()
    [void]$builder.Append('"')
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') {
            $slashes++
        }
        elseif ($character -eq '"') {
            [void]$builder.Append(('\' * (($slashes * 2) + 1)))
            [void]$builder.Append('"')
            $slashes = 0
        }
        else {
            [void]$builder.Append(('\' * $slashes))
            $slashes = 0
            [void]$builder.Append($character)
        }
    }
    [void]$builder.Append(('\' * ($slashes * 2)))
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Invoke-BoundedProcess {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter()][string[]]$ArgumentList = @(),
        [Parameter()][int]$TimeoutSeconds = 1800,
        [Parameter()][int[]]$ExpectedExitCodes = @(0)
    )
    Assert-Condition ($TimeoutSeconds -gt 0 -and $TimeoutSeconds -le 3600) 'process timeout is bounded'
    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $FilePath
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    if ($null -ne $start.PSObject.Properties['ArgumentList']) {
        foreach ($argument in $ArgumentList) { [void]$start.ArgumentList.Add($argument) }
    }
    else {
        $start.Arguments = ($ArgumentList | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' '
    }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $start
    $stopwatch = [Diagnostics.Stopwatch]::StartNew()
    if (-not $process.Start()) { throw "could not start $FilePath" }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        try { $process.Kill($true) } catch {
            & "$env:SystemRoot\System32\taskkill.exe" /PID $process.Id /T /F | Out-Null
        }
        throw "process timed out after $TimeoutSeconds seconds: $FilePath"
    }
    $process.WaitForExit()
    $stopwatch.Stop()
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    $result = [ordered]@{
        file = $FilePath
        arguments = @($ArgumentList)
        exit_code = $process.ExitCode
        duration_ms = $stopwatch.ElapsedMilliseconds
        stdout = if ($stdout.Length -gt 12000) { $stdout.Substring($stdout.Length - 12000) } else { $stdout }
        stderr = if ($stderr.Length -gt 12000) { $stderr.Substring($stderr.Length - 12000) } else { $stderr }
    }
    if ($ExpectedExitCodes -notcontains $process.ExitCode) {
        throw "unexpected exit code $($process.ExitCode) from $FilePath`n$($result.stderr)`n$($result.stdout)"
    }
    return [pscustomobject]$result
}

function New-InstallerTestContext {
    param([Parameter(Mandatory)][string]$Label)
    # Keep this deliberately short: native extension loading still encounters
    # Windows path limits even when setup itself is long-path aware.
    $base = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { Join-Path $env:TEMP 'swarm-installer-tests' }
    $profile = Join-Path $base ('p-' + [guid]::NewGuid().ToString('N').Substring(0, 12))
    $local = Join-Path $profile 'LocalAppData'
    $roaming = Join-Path $profile 'AppData'
    $user = Join-Path $profile 'UserProfile'
    $temp = Join-Path $profile 'Temp'
    foreach ($directory in @($local, $roaming, $user, $temp)) {
        [void](New-Item -ItemType Directory -Force -Path $directory)
    }
    return [pscustomobject]@{
        Label = $Label
        Profile = $profile
        LocalAppData = $local
        AppData = $roaming
        UserProfile = $user
        Temp = $temp
        InstallRoot = Join-Path $local 'Programs\SwarmInference'
        StateRoot = Join-Path $local 'SwarmInference'
        Evidence = Join-Path $profile 'evidence.json'
    }
}

function Enter-InstallerTestContext {
    param([Parameter(Mandatory)]$Context)
    $names = @('LOCALAPPDATA', 'APPDATA', 'USERPROFILE', 'TEMP', 'TMP', 'PATH', 'SWARM_STATE_ROOT')
    $saved = @{}
    foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    $system32 = Join-Path $env:SystemRoot 'System32'
    $cleanPath = @(
        $system32,
        $env:SystemRoot,
        (Join-Path $system32 'Wbem'),
        (Join-Path $system32 'WindowsPowerShell\v1.0')
    ) -join ';'
    $env:LOCALAPPDATA = $Context.LocalAppData
    $env:APPDATA = $Context.AppData
    $env:USERPROFILE = $Context.UserProfile
    $env:TEMP = $Context.Temp
    $env:TMP = $Context.Temp
    $env:PATH = $cleanPath
    $env:SWARM_STATE_ROOT = $Context.StateRoot
    return [pscustomobject]@{ Saved = $saved; CleanPath = $cleanPath }
}

function Exit-InstallerTestContext {
    param([Parameter(Mandatory)]$EnvironmentSnapshot)
    foreach ($entry in $EnvironmentSnapshot.Saved.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
}

function Assert-DeveloperCommandsAbsent {
    $where = Join-Path $env:SystemRoot 'System32\where.exe'
    foreach ($command in @('python.exe', 'uv.exe', 'git.exe')) {
        $probe = Invoke-BoundedProcess -FilePath $where -ArgumentList @($command) -TimeoutSeconds 15 -ExpectedExitCodes @(1)
        Assert-Condition ($probe.exit_code -eq 1) "$command must be unavailable on the sanitised PATH"
    }
}

function Invoke-SetupExecutable {
    param(
        [Parameter(Mandatory)][string]$SetupPath,
        [Parameter(Mandatory)]$Context,
        [Parameter()][string]$Backend = 'cpu',
        [Parameter()][string[]]$AdditionalArguments = @(),
        [Parameter()][int[]]$ExpectedExitCodes = @(0),
        [Parameter()][string]$LogName = 'setup.log'
    )
    $arguments = @(
        '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
        "/BACKEND=$Backend", "/PURGESTATE=0", "/DIR=$($Context.InstallRoot)",
        "/LOG=$(Join-Path $Context.Profile $LogName)"
    ) + $AdditionalArguments
    $result = Invoke-BoundedProcess -FilePath $SetupPath -ArgumentList $arguments -TimeoutSeconds 2400 -ExpectedExitCodes $ExpectedExitCodes
    if ($result.exit_code -eq 0) {
        $Context | Add-Member -NotePropertyName ActiveSetupPath -NotePropertyValue $SetupPath -Force
    }
    return $result
}

function Invoke-UninstallExecutable {
    param(
        [Parameter(Mandatory)]$Context,
        [Parameter()][bool]$PurgeState = $false,
        [Parameter()][string]$LogName = 'uninstall.log'
    )
    $uninstaller = Join-Path $Context.InstallRoot 'unins000.exe'
    Assert-Condition (Test-Path -LiteralPath $uninstaller -PathType Leaf) 'Inno uninstaller exists'
    $arguments = @(
        '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
        "/PURGESTATE=$([int]$PurgeState)", "/LOG=$(Join-Path $Context.Profile $LogName)"
    )
    return Invoke-BoundedProcess -FilePath $uninstaller -ArgumentList $arguments -TimeoutSeconds 600
}

function Wait-PathAbsent {
    param([Parameter(Mandatory)][string]$Path, [int]$TimeoutSeconds = 30)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ((Test-Path -LiteralPath $Path) -and [DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 100
    }
    Assert-Condition (-not (Test-Path -LiteralPath $Path)) "path was removed within $TimeoutSeconds seconds: $Path"
}

function Get-InstallRecord {
    param([Parameter(Mandatory)]$Context)
    $path = Join-Path $Context.InstallRoot 'app\install-record.json'
    Assert-Condition (Test-Path -LiteralPath $path -PathType Leaf) 'installation record exists'
    return Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
}

function Get-SwarmUninstallEntry {
    param([Parameter(Mandatory)]$Context)
    $keys = Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall' -ErrorAction SilentlyContinue
    return @($keys | ForEach-Object { Get-ItemProperty $_.PSPath } | Where-Object {
        $_.DisplayName -like 'Swarm Inference*' -and
        [string]::Equals([string]$_.InstallLocation.TrimEnd('\'), [string]$Context.InstallRoot.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)
    })
}

function Get-OwnedPathCount {
    param([Parameter(Mandatory)]$Context)
    $owned = [IO.Path]::GetFullPath((Join-Path $Context.InstallRoot 'runtime\Scripts')).TrimEnd('\')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $userPath) { return 0 }
    return @($userPath.Split(';', [StringSplitOptions]::RemoveEmptyEntries) | Where-Object {
        try { [IO.Path]::GetFullPath($_.Trim()).TrimEnd('\') -ieq $owned } catch { $false }
    }).Count
}

function Assert-NoOwnedService {
    $query = Invoke-BoundedProcess -FilePath (Join-Path $env:SystemRoot 'System32\schtasks.exe') -ArgumentList @('/Query', '/FO', 'CSV', '/NH') -TimeoutSeconds 30
    Assert-Condition ($query.Stdout -notmatch '\\SwarmInference-') 'installer must not create a cluster service before create/join'
}

function Assert-InstalledRuntime {
    param(
        [Parameter(Mandatory)]$Context,
        [Parameter(Mandatory)][string]$ExpectedVersion,
        [Parameter()][string]$ExpectedOperation = ''
    )
    $record = Get-InstallRecord $Context
    Assert-Condition ($record.product_version -eq $ExpectedVersion) "installed version is $ExpectedVersion"
    Assert-Condition ($record.installer_version -eq $ExpectedVersion) 'installer record uses the canonical package version'
    Assert-Condition ($record.installation_mode -eq 'native-windows') 'installation mode is native-windows'
    Assert-Condition ($record.selected_backend -eq 'cpu') 'CPU acceptance selected the CPU profile'
    Assert-Condition ([IO.Path]::GetFullPath($record.application_path) -eq [IO.Path]::GetFullPath($Context.InstallRoot)) 'record application path matches isolated root'
    Assert-Condition ([IO.Path]::GetFullPath($record.state_path) -eq [IO.Path]::GetFullPath($Context.StateRoot)) 'record state path is separate and correct'
    Assert-Condition ($null -ne $Context.PSObject.Properties['ActiveSetupPath']) 'successful setup path is tracked for release evidence'
    $releaseManifest = Join-Path (Split-Path -Parent $Context.ActiveSetupPath) 'release-manifest.json'
    Assert-Condition (Test-Path -LiteralPath $releaseManifest -PathType Leaf) 'external release manifest is available beside the accepted setup'
    $expectedManifestHash = "sha256:$((Get-FileHash -LiteralPath $releaseManifest -Algorithm SHA256).Hash.ToLowerInvariant())"
    Assert-Condition ($record.release_manifest_sha256 -eq $expectedManifestHash) 'installation record identifies the external release manifest'
    if ($ExpectedOperation) { Assert-Condition ($record.installation_operation -eq $ExpectedOperation) "installation operation is $ExpectedOperation" }
    $scripts = Join-Path $Context.InstallRoot 'runtime\Scripts'
    $env:PATH = "$scripts;$($script:InstallerCleanPath)"
    $version = Invoke-BoundedProcess -FilePath 'swarm.exe' -ArgumentList @('--version') -TimeoutSeconds 60
    Assert-Condition ($version.Stdout -match [regex]::Escape($ExpectedVersion)) 'swarm --version uses the installed runtime'
    $doctor = Invoke-BoundedProcess -FilePath 'swarm.exe' -ArgumentList @('node', 'doctor', '--json') -TimeoutSeconds 180
    $doctorJson = $doctor.Stdout | ConvertFrom-Json
    Assert-Condition ($doctorJson.status -eq 'pass') 'installed swarm node doctor passes'
    Assert-Condition ($doctorJson.backend_selection.selected_backend -eq 'torch-cpu') 'installed doctor selects torch-cpu'
    $python = Join-Path $scripts 'python.exe'
    $importCode = 'import json,pathlib,swarm_inference; print(json.dumps({"module":str(pathlib.Path(swarm_inference.__file__).resolve()),"version":swarm_inference.__version__}))'
    $import = Invoke-BoundedProcess -FilePath $python -ArgumentList @('-c', $importCode) -TimeoutSeconds 60
    $importJson = $import.Stdout | ConvertFrom-Json
    Assert-Condition ($importJson.version -eq $ExpectedVersion) 'wheel metadata and runtime version agree'
    Assert-Condition ($importJson.module -like "$($Context.InstallRoot)*") 'module imports only from installed runtime'
    Assert-Condition ($importJson.module -notlike "$(Resolve-Path .)*") 'module does not import from source checkout'
    Assert-Condition ((Get-OwnedPathCount $Context) -eq 1) 'user PATH has exactly one owned entry'
    $entry = @(Get-SwarmUninstallEntry $Context)
    Assert-Condition ($entry.Count -eq 1) 'Apps & Features has exactly one matching entry'
    Assert-Condition ($entry[0].DisplayVersion -eq $ExpectedVersion) 'Apps & Features version matches package version'
    Assert-NoOwnedService
    return [pscustomobject]@{ record = $record; doctor = $doctorJson; import = $importJson }
}

function Write-AcceptanceEvidence {
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)]$Document)
    $Document | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Path -Encoding utf8
}
