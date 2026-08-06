[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SetupA,
    [Parameter(Mandatory)][string]$SetupB,
    [Parameter(Mandatory)][string]$BrokenSetup,
    [string]$VersionA = '0.1.0rc2',
    [string]$VersionB = '0.1.0rc3',
    [string]$BrokenVersion = '0.1.0rc4',
    [string]$EvidencePath
)

. "$PSScriptRoot\windows_installer_test_lib.ps1"
$SetupA = (Resolve-Path -LiteralPath $SetupA).Path
$SetupB = (Resolve-Path -LiteralPath $SetupB).Path
$BrokenSetup = (Resolve-Path -LiteralPath $BrokenSetup).Path
$context = New-InstallerTestContext -Label 'upgrade-rollback'
$originalUserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$snapshot = $null
try {
    $snapshot = Enter-InstallerTestContext $context
    $script:InstallerCleanPath = $snapshot.CleanPath
    Assert-DeveloperCommandsAbsent

    $installA = Invoke-SetupExecutable -SetupPath $SetupA -Context $context -LogName 'version-a.log'
    $versionARecord = (Assert-InstalledRuntime -Context $context -ExpectedVersion $VersionA -ExpectedOperation 'install').record
    [void](New-Item -ItemType Directory -Force -Path $context.StateRoot)
    $marker = Join-Path $context.StateRoot 'durable-upgrade-marker.txt'
    Set-Content -LiteralPath $marker -Value ([guid]::NewGuid().ToString('N')) -Encoding utf8
    $markerHash = (Get-FileHash -LiteralPath $marker -Algorithm SHA256).Hash

    $upgradeB = Invoke-SetupExecutable -SetupPath $SetupB -Context $context -LogName 'version-b.log'
    $versionBRecord = (Assert-InstalledRuntime -Context $context -ExpectedVersion $VersionB -ExpectedOperation 'upgrade').record
    Assert-Condition ((Get-FileHash -LiteralPath $marker -Algorithm SHA256).Hash -eq $markerHash) 'durable state survives upgrade'
    Assert-Condition ($versionBRecord.previous_installed_version -eq $VersionA) 'upgrade record identifies previous version'
    $previousEntries = @(Get-ChildItem -LiteralPath (Join-Path $context.InstallRoot 'previous') -Force -ErrorAction SilentlyContinue)
    Assert-Condition ($previousEntries.Count -eq 0) 'previous runtime is removed only after successful upgrade'

    $downgradeRejected = Invoke-SetupExecutable -SetupPath $SetupA -Context $context -LogName 'downgrade-rejected.log' -ExpectedExitCodes @(1..255)
    Assert-Condition ($downgradeRejected.exit_code -ne 0) 'unintended downgrade is rejected'
    [void](Assert-InstalledRuntime -Context $context -ExpectedVersion $VersionB)
    Assert-Condition ((Get-FileHash -LiteralPath $marker -Algorithm SHA256).Hash -eq $markerHash) 'rejected downgrade preserves state'

    $downgradeAllowed = Invoke-SetupExecutable -SetupPath $SetupA -Context $context -LogName 'downgrade-allowed.log' -AdditionalArguments @('/ALLOWDOWNGRADE=1')
    [void](Assert-InstalledRuntime -Context $context -ExpectedVersion $VersionA -ExpectedOperation 'upgrade')
    Assert-Condition ((Get-FileHash -LiteralPath $marker -Algorithm SHA256).Hash -eq $markerHash) 'authorised downgrade preserves state'

    [void](Invoke-SetupExecutable -SetupPath $SetupB -Context $context -LogName 'version-b-before-rollback.log')
    [void](Assert-InstalledRuntime -Context $context -ExpectedVersion $VersionB -ExpectedOperation 'upgrade')
    $rollback = Invoke-SetupExecutable -SetupPath $BrokenSetup -Context $context -LogName 'broken-version.log' -ExpectedExitCodes @(1..255)
    Assert-Condition ($rollback.exit_code -ne 0) 'invalid upgrade returns failure'
    $restored = (Assert-InstalledRuntime -Context $context -ExpectedVersion $VersionB).record
    Assert-Condition ((Get-FileHash -LiteralPath $marker -Algorithm SHA256).Hash -eq $markerHash) 'rollback preserves durable state'
    Assert-Condition (@(Get-ChildItem -LiteralPath $context.InstallRoot -Filter '.runtime.*' -Force -ErrorAction SilentlyContinue).Count -eq 0) 'rollback leaves no failed staging runtime'
    Assert-NoOwnedService
    $rollbackSources = @(Get-ChildItem -LiteralPath $context.Profile -Filter 'broken-version.log*' -File -ErrorAction SilentlyContinue)
    Assert-Condition ($rollbackSources.Count -ge 2) 'setup and bootstrapper rollback diagnostics are retained'

    $target = if ($EvidencePath) { $EvidencePath } else { $context.Evidence }
    $rollbackDirectory = Join-Path (Split-Path -Parent $target) 'installer-rollback-logs'
    [void](New-Item -ItemType Directory -Force -Path $rollbackDirectory)
    $rollbackLogs = @($rollbackSources | ForEach-Object {
        $destination = Join-Path $rollbackDirectory $_.Name
        Copy-Item -LiteralPath $_.FullName -Destination $destination -Force
        $destination
    })

    [void](Invoke-UninstallExecutable -Context $context -PurgeState:$false -LogName 'upgrade-suite-uninstall.log')
    Assert-Condition (Test-Path -LiteralPath $marker -PathType Leaf) 'suite cleanup uninstall preserves state'
    $evidence = [ordered]@{
        schema_version = 1
        status = 'PASS'
        version_a = $VersionA
        version_b = $VersionB
        broken_version = $BrokenVersion
        install_a = $installA
        upgrade_b = $upgradeB
        downgrade_rejected = $downgradeRejected
        downgrade_allowed = $downgradeAllowed
        rollback_attempt = $rollback
        version_a_record = $versionARecord
        version_b_record = $versionBRecord
        restored_record = $restored
        state_marker_sha256 = "sha256:$($markerHash.ToLowerInvariant())"
        rollback_logs = $rollbackLogs
    }
    Write-AcceptanceEvidence -Path $target -Document $evidence
    Write-Output ($evidence | ConvertTo-Json -Depth 20 -Compress)
}
finally {
    [Environment]::SetEnvironmentVariable('Path', $originalUserPath, 'User')
    if ($null -ne $snapshot) { Exit-InstallerTestContext $snapshot }
}
