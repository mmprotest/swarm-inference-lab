[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SetupPath,
    [string]$ExpectedVersion = '0.1.0rc11',
    [string]$EvidencePath
)

. "$PSScriptRoot\windows_installer_test_lib.ps1"
$SetupPath = (Resolve-Path -LiteralPath $SetupPath).Path
$context = New-InstallerTestContext -Label 'uninstall-reinstall'
$originalUserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$snapshot = $null
try {
    $snapshot = Enter-InstallerTestContext $context
    $script:InstallerCleanPath = $snapshot.CleanPath
    Assert-DeveloperCommandsAbsent
    [void](Invoke-SetupExecutable -SetupPath $SetupPath -Context $context -LogName 'initial-install.log')
    [void](Assert-InstalledRuntime -Context $context -ExpectedVersion $ExpectedVersion -ExpectedOperation 'install')
    $markers = @(
        (Join-Path $context.StateRoot 'security\identity.acceptance'),
        (Join-Path $context.StateRoot 'models\cache.acceptance'),
        (Join-Path $context.StateRoot 'evidence\run.acceptance'),
        (Join-Path $context.StateRoot 'trust.acceptance')
    )
    foreach ($marker in $markers) {
        [void](New-Item -ItemType Directory -Force -Path (Split-Path -Parent $marker))
        Set-Content -LiteralPath $marker -Value 'durable-state' -Encoding utf8
    }
    [void](Invoke-UninstallExecutable -Context $context -PurgeState:$false -LogName 'preserve-uninstall.log')
    Wait-PathAbsent -Path $context.InstallRoot
    foreach ($marker in $markers) { Assert-Condition (Test-Path -LiteralPath $marker -PathType Leaf) "preserved $marker" }
    Assert-Condition ((Get-OwnedPathCount $context) -eq 0) 'owned PATH is removed on normal uninstall'

    [void](Invoke-SetupExecutable -SetupPath $SetupPath -Context $context -LogName 'reinstall.log')
    $reinstalled = (Assert-InstalledRuntime -Context $context -ExpectedVersion $ExpectedVersion -ExpectedOperation 'install').record
    foreach ($marker in $markers) { Assert-Condition (Test-Path -LiteralPath $marker -PathType Leaf) "reinstall retains $marker" }
    [void](Invoke-UninstallExecutable -Context $context -PurgeState:$true -LogName 'purge-uninstall.log')
    Wait-PathAbsent -Path $context.InstallRoot
    Assert-Condition (-not (Test-Path -LiteralPath $context.StateRoot)) 'explicit /PURGESTATE=1 removes durable state'
    Assert-Condition ((Get-OwnedPathCount $context) -eq 0) 'purge uninstall leaves no owned PATH entry'
    Assert-Condition (@(Get-SwarmUninstallEntry $context).Count -eq 0) 'purge uninstall removes Apps & Features entry'
    $evidence = [ordered]@{
        schema_version = 1
        status = 'PASS'
        version = $ExpectedVersion
        state_preserved_by_default = $true
        reinstall_retained_state = $true
        explicit_purge_removed_state = $true
        reinstalled_record = $reinstalled
        marker_count = $markers.Count
    }
    $target = if ($EvidencePath) { $EvidencePath } else { $context.Evidence }
    Write-AcceptanceEvidence -Path $target -Document $evidence
    Write-Output ($evidence | ConvertTo-Json -Depth 20 -Compress)
}
finally {
    [Environment]::SetEnvironmentVariable('Path', $originalUserPath, 'User')
    if ($null -ne $snapshot) { Exit-InstallerTestContext $snapshot }
}
