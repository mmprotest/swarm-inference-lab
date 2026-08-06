[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SetupPath,
    [string]$ExpectedVersion = '0.1.0rc4',
    [string]$Label = 'clean-cpu',
    [string]$EvidencePath
)

. "$PSScriptRoot\windows_installer_test_lib.ps1"
$SetupPath = (Resolve-Path -LiteralPath $SetupPath).Path
$context = New-InstallerTestContext -Label $Label
$originalUserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$snapshot = $null
try {
    $snapshot = Enter-InstallerTestContext $context
    $script:InstallerCleanPath = $snapshot.CleanPath
    Assert-DeveloperCommandsAbsent
    $install = Invoke-SetupExecutable -SetupPath $SetupPath -Context $context -LogName 'setup-clean.log'
    $installed = Assert-InstalledRuntime -Context $context -ExpectedVersion $ExpectedVersion -ExpectedOperation 'install'
    $repair = Invoke-SetupExecutable -SetupPath $SetupPath -Context $context -LogName 'setup-repair.log'
    $repaired = Assert-InstalledRuntime -Context $context -ExpectedVersion $ExpectedVersion -ExpectedOperation 'repair'
    [void](New-Item -ItemType Directory -Force -Path $context.StateRoot)
    $marker = Join-Path $context.StateRoot 'acceptance-state-marker.txt'
    Set-Content -LiteralPath $marker -Value 'preserve-me' -Encoding utf8
    $uninstall = Invoke-UninstallExecutable -Context $context -PurgeState:$false
    Wait-PathAbsent -Path $context.InstallRoot
    Assert-Condition (Test-Path -LiteralPath $marker -PathType Leaf) 'normal uninstall preserves durable state'
    Assert-Condition ((Get-OwnedPathCount $context) -eq 0) 'normal uninstall removes only the owned PATH entry'
    Assert-Condition (@(Get-SwarmUninstallEntry $context).Count -eq 0) 'normal uninstall removes Apps & Features entry'
    $evidence = [ordered]@{
        schema_version = 1
        status = 'PASS'
        label = $Label
        setup_path = $SetupPath
        setup_sha256 = "sha256:$((Get-FileHash -LiteralPath $SetupPath -Algorithm SHA256).Hash.ToLowerInvariant())"
        install = $install
        repair = $repair
        uninstall = $uninstall
        install_record = $installed.record
        repaired_record = $repaired.record
        state_preserved = $true
        developer_commands_absent = $true
        service_created_before_pairing = $false
    }
    $target = if ($EvidencePath) { $EvidencePath } else { $context.Evidence }
    Write-AcceptanceEvidence -Path $target -Document $evidence
    Write-Output ($evidence | ConvertTo-Json -Depth 20 -Compress)
}
finally {
    [Environment]::SetEnvironmentVariable('Path', $originalUserPath, 'User')
    if ($null -ne $snapshot) { Exit-InstallerTestContext $snapshot }
}
