[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^v\d+\.\d+\.\d+$')]
    [string]$Release,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$Commit,
    [string]$ColibriPath = (Join-Path $PSScriptRoot '..\..\third_party\colibri')
)

$ErrorActionPreference = 'Stop'
$resolved = (Resolve-Path -LiteralPath $ColibriPath).Path
$remote = 'https://github.com/JustVugg/colibri.git'
if ($PSCmdlet.ShouldProcess($resolved, "fetch $Release and checkout $Commit")) {
    & git -c "safe.directory=$($resolved.Replace('\', '/'))" -C $resolved fetch --no-tags origin "refs/tags/$Release:refs/tags/$Release"
    if ($LASTEXITCODE -ne 0) { throw "git fetch failed with exit code $LASTEXITCODE" }
    $tagCommit = (& git -c "safe.directory=$($resolved.Replace('\', '/'))" -C $resolved rev-parse "$Release^{}").Trim()
    if ($tagCommit -ne $Commit.ToLowerInvariant()) {
        throw "release $Release resolves to $tagCommit, not requested commit $Commit"
    }
    & git -c "safe.directory=$($resolved.Replace('\', '/'))" -C $resolved checkout --detach $Commit
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed with exit code $LASTEXITCODE" }
    Write-Host "Checked out $remote $Release at $Commit."
    Write-Warning 'Update recorded constants, review license/notices, refresh patches, and rerun the full build and test suite.'
}
