Set-StrictMode -Version Latest

function Get-SwarmUv {
    $command = Get-Command uv.exe -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\uv\uv.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    $storePackageRoot = Join-Path $env:LOCALAPPDATA "Packages"
    if (Test-Path -LiteralPath $storePackageRoot) {
        $storeCandidates = @()
        $pythonPackages = @(
            Get-ChildItem -LiteralPath $storePackageRoot -Directory |
                Where-Object {
                    $_.Name -like "PythonSoftwareFoundation.Python.*"
                }
        )
        foreach ($pythonPackage in $pythonPackages) {
            $storeCandidates += @(
                Get-ChildItem `
                    -LiteralPath $pythonPackage.FullName `
                    -Filter uv.exe `
                    -File `
                    -Recurse `
                    -ErrorAction SilentlyContinue |
                    Where-Object {
                        $_.FullName -like `
                            "*\LocalCache\local-packages\Python*\Scripts\uv.exe"
                    }
            )
        }
        $storeCandidates = @($storeCandidates | Sort-Object FullName -Descending)
        if ($storeCandidates.Count -gt 0) {
            return $storeCandidates[0].FullName
        }
    }

    return $null
}
