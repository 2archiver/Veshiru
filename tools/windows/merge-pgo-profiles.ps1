# SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

<#
.SYNOPSIS
    Merges multi-game LLVM clang-cl PGO raw/indexed profiles into a balanced veshiru-windows.profdata.
.DESCRIPTION
    Each subdirectory under -ProfileRoot (or each path passed to -SessionPaths) represents one
    game or workload session containing .profraw or .profdata files. Each session is first merged
    on its own, inspected for total counter volume, and then combined with inverse-volume weights
    so longer sessions do not drown out shorter Switch 1 / Switch 2 workloads.
#>
param(
    [string]$ProfileRoot = "build/pgo-raw",
    [string[]]$SessionPaths = @(),
    [string]$Output = "veshiru-windows.profdata",
    [string]$LlvmProfdata = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($LlvmProfdata)) {
    if ($env:LLVM_ROOT -and (Test-Path (Join-Path $env:LLVM_ROOT "bin/llvm-profdata.exe"))) {
        $LlvmProfdata = Join-Path $env:LLVM_ROOT "bin/llvm-profdata.exe"
    } else {
        $LlvmProfdata = "llvm-profdata"
    }
}

if ($SessionPaths.Count -eq 0) {
    if (-not (Test-Path $ProfileRoot)) {
        throw "ProfileRoot '$ProfileRoot' does not exist."
    }
    $SubDirs = Get-ChildItem -Path $ProfileRoot -Directory
    if ($SubDirs.Count -gt 0) {
        $SessionPaths = $SubDirs.FullName
    } else {
        $SessionPaths = @((Resolve-Path $ProfileRoot).Path)
    }
}

$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("veshiru-pgo-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

try {
    $SessionProfdata = @()
    $SessionTotals = @()
    $Index = 0

    foreach ($Session in $SessionPaths) {
        $Files = Get-ChildItem -Path $Session -Recurse -File |
            Where-Object { $_.Extension -in @(".profraw", ".profdata") }
        if ($Files.Count -eq 0) {
            Write-Warning "Skipping empty session directory: $Session"
            continue
        }

        $MergedSession = Join-Path $TempDir ("session-$Index.profdata")
        & $LlvmProfdata merge -output="$MergedSession" @($Files.FullName)
        if ($LASTEXITCODE -ne 0) {
            throw "llvm-profdata merge failed for session '$Session'"
        }

        $Summary = (& $LlvmProfdata show --all-functions "$MergedSession") -join "`n"
        $TotalSum = 1.0
        if ($Summary -match 'Maximum function count:\s*(\d+)') {
            $MaxCount = [double]$Matches[1]
            if ($MaxCount -gt 0) {
                $TotalSum = $MaxCount
            }
        }

        $SessionProfdata += $MergedSession
        $SessionTotals += $TotalSum
        $Index++
    }

    if ($SessionProfdata.Count -eq 0) {
        throw "No .profraw or .profdata inputs were found."
    }

    $MaxSession = ($SessionTotals | Measure-Object -Maximum).Maximum
    $WeightedArgs = @("merge", "-output=$Output")
    for ($i = 0; $i -lt $SessionProfdata.Count; $i++) {
        $Weight = [Math]::Max(1, [int][Math]::Round(($MaxSession / $SessionTotals[$i]) * 100.0))
        $WeightedArgs += "--weighted-input=$Weight,$($SessionProfdata[$i])"
    }

    & $LlvmProfdata @WeightedArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Final llvm-profdata weighted merge failed"
    }

    $Check = (& $LlvmProfdata show --topn=10 "$Output") -join "`n"
    if ($Check -notmatch '\?') {
        throw "Merged profile '$Output' does not contain MSVC-mangled ('?') clang-cl symbols!"
    }

    Write-Host "Merged $($SessionProfdata.Count) PGO session(s) into $Output"
} finally {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $TempDir
}
