# SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

<#
.SYNOPSIS
    Merges multi-game LLVM clang-cl PGO raw/indexed profiles into a balanced veshiru-windows.profdata.
.DESCRIPTION
    Each subdirectory under -ProfileRoot (or each path passed to -SessionPaths) represents one
    game or workload session containing .profraw or .profdata files (e.g. Switch 2 gameplay,
    Switch 1 gameplay, OpenFrameGen/FSR3 active session, GUI game-list/artwork/NUP startup).

    Each session is merged independently, measured by its total execution counter volume (sum of
    per-function execution counts rather than spinloop-skewed maximum function count), and then
    combined with bounded 64-bit inverse-volume weights so long sessions do not drown out shorter
    sessions and tiny sessions do not suffer runaway amplification.

    Finally, the merged profile is verified for MSVC-mangled ('?') clang-cl symbols and audited
    for function coverage across Veshiru's key runtime subsystems.
#>
param(
    [string]$ProfileRoot = "build/pgo-raw",
    [string[]]$SessionPaths = @(),
    [string]$Output = "veshiru-windows.profdata",
    [string]$LlvmProfdata = "",
    [int64]$BaseWeight = 100,
    [int64]$MaxWeight = 10000,
    [switch]$StrictCoverage
)

$ErrorActionPreference = "Stop"

if ($BaseWeight -lt 1) {
    throw "BaseWeight must be >= 1."
}
if ($MaxWeight -lt $BaseWeight) {
    throw "MaxWeight ($MaxWeight) must be >= BaseWeight ($BaseWeight)."
}

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

function Get-ProfileVolumeAndStats {
    param(
        [string]$ToolPath,
        [string]$ProfdataPath
    )

    $Lines = & $ToolPath show --all-functions --counts "$ProfdataPath"
    if ($LASTEXITCODE -ne 0) {
        throw "llvm-profdata show failed for '$ProfdataPath'"
    }

    [double]$TotalFunctionCount = 0.0
    [double]$MaxFunctionCount = 0.0
    [int64]$TotalFunctions = 0
    [int64]$HotFunctions = 0
    [bool]$HasMsvcSymbols = $false

    foreach ($Line in $Lines) {
        if (-not $HasMsvcSymbols -and $Line -match '^\s*\?') {
            $HasMsvcSymbols = $true
        }
        if ($Line -match '^\s*Function count:\s*(\d+)') {
            $Count = [double]$Matches[1]
            $TotalFunctionCount += $Count
            if ($Count -gt 0.0) {
                $HotFunctions++
            }
        } elseif ($Line -match '^\s*Total functions:\s*(\d+)') {
            $TotalFunctions = [int64]$Matches[1]
        } elseif ($Line -match '^\s*Maximum function count:\s*(\d+)') {
            $MaxFunctionCount = [double]$Matches[1]
        }
    }

    # Prefer summed per-function execution volume so spinloops (e.g. GpuSplit / Scheduler waits)
    # cannot distort session balancing the way Maximum function count does.
    $Volume = if ($TotalFunctionCount -gt 0.0) {
        $TotalFunctionCount
    } elseif ($MaxFunctionCount -gt 0.0) {
        Write-Warning "No per-function counts parsed for '$ProfdataPath'; falling back to Maximum function count."
        $MaxFunctionCount
    } else {
        1.0
    }

    return [PSCustomObject]@{
        Volume             = $Volume
        TotalFunctionCount = $TotalFunctionCount
        MaxFunctionCount   = $MaxFunctionCount
        TotalFunctions     = $TotalFunctions
        HotFunctions       = $HotFunctions
        HasMsvcSymbols     = $HasMsvcSymbols
        RawLines           = $Lines
    }
}

$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("veshiru-pgo-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

try {
    $SessionProfdata = @()
    $SessionNames = @()
    $SessionVolumes = @()
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

        $Stats = Get-ProfileVolumeAndStats -ToolPath $LlvmProfdata -ProfdataPath $MergedSession
        if (-not $Stats.HasMsvcSymbols) {
            throw "Session '$Session' does not contain MSVC-mangled ('?') clang-cl symbols!"
        }

        Write-Host ("Session [{0}] '{1}': files={2}, hot_funcs={3}/{4}, total_vol={5:N0}, max_func={6:N0}" -f `
            $Index, (Split-Path -Leaf $Session), $Files.Count, $Stats.HotFunctions, `
            $Stats.TotalFunctions, $Stats.TotalFunctionCount, $Stats.MaxFunctionCount)

        $SessionProfdata += $MergedSession
        $SessionNames += (Split-Path -Leaf $Session)
        $SessionVolumes += $Stats.Volume
        $Index++
    }

    if ($SessionProfdata.Count -eq 0) {
        throw "No .profraw or .profdata inputs were found."
    }

    $MaxVolume = ($SessionVolumes | Measure-Object -Maximum).Maximum
    $WeightedArgs = @("merge", "-output=$Output")
    for ($i = 0; $i -lt $SessionProfdata.Count; $i++) {
        $RawWeight = [Math]::Round(($MaxVolume / [Math]::Max(1.0, $SessionVolumes[$i])) * [double]$BaseWeight)
        [int64]$ClampedWeight = [int64][Math]::Min([double]$MaxWeight, [Math]::Max(1.0, $RawWeight))
        Write-Host ("  -> Weight for '{0}': {1} (ratio {2:N2}x)" -f `
            $SessionNames[$i], $ClampedWeight, ($MaxVolume / [Math]::Max(1.0, $SessionVolumes[$i])))
        $WeightedArgs += "--weighted-input=$ClampedWeight,$($SessionProfdata[$i])"
    }

    & $LlvmProfdata @WeightedArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Final llvm-profdata weighted merge failed"
    }

    $FinalStats = Get-ProfileVolumeAndStats -ToolPath $LlvmProfdata -ProfdataPath $Output
    if (-not $FinalStats.HasMsvcSymbols) {
        throw "Merged profile '$Output' does not contain MSVC-mangled ('?') clang-cl symbols!"
    }

    # Audit subsystem coverage so missing workloads (e.g. OpenFrameGen, ArtworkCache) are visible
    # before shipping a new veshiru-windows.profdata.
    $Subsystems = [ordered]@{
        "GpuSplit"      = "GpuSplit"
        "PipelineCache" = "PipelineCache"
        "TextureCache"  = "TextureCache"
        "BufferCache"   = "BufferCache"
        "Maxwell3D"     = "Maxwell3D"
        "ASTC"          = "DecompressBlock|ASTC"
        "OpenFrameGen"  = "OpenFrameGen"
        "Fsr3FrameGen"  = "Fsr3FrameGen"
        "FrameGenPacer" = "FrameGenPacer"
        "ArtworkCache"  = "ArtworkCache"
        "NupContainer"  = "NupContainer|FileSys@@"
    }

    $SymbolLines = $FinalStats.RawLines | Where-Object { $_ -match '^\s*\S+:$' }
    $MissingSubsystems = @()
    Write-Host "`nMerged profile subsystem coverage ($Output):"
    Write-Host ("  Total hot functions: {0} / {1}" -f $FinalStats.HotFunctions, $FinalStats.TotalFunctions)
    foreach ($Name in $Subsystems.Keys) {
        $Pattern = $Subsystems[$Name]
        $Count = ($SymbolLines | Where-Object { $_ -match $Pattern }).Count
        Write-Host ("  - {0,-15}: {1,5} symbols" -f $Name, $Count)
        if ($Count -eq 0) {
            $MissingSubsystems += $Name
        }
    }

    if ($MissingSubsystems.Count -gt 0) {
        $Msg = "Merged PGO profile has 0 symbols for subsystem(s): $($MissingSubsystems -join ', '). " +
               "Include gameplay + FrameGen + GUI library sessions for full 1.4 coverage."
        if ($StrictCoverage) {
            throw $Msg
        } else {
            Write-Warning $Msg
        }
    }

    Write-Host "`nMerged $($SessionProfdata.Count) PGO session(s) into $Output"
} finally {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $TempDir
}
