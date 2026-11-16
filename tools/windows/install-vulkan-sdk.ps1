# SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

$ErrorActionPreference = "Stop"

# Check if running as administrator
try {
    net session 1>$null 2>$null
} catch {
    Write-Host "This script must be run with administrator privileges!"
    Exit 1
}

$VulkanSDKVer = "1.4.341.1"
if ((Get-CimInstance -ClassName Win32_ComputerSystem).SystemType -match '(x64)') {
    $VulkanSDKArch = "X64"
    $VulkanSDKOs = "windows"
} else {
    $VulkanSDKArch = "ARM64"
    $VulkanSDKOs = "warm"
}
$VULKAN_SDK = "C:/VulkanSDK/$VulkanSDKVer"
$ExeFile = "vulkansdk-windows-$VulkanSDKArch-$VulkanSDKVer.exe"
$Uri = "https://sdk.lunarg.com/sdk/download/$VulkanSDKVer/$VulkanSDKOs/$ExeFile"
$ShaUri = "https://sdk.lunarg.com/sdk/sha/$VulkanSDKVer/$VulkanSDKOs/$ExeFile.txt"
$Destination = Join-Path $env:TEMP $ExeFile

# Check if Vulkan SDK is already installed
if (Test-Path $VULKAN_SDK) {
    Write-Host "-- Vulkan SDK already installed at $VULKAN_SDK"
    if ("$env:GITHUB_ACTIONS" -eq "true") {
        echo "VULKAN_SDK=$VULKAN_SDK" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
        echo "$VULKAN_SDK/Bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
    }
    return
}

try {
    Write-Host "Downloading Vulkan SDK $VulkanSDKVer from $Uri"
    Invoke-WebRequest -Uri $Uri -OutFile $Destination

    $ExpectedSha = $env:VULKAN_SDK_SHA256
    if ([string]::IsNullOrWhiteSpace($ExpectedSha)) {
        $ShaText = (Invoke-WebRequest -Uri $ShaUri -UseBasicParsing).Content
        if ($ShaText -match '([0-9a-fA-F]{64})') {
            $ExpectedSha = $Matches[1]
        } else {
            throw "Could not parse a valid 64-character SHA-256 digest from $ShaUri"
        }
    }

    $ActualSha = (Get-FileHash -Algorithm SHA256 -Path $Destination).Hash
    if ($ActualSha.ToLowerInvariant() -ne $ExpectedSha.Trim().ToLowerInvariant()) {
        throw "Vulkan SDK SHA-256 mismatch: expected $ExpectedSha, got $ActualSha"
    }

    $Sig = Get-AuthenticodeSignature -FilePath $Destination
    if ($Sig.Status -ne 'Valid') {
        throw "Vulkan SDK Authenticode signature is invalid (Status: $($Sig.Status))"
    }

    Write-Host "Verified Vulkan SDK $VulkanSDKVer (SHA-256: $ActualSha)"

    $Arguments = "--root `"$VULKAN_SDK`" --accept-licenses --default-answer --confirm-command install"
    Write-Host "Installing Vulkan SDK $VulkanSDKVer"
    $InstallProcess = Start-Process -FilePath $Destination -NoNewWindow -PassThru -Wait -ArgumentList $Arguments
    $ExitCode = $InstallProcess.ExitCode

    if ($ExitCode -ne 0) {
        Write-Host "Error installing Vulkan SDK $VulkanSDKVer (Error: $ExitCode)"
        Exit $ExitCode
    }
} finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $Destination
}

Write-Host "Finished installing Vulkan SDK $VulkanSDKVer"

if ("$env:GITHUB_ACTIONS" -eq "true") {
    echo "VULKAN_SDK=$VULKAN_SDK" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    echo "$VULKAN_SDK/Bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
}
