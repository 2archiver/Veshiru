<!--
SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Veshiru CI and release guide

Veshiru is a Windows-exclusive emulator. GitHub Actions builds, tests and publishes 64-bit Windows
packages only; Linux and macOS artifacts are not produced.

## Live workflows

| Workflow | File | Triggers | Purpose |
| --- | --- | --- | --- |
| `build-check` | `.github/workflows/build-check.yml` | Pull requests & pushes to `main`, manual dispatch | Compiles `veshiru`, `veshiru-cmd` and `tests` with LLVM 22 `clang-cl` (`-DENABLE_WERROR=ON`) and runs the CTest suite + `tools/nx2` checks (skipped automatically for docs-only PRs). |
| `release-binaries` | `.github/workflows/release-binaries.yml` | Manual dispatch; publishing from `main` only | Builds ThinLTO (`windows-x86_64`) and ThinLTO + PGO (`windows-x86_64-pgo`) archives, checks both console executables and their version, and uploads verified archives with `SHA256SUMS.txt`. Publishes only when explicitly selected. |

Every job declares an explicit `timeout-minutes` budget and least-privilege `permissions:` (`contents: read` on build/test jobs; `contents: write` only on tag/release jobs), and checks out repository code with `persist-credentials: false`.

## Supply-chain and checksum verification

All external toolchain and dependency downloads in CI are cryptographically verified before use:

- **LLVM 22.1.8 (`clang-cl` / `lld-link` / `llvm-lib` / `llvm-profdata`)**: pinned SHA-256 verified in `tools/windows/install-llvm.ps1`.
- **Vulkan SDK 1.4.341.1**: SHA-256 verified against LunarG's official digest endpoint (or `$env:VULKAN_SDK_SHA256`) plus Windows Authenticode signature verification (`Get-AuthenticodeSignature`) in `tools/windows/install-vulkan-sdk.ps1`.
- **Compiler caches (`ccache` 4.14.2 / `sccache` 0.19.0)**: pinned SHA-256 verified before archive extraction in `build-check.yml` and `release-binaries.yml`.
- **CPM source & binary packages (`cpmfile.json`)**: verified against pinned SHA-512 / SHA-256 digests in `CMakeModules/CPMUtil.cmake`, including cached `.cache/cpm` hash marker validation (`needs_refetch`).
- **Release archives & in-app updater**: `release-binaries.yml` generates `SHA256SUMS.txt` over `veshiru-windows-x86_64.zip` and `veshiru-windows-x86_64-pgo.zip`. The in-app updater (`UpdateDialog`) requires a matching entry in `SHA256SUMS.txt` (or GitHub's asset `sha256:` digest) and verifies the downloaded `.zip` before invoking `veshiru-updater.exe`.

## Version, tags and updater rules

`VERSION` at the repository root is the single source of truth for the semantic version and must contain one stable `X.Y.Z` line for a release.

- **No automatic releases** are published on pushes or on a schedule. Build checks continue to run on pull requests and pushes to `main`.
- **Package validation**: dispatch `release-binaries` from the candidate branch with `release_tag` set to `vX.Y.Z` matching `VERSION` and leave `publish_release` off. The Actions artifact `veshiru-package-vX.Y.Z-<commit>` includes the two updater-compatible archive names and `SHA256SUMS.txt`. No tag or release is created.
- **Stable tagged releases**: dispatch the same workflow from `main` with `publish_release` on only after validation and approval. Both the tag and `-DVESHIRU_BUILD_TAG` receive `vX.Y.Z`, `GenerateSCMRev.cmake` marks `IS_DEV_BUILD` as `false` (even on local tagged checkouts where `VESHIRU_BUILD_TAG` is omitted), and the release body is extracted from `## [X.Y.Z]` in `CHANGELOG.md`.
- **Update eligibility (`UpdateChecker::SelectUpdate`)**:
  - Installed `vX.Y.Z` stable builds are offered newer `vX.Y.Z` stable releases.
  - Installed `build-N` pre-release builds are offered newer `build-M` pre-releases (`M > N`) **and** newer `vX.Y.Z` stable releases so preview users transition onto stable releases automatically.
  - Release candidates (`-rcN`) are never auto-offered by the background update checker.

## Windows PGO profile (`veshiru-windows.profdata`)

`windows-x86_64-pgo` compiles `veshiru.exe` and `veshiru-cmd.exe` (`veshiru-updater.exe` explicitly strips PGO flags to stay lightweight and self-contained) with `-flto=thin -fprofile-use=veshiru-windows.profdata`.

`veshiru-windows.profdata` is an LLVM v13 indexed IR profile (`clang-cl` MSVC ABI, `?`-mangled C++ symbols and `D:\a\Veshiru\Veshiru\` normalized internal-linkage paths) merged with balanced session weights across Switch 1 and Switch 2 gameplay workloads (`77,537` symbols, `23,103` active hot functions covering `Core`, `Vulkan`, `VideoCommon`, `Shader`, `Dynarmic`, `AudioCore`, NUP zstd streaming and Qt UI paths).

### Retraining a multi-game PGO profile

Use the same LLVM major/minor version installed by `tools/windows/install-llvm.ps1` (LLVM 22.1.8):

1. Configure an instrumented Release build using the built-in `VESHIRU_PGO_GENERATE` CMake option:
   ```powershell
   cmake -S . -B build/pgo-instr -G Ninja `
     -DCMAKE_BUILD_TYPE=Release `
     -DVESHIRU_PGO_GENERATE="$PWD/build/pgo-raw" `
     "-DCMAKE_C_COMPILER=$env:LLVM_ROOT/bin/clang-cl.exe" `
     "-DCMAKE_CXX_COMPILER=$env:LLVM_ROOT/bin/clang-cl.exe" `
     "-DCMAKE_LINKER=$env:LLVM_ROOT/bin/lld-link.exe" `
     "-DCMAKE_AR=$env:LLVM_ROOT/bin/llvm-lib.exe"
   cmake --build build/pgo-instr --target veshiru --parallel
   ```
2. Record one subdirectory per representative game workload so no single long session dominates:
   ```powershell
   $env:LLVM_PROFILE_FILE = "$PWD/build/pgo-raw/game1/veshiru-%m-%p.profraw"
   .\build\pgo-instr\bin\veshiru.exe
   ```
   Repeat for 3–5 representative Switch 1 and Switch 2 titles (`game2`, `game3`, …) and exit `veshiru.exe` cleanly after each run so atomic counters flush on exit.
3. Merge and weight-balance all sessions into `veshiru-windows.profdata`:
   ```powershell
   pwsh ./tools/windows/merge-pgo-profiles.ps1 -ProfileRoot build/pgo-raw -Output veshiru-windows.profdata
   ```
4. Commit the updated `veshiru-windows.profdata` at the repository root.
