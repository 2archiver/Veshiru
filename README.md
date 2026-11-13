<!--
# SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
# SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# SPDX-License-Identifier: GPL-2.0-or-later
-->

<h2 align="center">⚠️ Veshiru has no official website ⚠️</h2>

> [!CAUTION]
> **Veshiru has no official website.** Download Veshiru **only** from this repository's [releases page](https://github.com/2archiver/Veshiru/releases). Any other "Veshiru" website or download is not built by this project and may be unsafe.

<p align="center">
  <img src="dist/branding/veshiru_icon_with_logo.png" alt="Veshiru" width="600">
</p>

# Veshiru

Veshiru is an experimental, open-source Nintendo Switch 2 emulator written in C++. It is an independent project with its own executable, profile, interface and development direction. It uses verified work from Eglantine, Lichii, Nemyu, Eden, yuzu and Sudachi as engineering references while preserving their contributions and licenses.

The current development target is a Windows 11 PC with an Intel Core i7-10750H, NVIDIA GTX 1650 4 GB, Intel UHD integrated graphics and 32 GB RAM. Performance decisions are made for that machine first: GTX 1650 Vulkan rendering, bounded VRAM use, asynchronous shader/pipeline work, proven ASTC decode improvements and low-overhead CPU/GPU threading. Veshiru is not affiliated with Nintendo.

## Downloads and setup

Download Veshiru only from this repository's [releases page](https://github.com/2archiver/Veshiru/releases). On Windows, extract the entire build archive and run **veshiru.exe**. Only use games, updates, firmware and keys you have legally obtained from your own console. Games, keys and firmware are not included.

### Existing Eglantine saves and settings

Veshiru uses its own profile: `%APPDATA%\veshiru` on Windows, and the `veshiru` directories under `$XDG_DATA_HOME`, `$XDG_CONFIG_HOME` and `$XDG_CACHE_HOME` on Linux and macOS. An existing portable `user/` directory still takes priority and is not imported or changed by the wizard.

Eglantine build 17 and other Eden-derived builds stored saves under `eden`; later Eglantine, Lichii and Nemyu builds used their own names. On the first non-portable GUI launch, Veshiru detects those profiles and offers an import. **Copy** is the default and leaves the source unchanged. Move retains a recovery backup; Link deliberately shares data. Existing Veshiru data is kept in backup directories rather than merged or deleted. Save bytes, title IDs, save-size files and compatible shader caches are preserved; imported storage paths are remapped to the new profile.

Back up `%APPDATA%\eden` (or the old Unix data/config/cache directories) before importing, and close the source emulator. `veshiru-cmd` refuses to initialize configuration until you import or explicitly choose **Start fresh** in the GUI, so a CLI-first launch cannot silently abandon existing saves.

## Live console and diagnostics

**Tools > Console** (`Ctrl+Shift+C`) opens an independent, non-modal live log window. It supports Trace/Debug/Info/Warning/Error/Critical levels, CPU/GPU/HLE/FS/Audio/Networking/Performance/Compatibility filters, search, copy, `.log` export, clear, auto-scroll and display pause. Pausing the display does not pause emulation or log capture.

The console backend is asynchronous and bounded, starts only while the window is open, rate-limits repeated messages and redacts personal paths/credential-like values. **Export diagnostics** creates a sanitized zip with logs, build information and hardware/graphics details. It does not include game files, saves, firmware, keys or personal documents. Veshiru does not add telemetry or automatic log uploads.

## NUP support

Veshiru loads NUP containers with base/update metadata, icons, localized titles, integrity verification and streamed zstd members. The target profile keeps a 1 GiB NUP decompression cache in host RAM (`EDEN_NX2_NUP_CACHE_MIB`) and can serve a sound sibling for a damaged `.zs` member (`EDEN_NUP_DAMAGED_FALLBACK=0` disables that fallback). Carried add-on content is exposed to the game when it is not installed separately, and update-only packages can use a matching base `.nup` in the same folder. The per-game add-on list shows carried Update/DLC entries and honors their disable switches.

## Compatibility and issue reports

Compatibility and performance vary by game, hardware and driver. Rendering errors, stutters and crashes remain possible. Passing automated tests does not establish that a game renders correctly, and this project makes no universal frame-rate guarantee.

When [reporting an issue](https://github.com/2archiver/Veshiru/issues), include:

- The exact build, game version, CPU, GPU and driver version.
- Relevant settings and steps to reproduce the problem.
- A screenshot and a diagnostic bundle from **Tools > Console**, after reviewing it.

Do not upload game files, keys, firmware or private account information.

## Building and contributing

See the [build guide](docs/Build.md), [user documentation](docs/user/README.md), [coding guidelines](docs/Coding.md) and [Veshiru development notes](docs/Veshiru.md). The GUI build target is `veshiru`; its output is `veshiru.exe` on Windows and `veshiru` on other desktop platforms. The CLI target is `veshiru-cmd`.

Keep changes focused and distinguish code-level checks from actual gameplay testing. Performance claims should be backed by identical-settings benchmarks on the target hardware.

## Artwork

Art created by YOB (yobslab). The supplied Veshiru logo, icon and library artwork are in [dist/branding](dist/branding). Windows, Linux and macOS icon resources are generated from that artwork by [Generate-VeshiruIcons.ps1](tools/Generate-VeshiruIcons.ps1).

## Credits and license

Veshiru inherits the work of the Eglantine, Lichii, Nemyu, Eden, yuzu and Sudachi projects and their contributors, including Eglantine's original developer Camille LaVey, SmileyTatsu's GTX 1650 ASTC/memory work, zortos293's Lichii compatibility work and UesleiDev's Nemyu NUP/interface work. Original copyright notices and third-party licenses are retained.

See [LICENSE.txt](LICENSE.txt) and the source files for applicable license terms.
