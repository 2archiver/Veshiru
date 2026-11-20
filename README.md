<!--
# SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
# SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# SPDX-License-Identifier: GPL-2.0-or-later
-->

<h2 align="center">Download Veshiru <b>only</b> from this repository's <a href="https://github.com/2archiver/Veshiru/releases">releases page</a>.</h2>

# Veshiru

Veshiru is an experimental, open-source Nintendo Switch 2 emulator for Windows. It is an
independent project with its own executable, profile, interface and direction. It builds on
verified work from Eglantine, Lichii, Nemyu, Eden, yuzu and Sudachi, with their licenses and
copyright notices kept.

Veshiru is **Windows-exclusive**: it is developed, built and tested for 64-bit Windows 10 and 11,
and only Windows packages are published. It is not affiliated with Nintendo.

## System requirements

- Windows 10 or 11, 64-bit.
- A Vulkan 1.1+ GPU. 4 GB of VRAM is the practical floor; 8 GB or more is recommended.
- 16 GB of system RAM (24 GB or more lets Veshiru keep decompressed NUP data in memory).
- A CPU with AVX2. Six or more cores help shader and pipeline background work.

The reference target is an Intel i7-10750H, an NVIDIA GTX 1650 Laptop GPU (4 GB) and 32 GB of
RAM; that machine is what the memory and pipeline defaults are tuned for.

## Installation

1. Download the latest build archive from the [releases page](https://github.com/2archiver/Veshiru/releases).
2. Extract the **entire** archive to a folder you can write to, and run `veshiru.exe`.
3. Point **Configure > General > Game Directories** at the folder holding your games.
4. Install firmware and keys you dumped from your own console under **File > Open Veshiru folder**.

Games, firmware and keys are not included and are not provided. Use only copies you legally own.

## Basic configuration

Veshiru sizes video memory, shader caches and pipeline workers automatically for the GPU it finds,
so the defaults suit most systems. The settings that matter day to day:

- **Graphics**: resolution scale (1x is native), anisotropic filtering, and the FSR/DLSS upscaler
  if your GPU supports one.
- **System**: console language and region, docked or handheld, and the memory layout for the game.
- **Controls**: keyboard, gamepad and motion bindings, per player.
- **Per-game settings**: anything set while a game is running applies to that game only.

Memory and VRAM controls exist but sit behind **Graphics > Advanced > Show memory expert options**.
They are for testing; the automatic values are what the hardware should get.

If a game's file is moved or deleted, its row stays in the list marked **Missing**, with the old
location in the tooltip — saves live in the NAND save data, so putting the file back restores the
game. Files under a folder you remove from the game list are not reported.

**Frame generation** (Open-Frame-Gen) adds interpolated frames on top of the game's own output. It
does not raise the game's simulation rate, and turning it on can lower native FPS; leave it off
unless you specifically want smoother-looking motion.

## Data and profiles

Veshiru uses `%APPDATA%\veshiru`. A portable `user\` folder next to `veshiru.exe` takes priority.
On first launch Veshiru offers to import saves and settings from earlier Eglantine, Lichii and
Nemyu profiles; **Copy** is the default and leaves the source untouched.

Logs are written to the `log` folder inside the profile. **Tools > Console** shows the same stream
live, and **Export diagnostics** produces a sanitised zip with logs and hardware details — no game
files, saves, firmware or keys. Veshiru adds no telemetry and uploads nothing on its own.

## Known limitations

- Very experimental. Crashes, rendering errors, audio glitches and stutter are expected.
- Compatibility and speed vary widely by game, GPU and driver; there is no universal frame-rate
  promise.
- Frame generation is cosmetic and costs native performance on some systems.
- Impersonation: only the releases page linked above is official.

## Reporting an issue

Include the build, game and version, your CPU, GPU and driver version, the settings that were in
use, and steps to reproduce. Attach a screenshot and a diagnostic bundle. Do not upload game files,
keys, firmware or account information.

## Building

The GUI target is `veshiru` (output `veshiru.exe`); the CLI target is `veshiru-cmd`. See the
[build guide](docs/Build.md), [coding guidelines](docs/Coding.md) and
[development notes](docs/Veshiru.md). Keep changes focused, and separate code-level checks from
real gameplay testing; performance claims need identical-settings benchmarks on the target
hardware.

## Credits and license

Veshiru inherits the work of the Eglantine, Lichii, Nemyu, Eden, yuzu and Sudachi projects and
their contributors, including Eglantine's developer Camille LaVey, SmileyTatsu's GTX 1650
ASTC/memory work, zortos293's Lichii compatibility work and UesleiDev's Nemyu NUP/interface work.
The optional **Open-Frame-Gen** Vulkan neural optical-flow frame generator adapts `NanoModel` and
pretrained weights from [Deadly-pro/Open-Frame-gen](https://github.com/Deadly-pro/Open-Frame-gen)
(`Copyright (c) 2026 Prithivi Nair, Prajwal R, Madhav S, Rashmi N Ugarakhod`, MIT License).

See [LICENSE.txt](LICENSE.txt) and the source files for the applicable terms.
