<!--
SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Veshiru development notes

Veshiru is an independent Nintendo Switch 2 emulator. Upstream projects are engineering
references, not a branding exercise: compare changes individually, keep save compatibility, and
prefer proven fixes over speculative rewrites.

## Hardware target

The first-class target is:

- Intel Core i7-10750H (6 cores / 12 threads)
- NVIDIA GTX 1650 4 GB as the primary Vulkan renderer
- Intel UHD integrated graphics present but not the default target
- 32 GB host RAM
- Windows 11

Fresh global profiles therefore seed Vulkan, native resolution, no extra AA, GPU ASTC decode with
BC3 recompression, disk/driver pipeline caches, asynchronous shaders/presentation, bounded native
draws, EDS2 where supported, GPU thread pinning on Windows and a conservative VRAM mode with
a configurable texture-cache limit. Imported profiles and per-game overrides are not rewritten.

The texture-cache limit (`texture_cache_memory_limit`, Advanced Graphics, MiB, per-game capable)
uses Auto = 0 by default: three quarters of the live device budget, clamped to the device. Lichii's
driver-budget refresh, out-of-memory backoff, allocator block-slack reserve and host commit bound
still apply on top. On a 4 GB card this keeps the cache below the WDDM budget instead of using a
fixed high-end cap.

## Upstream references

Verified during this development pass:

- Eglantine Community `nx2` and SmileyTatsu's Eglantine build 17/build 20: ASTC one-submit,
  decode-image pooling, scratch bucketing, in-place opacity scan, texture-cache memory limit,
  `R16G16B16A16_SFLOAT` HDR swapchain preservation and native asynchronous pipeline compilation.
- Lichii (`#948`–`#1907`): current compatibility/renderer baseline, NUP container/validation,
  budget-aware texture collection, `ShaderCacheV2`, Maxwell/Ampere shader lowering, `SET_SCISSOR`
  stride, `RenderTargetState` layout sync, NCZ bounds validation and Windows crash/minidump work.
- Nemyu (GitLab `nemyu-emulator/nemyu`, GitHub mirror `PlayerJV55/DeckEmuV2OOT`): NUP base-title
  peek, decompression-cache override, damaged-member fallback and slow-decompression diagnostics.
- Deadly-pro/Open-Frame-gen (`Copyright (c) 2026 Prithivi Nair, Prajwal R, Madhav S, Rashmi N Ugarakhod`,
  MIT License): `NanoModel` depthwise-separable optical-flow U-Net + Squeeze-and-Excitation
  architecture and `nano_v17_best` checkpoint weights, implemented natively in Vulkan compute
  shaders (`open_frame_gen_*.comp`) with BatchNorm-folded FP16 embedded weights.

Do not blindly merge whole forks. Port the smallest compatible change, preserve upstream SPDX
notices, and add a regression test or a targeted diagnostic where practical.

## Frame generation

**Configure > Graphics > Frame Generation** (`frame_gen_type`) offers three modes:

- **Off** (default)
- **LS1 (Lossless Scaling)** (`FrameGenType::Lsfg`) — optical-flow search/inpaint/blend compute pipeline
- **Open-Frame-Gen (Neural Optical Flow)** (`FrameGenType::OpenFrameGen`) — native Vulkan compute port
  of `Open-Frame-Gen`'s `NanoModel` U-Net (`open_frame_gen_{prep,dw3x3,pw1x1,se,head,warp}.comp`).
  Evaluates dense bidirectional optical flow and an occlusion mask at quarter resolution using
  packed `rgba16f` tensor layers and 87,995 BatchNorm-folded FP16 weights (~176 KiB embedded in
  `open_frame_gen_weights.cpp`), then performs bilinear flow upsampling, bidirectional warping and
  CAS sharpening at presentation resolution without Python, CUDA or TensorRT dependencies.

## High-resolution game artwork

Game-list icons are cached lossless up to 512x512 PNG in
`%APPDATA%\veshiru\cache\game_list\artwork\<title_id>.png` (`Common::ArtworkCache`) without
touching user custom overrides in `load/<title_id>/icon.*`, and rendered with `devicePixelRatioF()`
scaling on Windows HiDPI monitors. `IconSearch` queries Nintendo CDN catalogs by 16-hex Title ID
first and falls back to an edition-aware scored title matcher (`Nintendo Switch 2 Edition`,
Deluxe/Demo/Region disambiguation), silently auto-fetching missing artwork in the background and
exposing **Refresh Game Artwork** on the game-list context menu.

## Save compatibility

Eglantine build 17 kept Eden's `eden` data directory; later forks used `eglantine`, `lichii` and
`nemyu`. Veshiru detects all of them plus the older supported emulators. Copy is the default import
strategy. The importer verifies copied bytes, keeps the source unchanged, remaps stored paths by
path component, and journals interrupted imports. Existing destination data is backed up rather
than merged. Save data, title IDs, save-size files and compatible shader caches are preserved.

On top of the first-launch import, `Right-click game > Link to Eglantine` transfers a single
title's saves. Eglantine stores saves in the same `nand/user/save/0000000000000000/<user>/<title>`
NAND layout, so the transfer copies the data to a sibling staging directory, verifies the copy
(file list, sizes and SHA-256 content digests), moves the replaced directory aside to a timestamped
`*.veshiru-backup-*` folder instead of deleting it, and finally makes the Veshiru save directory a
junction pointing at the Eglantine save, so both emulators reference the same files. Junctions are
created and inspected natively (`FSCTL_SET_REPARSE_POINT` / `FSCTL_GET_REPARSE_POINT`) — no
`mklink` shell-out — and unlinking stages and verifies the standalone copy before removing the
junction. Every step reports its errors to the user instead of failing silently.

## Live console

`Tools > Console` opens a non-modal Qt window independent of the main window. The common logging
layer has a bounded MPSC console backend that is enabled only while the window is open. Logging
threads enqueue already-formatted messages; a worker sanitizes (redacting case-insensitive Windows
and POSIX user paths and credential/token headers), rate-limits and buffers them. `veshiru_log.txt`
flushes dirty buffers every second and on crash paths. The UI polls without blocking emulation, and
display pause stops UI updates only.

The console groups existing log classes into CPU, GPU, HLE, FS, Audio, Networking, Performance,
Compatibility and Other. It does not invent performance statistics. Diagnostic export writes a
sanitized zip containing logs, build/version information and hardware/graphics details; it never
includes game files, saves, firmware, keys or credentials.

## NUP and host memory

NUP remains a major container format. Hosts with `>= 24 GiB` RAM automatically default the host-RAM
NUP decompression cache to **1,536 MiB** (`1.5 GiB`; 256 MiB on smaller systems), overridable via
`VESHIRU_NUP_CACHE_MIB` or `EDEN_NX2_NUP_CACHE_MIB` (clamped to `[0, 16384]` MiB), and stream zstd
blocks in 512 KiB chunks; this is host memory, not GTX 1650 VRAM. On `>= 24 GiB` hosts, guest
physical-memory zero-page sweeps (`TrimZeroPages`) are also deferred while emulation is actively
running to avoid mid-gameplay `VirtualAlloc(MEM_RESET)` page-table churn. `EDEN_NUP_DAMAGED_FALLBACK=0`
disables serving a sound sibling for a damaged `.zs` member. `NupContainer::PeekBaseTitleId` reads
only package records so large collections can be scanned without parsing every container. The loader
uses that peek to find a matching sibling base for an update-only file, exposes carried add-ons to
the AOC/filesystem services when installed content is absent, and lists carried Update/DLC entries
in the existing per-game add-on dialog. Disabling Update selects the base; disabling DLC hides
carried add-ons. Installed content still takes precedence.

## Build and test

The authoritative desktop build is CI (`.github/workflows/build-check.yml`): Windows
clang-cl with Qt 6.8.3, tests enabled. Release binaries use
`.github/workflows/release-binaries.yml`; the Windows PGO build uses `veshiru-windows.profdata`.

Useful local checks that do not require the full Qt/Vulkan stack:

```sh
python3 tools/nx2/test_switches.py
python3 tools/nx2/test_cache_metrics.py
python3 tools/nx2/test_cache_bench.py
```

Gameplay performance must be measured on the target PC with identical settings and scenes. Track
average/min FPS, frame-time consistency, CPU/GPU utilisation, RAM/VRAM and shader stutters. Do not
claim a speedup from code inspection alone.
