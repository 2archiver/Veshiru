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
draws, EDS2 where supported, GPU thread pinning on Windows/Linux and a conservative VRAM mode with
a configurable texture-cache limit. Imported profiles and per-game overrides are not rewritten.

The texture-cache limit (`texture_cache_memory_limit`, Advanced Graphics, MiB, per-game capable)
uses Auto = 0 by default: three quarters of the live device budget, clamped to the device. Lichii's
driver-budget refresh, out-of-memory backoff, allocator block-slack reserve and host commit bound
still apply on top. On a 4 GB card this keeps the cache below the WDDM budget instead of using a
fixed high-end cap.

## Upstream references

Verified during this development pass:

- Eglantine Community `nx2` and SmileyTatsu's Eglantine build 17/build 20: ASTC one-submit,
  decode-image pooling, scratch bucketing, in-place opacity scan and texture-cache memory limit.
- Lichii build 177: current compatibility/renderer baseline, NUP container/validation, budget-aware
  texture collection, shader cache v2 and Windows crash/minidump work.
- Nemyu (GitLab `nemyu-emulator/nemyu`, GitHub mirror `PlayerJV55/DeckEmuV2OOT`): NUP base-title
  peek, decompression-cache override, damaged-member fallback and slow-decompression diagnostics.

Do not blindly merge whole forks. Port the smallest compatible change, preserve upstream SPDX
notices, and add a regression test or a targeted diagnostic where practical.

## Save compatibility

Eglantine build 17 kept Eden's `eden` data directory; later forks used `eglantine`, `lichii` and
`nemyu`. Veshiru detects all of them plus the older supported emulators. Copy is the default import
strategy. The importer verifies copied bytes, keeps the source unchanged, remaps stored paths by
path component, and journals interrupted imports. Existing destination data is backed up rather
than merged. Save data, title IDs, save-size files and compatible shader caches are preserved.

## Live console

`Tools > Console` opens a non-modal Qt window independent of the main window. The common logging
layer has a bounded MPSC console backend that is enabled only while the window is open. Logging
threads enqueue already-formatted messages; a worker sanitizes, rate-limits and buffers them. The
UI polls without blocking emulation. Display pause stops UI updates only.

The console groups existing log classes into CPU, GPU, HLE, FS, Audio, Networking, Performance,
Compatibility and Other. It does not invent performance statistics. Diagnostic export writes a
sanitized zip containing logs, build/version information and hardware/graphics details; it never
includes game files, saves, firmware, keys or credentials.

## NUP

NUP remains a major container format. The target profile sets `EDEN_NX2_NUP_CACHE_MIB=1024` because
the host has 32 GB RAM; this is host memory, not GTX 1650 VRAM. `EDEN_NUP_DAMAGED_FALLBACK=0`
disables serving a sound sibling for a damaged `.zs` member. `NupContainer::PeekBaseTitleId` reads
only package records so large collections can be scanned without parsing every container. The loader
uses that peek to find a matching sibling base for an update-only file, exposes carried add-ons to
the AOC/filesystem services when installed content is absent, and lists carried Update/DLC entries
in the existing per-game add-on dialog. Disabling Update selects the base; disabling DLC hides
carried add-ons. Installed content still takes precedence.

## Build and test

The authoritative desktop build is CI (`.github/workflows/build-check.yml`): Linux GCC and Windows
clang-cl with Qt 6.8.3, tests enabled. Release binaries use
`.github/workflows/release-binaries.yml`; the Windows PGO build uses `veshiru-windows.profdata`.

Useful local checks that do not require the full Qt/Vulkan stack:

```sh
CXX=g++ python3 tools/nx2/test_switches.py
python3 tools/nx2/test_cache_metrics.py
python3 tools/nx2/test_cache_bench.py
```

Gameplay performance must be measured on the target PC with identical settings and scenes. Track
average/min FPS, frame-time consistency, CPU/GPU utilisation, RAM/VRAM and shader stutters. Do not
claim a speedup from code inspection alone.
