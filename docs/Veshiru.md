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

### Sparse texture tiles that do not fit in video memory

Since 2.1.0, when device-local memory is exhausted while a sparse texture's tiles are being
backed (seen with Mario Kart World on the 4 GB target), `MemoryAllocator::AllocateSparseMemory`
places the remaining tiles on a compatible non-device-local heap within its budget instead of
failing the upload. 2.2.0 does not change that; it makes it visible. When it happens the log gets
a warning, at most once every 10 seconds, with counters only:

```
Video memory is full: sparse texture tiles are going to a non-device-local heap. Since the last
report: N tiles (X MiB) placed there, F that no heap had room for. This session: ...
```

A warning is only written when a spill happens, so a burst that ends inside the 10 seconds is
counted but not printed until the next spill; the session totals are printed once more, at Info,
when the renderer shuts down. Whether these tiles cost frame time, and how much, has **not been
measured**: there are no before/after numbers from the same scene, so nothing here or in the log
line says so. Moving tiles back into video memory once it frees up is not implemented.

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

- Octelly/Eglantine `hdrumble-mvp` (`a540ba5`): the RenHDrum integration, "input: HD Rumble and
  DualSense haptics through RenHDrum" by samaBR85 with follow-up commits by hermes (Octelly),
  GPL-3.0-or-later. Ported by hand into `renhdrum_pad.{h,cpp}`, `configure_hd_rumble.{h,cpp}`,
  the SDL driver and the HID service; see [HD rumble on Xbox-style pads](#hd-rumble-on-xbox-style-pads).
- Three-Dola's RenHDrum (`trzz1214/RenHDrum` v0.5.0, MIT OR Apache-2.0): the library itself. Its
  header is vendored in `externals/renhdrum/`; its prebuilt `renhdrum.dll` ships in the release
  packages, with notices in `third_party_notices/RenHDrum.txt`.

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

Open-Frame-Gen is opt-in and only runs with eligible Vulkan storage, sampling and compute support.
It pauses for non-SDR output and falls back to real frames if its resources cannot be prepared.
Interpolation can add latency or artifacts, especially around overlays, camera cuts or uneven
frame times. Frame-generation activity logs count **queued** frames, not display measurements.
Use game FPS and game-frame time to evaluate emulation performance. See the
[validation record](performance/1.7-validation.md) for device test status and model provenance.

## High-resolution game artwork

The game list keeps using local custom artwork, cached images, embedded icons and a placeholder
when no usable artwork is available. Existing sidecars under `<CacheDir>/artwork/<base_title_id>.png`
are read offline (lower-resolution or invalid sidecars can fall back to container artwork).
Replacing a sidecar does not modify a game container or the user's `load/<title_id>/icon.*` override.
The cache stores PNG and small versioned JSON metadata separately. A selected local image is kept
until replaced or removed; no automatic expiry triggers a network refresh. An over-quota write is
refused rather than evicting other artwork.

No background scraping or title-ID uploads are performed. The SteamGridDB API integration is
**disabled in standard builds** pending provider clearance and live verification. Validation-only
builds can compile it with `-DVESHIRU_EXPERIMENTAL_ONLINE_ARTWORK=ON`. The user must explicitly
review/edit a single title and supply their own API key (held in memory, not saved by this build).
Only the API host receives the key; the asset CDN receives a credential-free PNG request. Nothing
is replaced until the user confirms a preview. Timeouts, cancellation, invalid responses and rate
limits leave existing artwork alone. See the provider's [terms](https://www.steamgriddb.com/terms)
and [privacy policy](https://www.steamgriddb.com/privacy). No online assets are bundled or licensed
for redistribution by Veshiru.

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

## HD rumble on Xbox-style pads

**Status: [unverified on hardware].** Nobody on this project has felt this on a controller. What
is verified is listed at the end of this section; everything else here describes what the code
does, not how it feels.

The **HD Rumble...** button in the vibration settings (**Configure > Controls > Vibration >
Configure**) opens the settings. The feature is **off by
default** (`hd_rumble_enabled`). While it is off, or when `renhdrum.dll` is not next to
`veshiru.exe`, every controller is played by exactly the code it was played by before 2.2.0.

Where it comes from: the integration is ported from the `hdrumble-mvp` branch of Octelly's
Eglantine fork (samaBR85, hermes), and the engine is Three-Dola's RenHDrum, a separate library
loaded at run time. The release packages include its prebuilt v0.5.0 `renhdrum.dll`; the release
workflow pins the file's SHA-256 and fails if the download differs. The library is looked for in
two places only: the path in the `RENHDRUM_LIBRARY` environment variable, then the executable's
folder. It is a third-party binary that this project did not build.

### What an Xbox-style pad can and cannot play

A Switch controller has a linear resonant actuator: the game sets a frequency and an amplitude
for a low band (about 160 Hz) and a high band (about 320 Hz), every few milliseconds. An
Xbox-style pad (Xbox One/Series, Xbox 360, and third-party pads such as the PowerA wired
controller) has two eccentric rotating masses: a heavy weight in the left grip and a light one in
the right. Such a motor has one input, its drive level, and that sets speed and force together.

- **Frequency cannot be reproduced.** There is no way to ask the motor for 160 Hz at half
  strength. The mapping uses the *band* (low to the heavy motor, high to the light one) and
  otherwise only lets a band's frequency weaken it slightly, by at most 15 % one octave above
  its nominal frequency. Effects that differ only in pitch feel the same.
- **Short effects are blurred.** A weight needs tens of milliseconds to spin up and to stop. A
  5 ms tick from a Switch 2 game arrives as a soft bump or not at all. "Hit accents" (a 25 ms
  overdrive when a vibration rises suddenly) is a partial workaround, not a fix.
- **Weak vibrations may not turn the motor at all.** Below some drive level the weight does not
  move. The mapping applies a curve (exponent 0.6) to lift weak values; the right amount depends
  on the pad, which is what "Weak rumble boost" and "Motor strength" are for.
- **Left and right are not independent.** Both halves of the emulated controller land on the same
  two motors; each motor follows the stronger half.
- **Trigger motors** exist only on some pads and are reachable only on some Windows input paths.
  With SDL's default XInput path a pad has its two grip motors and nothing else. They are off
  unless "Also drive the trigger motors" is ticked *and* the pad reports them.

### Two modes

| Mode | Where the mapping runs | What is tested |
| --- | --- | --- |
| **Direct** (default) | `src/input_common/helpers/xbox_hd_rumble.{h,cpp}` | Unit tests, see below |
| **RenHDrum mix** | inside `renhdrum.dll` | Nothing in this repository can test it |

Both modes write to the pad through the same scheduler: a write happens when the target changed
by more than 1 % of the range (at most every 8 ms), immediately for a stop, and every 400 ms
while a vibration is held. Every write lasts at most one second, so if a stop is lost the motors
stop on their own.

### When the motors are stopped

Zeros are written, for controllers HD rumble is playing, when: the game sends a stop; emulation
is paused; emulation stops or the window is closed; the controller disconnects; and when the
master switch, the Xbox switch, the mode or the trigger switch is changed in the dialog (picked
up within about 100 ms, without waiting for the next vibration).

`veshiru-cmd` does not run the timer that refreshes held vibrations, so there a held Direct-mode
vibration ends after one second and the RenHDrum mix does not play. Use the Qt frontend for HD
rumble.

### Log lines

- On connect, once per pad, at Info: `HD rumble: Xbox-style pad VVVV:PPPP "name" through SDL
  XInput (grip motors: yes, trigger motors: not exposed on this path, wired: ...)`. This names
  the input path SDL opened the pad through and whether trigger motors are available. It is only
  printed when the library is loaded.
- `RenHDrum: renhdrum.dll not found, using the standard rumble` at Info when the library is
  missing.
- `HD rumble: the controller refused N rumble writes in a row: ...` at Warning, at most once
  every 10 seconds per pad.

### Settings

All are in the `Controls` group of the configuration file. No existing setting was renamed;
`enable_dualsense_haptics` (Veshiru's own DualSense haptics) is untouched and separate from
`renhdrum_dualsense_enabled`.

| Key | Default | Meaning |
| --- | --- | --- |
| `hd_rumble_enabled` | `false` | Master switch |
| `hd_rumble_gain` | `100` | Percent (25-1000) applied to the game's amplitudes |
| `hd_rumble_low_boost` | `0` | 0-100, lifts weak vibrations more than strong ones |
| `xbox_rumble_enabled` | `true` | HD rumble on Xbox-style pads (needs the master switch) |
| `xbox_rumble_mode` | `1` | `1` Direct, `0` RenHDrum mix |
| `xbox_motor_strength` | `100` | Percent (0-1000) applied after the mapping |
| `xbox_detail_mix` | `50` | Direct: share of the high band also played on the heavy motor |
| `xbox_hit_accents` | `50` | Direct: overdrive on a sudden rise, `0` disables it |
| `xbox_trigger_rumble` | `false` | Drive trigger motors when the pad reports them |
| `xbox_trigger_mix` | `20` | Direct: share of the high band played on the triggers |
| `renhdrum_dualsense_enabled` | `false` | RenHDrum's DualSense haptics instead of Veshiru's |
| `hd_rumble_usb_interval_ms`, `hd_rumble_envelope_5ms`, `hd_rumble_frequency_in_packet`, `hd_rumble_crisp_notes`, `hd_rumble_bluetooth_level`, `hd_rumble_bt_*` | upstream's | Pro Controller / Joy-Con opened through SDL |

### What is verified, and what is not

- **Verified by unit tests** (`src/tests/input_common/xbox_hd_rumble.cpp`, tag
  `[xbox_hd_rumble]`): the Direct mapping stays within the motor range, is monotonic in
  amplitude and strength, sends silence for zero, negative and NaN input, routes the bands to
  the right motors, and combines the two halves; the scheduler always passes a stop through,
  holds back small and too-frequent changes, refreshes a held vibration, retries a refused
  write, and plays and ends a hit accent.
- **Verified by CI**: the code builds with clang-cl and the tests pass. That is all CI shows.
- **Not verified**: anything felt on a controller, including the PowerA wired Xbox One pad this
  was written for; which input path that pad takes (XInput is expected with the default
  settings); the RenHDrum mix; the Pro Controller, Joy-Con and DualSense paths in Veshiru; the
  curve constants (0.6 exponent, 15 % frequency weight, 25 ms accent), which are reasoned, not
  measured.

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
