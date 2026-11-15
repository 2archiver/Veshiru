<!--
# SPDX-FileCopyrightText: Copyright 2026 Veshiru Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
-->

# Upstream audit for Veshiru 1.2.0

This is the audit record for the 1.2.0 release cycle. It lists what was inspected in Eglantine and
Lichii, what was ported into Veshiru, and what was deliberately not ported. A fix is only marked
**integrated** when the corresponding code is present in this tree *and* was built and tested;
anything that could not be built and tested in this environment is marked **deferred**, with the
reason.

## Baselines checked

| Upstream | Reference in the task | State when audited (2026-11-16) | Notes |
| --- | --- | --- | --- |
| Eglantine | `daily-20261109`, `78743f3` | `daily-20261115` at `258b346`; `v0.1.0-rc4` at `cf6cfe9` | Release page still lists `daily-20261115` as the newest daily build. |
| Lichii | build 177, `87d84d5` | build 228 at `be1f4d9`, default branch `5da051e` | Release page still lists `build-228` as the newest release. |
| Veshiru | tag `v1.1.0`, `07fcfa8` | `main` at `81af07d`, version `1.1.0`, no `v1.2.0` tag | No `v1.2.0` tag, draft, release or CI artifact existed, so no published tag was at risk. |

Both upstream trees were cloned in full and compared file-by-file against this tree. The surface is
large (Eglantine shares Veshiru's Eden ancestry but has diverged across ~2,970 files; Lichii differs
across ~590 non-Android files, including 141 in `src/video_core` and 138 in `src/core`). None of
those trees can be merged wholesale into a fork with Veshiru's own identity, Windows-only target,
NUP handling, profile import and console; and the divergence is not reviewable as a single diff.

## Audit matrix

Status values: **integrated** (present, built and tested), **deferred** (wanted, not validated),
**rejected** (not applicable or harmful here).

### Build, CI and packaging

| Upstream change | Files | Problem solved | Strategy | Status |
| --- | --- | --- | --- | --- |
| Eglantine `e79386b19e`, `bb4bac869d`: name CI artifacts and the updater after the identity; expose the release tag | `.github/workflows/*`, `CMakeModules/GenerateSCMRev.cmake` | Artifacts and release notes identified themselves with upstream names | Veshiru already composes its own identity from `VESHIRU_BUILD_TAG`/`VESHIRU_BUILD_PLATFORM`, but its workflows still hard-coded the upstream `eden-cli.exe` output and dead branch names. Ported the naming *intent* only, not Eglantine's files | integrated |
| Eglantine `99702443f9`: fix Windows builds of page-table and preset tests | `src/tests/**` | Windows test build failure | Veshiru's test targets differ; the affected targets are built from Veshiru's own sources and already compile in the Windows clang-cl job | rejected (no Veshiru equivalent) |
| Lichii: Windows zero-page pass, cheaper constant-buffer writes | `src/video_core/**` | Host-side GPU overhead | Not reviewable as an isolated patch; requires the Lichii buffer-cache layout, which Veshiru does not share | deferred |

### Shader, pipeline and Vulkan

| Upstream change | Files | Problem solved | Strategy | Status |
| --- | --- | --- | --- | --- |
| Eglantine `b54d1f064f`: AMD hardware regression fixes | `src/shader_recompiler/**`, `src/video_core/**` | AMD-specific rendering regression | Touches lowering paths whose Veshiru state differs; no AMD target hardware was available to validate | deferred |
| Eglantine `1b61c9544d`: mesh fallback for points, lines and primitive attributes | `src/video_core/renderer_vulkan/**` | Missing geometry in mesh-shader titles | Depends on Eglantine's mesh/task shader warp-op handling, which is not present in Veshiru | deferred |
| Eglantine `5943a945d7`, `5551c9576a`: `vk_pipeline_cache` native-skip conditions | `src/video_core/renderer_vulkan/vk_pipeline_cache.cpp` | Unnecessary native draws, shader build overhead | Requires Eglantine's native-skip machinery; Veshiru's pipeline cache has a different structure | deferred |
| Eglantine `aeab4fc3a2`: don't re-copy a command-list page after a CPU write | `src/video_core/buffer_cache/**` | Redundant GPU-side copies | Veshiru's buffer cache already has its own page-tracking; the patch does not apply as-is | deferred |
| Eglantine `258b346b9a`: fallback textures for damaged NUP members | `src/core/file_sys/**` | Damaged `NUP` members rendered with no texture | Veshiru already has an independent damaged-member fallback (`EDEN_NUP_DAMAGED_FALLBACK`) with tests | rejected (already covered) |

### GPU lifecycle, memory and scheduling

| Upstream change | Files | Problem solved | Strategy | Status |
| --- | --- | --- | --- | --- |
| Lichii build 177: stale GPU channel cleanup, second-game launch, guest-thread suspension at shutdown | `src/core/core.cpp`, `src/video_core/gpu.cpp`, `src/core/hle/kernel/**` | Deadlock/crash when launching a second game in the same process | Veshiru's GPU thread split (`NX2_GPU_THREAD_SPLIT.md`) and channel lifecycle differ substantially; a partial port risks the use-after-free it is meant to fix | deferred |
| Lichii build 177: Mario Kart World GPU loading deadlock | `src/video_core/**`, `src/core/memory/**` | Boot deadlock in MKW | Needs the Lichii sparse-memory reservation layout and real MKW gameplay to validate | deferred |
| Lichii build 177: bounded device-address allocation, GPU range reuse, cache budget caps | `src/video_core/**` | VRAM exhaustion on 4 GB cards | Veshiru already caps the texture cache to the reported device budget and has its own range allocator with tests (`host_mip_limits`, `stale_pages`, `gpu_dirty_memory`) | deferred (partially covered independently) |

### CPU/JIT

| Upstream change | Files | Problem solved | Strategy | Status |
| --- | --- | --- | --- | --- |
| Lichii build 177: faster x86-64 code generation and register allocation, UMONITOR/UMWAIT, MONITORX/MWAITX idle | `src/dynarmic/**`, `src/core/arm/**`, `src/core/cpu_manager.cpp` | Host CPU overhead and idle-core wake-up latency | `src/dynarmic` here is a different revision; the changes are not separable from it. Veshiru has independent JIT and idle instrumentation (`EDEN_NX2_IDLE_STATS`, `jit_codegen_options`, `jit_ordered_loads` tests) | deferred |

### Interface

| Upstream change | Files | Problem solved | Strategy | Status |
| --- | --- | --- | --- | --- |
| Lichii build 160 dark UI refresh: icons, scalable/scrollable settings, game-list metadata, per-game override indicators, redesigned overlay | `dist/qt_themes/**`, `src/veshiru/**` | Interface quality | Used as *design inspiration only*; Veshiru's blue interface, palette, assets and composition are original to this repository. No Lichii artwork, logo or layout was copied | rejected (inspiration only) |

## What this release actually integrates

The 1.2.0 release integrates no upstream gameplay or renderer code. It is a release-repair and
identity release:

1. **Release pipeline repair.** The release workflow's build trigger named a branch that no longer
   exists, so only a manual dispatch could produce a stable release. The build-check workflow named
   a stale feature branch. Both now target `main`.
2. **Executable identity.** `veshiru-cmd` was built as `eden-cli.exe` and the sound-room target as
   `eden-room`, both inherited upstream names. They are now `veshiru-cmd.exe` and `veshiru-room`.
   Every consumer of those paths (release job, installer, shader-sweep tool, documentation) was
   updated with them.
3. **Updater reachability.** The update checker compared only `build-N` tags, so a tagged `v1.2.0`
   release would have been invisible to an installed 1.1. Stable versions are now compared
   numerically and kept on a separate track from build-number pre-releases, with tests.
4. **Veshiru Blue interface.** The default theme was an inherited dark theme with a pink accent.
   It is replaced by an original deep-navy/electric-blue system, including the theme's raster
   assets, the Qt palette, the console colours and the What's New view.
5. **Identity in data.** The log file inside Veshiru's own profile is now `veshiru_log.txt` rather
   than the inherited `eden_log.txt`. Existing logs are left in place.

## Identity items deliberately deferred

- **`EDEN_*` environment variables.** 194 distinct debug/diagnostic variables across 94 source
  files still use the inherited `EDEN_` prefix (`EDEN_NX2_SHADER_DUMP`, `EDEN_NX2_GPU_PROFILE`,
  `EDEN_NX2_DEBUG_SERVER`, and so on). They are the documented debugging contract in `AGENTS.md`
  and `docs/nx2/`, and every tool under `tools/nx2/` passes them by name. A prefix rename is a
  mechanical but wide-reaching change that would break those tools and every existing bug report
  recipe at once, so it is deferred to its own reviewable change with the tools updated together.
  Setting the variables does not change emulator behaviour for a normal user.
- **Internal `Eden*` C++ identifiers.** `EdenPath`, `GetEdenPath`, `EDEN_DIR` and similar are
  compile-time names, invisible outside the source tree, and renames would touch every file that
  resolves a user path. Not worth the review cost in a release-repair change.
- **`dist/eden.bmp`, `dist/eden.ico`, desktop/metainfo files.** These are consumed by non-Windows
  packaging and the CLI window icon. `dist/veshiru.ico` is already the executable icon
  (`src/veshiru/veshiru.rc`) and the installer and shortcuts use `veshiru.exe`; replacing the
  remaining inherited asset filenames belongs with a packaging pass that also updates their
  consumers.

## Tests performed

| Check | Result |
| --- | --- |
| Workflow YAML parses (`yaml.safe_load`) | pass, all three workflows |
| Shell syntax (`bash -n`) on changed scripts | pass |
| `tools/ci/release-notes.sh` end-to-end with a stable tag and stubbed assets | pass; produces the Windows-only table and checksum note |
| Qt stylesheet and resource consistency (every `url(:/...)` resolves in `style.qrc`; brace balance) | pass |
| `.ui` XML well-formedness for changed UI files | pass |
| Updater version/build selection logic | compiled and run standalone against the new tests' cases: stable→stable, build→build, no downgrade, no cross-track offering, numeric ordering, wrong-platform rejection, malformed-tag rejection |
| Theme icon rendering | rendered and inspected; all accent pixels are in the blue palette with no residual pink fringing |
| `codespell` over changed files | no new findings (pre-existing findings unrelated to this change) |

## Tests not performed

- **Windows build of `veshiru` and `veshiru-cmd`.** No Windows toolchain, Qt 6, Vulkan SDK or MSVC
  is available in this environment, and the dependency fetches the configure step needs are not
  reachable. The C++ changes are small and self-contained, but a compile-only check of the full
  target was not possible here. The Windows clang-cl CI job is the first real verification.
- **Running the GUI, the console, or any game.** No gameplay or rendering validation was performed.
- **Windows scaling at 100/125/150/200%.** The stylesheet deliberately uses only spacing, radius and
  colour rules, and the existing configuration pages already scroll, but this was not verified on
  Windows.
- **Archive extraction and launch on a clean Windows 10/11 machine.**
- **Profile import, NUP base/update/DLC cases.** These paths were not modified by this change and
  their existing tests were not run here.
