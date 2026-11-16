# Changelog

All notable changes to Veshiru are documented here. Each release's notes are taken from its
`## [X.Y.Z]` section (release candidates use the section of the version they are a candidate for).
New entries go under `## [Unreleased]`; `tools/ci/bump-version.sh` turns them into a version section.

## [Unreleased]

## [1.3.0] - 2026-11-16

### Frame Generation & Vulkan Rendering

- **Open-Frame-Gen (`Frame Generation > Open-Frame-Gen (Neural Optical Flow)`)**: added a native
  Vulkan compute implementation of the MIT-licensed `Open-Frame-Gen` `NanoModel` optical-flow
  U-Net alongside LS1 (`Lossless Scaling`). Runs six precompiled SPIR-V compute passes with
  embedded BatchNorm-folded FP16 weights (~88K parameters), Squeeze-and-Excitation channel
  attention, bidirectional occlusion-masked warping and RCAS sharpening — no Python, CUDA or
  TensorRT dependencies required.
- **Native asynchronous shader/pipeline compilation & Switch 2 GPU compatibility**: integrated
  upstream Lichii (`#948`–`#1907`) and Eglantine (`65f5650aa9`, `09cc3d1dc9`) Vulkan, Maxwell/Ampere
  shader lowering, `SET_SCISSOR` stride, `RenderTargetState` layout synchronization, `R16G16B16A16_SFLOAT`
  HDR swapchain preservation (Mario Kart World), staging-buffer bounded trimming, sparse/null image
  safety and `ShaderCacheV2` pipeline remapping while preserving Veshiru's 3-stage front/back GPU
  thread split.

### High-Resolution Game List Artwork

- **Persistent HD artwork sidecar cache**: fetched game icons are now stored lossless at full
  resolution (up to 512x512 PNG) under `%APPDATA%\veshiru\cache\game_list\artwork\<title_id>.png`
  instead of only writing 256x256 JPEG overrides, and are rendered crisp on Windows HiDPI displays
  via `devicePixelRatioF()`.
- **Accurate Switch 1 & Switch 2 artwork lookup**: `IconSearch` now resolves Nintendo CDN icons by
  16-hex Title ID first before falling back to an edition-aware scored title matcher (`Nintendo Switch 2 Edition`,
  Deluxe/Demo/Region disambiguation), silently auto-fetches missing icons in the background, and
  adds **Refresh Game Artwork** to the game-list context menu.
- **Full-resolution NUP container icons**: `AppLoader_NUP` now returns raw PNG/JPEG container icon
  bytes before falling back to a 256x256 downscale.

### Performance, NUP Streaming & Host Memory

- **Balanced multi-game Windows PGO profile (`veshiru-windows.profdata`)**: replaced the single-game
  Ocarina of Time profile with a normalized multi-game LLVM 22 `clang-cl` profile spanning Switch 1
  and Switch 2 titles (`77,537` symbols, `23,103` hot functions), added `-DVESHIRU_PGO_GENERATE=<dir>`
  in CMake, and added `tools/windows/merge-pgo-profiles.ps1` for weighted multi-session retraining.
- **NUP streaming & 24+ GiB host RAM tuning**: increased NUP zstd decode chunking from 64 KiB to
  512 KiB, auto-scaled the default host-RAM NUP decompression cache to 1.5 GiB on systems with
  `>= 24 GiB` RAM (overridable via `VESHIRU_NUP_CACHE_MIB`), and deferred guest physical-memory
  zero-page sweeps during active gameplay when host RAM is abundant.

### Security, Updater, Migration & Diagnostics

- **Verified updates & build-tag tracking**: auto-detected `vX.Y.Z` tags in `GenerateSCMRev.cmake`
  when `VESHIRU_BUILD_TAG` is unset, passed `release_tag` into `VESHIRU_BUILD_TAG` on tagged CI
  releases, allowed `build-N` installations to upgrade cleanly to newer `vX.Y.Z` stable releases,
  enforced mandatory SHA-256 verification against `SHA256SUMS.txt` before extracting updates, and
  excluded `veshiru-updater.exe` from PGO runtime instrumentation.
- **Save transfer & profile migration safety**: guarded `MainWindow` configuration/migration/NUP
  actions while emulation is running, switched `ReadSymlinkTarget` on Windows to native
  `FSCTL_GET_REPARSE_POINT`, verified byte-level SHA-256 digests during **Link to Eglantine** save
  transfers (staging unlinks before removing junctions), and remapped imported `ShaderCacheV2`
  archives (`pipeline_v2.bin` / `<title_id>.vkch`) across profile migrations.
- **Console & log reliability**: added case-insensitive Windows/POSIX path and credential token
  redaction, periodic 1-second dirty flushing + crash-path flushing for `veshiru_log.txt`, `.log`
  export filter defaults in **Tools > Console**, startup version/GPU/VRAM logging, NCZ block-header
  decompression bounds checks, and pinned SHA-256 verification across CI tool/dependency downloads.

## [1.2.0] - 2026-11-16

### New blue Veshiru interface

- **Veshiru Blue** is the default interface for new and existing installations. It replaces the
  inherited pink accent with a deep-navy surface, an electric-blue primary accent and cyan
  highlights, and is applied consistently to the main window, menus, dialogs, game list,
  settings, the live console and the What's New view.
- The theme's navigation, checkbox, menu and radio assets were redrawn in the blue palette instead
  of being recoloured at runtime, so icons stay crisp at every Windows scaling level.
- Severity colours in the live console keep their own hue (warning amber, error red) rather than
  using the accent, so errors stay distinguishable from ordinary highlighted text.
- The older themes (Default, Dark, Midnight Blue and their colourful variants) remain available in
  **Configure > General > UI**. Veshiru Blue is selected automatically after the update.

### Windows build and packaging

- The CLI executable is now built and shipped as **`veshiru-cmd.exe`**; it was previously produced
  as `eden-cli.exe`, an inherited upstream name, even though the CMake target was already
  `veshiru-cmd`. The sound-room executable is `veshiru-room` for the same reason.
- The release workflow now builds from and publishes `main`. Its build trigger named branches that
  no longer exist, and the build-check workflow named a stale feature branch, so a stable release
  could not be produced from the default branch without a manual dispatch.
- The release job verifies `veshiru-cmd.exe` is present in the Windows package, and the release
  notes name Veshiru's own executables.
- Windows archives now ship a `SHA256SUMS.txt` covering every archive.
- Release notes generated by `tools/ci/release-notes.sh` describe Veshiru's Windows-only assets
  instead of the inherited Eglantine Linux/macOS tables.

### Updater

- A stable release (`vX.Y.Z`) can now be offered to an installed stable release. The update checker
  previously compared only `build-N` tags, so a tagged release was invisible to an installed 1.1.
  Build-number pre-releases and stable releases are still never offered across lines, versions are
  compared numerically (`1.10.0` is newer than `1.9.0`), and release candidates are never offered.

### Identity and data

- Log files are now written as `veshiru_log.txt` inside `%APPDATA%\veshiru\log`. Existing log
  files are left in place; nothing reads the old name automatically.
- Version, About dialog, CLI `--version`, archive metadata and release notes all report 1.2.0 from
  the single `VERSION` file.

### Added

- **Save transfer to Eglantine**: right-click a game and choose **Link to Eglantine** to share
  save data between Veshiru and Eglantine (both emulators now reference the same save files).
  The transfer copies and verifies the save data before linking, keeps a timestamped
  `*.veshiru-backup-*` copy of the replaced saves instead of deleting them, and every failure
  is shown to the user with the exact step and error instead of failing silently. Portable
  Eglantine installations are supported via folder selection; the legacy `eden` data folder
  used by Eglantine build 17 is detected automatically.

### Changed

- **Console appearance**: the in-app console now uses a dark background with larger, bright
  high-contrast text (similar to Ryubing's console), making log output much easier to read.
- "Link to Ryujinx" has been removed and replaced by "Link to Eglantine".

### Fixed

- Save transfer no longer fails silently: filesystem errors, missing saves and link failures
  are now reported with a visible message and written to the log.
- Windows junction creation no longer shells out to `cmd /c mklink /J`; it uses a native
  `FSCTL_SET_REPARSE_POINT` call, which is reliable inside a GUI application.
- Windows junction detection now uses the reparse point attribute, so existing linked saves
  are recognised correctly (boost reports junctions as ordinary directories).
- The console no longer shows a Critical "assert" (and no soft assert fires) the first time a
  Switch 2 title calls an unimplemented `nx2` service command that Veshiru auto-stubs; it is
  logged as a warning with the usual report file instead.

### Known limitations

- Windows builds and the Windows-only test suite in this change were not executed in the
  environment that prepared it: it has no Windows toolchain, Qt 6 or Vulkan SDK. The changes are
  limited to the release scripts, the interface stylesheet and assets, the updater and the log
  filename; the unit tests added for the updater were compiled and run standalone, not through the
  project's test target.
- No gameplay testing was performed for this release.

## [1.1.0] - 2026-11-13

### Performance

- General FPS performance refinements.
- Improved frame pacing and gameplay consistency.
- Ongoing CPU and GPU overhead optimisations.
- Smoother performance in demanding gameplay scenarios.
- General responsiveness improvements.

### Stability

- General stability refinements.
- Improved reliability during extended gameplay sessions.
- Minor fixes and internal optimisations.

### Compatibility

- Continued refinement of game compatibility.
- General improvements to rendering and emulation behaviour.

### Notes

- Actual FPS improvements vary depending on hardware, graphics drivers, emulator settings and
  individual games. Performance gains are not guaranteed across all titles.
- Veshiru is now a Windows-exclusive emulator: Linux and macOS packages are no longer built or
  published, and the documentation and CI reflect Windows-only support.
- New wave-style application icon.

## [0.1.0]

- Initial release candidate
