# Changelog

All notable changes to Veshiru are documented here. Each release's notes are taken from its
`## [X.Y.Z]` section (release candidates use the section of the version they are a candidate for).
New entries go under `## [Unreleased]`; `tools/ci/bump-version.sh` turns them into a version section.

## [Unreleased]

## [1.7.0] - 2026-11-18

### Native Windows diagnostics

- Replaced the Qt imitation terminal and logo with the standard Windows console. The Qt UI stays
  available. Normal output is white, warnings are yellow, and errors are red on a Windows
  console; redirects receive plain UTF-8. Added `--console`, `--no-console`, `--log-level` and
  `--log-file`, bounded asynchronous logging, and orderly Ctrl+C/Ctrl+Break handling.
- **Save Diagnostic Log…** replaces the export from the old console window. Logs retain build,
  host and configured-renderer diagnostics and redact common host paths and credential fields.

### Performance and frame generation

- Shallow library discovery no longer enumerates irrelevant assets inside immediate game folders.
  In the synthetic 110-title fixture, discovery median was **10.70–10.96 ms → 0.65–0.68 ms**,
  with identical candidate and directory-watch counts. This is not startup or gameplay FPS evidence.
- Open-Frame-Gen checks Vulkan device and image capabilities, pauses outside SDR, and falls back to
  game frames after preparation failures. Optional extra frames don't wait for presentation fences.
  Telemetry separates game FPS, queued generation, and original/generated Vulkan presents; the
  latter are not display scanout measurements. Display pacing and input latency still require a
  hardware pass. Packaged notices are included for the verified Nano weights.

### Artwork and upgrading

- Library discovery and startup no longer download artwork. Existing HD sidecars, custom artwork
  and container icons remain usable offline; online lookup is disabled in standard builds pending
  provider validation. No new API keys are saved to the profile.
- Keep a profile backup before upgrading. No save or game-content migration is required by these
  changes. To roll back, use the previous verified build and the unchanged profile (or its backup).
  The previous build may resume its original artwork-network behavior and console UI.
- [1.7 validation and release gate](docs/performance/1.7-validation.md): download-only CI packages
  can be checked without creating a release or tag. Windows and matched hardware validation remain
  release gates.

## [1.6.0] - 2026-11-17

### Ryujinx-Inspired Startup Console

- **Authentic terminal console at launch**: Veshiru now opens a dedicated startup console window
  automatically when the emulator launches, recreating the simple console experience of
  Ryujinx/Ryubing.
  - Pure black background with traditional CMD-style white, yellow and red text on a clean
    monospaced terminal surface (no gradients, borders or decorative styling).
  - Large, custom Veshiru ASCII logo printed at startup, followed by a compact system summary
    (CPU, cores/threads, OS, RAM/swap, Vulkan GPU) and a small version/build identifier
    (`Veshiru v1.6.0 | build | commit | date`).
  - Timestamped startup logs streamed live from the existing `Common::Log` live-console backend:
    white for INFO/SUCCESS, yellow for WARN, red for ERROR.
  - The console stays open throughout the application's runtime; closing it only hides it while
    log capture keeps running, and it can be reopened from the View > Console menu.
  - Reuses the existing logging and startup infrastructure — the live console backend is now
    refcounted so the diagnostic console window and the startup console can share it without one
    closing stopping the other.

### Game List Artwork & Ocarina of Time Online Lookup

- **Fixed Ocarina of Time artwork refresh and online lookup (regression from 1.5)**:
  - Dump-filename and platform-tagged queries such as `The Legend of Zelda Ocarina of Time Switch 2`
    now resolve to the canonical `The Legend of Zelda: Ocarina of Time` title (trademark symbols
    stripped, `Switch 2`/`NX2` platform tags dropped, series+subtitle catalog reconstruction), so the
    keyless Libretro/Wikipedia fallbacks receive the exact boxart name.
  - Added aliases for the full colon-less Zelda/OOT slugs and the `Nintendo Welcome Tour` form.
  - Wikipedia image hosts (`thumb.wikimedia.org`) are now allowlisted so REST-summary thumbnails
    resolve as well as `originalimage`.
  - nlib Title-ID icon URLs advertised over plain HTTP are upgraded to HTTPS so they pass through
    the validated, CA-checked request path.
  - Successful artwork resolutions are logged (`Artwork lookup ... resolved from <source>`) so the
    startup console and session log show where each icon came from.

### Tests

- Expanded the `artwork_cache` Catch2 suite with OOT refresh regressions: platform-tagged dump
  filenames, trademark stripping, series+subtitle reconstruction, and the bare-franchise guard.

## [1.5.0] - 2026-11-16

### Frame Generation, Pacing & Present Scheduling

- **Stable `FrameGenPacer` under real-world gameplay variance & hitches**:
  - Replaced single-frame reset triggers with hysteresis-based hitch handling: isolated shader
    compile or streaming hitches (`100–400 ms`) now skip generation for that single frame without
    destroying active generation levels or poisoning exponential backoff; only genuine loading
    stalls (`>= 400 ms`) reset active pacing and clear stale backoff for quick post-load recovery.
  - Raised frame-time variance tolerance (`35%` jitter window vs the previous `15%`), added a
    per-level probe-failure table (`LevelFailures`) so rejecting a higher multiplier (e.g. `2x`)
    no longer drops a healthy lower multiplier (e.g. `1x`), and added a base-FPS collapse guard
    that rejects probes which tank real game FPS while chasing 60 FPS.
  - Added per-frame `FreeFrames()` display-capacity trimming (`GenerationsWithoutWaiting`) and one
    extra spare frame in `PresentManager::SetImageCount()` so generated frames never starve or
    block the next real game frame on 60 Hz FIFO displays.
  - Replaced raw swapchain-acquire throttling with `FramePacer` (`frame_pacer.h`), pacing by real
    game frames and dividing each frame slot evenly across `game_frame_presents` so 30/60/90/120
    FPS pacing targets work accurately alongside frame generation.
  - Updated `FrameGen::LogActivity()` to report separate `game FPS + generated FPS = output FPS`,
    current multiplier limit, hitch skips, and display trims.
- **Open-Frame-Gen zero-allocation warp descriptor caching**: pre-allocated a 16-slot cached warp
  descriptor set table in `OpenFrameGen` keyed by `(prev_slot, destination, destination_view)`,
  eliminating per-frame descriptor allocations and `vkUpdateDescriptorSets` churn during active
  neural optical-flow generation.

### Vulkan Sync, Memory Translation & General FPS

- **Present-fence handoff ordering (`DueComposites`)**: routed all `Renderer::Composite` calls
  through a thread-safe ordered queue (`VideoCommon::DueComposites` + `GpuSplit::OwnsScheduler()`)
  so present fences signalled from service threads always composite on the scheduler owner in
  exact fence order without overlapping or racing command recording (upstream `#1992`/`#1996`).
- **Batched Switch 2 command-list copies & post-sync reads**: gated `SubmitNx2ListCopies()` on
  `!scheduler.DeferFlush()` so compute/command-list scenes batch submissions instead of flushing
  thousands of times per second, and allowed `Nx2ReadListCopy()` to reuse up-to-date copies for
  `draw_written` pages past host sync without forcing `Scheduler::Finish()` (`#2006`/`#2007`).
- **OOM staging recovery without splitting open command buffers**: `StagingBufferPool::ReleaseMemory()`
  now waits only on already-submitted GPU ticks (`CurrentTick() - 1`) without submitting the
  currently recording command buffer, preserving in-progress draw/dispatch bindings (`#1993`/`#1997`).
- **Faster GPU memory translation & submapped range queries**:
  - Added a lock-free per-thread 2048-entry `GpuToCpuAddress` translation cache keyed by
    `map_serial` (with `EDEN_NX2_NO_TRANSLATION_CACHE=1` escape hatch).
  - Stored `device_addr` in `MappingSnapshot::Run` and switched `GetSubmappedRange` /
    `GetSubmappedDeviceRange` to O(log R) binary search over runs (`SubmappedRangeFromRuns`,
    `#1820`), eliminating per-page table walks during buffer/texture binding.
  - *Note on correctness*: Eglantine's `HotPages` (`f27522bfb`) was intentionally not ported
    because unprotecting rewritten guest pages after a single per-frame upload allows subsequent
    mid-frame CPU writes to go untrapped and render stale GPU data.
- **Sampled CPU readback telemetry**: replaced the unconditional 1-in-64 `CPU read of GPU-written
  memory` `LOG_INFO` spam in `GPU::Impl::OnCPURead` with a lock-free atomic 10-second summary
  reporting total reads, sync GPU waits vs async in-flight hits, MiB transferred, and avg/max
  latencies.

### Shader Compilation, Pipeline Cache & 4 GB VRAM Management

- **Native mesh pipeline deduplication (`MeshVertexInputs`, `#2012`)**: lent the first seen
  leftover vertex input state to subsequent native mesh draw keys so unrelated vertex input
  registers no longer split identical mesh pipelines into redundant driver compiles (cutting
  duplicate mesh compiles in titles such as Ocarina of Time and Switch 2 mesh-shader workloads).
- **Safe driver pipeline cache persistence (`#1838`)**: added `PrebuildStartsWithBootBuilds()` so
  drivers that cannot safely read `VkPipelineCache` concurrently with builds (e.g. Intel Windows)
  persist boot and precache driver caches synchronously before starting background prebuilds.
- **4 GB GPU / DXGI budget stabilization & sparse OOM feedback**:
  - Ported `RetentionBudget()` low-water tracking with `16 MiB` upward hysteresis per 30-frame
    check (`#1860`) so Windows DXGI budget fluctuations no longer trigger texture eviction/re-upload
    thrashing.
  - Made `TextureCache` `pressure_margin` proportional on `<= 4 GiB` GPUs (`+256 MiB` step,
    `-64 MiB` relax, `2 GiB` cap vs `+512 MiB` / `-128 MiB` / `3 GiB` on larger GPUs), wired
    `MemoryAllocator::AllocateSparseMemory` OOM failures to increment `pressure_events`, and
    deduplicated `TrimmedHostInfo` via `HostInfoForOffset` to eliminate repeated ASTC trim warnings.

### Game List Artwork & Ocarina of Time Icon Refresh

- **Fixed wrong-game matching and stale icon refreshes**:
  - Hardened `ScoreArtworkCandidate` with a subtitle guard, extra significant candidate word
    penalties, restricted prefix matching, and clean subtitle-complete matching so `"The Legend of
    Zelda: Ocarina of Time"` (`0400486024DEC000`) can never match bare `"The Legend of Zelda"` or
    unrelated Zelda entries, and `"Hyrule Warriors: Age of Imprisonment"` cannot match `"Age of
    Calamity"`.
  - Switched `switchiconshowdown.com` secondary queries from the bare series prefix before `:` to
    the distinctive subtitle after `:` (`ExtractSubtitleQuery`).
  - Added No-Intro naming normalization (`BuildLibretroNameVariants`: `": "` -> `" - "`, leading
    `"The "` -> `", The"`) across Libretro Nintendo Switch, N64, 3DS, and GameCube boxart
    collections, plus a MediaWiki Action API `pageimages` fallback after Wikipedia REST summary.
  - Replaced `cpp-httplib`'s `set_follow_location(true)` with a validated `<= 4`-hop HTTPS
    redirect loop that reloads the bundled OpenSSL CA store (`VESHIRU_BUNDLED_OPENSSL`) on
    cross-host redirects and strips SteamGridDB bearer tokens outside `steamgriddb.com`.
  - Updated `RefreshGameArtwork` to clear cached sidecar PNG/JSON, failure cooldowns, and
    `game_list/<TID>.jpeg` thumbnails before searching, resolve canonical catalog titles, and
    overwrite stale wrong `TitleRole` values (`force_title = true`).
  - Propagated `package.title_id` through `FileSys::Nup::ProbeNupMetadata` into
    `NupEditDialog::FindIcon()` so NUP metadata icon lookups use the exact Program ID.

### Tests

- Expanded Catch2 unit test coverage across `frame_gen_pacer` (14 cases including `FramePacer` and
  `GenerationsWithoutWaiting`), `due_composites` (3 multithreaded ordering cases),
  `mesh_vertex_input` (4 deduplication cases), `eviction_age` (`RetentionBudget`),
  `pipeline_eviction_policy` (`PrebuildStartsWithBootBuilds`), `memory_manager_runs`
  (`GetSubmappedRange` / `GetSubmappedDeviceRange`), `vulkan_buffer_oom` (`ReleaseMemory`), and
  `artwork_cache` (OOT / Hyrule Warriors scoring, subtitle extraction, No-Intro Libretro variants,
  and cache clearing).

## [1.4.0] - 2026-11-16

### Open-Frame-Gen, FSR3 & Frame Generation Pacing

- **Low-VRAM tiled Open-Frame-Gen shaders (2.5x lower activation footprint)**: fused channel-tiled
  workgroups (`open_frame_gen_{pw1x1,head}.comp`) cut 1080p intermediate SSBO allocation to ~106 MiB
  with bit-exact numerical equivalence, keeping neural frame generation comfortable on 4 GB GPUs
  such as the GTX 1650.
- **Zero-idle overhead & smooth pacing**: added idle/at-target bypass, swapchain-level bypass when
  frame generation is inactive, hysteresis and exponential backoff in `FrameGenPacer`, immediate
  `DispatchWork()` scheduling, and `IsReady()` gating so menus and at-target gameplay incur zero
  optical-flow GPU cost.
- **Synchronization, staging memory & HDR correctness**:
  - Fixed Vulkan Sync2 / `Generate()` barriers and `free_mutex` wait synchronization in the present
    manager to prevent frame-gen presentation stalls and data races.
  - Made Open-Frame-Gen's ~352 KiB weights upload staging buffer lazy (allocated only on first
    optical-flow dispatch) and released promptly as soon as its upload tick retires.
  - Added smooth perceptual scRGB soft-compression in `open_frame_gen_prep.comp` for
    `VK_FORMAT_R16G16B16A16_SFLOAT` HDR swapchains so highlights above 80 nits no longer hard-clip
    to `[0, 1]` during motion estimation, while keeping SDR and HDR10 PQ paths unchanged.

### Switch 2 GPU Pipeline, Thread Split, ASTC & 4 GB VRAM Budgeting

- **Parallel multi-stage native shader translation**: raised `MaxFlushedTexturePoolBytes` from
  1 MiB to 8 MiB in `vk_pipeline_cache.cpp` so Switch 2 titles with 4+ MiB texture descriptor pools
  (such as Mario Kart World's 4.20 MiB pool) translate graphics shader stages concurrently on CPU
  worker threads instead of falling back to single-threaded GPU-thread stalls.
- **Three-stage GPU thread split hardening**: reset `wait_target`, `hook_requested`, `back_quiet`,
  and `drain_quiet_requested` cleanly in `GpuSplit::Stop()`, eliminated a mid-batch `Drain()` /
  `RequestIdleHook()` 8192-iteration spin race in `BackLoop()`, and added multi-session Catch2
  synchronization tests.
- **ASTC CPU & GPU decode performance + test suite**: added a 2048-entry block-mode LUT, exact
  integer LDR interpolation (eliminating per-channel `double` divisions), single-partition and
  coarse row-band worker batching (2.2x faster on small mips, bit-exact across 16,384 mixed blocks),
  CPU HDR void-extent support, GPU shader endpoint hoisting in `astc_decoder.comp`, and a dedicated
  `astc_decode.cpp` Catch2 suite covering 4x4 through 12x12 footprints.
- **4 GB GPU (GTX 1650) VRAM budgeting**: lowered the texture cache's dynamic `VK_EXT_memory_budget`
  tracking floor from 2 GiB to 1 GiB (with underflow guards), scaled Switch 2 native buffer-cache GC
  thresholds by VRAM tier (`192/384 MiB` on `<= 4.5 GiB` cards with device-pressure cooperation,
  `320/640 MiB` on 6 GB, `512 MiB / 1 GiB` on 8+ GB), and added a `128 MiB` trimmed ASTC array
  ceiling for `< 5 GiB` GPUs.
- **Upstream Switch 2 rendering & memory updates**: integrated `DrawIndexedIndirect`, native tiling
  unswizzle fixes, `a829b346` / `662af0bf` / `b54d1f06` / `a310a7e9` / PR `#890` video core and
  host memory improvements, and fixed a `constexpr` compilation bug in `textures/decoders.cpp`.

### Game List Artwork, Eglantine Linking, Updater & Packaging Fixes

- **Artwork & NUP icon fixes (1.3 audit Bugs #1, #3, #4)**: fixed stale icon caching, NUP
  container icon loading, and background artwork refresh/retry behavior; ensured `veshiru-cmd.exe`
  builds with the Windows console subsystem.
- **Eglantine linking & first-launch migration safety (Bug #2)**:
  - Removed `ASSERT(uuid)` crash in `GetProfileID()`, guarded `program_id == 0` in `Link to Eglantine`,
    and added automatic Eglantine profile UUID resolution (`profiles.dat` + save directory scan)
    with full UTF-8 / wide-path support for non-ASCII Windows user profiles.
  - Prevented `UserDataMigrator` from throwing uncaught exceptions out of `MainWindow`'s constructor
    and allowed existing interior directory junctions when linking a legacy profile (`Strategy::Link`).
- **Configuration, updater & Windows packaging fixes (Audit B1, B2, B4, B5, B6)**:
  - Removed hardcoded `EDEN_NX2_NUP_CACHE_MIB=1024` from `dist/nx2_switches.txt` so 24+ GiB systems
    auto-scale to 1536 MiB, and added `VESHIRU_*` to the startup switch allowlist.
  - Ensured `veshiru-updater.exe` overrides global `/MD` with `/MT` for standalone execution from
    `%TEMP%`, handles `OpenProcess` `ERROR_ACCESS_DENIED` safely without racing file replacement,
    and falls back to `SHA256SUMS.txt` when GitHub's `asset["digest"]` field is absent.
  - Passed `--plugindir build/release/bin/plugins` to `windeployqt` in `release-binaries.yml` to
    eliminate duplicate root Qt plugin directories.
  - Rewrote `tools/windows/merge-pgo-profiles.ps1` to balance sessions by 64-bit summed per-function
    volume with bounded weights and subsystem coverage reporting.

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
