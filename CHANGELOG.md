# Changelog

All notable changes to Veshiru are documented here, in engineering voice: what changed, why, and what
has been verified. Each release's page shows the short, user-facing notes in
`docs/release-notes/<X.Y.Z>.md` when that file exists, and this file's `## [X.Y.Z]` section
otherwise (release candidates use the section of the version they are a candidate for). New entries
go under `## [Unreleased]`; `tools/ci/bump-version.sh` turns them into a version section.

## [Unreleased]

## [2.9.0] - 2026-11-23

### Fixed

- **VTX-HLE: UI layers the guest wrote into descriptor pools now resolve.** The SDK's bulk
  descriptor table writes (`nvnTexturePoolWriteDescriptorTableData`, which carries no driver
  function id: the bootstrap loader returns NULL and the SDK's own code runs) left Tegra texture
  headers in pool guest memory and nothing in the recorder's registration table, so every draw
  sampling those ids bound the transparent dummy and whole UI layers vanished (Mario Kart World's
  character select: ~69 ids a frame in the unsupported TSV,
  `descriptor fallback (graphics): texture handle names no texture`). `Recorder` now implements
  `PoolReconciler` (`video_core/nvn/pool_reconcile.h`, published through `Sink::SetPoolReconciler`
  with the guest-memory pointer's lifecycle): when a draw's handle names no texture, the replay
  asks the recorder, which reads the pool entry at `base + id * sizeof(TICEntry)` (and at double
  stride for padded pools) and resolves the storage the header names by guest address against the
  live textures (`textures_by_address`, width/height checked, plain texture over view). A hit
  registers the id in the descriptor table so later submissions carry it; counters
  `pool_table_reconciled` / `pool_table_unresolved` join the recorder rows of the unsupported TSV.
  See the NVN `DESIGN.md`, "Guest-written descriptor tables".

### Changed

- **Optimized defaults for everyone (2.9).** New global defaults, the first-run profile
  (`frontend_common/config.cpp`) and the recommended-settings table
  (`common/recommended_settings.h`) now agree on the set the 2.7/2.8 measurements on the
  i7-10750H + GTX 1650 4 GB reference machine came from: `graphics_path` = VTX-HLE (NVN) for
  Switch 2 titles, `use_asynchronous_shaders`, `async_presentation`, `nvn_pipeline_library` and
  `build_all_cached_shaders` on, `native_async_draws` = Bounded, `renderer_force_max_clock` on
  (a downclock between frames is a hitch), `precache_shaders` = Ask so the load screen builds the
  driver cache once. Existing configs keep their values; `Use recommended settings` and brand-new
  profiles pick the new set up.
- **Version** bumped to 2.9.0; release notes in `docs/release-notes/2.9.0.md`.

### Investigated

- **Veshiru 1.3's Mario Kart World work, re-checked for 2.9.** 1.3 brought the R16G16B16A16_SFLOAT
  HDR swapchain preservation for MKW and the Eglantine MKW fixes (`#870` legacy GPU timestamps,
  `#871` lighting sparse atlas) into the emulated path; 2.9 verifies the VTX-HLE path keeps their
  equivalents (HDR present tone-mapping, sparse/tiled storages) and that the MKW stability knobs
  (`mkw_30fps`, `mkw_dynamic_fps`) stay off in every optimized profile.
- **Eglantine `daily-20261122`, Lichii and Orchid releases** surveyed for 2.9 capability ideas
  (async pipeline translation workers, graphics-server pool sizing, host page-out ordering); the
  NVN replay's own counters guided what landed here.

## [2.8.0] - 2026-11-23

### Added

- **VTX-HLE (NVN) high-level graphics path for Switch 2 titles.** Ported from
  [Eglantine](https://github.com/Djihads80/Eglantine) (`810f44f5` + PR `#948` through `93b06ee7`),
  bringing VTX-HLE by Vortex727 (`https://github.com/Vortex727/FUKKO`) and FUKKO by Vortex727
  (`https://github.com/Vortex727/FUKKO`), with bridge design informed by `nxnative` by seedds
  (MIT), ShiftX2 fixes by Duck912, and DirectXTex BC7 encoding (MIT):
  - **NVN HLE bridge & command recording** (`src/core/hle/nvn/`, `src/video_core/nvn/`,
    `src/core/memory/page_tracker_state.*`): intercepts Switch 2 NVN2 calls (`functions_*`,
    `hle_runtime`, pools, buffers, textures, samplers, programs, command buffers, queues, syncs,
    events, windows, native image, and DLSS thunks), records command buffers into compact record
    streams, and tracks guest CPU writes / lazy GPU write-back pages only while VTX-HLE is active.
  - **Vulkan NVN replay & shader cache** (`src/video_core/renderer_vulkan/nvn/`,
    `src/video_core/host_shaders/nvn_*.comp`, `src/video_core/shader_log.*`): replays recorded
    NVN command lists on Vulkan with Ampere shader lowering, raw global memory access via buffer
    device addresses, smart BC stand-ins for ASTC textures (`bc_encoder.comp`, `bc7_encoder.comp`,
    `nvn_bc_copy.comp`), disk pipeline cache v18 (`nvn_pipelines.*.bin` and `nvn_pipelines.vkcache`)
    with automatic transition cleanup (`ClearShaderCachesForNvn`), and periodic shader refresh mode.
  - **Per-game & global Graphics Path settings** (`src/common/settings*`,
    `src/common/vtx_hle_settings.*`, `src/veshiru/configuration/vtx_hle_grey.*`,
    `configure_graphics.*`, `configure_per_game.cpp`): adds **Graphics → Graphics path**
    (`Emulated GPU` default, `VTX-HLE (NVN)` opt-in, plus `EDEN_NX2_NVN_HLE=1/0`), VTX-HLE toggles
    (`nvn_in_pass_queries`, `nvn_pipeline_library`, `nvn_sync_diet`, `nvn_shader_refresh`,
    `nvn_smart_bc`), and greys out Emulated-GPU-only controls while VTX-HLE is selected.
  - **NVN unit tests & notices** (`src/tests/nvn/`, `vtx-hle/NOTICE`,
    `third_party_notices/VTX-HLE.txt`).
- **Pre-NVN stability & Veshiru parity fixes.** Preserved Veshiru's existing features alongside the
  port (`EGLANTINE_STREAMLINE` / NVIDIA DLSS, FSR3/FSR4, `mesh_as_compute`, `Nx2FastDraw`,
  `DownloadLanding`, `Nx2Spikes`, `VulkanPipelineCacheVersion = 54`, Ocarina of Time stability
  profile, zero-page trim deferral, atomic game-list refresh, `union Result`, and `IsBlockLinearSizeCompatible`).

## [2.7.0] - 2026-11-22

### Added

- **Ocarina of Time runs a per-title stability profile on cards that cannot build pipelines
  mid-scene.** The 2026-11-22 logs from the reference machine (i7-10750H, GTX 1650 4 GB, 32 GiB)
  show three things happening in the same first visit to a busy area: 56 pipelines built on demand
  in one 10 s window with 5.8 s of draws held for them and single frames of 0.1-18.6 s; compute
  dispatches skipped for the frame because their pipeline was still building (a per-frame billboard
  drawn where it was the previous frame, which is what the fairy's up-and-down flicker is); and
  30-353 CPU reads of GPU-written memory per 10 s at 5-7 ms average, 60-140 ms worst. `Core::
  GameSettings` now names the title (`WantsStabilityProfile`) and, where the user's own
  configuration does not already provide each part, builds the whole recorded pipeline cache while
  the game loads and keeps it built (`Settings::BuildAllCachedShaders`, fed by a new session flag so
  nothing is written to disk), forces reactive flushing, and keeps the GPU engines' inner writes
  invalidating the CPU view. `EDEN_NX2_OOT_PROFILE=0` turns the whole profile off for one session so
  the same machine can A/B it, and one log line names what was turned on. `[verified: unit tests for
  the title list and every combination of the profile rule, including the session switch; the
  three-log analysis in docs/performance/2.7-oot-i7-10750h-gtx1650.md comes from the attached
  logs; unverified: that the flicker and the freezes are gone on hardware]`
- **Images that go over the memory budget are counted instead of truncated.** The allocator logged
  the first 32 over-budget image allocations per process and then stopped; all three 2026-11-22
  Ocarina logs stop at exactly 32, so none of them says how many images really went over, and the
  same five shapes filled every line. It now logs one warning per distinct shape (up to 64) and a
  running total with the shape count every 256 allocations. `[verified: the shape key packs width,
  height, depth, mip levels, layers and format without overlapping bits, and the cap keeps the
  logging bounded; unverified: the effect on a capture, which needs a session on hardware]`
- **The `nx2:vi2 last 600 presents` line counts the freezes separately from the slow frames.** A
  window of 600 presents read `10 fps` in three of the 2026-11-22 logs while 95% of its frames were
  16.7 ms, because 30 of them were over 100 ms and three of those lasted 1.2-18.6 s. The average is
  not a frame rate when a handful of frames last seconds, so the line now reports `N over 1 s`
  beside `N over 100 ms`. `[verified: the counts are taken from the same per-frame interval the
  existing over-100 ms count uses; the logs it was built from contain 486 ms-18.6 s frames;
  unverified: nothing, the format change is additive]`

### Changed

- **The all-zero guest page trim no longer runs while the game is presenting a frame.** A pass
  reads the whole resident guest working set under VAD locks: the 2026-11-22 logs have it taking
  1700 ms and 2149 ms in the middle of Ocarina's gameplay, which the player sees as a freeze. The
  saving is optional, so `Core::ShouldTrimZeroPages` now holds the pass back while the guest is
  presenting and lets it run when the game pauses, loads or sits in a menu; under real host memory
  pressure it still runs, once per escape interval instead of every 5-30 s (120 s on a 32 GiB host,
  20 s on a 16 GiB one). One log line explains the first deferral. `[verified: unit tests over the
  whole truth table of the predicate; unverified: the effect on a session on hardware]`
- **Release pages carry short, user-facing notes.** `tools/ci/release-description.sh <version>`
  prints `docs/release-notes/<version>.md` when it exists and the CHANGELOG section otherwise
  (release candidates use the notes of the release they are a candidate for); the release workflow
  and `release-notes.sh` both take it from there, and the Windows-only note, the download table and
  a link to `CHANGELOG.md` follow. The 2.7.0 notes describe the Ocarina of Time changes in the terms
  a player uses (freezes, flicker, second-long pauses), say what the release costs (a longer first
  load) and what to do if something still stutters, in a "What's better" / "Good to know" shape
  rather than a list of defects. `docs/release-notes/README.md` says how to write them.
  `[verified: the script and the workflow's notes assembly were run for 2.7.0 and for a version
  without notes (2.6.0); unverified: the rendered release page, which only a publish run produces]`
- **Ocarina of Time on the reference machine: analysis and settings** in
  `docs/performance/2.7-oot-i7-10750h-gtx1650.md`, covering all three 2026-11-22 logs (frame-gen
  off, frame-gen on, and FPS Unlock off), what the freezes and the flicker come from, what 2.7
  changes and what it deliberately does not, and where the release stands against Fritchii's
  4-8 GB fixes (the retention it describes is 2.5/2.6 work of this tree's own; nothing here is
  ported from it, and the one credited port is the skipped empty flush from an earlier round).

## [2.6.0] - 2026-11-22

### Changed

- **Star Fox (Switch 2) and Donkey Kong Bananza run with the guest clock at Normal for the session
  when "CPU Clock" is inherited from the global profile as Boost.** `Core::Timing::GetClockTicks`
  derives the guest tick counter from the host wall clock in multicore and multiplies it by
  1734/1020 for Boost, while kernel sleeps and GPU timestamps stay at wall time, so those two
  titles saw guest time run 1.7x fast: a 60 Hz scene presented about 39 times a second, and Bananza
  measured its own CPU frames over budget and fell back to swap interval 2 (30 FPS) with vblank
  headroom left. `Core::GameSettings` now names the titles that pace themselves by that clock
  (`NeedsGuestClockPacing`) and holds their clock at Normal when the inherited clock is Boost
  (`HoldsClockAtNormal`); a clock chosen for the game itself, Overclock included, is left alone and
  nothing on disk is rewritten. Adapted from Orchid 0.12.0's "keep Normal CPU clock for Donkey Kong
  Bananza and Star Fox" (MythosOfs/Orchid PR #180, Eglantine lineage), which measured Star Fox
  39.19 → 58.22 FPS and Bananza 57.75 → 59.74 mean FPS with its 30-FPS drops going 16 → 0 on an
  RX 9070 XT. `[verified: unit tests for the title list and every combination of the hold rule;
  unverified: frame rates on the target machine]`
- **`Common::RangeMap`'s `Map`/`Unmap` walk the tree twice instead of four times.** The map behind
  `Tegra::MemoryManager`'s PTE-kind tracking found the two surrounding values with its own
  `lower_bound`/`upper_bound` and then searched again for the erase range. It now keeps the two
  iterators it already has, erases the range between them and inserts with a hint. Adapted from
  Orchid `f4edd9c8` / Eglantine `c6dd1d0214` (goof787lord), with Orchid's randomized flat-model
  test. `[verified: standalone host benchmark (g++ 12.2.0, -O2), 3,000,000 random Map calls, three
  runs each: 2.5's header 1.014-1.068 s (median 342 ns per Map), 2.6's 0.637-0.681 s (median
  214 ns per Map, 37% less), identical final-state digest in both builds; a 40-seed x 2000-operation
  randomized check against a flat model passes for both headers; unverified: the effect inside a
  game]`
- **Deep Current is a step brighter and bluer.** Every token moved: app background `#07111F` →
  `#0A1526`, surface `#0B1B2D` → `#0E2140`, elevated `#10253B` → `#14304F`, border `#1D3B58` →
  `#274C74`, primary blue `#258BFF` → `#3D9BFF`, bright blue `#3BA7FF` → `#5FBEFF`, cyan
  `#42D9F5` → `#5FE1FA`, with hover/pressed/selection, the text ramp and faint lifted to match
  (`dist/qt_themes/veshiru/style.qss`, `VeshiruPalette()`, both link colours, and the token sheet
  that is the record). Filled accent surfaces — the selected menu row, the default button in all
  three states and the item-view highlight — take dark navy text (`#06121F`) because white on
  `#3D9BFF` is only 2.9:1; contrast computed from the committed colours: text 16.75:1, muted
  5.30:1, faint 5.50:1 (3.68:1 in 2.5), accent 6.38:1 on the window, selected-row 6.58:1. The
  HOME-applet news artwork was regenerated with the same gradient. `[verified: contrast ratios
  computed from the exact hex values and the stylesheet's compositing; unverified: how it looks on
  a real panel]`
- **The interface applies its theme stylesheet once.** `MainWindow::UpdateUITheme` read the theme
  file through a `QTextStream` twice; the second `readAll()` returns nothing, and that empty result
  was set as the *window's* stylesheet, making Qt re-resolve the style of every child for no visual
  change. The sheet is now read once and set on the application only. `[verified: the call path and
  Qt's handling of an empty widget stylesheet; unverified: measured restyle time on Windows]`

### Added

- **`docs/performance/2.6-performance-and-ui.md`** records the measurement, the palette token
  table, the contrast table, what was audited and deliberately not changed (on-demand pipeline
  compilation, vertex-input dynamic state, storage-binding reuse, texture-cache limits) and how to
  check the clock hold on the NH77DB.
- **`docs/2.6-arena-agent-patch.txt`** is a plain-text patch manifest: the exact edits, the
  verification run for each, and the limits of what was proved.
- **A randomized Range Map test** (`src/tests/common/range_map.cpp`) compares 2000 random maps
  against a flat model at every address, and **`src/tests/core/game_settings.cpp`** covers the
  paced-title list and the hold rule.

### Known issues / not addressed in 2.6

- **No FPS number is claimed for this build on the reference machine.** The only measured frame
  rates quoted above are Orchid's, from its own hardware and build. Run the checklist in
  `docs/performance/2.5-operator-checklist.md` for anything stronger.
- **On-demand pipeline compilation still dominates Kakariko's stutter** on the 4 GB GTX 1650
  (`docs/performance/2.5-oot-switch2-gtx1650.md`). 2.6 does not change pipeline scheduling: this
  tree already builds deferred pipelines on worker threads with priority promotion, and the
  reference projects' pipeline ownership work describes what it already does.
- **Boost remains the wrong clock for those two titles**, and the hold applies only when the clock
  is inherited; a per-game Boost is still honoured and still costs the frame rate.

### Verified

- Host `g++ 12.2.0` standalone harnesses for the Range Map change: the 3-run benchmark table above,
  the digest that proves both headers end in the same map, and the 40-seed randomized model check.
- Palette contrast arithmetic from the committed hex values, and the regenerated news JPEGs (15427
  and 4687 bytes) rendered and inspected.
- A source grep that the old palette hexes no longer appear in the theme or the palette code.
- **Windows clang-cl production build and the full test suite, on this release's head:** CI run
  [44984651181](https://github.com/2archiver/Veshiru/actions/runs/44984651181) passed configure,
  `cmake --build build` with `ENABLE_WERROR=ON`, the console-subsystem and import checks, and
  `ctest --test-dir build --output-on-failure --no-tests=error`, which includes the new Range Map
  and game-settings cases.

### Unverified

- Every frame-rate claim for Veshiru itself, and the clock hold in a running game: no title was
  booted for 2.6.
- How the palette looks on a panel, and the restyle saving in a running GUI.

## [2.5.0] - 2026-11-22

### Changed

- **Synchronize Core Speed is on by default.** `Core.sync_core_speed` now defaults to `true` in
  the built-in value, the first-run profile that the configuration seeder writes, and the
  recommended-settings table, so a fresh profile never runs the guest clock ahead of the speed
  limit; the change applies to fresh installs and to "Use recommended settings". Existing
  configurations keep their stored values; nothing on disk is rewritten. The `first-run config`
  test now expects the new default. `[verified: settings/recommended/seed/test all changed
  together; unverified: hardware run — CI is the build gate for this branch]`

### Fixed

- **The HOME Menu news applet loads the Veshiru icon.** The news feed built its `list_image` and
  `main_image` binaries from `eden-emu.dev`-hosted files that Veshiru cannot rely on, so the
  news area rendered without an icon (or with another project's). The bundled default images
  are now embedded JPEGs (800x450 and 320x180, rendered from `dist/branding/veshiru_icon.png`;
  `tools/ci/embed-news-images.py` regenerates `news_images.h`), the per-release image download
  is gone, and the feed works offline. `<news_id>_{small,large}.jpg` files dropped into the
  news images cache directory are still honoured ahead of the bundled artwork. The msgpack
  `topic_id` is `veshiru`, and the topic list the applet queries advertises `veshiru` as well
  (previously `eden`). `[verified: the embedded assets and encoder output were inspected and
  are well-formed JPEGs; unverified: HOME Menu rendering on hardware]`

- **The `veshiru-cmd` loader error message told users to visit "the Eden wiki" and "the Eden
  discord", and the GitHub issue template offered an "Eden Discord" contact link.** Both now
  point at the Veshiru repository and its issue tracker. (The remaining `EDEN_*` environment
  variables and `EdenPath` directory names are the fork's on-disk and IPC-compatible names;
  renaming them is a migration, not a string change. The Discord IPC environment variable names
  stay as-is.) `[verified: grep for user-facing "Eden" strings in the launch path and
  .github comes back clean; the Discord icon URL host and the news topic id are already
  Veshiru]`

### Added

- **Ocarina of Time (Switch 2) performance notes for the GTX 1650.**
  `docs/performance/2.5-oot-switch2-gtx1650.md` reads all five session logs from the 2026-11-22
  testing (title 0400486024DEC000 v2.0.0 on a Metabox NH77DB: i7-10750H, GTX 1650 4 GB,
  31.8 GiB RAM), quantifies the Kakariko walk-through (up to `frame 106381 us`, 212 of 600
  presents over 100 ms, up to 7.8 s of draws held per 10 s window, one 39.4 s frame on first
  entry), and gives the per-game settings that matter on this card (lower the texture cache
  memory limit below 3935, `precache_shaders` Ask/Always, frame generation only above ~30 real
  FPS). It also records what this machine cannot do, so expectations stay honest. No performance
  renderer change ships in 2.5.0. `[verified: every number quoted in the document comes from the
  five attached logs; not verified: a post-change session on hardware]`
- **Pipeline timeline tracing (`EDEN_NX2_PIPELINE_BUILD_TRACE=<file>`) gains three markers.**
  Boot-time pipeline loads now log `SUBMIT` (a record queued to the job scheduler), and each
  deferred SPIR-V emission logs `EMIT-BEG`/`EMIT-END`, so a cold load can be attributed end to end
  with the existing driver `BEGIN`/`END` (vkCreate*Pipelines) lines: queue → emission → driver
  create call → usable. No behaviour change when the variable is unset.
- **Benchmark contract and operator checklist** (`docs/performance/2.5-benchmark-contract.md`,
  `docs/performance/2.5-operator-checklist.md`). The contract defines matched before/after
  captures (≥ 3 per scenario × cache condition, same machine/firmware/title/save/route/settings/
  driver/AC/output, frame-gen off, sync on, separate cache dirs), frame metrics (mean real FPS,
  p50/p99/p99.9 intervals, hitch counts > 100 ms and > 250 ms), and the [2.5.0] Kakariko gate
  (mean ≥ 10 real FPS **and** p99 ≤ 100 ms on the 60 s route). No release claim may be quoted
  without running it.

### Known issues / not addressed in 2.5.0

- **Discord Rich Presence still displays "Eden" as the application name.** The bundled client ID
  is the Eden-lineage community application; Discord renders the registered application name and
  no client-side payload can override it. 2.5.0 keeps it so presence works out of the box, and
  documents the swap to a Veshiru-owned application in `src/qt_common/discord/discord_application.h`
  (create the application at discord.com/developers/applications, replace
  `VESHIRU_DISCORD_APPLICATION_ID`, update the `discord-<id>` cleanup key in
  `dist/installer.nsi`). Everything the client sends is already Veshiru ("Playing <game>",
  "Veshiru <version>", "large_text": "Veshiru", icon from this repo); only the application name
  is not. Users who do not want the mislabelled entry can turn it off under Configure → Web →
  "Show current game in Discord (Rich Presence)".
- **Kakariko stays heavy on a cold shader cache on 4 GB cards.** 2.5.0 ships the analysis and
  the recommended per-game settings, not a renderer fix; see the notes document.
- **The one 39.4 s frame** (Kakariko first entry, 78 pipelines built in 10 s, 21 s of draws
  held) is documented as an investigation, not yet diagnosed end to end.
- **`precache_shaders` stays `Off` in the recommended table** (right for a cold cache and for
  boot time); turning it on per game is a valid choice once the cache is warm.
- **No performance numbers are claimed for 2.5.0.** The renderer paths, the Kakariko real-FPS
  gate and every VRAM figure are documented, but no before/after capture exists yet on the target
  machine; 2.5.0 ships the fixes, the analysis and the measurement tooling. Hardware verification
  belongs to `docs/performance/2.5-operator-checklist.md` and is the release blocker it documents.

## [2.4.1] - 2026-11-21

Ports six self-contained stability and hot-path fixes from Lichii `nx2` `3165970a`, with the
upstream commit named in each file. No frame-generation behaviour changed.

### Fixed

- **`UniqueFunction` owns a callable given as an lvalue.** It previously stored `Functor` with a
  forwarding reference, so a callable passed by name as `const` was kept as a reference;
  `QueueVulkanPipelineCacheSave` queues `const auto save = ...`, and the serialization worker ran
  that lambda after its stack frame returned. It now stores `std::decay_t<Functor>`, constrains
  construction and calls through `std::invoke`. Ported from Lichii `249d889b`.
  `[verified: host g++ harness covers const/non-const lvalues, std::ref, member pointers and the
  construction constraints; unverified: the original crash on hardware]`
- **Shader `TrackAddend` no longer recurses without a depth limit.** A long `IAdd32` chain could
  overflow the host stack while translating a shader; it now uses an explicit stack with the same
  visit order. Ported from Lichii `19dcffea`.
  `[verified: same visit order and ambiguity rule by inspection; unverified: a title with the long
  chain]`
- **A failed `ScratchBuffer::resize_destructive` leaves a consistent buffer.** The old order
  raised the capacity before allocating, so a failed growth left a capacity with no storage. It
  now releases first; a failure leaves an empty, reusable buffer. Ported from Lichii `64ea088a`
  (the ScratchBuffer part only). `[verified: host g++ failure-injection harness and a new ctest
  case; unverified: memory pressure in a game]`
- **Sparse texture segments with no device translation are skipped, not asserted.** A segment can
  lose its translation between the submapped-range query and the lookup. The first eight are
  logged, the rest are skipped, and iteration continues. Ported from Lichii `4a558454` (originally
  Eglantine `19dfa543`). `[unverified: a title that hits the race]`
- **Unmapped guest writes no longer flood the log.** `Write`, `WriteExclusive` and
  `WriteExclusive128` now log the first 1000 and then one per 100000, as unmapped reads already
  did. A guest spinning on a bad pointer could otherwise write a ~100 MB log and slow the run.
  Ported from Lichii `528dae4b`. `[verified: code path by inspection; unverified: the pathological
  guest]`

### Changed

- **The GPU snapshot copier parks its waiter.** After 4096 spins, `WaitFor` now waits on the
  completion atomic and the copier notifies after each job, instead of yielding in a loop. Ported
  from Lichii `4a3dbf64`. `[unverified: frame-time effect on the GTX 1650]`

### Verified

- Windows clang-cl production build and ctest: CI run
  [44892557878](https://github.com/2archiver/Veshiru/actions/runs/44892557878) passed configure,
  `cmake --build build` with `ENABLE_WERROR=ON`, both executable checks and
  `ctest --test-dir build --output-on-failure --no-tests=error` on commit `5838714`.
- Host `g++ -std=c++20` harnesses for `UniqueFunction` (const/non-const lvalues, `std::ref`,
  member pointers, construction constraints) and `ScratchBuffer` (failure injection and reuse).

### Not ported

- Lichii's 4 MiB device-buffer blocks: Veshiru already uses 8 MiB; the extra step is unmeasured.
- Spill-collect GC change: Veshiru has no host-spill collect path.
- Texel-buffer clamp, GPU-split parking and GPFIFO inline headers: deferred; reasons and exact
  commits are in [the comparison](docs/2.4-lichii-comparison.md).

## [2.4.0] - 2026-11-21

Versioned List, Discord Rich Presence, crash dumps, the HOME Menu news applet, notification icons
and the About wordmark. The Windows clang-cl production build and ctest passed on CI (run
44887537553, commit 4dc37f0, `-DENABLE_WERROR=ON`). GUI rendering and live Discord/crash behaviour
still need a desktop run, so those entries are marked unverified.

### Added

- **Versioned List rebuilt from the active theme palette.** The 2.3 view painted its rows from
  `QPalette::Base`, the near-black `#07111F` app background, which read as a black panel inside
  the dark blue window. The new delegate takes every colour from `QPalette` (row surface
  `QPalette::Mid` = `#132D46` on Deep Current), enforces WCAG AA at paint time, and adds a version
  pill, dimmed title ID, file size, play time, hover/selected accent bar and a marked `Missing`
  row that keeps the old path in its model tooltip. No raw background or text hex value remains in
  the mode. `[verified: contrast computed from the shipped palette — title 12.66:1, secondary
  7.68:1, badge 5.37:1 after the runtime adjustment, selected title 8.95:1, row surface 1.34:1
  against the window; unverified: rendering in a running GUI]`
- **Discord Rich Presence ships its own application ID.** The user-facing Application ID field is
  gone; presence is on by default behind one checkbox in Configure → Web. The build handshakes
  with the constant in `src/qt_common/discord/discord_application.h`, sends "Playing <game>",
  the resolved game version, elapsed time and the Veshiru icon, and stays at Debug when Discord
  is missing. `[verified: payload built and asserted by a host g++ harness, including the assets
  object, escaping and the digits-only ID; unverified: live Discord on a desktop]`
- **Notification-area icon and toasts use the embedded Veshiru icon.** A tray icon is created from
  `:/img/veshiru.ico` (never a relative path), the application window icon is set at startup, and
  update-available, install-finished, profile-import-finished, game-started (only while the
  window is inactive) and Qt critical-error toasts all pass that icon explicitly. `[unverified: tray/toast appearance on a
  real desktop]`

### Changed

- **Crash dumps removed.** `eden_crash.dmp`, the Breakpad integration and the `crash_dumps`
  directory are gone; the Windows handler still logs the exception code, faulting module+offset,
  thread name and stack as one Critical line, written and flushed directly by
  `TryWriteCrash`. Ordinary logging and the sanitised Export diagnostics zip are unchanged.
  `[verified: no source, CMake or installer reference to eden_crash/breakpad/CrashDumpsDir remains; the crash line
  path is unchanged and flushes per line; unverified: forced crash on Windows]`
- **HOME Menu news applet opens GitHub.** "Open Forgejo" → "Open GitHub", "View more on GitHub",
  the link allow-list now accepts `github.com`, the news author/topic is Veshiru, and the update
  dialog plus the remaining user-facing Eden-Forgejo links point at
  `https://github.com/2archiver/Veshiru`. `.forgejo` CI is untouched.
  `[unverified: the in-emulator news applet]`
- **No performance or frame-generation code changed in 2.4.** Source comparison against Lichii
  `nx2` `b198feb` shows the reduced-resolution flow pass, hitch skip, pacer and VRAM cap already
  present and more developed than Lichii's. Without a GPU to measure on, changing them would be a
  guess. See [the 2.4 comparison](docs/2.4-lichii-comparison.md). `[source-level only]`

### Fixed

- **The About dialog showed the Lichii wordmark.** `dist/veshiru_wordmark.png` and its 2x variant
  were the upstream project's artwork; both are regenerated as a Veshiru wordmark in the Deep
  Current accent colours. `[verified: asset inspected at 1x and 2x on dark and light backgrounds;
  unverified: the dialog in a running GUI]`

### Verified

- Windows clang-cl production build and tests: CI run
  [44887537553](https://github.com/2archiver/Veshiru/actions/runs/44887537553) passed configure,
  `cmake --build build` with `ENABLE_WERROR=ON`, the console-subsystem and import checks, and
  `ctest --test-dir build --output-on-failure --no-tests=error`. The exact configure line is in
  `.github/workflows/build-check.yml`.
- Discord payload/application-ID logic: standalone host `g++ -std=c++20` build of
  `discord_payload.cpp` with all assertions passing (`ALL CHECKS PASSED (0)`).
- Contrast values above, computed from the exact `VeshiruPalette()` colours and the delegate's
  compositing/adjustment rules.
- A source/CMake grep for `eden_crash`, `breakpad`, `CrashDumpsDir` and
  `VESHIRU_CRASH_DUMPS` finds no hits outside this changelog.
- The regenerated wordmark rendered on `#07111F` and `#F3F6FA` and inspected.

### Unverified

- GUI rendering of the Versioned List, the tray icon/toasts and the About dialog: CI proves it
  compiles and the tests pass, not how it looks.
- Live Discord presence with the client open and closed.
- A forced host crash on Windows writing no `.dmp` while still flushing the log line.
- The HOME Menu news applet's new button and link.

## [2.3.0] - 2026-11-21

Every entry carries a tag. `[verified: <how>]` means it was run, or a CI job proved it; a green
CI means the code builds and the tests pass, not that a game or Discord behaves. `[unverified]`
means nobody has observed it. The Verified and Unverified sections below summarize.

### Fixed

- **The Windows production build, broken on main since the 2.3 library merge.** `game_list.cpp`
  called `isRowHidden` (removed with the view change) and used `Veshiru` without its include;
  both compile errors are repaired, and CI compiles the Discord RPC TU (`USE_DISCORD_PRESENCE=ON`)
  so a broken include cannot ship silently again. `[verified: Windows clang-cl build in CI]`

### Added

- **Discord Rich Presence over a local IPC client (Windows).** Presence is opt-in, uses a
  user-supplied Discord Application ID (none is bundled), and only sends the game title, version
  and elapsed-time timestamp plus the required local IPC metadata. The worker owns one persistent
  pipe connection with overlapped I/O, a 15-second send rate limit, and reconnect backoff of 1 to
  60 seconds with a bounded shutdown. Artwork lookups and the third-party RPC library were
  removed. The activity name is derived from the build version, and the payload escaping is
  covered by unit tests. `[verified: payload/version unit tests; unverified: live Discord behavior
  on a real desktop]`
- **Versioned List game-library mode.** A new compact, keyboard/controller-friendly card list shows
  cover art, title, content type, play time and the installed game version at a glance. The
  version comes from one shared resolver (`GameListVersion::Resolve`: enabled update > control
  metadata > display version > "1.0.0" only when a base NACP exists); an unknown version hides
  the badge, and Discord reports the same version. `[verified: resolver unit tests; unverified:
  badge rendering in a running GUI]`

### Changed

- **Vulkan storage-binding cache reads only stable GPU mapping generations.** Cache hits are
  bypassed while a page-table update is in progress, so a service thread cannot reuse a binding
  against a partially updated map, and a lockstep test pins that every completed page-table
  operation (map, sparse map, unmap, remap) advances the generation. Adapted from Orchid's
  `buffer_cache: reuse storage bindings while GPU mappings are stable` change. `[verified:
  mapping-generation and lockstep unit tests; unverified: gameplay effect on target hardware]`
- **Game-directory updates are debounced by 30 ms.** Watcher bursts of both kinds (game
  directories and external content) are tracked independently and coalesce into one asynchronous
  rescan instead of repeatedly clearing and rebuilding the library. Manual and settings-driven
  refreshes remain immediate. `[verified: coalescer unit tests; unverified: under an OS
  filesystem watcher]`
- **The library refresh is incremental.** Rows are cached in memory for the session keyed by
  canonical path, file size and modification time, so a refresh re-parses only new or modified
  files. Play time is re-read when a row is rebuilt, boot support never depends on the cache, and
  the cache is cleared when keys or firmware are installed. Every refresh logs
  `library refresh: N entries, X reused, Y parsed, Z ms (reason: ...)`. `[verified: cache unit
  tests; unverified: large-library timing on target hardware]`
- **Diagnostic logs redact private identifiers**, including emulated console serials, device and
  account names, network-interface labels, configured endpoints, controller/audio identifiers, IP
  addresses, and MAC addresses. Address redaction is a single-pass scanner that no longer
  mistakes Vulkan/driver version numbers for IPv4 addresses (the old regexes redacted
  `Vulkan 1.3.280` and `551.86.0.0`). `[verified: 18-case unit test and a 100k-line benchmark
  printed in the CI log (locally 1578 ms regexes vs 46 ms scanner)]`
- **A built-in performance summary** for honest FPS measurement: Debug > Performance Summary
  (off / 10 s / 30 s), Ctrl+F10 for an on-demand window, or `VESHIRU_PERF_SUMMARY=<seconds>`.
  One fixed-format Info line per window with native FPS, nearest-rank frame-time percentiles,
  1%/0.1% lows, slow-frame counts, synchronous-readback waits, pipeline builds and generated
  presents. Off by default; a single branch when off. `[verified: format definitions and flag
  plumbing compile in CI; unverified: measured on target hardware]`

### Known issues

- Discord Rich Presence requires a user-supplied Application ID in Settings and is Windows-only in
  this build; it is off by default.
- The Windows production build of PR #22 was broken (`isRowHidden`, missing include); this release
  restores it. Main carried the broken commit, so 2.3.0 is the first release whose artifacts are
  expected to build since 2.2.0.

### Verified
- Unit tests run by the CI test job on the release head: Discord payload construction, escaping
  and version derivation (`src/tests/common/discord_payload.cpp`, 8 cases); watcher refresh
  coalescing (`src/tests/qt_common/refresh_coalescer.cpp`, 2 cases); the game-list version
  resolver (`src/tests/qt_common/game_version.cpp`, 5 cases); address redaction
  (`src/tests/common/address_redaction.cpp`, 18 cases plus a printed benchmark); the
  mapping-generation lockstep (`src/tests/video_core/memory_manager_runs.cpp`); the parse cache
  (`src/tests/qt_common/game_list_parse_cache.cpp`, 3 cases).
- The Windows clang-cl production build and the full test suite on the exact release commit.

### Unverified
- Live Discord behavior on a real desktop (no Application ID was supplied; none is bundled).
- GUI behavior of the Versioned List badge, the Debug > Performance Summary menu and Ctrl+F10.
- Watcher-driven refreshes under a real OS filesystem watcher, and large-library refresh timing.
- Any gameplay or FPS effect: no target-hardware measurement exists for this release.

### Off by default
- Discord Rich Presence (opt-in; requires a user-supplied Application ID in Settings).
- The periodic performance summary (off / 10 s / 30 s; Ctrl+F10 and VESHIRU_PERF_SUMMARY also
  available). When off it costs one atomic load per present.
- No experimental performance change from the 2.3 review is enabled in this release.

### Performance status

- The supplied 2.2.0 log shows the native game rate varying well below 30 FPS in gameplay even while
  frame generation queues extra presents. Its frame and GPU-readback diagnostics point to runtime
  workload bottlenecks, not a proven single fix. No general emulator-core optimization is claimed
  or a 30 FPS result promised without a repeatable scene benchmark. See
  [the 2.3 performance review](docs/2.3-performance-review.md).

## [2.2.0] - 2026-11-20

Every entry carries a tag. `[verified: <how>]` means it was run, or a CI job proved it; a green
CI means the code builds and the tests pass, not that a game or a controller behaves. `[unverified]`
means nobody has observed it.

### Added

- **HD rumble (off by default), for Xbox-style pads, Pro Controller / Joy-Con and DualSense.**
  The **HD Rumble...** button in the vibration settings opens it. Ported from Octelly's Eglantine
  `hdrumble-mvp` branch (samaBR85, hermes) and played by Three-Dola's RenHDrum v0.5.0, whose
  prebuilt `renhdrum.dll` now ships in the Windows packages (third-party binary, SHA-256 pinned
  in the release workflow, notices in `third_party_notices/RenHDrum.txt`). With the switch off,
  or without the DLL, rumble is played by the same code as in 2.1.0. Details and limits:
  [HD rumble on Xbox-style pads](https://github.com/2archiver/Veshiru/blob/v2.2.0/docs/Veshiru.md#hd-rumble-on-xbox-style-pads).
  - Xbox-style pads get a "Direct" mode mapped by Veshiru itself (low band to the heavy motor,
    high band to the light one) and RenHDrum's own mix as the alternative.
    `[verified: 14 unit test cases for the Direct mapping and the write scheduler, run by CI]`
    `[unverified on hardware]`: nobody has felt it on a controller, the PowerA wired Xbox One pad
    included. Eccentric-mass motors cannot reproduce HD rumble frequencies or very short ticks.
  - Motors are zeroed on pause, stop, exit, disconnect and when the feature is switched off, and
    every write expires after one second. `[verified: scheduler unit tests]` for the stop and the
    expiry logic; `[unverified]` on a real pad.
  - One Info log line per Xbox-style pad names the SDL input path and whether trigger motors are
    available; refused rumble writes are logged at most once every 10 seconds. `[unverified]`
  - RenHDrum mix, Pro Controller / Joy-Con and DualSense paths. `[unverified]`: the mapping is
    inside the DLL and none of these controllers was available.

### Changed

- **Mario Kart World, sparse texture tiles outside video memory are now reported.** When
  device-local memory runs out and tiles go to a non-device-local heap (the 2.1.0 behaviour,
  unchanged), the log gets a warning at most once every 10 seconds with the tiles and MiB placed
  there since the last report, the tiles no heap had room for, and the session totals; the totals
  are logged once more at shutdown. Counters only: no claim about frame rate is made, because
  none was measured. `[verified: 4 unit test cases for the counting and the rate limit, run by
  CI]` `[unverified]` in a game: nobody has seen the line in a Mario Kart World log yet.

- **`aoc:u` PrepareAddOnContent says whether the add-on exists.** It still always reports
  success, exactly as the stub did, so no title behaves differently. The log now has an Info line
  when the index names an add-on that ListAddOnContent lists for the calling title, and a warning
  with the number of listed add-ons when it does not (or when add-ons are disabled for the
  title). `[verified: 3 unit test cases for the index check, run by CI]` `[unverified]` in a
  game: no title that calls it was run.

- **Shader builds that a draw is waiting for run at normal thread priority (Windows).** Shader
  workers run below normal priority so that background builds give way to the game. A build was
  already moved to the front of the queue once a draw waited for it, but one that had started
  kept the low priority while the GPU thread was blocked on it. The worker running such a build
  is now raised until that stage ends; builds nothing has waited for are untouched. Taken from
  an owner's Donkey Kong Bananza log on a GTX 1650 laptop, where draws were held 48.5 s out of
  833 s for pipelines still building on a first run with an empty shader cache.
  `[verified: 6 unit test cases for when a worker is raised and when that ends, run by CI; the
  scheduler's 31 cases also pass under ThreadSanitizer in a Linux build]` `[unverified]` in a
  game: no frame time was measured before or after, and the log cannot say how much of the
  waiting was priority.
  It does not change the frame rate once shaders are built: that log's 14 FPS in gameplay is
  time on the GPU command threads, which this release does not address.
  [What the log shows](https://github.com/2archiver/Veshiru/blob/v2.2.0/docs/bananza-2.2-log-review.md).

### Build and packaging

- **The Windows packages bundle a third-party binary, `renhdrum.dll` (RenHDrum v0.5.0).** The
  release workflow that builds the packages fails unless the downloaded file has the SHA-256
  pinned in the workflow, is an x86-64 PE, exports the 18 functions Veshiru resolves by name, and
  is byte-for-byte the copy that ends up next to `veshiru.exe`. `[verified: those checks are
  steps of the workflow run that produced the packages of this release]` `[unverified]`: the
  library was never loaded or called by anyone working on this release.
- Every GitHub Action in the three workflows is pinned to a commit. `[verified: grep; the
  workflows ran]`

### Documentation

- The "`main.npdm` is parsed rather than assumed" entry was listed under 1.5.0, 1.6.0 and 1.8.0
  as well as 2.0.0. The fix first shipped in 2.0.0; the other three entries are removed.
  `[verified: commit range]`
- `docs/CI.md` said `build-check` runs the `tools/nx2` scripts. It does not, and the page now
  says so. `[verified: the workflow file]`

### Known gaps in this release

- The written brief for 2.2.0 was lost while the work was in progress and could not be recovered
  from the repository. Its sections 3 and 7, part of 6 and 8, and its release gates G1 to G6 are
  unknown, so nothing was implemented for them and nothing here claims they were met. The owner
  chose to release without them, against a substitute list of six checks.
  [The 2.2.0 report](https://github.com/2archiver/Veshiru/blob/v2.2.0/docs/2.2-report.md)
  lists what is known, what is not, and those checks.
- Nothing in this release was tried on a controller or in a game, and no performance figure
  was measured for it.

## [2.1.0] - 2026-11-20

### Fixed (source-level verification)

- Zero or absent DLC metadata no longer produces a zero AOC base. Use a matching carried
  add-on's namespace or the conventional application DLC base; preserve non-zero metadata.
  Populate the base in synthesized NUP control data too. Pin its NACP offset at `0x3070`.

### Changed (UNVERIFIED - needs runtime test)

- Donkey Kong Bananza: log the returned base and the actual carried/requested IDs, and return
  FS TargetNotFound for missing data storage. **Reaching the menu is unverified.**
- Mario Kart World: `MemoryAllocator::AllocateSparseMemory` tries a compatible non-local type
  within budget after device-local exhaustion, before rolling back a sparse tile upload. The
  GTX 1650 log recorded 213 failed tile allocations. No same-scene FPS measurement yet.

### Known issues and credits

- Guest software can still abort on a missing required add-on. GPU paging and readback stalls
  are still possible; no changes to frame generation, shader cache, JIT, or PGO profiles.
- Thanks to Orchid, Lichii (both forks) and Eglantine PR #877 for DLC metadata precedents.
  See [the investigation and retest plan](docs/2.1-investigation.md) for evidence and limitations.

## [2.0.0] - 2026-11-20

### Memory: one VRAM policy instead of three fighting margins

- **Removed the 2,048 MiB "keep free" behaviour**: Veshiru 1.x stacked three margins under the
  driver budget (a base share, a reaction to every allocation that missed the budget, and a reserve
  for the allocator's block space). They reacted to each other: a miss raised the margin by
  256–512 MiB, the raised margin made the next allocations miss, and on a 4 GB card the margin
  reached its 2 GiB cap, so the cache held half the card free and re-decoded textures every frame
  (`Out of device memory: keeping 2048 MiB more free` once per few hundred milliseconds in the
  supplied log). A single `Common::VramPolicy` now sizes the reserve from the driver budget alone:
  an eighth of it, at most 1 GiB — about 387 MiB on the GTX 1650's ~3 GiB budget, so the texture
  cache may fill ~2,709 MiB instead of ~1,734 MiB. The reserve settles instead of ratcheting up.
- **Removed the dead allocator-pressure feedback**: `runtime.MemoryPressureEvents()`, the
  `pressure_events` counter behind it and the unused `pressure_margin`, `pressure_quiet_frames`,
  `slack_margin` and `seen_pressure_events` state are gone. An allocation that still misses the
  budget is served by the allocator's existing relax/try-fallback path, which did not need the
  margin loop.
- **Small images no longer count as VRAM pressure**: every `is over the memory budget; allocating
  it anyway` line in the supplied Mario Kart log was the same 64x64, 7-level depth image — 21 KiB.
  The Vulkan allocator's own budget estimate refused it, which counted as pressure and made the
  texture cache evict recently used images. Images that cannot exceed 4 MiB now skip the budget
  check and fail only if the allocation really fails.
- **Removed the second hidden cap**: a discrete GPU's budget was capped again at 6 GiB plus 1 GiB
  per resolution scale step unless *VRAM Usage Mode* was *Aggressive*. On a 12 GB card that left
  over 4 GiB unused while the cache evicted to stay under the cap. The driver budget less the
  reserve is now the whole answer.

### Frame delivery: fewer hitches while playing

- **NVIDIA pipeline-compile workers**: the driver stage of the pipeline build
  (`vkCreateGraphicsPipelines`/`vkCreateComputePipelines`) took 201 ms at the median in the supplied
  log with six workers on a 12-thread laptop. NVIDIA's driver serialises that work behind an
  internal lock, so the extra workers only queued while holding their shaders' memory. NVIDIA now
  runs two driver workers; other vendors keep the previous scaling. The previously dead
  `has_broken_parallel_compiling` flag is now set for Intel's Windows driver, which shares the lock
  too and keeps one worker.
- **NUP decompression batching**: `DecodeTo` released and re-acquired the member's lock once per
  512 KiB chunk, so one sequential read of a big asset made dozens of turns: the log shows
  `Archive.min.bntx.zs` (38.6 MiB) taking 514 ms, `CmnTex_Tera_Product_Proc_Nrm.bntx.zs` 228 ms and
  `main` 193 ms on the thread that asked for the bytes. A read now decodes up to 16 chunks per turn,
  which is five turns for that 38.6 MiB file instead of 77, while still yielding to another reader.

### Profile applet: no more stale black frame

- **Compositor presents when the visible layer set changes**: `HardwareComposer::ComposeLocked` only
  presented when a producer handed over a new buffer. A layer that just became visible (the profile
  applet over Mario Kart World) or went away left the previous composite on screen until the next
  buffer arrived — the reported two-second black frame. The composer now also presents when the set
  of layers that should be on screen differs from what the frame can show, without consuming extra
  producer buffers.

### Game list: a file that goes missing says so

- **Moved or deleted game files are reported**: a title whose file moved or was deleted simply
  stopped appearing, with nothing to distinguish it from never having been added — the reported
  "I deleted my MKW `.nup` and the game vanished". The worker keeps the paths the last scan listed
  and, on the next scan, lists a **Missing** row for each one now absent from a directory still
  being scanned, with its old location and a note that saves live in the NAND save data, not in the
  game file. Removing a folder from the game list stays silent, since that is not a game going
  missing.

### Settings

- **Memory knobs behind "Show memory expert options"**: *VRAM Usage Mode*, *VRAM Reserve* and
  *Texture Cache Memory Limit* leave the normal Advanced tab and appear only when that box is
  ticked (or when a per-game/global override already exists, so an existing choice stays editable).
  Veshiru sizes this automatically from the GPU's driver budget; showing the knobs invited users to
  cap the cache at 2 GiB and stutter.

### Boot

- **`main.npdm` is parsed rather than assumed**: the NCA loader read the eight bytes at offset
  0x28 as a heap size without checking the `META` magic, and the supplied HOME Menu log shows a
  file where that read produced `0x7463656c` — the ASCII bytes of a text field — as a 1.8 GiB heap,
  which then sized the pointer buffer for a heap that does not exist. The heap size is now read
  only from a real NPDM, and anything unreasonable falls back to the default pointer buffer.

### Tests

- Added `src/tests/common/vram_policy.cpp`, covering the reserve, the critical threshold, and the
  4 GB case the report described (the cache keeps more than 2 GiB of a ~3 GiB budget).

## [1.9.0] - 2026-11-18

### The Legend of Zelda: Ocarina of Time & 4–8 GiB GPU Performance

- **Play Report (`prepo`) command `10500` support**: Implemented `SaveReport` command `10500` on
  `prepo` (eliminating fatal guest IPC panic `1A80A` during Ocarina of Time sessions) and added
  domain request error naming in `hle_ipc` for clearer diagnostics.
- **Anti-gloom async pipeline & compute protection**: Stopped asynchronous draw and compute skip
  policies from dropping unbuilt storage-image-writing graphics draws or image-only tonemapping,
  auto-exposure, and lighting compute dispatches, preventing temporarily dark or gloomy scene
  transitions while shaders compile in the background.
- **4 GiB VRAM ASTC→BCn chunked transcoding & staging safety**: Split large multi-layer GPU
  ASTC→BCn transcodes into VRAM-tiered 32/48/64 MiB layer chunks with post-decode in-flight
  throttling (fixing cold-cache staging buffer reuse corruption and mid-batch pipeline stalls),
  automatic queue-drain retry on transient `VK_ERROR_OUT_OF_DEVICE_MEMORY`, and inter-chunk staging
  flushes on sub-5 GiB GPUs, while keeping dedicated transient RGBA8 scratch allocations from
  falsely triggering persistent texture pressure eviction.
- **Unified-RAM / iGPU budget scaling (no more 2,048 MiB UMA & 4 GiB caps)**: Sized integrated and
  unified-memory GPUs across both BIOS UMA carveouts and shared system GPU heaps scaled to total
  unified RAM (plus full dedicated BIOS UMA utilization on handhelds and 24–128 GiB APUs), unlocked
  the *Texture Cache Memory Limit* UI spinbox up to the full unified memory ceiling instead of
  clamping at 512/2,048/4,096 MiB, and expanded live iGPU budgets cleanly when an explicit cache
  limit is configured while preserving host commit-room protection.
- **`v1.8.0` frame-time & streaming regression fixes**: Replaced four per-frame full-LRU texture
  walks (`DeleteSupersededImages`, `RetireStaleTargets`, and `BadOverlap` GC passes) with an O(1)
  dense overlapping-image index, tracked in-flight background `vkFreeMemory` bytes per heap so
  async frees never trigger phantom VRAM eviction loops or false budget pressure, eliminated
  duplicate `VkImage` creation in render-target allocation and raised device-buffer pool blocks to
  8 MiB, gated Windows `PrefaultGuestMemory` with a `K32QueryWorkingSetEx` residency check and
  zero-allocation contiguous iteration, switched Windows `RealVfsFilesystem` read hits to shared
  locking with `FILE_SKIP_SET_EVENT_ON_HANDLE`, short-circuited capped speculative readback walks,
  and stopped shared-cache JIT invalidations of uncompiled pages from halting running CPU cores.
- **Transient render-target & texture cache optimizations**: Skipped redundant guest uploads for
  new single-level 2D render targets completely covered by prior GPU-written images, added a fast
  path with per-frame LRU deduplication in `PrepareImage`, cached graphics descriptor table host
  page pointers with 64-bit word comparison, and enabled depth/stencil-write render-pass tracking
  to avoid unneeded feedback-loop transitions.
- **DMA pushbuffer, staging pool & cyclic readback speedups**: Restricted per-entry pushbuffer
  writer sync waits to titles requiring them, skipped unnecessary dirty checks when record reads
  are bypassed, switched deferred Vulkan staging buffer reclamation to sorted tick binary search,
  and pre-armed sequential GPU-to-CPU ring-buffer readback pages on step detection to cut first-hit
  synchronous CPU readback stalls.
- **Configurable VRAM Reserve (`vram_reserve_mib`) & Ocarina of Time AA tip**: Added a **VRAM
  Reserve** setting under *Settings → Graphics → Advanced* (`Auto` default: 1/8 of driver budget
  up to 1,024 MiB) so 4–8 GiB GPUs can reclaim extra headroom for textures, and added an
  `EDEN_NX2_OOT_AA=0` override matching *Settings → Graphics → Ocarina of Time anti-aliasing → No
  sharpening* (recommended on 4–6 GiB GPUs such as GTX 1650 to bypass three post-process compute
  passes for substantial GPU frame-time savings).

### Vulkan, SPIR-V & Texture Compatibility

- **Lean multi-buffer SPIR-V global-memory helpers**: Marked generated multi-buffer global-memory
  load, store, atomic, and CAS helper functions `DontInline` across classic graphics stages while
  keeping single-buffer routes (including NVIDIA `OpSelect` single-buffer compute shaders),
  branch-routed `Compute`, `Task`, `Mesh`, and emulated mesh-as-compute shaders inline, slashing
  SPIR-V bloat and pipeline compile times while preserving Ocarina of Time foliage and tree
  accuracy.
- **AMD mesh-shader fallback & NVIDIA driver reporting**: Enabled compute mesh-shader fallback
  properly on AMD hardware where needed, formatted NVIDIA proprietary driver versions using their
  10-8-8-6 bit scheme, added a quiet `A8B8G8R8_UNORM` fallback for zeroed guest texture headers,
  and gated byte-offset texture header handles strictly to titles that use them.

### Switch HOME Menu (`qlaunch`) Full Library & `.nup` / `.nx` Integration

- **Full `.nup`, `.nx`, NSP, XCI & NRO discovery on HOME Menu**: Synthesized decrypted Control NCAs
  with locale-normalized NACP metadata, cached HD artwork fallback, and strict ≤128 KiB baseline
  JPEG icons (with automatic PNG/JPEG transcoding and procedural fallback tiles) so Switch 1 and
  Switch 2 `.nup` and `.nx` packages, uninstalled containers, and homebrew appear on HOME Menu with
  titles, publishers, display versions, real occupied sizes, and artwork — including via `-qlaunch`
  and regardless of external-content directory settings.
- **Switch 2 title aliasing & seamless frontend launch handoff**: Deterministically aliased Switch
  2 `0x0400...` Program IDs onto HOME Menu-compatible IDs exclusively for system applets, fixed a
  byte-indexing bug in `ListApplicationIcon` and transfer-memory handling in
  `ListApplicationTitle`, deduplicated multi-source application records, and routed HOME Menu game
  launches through the frontend with preselected user profile preservation, full per-game settings,
  and Switch 2 extended DRAM.
- **HOME Menu, User Page & Power Menu polish**: Implemented missing `ns`, `am`, `acc`, and `friend`
  calls (`GetNintendoAccountUserResourceCache`, `GetPlayHistoryStatistics`, `IGlobalStateController`
  sleep/reboot handlers, rights/record/gamecard queries), guarded non-NCA manual provider lookups
  and embedded `GRenderWindow` handle connections, eliminated duplicate HLE shutdown stops, added
  friendly log names for firmware applets, and cleaned up false-alarm avatar and network log spam.

## [1.8.0] - 2026-11-18

### Mario Kart World & Switch 2 Frame-Time Improvements

- **Cyclic GPU-to-CPU readback tracking and coalescing**: Expanded learned readback page tracking
  from 512 to 4,096 pages (up to 24 MiB/frame) so multi-thousand-page Switch 2 compute culling and
  visibility rings remain warm across wrap-arounds instead of dropping back to synchronous CPU
  readback flushes. Contiguous early-download pages are merged into single copy spans, same-buffer
  readback copies batch into one download call, and speculative `cpu_read_ranges` prefetching is
  bounded to a forward-biased 2 MiB window so isolated 4 KiB reads do not stall on multi-megabyte
  buffers.
- **Present fence pacing on Mario Kart World**: Adaptive `nx2:vi2` presentation now skips the
  blocked 3–4 ms current-frame suspension probe when Mario Kart World's current frame fence is
  still waiting on a post-present CPU semaphore release, pacing directly against the previous
  frame's fence while keeping immediate pass-through when the current fence is already signaled.
- **Unreported occlusion query suppression**: When a guest leaves Z-pass pixel counting enabled past
  the 1,024-slot overflow guard without resetting or reporting, host Vulkan occlusion queries now
  stay paused until the next guest reset or report instead of repeatedly cycling query pools.
- **Parallel read-only Windows VFS I/O**: Read-only `RealVfsFile` reads on Windows now run through
  cached, reference-counted overlapped file handles (up to 4,096 open handles independent of the C
  runtime `FILE*` cap) without holding `RealVfsFilesystem::list_lock` across disk I/O, allowing
  concurrent multi-core RomFS, audio, and video streaming while cleanly releasing cached handles on
  file and directory mutations.

### Vulkan Renderer, Memory & Cache Efficiency

- **Shader stage translation cache**: Added a thread-safe `StageCache` with RAM-tiered capacity
  that deduplicates translated IR and zstd-compressed emitted SPIR-V across pipelines sharing stage
  programs and constant/resource reads, accelerating shader compilation and cold pipeline builds.
- **Vulkan descriptor, layout & render-pass overhead reductions**: Added cached `VkPipelineLayout`
  reuse, tighter fast-draw uniform and storage buffer pathways, render-pass retention across safe
  state transitions, and async draw/dispatch discard policy for unbuilt Switch 2 compute dispatches
  whose output buffers already hold GPU data and color draws targeting historical render targets.
- **Texture, ASTC & low-VRAM management**: Improved 4 GB and tight-VRAM buffer/texture garbage
  collection guards so device-pressure collection avoids stalling on live GPU-written working sets,
  offloaded deferred Vulkan memory frees to a background worker, tracked stale render targets and
  superseded images safely, checked resolution-scaled image fit before allocation, and optimized
  ASTC upload and opaque-alpha handling.
- **DMA copy hazard & shader cache robustness**: Tracked GPU-written destination subranges across
  buffer `DMACopy` operations so guest memory and texture uploads never read stale bytes, and
  chunked large shader cache v2 reads cleanly past 2 GiB on Windows.

### Shader Recompiler, CPU JIT & System Services

- **Ampere shader compiler improvements**: Expanded NVIDIA global-memory buffer routing and
  non-inlined bound-buffer selection helpers, added stage-buffer output (`ISBE`) and additional
  Ampere instruction lowerings, and reduced redundant pass overhead during shader translation.
- **CPU, host memory & NX2 services**: Added shared x86-64 Dynarmic JIT cache reuse, guest memory
  prefaulting and faster Windows memory/zero-page queries, GPU-only region tracking, VP9 range
  encoder fixes, and `nx2:gsv` service stability improvements.

### Tests

- Added and expanded Catch2 test coverage for large cyclic readback rings and forward-windowed
  speculative downloads, `nx2:vi2` present fence planning, concurrent Windows `RealVfsFile` reads
  and filesystem mutations, `StageCache` translation reuse, `PipelineLayoutCache`, stale render
  targets, superseded images, scaled image sizing, ASTC opacity, Ampere stage-buffer outputs, async
  draw/dispatch policy, VP9 range encoding, GPU-only regions, and shared x64 JIT caches.

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
