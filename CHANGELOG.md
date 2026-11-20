# Changelog

All notable changes to Veshiru are documented here. Each release's notes are taken from its
`## [X.Y.Z]` section (release candidates use the section of the version they are a candidate for).
New entries go under `## [Unreleased]`; `tools/ci/bump-version.sh` turns them into a version section.

## [Unreleased]

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
  [HD rumble on Xbox-style pads](docs/Veshiru.md#hd-rumble-on-xbox-style-pads).
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

### Boot

- **`main.npdm` is parsed rather than assumed**: the NCA loader read the eight bytes at offset
  0x28 as a heap size without checking the `META` magic, and the supplied HOME Menu log shows a
  file where that read produced `0x7463656c` — the ASCII bytes of a text field — as a 1.8 GiB heap,
  which then sized the pointer buffer for a heap that does not exist. The heap size is now read
  only from a real NPDM, and anything unreasonable falls back to the default pointer buffer.

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

### Boot

- **`main.npdm` is parsed rather than assumed**: the NCA loader read the eight bytes at offset
  0x28 as a heap size without checking the `META` magic, and the supplied HOME Menu log shows a
  file where that read produced `0x7463656c` — the ASCII bytes of a text field — as a 1.8 GiB heap,
  which then sized the pointer buffer for a heap that does not exist. The heap size is now read
  only from a real NPDM, and anything unreasonable falls back to the default pointer buffer.

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

### Boot

- **`main.npdm` is parsed rather than assumed**: the NCA loader read the eight bytes at offset
  0x28 as a heap size without checking the `META` magic, and the supplied HOME Menu log shows a
  file where that read produced `0x7463656c` — the ASCII bytes of a text field — as a 1.8 GiB heap,
  which then sized the pointer buffer for a heap that does not exist. The heap size is now read
  only from a real NPDM, and anything unreasonable falls back to the default pointer buffer.

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
