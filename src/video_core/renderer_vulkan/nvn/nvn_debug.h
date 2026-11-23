// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Diagnostics of the draw path, all gated by environment variables (nothing runs when they are
// unset; the configuration is read once):
//
//   EDEN_NX2_NVN_GPU_PROFILE=1|2|frame  GPU time per render pass, dispatch, copy and clear from
//                                  timestamp queries, reported every 10 s (2: also one interval per
//                                  draw; frame: only a timestamp pair per command buffer, logged as
//                                  GPU ms per frame, no drain inside the frame: the mode that can
//                                  judge barrier changes); see nvn_gpu_profile.h and DESIGN.md
//                                  "GPU profiler"
//   EDEN_NX2_NVN_NO_ZERO_SIZE=1    experiment: storage buffers bound with an address and size 0 bind
//                                  the dummy buffer again (the behavior before they bound the rest
//                                  of their pool); changes what compute/effect shaders do
//   EDEN_NX2_NVN_NO_HOST_IMPORT=1  experiment: CPU-accessible pools use device-local mirrors instead
//                                  of imported host memory (1-2 FPS in DK: see DESIGN.md "GPU profiler")
//   EDEN_NX2_NVN_SHADOW_STORAGE=0  switches off the device-local shadows that written storage
//                                  ranges of imported pools (<= EDEN_NX2_NVN_SHADOW_MAX_KIB, default
//                                  1024) are bound through: the host memory is bound as before
//   EDEN_NX2_NVN_SHADOW_LOG=1      logs the first shadowed ranges (see DESIGN.md "Storage shadows")
//   EDEN_NX2_NVN_LOG_BACKING=<hash>[:<max dispatches, default 8>]  logs guest storage binding
//                                  backing and atomic-target status for matching compute dispatches
//   EDEN_NX2_NVN_CONDITION_PREFETCH=0  render-condition predicates are no longer computed ahead in
//                                  batches: every condition that is not enabled/disabled ends the
//                                  pass and computes its own (see DESIGN.md "Predicates computed
//                                  ahead")
//   EDEN_NX2_NVN_DRAWLOG=<n>       logs the first n draws of every selected frame (pipeline stage
//                                  hashes, topology, state words, viewport/scissor, targets, bound
//                                  textures and buffers) and a per-target draw table at the end of
//                                  the frame. An indirect draw's `IND=` words are guest memory at
//                                  record time; a mesh-task indirect draw also gets a
//                                  `[nvn] watch draw-args f<frame> #<draw> ...` line at completion
//                                  with the {count, first} the GPU read (`draw-count`: count word).
//                                  `rc=cond`: the draw is gated by a GPU render-condition predicate
//                                  (the GPU may discard it; the emulated path's log omits draws whose
//                                  condition its CPU resolved false)
//   EDEN_NX2_NVN_DUMP_TARGETS=<d>  writes every render pass's color/depth targets of the selected
//                                  frame to <d>/f<frame>_p<pass>_<slot>_*.tga, plus statistics;
//                                  depth and every colour format the 8-bit TGA cannot hold exactly
//                                  (float, 10/16/32-bit) also keep their texels in <name>.tga.raw
//   EDEN_NX2_NVN_DUMP_FRAME=<n>[,<n>...]  the selected frames: the 1-based present numbers the
//                                  frames' work is recorded for (default: every 120th starting at
//                                  1); each number selects EDEN_NX2_NVN_DUMP_COUNT frames
//   EDEN_NX2_NVN_DUMP_MAX=<n>      dump files per frame (default 400)
//   EDEN_NX2_NVN_DUMP_WIDTH=<w>    only targets of this width
//   EDEN_NX2_NVN_DUMP_THUMB=<w>    the TGA files are box-downscaled to at most w pixels wide (the
//                                  statistics still cover every pixel; a downscaled target keeps no
//                                  .raw file)
//   EDEN_NX2_NVN_CACHE_LOG=1       logs every pipeline compiled at run time with the state words
//                                  that differ from a cached pipeline of the same programs
//
// Experiments (change what the replay draws; never set in normal runs):
//   EDEN_NX2_NVN_FACE=1|2          1: invert the front face, 2: no culling
//   EDEN_NX2_NVN_NO_DEPTH=1        depth test off
//   EDEN_NX2_NVN_RELAX_DEPTH_EQUAL=1  test depth func EQUAL as LESS_OR_EQUAL (default: exact; see
//                                  DESIGN.md "Depth EQUAL after a pre-pass")
//   EDEN_NX2_NVN_DEPTH_PROBE=1|2|3 EQUAL draws test LEQUAL / GEQUAL / ALWAYS and write depth (measures how
//                                  far their depth is from the pre-pass; dump the depth target)
//   EDEN_NX2_NVN_ONLY_VS=h,h,..    draw only draws of the selected frame whose vertex program hash (hex) is listed
//   EDEN_NX2_NVN_ONLY_DRAWS=a,b,.. draw only these draws (log numbering) of the selected frame
//   EDEN_NX2_NVN_FLIP_VIEWPORT=1   positive-height viewports (the rule is negative; with it set
//                                  use _FACE=1 too, see nvn_viewport.h)
//   EDEN_NX2_NVN_DEPTH_MODE=0|1    override the device depth mode (0 near=-w, 1 near=0)

#pragma once

#include <atomic>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "video_core/vulkan_common/vulkan_wrapper.h"

#include "common/common_types.h"

namespace Vulkan::Nvn::Debug {

struct Config {
    u32 draw_log{};
    std::string dump_dir;
    /// EDEN_NX2_NVN_DUMP_SPIRV=<d>: the translated SPIR-V of every stage is written to <d>.
    std::string dump_spirv;
    /// EDEN_NX2_NVN_DUMP_FRAME=<n>[,<n>...] (empty: every 120th frame from 1).
    std::vector<u64> dump_frames;
    u32 dump_max{400};
    u32 dump_width{}; ///< only targets of this width (0: any)
    u32 dump_thumb{}; ///< EDEN_NX2_NVN_DUMP_THUMB: TGA width cap (0: full size)
    /// EDEN_NX2_NVN_DUMP_DIVISOR=n: float targets are divided by n in the TGA (HDR values above 1
    /// stay distinguishable); the logged statistics are unscaled.
    u32 dump_divisor{1};
    u32 face{};
    bool no_depth{};
    /// EDEN_NX2_NVN_DUMP_COUNT=n: n consecutive frames from each DUMP_FRAME number are selected.
    u32 dump_count{1};
    /// EDEN_NX2_NVN_DUMP_PASS=p: only pass p (1-based pass index of the frame) is dumped.
    u32 dump_pass{};
    /// EDEN_NX2_NVN_DUMP_DRAW=<n>: logs the CPU view of the vertex and uniform buffers of draw n.
    u32 dump_draw{};
    u32 depth_probe{};
    /// EDEN_NX2_NVN_ONLY_DRAWS=a,b,..: the draws (1-based, in the log's numbering) of the selected
    /// frame that are drawn; every other draw of that frame is skipped.
    std::vector<u32> only_draws;
    /// EDEN_NX2_NVN_ONLY_VS=h,h,..: (hex) only draws whose vertex program hash is listed are drawn in
    /// the selected frame.
    std::vector<u64> only_vs;
    /// EDEN_NX2_NVN_OPLOG=1: logs the buffer-writing ops (clears, copies, inline updates, dispatches,
    /// reports, barriers) of the selected frame in submission order.
    bool op_log{};
    /// EDEN_NX2_NVN_WATCH=<hex address>[:<hex bytes>] (with OPLOG): the range is copied back after
    /// every dispatch/barrier/render target change of the selected frame and logged.
    u64 watch_address{};
    u32 watch_size{64};
    /// EDEN_NX2_NVN_PEEK=<hex program hash>:<storage slot>:<hex offset>:<words> (with OPLOG): at
    /// every dispatch of that compute program in the selected frames, the guest-memory view of
    /// `words` 32-bit words at the storage binding's address + offset is logged (statistics and
    /// the first words), plus the textures the dispatch binds.
    u64 peek_program{};
    u32 peek_slot{};
    u64 peek_offset{};
    u32 peek_words{16};
    /// EDEN_NX2_NVN_LOG_BACKING=<hex program hash>[:<max dispatches>] (default max: 8).
    u64 log_backing_program{};
    u32 log_backing_max{8};
    /// EDEN_NX2_NVN_POKE=<word index>:<hex value> (diagnostic, wrong on purpose): with PEEK, that
    /// word of the peeked guest range is overwritten before each peeked dispatch.
    bool poke_set{};
    u32 poke_index{};
    u32 poke_value{};
    /// EDEN_NX2_NVN_PEEK_CHASE=<word index>:<hex offset>:<words> (with PEEK): the u64 at that word
    /// of the peeked range is a pointer; the guest words at pointer + offset are logged, and with
    /// WATCH=peek the GPU view of them too.
    u32 chase_word{};
    u64 chase_offset{};
    u32 chase_words{};
    bool cache_log{};
    /// EDEN_NX2_NVN_ASYNC_WORKERS=<n>: threads of the run-time pipeline compile workers (0 = the
    /// automatic count, `clamp(hardware threads / 4, 2, 6)`).
    u32 async_workers{};
    /// EDEN_NX2_NVN_GPU_PROFILE: 0 off, 1 intervals, 2 intervals and one per draw, 3 (`frame`)
    /// only a pair per command buffer.
    u32 gpu_profile{};
    /// EDEN_NX2_NVN_NO_ZERO_SIZE=1: size-0 storage entries stay unbound (dummy buffer).
    bool no_zero_size{};
    /// EDEN_NX2_NVN_NO_STATE_ELIDE=1 (experiment): every draw records its push constants,
    /// descriptors and vertex/index buffers again, as before the pass state tracking.
    bool no_state_elide{};
    /// EDEN_NX2_NVN_COSTSPLIT=<class>:<start>[:<modes>][:<period>] (MEASUREMENT ONLY, the picture
    /// is wrong on purpose): the raw text, parsed by CostSplit::Parse (nvn_costsplit.h).
    std::string costsplit;
    /// EDEN_NX2_NVN_VERTEX_STAGE=0 (A/B, kill switch): passes read their vertex buffers from the
    /// imported pools instead of device-local copies staged at pass begin by a compute gather
    /// (nvn_vertex_stage.h); on by default. Off also turns index staging off.
    bool vertex_stage{true};
    /// EDEN_NX2_NVN_INDEX_STAGE=0 (A/B, kill switch): with vertex staging on, the direct indexed
    /// draws of a pass bind their guest index buffer instead of a copy staged in the same gather;
    /// on by default.
    bool index_stage{true};
    /// EDEN_NX2_NVN_FEEDBACK_LOG=1: logs the first feedback copies (shape, sampled view, targets).
    bool feedback_log{};
    /// EDEN_NX2_NVN_SHADOW_STORAGE=0 switches the device-local shadows of written storage ranges in
    /// imported pools off (nvn_draw_shadow.cpp); on by default.
    bool shadow_storage{true};
    /// EDEN_NX2_NVN_SHADOW_MAX_KIB: longest written storage range that is shadowed (default 1024).
    u64 shadow_max_bytes{u64{1} << 20};
    /// EDEN_NX2_NVN_SHADOW_LOG=1: logs the first shadowed ranges (pool, range, program).
    bool shadow_log{};
    /// EDEN_NX2_NVN_POOL_MIRROR: storage bindings and indirect arguments of imported pools use a
    /// persistent VRAM mirror of the pool (nvn_pool_mirror.h, DESIGN.md "Pool mirrors") instead of
    /// the guest memory and the per-command-buffer storage shadows. 0 off; 1 a written storage
    /// range makes its pool's mirror (read-only ranges use existing mirrors); 2 every storage
    /// range does.
    u32 pool_mirror{0};
    /// EDEN_NX2_NVN_POOL_MIRROR_MAX_MIB: VRAM all pool mirrors together may take (default 4096).
    u64 pool_mirror_max_bytes{u64{4096} << 20};
    /// EDEN_NX2_NVN_STORAGE_BUDGET_MIB: the storage cache budget (0: the default, 6144 MiB).
    u64 storage_budget_bytes{0};
    /// EDEN_NX2_NVN_TRIM_AGE_PRESENTS: presents after which an idle render target that can be
    /// rebuilt leaves the storage cache (default 0 = off: it dropped GPU-written lighting targets in OoT; 300 = the old rule).
    u64 trim_age_presents{0};
    /// EDEN_NX2_NVN_TRIM_LEGACY=1: the storage cache trim without the age and device budget rules.
    bool trim_legacy{false};
    /// EDEN_NX2_TRY_VMA_DEVICE_STRICT=1 (candidate): the replay's device-local images and buffers
    /// require device-local memory; over the budget they are made there anyway (the driver
    /// pages) and host memory is the last resort (MemoryAllocator::TrackSpills).
    bool try_vma_device_strict{false};
    /// EDEN_NX2_TRY_STORAGE_VRAM=0 opts out of mirroring bounded imported storage pools and
    /// redirecting raw-table aliases into the same copy, with full-pool raw write tracking.
    bool try_storage_vram{true};
    /// EDEN_NX2_TRY_STORAGE_VRAM_MAX_MIB: largest pool eligible for raw-table mirroring (64 MiB).
    u64 try_storage_vram_max_bytes{u64{64} << 20};
    /// EDEN_NX2_NVN_WB_MONITOR=1 (diagnostic): arms a CPU access trap on every pool mirror range
    /// the write-back just merged (up to 32 MiB per entry) and counts, per pool, how often the guest
    /// touches those pages before the next merge: the data that decides lazy write-back. Slows the
    /// guest (a trapped access costs microseconds); reported in the replay heartbeat.
    bool wb_monitor{false};
    /// EDEN_NX2_TRY_WB_LINES=0|1|8|32 (default 8, 0 opts out): the pool mirrors' write-back merge
    /// stores a unit of 8 (1, 8) or 32 words with two or more changed words whole (the unchanged
    /// words as the guest holds them just before) instead of every changed word on its own:
    /// scattered 4-byte stores into host memory cost 10-20 ns each over PCIe, a full line 3-5 ns
    /// (WriteBackMerge::SetLineMode). Only on pages guarded by an access trap (MirrorLineGuard): a
    /// page the guest touches goes back to the exact word merge, so no CPU store is lost.
    u32 try_wb_lines{8};
    /// EDEN_NX2_TRY_DISPATCH_ORDER=1 (candidate, A/B pending): a dispatch is ordered after the shader
    /// writes of an earlier dispatch only by a game barrier, as NVN orders them (no full barrier
    /// between two dispatches the game did not separate); everything else still orders after them.
    bool try_dispatch_order{false};
    /// EDEN_NX2_TRY_FLUSH_TRACKED=0 opts out (on by default): a guest flush of a pool mirror's range
    /// (nvnMemoryPoolFlushMappedRange; OoT flushes whole pools every other frame and writes 17 pages
    /// of 2.5 M in all) only makes the mirror copy in again the pages the page tracker saw the guest
    /// write (and the ones the mirror does not hold); a host service's write (Sink::NotifyHostWrite)
    /// still counts in full. Every host writer of guest memory tells the tracker (DESIGN.md "Guest
    /// flushes of mirrored pools"), so the flush only has to find the guest CPU's stores.
    bool try_flush_tracked{true};
    /// EDEN_NX2_NVN_MIRROR_HASH_ALWAYS=1: mirror pools (no host import) hash every guest page of
    /// every range they use at each submission, as before the page tracker gated it (A/B).
    bool mirror_hash_always{false};
    /// EDEN_NX2_NVN_RAW_SPLIT_IMPORT (default on, =0 off): import host-discontiguous CPU-visible
    /// pools run by run so raw global loads/stores reach them (OoT's Kokiri pool).
    bool raw_split_import{true};
    /// EDEN_NX2_NVN_TEX_HASH_ALWAYS=1: CPU-sourced textures are hashed at every due check even when
    /// the page tracker saw none of their pages written (A/B of Impl::tex_hash_tracked).
    bool tex_hash_always{false};
    /// EDEN_NX2_NVN_TEX_PAGE_HASH=0: a tracked write to a trusted texture hashes it whole again
    /// instead of only the written pages (A/B of Impl::tex_page_hash).
    bool tex_page_hash{true};
    /// EDEN_NX2_NVN_TEX_FLUSH_TRUSTED=0: a tracked guest flush forces a full hash of the textures
    /// it covers (A/B of Impl::tex_flush_trusted).
    bool tex_flush_trusted{true};
    /// EDEN_NX2_TRY_INVALIDATE_WRITEBACK=<ms> (candidate, off; 1 = 250 ms): nvnMemoryPoolInvalidate
    /// MappedRange (the game's "I read what the GPU wrote now") waits, at most that long, until
    /// the submissions made before it that write its range have completed, so the CPU reads what
    /// the GPU wrote there like on the console, where the GPU runs as the game submits. Without
    /// it the replay lag (frames) decides what the CPU sees (DESIGN.md "Invalidate").
    u32 try_invalidate_writeback{0};
    /// EDEN_NX2_TRY_BARRIER_MERGE=1 (candidate, off; 2/3/4 = one part only, BarrierMergeMask): the
    /// replay's own buffer commands (shadow and mirror copies, write-back merges) skip the barrier in
    /// front of them when no byte they touch conflicts with the buffer commands since the last barrier
    /// (nvn_barrier_track.h), the inline uniform updates record no barrier when no copy follows, and
    /// the layout transitions a draw needs before its pass share one barrier with the attachments'
    /// (DESIGN.md "Barrier merging").
    u32 try_barrier_merge{0};
    /// EDEN_NX2_TRY_STAGE_SPAN=1 (candidate, off by default): vertex staging scans past render target
    /// changes, so the small passes of one window (a game that switches targets every few draws)
    /// share one gather when together they hold the minimum number of draws, and a mesh indirect
    /// argument conversion does not drop the staged copies (it writes only its own arena).
    bool try_stage_span{false};
    /// EDEN_NX2_TRY_UBO_MIRROR=1 (candidate, off by default): read-only uniform buffer ranges of
    /// imported pools are served from device-local pool mirrors (made at the pool's first uniform
    /// binding, at most TRY_STORAGE_VRAM_MAX_MIB each), kept current by a copy-in of the pages the
    /// guest wrote at every pass begin, instead of read over PCIe (CostSplit `vramu`).
    bool try_ubo_mirror{false};
    /// EDEN_NX2_TRY_VTX_MIRROR=0 opts out (on by default): the vertex and index ranges a render
    /// pass stages are served from device-local pool mirrors (read-only: no raw table entry, made
    /// stale by raw-writing draws) that copy in only the stale pages of the ranges used, at most
    /// VTX_MIRROR_COPY_BYTES per pass, instead of copying every range of every pass with 32+ draws
    /// (VertexStage) or reading it over PCIe (passes of fewer draws).
    bool try_vtx_mirror{true};
    /// EDEN_NX2_TRY_VTXCOV_INPASS / _SHADERWRITES / _BOUND / _BIGPOOL: VtxCovSwitch bits
    /// (nvn_vtxcov.h), all on by default (Load sets VTXCOV_ALL); =0 turns one off and
    /// EDEN_NX2_TRY_VTXCOV=0 turns all four off. They act with TRY_VTX_MIRROR.
    u32 try_vtxcov{0};
    /// EDEN_NX2_NVN_VTXCOV_DIAG=1 (opt-in): every
    /// vertex/index bind is counted by where it was read from or why it stayed in host memory,
    /// logged as `[nvn] vtxcov:` every 10 s of presents.
    bool vtxcov_diag{false};
    /// EDEN_NX2_NVN_DIAG_RO_RAW_KEEP=1 (measurement only, the picture may be wrong): a raw-writing
    /// draw leaves the read-only mirrors' pages valid. The gap between this and the real switch is
    /// the cost of Impl::MirrorInvalidateReadOnly (the copies the invalidation makes the next
    /// draws repeat); identical pictures mean no raw store of the run reached a read-only mirror.
    bool diag_ro_raw_keep{false};
    /// EDEN_NX2_TRY_RAW_KEEP_ENTRIES=0 opts out (on by default): a draw with raw global memory
    /// does not end every live pool-mirror entry (merge + new whole-pool snapshots at each of the
    /// ~55 raw draws a present on OoT). The raw table points at the mirrors, so raw reads and
    /// writes already see the entries' bytes; the entries (and the raw-writing draws' whole-pool
    /// ones) end at a conflicting use or the end of the command buffer. Plain copy shadows (pools
    /// without a table mirror) still end.
    bool try_raw_keep_entries{true};
    /// EDEN_NX2_TRY_REFUSED_SHADOW=0 opts out (on by default): with pool mirrors on, a written
    /// storage range of at most EDEN_NX2_NVN_REFUSED_SHADOW_MAX_KIB in an imported pool that gets
    /// no mirror (over TRY_STORAGE_VRAM_MAX_MIB, the budget, unaligned) is bound through a copy
    /// shadow, as with mirrors off, instead of staying in host memory where its stores and atomics
    /// cross PCIe.
    bool try_refused_shadow{true};
    /// EDEN_NX2_NVN_REFUSED_SHADOW_MAX_KIB: longest such range (default 4096; OoT's CMAA edge
    /// list is 1.76 MiB, over the 1 MiB SHADOW_MAX_KIB of the mirrors-off path).
    u64 refused_shadow_max_bytes{u64{4} << 20};
    /// EDEN_NX2_NVN_CONDITION_PREFETCH=0: every GPU render condition computes its own predicate at
    /// its point (no batch of the conditions ahead, Recorder::PrefetchConditions); on by default.
    bool condition_prefetch{true};
    /// EDEN_NX2_NVN_UPLOADLOG=1: logs every texture upload with its size, format, decode kind and
    /// the time of each step (guest read, hash, fill/decode, command recording).
    bool upload_log{};
    /// EDEN_NX2_NVN_STORAGE_CENSUS=1 logs a detailed replay-side inventory every 900 presents.
    bool storage_census{};
    /// EDEN_NX2_NVN_ASTC_VERIFY=1: every storage decoded on the GPU is also decoded on the CPU,
    /// read back and compared (logs the mismatches; slow, one GPU wait per storage).
    bool astc_verify{};
    bool equal_lequal{};
    bool flip_viewport{};
    s32 depth_mode{-1};
    /// Early submission (nvn_exec.cpp Recorder): a submission's command buffers go to the queue in
    /// pieces while the rest is still being recorded, so the GPU starts before the replay is done.
    /// `early_first_us`: CPU time of the first piece; `early_interval_us`: of the later ones (both
    /// only at a point where no render pass is open; 0 interval = off); `early_pass_us`: a piece
    /// this long with a pass still open ends the pass and goes anyway (0 = never inside a pass).
    /// EDEN_NX2_NVN_EARLY_SUBMIT_US=<interval>, _EARLY_FIRST_US=<n>, _EARLY_PASS_US=<n>.
    u32 early_first_us{300};
    u32 early_interval_us{1500};
    u32 early_pass_us{4000};
    /// Any per-draw diagnostic is on.
    bool tracing{};
};

/// EDEN_NX2_NVN_TIMELINE=1: every submission and present logs its hand-offs (enqueued by the guest,
/// recording begin/end on the replay thread, GPU done and completion callback on the completion
/// thread) as `[nvn] tl <us> <event> <seq> [detail]`, to see which thread waits for which. The
/// event model logs its signals, waits and CPU changes the same way (`detail`: queue, address,
/// values). `<us>` is steady-clock microseconds since its epoch, so lines written by the core
/// (`[nvn] tl ... guest_wait_*`) share the time base.
extern bool g_timeline;
void TimelineEvent(const char* event, u64 seq, std::string_view detail = {});
[[nodiscard]] inline bool Timeline() {
    return g_timeline;
}

/// The configuration, read once from the environment during static initialization and refreshed by
/// `Reload()` once the frontend applies switch files. A plain global, so every read on a hot path
/// is one load (no function-local-static guard).
extern Config g_config;
[[nodiscard]] inline const Config& Get() {
    return g_config;
}
void Reload();

/// `present_index` is the 1-based present number the work being recorded belongs to.
[[nodiscard]] bool FrameSelected(u64 present_index);

// ---- On-demand diagnostics (diag request file: `nvn-pass <n>`, `nvn-draw <n>`) ----------------
//
// A request applies to the next whole frame the replay records (the first frame that starts after
// the next present): `nvn-pass <n>` dumps the render targets of pass n (EDEN_NX2_NVN_DUMP_PASS
// with the request folder as EDEN_NX2_NVN_DUMP_TARGETS) and appends the dump statistics to
// `nvn-pass-<n>.txt`; `nvn-draw <n>` logs the state and the CPU view of the vertex and uniform
// buffers of draw n (EDEN_NX2_NVN_DUMP_DRAW) and also writes them to `nvn-draw-<n>.txt`. Pass and
// draw numbers are the 1-based numbers of the draw log.

/// What a request bound to a frame asks for.
struct RequestState {
    u64 frame{};     ///< 1-based present number of the frame the request applies to
    u32 pass{};      ///< 0: none
    u32 draw{};      ///< 0: none
    std::filesystem::path directory;
};

/// 0: no request; 1: armed, waits for the next frame to start; 2: bound to a frame, running.
extern std::atomic<u32> g_request_phase;

/// Any per-draw diagnostic is on: the environment configuration or a pending request. One load of
/// the configuration and one relaxed atomic load.
[[nodiscard]] inline bool Tracing() {
    return g_config.tracing || g_request_phase.load(std::memory_order_relaxed) != 0;
}

/// Arms a request (any thread). `pass` / `draw` of 0 leave that part as it is; requests armed
/// before the replay binds them merge into one.
void RequestDump(u32 pass, u32 draw, std::filesystem::path directory);

/// Replay thread, once per present, after the present counter was incremented to `present_index`:
/// binds an armed request to the frame that starts now and retires a bound request whose frame has
/// just ended (returns true then). One relaxed load while no request is pending.
bool OnPresent(u64 present_index);

/// Replay thread: the request bound to `frame`, null when there is none.
[[nodiscard]] const RequestState* RequestFor(u64 frame);

/// A new context starts: forgets an armed or bound request of an earlier one (its frame numbers
/// count that context's presents).
void ResetRequests();

/// Appends `text` and a newline to the file `path` (the report of a request).
void AppendLine(const std::filesystem::path& path, std::string_view text);

/// Bytes per pixel of the aspect of an image (0: not convertible).
[[nodiscard]] u32 PixelBytes(VkFormat format, VkImageAspectFlags aspect);

/// A dump the replay thread recorded a copy for; written when the submission completed.
struct PendingDump {
    std::string path;
    u32 width{};
    u32 height{};
    VkFormat format{};
    VkImageAspectFlags aspect{};
    /// The request report the completion appends its statistics line to (empty: none).
    std::filesystem::path report;
    /// Width cap of the written TGA (0: full size): box-downscaled by the smallest integer factor
    /// that fits; the depth .raw file is not written then.
    u32 thumb_width{};
};

/// Converts `pixels` (tightly packed, PixelBytes per pixel) to a top-down 32-bit TGA file and
/// returns a one-line statistics summary over every pixel (mean color, fraction of non-black
/// pixels, range).
std::string WriteTga(const PendingDump& dump, std::span<const u8> pixels);

/// Converts `width * height` pixels of `format` (tightly packed, PixelBytes per pixel, first row at
/// the top) to RGBA8 for a PNG: unorm and sRGB as stored, floats clamped to [0, 1], depth as grey.
/// False when the format cannot be converted.
[[nodiscard]] bool ToRgba8(VkFormat format, VkImageAspectFlags aspect, u32 width, u32 height,
                           std::span<const u8> pixels, std::vector<u8>& rgba);

} // namespace Vulkan::Nvn::Debug
