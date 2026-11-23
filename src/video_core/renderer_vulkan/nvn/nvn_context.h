// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <mutex>

#include "common/common_types.h"
#include "core/frontend/framebuffer_layout.h"
#include "video_core/nvn/sink.h"
#include "video_core/renderer_vulkan/nvn/nvn_vtxcov.h"

namespace Core::Frontend {
class EmuWindow;
}

namespace VideoCore {
struct RendererSettings;
class ShaderNotify;
class LoadProgressSink;
}

namespace Vulkan {
class Device;
class MemoryAllocator;
class PresentManager;
class Scheduler;
struct Frame;
} // namespace Vulkan

namespace Vulkan::Nvn {

/// Where the replay hands finished images over for display.
class PresentTarget {
public:
    virtual ~PresentTarget() = default;

    /// Returns a frame sized for the window with an image that can be written. Blocks until one is
    /// free. Called from the replay thread.
    virtual Frame* AcquireFrame() = 0;

    /// Queues `frame` for display. Its render_ready semaphore has already been submitted for
    /// signalling. Called from the replay thread.
    virtual void PresentFrame(Frame* frame) = 0;

    /// Gives back a frame from AcquireFrame that no submitted work uses (recording failed).
    virtual void ReleaseFrame(Frame* frame) {
        static_cast<void>(frame);
    }

    /// The context's thread-safe allocator, valid from Attach until Detach. Frame images the
    /// target creates with it must be released in Detach (the allocator dies right after).
    virtual void Attach(MemoryAllocator& allocator) {
        static_cast<void>(allocator);
    }
    /// Drains queued presentation work before the context waits for the device to go idle.
    virtual void Drain() {}
    virtual void Detach() {}

    /// VkFormat of the images AcquireFrame hands out (0 = VK_FORMAT_UNDEFINED when unknown:
    /// treated as UNORM).
    [[nodiscard]] virtual s32 FrameFormat() const {
        return 0;
    }

    /// VkFormat of the swapchain images the frames are copied into (what DLSS frame generation
    /// is told is its colour buffer format). Defaults to the frame format.
    [[nodiscard]] virtual s32 SwapchainImageFormat() const {
        return FrameFormat();
    }

    /// The window's layout as of now: the size AcquireFrame sizes frames to and the rectangle of
    /// it the picture goes into (aspect ratio setting). A zero width means the target has no
    /// window: the replay then fits the picture into the frame.
    [[nodiscard]] virtual Layout::FramebufferLayout ScreenLayout() const {
        return Layout::FramebufferLayout{.width = 0, .height = 0, .screen = {}, .is_srgb = false};
    }

    /// How many frames besides the one a present is drawn into can be acquired for the same
    /// present (frame generation's generated frames) without waiting for a display.
    [[nodiscard]] virtual size_t MaxExtraFrames() const {
        return 0;
    }

    /// True when a separate present thread presents the frames (asynchronous presentation).
    [[nodiscard]] virtual bool UsesPresentThread() const {
        return false;
    }
};

struct ContextOptions {
    /// Zero-copy import of CPU-accessible pools with VK_EXT_external_memory_host. When false (or
    /// when the import fails) such pools get a device-local mirror.
    bool allow_host_import{true};
    /// Back memory pools (host import of the chunk around them, device-local buffers of GPU-only
    /// pools) on a worker thread right after their creation instead of at the first command that
    /// needs them.
    bool background_backing{true};
    /// How long a stream waits for something nobody provides (a fence nobody has submitted, an
    /// event nobody signals) before it goes on without the wait; counted and logged.
    std::chrono::milliseconds parked_timeout{5000};
    /// Evaluate render conditions on the GPU with VK_EXT_conditional_rendering. When false (or
    /// when the extension is missing) each condition flushes the work so far, waits for it and
    /// decides on the CPU.
    bool allow_conditional_rendering{true};
    /// SAMPLES_PASSED occlusion queries inside the render passes, reports written when the pass
    /// ends (needs the host query reset). When false a query spans whole passes and every report
    /// ends the pass. The renderer's context takes the nvn_in_pass_queries setting (on).
    bool in_pass_queries{false};
    /// Early submission of a long submission's recorded part (EDEN_NX2_NVN_EARLY_SUBMIT_US). Tests
    /// that count render passes turn it off: a piece ends the open pass after a recording time.
    bool early_submit{true};
    /// Test hook: every queue submission with a sequence number from this on reports
    /// VK_ERROR_DEVICE_LOST (the real submission has already run), exercising the loss handling.
    /// 0 = off.
    u64 simulate_device_loss_from_seq{0};
    /// Call Device::ReportLoss on a device loss (a log line, then a 15 s wait for the log and
    /// Nsight Aftermath to flush). Tests turn it off.
    bool report_device_loss{true};
    /// Persistent VRAM mirrors of imported pools for storage bindings and indirect arguments
    /// (DESIGN.md "Pool mirrors"; modes as EDEN_NX2_NVN_POOL_MIRROR: 0 off, 1 written storage
    /// ranges create them, 2 every storage range does); also needs guest memory that tracks CPU
    /// writes (GuestMemory::Track). Unset: EDEN_NX2_NVN_POOL_MIRROR decides.
    std::optional<u32> pool_mirror{};
    /// Test override for EDEN_NX2_TRY_STORAGE_VRAM (on by default where the device supports it).
    std::optional<bool> try_storage_vram{};
    /// Test override for EDEN_NX2_TRY_STORAGE_VRAM_MAX_MIB.
    std::optional<u64> try_storage_vram_max_bytes{};
    /// Test overrides for EDEN_NX2_TRY_VTX_MIRROR / EDEN_NX2_TRY_UBO_MIRROR (read-only pool
    /// mirrors for vertex, index and uniform ranges; they also need `pool_mirror` != 0).
    std::optional<bool> vtx_mirror{};
    std::optional<bool> ubo_mirror{};
    /// Test overrides for EDEN_NX2_TRY_VTXCOV_* (VtxCovSwitch bits, nvn_vtxcov.h; they need
    /// `vtx_mirror`) and EDEN_NX2_NVN_VTXCOV_DIAG.
    std::optional<u32> vtxcov{};
    std::optional<bool> vtxcov_diag{};
    /// Dispatches are ordered after earlier dispatches' shader writes only by the game's barriers
    /// (NVN semantics; DESIGN.md "Dispatch ordering"). Unset: EDEN_NX2_TRY_DISPATCH_ORDER decides.
    std::optional<bool> dispatch_order_by_barriers{};
    /// A guest flush of a pool mirror's range (nvnMemoryPoolFlushMappedRange) only counts for the
    /// pages the page tracker saw the guest write or the mirror does not hold. Unset:
    /// EDEN_NX2_TRY_FLUSH_TRACKED decides (on unless 0).
    std::optional<bool> flush_tracked{};
    /// Mirror pools (Pool::Backing::Mirror) hash only the guest pages the page tracker saw written
    /// (or that something it does not see wrote) instead of every page of every range at each
    /// submission; needs GuestMemory::Track. Unset: on unless EDEN_NX2_NVN_MIRROR_HASH_ALWAYS.
    std::optional<bool> mirror_hash_tracked{};
    /// Milliseconds a guest nvnMemoryPoolInvalidateMappedRange waits for the GPU's unfinished
    /// writes to its range (0 = off). Unset: EDEN_NX2_TRY_INVALIDATE_WRITEBACK decides.
    std::optional<u32> invalidate_wait_ms{};
    /// Test override for EDEN_NX2_TRY_RAW_KEEP_ENTRIES (raw-global draws keep mirror entries live).
    std::optional<bool> raw_keep_entries{};
    /// Test override for EDEN_NX2_TRY_REFUSED_SHADOW (copy shadows for written ranges of pools
    /// without a mirror while pool mirrors are on).
    std::optional<bool> refused_shadow{};
    /// The status bar's shader build counter (PipelineCacheOptions::shader_notify). The renderer's
    /// context passes the GPU's notifier; tests leave it null.
    VideoCore::ShaderNotify* shader_notify{};
    /// The loading screen's progress sink (PipelineCacheOptions::load_progress). The renderer's
    /// context passes the GPU's; tests leave it null.
    VideoCore::LoadProgressSink* load_progress{};
    /// Cached storages with an image above this many bytes are trimmed (Context::Impl::
    /// TrimStorageCache). Unset: EDEN_NX2_NVN_STORAGE_BUDGET_MIB, else 6 GiB.
    std::optional<u64> storage_cache_budget_bytes{};
    /// Override the device-local sparse page limit.
    std::optional<u64> sparse_device_budget_bytes{};
    /// Presents after which an idle render target that can be rebuilt leaves the cache even under
    /// the budget (0: never). Unset: EDEN_NX2_NVN_TRIM_AGE_PRESENTS, else 0 (off: the age rule dropped GPU-written lighting targets in OoT).
    std::optional<u64> storage_trim_age_presents{};
    /// The trim of before the age and device budget rules. Unset: EDEN_NX2_NVN_TRIM_LEGACY.
    std::optional<bool> storage_trim_legacy{};
    /// Device-local allocations require device-local memory (MemoryAllocator::TrackSpills).
    /// Unset: EDEN_NX2_TRY_VMA_DEVICE_STRICT.
    std::optional<bool> vma_device_strict{};
    /// Runs on the replay thread after every queue submission. The renderer's context sets it to
    /// feed TurboMode (Force maximum clocks): that thread keeps the GPU at full clocks only while
    /// submissions keep coming, and the replay's submissions do not go through the scheduler.
    std::function<void()> on_queue_submit{};
    /// The replay's own buffer commands (shadow and mirror copies, write-back merges), the inline
    /// uniform updates and the pass-begin layout transitions skip barriers that order nothing
    /// (nvn_barrier_track.h; DESIGN.md "Barrier merging"). The value of EDEN_NX2_TRY_BARRIER_MERGE
    /// (0 off, 1 all, 2..4 one part: BarrierMergeMask). Unset: the switch decides.
    std::optional<u32> barrier_merge{};
    /// Interval of the opt-in shader refresh mode (zero: every present, for headless diagnostics).
    std::chrono::seconds shader_refresh_interval{60};
};

/// Counters for things the replay cannot do (yet) or had to work around. All monotonic.
struct ContextStats {
    u64 pools_imported{};
    u64 pools_mirrored{};
    u64 pools_mirrored_discontiguous{}; ///< ... guest storage is several host runs
    u64 pools_mirrored_no_import{};     ///< ... host import unavailable or failed
    u64 pools_split_imported{};         ///< ... several host runs, each imported (raw global)
    u64 split_runs{};                   ///< ... in this many runs
    u64 mirror_uploads{};               ///< range uses that copied guest pages into a mirror
    u64 mirror_upload_runs{};           ///< ... as this many copies
    u64 mirror_upload_bytes{};          ///< ... of this many bytes
    u64 mirror_ranges_clean{};          ///< range uses that needed no copy
    u64 mirror_pages_hashed{};          ///< guest pages compared with what the mirror holds
    u64 mirror_pages_trusted{};         ///< ... not compared: the page tracker saw no write
    u64 staging_chunks_reused{};
    u64 mirror_pages_gpu_owned{};       ///< ... not compared: pending GPU writes own the page
    // Pool mirrors (EDEN_NX2_NVN_POOL_MIRROR; Context::Impl::AtomicStats vram_* for the meaning).
    u64 vram_mirrors{};
    u64 vram_mirror_refused{};
    u64 vram_copy_in_runs{};
    u64 vram_copy_in_bytes{};
    u64 vram_bindings{};
    u64 vram_bindings_host{};
    u64 vram_write_ranges{};
    u64 vram_write_bytes{};
    u64 vram_write_backs_plain{};
    u64 vram_invalidate_all{};
    u64 vram_mirrors_read_only{};
    u64 vram_mirrors_promoted{};
    u64 vram_invalidate_read_only{};
    u64 vram_ro_written_through{};
    u64 vram_raw_alias_rebuilds{};
    u64 vram_vertex_ranges{};
    u64 vram_vertex_budget_skips{};
    /// EDEN_NX2_NVN_VTXCOV_DIAG: vertex and index binds (summed) by VtxCov (nvn_vtxcov.h).
    std::array<u64, VTXCOV_REASONS> vtxcov_binds{};
    u64 vram_flush_pages_kept{};   ///< flushed pages a mirror kept (EDEN_NX2_TRY_FLUSH_TRACKED)
    u64 vram_flush_pages_taken{};  ///< flushed pages the mirror had to copy in again
    u64 pools_device_local{};
    u64 pools_unsupported{};
    u64 host_import_failures{};
    u64 textures_created{};
    u64 textures_failed{};
    u64 textures_transcode_unsupported{};
    u64 alias_unsupported{};
    u64 alias_events{};
    // Memory work (nvn_storage.cpp, nvn_upload.cpp).
    u64 storages_created{};
    u64 storages_reused{};
    u64 storages_evicted{};
    u64 storages_superseded_evicted{}; ///< ... of which GPU-written ones a later GPU write replaced
    u64 storages_aged_evicted{};       ///< ... of which idle render targets past the age limit
    u64 storages_pressure_evicted{};   ///< ... of which ones under the budget, for the device budget
    u64 storage_cache_bytes{};         ///< image bytes the storage cache holds now (estimate)
    u64 alias_transfers{};
    u64 alias_byte_transfers{}; ///< alias transfers made through the pool's bytes (nvn_alias_gpu.h)
    u64 alias_overlaps_seen{};
    u64 uploads{};
    u64 upload_bytes{};
    u64 cpu_checks{};
    u64 cpu_changes{};
    u64 host_write_uploads{};
    u64 decode_ns{};
    u64 decode_bytes{};
    u64 hash_ns{};
    u64 hash_bytes{};
    u64 depth_upload_unsupported{};
    u64 virtual_maps{};
    u64 virtual_unresolved{};
    u64 virtual_sparse_binds{};
    u64 sparse_write_pool_checks{};
    u64 driver_bank_uploads{};
    u64 handle_bank_uploads{};
    u64 pipeline_state_reuses{};
    u64 texels_flushed{};
    u64 ops_unsupported{};
    u64 ops_failed{};
    u64 submissions{};
    u64 presents{};
    u64 held_back_submissions{}; ///< Submissions that waited for a sync nobody had fenced yet.
    u64 parked_timeouts{};       ///< Submissions run without a wait whose fence was never submitted.
    // Events (WaitEvent / SignalEvent records, DESIGN.md "Event waits").
    u64 event_signals{};       ///< SignalEvent records replayed.
    u64 event_waits{};         ///< WaitEvent records the event model satisfied.
    u64 event_waits_blocked{}; ///< ... that suspended their stream at least once.
    u64 waits_unsatisfied{};   ///< WaitEvent records released by the timeout (nothing signalled).
    // Indirect draws (NvnIndirect).
    u64 indirect_draws{};         ///< Indirect draw commands recorded.
    u64 indirect_counted_draws{}; ///< ... of which with a GPU draw count.
    u64 indirect_unsupported{};   ///< Indirect draws dropped.
    u64 mesh_task_draws{};        ///< Mesh-task draw commands recorded (direct and indirect).
    // Raw global memory (guest address -> device address table in the driver bank).
    u64 raw_global_draws{};          ///< Draws/dispatches whose shaders use the table.
    u64 raw_global_pools_skipped{};  ///< Mirror/virtual/unbacked pools left out of the table.
    u64 raw_global_ranges_dropped{}; ///< Ranges over the table's capacity, left out.
    u64 raw_global_rebuilds{};
    // Storage shadows (written storage ranges of imported pools bound through device memory).
    u64 shadow_creates{};
    u64 shadow_redirects{};
    u64 shadow_flush_segment{};
    u64 shadow_flush_conflict{};
    u64 shadow_flush_raw{};
    u64 shadow_bytes_in{};
    u64 shadow_bytes_out{};
    u64 shadow_skipped_alias{};
    u64 shadow_skipped_open_pass{};
    u64 shadow_skipped_no_room{};
    // Render conditions and counters (NvnIndirect).
    u64 render_conditions_gpu{};        ///< Predicates evaluated on the GPU (computed ahead too).
    u64 render_conditions_prefetched{}; ///< Conditions served by a predicate computed ahead.
    u64 render_conditions_cpu{};        ///< Conditions evaluated on the CPU (fallback).
    u64 render_conditions_false{};      ///< CPU-evaluated conditions that turned rendering off.
    u64 render_conditions_unresolved{}; ///< Report address outside every pool (drew anyway).
    u64 render_ops_gated{};             ///< Draws/clears recorded under conditional rendering.
    u64 render_ops_skipped{};           ///< Draws/clears dropped by SetRenderEnable(false).
    u64 counter_reports_timestamp{};
    u64 counter_reports_samples{};
    u64 counter_reports_value{};
    u64 counter_reports_unsupported{};  ///< Counter types written as 0.
    u64 counter_reports_samples_uncounted{}; ///< SAMPLES_PASSED reports after uncounted draws.
    u64 counter_occlusion_folds{};
    u64 device_lost{};                  ///< 1 once the Vulkan device was lost.
    u64 dropped_after_device_loss{};    ///< Submissions/presents completed without running.
};

/// Executes NVN command lists on Vulkan. Implements the sink the recorder submits to.
///
/// Threads: the caller's thread only queues work; "NvnReplay" records and submits it in global
/// order, "NvnComplete" waits for the GPU, applies mirror write-backs and runs the completion
/// callback in seq order.
class Context final : public VideoCommon::Nvn::Sink {
public:
    /// The renderer's constructor. The context allocates from its own thread-safe VMA allocator,
    /// because the renderer's one is externally synchronized and the replay runs on its own
    /// thread; `scheduler` only lends its submit mutex, which serializes queue submission with the
    /// renderer's own submissions. `shader_notify` is the GPU's build counter the pipeline
    /// cache's compiles are counted in; `load_progress` is where its disk preload reports.
    Context(const Device& device, Scheduler& scheduler, PresentManager& present_manager,
            Core::Frontend::EmuWindow& render_window, VideoCore::ShaderNotify& shader_notify,
            VideoCore::LoadProgressSink& load_progress,
            std::function<void()> on_queue_submit = {});

    /// Headless / test constructor. `submit_mutex` serializes queue submission with the other users
    /// of the device's graphics queue; `present` may be null (Present then only completes).
    Context(const Device& device, std::mutex& submit_mutex, std::unique_ptr<PresentTarget> present,
            ContextOptions options = {});

    ~Context() override;

    void CreatePool(const VideoCommon::Nvn::PoolDesc& desc) override;
    void CreateTexture(const VideoCommon::Nvn::TextureDesc& desc) override;
    void CreateSampler(const VideoCommon::Nvn::SamplerDesc& desc) override;
    void CreateProgram(const VideoCommon::Nvn::ProgramDesc& desc) override;
    void Destroy(u64 serial) override;
    void Submit(VideoCommon::Nvn::Submission&& submission) override;
    void Present(const VideoCommon::Nvn::PresentRequest& request) override;
    void MapVirtual(u64 pool_serial, std::vector<VideoCommon::Nvn::VirtualMapping>&& mappings) override;
    void NotifyCpuWrite(u64 guest_address, u64 size) override;
    void NotifyHostWrite(u64 guest_address, u64 size) override;
    [[nodiscard]] u64 GpuWriteMarker() override;
    [[nodiscard]] VideoCommon::Nvn::GpuWriteState PollGpuWrites(u64 marker, u64 guest_address,
                                                                u64 size, u64 max_seq) override;
    void NotifyTextureWritten(u64 texture_serial) override;
    void NotifyEventCpu(u64 address, VideoCommon::Nvn::EventCpuChange change, u32 value) override;
    bool FlushTextureToGuest(u64 texture_serial) override;
    bool Wait(u64 seq, std::chrono::nanoseconds timeout) override;
    void SetCompletionCallback(std::function<void(u64 seq)> callback) override;
    void SetFailureCallback(std::function<void(std::string reason)> callback) override;
    void SetGuestMemory(VideoCommon::Nvn::GuestMemory* memory) override;
    void SetDeviceModes(u32 window_origin, u32 depth_mode) override;
    void SetTitleId(u64 title_id) override;

    /// Serves the renderer's pending screenshot request (RendererBase::RequestScreenshot) from the
    /// next presented frame: the frame is read back and rendered into the request's buffer, then
    /// its callback runs (callback(false): rows top first). `settings` must outlive the context.
    void SetScreenshotRequests(VideoCore::RendererSettings& settings);

    [[nodiscard]] ContextStats Stats() const;
    void CollectStats(std::vector<VideoCommon::Nvn::SinkStat>& out) const override;
    [[nodiscard]] std::string DescribeState() const override;
    [[nodiscard]] std::string DescribeMemory() const override;
    void NoteTextureHolders(VideoCommon::Nvn::TextureHolderReport&& report) override;

    /// Implementation detail, defined in nvn_impl.h.
    struct Impl;

private:
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan::Nvn
