// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>

#include <fmt/format.h>

#include "common/assert.h"
#include "common/boot_counter.h"
#include "common/cityhash.h"
#include "common/diag_request.h"
#include "common/frame_capture_list.h"
#include "common/frame_stats.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/perf_stats.h"
#include "video_core/game_frame_size.h"
#include "video_core/nvn/timestamp_mode.h"
#include "video_core/nvn_draw_probe.h"
#include "video_core/nx2.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_vulkan/nvn/nvn_condition_prefetch.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_settings_notice.h"
#include "video_core/renderer_vulkan/nvn/nvn_sparse.h"
#include "video_core/renderer_vulkan/nvn/nvn_submit_policy.h"
#include "video_core/renderer_vulkan/nvn/nvn_upload.h"
#include "video_core/renderer_vulkan/present/filters.h"
#include "video_core/renderer_vulkan/present/fsr.h"
#include "video_core/renderer_vulkan/present/fsr3.h"
#include "video_core/renderer_vulkan/present/layer.h"
#include "video_core/renderer_vulkan/present/sgsr.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/present/window_adapt_pass.h"
#include "video_core/renderer_vulkan/vk_present_manager.h"
#include "video_core/shader_log.h"
#include "video_core/textures/decoders.h"
#include "video_core/upscaler.h"

namespace Vulkan::Nvn {
namespace {

constexpr VkPipelineStageFlags ALL_STAGES = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
constexpr VkAccessFlags ALL_WRITES = VK_ACCESS_MEMORY_WRITE_BIT;
constexpr VkAccessFlags ALL_ACCESS = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
/// Tiled uploads below this many linear bytes untile on the CPU: a few dozen microseconds there,
/// less than a buffer allocation, a dispatch and a barrier more (switch2-recomp's GPU untile uses
/// the same 64 KiB floor).
constexpr u64 UNTILE_GPU_MIN_BYTES = u64{64} << 10;
/// Binary semaphores the present submission can signal: the presented frame and the frames
/// generated for it (SubmitSegments adds the queue's timeline semaphore to them).
constexpr size_t MaxPresentSignals = 15;

bool IsTimestampReport(const VC::ReportCounterPayload& p) {
    return p.counter_type == CounterType::Timestamp || p.counter_type == CounterType::TimestampTop;
}

u32 Mip(u32 size, u32 level) {
    return (std::max)(size >> level, 1u);
}

u32 CeilDiv(u32 value, u32 divisor) {
    return (value + divisor - 1) / divisor;
}

struct LevelExtent {
    u32 width;
    u32 height;
    u32 depth;
};

LevelExtent ExtentOfLevel(const Storage& s, u32 level) {
    return {
        Mip(s.extent.width, level),
        Mip(s.extent.height, level),
        s.type == VK_IMAGE_TYPE_3D ? Mip(s.extent.depth, level) : 1u,
    };
}

VkImageAspectFlags TransferAspect(const Storage& s) {
    if (s.host.depth) {
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    if (s.host.stencil) {
        return VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    return VK_IMAGE_ASPECT_COLOR_BIT;
}

/// Aspects an image-to-image copy has to name: both planes of a depth/stencil image.
VkImageAspectFlags CopyAspect(const Storage& s) {
    if (s.host.depth && s.host.stencil) {
        return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    return TransferAspect(s);
}

/// A view and framebuffer kept alive until the GPU is done with a partial clear.
struct ClearHolder {
    vk::ImageView view;
    vk::Framebuffer framebuffer;
};

struct StagingRange {
    VkBuffer buffer;
    u64 offset;
    u8* pointer;
};

struct CopyGeometry {
    std::vector<VkBufferImageCopy> regions;
    u64 span{}; ///< Bytes the buffer side covers, from the first region's buffer offset.
    bool padded{};
};

/// Records one submission (or a present) into one or more command buffers and submits them.
class Recorder final : public DrawHost {
public:
    Recorder(Context::Impl& impl_, u32 queue_index_, InFlight& flight_, bool present = false)
        : impl{impl_}, queue_index{queue_index_}, flight{flight_},
          queue{impl_.GetQueue(queue_index_)}, draw{impl_.GetDrawContext(queue_index_)} {
        if (impl.gpu_profiler && !flight.gpu_prof) {
            flight.gpu_prof = impl.gpu_profiler->NewFlight(present);
            if (flight.gpu_prof) {
                flight.gpu_prof->seq = flight.seq;
                if (impl.costsplit) {
                    if (const auto& slot = impl.costsplit->Current()) {
                        flight.gpu_prof->costsplit_slot = static_cast<s64>(slot->index);
                        flight.gpu_prof->costsplit_measured = slot->measured;
                    }
                }
            }
        }
        BeginSegment();
        ApplyPendingStandins();
        draw.BeginSubmission(*this, EmptySubmission());
    }

    static const VC::Submission& EmptySubmission() {
        static const VC::Submission empty{};
        return empty;
    }

    /// The submission whose descriptor pool snapshots the draw path resolves handles against.
    void BindSubmission(const VC::Submission& submission) {
        draw.BeginSubmission(*this, submission);
    }

    ~Recorder() {
        if (!finished) {
            Abort();
        }
    }

    /// Replays `list` from word `start`. True: it ran to its End record. False: it stopped at a
    /// WaitEvent record the event model does not satisfy; `stop` is that record's word index, and
    /// the wait is looked at again when the list is replayed from `stop` (after something else may
    /// have changed the event). With `release` set the first such wait is passed instead (logged
    /// and counted); `release` is cleared when it was used.
    bool RecordList(std::span<const u64> list, size_t start, size_t& stop, bool& release);
    /// Records replayed so far (the record a RecordList call stopped at is not one).
    [[nodiscard]] u64 RecordsReplayed() const {
        return records_replayed;
    }
    /// The stream stops at a wait: everything recorded so far goes to the GPU, so that a stream
    /// that runs meanwhile, and the waits it makes, only name submitted segments (the storage
    /// layouts and timeline values follow the order of recording, which must be the order of
    /// submission). Does nothing when nothing was recorded since the last submitted piece.
    void Suspend();
    /// EDEN_NX2_NVN_OPLOG diagnostics.
    void LogOp(VC::Op op, const void* payload, size_t payload_size);
    u32 op_log_count{0};
    /// GPU command capture: where the capture wants the presented frame as a PNG (empty: no).
    /// Set by ExecutePresent before RecordPresent.
    std::filesystem::path gpu_capture_png;
    /// EDEN_NX2_NVN_WATCH: copies the watched range back for a log line at completion (outside a
    /// pass).
    void WatchSnapshot(std::string tag, u64 peek_address = 0, u32 size_override = 0);
    [[nodiscard]] bool RecordPresent(Texture& texture, Frame& frame, std::vector<Frame*>& generated,
                                     const VideoCommon::Nvn::PresentCrop& window_crop);
    /// Copies the present source into the context's scratch image in its raw (UNORM) format. On
    /// return the image is in the TRANSFER_SRC layout, or in GENERAL with a sampled view
    /// (`sampled`: the presentation passes read it).
    VkImage RawPresentCopy(Storage& s, VkFormat raw_format, bool sampled = false);
    /// A readback of the presented picture at a size of its own (captures, screenshots).
    struct PictureReadback {
        vk::Image image;
        vk::Buffer buffer;
        Picture picture;
    };
    /// Creates the image and host buffer of a `width` x `height` readback of a present into
    /// `frame` (the frame's format, so the values are the ones the present blit writes).
    PictureReadback CreatePictureReadback(const Frame& frame, u32 width, u32 height);
    /// Records the readback: black, then `source` (TRANSFER_SRC layout; null: no picture) blitted
    /// into `target` with `filter`, copied to the host buffer. The image is released with the
    /// flight.
    void RecordPictureReadback(PictureReadback& readback, VkImage source,
                               VkImageLayout source_layout, const PixelRect& source_rect,
                               VkFilter filter, const PixelRect& target);
    void Finish(std::span<const VkSemaphore> binary_signals);
    /// The submission being replayed (the lists after the current one are looked at to decide
    /// whether occlusion samples are still wanted).
    void SetLookahead(const VC::Submission& submission, size_t list) {
        lookahead_submission = &submission;
        lookahead_list = list;
    }

    // ---- DrawHost (services for the draw path) ----
    Context::Impl& HostImpl() override {
        return impl;
    }
    InFlight& HostFlight() override {
        return flight;
    }
    vk::CommandBuffer HostCmd() override {
        return Cmd();
    }
    void HostFlushBarrier() override {
        FlushPending(true);
    }
    void HostBarrierBeforePass() override {
        if (impl.draw_resources->Cache().SyncDiet()) {
            FlushPending(true);
        }
    }
    void HostMarkWritten() override {
        barrier_pending = true;
    }
    void HostMarkDispatchWritten() override {
        dispatch_writes_pending = true;
    }
    void HostFlushBarrierBeforeDispatch() override {
        if (barrier_pending || !buffer_ops.Empty()) {
            MemoryBarrier(true);
            return;
        }
        if (!dispatch_writes_pending) {
            return;
        }
        impl.stats.dispatch_barriers_left.fetch_add(1, std::memory_order_relaxed);
        if (!impl.dispatch_order_by_barriers) {
            MemoryBarrier(true);
        }
    }
    u32 HostBarrierMerge() const override {
        return impl.barrier_merge;
    }
    void HostBeginBufferOp(std::span<const BufferAccess> accesses) override {
        if ((impl.barrier_merge & BARRIER_MERGE_BUFFERS) == 0) {
            FlushPending(true);
            return;
        }
        // Something the tracker does not know wrote (a game dispatch, a draw, an image copy, any
        // op that marked the barrier): the old rule. Else the ranges decide.
        if (barrier_pending || dispatch_writes_pending) {
            MemoryBarrier(true);
            impl.stats.barrier_merge_ops_unknown_writes.fetch_add(1, std::memory_order_relaxed);
        } else if (buffer_ops.Conflicts(accesses)) {
            MemoryBarrier(true);
            impl.stats.barrier_merge_ops_hazard.fetch_add(1, std::memory_order_relaxed);
        } else if (buffer_ops.Empty()) {
            impl.stats.barrier_merge_ops_clean.fetch_add(1, std::memory_order_relaxed);
        } else {
            impl.stats.barrier_merge_ops_left_out.fetch_add(1, std::memory_order_relaxed);
        }
        buffer_ops.Note(accesses);
    }
    void HostEndBufferOp() override {
        if ((impl.barrier_merge & BARRIER_MERGE_BUFFERS) == 0) {
            barrier_pending = true;
        }
        // On: the ranges noted by HostBeginBufferOp are the pending writes.
    }
    void HostNeedHostReadBarrier() override {
        host_barrier_needed = true;
    }
    void HostTransition(Storage& storage, VkImageLayout layout) override {
        const std::pair<Storage*, VkImageLayout> image{&storage, layout};
        PrepareImagesSpan(std::span{&image, 1}, {}, true, true);
    }
    void HostTransitionMany(std::span<const std::pair<Storage*, VkImageLayout>> images,
                            bool note_writes) override {
        // Every caller flushes the pending barrier itself after the transitions (before the
        // dispatch: HostFlushBarrierBeforeDispatch), so images already in layout record nothing.
        PrepareImagesSpan(images, {}, false, true, note_writes);
    }
    void HostNoteImageWrite(Storage& storage) override {
        NoteGpuWrite(storage);
    }
    void HostNoteTouched(const std::shared_ptr<Storage>& storage) override {
        NoteTouched(storage);
    }
    bool HostEnsureContents(Storage& storage, bool will_overwrite) override {
        return EnsureContents(storage, will_overwrite);
    }
    bool HostEnsureAttachmentContents(Storage& storage) override;
    bool HostNeedsCommands(Storage& storage, bool will_overwrite) override {
        storage.last_used_seq = flight.seq;
        storage.last_used_present = impl.stats.presents.load(std::memory_order_relaxed);
        return impl.StorageNeedsCommands(storage, will_overwrite, flight.seq);
    }
    bool HostNeedsNativeUpdate(Storage& storage) override {
        storage.last_used_seq = flight.seq;
        storage.last_used_present = impl.stats.presents.load(std::memory_order_relaxed);
        return impl.StorageNeedsCommands(storage, false, flight.seq, true);
    }
    bool HostPrepareScaledTarget(Storage& native) override {
        return PrepareScaledTarget(native);
    }
    void HostMakeNativeCurrent(Storage& storage) override {
        MakeNativeCurrent(storage);
    }
    bool HostResolvePoolRange(u64 address, u64 size, Pool*& pool, u64& offset,
                              std::string_view op) override {
        return ResolvePoolRange(address, size, pool, offset, op, false);
    }
    std::span<const u64> HostRecordsAhead() override {
        return next_record < current_list.size() ? current_list.subspan(next_record)
                                                 : std::span<const u64>{};
    }
    std::span<const std::vector<u64>> HostFollowingLists() override {
        if (lookahead_submission == nullptr) {
            return {};
        }
        return std::span{lookahead_submission->lists}.subspan(lookahead_list + 1);
    }
    void HostUploadRange(Pool& pool, u64 offset, u64 size) override {
        UploadRange(pool, offset, size);
    }
    bool HostMirrorNeedsUpload(Pool& pool, u64 offset, u64 size) override {
        return MirrorNeedsUpload(pool, offset, size);
    }
    void HostNotePoolWrite(Pool& pool, u64 offset, u64 size) override {
        NotePoolWrite(pool, offset, size);
    }
    void HostNoteMirrorWriteBack(Pool& pool, u64 offset, u64 size) override {
        NotePoolWrite(pool, offset, size, false);
    }
    void HostNoteRawGlobalWrite(bool written_through) override {
        // Raw global tables resolve every VRAM-mirrored pool to its device-local mirror; writes to
        // non-mirrored pools never make a mirror stale. Keep mirror pages current instead of
        // copying their GPU-written contents back in from stale guest memory.
        condition_prefetch.Clear();
        host_barrier_needed = true;
        flight.raw_writes = true;
        if (!written_through) {
            flight.raw_writes_unaliased = true;
        }
        // A raw pointer reaches any range of the table: no submit-time uniform copy of them.
        impl.uniform_snapshots.NoteRawWrite(impl.raw_table.generation, impl.raw_table.entries,
                                            flight.seq);
    }
    void HostNoteShadowedWrite(const Pool& pool, u64 offset, u64 size) override {
        impl.NoteNonInlinePoolWrite(pool.address + offset, size, flight.seq);
        // The pool bytes take these writes when the shadow is copied back; a later condition on
        // them would copy it back first (ResolvePoolRange), a predicate computed now would not.
        condition_prefetch.Invalidate(pool, offset, size);
        // A uniform binding of these bytes reads them through the pool from now on (the copy
        // back comes before it), not from a submit-time copy.
        impl.uniform_snapshots.NoteWrite(pool.address + offset, pool.address + offset + size,
                                         flight.seq);
    }
    void HostUnsupported(std::string_view op, const std::string& reason) override {
        Unsupported(op, reason);
    }
    void HostFailed(std::string_view op, const std::string& reason) override {
        Failed(op, reason);
    }
    bool HostBeginRenderCondition() override {
        return BeginRenderCondition();
    }
    void HostEndRenderCondition() override {
        EndRenderCondition();
    }
    bool HostConditional() const override {
        return render_state == RenderState::Conditional;
    }
    /// A conditional-rendering region cannot straddle a render pass begin/end: the draw path ends
    /// it before it changes passes and starts it again inside the new pass.
    bool HostSuspendCondition() override {
        if (!condition_open) {
            return false;
        }
        impl.counters_engine->EndConditional(Cmd());
        condition_open = false;
        return true;
    }
    void HostResumeCondition() override {
        impl.counters_engine->BeginConditional(Cmd(), condition_slot);
        condition_open = true;
    }
    void HostBeforeDraw() override {
        if (queue.occlusion_counting && !occlusion && InPassCounters()) {
            // The draw counts: a query of its own inside the pass (it ends with the pass, at a
            // report or at a reset).
            occlusion = impl.counters_engine->BeginOcclusion(Cmd(), Counters());
        }
    }
    void HostPassEnding() override {
        if (occlusion && InPassCounters()) {
            impl.counters_engine->EndOcclusion(Cmd(), Counters(), *occlusion);
            counter_batch.AddQuery(*occlusion);
            occlusion.reset();
        }
    }
    void HostPassEnded() override {
        // Reports must be written before anything outside the pass can read them; a batch of
        // queries and resets only (no report yet) waits for the next pass, up to half its room.
        if (counter_batch.HasReports() || !counter_batch.Room(CounterBatch::MAX_EVENTS / 2)) {
            ResolveCounterBatch();
        }
    }
    void HostPauseCounting() override {
        if (!InPassCounters()) {
            // The query spans passes: the clear's own pass must not count into it.
            FoldOcclusion();
        }
    }
    bool HostReportPending(const Pool& pool, u64 offset, u64 size) override {
        return counter_batch.Writes(&pool, offset, size);
    }
    bool HostReportsPending() override {
        return counter_batch.HasReports();
    }
    void HostDumpImage(Storage& storage, u32 level, u32 layer, VkImageAspectFlags aspect,
                       Debug::PendingDump dump) override {
        const u32 bytes = Debug::PixelBytes(dump.format, aspect);
        if (!storage.image || bytes == 0) {
            return;
        }
        dump.width = (std::max)(storage.extent.width >> level, 1u);
        dump.height = (std::max)(storage.extent.height >> level, 1u);
        dump.aspect = aspect;
        InFlight::Dump entry;
        entry.buffer =
            impl.CreateHostBuffer(static_cast<VkDeviceSize>(dump.width) * dump.height * bytes,
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::Download);
        PrepareImages({{&storage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL}});
        FlushPending();
        const bool is_3d = storage.type == VK_IMAGE_TYPE_3D;
        const VkBufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {aspect, level, is_3d ? 0u : layer, 1},
            .imageOffset = {0, 0, is_3d ? static_cast<s32>(layer) : 0},
            .imageExtent = {dump.width, dump.height, 1},
        };
        Cmd().CopyImageToBuffer(*storage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *entry.buffer,
                                region);
        barrier_pending = true;
        host_barrier_needed = true;
        entry.info = std::move(dump);
        flight.dumps.push_back(std::move(entry));
    }
    void HostDumpTexture(Storage& storage, const DiagTextures::Job& job) override {
        // Per submission, so a request over many draws cannot pile up host memory.
        constexpr u64 MaxFlightBytes = u64{256} << 20;
        if (!storage.image || !storage.contents_valid) {
            DiagTextures::Skip(job, "the texture had no contents when its pass ended");
            return;
        }
        if (storage.samples != VK_SAMPLE_COUNT_1_BIT) {
            DiagTextures::Skip(job, "multisampled: not captured");
            return;
        }
        const VkImageAspectFlags aspect = TransferAspect(storage);
        if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT) {
            DiagTextures::Skip(job, "stencil-only: not captured");
            return;
        }
        const u32 width = storage.extent.width;
        const u32 height = storage.extent.height;
        DiagTextures::Readback readback{};
        readback.epoch = job.epoch;
        readback.slot = job.slot;
        readback.format = storage.host.format;
        readback.aspect = aspect;
        readback.width = width;
        readback.height = height;
        readback.pixel_format = storage.host.pixel_format;
        u64 size = 0;
        if (const u32 bytes = Debug::PixelBytes(storage.host.format, aspect); bytes != 0) {
            size = static_cast<u64>(width) * height * bytes;
        } else if (const u32 bc_bytes = BcStandinBlockBytes(storage.host); bc_bytes != 0) {
            // The BC blocks the stand-in holds, decoded as that BC format.
            const bool srgb = VideoCore::Surface::IsPixelFormatSRGB(storage.host.pixel_format);
            readback.bcn = true;
            using VideoCore::Surface::PixelFormat;
            switch (BcStandinOf(storage.host)) {
            case AstcStandin::Bc1:
                readback.pixel_format = srgb ? PixelFormat::BC1_RGBA_SRGB : PixelFormat::BC1_RGBA_UNORM;
                break;
            case AstcStandin::Bc7:
                readback.pixel_format = srgb ? PixelFormat::BC7_SRGB : PixelFormat::BC7_UNORM;
                break;
            case AstcStandin::Bc4:
                readback.pixel_format = PixelFormat::BC4_UNORM;
                readback.luminance = true;
                break;
            case AstcStandin::Bc5:
                readback.pixel_format = PixelFormat::BC5_UNORM;
                readback.luminance = true;
                break;
            default:
                readback.pixel_format = srgb ? PixelFormat::BC3_SRGB : PixelFormat::BC3_UNORM;
                break;
            }
            size = BcBlocksSize(width, height, 1, bc_bytes);
        } else if (VideoCore::Surface::IsPixelFormatBCn(storage.host.pixel_format)) {
            const BlockInfo block = FormatBlockInfo(storage.desc.format);
            if (block.bytes == 0) {
                DiagTextures::Skip(job, "block layout unknown: not captured");
                return;
            }
            readback.bcn = true;
            size = static_cast<u64>(CeilDiv(width, block.width)) * CeilDiv(height, block.height) *
                   block.bytes;
        } else {
            DiagTextures::Skip(job, fmt::format("host format {} cannot be converted: not captured",
                                                static_cast<s32>(storage.host.format)));
            return;
        }
        if (flight.texture_dump_bytes + size > MaxFlightBytes) {
            DiagTextures::Skip(job, "readback budget of the submission exceeded: not captured");
            return;
        }
        flight.texture_dump_bytes += size;
        InFlight::TextureDump entry;
        entry.buffer =
            impl.CreateHostBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::Download);
        PrepareImages({{&storage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL}});
        FlushPending();
        const VkBufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {aspect, 0, 0, 1},
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, 1},
        };
        Cmd().CopyImageToBuffer(*storage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *entry.buffer,
                                region);
        barrier_pending = true;
        host_barrier_needed = true;
        entry.info = readback;
        flight.texture_dumps.push_back(std::move(entry));
    }
    /// Render conditions (SetRenderEnable / SetRenderEnableConditional). The draw path wraps every
    /// draw, DrawTexture and ClearColor / ClearDepthStencil like this (compute dispatches are not
    /// gated, as in nxnative; they run outside any conditional region):
    ///     if (!BeginRenderCondition()) return;
    ///     ...record the op...
    ///     EndRenderCondition();
    /// false means rendering is disabled and nothing must be recorded. With a GPU condition active,
    /// the op is bracketed by vkCmdBegin/EndConditionalRenderingEXT (valid inside and outside a
    /// render pass). Also starts the SAMPLES_PASSED occlusion query before the first draw that
    /// follows a ResetCounter / ReportCounter, so call it before opening a render pass.
    [[nodiscard]] bool BeginRenderCondition();
    void EndRenderCondition();

private:
    struct Segment {
        VkCommandBuffer raw{};
        vk::CommandBuffer cmd;
        bool ended{};
        std::vector<VkSemaphore> wait_semaphores;
        std::vector<u64> wait_values;
        u64 signal_value{};
        u32 prof_begin{GpuProf::NONE}; ///< timestamp at the start of the command buffer
    };

    /// The command buffer of the open segment. A barrier request still waiting for its next
    /// command (EDEN_NX2_TRY_BARRIER_MERGE, `deferred_barrier`) is recorded first, so everything
    /// recorded through this handle comes after it. Whoever keeps the handle across a barrier
    /// request (HostFlushBarrier, HostTransition*, HostBeginBufferOp, HostBarrierBeforePass,
    /// HostFlushBarrierBeforeDispatch) must ask again after the request.
    vk::CommandBuffer Cmd() {
        if (deferred_barrier.pending) {
            EmitDeferredBarrier();
        }
        return segments.back().cmd;
    }

    void BeginSegment(bool keep_shadows = false);
    void EndSegment(bool flush_shadows = true);
    void Abort();

    /// Ends the render pass the draw path left open, if there is one. The only place that ends it:
    /// counters, conditions and EndSegment call it before any command that must be recorded
    /// outside a render pass. Nothing draws yet, so there is no pass to end.
    void EndDrawPass(BreakCause cause = BreakCause::OtherOp) {
        draw.EndPass(cause);
        FlushDeferredReports();
    }
    /// Zero-fills of timestamp reports written inside a render pass (a fill must be outside one),
    /// recorded as soon as the pass has ended.
    void FlushDeferredReports();
    /// A SAMPLES_PASSED report follows in the rest of the submission before the next reset.
    [[nodiscard]] bool SamplesReportFollows() const;
    /// `defer`: the barrier may wait for the next command (the draw path's requests, which
    /// re-fetch the command buffer after them); with EDEN_NX2_TRY_BARRIER_MERGE a request that
    /// finds one waiting joins it whether it defers or not (no command came between them).
    void MemoryBarrier(bool defer = false);
    void FlushPending(bool defer = false) {
        if (barrier_pending || dispatch_writes_pending || !buffer_ops.Empty()) {
            MemoryBarrier(defer);
        }
    }
    /// Records the barrier request waiting for its next command (`deferred_barrier`).
    void EmitDeferredBarrier();
    void PrepareImages(std::initializer_list<std::pair<Storage*, VkImageLayout>> images,
                       std::span<const VkImageMemoryBarrier> extra = {}) {
        PrepareImagesSpan(std::span{images.begin(), images.size()}, extra);
    }
    /// Transitions `images` (and records `extra`) in one barrier that also is the pending memory
    /// barrier. When none needs one: the pending barrier alone, unless `flush_if_none` is false.
    void PrepareImagesSpan(std::span<const std::pair<Storage*, VkImageLayout>> images,
                           std::span<const VkImageMemoryBarrier> extra = {},
                           bool flush_if_none = true, bool defer = false, bool note_writes = true);

    StagingRange AllocStaging(u64 size, u64 alignment);
    void UploadRange(Pool& pool, u64 offset, u64 size);
    bool MirrorNeedsUpload(Pool& pool, u64 offset, u64 size);
    u64 GuestPageHash(const Pool& pool, u64 page) const;
    /// Mirror pools: registers the pool with the page tracker at the first call, then turns the
    /// pages the guest CPU wrote since the last call into untrusted ones. False: no tracking (every
    /// page is hashed, nothing is trusted).
    bool MirrorTakeCpuWrites(Pool& pool);
    /// Classify the page against the pending GPU writes sampled at the first page check.
    MirrorWriteCoverage MirrorPageWrites(Pool& pool, u64 page, bool& pending_read);
    /// The replay wrote a pool range on the GPU (through the pool's own buffer unless
    /// `invalidate_mirror` is false: a pool mirror's write-back, after which the mirror and the
    /// guest bytes agree).
    void NotePoolWrite(Pool& pool, u64 offset, u64 size, bool invalidate_mirror = true,
                       u64 row_bytes = 0, u64 row_stride = 0, u32 row_count = 0);
    void NoteTouched(const std::shared_ptr<Storage>& storage);
    [[nodiscard]] bool HasLinearWritebackOverlapping(const Pool& pool, u64 begin, u64 end) const;
    /// Writes GPU-modified linear storages in CPU-readable pools back to their pools so guest
    /// CPU readers (and intra-submission buffer ops over those addresses) see the GPU's texels.
    void FlushLinearWritebacks(const Pool* filter_pool = nullptr, u64 filter_begin = 0,
                               u64 filter_end = 0);
    void WriteReadbacks();

    void Unsupported(std::string_view op, const std::string& reason);
    void Failed(std::string_view op, const std::string& reason);
    void LogOnce(const std::string& message);

    [[nodiscard]] Texture* FindTexture(u64 serial) const;
    bool ResolveRef(const VC::TextureRef& ref, Storage*& storage, std::shared_ptr<Storage>& owner,
                    u32& level, u32& layer, std::string_view op);
    /// `flush_shadows`: the caller reads or writes the range through host memory (draw-path callers
    /// decide for themselves, see DrawContext::ShadowPolicy).
    bool ResolvePoolRange(u64 address, u64 size, Pool*& pool, u64& offset, std::string_view op,
                          bool flush_shadows = true);
    bool MakeCopyGeometry(const Storage& s, u32 level, u32 layer, const VC::CopyRegion& region,
                          u32 row_stride, u32 image_stride, u64 buffer_offset,
                          CopyGeometry& out) const;
    /// Brings the storage's image up to date before the GPU reads or writes it: creates the image,
    /// uploads the guest bytes on first use, uploads them again when the CPU changed them, and
    /// copies in what overlapping textures wrote. `will_overwrite`: the whole storage is about to
    /// be overwritten without being read, so none of that is needed. False when the image cannot
    /// be created (the op must be dropped).
    /// `keep_scaled`: the caller reads the scaled twin when it is the newer image (present), so
    /// the guest-sized image is only made current when guest bytes have to change it anyway.
    /// Without it a storage whose twin is newer gets its guest-sized image from the twin first.
    bool EnsureContents(Storage& s, bool will_overwrite, bool keep_scaled = false);
    /// Resolution scaling (nvn_rescale.h). The guest-sized image made from the twin (a downscale
    /// blit) when the twin is newer; outside a render pass.
    void MakeNativeCurrent(Storage& s);
    /// Before a pass draws into the twin of `native`: creates the twin's image and, when the
    /// guest-sized image holds GPU-written pixels the twin does not, upscales them into it (the
    /// pass loads what is there). False when the twin cannot be made.
    bool PrepareScaledTarget(Storage& native);
    /// The storage a present reads: the twin when it is the newer image, else the storage.
    [[nodiscard]] Storage& PresentSource(Storage& guest);
    /// A copy between two images the Resolution setting scales, done on their twins. False when
    /// the copy is not one (nothing recorded).
    bool CopyTextureScaled(Storage& src, Storage& dst, u32 src_layer, u32 dst_layer, u32 layers,
                           const Rescale::CopyRegion& region, bool dst_covered);
    /// The GPU writes the storage (PrepareImages calls it for the layouts that write).
    void NoteGpuWrite(Storage& s);
    void UploadStorage(Storage& s);
    /// A tile-backed sparse storage (nvn_sparse.h): binds device memory for the tiles the pool
    /// mapped since the last use, takes it back from the unmapped ones, and uploads the mapped
    /// tiles whose bytes changed. `will_overwrite`: binds only.
    void SyncSparse(Storage& s, bool will_overwrite);
    /// Decodes ASTC on the GPU from imported guest memory or staged compressed bytes.
    /// Returns false when the CPU must decode it instead.
    bool UploadAstcOnGpu(Storage& s, const UploadPlan& plan, Pool& pool, const u8* data, u64 size);
    bool AstcGpuLevels(const Storage& s, const UploadPlan& plan, u64 input_offset,
                       const AstcGpuDecoder& decoder, std::vector<AstcGpuDecoder::Level>& levels);
    /// UploadAstcOnGpu for a BC stand-in: decodes into the BC scratch, chunk by chunk, and encodes.
    bool UploadAstcBcOnGpu(Storage& s, const UploadPlan& plan, Pool& pool, const u8* data,
                           u64 size);
    /// A BC stand-in's RGBA8 texels decoded by the CPU into `staging` (`plan`'s pieces): copied
    /// into the BC scratch, encoded and copied into the image.
    void UploadBcFromStaging(Storage& s, const UploadPlan& plan, const StagingRange& staging);
    /// Records every chunk of a BC stand-in upload: `fill` writes the chunk's RGBA8 texels into
    /// layers [0, chunk.layers) of the scratch (GENERAL layout), then they are encoded and copied
    /// into `s` at the chunk's level, layers and offset. `guest_contents`: the texels are the
    /// guest's bytes (an upload), not a GPU write. False unless every chunk was recorded.
    /// The flight's BcUploadTimes with room for one more upload (created on first use), null
    /// without timestamps or when it is full (counted untimed).
    BcUploadTimes* BcTimes();
    bool RecordBcUpload(Storage& s, std::span<const BcChunk> chunks, bool guest_contents,
                        const std::function<void(const BcChunk&, BcScratch&)>& fill);
    /// A whole upload of smart stand-in `s` (nvn_format.h SmartBc) from guest bytes `data`: their
    /// class must fit its format. A stand-in kept at BC3 while its bytes were zero takes the format
    /// of the first blocks; one whose new bytes do not fit (alpha in BC1/BC4, colour in BC4/BC5)
    /// is recreated as BC3 and stays so (logged once per texture, counted). A BC4/BC5 one with an
    /// sRGB view (Storage::standin_srgb_view) changes to BC1/BC3. False when the new image could
    /// not be created.
    bool RefitStandin(Storage& s, const u8* data, u64 size);
    /// Keeps smart stand-in `s` at BC3 for good (`op` copies or partially uploads it, which only
    /// works within one format and must not lose texels to a later change of format): recreated
    /// as BC3 when it is another format, its contents then uploaded again from the guest bytes by
    /// the caller's EnsureContents. False when the new image could not be created.
    bool PinStandin(Storage& s, std::string_view op);
    /// Replaces the image of BC stand-in `s` by an empty one of `standin`'s format: the old image
    /// and the views of every texture over it go after this submission.
    bool Restandin(Storage& s, AstcStandin standin, bool deferred = false);
    /// Applies the stand-in format changes a submission asked for (Storage::pending_standin): no
    /// binding of the new submission can hold a key of the old format yet.
    void ApplyPendingStandins();
    /// Untiles a block-linear storage that needs no decode on the GPU (nvn_untile_gpu.h): from its
    /// imported or GPU-only pool in place, else from a copy of the tiled bytes `data` in staging,
    /// into a device-local buffer copied into the image with the CPU path's regions. False (nothing
    /// recorded, the reason logged once) when the shader cannot take it; the caller then untiles
    /// on the CPU.
    bool UploadUntileOnGpu(Storage& s, const UploadPlan& plan, Pool& pool, const u8* data,
                           u64 size);
    /// EDEN_NX2_NVN_ASTC_VERIFY: runs everything recorded so far, reads the decoded image back and
    /// compares it with the CPU decode of the same bytes (logs differences).
    void VerifyAstcGpu(Storage& s, const UploadPlan& plan, const u8* data, u64 size);
    void RefreshFromGuest(Storage& s);
    void ResolveOverlaps(Storage& s);
    bool TransferOverlap(Storage& source, Storage& destination);
    /// Moves the bytes `source` and `destination` share in their pool from one image to the other
    /// through the pool's byte layout (nvn_alias_gpu.h), for textures whose layouts differ.
    /// False with `why` set when it cannot be expressed (nothing recorded).
    bool TransferOverlapBytes(Storage& source, Storage& destination, const char*& why);
    /// CopyBufferToTexture into a decoded stand-in (ASTC/BCn): decodes the source blocks on the
    /// CPU.
    bool DecodeBufferToTexture(Storage& s, u32 level, u32 layer,
                               const VC::CopyBufferTexturePayload& p, Pool& pool, u64 offset);

    void ClearTexture(const VC::ClearTexturePayload& p);
    void ClearTexturePartial(Storage& s, u32 level, u32 layer, u32 layer_count, u32 x, u32 y,
                             u32 width, u32 height, const VC::ClearTexturePayload& p);
    void ClearBuffer(const VC::ClearBufferPayload& p);
    void CopyBufferToBuffer(const VC::CopyBufferToBufferPayload& p);
    void CopyBufferToTexture(const VC::CopyBufferTexturePayload& p);
    void CopyTextureToBuffer(const VC::CopyBufferTexturePayload& p);
    void CopyTextureToTexture(const VC::CopyTextureToTexturePayload& p);
    void FenceSync(const VC::FenceSyncPayload& p);
    void WaitSync(const VC::WaitSyncPayload& p);
    void SignalEvent(const VC::SignalEventPayload& p);
    /// False when the event model does not satisfy the wait and `release` is not set.
    [[nodiscard]] bool WaitEvent(const VC::WaitEventPayload& p, bool& release);
    void ReportValue(const VC::ReportValuePayload& p);
    void ReportCounter(const VC::ReportCounterPayload& p);
    void ResetCounter(const VC::ResetCounterPayload& p);
    void SetRenderCondition(const VC::SetRenderConditionPayload& p);
    void EvaluateConditionOnCpu(Pool& pool_a, u64 offset_a, Pool& pool_b, u64 offset_b, u32 mode);
    /// GPU conditions: computes the predicate of the condition at `address` (its report pair
    /// resolved to pool_a/offset_a, pool_b/offset_b) together with those of the conditions that
    /// follow it in the list, in one batch, and makes it the current one. Outside a render pass.
    void PrefetchConditions(u64 address, u32 mode, Pool& pool_a, u64 offset_a, Pool& pool_b,
                            u64 offset_b);
    /// The report pair of the condition at `address` without logging a failure (a later
    /// condition the lookahead looks at; its own SetRenderCondition reports what fails).
    [[nodiscard]] bool ResolveReportPairQuietly(u64 address, Pool*& pool_a, u64& offset_a,
                                                Pool*& pool_b, u64& offset_b);
    /// The submission's counter state (query blocks, timestamp reports).
    [[nodiscard]] CounterFlight& Counters();
    /// Ends the open occlusion query, adding its samples to the queue's accumulator.
    void FoldOcclusion();
    /// Occlusion queries live inside render passes (CounterEngine::InPassQueries; false while no
    /// counter op created the engine).
    [[nodiscard]] bool InPassCounters() const noexcept {
        return impl.counters_engine && impl.counters_engine->InPassQueries();
    }
    /// In-pass queries: writes the reports of counter_batch (outside a render pass).
    void ResolveCounterBatch();
    /// In-pass queries: room in counter_batch for a report or reset (its query and itself, and the
    /// query the pass end adds): a full batch is written now, ending the open pass when there is
    /// one (`cause`). Without it a run of resets outside a pass would overflow the batch.
    void ReserveCounterEvents(BreakCause cause);
    /// ReportCounter / ResetCounter(SAMPLES_PASSED) leave the open render pass open (creates the
    /// counter engine).
    [[nodiscard]] bool CounterOpKeepsPass(VC::Op op, const void* payload, size_t payload_size);
    /// Submits the segments recorded so far and waits for them (CPU render conditions).
    void FlushAndWait();
    /// Submits segments[submitted_segments..]; `binary_signals` go on the last one.
    void SubmitSegments(std::span<const VkSemaphore> binary_signals);
    bool RecordWindowFilter(Texture& texture, VkFormat source_format, const Frame& frame,
                            const PixelRect& crop, const PixelRect& picture);
    void WarnUnsupportedPresentPasses(bool applied_filter);
    /// Acquires `count` more frames, records the generated frames into them and lists them in
    /// presentation order.
    void GenerateFrames(size_t count, const Frame& frame, std::vector<Frame*>& generated);
    /// Profiler tag of an op on texture `serial` (formats and size) / of a storage; only called
    /// with the profiler on.
    [[nodiscard]] GpuProf::Tag ProfTexture(GpuProf::Kind kind, u64 serial, u32 count = 0) const;
    [[nodiscard]] static GpuProf::Tag ProfStorage(GpuProf::Kind kind, const Storage& s,
                                                  u64 sub = 0);
    /// Ends the current command buffer, submits everything recorded so far and starts a new one,
    /// when enough recording time has passed (EDEN_NX2_NVN_EARLY_SUBMIT_US): the GPU runs the first
    /// part of a big submission while the rest is still recorded. Called between ops only.
    void MaybeSubmitEarly();
    /// Ends the current command buffer, submits everything recorded so far and starts a new one.
    void SubmitPiece(bool flush_shadows = true);

    Context::Impl& impl;
    u32 queue_index;
    InFlight& flight;
    QueueState& queue;
    DrawContext& draw;

    std::vector<Segment> segments;
    std::vector<std::pair<VkSemaphore, u64>> pending_waits;
    /// Render-condition predicates computed ahead of their SetRenderCondition (PrefetchConditions;
    /// what invalidates them: nvn_condition_prefetch.h).
    ConditionPrefetch condition_prefetch;
    /// PrefetchConditions scratch: the batch, and the 8-byte granules the reports ahead write.
    std::vector<ConditionPrefetch::Entry> prefetch_batch;
    std::vector<CounterEngine::ConditionInput> prefetch_inputs;
    std::unordered_set<u64> prefetch_written;
    bool barrier_pending{false};
    /// Game dispatches wrote since the last memory barrier (HostMarkDispatchWritten): every flush
    /// but the one before a dispatch (HostFlushBarrierBeforeDispatch) emits the barrier for them.
    bool dispatch_writes_pending{false};
    /// EDEN_NX2_TRY_BARRIER_MERGE: the buffer ranges of the replay's own commands since the last
    /// barrier (HostBeginBufferOp); empty with the switch off. Any flush emits the barrier for
    /// them.
    BarrierTracker buffer_ops;
    /// EDEN_NX2_TRY_BARRIER_MERGE (adjacent requests): the barrier the draw path asked for that
    /// waits for the next command recorded through Cmd(): a memory barrier (all stages, as
    /// MemoryBarrier) and the layout transitions requested since, one entry per image. Another
    /// request before any command joins it; EndSegment records what is left.
    struct DeferredBarrier {
        bool pending{false};
        std::vector<VkImageMemoryBarrier> images;
    } deferred_barrier;
    /// Image barriers of the PrepareImages call in progress (kept to avoid a heap allocation per
    /// call).
    std::vector<VkImageMemoryBarrier> image_barriers;
    bool host_barrier_needed{false};
    bool segment_signaled{false};

    enum class RenderState { Enabled, Disabled, Conditional };
    RenderState render_state{RenderState::Enabled};
    u32 condition_slot{};              ///< Scratch slot of the active GPU condition.
    bool condition_open{false};        ///< Between BeginRenderCondition and EndRenderCondition.
    std::optional<QueryRef> occlusion; ///< The open SAMPLES_PASSED query of the current segment.
    /// In-pass queries: the counter events of the open pass (and of earlier passes without a
    /// report), written by ResolveCounterBatch.
    CounterBatch counter_batch;
    size_t submitted_segments{0};
    std::chrono::steady_clock::time_point piece_start{std::chrono::steady_clock::now()};
    u32 ops_since_check{0};
    /// Records replayed (ops, including the ones that record nothing) and the count at the last
    /// submitted piece: Suspend has something to submit when they differ.
    u64 records_replayed{0};
    u64 records_at_piece{0};
    /// The last RecordList call stopped at a wait (the replay resumes at that record).
    bool resumed_wait{false};
    /// The signals a satisfied wait depends on (kept to avoid a heap allocation per wait).
    std::vector<VC::SignalPoint> event_deps;
    /// EDEN_NX2_NVN_TIMELINE: the current blocked wait was logged (`wait_blocked` once per wait).
    bool wait_block_noted{false};
    u32 early_pieces{0};
    size_t flushed_arena{0};   ///< flight.arena chunks before this index were flushed already
    size_t flushed_staging{0}; ///< the same for flight.staging
    std::shared_ptr<ShadowChunk> alias_scratch;
    std::vector<std::pair<u64, u64>> mirror_pending_writes;
    std::vector<std::pair<VkBuffer, u64>> deferred_zero; ///< (buffer, offset) of 8-byte zero-fills
    const VC::Submission* lookahead_submission{};
    size_t lookahead_list{0};
    std::span<const u64> current_list;
    size_t next_record{0}; ///< word index of the record after the one being replayed
    bool finished{false};
    /// PrepareImages records GPU writes (Storage::origin); uploads and overlap transfers are not
    /// writes of the texture.
    bool track_writes{true};

    // Staging arena.
    u64 chunk_used{};
    u64 chunk_size{};
    u8* chunk_pointer{};

    // Mirror write-back.
    u64 download_size{};
    std::vector<InFlight::Readback> readbacks;
    std::unordered_set<Pool*> dirty_pool_set;
    std::vector<std::shared_ptr<Storage>> linear_writebacks;
    std::unordered_set<Storage*> linear_writeback_set;
    std::vector<std::shared_ptr<Storage>> touched;
    std::unordered_set<Storage*> touched_set;
};

void Recorder::BeginSegment(bool keep_shadows) {
    Segment segment;
    segment.raw = impl.AcquireCommandBuffer();
    flight.command_buffers.push_back(segment.raw);
    segment.cmd = vk::CommandBuffer{segment.raw, impl.dld};
    segment.cmd.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    for (const auto& [semaphore, value] : pending_waits) {
        segment.wait_semaphores.push_back(semaphore);
        segment.wait_values.push_back(value);
    }
    pending_waits.clear();
    segment.signal_value = queue.next_value++;
    if (flight.gpu_prof) {
        segment.prof_begin = flight.gpu_prof->StampSegment(segment.cmd);
    }
    segments.push_back(std::move(segment));
    // The first op of a segment must be ordered after whatever earlier submissions on this
    // queue left running (same VkQueue, but no implicit dependency between submissions).
    barrier_pending = true;
    buffer_ops.Clear();
    condition_prefetch.Clear();
    segment_signaled = false;
    draw.OnSegmentBegin(keep_shadows);
}

void Recorder::EndSegment(bool flush_shadows) {
    EndRenderCondition();
    EndDrawPass(BreakCause::SegmentEnd);
    // Storage shadows go back to their pools before the host barrier below.
    if (flush_shadows || segment_signaled) {
        draw.FlushShadows();
    }
    FlushLinearWritebacks();
    // Queries cannot span command buffers: fold the samples counted so far into the queue's
    // accumulator (the next draw of the next segment starts a new query); in-pass queries: the
    // events the passes left (queries and resets without a report) go into it as well.
    FoldOcclusion();
    ResolveCounterBatch();
    // Predicates computed ahead live in this command buffer's slots and ordering only.
    condition_prefetch.Clear();
    if (deferred_barrier.pending) {
        EmitDeferredBarrier();
    }
    if (flight.gpu_prof) {
        flight.gpu_prof->EndOps(segments.back().cmd);
    }
    Segment& segment = segments.back();
    if (host_barrier_needed) {
        const VkMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = ALL_WRITES,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT,
        };
        segment.cmd.PipelineBarrier(ALL_STAGES, VK_PIPELINE_STAGE_HOST_BIT, 0, barrier);
        host_barrier_needed = false;
    }
    if (flight.gpu_prof) {
        // After the host barrier, so the segment covers everything the GPU runs for it.
        const u32 end = flight.gpu_prof->StampSegment(segment.cmd);
        flight.gpu_prof->Add(segment.prof_begin, end, GpuProf::Tag{.kind = GpuProf::Kind::Segment});
    }
    segment.cmd.End();
    segment.ended = true;
}

void Recorder::Abort() {
    finished = true;
    deferred_barrier = {};
    try {
        EndRenderCondition();
        draw.EndPass();
    } catch (...) {
    }
    draw.EndSubmission();
    if (!segments.empty() && !segments.back().ended) {
        // A query or conditional scope still open would make the command buffer invalid.
        if (condition_open) {
            impl.counters_engine->EndConditional(Cmd());
            condition_open = false;
        }
        if (occlusion) {
            impl.counters_engine->EndOcclusion(Cmd(), Counters(), *occlusion);
            occlusion.reset();
        }
        counter_batch.Clear();
    }
    for (Segment& segment : segments) {
        if (!segment.ended) {
            void(impl.dld.vkEndCommandBuffer(segment.raw));
            segment.ended = true;
        }
    }
    std::erase_if(impl.submit_batch.entries,
                  [&](const Context::Impl::SubmitBatch::Entry& e) { return e.flight_seq == flight.seq; });
    std::erase_if(impl.submit_batch.marks,
                  [&](const Context::Impl::SubmitBatch::Mark& m) { return m.flight == &flight; });
    if (flight.gpu_submitted_value != 0) {
        // Early pieces of this submission are on the GPU and use its arena, staging and
        // resources, which a failed flight releases at once: let them finish first.
        const VkSemaphore semaphore = *queue.timeline;
        const u64 value = flight.gpu_submitted_value;
        const VkSemaphoreWaitInfo wait_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .pNext = nullptr,
            .flags = 0,
            .semaphoreCount = 1,
            .pSemaphores = &semaphore,
            .pValues = &value,
        };
        void(impl.dld.vkWaitSemaphores(impl.logical, &wait_info, 5'000'000'000ull));
    }
    // The commands never ran, so the layouts they would have left are not real.
    for (const auto& storage : touched) {
        storage->current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    }
    impl.ErasePendingPoolWritesForSeq(flight.seq);
    // Pool mirror pages this submission was to copy in (and GPU writes it was to make there and
    // merge back) never happened: no mirror page is known to be current.
    impl.MirrorInvalidateAll();
    // Mirror ranges marked as GPU-written have nothing to write back.
    for (Pool* pool : dirty_pool_set) {
        std::scoped_lock lock{pool->dirty_mutex};
        std::erase_if(pool->gpu_dirty, [&](const Pool::Dirty& d) { return d.seq == flight.seq; });
    }
    for (const std::shared_ptr<Storage>& storage : linear_writebacks) {
        if (storage->linear_writeback_seq == flight.seq) {
            storage->linear_writeback_seq = 0;
        }
    }
    linear_writebacks.clear();
    linear_writeback_set.clear();
    readbacks.clear();
}

void Recorder::MemoryBarrier(bool defer) {
    // One barrier for every ordering point since the last one (game ORDER_PRIMITIVES /
    // ORDER_INDIRECT_DATA barriers: a wait for idle on the console, nvn_barrier.h; replay copies,
    // pass ends, uploads), and every later command relies on it: all stages on both sides.
    // Only writes need to be made available; a read in the source access mask does nothing
    // (write-after-read needs the execution dependency alone).
    const bool adjacent = (impl.barrier_merge & BARRIER_MERGE_ADJACENT) != 0;
    if (adjacent && deferred_barrier.pending) {
        // A request is waiting for its next command and none came: it already is this barrier.
        impl.stats.barrier_merge_folded.fetch_add(1, std::memory_order_relaxed);
    } else if (adjacent && defer) {
        deferred_barrier.pending = true;
    } else {
        const VkMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = ALL_WRITES,
            .dstAccessMask = ALL_ACCESS,
        };
        Cmd().PipelineBarrier(ALL_STAGES, ALL_STAGES, 0, barrier);
    }
    barrier_pending = false;
    impl.stats.memory_barriers.fetch_add(1, std::memory_order_relaxed);
    dispatch_writes_pending = false;
    buffer_ops.Clear();
}

void Recorder::EmitDeferredBarrier() {
    const VkMemoryBarrier memory_barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = ALL_WRITES,
        .dstAccessMask = ALL_ACCESS,
    };
    // Straight into the segment's command buffer: Cmd() would ask for this barrier again.
    deferred_barrier.pending = false;
    if (deferred_barrier.images.empty()) {
        segments.back().cmd.PipelineBarrier(ALL_STAGES, ALL_STAGES, 0, memory_barrier);
    } else {
        segments.back().cmd.PipelineBarrier(ALL_STAGES, ALL_STAGES, 0, memory_barrier, {},
                                            deferred_barrier.images);
        deferred_barrier.images.clear();
    }
    impl.stats.barrier_merge_emitted.fetch_add(1, std::memory_order_relaxed);
}

void Recorder::NoteTouched(const std::shared_ptr<Storage>& storage) {
    // One compare per use: the set is per submission, and so is the sequence number.
    if (storage->touched_seq == flight.seq) {
        return;
    }
    storage->touched_seq = flight.seq;
    if (touched_set.insert(storage.get()).second) {
        touched.push_back(storage);
    }
}

void Recorder::PrepareImagesSpan(std::span<const std::pair<Storage*, VkImageLayout>> images,
                                 std::span<const VkImageMemoryBarrier> extra, bool flush_if_none,
                                 bool defer, bool note_writes) {
    std::vector<VkImageMemoryBarrier>& barriers = image_barriers;
    barriers.assign(extra.begin(), extra.end());
    for (const auto& [storage, layout] : images) {
        if (note_writes && track_writes && (layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ||
                             layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL ||
                             layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL ||
                             layout == VK_IMAGE_LAYOUT_GENERAL)) {
            NoteGpuWrite(*storage);
        }
        if (storage->current_layout == layout) {
            continue;
        }
        barriers.push_back(VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = ALL_WRITES, // see MemoryBarrier
            .dstAccessMask = ALL_ACCESS,
            .oldLayout = storage->current_layout,
            .newLayout = layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *storage->image,
            .subresourceRange{
                .aspectMask = storage->aspect,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        });
        storage->current_layout = layout;
        impl.stats.image_transitions.fetch_add(1, std::memory_order_relaxed);
    }
    if (barriers.empty()) {
        if (flush_if_none) {
            FlushPending(defer);
        }
        return;
    }
    const bool adjacent = (impl.barrier_merge & BARRIER_MERGE_ADJACENT) != 0;
    if (adjacent && extra.empty() && (defer || deferred_barrier.pending)) {
        // The transitions wait for the next command (or join the request that does): one barrier
        // for every request until then. An image asked for again goes from where the first request
        // moved it from to where this one moves it to (Storage::current_layout was kept up to
        // date); the global memory barrier of both is the same one.
        for (const VkImageMemoryBarrier& barrier : barriers) {
            const auto same = std::ranges::find(deferred_barrier.images, barrier.image,
                                                &VkImageMemoryBarrier::image);
            if (same != deferred_barrier.images.end()) {
                same->newLayout = barrier.newLayout;
            } else {
                deferred_barrier.images.push_back(barrier);
            }
        }
        if (deferred_barrier.pending) {
            impl.stats.barrier_merge_folded.fetch_add(1, std::memory_order_relaxed);
        }
        deferred_barrier.pending = true;
    } else {
        const VkMemoryBarrier memory_barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = ALL_WRITES,
            .dstAccessMask = ALL_ACCESS,
        };
        Cmd().PipelineBarrier(ALL_STAGES, ALL_STAGES, 0, memory_barrier, {}, barriers);
    }
    barrier_pending = false;
    impl.stats.memory_barriers.fetch_add(1, std::memory_order_relaxed);
    dispatch_writes_pending = false;
    buffer_ops.Clear();
}

StagingRange Recorder::AllocStaging(u64 size, u64 alignment) {
    chunk_used = (chunk_used + alignment - 1) / alignment * alignment;
    if (flight.staging.empty() || chunk_used + size > chunk_size) {
        const u64 new_size = (std::max)(size, Context::Impl::STAGING_CHUNK);
        flight.staging.push_back(impl.AcquireStaging(new_size));
        chunk_size = new_size;
        chunk_pointer = flight.staging.back().Mapped().data();
        chunk_used = 0;
    }
    const StagingRange range{*flight.staging.back(), chunk_used, chunk_pointer + chunk_used};
    chunk_used += size;
    return range;
}

u64 Recorder::GuestPageHash(const Pool& pool, u64 page) const {
    constexpr u64 UNREADABLE = 0x9E3779B97F4A7C15ull;
    const u64 begin = page * Pool::MIRROR_PAGE;
    const u64 length = (std::min)(Pool::MIRROR_PAGE, pool.size - begin);
    const u8* bytes =
        pool.host ? pool.host + begin : impl.GuestPointer(pool.address + begin, length);
    std::array<u8, Pool::MIRROR_PAGE> local;
    if (bytes == nullptr) {
        // A page straddling two host runs (pool addresses are not always page aligned).
        VC::GuestMemory* memory = impl.guest.load(std::memory_order_acquire);
        if (!memory || !memory->Read(pool.address + begin, local.data(), length)) {
            return UNREADABLE;
        }
        bytes = local.data();
    }
    return Common::CityHash64(reinterpret_cast<const char*>(bytes), static_cast<size_t>(length));
}

bool Recorder::MirrorTakeCpuWrites(Pool& pool) {
    return impl.TakePoolCpuWrites(pool);
}

MirrorWriteCoverage Recorder::MirrorPageWrites(Pool& pool, u64 page, bool& pending_read) {
    if (!pending_read) {
        // Taken before the page is hashed: the completion writes a range back before it drops it
        // from gpu_dirty, so a range missing here is in the guest bytes the hash reads.
        mirror_pending_writes.clear();
        std::scoped_lock lock{pool.dirty_mutex};
        for (const Pool::Dirty& dirty : pool.gpu_dirty) {
            mirror_pending_writes.emplace_back(dirty.begin, dirty.end);
        }
        pending_read = true;
    }
    const u64 begin = page * Pool::MIRROR_PAGE;
    const u64 end = (std::min)(begin + Pool::MIRROR_PAGE, pool.size);
    return ClassifyMirrorWrites(begin, end, mirror_pending_writes);
}

bool Recorder::MirrorNeedsUpload(Pool& pool, u64 offset, u64 size) {
    if (pool.backing != Pool::Backing::Mirror || size == 0 || offset >= pool.size) {
        return false;
    }
    if (!pool.buffer || pool.mirror_pages.empty()) {
        impl.EnsureBacking(pool);
        if (!pool.buffer || pool.mirror_pages.empty()) {
            return false;
        }
    }
    const bool tracked = MirrorTakeCpuWrites(pool);
    bool pending_read = false;
    const u32 stamp = static_cast<u32>(flight.seq) + 1;
    const u64 first = offset / Pool::MIRROR_PAGE;
    const u64 last = (std::min)(offset + size, pool.size) - 1;
    u64 hashed = 0;
    u64 trusted = 0;
    u64 gpu_owned = 0;
    bool dirty = false;
    for (u64 index = first; index <= last / Pool::MIRROR_PAGE; ++index) {
        Pool::MirrorPage& page = pool.mirror_pages[static_cast<size_t>(index)];
        const bool current_epoch =
            !impl.mario_kart_world || pool.split_runs.empty() ||
            (pool.raw_written_through ? page.raw_epoch >= pool.split_raw_epoch_unaliased_seen
                                      : page.raw_epoch == pool.split_raw_epoch_seen);
        if (page.valid && page.checked == stamp && current_epoch) {
            continue;
        }
        if (page.valid && page.trusted && current_epoch) {
            ++trusted;
            continue;
        }
        const auto coverage = MirrorPageWrites(pool, index, pending_read);
        if (coverage == MirrorWriteCoverage::Full) {
            ++gpu_owned;
            continue;
        }
        const bool settled = tracked && coverage == MirrorWriteCoverage::None;
        ++hashed;
        if (page.valid && page.hash == GuestPageHash(pool, index)) {
            page.checked = stamp;
            page.raw_epoch = pool.split_raw_epoch_seen;
            page.trusted = settled;
            continue;
        }
        dirty = true;
        break;
    }
    if (hashed != 0) {
        impl.stats.mirror_pages_hashed.fetch_add(hashed, std::memory_order_relaxed);
    }
    if (trusted != 0) {
        impl.stats.mirror_pages_trusted.fetch_add(trusted, std::memory_order_relaxed);
    }
    if (gpu_owned != 0) {
        impl.stats.mirror_pages_gpu_owned.fetch_add(gpu_owned, std::memory_order_relaxed);
    }
    return dirty;
}

void Recorder::UploadRange(Pool& pool, u64 offset, u64 size) {
    if (pool.backing != Pool::Backing::Mirror || size == 0 || offset >= pool.size) {
        return;
    }
    if (!pool.buffer || pool.mirror_pages.empty()) {
        impl.EnsureBacking(pool);
        if (!pool.buffer || pool.mirror_pages.empty()) {
            return;
        }
    }
    // The guest pages of the range that differ from what the mirror holds (all of them the first
    // time), as runs of consecutive pages. The mirror is made equal to these pages from here on,
    // so the bookkeeping is updated now.
    const bool tracked = MirrorTakeCpuWrites(pool);
    bool pending_read = false;
    const u32 stamp = static_cast<u32>(flight.seq) + 1;
    const u64 first = offset / Pool::MIRROR_PAGE;
    const u64 last = (std::min)(offset + size, pool.size) - 1;
    u64 hashed = 0;
    u64 trusted = 0;
    u64 gpu_owned = 0;
    std::vector<std::pair<u64, u64>> parts;
    for (u64 index = first; index <= last / Pool::MIRROR_PAGE; ++index) {
        Pool::MirrorPage& page = pool.mirror_pages[static_cast<size_t>(index)];
        const bool current_epoch =
            !impl.mario_kart_world || pool.split_runs.empty() ||
            (pool.raw_written_through ? page.raw_epoch >= pool.split_raw_epoch_unaliased_seen
                                      : page.raw_epoch == pool.split_raw_epoch_seen);
        if (page.valid && page.checked == stamp && current_epoch) {
            continue;
        }
        if (page.valid && page.trusted && current_epoch) {
            ++trusted;
            continue;
        }
        const auto coverage = MirrorPageWrites(pool, index, pending_read);
        if (coverage == MirrorWriteCoverage::Full) {
            ++gpu_owned;
            continue; // UploadRange would discard this whole page while the GPU owns it.
        }
        const bool settled = tracked && coverage == MirrorWriteCoverage::None;
        const bool partial_gpu = coverage == MirrorWriteCoverage::Partial;
        ++hashed;
        const u64 hash = GuestPageHash(pool, index);
        const bool changed = !page.valid || page.hash != hash;
        if (!partial_gpu) {
            page.hash = hash;
            page.valid = true;
            page.checked = stamp;
            page.raw_epoch = pool.split_raw_epoch_seen;
            page.trusted = settled;
        } else {
            page.valid = false;
            page.checked = 0;
            page.trusted = false;
        }
        if (!changed) {
            continue;
        }
        const u64 begin = index * Pool::MIRROR_PAGE;
        const u64 end = (std::min)(begin + Pool::MIRROR_PAGE, pool.size);
        if (!parts.empty() && parts.back().second == begin) {
            parts.back().second = end;
        } else {
            parts.emplace_back(begin, end);
        }
    }
    if (hashed != 0) {
        impl.stats.mirror_pages_hashed.fetch_add(hashed, std::memory_order_relaxed);
    }
    if (trusted != 0) {
        impl.stats.mirror_pages_trusted.fetch_add(trusted, std::memory_order_relaxed);
    }
    if (gpu_owned != 0) {
        impl.stats.mirror_pages_gpu_owned.fetch_add(gpu_owned, std::memory_order_relaxed);
    }
    if (parts.empty()) {
        impl.stats.mirror_ranges_clean.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // Keep the same ownership snapshot for the hashes and copies. Completion can remove a range
    // after the snapshot, but those bytes already have the GPU's value in the mirror.
    for (const auto& [dirty_begin, dirty_end] : mirror_pending_writes) {
        std::vector<std::pair<u64, u64>> next;
        for (const auto& [begin, end] : parts) {
            if (dirty_end <= begin || dirty_begin >= end) {
                next.emplace_back(begin, end);
                continue;
            }
            if (dirty_begin > begin) {
                next.emplace_back(begin, dirty_begin);
            }
            if (dirty_end < end) {
                next.emplace_back(dirty_end, end);
            }
        }
        parts = std::move(next);
        if (parts.empty()) {
            return;
        }
    }
    u64 copied = 0;
    for (const auto& [begin, end] : parts) {
        copied += end - begin;
    }
    // Earlier commands may still read or write these bytes.
    GpuProf::Scope prof{flight.gpu_prof.get(), Cmd(),
                        flight.gpu_prof ? GpuProf::Tag{.kind = GpuProf::Kind::Upload,
                                                       .count = static_cast<u32>(copied),
                                                       .hash_a = GpuProf::SUB_MIRROR}
                                        : GpuProf::Tag{}};
    MemoryBarrier();
    VC::GuestMemory* memory = impl.guest.load(std::memory_order_acquire);
    for (const auto& [begin, end] : parts) {
        const u64 length = end - begin;
        const StagingRange staging = AllocStaging(length, 64);
        if (pool.host) {
            std::memcpy(staging.pointer, pool.host + begin, length);
        } else if (!memory || !memory->Read(pool.address + begin, staging.pointer, length)) {
            std::memset(staging.pointer, 0, length);
        }
        const VkBufferCopy region{.srcOffset = staging.offset, .dstOffset = begin, .size = length};
        Cmd().CopyBuffer(staging.buffer, pool.Handle(), region);
    }
    impl.stats.mirror_uploads.fetch_add(1, std::memory_order_relaxed);
    impl.stats.mirror_upload_runs.fetch_add(parts.size(), std::memory_order_relaxed);
    impl.stats.mirror_upload_bytes.fetch_add(copied, std::memory_order_relaxed);
    barrier_pending = true;
}

void Recorder::NotePoolWrite(Pool& pool, u64 offset, u64 size, bool invalidate_mirror,
                             u64 row_bytes, u64 row_stride, u32 row_count) {
    impl.NoteNonInlinePoolWrite(pool.address + offset, size, flight.seq);
    if (impl.invalidate_wait_ms != 0) {
        // Where a guest invalidate has to wait for this submission (nvn_gpu_writes.h).
        impl.gpu_writes.Note(pool.address + offset, pool.address + offset + size, flight.seq,
                             impl.completed_seq.load(std::memory_order_acquire));
    }
    draw.OnPoolWrite(pool, offset, size);
    condition_prefetch.Invalidate(pool, offset, size);
    if (invalidate_mirror) {
        // (A mirror entry's write-back repeats stores noted at their draw, or raw stores.)
        impl.uniform_snapshots.NoteWrite(pool.address + offset, pool.address + offset + size,
                                         flight.seq);
    }
    switch (pool.backing) {
    case Pool::Backing::Imported:
        host_barrier_needed = true;
        if (invalidate_mirror) {
            impl.MirrorInvalidate(pool, offset, size);
        }
        break;
    case Pool::Backing::Mirror: {
        {
            std::scoped_lock lock{pool.dirty_mutex};
            pool.gpu_dirty.push_back({offset, offset + size, flight.seq});
        }
        // The write-back (and a report's timestamp) reaches the guest bytes untracked: the pages
        // are hashed again once it is done (MirrorPageWrites).
        if (size != 0 && offset < pool.size && !pool.mirror_pages.empty()) {
            const u64 end = (std::min)(offset + size, pool.size);
            const size_t limit = pool.mirror_pages.size();
            for (u64 page = offset / Pool::MIRROR_PAGE; page * Pool::MIRROR_PAGE < end && page < limit;
                 ++page) {
                Pool::MirrorPage& mirror_page = pool.mirror_pages[static_cast<size_t>(page)];
                mirror_page.trusted = false;
                mirror_page.valid = false;
                mirror_page.checked = 0;
            }
        }
        dirty_pool_set.insert(&pool);
        for (const InFlight::Readback& existing : readbacks) {
            if (existing.pool.get() == &pool && existing.pool_offset == offset &&
                existing.size == size && existing.row_bytes == row_bytes &&
                existing.row_stride == row_stride && existing.row_count == row_count) {
                return;
            }
        }
        readbacks.push_back({impl.pools.at(pool.serial), offset, size, download_size, row_bytes,
                             row_stride, row_count});
        download_size += (size + 15) / 16 * 16;
        break;
    }
    default:
        break;
    }
}

void Recorder::WriteReadbacks() {
    if (readbacks.empty()) {
        return;
    }
    flight.download = impl.CreateHostBuffer(download_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                            MemoryUsage::Download);
    MemoryBarrier();
    for (const InFlight::Readback& readback : readbacks) {
        const VkBufferCopy region{
            .srcOffset = readback.pool_offset,
            .dstOffset = readback.download_offset,
            .size = readback.size,
        };
        Cmd().CopyBuffer(readback.pool->Handle(), *flight.download, region);
    }
    host_barrier_needed = true;
    flight.readbacks = readbacks;
    for (Pool* pool : dirty_pool_set) {
        flight.dirty_pools.push_back(impl.pools.at(pool->serial));
    }
}

void Recorder::LogOnce(const std::string& message) {
    if (impl.logged_messages.insert(message).second) {
        LOG_WARNING(Render_Vulkan, "[nvn] {}", message);
    }
}

void Recorder::Unsupported(std::string_view op, const std::string& reason) {
    impl.stats.ops_unsupported.fetch_add(1, std::memory_order_relaxed);
    LogOnce(fmt::format("unsupported {}: {}", op, reason));
}

void Recorder::Failed(std::string_view op, const std::string& reason) {
    impl.stats.ops_failed.fetch_add(1, std::memory_order_relaxed);
    LogOnce(fmt::format("{} dropped: {}", op, reason));
}

Texture* Recorder::FindTexture(u64 serial) const {
    const auto it = impl.textures.find(serial);
    return it == impl.textures.end() ? nullptr : it->second.get();
}

bool Recorder::ResolveRef(const VC::TextureRef& ref, Storage*& storage,
                          std::shared_ptr<Storage>& owner, u32& level, u32& layer,
                          std::string_view op) {
    Texture* texture = FindTexture(ref.serial);
    if (!texture) {
        Failed(op, fmt::format("unknown texture {:#x}", ref.serial));
        return false;
    }
    if (!texture->storage) {
        Failed(op, fmt::format("texture {:#x} has no image", ref.serial));
        return false;
    }
    if (!impl.EnsureImage(*texture->storage)) {
        Failed(op, fmt::format("texture {:#x}: the image could not be created", ref.serial));
        return false;
    }
    owner = texture->storage;
    storage = owner.get();
    level = ref.level;
    layer = ref.layer;
    // TextureRef level/layer already include the view's base level and layer.
    if (level >= storage->levels || layer >= storage->layers) {
        Failed(op, fmt::format("texture {:#x} level {} layer {} out of range", ref.serial, level,
                               layer));
        return false;
    }
    impl.NoteUse(*texture, flight.seq);
    NoteTouched(owner);
    return true;
}

bool Recorder::ResolvePoolRange(u64 address, u64 size, Pool*& pool, u64& offset,
                                std::string_view op, bool flush_shadows) {
    if (!impl.ResolveBuffer(address, size, pool, offset)) {
        const std::string reason = impl.DescribeUnresolved(address, size);
        if (reason.find("virtual") != std::string::npos) {
            impl.stats.virtual_unresolved.fetch_add(1, std::memory_order_relaxed);
        }
        Failed(op, reason);
        return false;
    }
    impl.EnsureBacking(*pool);
    if (pool->backing == Pool::Backing::None) {
        Unsupported(op, fmt::format("pool {:#x} has no backing", pool->address));
        return false;
    }
    impl.NoteUse(*pool, flight.seq);
    if (!linear_writebacks.empty() && !draw.PassOpen()) {
        const u64 clamped = (std::min)(pool->size - offset, (std::max<u64>)(size, 1));
        FlushLinearWritebacks(pool, offset, offset + clamped);
    }
    if (flush_shadows && draw.HasShadows()) {
        // An op outside the draw path touches the bytes through host memory.
        draw.FlushShadowsOverlapping(*pool, offset, (std::max<u64>)(size, 1));
    }
    return true;
}

bool Recorder::MakeCopyGeometry(const Storage& s, u32 level, u32 layer,
                                const VC::CopyRegion& region, u32 row_stride, u32 image_stride,
                                u64 buffer_offset, CopyGeometry& out) const {
    const BlockInfo block = FormatBlockInfo(s.desc.format);
    if (!block.bytes || s.samples != VK_SAMPLE_COUNT_1_BIT) {
        return false;
    }
    const LevelExtent extent = ExtentOfLevel(s, level);
    const u32 x = static_cast<u32>((std::max)(region.x, 0));
    const u32 y = static_cast<u32>((std::max)(region.y, 0));
    if (x >= extent.width || y >= extent.height) {
        return false;
    }
    const u32 width = region.width > 0
                          ? (std::min)(static_cast<u32>(region.width), extent.width - x)
                          : extent.width - x;
    const u32 height = region.height > 0
                           ? (std::min)(static_cast<u32>(region.height), extent.height - y)
                           : extent.height - y;
    const bool is_3d = s.type == VK_IMAGE_TYPE_3D;
    u32 z = static_cast<u32>((std::max)(region.z, 0));
    u32 depth_count = region.depth > 0 ? static_cast<u32>(region.depth) : 1u;
    u32 first_layer = layer;
    u32 layer_count = 1;
    u32 slices = 1;
    if (is_3d) {
        if (z >= extent.depth) {
            return false;
        }
        depth_count = (std::min)(depth_count, extent.depth - z);
        slices = depth_count;
    } else {
        first_layer = layer + z;
        if (first_layer >= s.layers) {
            return false;
        }
        layer_count = (std::min)(depth_count, s.layers - first_layer);
        slices = layer_count;
        z = 0;
    }
    if (x % block.width || y % block.height) {
        return false;
    }
    const u32 cols = CeilDiv(width, block.width);
    const u32 rows = CeilDiv(height, block.height);
    const u32 row_bytes = cols * block.bytes;
    const u32 rs = row_stride ? row_stride : row_bytes;
    if (rs < row_bytes) {
        return false;
    }
    const u32 is = image_stride ? image_stride : rs * rows;
    if (is < rs * rows && slices > 1) {
        return false;
    }
    out.span = static_cast<u64>(slices - 1) * is + static_cast<u64>(rows - 1) * rs + row_bytes;
    out.padded = rs != row_bytes || (slices > 1 && is != rs * rows);
    const VkImageSubresourceLayers subresource{
        .aspectMask = TransferAspect(s),
        .mipLevel = level,
        .baseArrayLayer = first_layer,
        .layerCount = layer_count,
    };
    out.regions.clear();
    if (buffer_offset % 4 == 0 && buffer_offset % block.bytes == 0 && rs % block.bytes == 0 &&
        is % rs == 0) {
        out.regions.push_back({
            .bufferOffset = buffer_offset,
            .bufferRowLength = rs / block.bytes * block.width,
            .bufferImageHeight = is / rs * block.height,
            .imageSubresource = subresource,
            .imageOffset = {static_cast<s32>(x), static_cast<s32>(y), static_cast<s32>(z)},
            .imageExtent = {width, height, is_3d ? depth_count : 1u},
        });
        return true;
    }
    // Strides the image copy cannot express: one region per row.
    if (buffer_offset % 4 != 0) {
        return false;
    }
    for (u32 slice = 0; slice < slices; ++slice) {
        for (u32 row = 0; row < rows; ++row) {
            const u64 offset =
                buffer_offset + static_cast<u64>(slice) * is + static_cast<u64>(row) * rs;
            if (offset % 4 != 0 || offset % block.bytes != 0) {
                return false;
            }
            VkImageSubresourceLayers row_layers = subresource;
            row_layers.layerCount = 1;
            if (!is_3d) {
                row_layers.baseArrayLayer = first_layer + slice;
            }
            out.regions.push_back({
                .bufferOffset = offset,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = row_layers,
                .imageOffset = {static_cast<s32>(x), static_cast<s32>(y + row * block.height),
                                static_cast<s32>(is_3d ? z + slice : 0)},
                .imageExtent = {width, (std::min)(block.height, height - row * block.height), 1u},
            });
        }
    }
    return true;
}

bool Recorder::EnsureContents(Storage& s, bool will_overwrite, bool keep_scaled) {
    if (!impl.EnsureImage(s)) {
        return false;
    }
    s.last_used_seq = flight.seq;
    s.last_used_present = impl.stats.presents.load(std::memory_order_relaxed);
    if (s.scale_owner != nullptr) {
        return true; // a twin: the guest storage that owns it decides what is current
    }
    if (s.scaled && !will_overwrite && Rescale::NativeIsStale(s.scale_state) &&
        (!keep_scaled || impl.StorageNeedsCommands(s, false, flight.seq, true))) {
        // The guest-sized image is about to be read, or changed by guest bytes: the twin's pixels
        // are the newest ones, bring them over first.
        MakeNativeCurrent(s);
    }
    if (s.sparse) {
        SyncSparse(s, will_overwrite);
        return true;
    }
    if (!s.contents_valid) {
        s.contents_valid = true;
        if (!will_overwrite) {
            UploadStorage(s);
        }
    } else if (!will_overwrite) {
        RefreshFromGuest(s);
    }
    if (will_overwrite) {
        s.force_upload = false;
        s.force_check = false;
    } else {
        ResolveOverlaps(s);
    }
    return true;
}

namespace {

/// The pool of `s` when its bytes are CPU-readable guest memory (not virtual, not CPU_NO_ACCESS)
/// and `s` starts inside it; null otherwise.
Pool* CpuReadablePoolOf(Context::Impl& impl, const Storage& s) {
    const auto it = impl.pools.find(s.pool_serial);
    if (it == impl.pools.end() || !it->second) {
        return nullptr;
    }
    Pool& pool = *it->second;
    if ((pool.flags & PoolFlag::CpuNoAccess) != 0 || pool.IsVirtual() ||
        s.pool_offset >= pool.size) {
        return nullptr;
    }
    return &pool;
}

/// True when GPU writes to `s` should be copied back to its CPU-readable pool so guest CPU
/// reads of the linear texels observe what the GPU rendered or copied into the image.
bool ShouldWriteBackLinear(Context::Impl& impl, const Storage& s) {
    if (!IsLinearStorage(s.desc) || s.sparse || s.scale_owner != nullptr ||
        s.host.needs_transcode || s.host.guest_layout_differs || s.host.depth ||
        s.host.stencil || s.aspect != VK_IMAGE_ASPECT_COLOR_BIT ||
        s.samples != VK_SAMPLE_COUNT_1_BIT || s.levels != 1 || s.layers != 1 ||
        s.type == VK_IMAGE_TYPE_3D || CpuReadablePoolOf(impl, s) == nullptr) {
        return false;
    }
    return FormatBlockInfo(s.desc.format).bytes != 0;
}

/// The pool's CPU write tracking registration for the overlap rule, created on first use; 0 when
/// the guest memory has no write tracking.
u64 TakeoverTrackOf(Context::Impl& impl, Pool& pool) {
    if (pool.takeover_track == 0 && !pool.takeover_track_tried) {
        pool.takeover_track_tried = true;
        if (VC::GuestMemory* const memory = impl.guest.load(std::memory_order_acquire)) {
            pool.takeover_track = memory->Track(pool.address, pool.size);
        }
    }
    return pool.takeover_track;
}

/// SampleGuestBytes of the guest bytes under `s`; nothing when its pool has no CPU-readable bytes
/// (virtual or CPU_NO_ACCESS pools).
std::optional<u64> GuestSignatureOf(Context::Impl& impl, const Storage& s) {
    const Pool* const found = CpuReadablePoolOf(impl, s);
    if (found == nullptr) {
        return std::nullopt;
    }
    const Pool& pool = *found;
    const u64 size = (std::min)(s.layout.size, pool.size - s.pool_offset);
    if (size == 0) {
        return std::nullopt;
    }
    std::vector<u8> scratch;
    const u8* const data = impl.GuestBytes(pool, s.pool_offset, size, scratch);
    if (data == nullptr) {
        return std::nullopt;
    }
    return SampleGuestBytes(data, size, impl.mario_kart_world);
}

/// EDEN_NX2_NVN_LEGACY_FIRST_UPLOAD=1: overlapping GPU-written storages are copied over a texture
/// uploaded from guest memory even when the CPU rewrote those bytes after the GPU write (A/B).
bool LegacyFirstUpload() {
    static const bool legacy = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_LEGACY_FIRST_UPLOAD");
        const bool on = value != nullptr && *value != 0 && value[0] != '0';
        if (on) {
            LOG_INFO(Render_Vulkan, "[nvn] EDEN_NX2_NVN_LEGACY_FIRST_UPLOAD: overlapping GPU "
                                    "contents replace CPU-rewritten uploads (legacy)");
        }
        return on;
    }();
    return legacy;
}

} // Anonymous namespace

bool Recorder::HostEnsureAttachmentContents(Storage& s) {
    const bool fresh_color = impl.mario_kart_world && !s.contents_valid && !s.scale_owner &&
                             !s.sparse && s.aspect == VK_IMAGE_ASPECT_COLOR_BIT &&
                             (s.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0;
    if (!EnsureContents(s, true)) {
        return false;
    }
    if (!fresh_color) {
        return true;
    }
    const auto block = FormatBlockInfo(s.desc.format);
    if (block.width != 1 || block.height != 1) {
        return true;
    }
    // The render pass loads the attachment. Preserve available guest contents and copy any newer
    // GPU aliases over them. A fresh GPU-only image needs defined pixels under a partial draw.
    if (CpuReadablePoolOf(impl, s) != nullptr) {
        UploadStorage(s);
    }
    if (s.origin == StorageOrigin::None) {
        const bool saved = std::exchange(track_writes, false);
        PrepareImages({{&s, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
        track_writes = saved;
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                            VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        Cmd().ClearColorImage(*s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, {}, range);
        barrier_pending = true;
    }
    ResolveOverlaps(s);
    return true;
}

void Recorder::NoteGpuWrite(Storage& s) {
    if (Storage* const owner = s.scale_owner) {
        // A pass, copy or clear wrote the scaled twin: the guest storage it stands for holds
        // GPU-written pixels (stamped as such for overlapping textures) that only the twin has.
        NoteGpuWrite(*owner);
        owner->scale_state = Rescale::AfterScaledWrite();
        return;
    }
    const bool linear_wb = ShouldWriteBackLinear(impl, s);
    if (linear_wb) {
        // Pending write-back will overwrite the guest bytes; takeover signature and page tracking
        // are refreshed once the submission completes (Context::Impl::SettleLinearWriteback).
        s.gpu_guest_sig = 0;
        s.gpu_guest_sig_valid = false;
        s.gpu_takeover_armed = false;
        s.linear_writeback_seq = flight.seq;
        std::shared_ptr<Storage> owner = s.shared_from_this();
        NoteTouched(owner);
        if (linear_writeback_set.insert(&s).second) {
            linear_writebacks.push_back(std::move(owner));
        }
    } else if (s.origin != StorageOrigin::Gpu) {
        // The guest bytes these GPU contents supersede. GPU writes never reach guest memory, so
        // the bytes only differ from this later if the CPU rewrote them (ResolveOverlaps).
        const std::optional<u64> signature = GuestSignatureOf(impl, s);
        s.gpu_guest_sig = signature.value_or(0);
        s.gpu_guest_sig_valid = signature.has_value();
        // CPU writes from here on: armed pages turn dirty on the guest's next write to them.
        s.gpu_takeover_armed = false;
        if (Pool* const pool = CpuReadablePoolOf(impl, s)) {
            VC::GuestMemory* const memory = impl.guest.load(std::memory_order_acquire);
            if (const u64 track = TakeoverTrackOf(impl, *pool); track != 0 && memory != nullptr) {
                memory->Arm(track, pool->address + s.pool_offset,
                            (std::min)(s.layout.size, pool->size - s.pool_offset));
                s.gpu_takeover_armed = true;
            }
        }
    }
    s.origin = StorageOrigin::Gpu;
    s.contents_valid = true;
    s.watch_valid = false;
    s.gpu_write_gen = s.sync_gen = ++impl.write_clock;
    if (const auto it = impl.storage_index.find(s.pool_serial); it != impl.storage_index.end()) {
        s.pool_gen_seen = ++it->second.write_gen;
    }
    if (s.scaled) {
        s.scale_state = Rescale::AfterNativeWrite();
    }
}

bool Recorder::HasLinearWritebackOverlapping(const Pool& pool, u64 begin, u64 end) const {
    for (const std::shared_ptr<Storage>& owner : linear_writebacks) {
        const Storage& s = *owner;
        const u64 s_end = s.pool_offset + s.layout.size;
        if (s.pool_serial == pool.serial && s.pool_offset < end && begin < s_end) {
            return true;
        }
    }
    return false;
}

void Recorder::FlushLinearWritebacks(const Pool* filter_pool, u64 filter_begin, u64 filter_end) {
    if (linear_writebacks.empty()) {
        return;
    }
    std::vector<std::shared_ptr<Storage>> pending;
    if (filter_pool == nullptr) {
        pending.swap(linear_writebacks);
        linear_writeback_set.clear();
    } else {
        std::vector<std::shared_ptr<Storage>> remaining;
        remaining.reserve(linear_writebacks.size());
        for (std::shared_ptr<Storage>& owner : linear_writebacks) {
            const Storage& s = *owner;
            const u64 s_begin = s.pool_offset;
            const u64 s_end = s.pool_offset + s.layout.size;
            if (s.pool_serial == filter_pool->serial && s_begin < filter_end &&
                filter_begin < s_end) {
                linear_writeback_set.erase(owner.get());
                pending.push_back(std::move(owner));
            } else {
                remaining.push_back(std::move(owner));
            }
        }
        linear_writebacks = std::move(remaining);
    }
    if (pending.size() > 1) {
        std::ranges::stable_sort(pending, {}, [](const std::shared_ptr<Storage>& entry) {
            return entry->gpu_write_gen;
        });
    }
    for (std::shared_ptr<Storage>& owner : pending) {
        Storage& s = *owner;
        if (!s.image || s.origin != StorageOrigin::Gpu) {
            s.linear_writeback_seq = 0;
            continue;
        }
        if (s.scaled && Rescale::NativeIsStale(s.scale_state)) {
            MakeNativeCurrent(s);
        }
        Pool* const pool = CpuReadablePoolOf(impl, s);
        if (!pool) {
            s.linear_writeback_seq = 0;
            continue;
        }
        impl.EnsureBacking(*pool);
        if (pool->backing != Pool::Backing::Imported && pool->backing != Pool::Backing::Mirror) {
            s.linear_writeback_seq = 0;
            continue;
        }
        const BlockInfo block = FormatBlockInfo(s.desc.format);
        const u64 available = pool->size - s.pool_offset;
        const u64 row_bytes = static_cast<u64>(CeilDiv(s.extent.width, block.width)) * block.bytes;
        const u32 stride =
            s.desc.stride ? s.desc.stride
                          : static_cast<u32>(s.layout.layer_size / (std::max)(s.extent.height, 1u));
        if (block.bytes == 0 || row_bytes == 0 || available < row_bytes || stride < row_bytes) {
            s.linear_writeback_seq = 0;
            continue;
        }
        u32 copy_rows = s.extent.height;
        if (copy_rows > 1 && static_cast<u64>(copy_rows - 1) * stride + row_bytes > available) {
            copy_rows = static_cast<u32>((available - row_bytes) / stride + 1);
        }
        CopyGeometry geometry;
        const VC::CopyRegion region{0, 0, 0, static_cast<s32>(s.extent.width),
                                    static_cast<s32>(copy_rows), 1};
        if (!MakeCopyGeometry(s, 0, 0, region, stride, 0, s.pool_offset, geometry) ||
            s.pool_offset + geometry.span > pool->size) {
            s.linear_writeback_seq = 0;
            continue;
        }
        impl.NoteUse(*pool, flight.seq);
        if (draw.HasShadows()) {
            draw.FlushShadowsOverlapping(*pool, s.pool_offset, geometry.span);
        }
        if (geometry.padded) {
            // Bytes between rows are not written: the mirror must hold the guest's version of them.
            UploadRange(*pool, s.pool_offset, geometry.span);
        }
        const bool saved = std::exchange(track_writes, false);
        PrepareImages({{&s, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL}});
        track_writes = saved;
        Cmd().CopyImageToBuffer(*s.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pool->Handle(),
                                geometry.regions);
        if (geometry.padded) {
            NotePoolWrite(*pool, s.pool_offset, geometry.span, true, row_bytes, stride, copy_rows);
        } else {
            NotePoolWrite(*pool, s.pool_offset, geometry.span);
        }
        barrier_pending = true;
    }
}

void Recorder::MakeNativeCurrent(Storage& s) {
    if (!s.scaled || !Rescale::NativeIsStale(s.scale_state)) {
        return;
    }
    Storage& twin = *s.scaled;
    if (!impl.EnsureImage(s) || !impl.EnsureImage(twin)) {
        LogOnce("resolution scale: the guest-sized image of a scaled texture could not be made; "
                "reads of it see stale pixels");
        return;
    }
    NoteTouched(s.shared_from_this());
    NoteTouched(s.scaled);
    const bool saved = track_writes;
    track_writes = false;
    PrepareImages({{&twin, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
                   {&s, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
    track_writes = saved;
    impl.RecordScaleBlit(Cmd(), s, false);
    barrier_pending = true;
    s.scale_state = Rescale::AfterBlit();
    impl.stats.rescale_downscales.fetch_add(1, std::memory_order_relaxed);
}

bool Recorder::PrepareScaledTarget(Storage& native) {
    Storage* const twin = native.scaled.get();
    if (twin == nullptr || !impl.EnsureImage(*twin)) {
        return false;
    }
    if (native.scale_state == Rescale::State::Native && native.image &&
        native.origin != StorageOrigin::None && native.contents_valid) {
        // GPU-written pixels (or an upload) only the guest-sized image has: the pass loads them.
        NoteTouched(native.shared_from_this());
        NoteTouched(native.scaled);
        const bool saved = track_writes;
        track_writes = false;
        PrepareImages({{&native, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
                       {twin, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
        track_writes = saved;
        impl.RecordScaleBlit(Cmd(), native, true);
        barrier_pending = true;
        native.scale_state = Rescale::AfterBlit();
        impl.stats.rescale_upscales.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

Storage& Recorder::PresentSource(Storage& guest) {
    if (guest.scaled && Rescale::TwinIsCurrent(guest.scale_state) &&
        impl.EnsureImage(*guest.scaled)) {
        NoteTouched(guest.scaled);
        return *guest.scaled;
    }
    return guest;
}

bool Recorder::CopyTextureScaled(Storage& src, Storage& dst, u32 src_layer, u32 dst_layer,
                                 u32 layers, const Rescale::CopyRegion& region, bool dst_covered) {
    const Rescale::Factor factor = impl.RescaleFactor();
    if (&src == &dst || !src.scaled || !Rescale::TwinIsCurrent(src.scale_state) ||
        dst.scale_owner != nullptr ||
        Rescale::Classify(factor, impl.RescaleShapeOf(dst, false)) != Rescale::Reason::Eligible) {
        return false;
    }
    // Guest bytes (an upload, an overlapping texture) that change the guest-sized source make the
    // twin stale: the plain copy runs, after the downscale or upload that makes it current.
    if (impl.StorageNeedsCommands(src, false, flight.seq, true)) {
        return false;
    }
    Storage* const dst_twin = impl.ScaledTwinOf(dst);
    if (dst_twin == nullptr) {
        return false;
    }
    Storage& src_twin = *src.scaled;
    if (!impl.EnsureImage(src_twin) || !impl.EnsureImage(*dst_twin)) {
        return false;
    }
    const Rescale::CopyRegion scaled =
        Rescale::ScaleCopy(factor, region, src_twin.extent.width, src_twin.extent.height,
                           dst_twin->extent.width, dst_twin->extent.height);
    if (scaled.width == 0 || scaled.height == 0) {
        return false;
    }
    // The destination keeps what the copy does not cover: its own guest-side machinery first
    // (uploads, overlapping textures), then its twin made current.
    if (!EnsureContents(dst, dst_covered, true) || (!dst_covered && !PrepareScaledTarget(dst))) {
        return false;
    }
    NoteTouched(src.scaled);
    NoteTouched(dst.scaled);
    PrepareImages({{&src_twin, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
                   {dst_twin, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
    const VkImageCopy copy{
        .srcSubresource = {CopyAspect(src_twin), 0, src_layer, layers},
        .srcOffset = {scaled.src_x, scaled.src_y, 0},
        .dstSubresource = {CopyAspect(*dst_twin), 0, dst_layer, layers},
        .dstOffset = {scaled.dst_x, scaled.dst_y, 0},
        .extent = {scaled.width, scaled.height, 1},
    };
    Cmd().CopyImage(*src_twin.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *dst_twin->image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copy);
    barrier_pending = true;
    impl.stats.rescale_copies.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void Recorder::UploadStorage(Storage& s) {
    const Common::FrameStats::Scope stall{Common::FrameStats::Cause::TextureUpload};
    const bool again = s.origin != StorageOrigin::None || s.force_upload;
    s.force_upload = false;
    s.force_check = false;
    if (s.samples != VK_SAMPLE_COUNT_1_BIT) {
        return;
    }
    const auto pool_it = impl.pools.find(s.pool_serial);
    if (pool_it == impl.pools.end()) {
        return;
    }
    Pool& pool = *pool_it->second;
    if (!pool.pending.load(std::memory_order_acquire) && pool.backing == Pool::Backing::None) {
        return; // no backing: nothing to read
    }
    if (s.pool_offset >= pool.size) {
        return;
    }
    const u64 size = (std::min)(s.layout.size, pool.size - s.pool_offset);
    GpuProf::Scope prof{flight.gpu_prof.get(), Cmd(),
                        flight.gpu_prof
                            ? ProfStorage(GpuProf::Kind::Upload, s, GpuProf::SUB_STORAGE)
                            : GpuProf::Tag{}};
    if (size == 0) {
        return;
    }
    const bool cpu_visible = pool.IsVirtual() || (pool.flags & PoolFlag::CpuNoAccess) == 0;
    u64 upload_tracked_pool = 0;
    u32 upload_tracked_gen = 0;
    if (cpu_visible && impl.tex_hash_tracked) {
        // Sample the physical pool's write generation before GuestBytes gathers the data so any
        // concurrent CPU write after the gather shows a strictly newer generation.
        Pool* track_pool = &pool;
        u64 track_offset = s.pool_offset;
        const bool resolved =
            !pool.IsVirtual() ||
            (impl.ResolveVirtualRange(pool, track_offset, size, track_pool, track_offset) &&
             track_pool != nullptr);
        if (resolved && (track_pool->flags & PoolFlag::CpuNoAccess) == 0 &&
            impl.TakePoolCpuWrites(*track_pool)) {
            upload_tracked_pool = track_pool->serial;
            upload_tracked_gen = track_pool->cpu_write_gen;
        }
    }
    const auto log_begin = std::chrono::steady_clock::now();
    std::vector<u8> scratch;
    const u8* data = impl.GuestBytes(pool, s.pool_offset, size, scratch);
    const auto log_guest = std::chrono::steady_clock::now();
    if (!data) {
        return;
    }
    const UploadPlan plan = PlanStorageUpload(s, size);
    if (plan.unsupported) {
        if (s.host.guest_layout_differs) {
            impl.stats.depth_upload_unsupported.fetch_add(1, std::memory_order_relaxed);
        }
        LogOnce(fmt::format("storage upload skipped (format {:#x}): {}", s.desc.format,
                            plan.unsupported));
        return;
    }
    if (plan.pieces.empty() || plan.staging_size == 0) {
        return;
    }
    if (s.host.bc_standin && !RefitStandin(s, data, size)) {
        return; // no image to upload into (EnsureImage logged it)
    }
    // Hash before the fill: bytes a decoder rewrites meanwhile must look changed at the next check
    // (the image holds what the fill read, the hash must not describe newer bytes).
    const auto log_planned = std::chrono::steady_clock::now();
    if (cpu_visible) {
        const auto begin = std::chrono::steady_clock::now();
        s.guest_hash = HashGuestBytes(data, size);
        s.sample_hash = SampleGuestBytes(data, size, impl.mario_kart_world);
        s.cpu_trusted_pool = upload_tracked_pool;
        s.cpu_trusted_gen = upload_tracked_gen;
        s.cpu_full_gen = upload_tracked_gen;
        s.page_hashes.clear();
        impl.stats.hash_ns.fetch_add(
            static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - begin)
                                 .count()),
            std::memory_order_relaxed);
        impl.stats.hash_bytes.fetch_add(size, std::memory_order_relaxed);
    }
    const auto log_hashed = std::chrono::steady_clock::now();
    auto log_staged = log_hashed;
    auto log_filled = log_hashed;
    UploadStats upload_stats;
    bool on_gpu = false;
    if ((s.astc_gpu || s.host.bc_standin) && impl.astc_gpu_enabled &&
        plan.decode == UploadPlan::Decode::Astc) {
        on_gpu = s.host.bc_standin ? UploadAstcBcOnGpu(s, plan, pool, data, size)
                                   : UploadAstcOnGpu(s, plan, pool, data, size);
        if (on_gpu && Debug::Get().astc_verify && !s.host.bc_standin) {
            VerifyAstcGpu(s, plan, data, size);
        }
        if (!on_gpu) {
            impl.stats.astc_cpu_fallbacks.fetch_add(1, std::memory_order_relaxed);
        }
    }
    // Block-linear bytes the CPU would only untile: the GPU does it.
    bool untiled_on_gpu = false;
    if (!on_gpu && !plan.linear && impl.untile_gpu_enabled &&
        plan.staging_size >= UNTILE_GPU_MIN_BYTES) {
        untiled_on_gpu = UploadUntileOnGpu(s, plan, pool, data, size);
        if (!untiled_on_gpu) {
            impl.stats.untile_cpu_fallbacks.fetch_add(1, std::memory_order_relaxed);
        }
    }
    StagingRange staging{};
    if (on_gpu) {
        impl.stats.astc_gpu_uploads.fetch_add(1, std::memory_order_relaxed);
        impl.stats.astc_gpu_bytes.fetch_add(size, std::memory_order_relaxed);
        log_staged = log_filled = std::chrono::steady_clock::now();
    } else if (untiled_on_gpu) {
        log_staged = log_filled = std::chrono::steady_clock::now();
    } else {
        staging = AllocStaging(plan.staging_size, 64);
        log_staged = std::chrono::steady_clock::now();
        FillStorageUpload(s, plan, data, size, staging.pointer, &upload_stats);
        log_filled = std::chrono::steady_clock::now();
        impl.stats.uploads.fetch_add(1, std::memory_order_relaxed);
        impl.stats.upload_bytes.fetch_add(plan.staging_size, std::memory_order_relaxed);
        impl.stats.decode_ns.fetch_add(upload_stats.decode_ns, std::memory_order_relaxed);
        impl.stats.decode_bytes.fetch_add(upload_stats.decode_bytes, std::memory_order_relaxed);
        if (s.host.bc_standin) {
            UploadBcFromStaging(s, plan, staging);
        } else {
            const std::vector<VkBufferImageCopy> regions =
                MakeUploadRegions(s, plan, staging.offset);
            const bool saved = track_writes;
            track_writes = false;
            PrepareImages({{&s, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
            track_writes = saved;
            Cmd().CopyBufferToImage(staging.buffer, *s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    regions);
            barrier_pending = true;
        }
    }
    if (Debug::Get().upload_log) {
        const auto log_end = std::chrono::steady_clock::now();
        const auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        LOG_INFO(
            Render_Vulkan,
            "[nvn] upload: present {} seq {} {}x{}x{} lv {} ly {} fmt {:#x} {} {} {} {:.2f} MB "
            "guest -> {:.2f} MB staged, pieces {}, pool flags {:#x} {} | ms: total {:.2f} "
            "guest {:.2f} plan {:.2f} hash {:.2f} staging {:.2f} fill {:.2f} record {:.2f} "
            "decode {:.2f}",
            impl.stats.presents.load(std::memory_order_relaxed), flight.seq, s.extent.width,
            s.extent.height, s.extent.depth, s.levels, s.layers, s.desc.format,
            on_gpu && s.host.bc_standin               ? "astcgpu-bc"
            : on_gpu                                  ? "astcgpu"
            : untiled_on_gpu                          ? "untilegpu"
            : plan.decode == UploadPlan::Decode::Astc ? "astc"
            : plan.decode == UploadPlan::Decode::Bcn  ? "bcn"
                                                      : "plain",
            plan.linear ? "linear" : "tiled", again ? "again" : "first", size / 1048576.0,
            plan.staging_size / 1048576.0, plan.pieces.size(), pool.flags,
            pool.IsVirtual() ? "virtual" : (pool.host ? "host" : "nohost"), ms(log_begin, log_end),
            ms(log_begin, log_guest), ms(log_guest, log_planned), ms(log_planned, log_hashed),
            ms(log_hashed, log_staged), ms(log_staged, log_filled), ms(log_filled, log_end),
            upload_stats.decode_ns / 1e6);
    }

    // From here on the contents are the guest's bytes at this moment. A CPU-accessible pool can
    // change them behind our back: remember what they were, to notice. A GPU-only pool's bytes
    // are the game's initial fill and never change again.
    s.contents_valid = true; // RefitStandin may have replaced the image above.
    s.origin = cpu_visible ? StorageOrigin::Cpu : StorageOrigin::Gpu;
    s.linear_writeback_seq = 0;
    s.watch_valid = false;
    s.unchanged_checks = 0;
    const u64 now = impl.PollTick();
    s.next_check_tick = now + 1;
    s.next_full_tick = now + FULL_VERIFY_PRESENTS;
    // The first upload predates every GPU write to overlapping storages (they are copied over it
    // by ResolveOverlaps, unless the CPU rewrote their bytes after that write); a later one has
    // the CPU's newer bytes.
    s.sync_gen = again ? impl.write_clock : 0;
    s.pool_gen_seen = ~u64{0};
    if (s.scaled) {
        s.scale_state =
            Rescale::AfterNativeWrite(); // the guest bytes changed the guest-sized image
    }
}

void Recorder::SyncSparse(Storage& s, bool will_overwrite) {
    SparseBacking& sparse = *s.sparse;
    s.contents_valid = true;
    if (!sparse.HasWork()) {
        return;
    }
    const Common::FrameStats::Scope stall{Common::FrameStats::Cause::TextureUpload};
    SparseBacking::Uploads uploads;
    if (!sparse.Prepare(impl, s, uploads)) {
        return;
    }
    // Unmapped coarse levels have real memory on MKW. Initialize it once on residency,
    // before partial GPU copies can leave arbitrary old allocation contents visible.
    if (!uploads.zero_tails.empty() && s.aspect == VK_IMAGE_ASPECT_COLOR_BIT) {
        const auto& sparse_layout = sparse.Layout();
        const BlockInfo block = s.host.bc_standin ? BlockInfo{BcStandinBlockBytes(s.host), 4, 4}
                                : s.host.needs_transcode
                                    ? BlockInfo{DecodedTexelBytes(s.host), 1, 1}
                                    : sparse_layout.block;
        for (u32 layer : uploads.zero_tails) {
            u64 bytes = 0;
            std::vector<VkBufferImageCopy> regions;
            for (u32 level = static_cast<u32>(sparse_layout.level.size()); level < s.levels;
                 ++level) {
                const u32 width = (std::max)(s.extent.width >> level, 1u);
                const u32 height = (std::max)(s.extent.height >> level, 1u);
                bytes = (bytes + 15) / 16 * 16;
                regions.push_back({.bufferOffset = bytes,
                                   .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, layer, 1},
                                   .imageExtent = {width, height, 1}});
                bytes += u64{(width + block.width - 1) / block.width} *
                         ((height + block.height - 1) / block.height) * block.bytes;
            }
            if (bytes != 0) {
                const StagingRange staging = AllocStaging(bytes, 16);
                std::memset(staging.pointer, 0, static_cast<size_t>(bytes));
                for (auto& region : regions) {
                    region.bufferOffset += staging.offset;
                }
                const bool saved = track_writes;
                track_writes = false;
                PrepareImages({{&s, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
                track_writes = saved;
                Cmd().CopyBufferToImage(staging.buffer, *s.image,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regions);
                barrier_pending = true;
            }
        }
    }
    if (will_overwrite || (uploads.tiles.empty() && uploads.tails.empty())) {
        return;
    }
    const auto pool_it = impl.pools.find(s.pool_serial);
    if (pool_it == impl.pools.end()) {
        return;
    }
    const Pool& pool = *pool_it->second;
    const SparseLayout& layout = sparse.Layout();
    const u64 tile_bytes = SparseTileStagingBytes(s, layout);
    const UploadPlan& tail_plan = sparse.TailPlan();
    const u64 tail_bytes = layout.TailSize();
    const auto align = [](u64 value) { return (value + 63) / 64 * 64; };
    // One staging buffer per chunk of at most 64 MiB.
    constexpr u64 CHUNK_BYTES = u64{64} << 20;
    size_t next_tile = 0;
    size_t next_tail = 0;
    while (next_tile < uploads.tiles.size() || next_tail < uploads.tails.size()) {
        u64 total = 0;
        const size_t first_tile = next_tile;
        std::vector<u64> tile_offsets;
        while (next_tile < uploads.tiles.size() &&
               (total == 0 || total + tile_bytes <= CHUNK_BYTES)) {
            total = align(total);
            tile_offsets.push_back(total);
            total += tile_bytes;
            ++next_tile;
        }
        const size_t first_tail = next_tail;
        std::vector<u64> tail_offsets;
        while (next_tail < uploads.tails.size() &&
               (total == 0 || total + tail_plan.staging_size <= CHUNK_BYTES)) {
            total = align(total);
            tail_offsets.push_back(total);
            total += tail_plan.staging_size;
            ++next_tail;
        }
        const StagingRange staging = AllocStaging(total, 64);
        std::vector<u64> tile_pools(tile_offsets.size(), 0);
        std::vector<u32> tile_gens(tile_offsets.size(), 0);
        if (impl.tex_hash_tracked) {
            for (size_t i = 0; i < tile_offsets.size(); ++i) {
                const u32 id = uploads.tiles[first_tile + i];
                Pool* phys = nullptr;
                u64 phys_offset = 0;
                if (impl.ResolveVirtualRange(pool, s.pool_offset + layout.TileOffset(id),
                                             SPARSE_TILE_BYTES, phys, phys_offset) &&
                    phys != nullptr && (phys->flags & PoolFlag::CpuNoAccess) == 0 &&
                    impl.TakePoolCpuWrites(*phys)) {
                    tile_pools[i] = phys->serial;
                    tile_gens[i] = phys->cpu_write_gen;
                }
            }
        }
        std::vector<u64> hashes(tile_offsets.size());
        const auto log_begin = std::chrono::steady_clock::now();
        RunParallel(tile_offsets.size(), [&](size_t i) {
            const u32 id = uploads.tiles[first_tile + i];
            std::vector<u8> bytes(static_cast<size_t>(SPARSE_TILE_BYTES), 0);
            impl.GatherVirtual(pool, s.pool_offset + layout.TileOffset(id), SPARSE_TILE_BYTES,
                               bytes.data());
            hashes[i] =
                Common::CityHash64(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            DecodeSparseTile(
                s, layout, bytes,
                std::span<u8>{staging.pointer + tile_offsets[i], static_cast<size_t>(tile_bytes)});
        });
        std::vector<VkBufferImageCopy> regions;
        regions.reserve(tile_offsets.size() + tail_offsets.size() * tail_plan.pieces.size());
        for (size_t i = 0; i < tile_offsets.size(); ++i) {
            const u32 id = uploads.tiles[first_tile + i];
            const SparseTileRef ref = layout.Tile(id);
            const SparseRect rect = layout.TileRect(ref);
            regions.push_back({
                .bufferOffset = staging.offset + tile_offsets[i],
                .bufferRowLength = layout.tile_width,
                .bufferImageHeight = layout.tile_height,
                .imageSubresource{
                    .aspectMask = s.aspect,
                    .mipLevel = ref.level,
                    .baseArrayLayer = ref.layer,
                    .layerCount = 1,
                },
                .imageOffset = {static_cast<s32>(rect.x), static_cast<s32>(rect.y), 0},
                .imageExtent = {rect.width, rect.height, 1},
            });
            sparse.SetTileHash(id, hashes[i]);
            sparse.SetTileVerifiedGen(id, tile_pools[i], tile_gens[i]);
        }
        for (size_t i = 0; i < tail_offsets.size(); ++i) {
            const u32 layer = uploads.tails[first_tail + i];
            std::vector<u8> bytes(static_cast<size_t>(tail_bytes), 0);
            impl.GatherVirtual(pool,
                               s.pool_offset + static_cast<u64>(layer) * layout.layer_size +
                                   layout.tail_offset,
                               tail_bytes, bytes.data());
            FillStorageUpload(s, tail_plan, bytes.data(), bytes.size(),
                              staging.pointer + tail_offsets[i]);
            for (VkBufferImageCopy region :
                 MakeUploadRegions(s, tail_plan, staging.offset + tail_offsets[i])) {
                region.imageSubresource.baseArrayLayer = layer;
                regions.push_back(region);
            }
        }
        impl.stats.uploads.fetch_add(tile_offsets.size() + tail_offsets.size(),
                                     std::memory_order_relaxed);
        impl.stats.upload_bytes.fetch_add(total, std::memory_order_relaxed);
        if (Debug::Get().upload_log) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] sparse upload: {}x{}x{} fmt {:#x}: {} tiles, {} tails, {:.2f} MB "
                     "staged in {:.2f} ms ({} tiles mapped, {:.0f} MiB bound)",
                     s.extent.width, s.extent.height, layout.layers, s.desc.format,
                     tile_offsets.size(), tail_offsets.size(), total / 1048576.0,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                               log_begin)
                         .count(),
                     sparse.MappedTiles(), sparse.BoundBytes() / 1048576.0);
        }
        if (regions.empty()) {
            continue;
        }
        const bool saved = track_writes;
        track_writes = false;
        PrepareImages({{&s, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
        track_writes = saved;
        if (s.host.bc_standin) {
            std::vector<BcChunk> chunks;
            chunks.reserve(regions.size());
            for (const auto& region : regions) {
                chunks.push_back({region.imageSubresource.mipLevel,
                                  region.imageSubresource.baseArrayLayer, 1,
                                  region.imageExtent.width, region.imageExtent.height,
                                  static_cast<u32>(region.imageOffset.x),
                                  static_cast<u32>(region.imageOffset.y)});
            }
            size_t next = 0;
            if (!RecordBcUpload(s, chunks, true, [&](const BcChunk&, BcScratch& scratch) {
                    VkBufferImageCopy region = regions[next++];
                    region.imageSubresource.mipLevel = 0;
                    region.imageSubresource.baseArrayLayer = 0;
                    region.imageOffset = {0, 0, 0};
                    Cmd().CopyBufferToImage(staging.buffer, *scratch.image, VK_IMAGE_LAYOUT_GENERAL,
                                            region);
                })) {
                // Prepare cleared the dirty flags; failed uploads must be retried.
                sparse.MarkAllDirty();
                s.contents_valid = false;
                return;
            }
        } else {
            Cmd().CopyBufferToImage(staging.buffer, *s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    regions);
        }
        barrier_pending = true;
    }
    // From here on the contents are the guest's bytes (a texture the GPU wrote keeps its origin).
    if (s.origin == StorageOrigin::None) {
        s.origin = StorageOrigin::Cpu;
    }
}

bool Recorder::AstcGpuLevels(const Storage& s, const UploadPlan& plan, u64 input_offset,
                             const AstcGpuDecoder& decoder,
                             std::vector<AstcGpuDecoder::Level>& levels) {
    if (plan.linear || s.type != VK_IMAGE_TYPE_2D || s.samples != VK_SAMPLE_COUNT_1_BIT ||
        s.layers == 0 || s.layout.layer_size > 0xFFFFFFFFull || (s.layout.layer_size & 15) != 0) {
        return false;
    }
    const BlockInfo block = FormatBlockInfo(s.desc.format);
    levels.clear();
    levels.reserve(plan.pieces.size());
    for (const UploadPiece& piece : plan.pieces) {
        if (piece.layer != 0) {
            continue;
        }
        if (piece.slices != 1 ||
            ((input_offset + piece.guest_offset) % decoder.OffsetAlignment()) != 0) {
            return false;
        }
        levels.push_back({
            .level = piece.level,
            .layers = s.layers,
            .columns = piece.cols,
            .rows = piece.rows,
            .offset = piece.guest_offset,
            .layer_stride = static_cast<u32>(s.layout.layer_size),
            .block_width = block.width,
            .block_height = block.height,
            .tile_height = piece.tile_h,
            .tile_depth = piece.tile_d,
        });
    }
    // Every level of every layer must be in the plan (a level that starts past the storage is
    // skipped by the plan and cannot be decoded either).
    return !levels.empty() && plan.pieces.size() == levels.size() * s.layers;
}

bool Recorder::UploadAstcOnGpu(Storage& s, const UploadPlan& plan, Pool& pool, const u8* data,
                               u64 size) {
    AstcGpuDecoder* const decoder = impl.AstcDecoder();
    if (!decoder || size != s.layout.size) {
        return false;
    }
    std::vector<AstcGpuDecoder::Level> levels;
    VkBuffer input{};
    u64 input_offset{};
    u64 input_size = size;
    if (!pool.pending.load(std::memory_order_acquire) && pool.backing == Pool::Backing::Imported &&
        AstcGpuLevels(s, plan, s.pool_offset, *decoder, levels)) {
        input = pool.Handle();
        input_offset = s.pool_offset;
        input_size = pool.size - s.pool_offset;
        // PrepareImages orders writes from overlapping storage shadows before the decode.
        if (draw.HasShadows()) {
            draw.FlushShadowsOverlapping(pool, s.pool_offset, s.layout.size);
        }
    } else {
        // MKW streams ASTC 8x8 textures into its lighting atlases. Stage the compressed GOBs
        // from mirrored pools to keep the CPU replay thread out of large RGBA8 decodes.
        constexpr u64 MinStagedAstcBytes = u64{64} << 10;
        constexpr u64 MaxStagedAstcBytes = u64{64} << 20;
        if (!impl.mario_kart_world || s.sparse ||
            size < MinStagedAstcBytes || size > MaxStagedAstcBytes ||
            size > impl.device.GetMaxStorageBufferRange() ||
            !AstcGpuLevels(s, plan, 0, *decoder, levels)) {
            return false;
        }
        try {
            const StagingRange staging = AllocStaging(size, decoder->OffsetAlignment());
            std::memcpy(staging.pointer, data, static_cast<size_t>(size));
            input = staging.buffer;
            input_offset = staging.offset;
        } catch (const vk::Exception& exception) {
            LogOnce(fmt::format("MKW ASTC staging allocation failed: {}", exception.what()));
            return false;
        }
    }
    const bool saved = track_writes;
    track_writes = false;
    PrepareImages({{&s, VK_IMAGE_LAYOUT_GENERAL}});
    track_writes = saved;
    auto views = std::make_shared<std::vector<vk::ImageView>>(
        decoder->Record(Cmd(), input, input_offset, input_size, *s.image, levels));
    flight.keepalive.push_back(std::move(views));
    barrier_pending = true;
    return true;
}

bool Recorder::UploadAstcBcOnGpu(Storage& s, const UploadPlan& plan, Pool& pool, const u8* data,
                                 u64 size) {
    AstcGpuDecoder* const decoder = impl.AstcDecoder();
    std::vector<AstcGpuDecoder::Level> levels;
    if (!decoder || size != s.layout.size || s.layout.size > pool.size - s.pool_offset) {
        return false;
    }
    VkBuffer input{};
    u64 input_offset{};
    u64 input_size{};
    if (!pool.pending.load(std::memory_order_acquire) && pool.backing == Pool::Backing::Imported &&
        AstcGpuLevels(s, plan, s.pool_offset, *decoder, levels)) {
        input = pool.Handle();
        input_offset = s.pool_offset;
        input_size = pool.size - s.pool_offset;
    } else {
        // Keep MKW's streaming sources on the GPU decode path with mirrored guest pools too.
        if (!impl.mario_kart_world || size < (u64{64} << 10) ||
            size > (u64{64} << 20) || size > impl.device.GetMaxStorageBufferRange() ||
            !AstcGpuLevels(s, plan, 0, *decoder, levels)) {
            return false;
        }
        const StagingRange staging = AllocStaging(size, decoder->OffsetAlignment());
        std::memcpy(staging.pointer, data, static_cast<size_t>(size));
        input = staging.buffer;
        input_offset = staging.offset;
        input_size = size;
    }
    // The decoder writes level 0 of the scratch: one dispatch per chunk of layers, each bound from
    // its first layer's bytes.
    std::vector<BcChunk> chunks;
    std::vector<AstcGpuDecoder::Level> chunk_levels;
    for (const AstcGpuDecoder::Level& level : levels) {
        const u32 width = (std::max)(s.extent.width >> level.level, 1u);
        const u32 height = (std::max)(s.extent.height >> level.level, 1u);
        for (const BcChunk& chunk : SplitBcLayers(level.level, 0, s.layers, width, height)) {
            AstcGpuDecoder::Level part = level;
            part.level = 0;
            part.layers = chunk.layers;
            part.offset += u64{chunk.first_layer} * level.layer_stride;
            if ((input_offset + part.offset) % decoder->OffsetAlignment() != 0) {
                return false;
            }
            chunks.push_back(chunk);
            chunk_levels.push_back(part);
        }
    }
    if (draw.HasShadows()) {
        draw.FlushShadowsOverlapping(pool, s.pool_offset, s.layout.size);
    }
    FlushPending(); // earlier writes of the pool's bytes before the decoder reads them
    size_t next = 0;
    return RecordBcUpload(s, chunks, true, [&](const BcChunk&, BcScratch& scratch) {
        auto views = std::make_shared<std::vector<vk::ImageView>>(
            decoder->Record(Cmd(), input, input_offset, input_size, *scratch.image,
                            std::span{&chunk_levels[next++], 1}));
        flight.keepalive.push_back(std::move(views));
    });
}

void Recorder::UploadBcFromStaging(Storage& s, const UploadPlan& plan,
                                   const StagingRange& staging) {
    // The plan's pieces by level, in layer order: runs of consecutive layers fill one chunk.
    std::vector<const UploadPiece*> pieces;
    pieces.reserve(plan.pieces.size());
    for (const UploadPiece& piece : plan.pieces) {
        pieces.push_back(&piece);
    }
    std::ranges::stable_sort(pieces, [](const UploadPiece* a, const UploadPiece* b) {
        return a->level != b->level ? a->level < b->level : a->layer < b->layer;
    });
    std::vector<BcChunk> chunks;
    std::vector<size_t> chunk_first; // index in `pieces` of each chunk's first layer
    for (size_t i = 0; i < pieces.size();) {
        size_t end = i + 1;
        while (end < pieces.size() && pieces[end]->level == pieces[i]->level &&
               pieces[end]->layer == pieces[end - 1]->layer + 1) {
            ++end;
        }
        size_t first = i;
        for (const BcChunk& chunk :
             SplitBcLayers(pieces[i]->level, pieces[i]->layer, static_cast<u32>(end - i),
                           pieces[i]->width, pieces[i]->height)) {
            chunks.push_back(chunk);
            chunk_first.push_back(first);
            first += chunk.layers;
        }
        i = end;
    }
    size_t next = 0;
    const auto fill = [&](const BcChunk& chunk, BcScratch& scratch) {
        const size_t first = chunk_first[next++];
        std::vector<VkBufferImageCopy> regions;
        regions.reserve(chunk.layers);
        for (u32 i = 0; i < chunk.layers; ++i) {
            const UploadPiece& piece = *pieces[first + i];
            regions.push_back({
                .bufferOffset = staging.offset + piece.staging_offset,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource{
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel = 0,
                    .baseArrayLayer = i,
                    .layerCount = 1,
                },
                .imageOffset = {0, 0, 0},
                .imageExtent = {piece.width, piece.height, 1},
            });
        }
        Cmd().CopyBufferToImage(staging.buffer, *scratch.image, VK_IMAGE_LAYOUT_GENERAL, regions);
    };
    if (!RecordBcUpload(s, chunks, true, fill)) {
        s.contents_valid = false;
        LogOnce(fmt::format("BC stand-in upload (format {:#x}) failed: retrying on next use",
                            s.desc.format));
    }
}

bool Recorder::Restandin(Storage& s, AstcStandin standin, bool deferred) {
    if (deferred) {
        // The image keeps its format until the next submission (see Storage::pending_standin);
        // its contents are re-uploaded once it is recreated, as this one's upload wrote the old one.
        s.pending_standin = standin;
        impl.pending_standins.push_back(s.shared_from_this());
        return true;
    }
    if (s.image) {
        // Earlier submissions and this one may still read the old image through its views.
        auto old_image = std::make_shared<vk::Image>(std::move(s.image));
        auto old_views = std::make_shared<std::vector<vk::ImageView>>(
            impl.draw_resources ? impl.draw_resources->TakeStorageViews(s)
                                : std::vector<vk::ImageView>{});
        impl.retired.push_back({flight.seq, [old_image, old_views]() mutable {
                                    old_views->clear();
                                    *old_image = vk::Image{};
                                }});
        s.image = vk::Image{};
        if (s.cached) {
            impl.storage_cache_bytes -= (std::min)(impl.storage_cache_bytes, s.vram_bytes);
        }
    }
    impl.SetStandin(s, standin);
    s.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    s.contents_valid = false;
    NoteTouched(s.shared_from_this()); // the cache trim of EnsureImage leaves it alone
    return impl.EnsureImage(s);
}

bool Recorder::RefitStandin(Storage& s, const u8* data, u64 size) {
    if (!s.smart_standin) {
        return true;
    }
    const AstcStandin current = BcStandinOf(s.host);
    if (s.standin_pinned) {
        return current == AstcStandin::Bc3 || Restandin(s, AstcStandin::Bc3);
    }
    const bool srgb =
        VideoCore::Surface::IsPixelFormatSRGB(s.host.pixel_format) || s.standin_srgb_view;
    if (s.standin_fresh) {
        s.standin_fresh = false; // EnsureImage classified these bytes a moment ago
    } else {
        bool empty = false;
        const auto channels = impl.ClassifyStandin(s, data, size, empty);
        if (!empty && s.standin_empty) {
            // The first blocks of a stand-in created over zero bytes choose its format.
            s.standin_empty = false;
            s.standin_class = channels;
            const AstcStandin chosen = SmartStandinFor(channels, srgb, impl.smart_bc);
            return chosen == current || Restandin(s, chosen, true);
        }
        if (!empty && !StandinHolds(current, channels)) {
            static constexpr std::array<const char*, 4> NAMES{"L", "LA", "RGB", "RGBA"};
            const u64 refits = impl.smart_total.refits.fetch_add(1, std::memory_order_relaxed);
            impl.smart_window.refits.fetch_add(1, std::memory_order_relaxed);
            if (refits < 64) {
                LOG_INFO(Render_Vulkan,
                         "[nvn] smart BC: {} stand-in {}x{}x{} ({} levels, format {:#x}, pool "
                         "{:#x} +{:#x}) was {}, new bytes are {}: recreated as BC3",
                         AstcStandinName(current), s.extent.width, s.extent.height, s.layers,
                         s.levels, s.desc.format, s.pool_serial, s.pool_offset,
                         NAMES[static_cast<size_t>(s.standin_class)],
                         NAMES[static_cast<size_t>(channels)]);
            }
            s.standin_pinned = true;
            s.standin_class = Tegra::Texture::ASTC::Channels::Rgba;
            return Restandin(s, AstcStandin::Bc3);
        }
    }
    // This upload was forced by an sRGB view. The draw needs the compatible image now.
    if (s.standin_srgb_view && (current == AstcStandin::Bc4 || current == AstcStandin::Bc5)) {
        s.pending_standin.reset();
        return Restandin(s, SmartStandinFor(s.standin_class, true, impl.smart_bc));
    }
    return true;
}

void Recorder::ApplyPendingStandins() {
    if (impl.pending_standins.empty()) {
        return;
    }
    std::vector<std::weak_ptr<Storage>> pending{std::move(impl.pending_standins)};
    impl.pending_standins.clear();
    for (const std::weak_ptr<Storage>& weak : pending) {
        const std::shared_ptr<Storage> storage = weak.lock();
        if (!storage || !storage->pending_standin) {
            continue;
        }
        const AstcStandin standin = *storage->pending_standin;
        storage->pending_standin.reset();
        Restandin(*storage, standin, false);
    }
}

bool Recorder::PinStandin(Storage& s, std::string_view op) {
    if (!s.smart_standin || s.standin_pinned) {
        return true;
    }
    s.standin_pinned = true;
    s.standin_empty = false;
    s.standin_fresh = false;
    s.standin_class = Tegra::Texture::ASTC::Channels::Rgba;
    impl.smart_total.pinned.fetch_add(1, std::memory_order_relaxed);
    impl.smart_window.pinned.fetch_add(1, std::memory_order_relaxed);
    if (BcStandinOf(s.host) == AstcStandin::Bc3) {
        return true;
    }
    LogOnce(fmt::format("smart BC: {} of a {} stand-in (format {:#x}) recreates it as BC3",
                        op, AstcStandinName(BcStandinOf(s.host)), s.desc.format));
    if (!s.image) {
        impl.SetStandin(s, AstcStandin::Bc3);
        return true;
    }
    // Only uploads wrote it: the caller's EnsureContents uploads the guest bytes again.
    return Restandin(s, AstcStandin::Bc3);
}

bool Recorder::RecordBcUpload(Storage& s, std::span<const BcChunk> chunks, bool guest_contents,
                              const std::function<void(const BcChunk&, BcScratch&)>& fill) {
    BcGpuEncoder* const encoder = impl.BcEncoder();
    if (!encoder || chunks.empty()) {
        return false;
    }
    const u32 block_bytes = BcStandinBlockBytes(s.host);
    const AstcStandin kind = BcStandinOf(s.host);
    if (kind == AstcStandin::Bc7 && !encoder->HasBc7()) {
        return false;
    }
    const bool saved = track_writes;
    track_writes = track_writes && !guest_contents;
    PrepareImages({{&s, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
    track_writes = saved;
    // The upload's GPU time: a timestamp pair around its chunks (CompleteBcTimes).
    BcUploadTimes* const times = BcTimes();
    u32 query = 0;
    if (times) {
        query = static_cast<u32>(times->uploads.size() * 2);
        times->uploads.push_back({
            .width = s.extent.width,
            .height = s.extent.height,
            .layers = s.layers,
            .levels = s.levels,
            .format = s.desc.format,
            .kind = AstcStandinName(kind),
        });
        Cmd().WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *times->pool, query);
    }
    bool recorded = false;
    for (const BcChunk& chunk : chunks) {
        const std::shared_ptr<BcScratch> scratch =
            impl.AcquireBcScratch(chunk.width, chunk.height, chunk.layers);
        if (!scratch) {
            if (times) {
                Cmd().WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *times->pool, query + 1);
            }
            return false;
        }
        flight.keepalive.push_back(scratch);
        // The scratch is shared by every upload: whatever used it last (here or in an earlier
        // submission) is done before it is written again; its old texels are not kept.
        const VkMemoryBarrier reuse{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        const VkImageMemoryBarrier to_general{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *scratch->image,
            .subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS},
        };
        Cmd().PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, reuse, {}, to_general);
        fill(chunk, *scratch);
        const VkMemoryBarrier texels_written{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        Cmd().PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, texels_written);
        const u32 blocks_x = (chunk.width + 3) / 4;
        const u32 blocks_y = (chunk.height + 3) / 4;
        const u64 blocks_size = BcBlocksSize(chunk.width, chunk.height, chunk.layers, block_bytes);
        if (kind == AstcStandin::Bc7) {
            encoder->RecordBc7(Cmd(), *scratch->view, chunk.width, chunk.height, chunk.layers,
                               impl.bc7_quality, *scratch->modes,
                               BcBlocksSize(chunk.width, chunk.height, chunk.layers, 8),
                               *scratch->blocks, blocks_size);
        } else {
            encoder->Record(Cmd(), *scratch->view, chunk.width, chunk.height, chunk.layers,
                            BcBlocksOf(kind, s.smart_standin), s.smart_standin && impl.smart_bc.refine,
                            *scratch->blocks, blocks_size);
        }
        const VkMemoryBarrier blocks_written{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        };
        Cmd().PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, blocks_written);
        // Edge blocks: the extent may end inside a block where it reaches the level's edge.
        const VkBufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = blocks_x * 4,
            .bufferImageHeight = blocks_y * 4,
            .imageSubresource{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = chunk.level,
                .baseArrayLayer = chunk.first_layer,
                .layerCount = chunk.layers,
            },
            .imageOffset = {static_cast<s32>(chunk.x), static_cast<s32>(chunk.y), 0},
            .imageExtent = {chunk.width, chunk.height, 1},
        };
        Cmd().CopyBufferToImage(*scratch->blocks, *s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                region);
        if (times) {
            times->uploads[query / 2].bytes += blocks_size;
        }
        recorded = true;
    }
    if (times) {
        Cmd().WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *times->pool, query + 1);
    }
    if (recorded) {
        barrier_pending = true;
        impl.stats.astc_bc_encodes.fetch_add(1, std::memory_order_relaxed);
    }
    return recorded;
}

BcUploadTimes* Recorder::BcTimes() {
    if (!impl.bc_timing) {
        return nullptr;
    }
    if (!flight.bc_times) {
        auto times = std::make_unique<BcUploadTimes>();
        try {
            times->pool = impl.device.GetLogical().CreateQueryPool({
                .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queryType = VK_QUERY_TYPE_TIMESTAMP,
                .queryCount = BcUploadTimes::CAPACITY * 2,
                .pipelineStatistics = 0,
            });
        } catch (const vk::Exception&) {
            impl.bc_timing = false;
            return nullptr;
        }
        if (impl.device.IsHostQueryResetSupported()) {
            impl.device.GetLogical().ResetQueryPool(*times->pool, 0, BcUploadTimes::CAPACITY * 2);
        } else {
            Cmd().ResetQueryPool(*times->pool, 0, BcUploadTimes::CAPACITY * 2);
        }
        times->uploads.reserve(BcUploadTimes::CAPACITY);
        flight.bc_times = std::move(times);
    }
    if (flight.bc_times->uploads.size() >= BcUploadTimes::CAPACITY) {
        impl.bc_window.untimed.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    return flight.bc_times.get();
}

bool Recorder::UploadUntileOnGpu(Storage& s, const UploadPlan& plan, Pool& pool, const u8* data,
                                 u64 size) {
    if (plan.decode != UploadPlan::Decode::None) {
        LogOnce(fmt::format("GPU untile: format {:#x} untiles on the CPU, which decodes its blocks",
                            s.desc.format));
        return false;
    }
    UntileGpu* const untile = impl.Untile();
    if (!untile) {
        return false; // its pipeline is still being built (or failed, logged once)
    }
    const BlockInfo block = FormatBlockInfo(s.desc.format);
    const u64 alignment = (std::max)(untile->OffsetAlignment(), u64{64});
    const auto align = [alignment](u64 value) {
        return (value + alignment - 1) / alignment * alignment;
    };
    // The CPU path's pieces and copy regions, at output offsets the shader can bind.
    UploadPlan linear = plan;
    std::vector<UntileGpu::Piece> pieces;
    pieces.reserve(linear.pieces.size());
    // The shader reads the pool's bytes in place, when the command runs (as the console's GPU
    // reads a texture), where the GPU has them: an imported pool is the guest's own memory, a
    // GPU-only pool's device-local buffer is the GPU's copy of it (its initial bytes uploaded
    // once, later CPU writes pushed in by NotifyHostWriteNow, GPU writes made there). Mirrors,
    // virtual pools and storages whose last GOBs lie past their bytes (the CPU path pads those
    // with zeros) go through staging. (`size` never reaches past the pool: UploadStorage clamps
    // it.)
    bool direct =
        !pool.pending.load(std::memory_order_acquire) &&
        (pool.backing == Pool::Backing::Imported || pool.backing == Pool::Backing::DeviceLocal);
    if (direct && pool.backing == Pool::Backing::DeviceLocal && pool.upload_pending) {
        impl.EnsureBacking(pool); // the guest's initial bytes into the buffer, once
        direct = pool.backing == Pool::Backing::DeviceLocal && !pool.upload_pending;
    }
    u64 output_size = 0;
    u64 input_size = 0;
    for (UploadPiece& piece : linear.pieces) {
        UntileGpu::Piece gpu{
            .input_offset = 0,
            .output_offset = 0,
            .columns = piece.cols,
            .rows = piece.rows,
            .slices = piece.slices,
            .bytes_per_block = block.bytes,
            .tile_height = piece.tile_h,
            .tile_depth = piece.tile_d,
        };
        const u64 needed = UntileGpu::InputSize(gpu);
        if (!UntileGpu::Fits(gpu) || needed > untile->MaxRange() ||
            UntileGpu::OutputSize(gpu) > untile->MaxRange()) {
            LogOnce(fmt::format("GPU untile: a {}x{}x{} level of format {:#x} is beyond the "
                                "shader's 32-bit addressing, untiled on the CPU",
                                piece.width, piece.height, piece.depth, s.desc.format));
            return false;
        }
        output_size = align(output_size);
        piece.staging_offset = output_size;
        gpu.output_offset = output_size;
        output_size += piece.staging_size;
        input_size = align(input_size);
        gpu.input_offset = input_size;
        input_size += needed;
        if (piece.guest_offset + needed > size ||
            (s.pool_offset + piece.guest_offset) % untile->OffsetAlignment() != 0) {
            direct = false;
        }
        pieces.push_back(gpu);
    }
    if (pieces.empty()) {
        return false;
    }
    std::shared_ptr<vk::Buffer> output;
    try {
        // Whole 32-bit words: the shader's u32 view of the last piece's range.
        output = std::make_shared<vk::Buffer>(impl.CreateHostBuffer(
            (output_size + 3) & ~u64{3},
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            MemoryUsage::DeviceLocal));
    } catch (const vk::Exception& exception) {
        LogOnce(fmt::format("GPU untile: no device-local output buffer ({}), untiled on the CPU",
                            exception.what()));
        return false;
    }
    VkBuffer input = VK_NULL_HANDLE;
    if (direct) {
        for (size_t i = 0; i < pieces.size(); ++i) {
            pieces[i].input_offset = s.pool_offset + linear.pieces[i].guest_offset;
        }
        // The shader reads the pool's bytes through host memory: storage shadows over them first
        // (FlushPending below orders their copies before the dispatches).
        if (draw.HasShadows()) {
            draw.FlushShadowsOverlapping(pool, s.pool_offset, size);
        }
        input = pool.Handle();
    } else {
        // The tiled bytes as they are (a copy, not an untile), zero-padded to whole GOBs, in 1 MiB
        // slices over the decode workers: one thread fills fresh upload memory slower than the
        // CPU untile's workers do (OoT's 288 MiB D16 storage: 381 ms against 189 ms).
        constexpr u64 COPY_SLICE = u64{1} << 20;
        struct Slice {
            u8* destination;
            const u8* source;
            u64 copy; ///< bytes from `source`
            u64 zero; ///< zero bytes after them
        };
        const StagingRange staging = AllocStaging(input_size, alignment);
        std::vector<Slice> slices;
        for (size_t i = 0; i < pieces.size(); ++i) {
            const u64 guest_offset = linear.pieces[i].guest_offset;
            const u64 needed = UntileGpu::InputSize(pieces[i]);
            const u64 available = (std::min)(needed, size - guest_offset);
            u8* const destination = staging.pointer + pieces[i].input_offset;
            for (u64 offset = 0; offset < needed; offset += COPY_SLICE) {
                const u64 end = (std::min)(offset + COPY_SLICE, needed);
                const u64 copy = offset < available ? (std::min)(end, available) - offset : 0;
                slices.push_back({destination + offset,
                                  copy != 0 ? data + guest_offset + offset : nullptr, copy,
                                  end - offset - copy});
            }
            pieces[i].input_offset += staging.offset;
        }
        const auto fill = [&slices](size_t index) {
            const Slice& slice = slices[index];
            if (slice.copy != 0) {
                std::memcpy(slice.destination, slice.source, static_cast<size_t>(slice.copy));
            }
            if (slice.zero != 0) {
                std::memset(slice.destination + slice.copy, 0, static_cast<size_t>(slice.zero));
            }
        };
        if (input_size >= COPY_SLICE) {
            RunParallel(slices.size(), fill);
        } else {
            for (size_t i = 0; i < slices.size(); ++i) {
                fill(i);
            }
        }
        input = staging.buffer;
    }
    FlushPending(); // earlier writes of the pool's bytes before the shader reads them
    untile->Record(Cmd(), input, **output, pieces);
    barrier_pending = true; // the shader's writes before the copy reads them
    const bool saved = track_writes;
    track_writes = false;
    PrepareImages({{&s, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
    track_writes = saved;
    Cmd().CopyBufferToImage(**output, *s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            MakeUploadRegions(s, linear, 0));
    flight.keepalive.push_back(std::move(output));
    barrier_pending = true;
    impl.stats.untile_gpu_uploads.fetch_add(1, std::memory_order_relaxed);
    impl.stats.untile_gpu_bytes.fetch_add(linear.staging_size, std::memory_order_relaxed);
    if (direct) {
        impl.stats.untile_gpu_direct.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

void Recorder::VerifyAstcGpu(Storage& s, const UploadPlan& plan, const u8* data, u64 size) {
    FlushAndWait();
    std::vector<u8> expected(static_cast<size_t>(plan.staging_size));
    FillStorageUpload(s, plan, data, size, expected.data(), nullptr);
    vk::Buffer download = impl.CreateHostBuffer(plan.staging_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                MemoryUsage::Download);
    const std::vector<VkBufferImageCopy> regions = MakeUploadRegions(s, plan, 0);
    impl.ImmediateSubmit([&](vk::CommandBuffer cmd) {
        cmd.CopyImageToBuffer(*s.image, VK_IMAGE_LAYOUT_GENERAL, *download, regions);
    });
    download.Invalidate();
    const u8* const actual = download.Mapped().data();
    u64 bad_texels = 0;
    u32 max_diff = 0;
    std::string first;
    for (const UploadPiece& piece : plan.pieces) {
        const u8* a = actual + piece.staging_offset;
        const u8* e = expected.data() + piece.staging_offset;
        const u64 texels = piece.staging_size / 4;
        for (u64 i = 0; i < texels; ++i) {
            u32 diff = 0;
            for (u32 c = 0; c < 4; ++c) {
                const u32 x = a[i * 4 + c];
                const u32 y = e[i * 4 + c];
                diff = (std::max)(diff, x > y ? x - y : y - x);
            }
            if (diff != 0) {
                if (bad_texels == 0) {
                    first = fmt::format(
                        "level {} layer {} texel ({}, {}): gpu {:02x}{:02x}{:02x}{:02x} "
                        "cpu {:02x}{:02x}{:02x}{:02x}",
                        piece.level, piece.layer, i % piece.width, i / piece.width, a[i * 4],
                        a[i * 4 + 1], a[i * 4 + 2], a[i * 4 + 3], e[i * 4], e[i * 4 + 1],
                        e[i * 4 + 2], e[i * 4 + 3]);
                }
                ++bad_texels;
                max_diff = (std::max)(max_diff, diff);
            }
        }
    }
    static Common::BootCounter checked;
    static Common::BootCounter mismatched;
    const u64 n = checked.Next() + 1;
    if (bad_texels != 0) {
        const u64 m = mismatched.Next() + 1;
        if (m <= 40) {
            LOG_WARNING(
                Render_Vulkan,
                "[nvn] astc verify: storage {}x{}x{} lv {} ly {} fmt {:#x}: {} of {} texels "
                "differ, max channel difference {}; {}",
                s.extent.width, s.extent.height, s.extent.depth, s.levels, s.layers, s.desc.format,
                bad_texels, plan.staging_size / 4, max_diff, first);
        }
    }
    if (n % 50 == 0) {
        LOG_INFO(Render_Vulkan, "[nvn] astc verify: {} storages compared, {} differ", n,
                 mismatched.Count());
    }
}

void Recorder::RefreshFromGuest(Storage& s) {
    if (s.samples != VK_SAMPLE_COUNT_1_BIT) {
        return;
    }
    impl.DetectGuestChange(s, flight.seq);
    if (s.force_upload) {
        UploadStorage(s);
    }
}

void Recorder::ResolveOverlaps(Storage& s) {
    const auto index_it = impl.storage_index.find(s.pool_serial);
    if (index_it == impl.storage_index.end()) {
        return;
    }
    Context::Impl::PoolIndex& index = index_it->second;
    if (s.pool_gen_seen == index.write_gen) {
        return;
    }
    const u64 begin = s.pool_offset;
    const u64 end = begin + s.layout.size;
    std::vector<Storage*> sources;
    const u64 scan_from = begin > index.max_size ? begin - index.max_size : 0;
    for (auto it = index.spans.lower_bound(scan_from); it != index.spans.end() && it->first < end;
         ++it) {
        Storage* const other = it->second;
        if (other == &s || (!other->image && !Rescale::NativeIsStale(other->scale_state)) ||
            other->gpu_write_gen <= s.sync_gen || it->first + other->layout.size <= begin) {
            continue;
        }
        sources.push_back(other);
    }
    std::sort(sources.begin(), sources.end(), [](const Storage* a, const Storage* b) {
        return a->gpu_write_gen < b->gpu_write_gen;
    });
    for (Storage* const other : sources) {
        const char* why = nullptr;
        if (s.origin == StorageOrigin::Cpu && s.gpu_write_gen == 0) {
            // `s` holds only bytes uploaded from guest memory. When the CPU rewrote the source's
            // bytes after its GPU write (OoT reuses a GPU-written 260x260 target's memory for a
            // CPU-written vertex heightmap after an area load), the upload is newer than the
            // source's pixels: copying them over it would leave the texture stale for good, as
            // the guest bytes never change again. Unchanged bytes (a view or alias the game
            // places over GPU output, no CPU write since) still take the GPU contents.
            // Two signals of a CPU write after the source's GPU takeover: the page tracker (armed
            // at the takeover; sees any write over the overlapping pages, also of the same bytes)
            // and, for pools without tracking, a changed guest signature.
            bool rewritten = false;
            const char* how = "";
            const bool writeback_settled =
                other->linear_writeback_seq == 0 || impl.SettleLinearWriteback(*other);
            if (writeback_settled && !LegacyFirstUpload()) {
                if (other->gpu_takeover_armed) {
                    Pool* const pool = CpuReadablePoolOf(impl, *other);
                    VC::GuestMemory* const memory = impl.guest.load(std::memory_order_acquire);
                    const u64 from = (std::max)(begin, other->pool_offset);
                    const u64 to = (std::min)(end, other->pool_offset + other->layout.size);
                    if (pool != nullptr && pool->takeover_track != 0 && memory != nullptr &&
                        to > from &&
                        memory->AnyWritten(pool->takeover_track, pool->address + from, to - from)) {
                        rewritten = true;
                        how = " (page tracker)";
                    }
                }
                if (!rewritten && other->gpu_guest_sig_valid) {
                    const std::optional<u64> now = GuestSignatureOf(impl, *other);
                    if (now.has_value() && *now != other->gpu_guest_sig) {
                        rewritten = true;
                        how = " (signature)";
                    }
                }
            }
            static std::atomic<u64> overlap_copies{0};
            const u64 n = overlap_copies.fetch_add(1, std::memory_order_relaxed) + 1;
            if (n <= 16 || n % 1000 == 0) {
                LOG_INFO(Render_Vulkan,
                         "[nvn] overlap #{} into a texture uploaded from guest memory: dst {}x{} "
                         "format {:#x} pool {:#x}+{:#x} size {:#x} <- src {}x{} format {:#x} "
                         "+{:#x} size {:#x} gpu_write_gen {}: {}{}",
                         n, s.extent.width, s.extent.height, s.desc.format, s.pool_serial,
                         s.pool_offset, s.layout.size, other->extent.width, other->extent.height,
                         other->desc.format, other->pool_offset, other->layout.size,
                         other->gpu_write_gen,
                         rewritten ? "skipped, the CPU wrote the bytes after the GPU write"
                         : writeback_settled &&
                                 (other->gpu_takeover_armed || other->gpu_guest_sig_valid)
                             ? "copied, no CPU write since the GPU write"
                             : "copied, no CPU write tracking for the source",
                         how);
            }
            if (rewritten) {
                impl.stats.overlaps_skipped_cpu_rewrite.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
        }
        // The source's guest-sized image is read: it must hold its newest pixels.
        MakeNativeCurrent(*other);
        if (TransferOverlap(*other, s) || TransferOverlapBytes(*other, s, why)) {
            if (s.scaled) {
                s.scale_state = Rescale::AfterNativeWrite(); // overlapping bytes were copied in
            }
            continue;
        }
        // The bytes were written through a texture of another shape and no transfer expresses it.
        const std::pair<u64, u64> pair{reinterpret_cast<u64>(other), reinterpret_cast<u64>(&s)};
        impl.stats.alias_events.fetch_add(1, std::memory_order_relaxed);
        // A reader the GPU wrote before holds contents the alias overwrote (a stale read);
        // otherwise it holds the guest's initial bytes, which the alias overwrote too.
        const bool after_own_write = s.gpu_write_gen != 0;
        (after_own_write ? impl.stats.alias_events_stale : impl.stats.alias_events_initial)
            .fetch_add(1, std::memory_order_relaxed);
        if (impl.alias_logged.insert(pair).second) {
            impl.stats.alias_unsupported.fetch_add(1, std::memory_order_relaxed);
            LogOnce(fmt::format(
                "overlapping textures cannot exchange contents: pool {:#x} +{:#x} ({}x{}, format "
                "{:#x}) was written, +{:#x} ({}x{}, format {:#x}) reads it ({}): {}",
                s.pool_serial, other->pool_offset, other->extent.width, other->extent.height,
                other->desc.format, s.pool_offset, s.extent.width, s.extent.height, s.desc.format,
                after_own_write ? "the reader was written by the GPU before"
                                : "the reader holds the guest's initial bytes",
                why != nullptr ? why : "no reason recorded"));
        }
    }
    s.sync_gen = impl.write_clock;
    s.pool_gen_seen = index.write_gen;
}

bool Recorder::TransferOverlap(Storage& source, Storage& destination) {
    if (!source.image || !destination.image || source.samples != VK_SAMPLE_COUNT_1_BIT ||
        destination.samples != VK_SAMPLE_COUNT_1_BIT ||
        source.pool_offset != destination.pool_offset) {
        return false;
    }
    const bool depth = source.host.depth || source.host.stencil;
    if (depth != (destination.host.depth || destination.host.stencil)) {
        return false;
    }
    const BlockInfo block = FormatBlockInfo(source.desc.format);
    const BlockInfo other_block = FormatBlockInfo(destination.desc.format);
    if (!block.bytes || block.bytes != other_block.bytes || block.width != other_block.width ||
        block.height != other_block.height) {
        return false;
    }
    if (depth ? source.host.format != destination.host.format
              : (source.host.format != destination.host.format &&
                 (source.host.needs_transcode || destination.host.needs_transcode ||
                  block.width != 1 || block.height != 1))) {
        return false;
    }
    const bool linear = IsLinearStorage(source.desc);
    if (linear != IsLinearStorage(destination.desc) ||
        (linear && source.desc.stride != destination.desc.stride)) {
        return false;
    }
    const u32 layers = source.layout.layer_size == destination.layout.layer_size
                           ? (std::min)(source.layers, destination.layers)
                           : 1u;
    const u32 level_limit = linear ? 1u : (std::min)(source.levels, destination.levels);
    const VkImageAspectFlags aspect =
        depth ? (source.aspect & destination.aspect)
              : static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_COLOR_BIT);
    std::vector<VkImageCopy> copies;
    for (u32 level = 0; level < level_limit; ++level) {
        if (!linear) {
            u32 source_h, source_d, destination_h, destination_d;
            TextureLevelTiles(source.desc, level, source_h, source_d);
            TextureLevelTiles(destination.desc, level, destination_h, destination_d);
            if (TextureLevelOffset(source.desc, level) !=
                    TextureLevelOffset(destination.desc, level) ||
                TextureLevelGobs(source.desc, level) != TextureLevelGobs(destination.desc, level) ||
                source_h != destination_h || source_d != destination_d) {
                break; // the levels' bytes are laid out differently from here on
            }
        }
        const LevelExtent from = ExtentOfLevel(source, level);
        const LevelExtent to = ExtentOfLevel(destination, level);
        const u32 width = (std::min)(from.width, to.width);
        const u32 height = (std::min)(from.height, to.height);
        if (source.host.bc_standin &&
            (!BcRegionAligned(0, 0, width, height, from.width, from.height) ||
             !BcRegionAligned(0, 0, width, height, to.width, to.height))) {
            break; // the copy would end inside a 4x4 block of one of the BC stand-ins
        }
        copies.push_back({
            .srcSubresource{
                .aspectMask = aspect, .mipLevel = level, .baseArrayLayer = 0, .layerCount = layers},
            .srcOffset = {0, 0, 0},
            .dstSubresource{
                .aspectMask = aspect, .mipLevel = level, .baseArrayLayer = 0, .layerCount = layers},
            .dstOffset = {0, 0, 0},
            .extent = {(std::min)(from.width, to.width), (std::min)(from.height, to.height),
                       (std::min)(from.depth, to.depth)},
        });
    }
    if (copies.empty()) {
        return false;
    }
    NoteTouched(source.shared_from_this());
    const bool saved = track_writes;
    track_writes = false;
    PrepareImages({{&source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
                   {&destination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
    track_writes = saved;
    Cmd().CopyImage(*source.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *destination.image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copies);
    barrier_pending = true;
    impl.stats.alias_transfers.fetch_add(1, std::memory_order_relaxed);
    return true;
}

namespace {

/// The rows of one (layer, level) of a storage that touch the window of an alias transfer, with
/// where their tight texels go in the scratch buffer.
struct AliasPiece {
    AliasTransferGpu::Piece gpu;
    VkBufferImageCopy region;
};

/// The pieces of `s` that touch [window_begin, window_end) of its pool, each cut down to the block
/// rows whose bytes reach the window (a 2D piece; a 3D one stays whole): the shared bytes of a big
/// texture and a small one cost the small one's rows. `tight` is advanced by their texels (256-byte
/// aligned). False when a piece lies too far from the window for the shader's 32-bit offsets.
bool CollectAliasPieces(const Storage& s, const UploadPlan& plan, u64 window_begin, u64 window_end,
                        u64& tight, std::vector<AliasPiece>& pieces) {
    const BlockInfo block = FormatBlockInfo(s.desc.format);
    for (const UploadPiece& piece : plan.pieces) {
        const u64 begin = s.pool_offset + piece.guest_offset;
        // Bytes of a block row (a row of GOB blocks; linear: one pitch) and the piece's span.
        u64 row_bytes = 0;
        u32 rows_per_step = 1;
        u64 span = 0;
        u32 pitch = 0;
        if (plan.linear) {
            // The pitch the upload reads the guest rows at (nvn_upload.cpp).
            row_bytes = s.layout.layer_size / (std::max)(piece.height, 1u);
            pitch = static_cast<u32>(row_bytes);
            span = row_bytes * piece.rows;
        } else {
            const u64 gobs_in_x = (static_cast<u64>(piece.cols) * block.bytes + 63) / 64;
            row_bytes = gobs_in_x << (9 + piece.tile_h + piece.tile_d);
            rows_per_step = 8u << piece.tile_h;
            span = Tegra::Texture::CalculateSize(true, block.bytes, piece.cols, piece.rows,
                                                 piece.slices, piece.tile_h, piece.tile_d);
        }
        const u64 end = begin + span;
        if (begin >= window_end || end <= window_begin || row_bytes == 0) {
            continue;
        }
        u32 first_row = 0;
        u32 rows = piece.rows;
        if (piece.slices == 1) {
            // Block rows follow each other at row_bytes in the pool; the other block rows keep
            // their bytes.
            const u64 first_step = (begin < window_begin ? window_begin - begin : 0) / row_bytes;
            const u64 end_step = (window_end - begin + row_bytes - 1) / row_bytes;
            first_row = static_cast<u32>((std::min<u64>)(first_step * rows_per_step, piece.rows));
            rows =
                static_cast<u32>((std::min<u64>)(end_step * rows_per_step, piece.rows)) - first_row;
            if (rows == 0) {
                continue;
            }
        }
        const s64 base = static_cast<s64>(begin + first_row / rows_per_step * row_bytes) -
                         static_cast<s64>(window_begin);
        if (base > (std::numeric_limits<s32>::max)() || base < (std::numeric_limits<s32>::min)()) {
            return false;
        }
        tight = (tight + 255) / 256 * 256;
        const u32 first_texel_row = first_row * block.height;
        const u32 texel_rows =
            (std::min)(piece.height, (first_row + rows) * block.height) - first_texel_row;
        pieces.push_back({
            .gpu{
                .tight_offset = static_cast<u32>(tight),
                .piece_base = static_cast<s32>(base),
                .cols = piece.cols,
                .rows = rows,
                .slices = piece.slices,
                .bytes_per_texel = block.bytes,
                .tile_height = piece.tile_h,
                .tile_depth = piece.tile_d,
                .pitch = pitch,
            },
            .region{
                .bufferOffset = tight,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource{
                    .aspectMask = plan.aspect,
                    .mipLevel = piece.level,
                    .baseArrayLayer = piece.layer,
                    .layerCount = 1,
                },
                .imageOffset = {0, static_cast<s32>(first_texel_row), 0},
                .imageExtent = {piece.width, texel_rows, piece.depth},
            },
        });
        tight += static_cast<u64>(piece.cols) * rows * piece.slices * block.bytes;
    }
    return true;
}

} // Anonymous namespace

bool Recorder::TransferOverlapBytes(Storage& source, Storage& destination, const char*& why) {
    if (!source.image || !destination.image) {
        why = "a texture has no image";
        return false;
    }
    if (source.samples != VK_SAMPLE_COUNT_1_BIT || destination.samples != VK_SAMPLE_COUNT_1_BIT) {
        why = "multisampled texture";
        return false;
    }
    AliasTransferGpu* const gpu = impl.AliasGpu();
    if (gpu == nullptr) {
        why = "this device cannot run the alias transfer shader";
        return false;
    }
    const UploadPlan source_plan = PlanStorageUpload(source, source.layout.size);
    const UploadPlan destination_plan = PlanStorageUpload(destination, destination.layout.size);
    for (const UploadPlan* plan : {&source_plan, &destination_plan}) {
        if (plan->unsupported != nullptr) {
            why = plan->unsupported;
            return false;
        }
        if (plan->decode != UploadPlan::Decode::None) {
            why = "a compressed texture decoded to a stand-in format";
            return false;
        }
        if (plan->pieces.empty()) {
            why = "a texture with no level in its storage";
            return false;
        }
        if (std::popcount(static_cast<u32>(plan->aspect)) != 1) {
            why = "a texture with several aspects";
            return false;
        }
    }
    const u64 window_begin = (std::max)(source.pool_offset, destination.pool_offset);
    const u64 window_end = (std::min)(source.pool_offset + source.layout.size,
                                      destination.pool_offset + destination.layout.size);
    if (window_end <= window_begin) {
        return true;
    }
    std::vector<AliasPiece> source_pieces;
    std::vector<AliasPiece> destination_pieces;
    u64 cursor = 0;
    if (!CollectAliasPieces(source, source_plan, window_begin, window_end, cursor, source_pieces) ||
        !CollectAliasPieces(destination, destination_plan, window_begin, window_end, cursor,
                            destination_pieces)) {
        why = "a piece lies too far from the shared bytes";
        return false;
    }
    if (source_pieces.empty() || destination_pieces.empty()) {
        return true; // the shared bytes are padding of one of them
    }
    cursor = (cursor + 255) / 256 * 256;
    const u64 window_offset = cursor;
    const u64 window_size = window_end - window_begin;
    const u64 total = window_offset + window_size;
    if (total > gpu->MaxScratchBytes() || window_size > (std::numeric_limits<u32>::max)()) {
        why = "the shared bytes do not fit the transfer scratch buffer";
        return false;
    }
    const bool reused_scratch = alias_scratch && alias_scratch->size >= total;
    if (!reused_scratch) {
        std::shared_ptr<ShadowChunk> scratch = impl.draw_resources->AcquireShadow(total);
        if (!scratch) {
            why = "no scratch memory for the transfer";
            return false;
        }
        flight.shadow.push_back(scratch);
        alias_scratch = std::move(scratch);
    }

    NoteTouched(source.shared_from_this());
    const bool saved = track_writes;
    track_writes = false;
    PrepareImages({{&source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
                   {&destination, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL}});
    track_writes = saved;
    vk::CommandBuffer cmd = Cmd();
    const auto barrier = [&cmd](VkPipelineStageFlags from, VkAccessFlags written,
                                VkPipelineStageFlags to, VkAccessFlags used) {
        cmd.PipelineBarrier(from, to, 0,
                            VkMemoryBarrier{
                                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                .pNext = nullptr,
                                .srcAccessMask = written,
                                .dstAccessMask = used,
                            });
    };
    if (reused_scratch) {
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    }
    const VkBuffer buffer = *alias_scratch->buffer;
    const auto regions = [](const std::vector<AliasPiece>& pieces) {
        std::vector<VkBufferImageCopy> out;
        out.reserve(pieces.size());
        for (const AliasPiece& piece : pieces) {
            out.push_back(piece.region);
        }
        return out;
    };
    const std::vector<VkBufferImageCopy> destination_regions = regions(destination_pieces);
    cmd.CopyImageToBuffer(*source.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer,
                          regions(source_pieces));
    cmd.CopyImageToBuffer(*destination.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer,
                          destination_regions);
    constexpr VkAccessFlags SHADER_ACCESS = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    const auto to_shader = [&] {
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, SHADER_ACCESS);
    };
    to_shader();
    gpu->Begin(cmd, buffer, total);
    const auto run = [&](AliasTransferGpu::Mode mode, const std::vector<AliasPiece>& pieces) {
        for (const AliasPiece& piece : pieces) {
            gpu->Dispatch(cmd, mode, static_cast<u32>(window_offset), static_cast<u32>(window_size),
                          piece.gpu);
        }
    };
    // The destination's own bytes first (what the source did not write stays), then the source over
    // them, then the window back into the destination's texels.
    run(AliasTransferGpu::Mode::Scatter, destination_pieces);
    to_shader();
    run(AliasTransferGpu::Mode::Scatter, source_pieces);
    to_shader();
    run(AliasTransferGpu::Mode::Gather, destination_pieces);
    barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    track_writes = false;
    PrepareImages({{&destination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
    track_writes = saved;
    cmd = Cmd();
    cmd.CopyBufferToImage(buffer, *destination.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          destination_regions);
    barrier_pending = true;
    impl.stats.alias_byte_transfers.fetch_add(1, std::memory_order_relaxed);
    impl.stats.alias_byte_bytes.fetch_add(window_size, std::memory_order_relaxed);
    return true;
}

void Recorder::ClearTexture(const VC::ClearTexturePayload& p) {
    Storage* storage{};
    std::shared_ptr<Storage> owner;
    u32 level{};
    u32 ref_layer{};
    if (!ResolveRef(p.texture, storage, owner, level, ref_layer, "ClearTexture")) {
        return;
    }
    Storage& s = *storage;
    if (s.host.depth || s.host.stencil) {
        Unsupported("ClearTexture", "depth/stencil textures are cleared by the draw path");
        return;
    }
    if (s.samples != VK_SAMPLE_COUNT_1_BIT) {
        Unsupported("ClearTexture", "multisampled texture");
        return;
    }
    if (s.host.bc_standin) {
        // Vulkan clears no compressed image (EDEN_NX2_NVN_ASTC_STANDIN=rgba8 keeps it clearable).
        Unsupported("ClearTexture", "BC stand-in of an ASTC texture");
        return;
    }
    const LevelExtent extent = ExtentOfLevel(s, level);
    const u32 x = static_cast<u32>((std::max)(p.region.x, 0));
    const u32 y = static_cast<u32>((std::max)(p.region.y, 0));
    if (x >= extent.width || y >= extent.height) {
        Failed("ClearTexture", "region outside the texture");
        return;
    }
    const u32 width = p.region.width > 0
                          ? (std::min)(static_cast<u32>(p.region.width), extent.width - x)
                          : extent.width - x;
    const u32 height = p.region.height > 0
                           ? (std::min)(static_cast<u32>(p.region.height), extent.height - y)
                           : extent.height - y;
    const bool is_3d = s.type == VK_IMAGE_TYPE_3D;
    u32 first_layer = ref_layer;
    u32 layer_count = 1;
    u32 z = 0;
    u32 depth = 1;
    if (is_3d) {
        z = static_cast<u32>((std::max)(p.region.z, 0));
        if (z >= extent.depth) {
            Failed("ClearTexture", "slice outside the texture");
            return;
        }
        depth = p.region.depth > 0 ? (std::min)(static_cast<u32>(p.region.depth), extent.depth - z)
                                   : extent.depth - z;
    } else if (p.layer_count != 0) {
        layer_count = p.layer_count;
    } else {
        first_layer = ref_layer + static_cast<u32>((std::max)(p.region.z, 0));
        layer_count = p.region.depth > 0 ? static_cast<u32>(p.region.depth) : 1u;
    }
    if (first_layer >= s.layers) {
        Failed("ClearTexture", "layer out of range");
        return;
    }
    layer_count = (std::min)(layer_count, s.layers - first_layer);

    const bool full_level = x == 0 && y == 0 && width == extent.width && height == extent.height &&
                            (!is_3d || (z == 0 && depth == extent.depth));
    const bool everything = full_level && s.levels == 1 && layer_count == s.layers;
    EnsureContents(s, everything);
    if (full_level) {
        PrepareImages({{&s, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
        VkClearColorValue color{};
        if (p.is_integer) {
            std::memcpy(color.uint32, p.color, sizeof(color.uint32));
        } else {
            std::memcpy(color.float32, p.color, sizeof(color.float32));
        }
        const VkImageSubresourceRange range{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = level,
            .levelCount = 1,
            .baseArrayLayer = is_3d ? 0u : first_layer,
            .layerCount = is_3d ? 1u : layer_count,
        };
        Cmd().ClearColorImage(*s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, color, range);
        if (everything && s.scaled && impl.EnsureImage(*s.scaled)) {
            // A twin of this texture exists: it takes the same clear instead of being made from
            // the cleared guest-sized image by a blit later.
            Storage& twin = *s.scaled;
            NoteTouched(s.scaled);
            const bool saved = track_writes;
            track_writes = false;
            PrepareImages({{&twin, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
            track_writes = saved;
            Cmd().ClearColorImage(*twin.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, color, range);
            s.scale_state = Rescale::AfterBlit(); // both hold the cleared colour
        }
        barrier_pending = true;
        return;
    }
    if (is_3d) {
        Unsupported("ClearTexture", "partial clear of a 3D texture");
        return;
    }
    ClearTexturePartial(s, level, first_layer, layer_count, x, y, width, height, p);
}

void Recorder::ClearTexturePartial(Storage& s, u32 level, u32 layer, u32 layer_count, u32 x, u32 y,
                                   u32 width, u32 height, const VC::ClearTexturePayload& p) {
    if (!s.host.attachable) {
        Unsupported("ClearTexture", "partial clear of a texture that cannot be a render target");
        return;
    }
    auto pass_it = impl.clear_render_passes.find(s.host.format);
    if (pass_it == impl.clear_render_passes.end()) {
        const VkAttachmentDescription attachment{
            .flags = 0,
            .format = s.host.format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };
        const VkAttachmentReference reference{
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };
        const VkSubpassDescription subpass{
            .flags = 0,
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .inputAttachmentCount = 0,
            .pInputAttachments = nullptr,
            .colorAttachmentCount = 1,
            .pColorAttachments = &reference,
            .pResolveAttachments = nullptr,
            .pDepthStencilAttachment = nullptr,
            .preserveAttachmentCount = 0,
            .pPreserveAttachments = nullptr,
        };
        pass_it =
            impl.clear_render_passes
                .emplace(s.host.format, impl.device.GetLogical().CreateRenderPass({
                                            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                                            .pNext = nullptr,
                                            .flags = 0,
                                            .attachmentCount = 1,
                                            .pAttachments = &attachment,
                                            .subpassCount = 1,
                                            .pSubpasses = &subpass,
                                            .dependencyCount = 0,
                                            .pDependencies = nullptr,
                                        }))
                .first;
    }
    const LevelExtent extent = ExtentOfLevel(s, level);
    auto holder = std::make_shared<ClearHolder>();
    // The view is only an attachment: an sRGB image with storage usage (HostFormat::storage_view)
    // must not pass that usage on to a view of its own format.
    const VkImageViewUsageCreateInfo attachment_usage{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
        .pNext = nullptr,
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    };
    holder->view = impl.device.GetLogical().CreateImageView({
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = &attachment_usage,
        .flags = 0,
        .image = *s.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        .format = s.host.format,
        .components{},
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = level,
            .levelCount = 1,
            .baseArrayLayer = layer,
            .layerCount = layer_count,
        },
    });
    const VkImageView view = *holder->view;
    holder->framebuffer = impl.device.GetLogical().CreateFramebuffer({
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .renderPass = *pass_it->second,
        .attachmentCount = 1,
        .pAttachments = &view,
        .width = extent.width,
        .height = extent.height,
        .layers = layer_count,
    });
    flight.keepalive.push_back(holder);

    PrepareImages({{&s, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}});
    const VkRenderPassBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = nullptr,
        .renderPass = *pass_it->second,
        .framebuffer = *holder->framebuffer,
        .renderArea = {{static_cast<s32>(x), static_cast<s32>(y)}, {width, height}},
        .clearValueCount = 0,
        .pClearValues = nullptr,
    };
    Cmd().BeginRenderPass(begin_info, VK_SUBPASS_CONTENTS_INLINE);
    VkClearAttachment attachment{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .colorAttachment = 0,
        .clearValue = {},
    };
    if (p.is_integer) {
        std::memcpy(attachment.clearValue.color.uint32, p.color,
                    sizeof(attachment.clearValue.color.uint32));
    } else {
        std::memcpy(attachment.clearValue.color.float32, p.color,
                    sizeof(attachment.clearValue.color.float32));
    }
    const VkClearRect rect{
        .rect = {{static_cast<s32>(x), static_cast<s32>(y)}, {width, height}},
        .baseArrayLayer = 0,
        .layerCount = layer_count,
    };
    Cmd().ClearAttachments(attachment, rect);
    Cmd().EndRenderPass();
    barrier_pending = true;
}

void Recorder::ClearBuffer(const VC::ClearBufferPayload& p) {
    if (p.size == 0) {
        return;
    }
    Pool* pool{};
    u64 offset{};
    if (!ResolvePoolRange(p.address, p.size, pool, offset, "ClearBuffer")) {
        return;
    }
    if (offset % 4 != 0 || p.size % 4 != 0) {
        Failed("ClearBuffer",
               fmt::format("address {:#x} / size {:#x} not 4-byte aligned", p.address, p.size));
        return;
    }
    FlushPending();
    Cmd().FillBuffer(pool->Handle(), offset, p.size, p.value);
    NotePoolWrite(*pool, offset, p.size);
    barrier_pending = true;
}

void Recorder::CopyBufferToBuffer(const VC::CopyBufferToBufferPayload& p) {
    if (p.size == 0) {
        return;
    }
    Pool* src{};
    Pool* dst{};
    u64 src_offset{};
    u64 dst_offset{};
    if (!ResolvePoolRange(p.src, p.size, src, src_offset, "CopyBufferToBuffer") ||
        !ResolvePoolRange(p.dst, p.size, dst, dst_offset, "CopyBufferToBuffer")) {
        return;
    }
    if (src == dst && src_offset < dst_offset + p.size && dst_offset < src_offset + p.size) {
        Unsupported("CopyBufferToBuffer", "overlapping source and destination");
        return;
    }
    UploadRange(*src, src_offset, p.size);
    FlushPending();
    const VkBufferCopy region{.srcOffset = src_offset, .dstOffset = dst_offset, .size = p.size};
    Cmd().CopyBuffer(src->Handle(), dst->Handle(), region);
    NotePoolWrite(*dst, dst_offset, p.size);
    barrier_pending = true;
}

void Recorder::CopyBufferToTexture(const VC::CopyBufferTexturePayload& p) {
    Storage* storage{};
    std::shared_ptr<Storage> owner;
    u32 level{};
    u32 layer{};
    if (!ResolveRef(p.texture, storage, owner, level, layer, "CopyBufferToTexture")) {
        return;
    }
    Storage& s = *storage;
    if (s.host.guest_layout_differs) {
        impl.stats.depth_upload_unsupported.fetch_add(1, std::memory_order_relaxed);
        Unsupported("CopyBufferToTexture",
                    "depth/stencil texel layout differs between guest and host");
        return;
    }
    Pool* pool{};
    u64 offset{};
    if (!ResolvePoolRange(p.buffer, 1, pool, offset, "CopyBufferToTexture")) {
        return;
    }
    if (s.host.needs_transcode) {
        DecodeBufferToTexture(s, level, layer, p, *pool, offset);
        return;
    }
    CopyGeometry geometry;
    if (!MakeCopyGeometry(s, level, layer, p.region, p.row_stride, p.image_stride, offset,
                          geometry)) {
        Unsupported("CopyBufferToTexture", "region/stride/alignment cannot be copied");
        return;
    }
    if (offset + geometry.span > pool->size) {
        Failed("CopyBufferToTexture", "source range leaves the memory pool");
        return;
    }
    if (HasLinearWritebackOverlapping(*pool, offset, offset + geometry.span)) {
        EndDrawPass();
        FlushLinearWritebacks(pool, offset, offset + geometry.span);
    }
    const LevelExtent extent = ExtentOfLevel(s, level);
    const bool covers_level =
        p.region.x <= 0 && p.region.y <= 0 &&
        (p.region.width <= 0 || static_cast<u32>(p.region.width) >= extent.width) &&
        (p.region.height <= 0 || static_cast<u32>(p.region.height) >= extent.height);
    EnsureContents(s, covers_level && s.levels == 1 && s.layers == 1 && s.type != VK_IMAGE_TYPE_3D);
    UploadRange(*pool, offset, geometry.span);
    PrepareImages({{&s, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
    Cmd().CopyBufferToImage(pool->Handle(), *s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            geometry.regions);
    barrier_pending = true;
}

bool Recorder::DecodeBufferToTexture(Storage& s, u32 level, u32 layer,
                                     const VC::CopyBufferTexturePayload& p, Pool& pool,
                                     u64 offset) {
    const BlockInfo block = FormatBlockInfo(s.desc.format);
    if (!block.bytes || s.samples != VK_SAMPLE_COUNT_1_BIT) {
        Unsupported("CopyBufferToTexture", "multisampled or layout-less compressed texture");
        return false;
    }
    const LevelExtent extent = ExtentOfLevel(s, level);
    const u32 x = static_cast<u32>((std::max)(p.region.x, 0));
    const u32 y = static_cast<u32>((std::max)(p.region.y, 0));
    if (x >= extent.width || y >= extent.height) {
        Failed("CopyBufferToTexture", "region outside the texture");
        return false;
    }
    if (x % block.width || y % block.height) {
        Unsupported("CopyBufferToTexture", "region of a compressed texture is not block aligned");
        return false;
    }
    const u32 width = p.region.width > 0
                          ? (std::min)(static_cast<u32>(p.region.width), extent.width - x)
                          : extent.width - x;
    const u32 height = p.region.height > 0
                           ? (std::min)(static_cast<u32>(p.region.height), extent.height - y)
                           : extent.height - y;
    if (s.host.bc_standin && !BcRegionAligned(x, y, width, height, extent.width, extent.height)) {
        Unsupported("CopyBufferToTexture",
                    "region of a BC stand-in is not aligned to its 4x4 blocks");
        return false;
    }
    const bool is_3d = s.type == VK_IMAGE_TYPE_3D;
    u32 z = static_cast<u32>((std::max)(p.region.z, 0));
    u32 depth_count = p.region.depth > 0 ? static_cast<u32>(p.region.depth) : 1u;
    u32 first_layer = layer;
    u32 slices = 1;
    if (is_3d) {
        if (z >= extent.depth) {
            Failed("CopyBufferToTexture", "slice outside the texture");
            return false;
        }
        depth_count = (std::min)(depth_count, extent.depth - z);
        slices = depth_count;
    } else {
        first_layer = layer + z;
        if (first_layer >= s.layers) {
            Failed("CopyBufferToTexture", "layer outside the texture");
            return false;
        }
        depth_count = (std::min)(depth_count, s.layers - first_layer);
        slices = depth_count;
        z = 0;
    }
    const u32 cols = CeilDiv(width, block.width);
    const u32 rows = CeilDiv(height, block.height);
    const u64 row_bytes = static_cast<u64>(cols) * block.bytes;
    const u64 row_stride = p.row_stride ? p.row_stride : row_bytes;
    const u64 image_stride = p.image_stride ? p.image_stride : row_stride * rows;
    if (row_stride < row_bytes || (slices > 1 && image_stride < row_stride * rows)) {
        Unsupported("CopyBufferToTexture", "strides smaller than the region");
        return false;
    }
    const u64 span = (slices - 1) * image_stride + (rows - 1) * row_stride + row_bytes;
    if (offset + span > pool.size) {
        Failed("CopyBufferToTexture", "source range leaves the memory pool");
        return false;
    }
    if (HasLinearWritebackOverlapping(pool, offset, offset + span)) {
        EndDrawPass();
        FlushLinearWritebacks(&pool, offset, offset + span);
    }
    {
        // The CPU reads the blocks: bytes a submission still in flight writes would be stale.
        std::scoped_lock lock{pool.dirty_mutex};
        for (const Pool::Dirty& dirty : pool.gpu_dirty) {
            if (dirty.begin < offset + span && dirty.end > offset) {
                Unsupported("CopyBufferToTexture",
                            "the blocks were written by the GPU: they cannot be decoded on the "
                            "CPU");
                return false;
            }
        }
    }
    std::vector<u8> scratch;
    const u8* data = impl.GuestBytes(pool, offset, span, scratch);
    if (!data) {
        Failed("CopyBufferToTexture", "source bytes are not readable");
        return false;
    }
    std::vector<u8> blocks(static_cast<size_t>(row_bytes * rows * slices));
    for (u32 slice = 0; slice < slices; ++slice) {
        for (u32 row = 0; row < rows; ++row) {
            std::memcpy(blocks.data() + (static_cast<u64>(slice) * rows + row) * row_bytes,
                        data + slice * image_stride + row * row_stride,
                        static_cast<size_t>(row_bytes));
        }
    }
    const u64 out_size = static_cast<u64>(width) * height * slices * DecodedTexelBytes(s.host);
    const StagingRange staging = AllocStaging(out_size, 64);
    const auto begin = std::chrono::steady_clock::now();
    DecodeBlocks(s.host, block, blocks, width, height, slices,
                 std::span<u8>{staging.pointer, static_cast<size_t>(out_size)});
    impl.stats.decode_ns.fetch_add(
        static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - begin)
                             .count()),
        std::memory_order_relaxed);
    impl.stats.decode_bytes.fetch_add(out_size, std::memory_order_relaxed);
    const bool covers = x == 0 && y == 0 && width == extent.width && height == extent.height &&
                        s.levels == 1 && s.layers == 1 && !is_3d;
    if (s.host.bc_standin && !PinStandin(s, "CopyBufferToTexture")) {
        Failed("CopyBufferToTexture", "the BC stand-in could not be recreated as BC3");
        return false;
    }
    EnsureContents(s, covers);
    if (s.host.bc_standin) {
        // 2D only (ToBcStandin): the slices are layers, decoded tightly one after the other.
        const u64 layer_bytes = u64{width} * height * 4;
        const std::vector<BcChunk> chunks =
            SplitBcLayers(level, first_layer, slices, width, height, x, y);
        const bool recorded =
            RecordBcUpload(s, chunks, false, [&](const BcChunk& chunk, BcScratch& scratch) {
                const VkBufferImageCopy region{
                    .bufferOffset =
                        staging.offset + u64{chunk.first_layer - first_layer} * layer_bytes,
                    .bufferRowLength = 0,
                    .bufferImageHeight = 0,
                    .imageSubresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, chunk.layers},
                    .imageOffset = {0, 0, 0},
                    .imageExtent = {width, height, 1},
                };
                Cmd().CopyBufferToImage(staging.buffer, *scratch.image, VK_IMAGE_LAYOUT_GENERAL,
                                        region);
            });
        if (!recorded) {
            Failed("CopyBufferToTexture", "the BC stand-in could not be encoded");
        }
        return recorded;
    }
    PrepareImages({{&s, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
    const VkBufferImageCopy region{
        .bufferOffset = staging.offset,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource{
            .aspectMask = TransferAspect(s),
            .mipLevel = level,
            .baseArrayLayer = first_layer,
            .layerCount = is_3d ? 1u : slices,
        },
        .imageOffset = {static_cast<s32>(x), static_cast<s32>(y), static_cast<s32>(is_3d ? z : 0)},
        .imageExtent = {width, height, is_3d ? slices : 1u},
    };
    Cmd().CopyBufferToImage(staging.buffer, *s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);
    barrier_pending = true;
    return true;
}

void Recorder::CopyTextureToBuffer(const VC::CopyBufferTexturePayload& p) {
    Storage* storage{};
    std::shared_ptr<Storage> owner;
    u32 level{};
    u32 layer{};
    if (!ResolveRef(p.texture, storage, owner, level, layer, "CopyTextureToBuffer")) {
        return;
    }
    Storage& s = *storage;
    if (s.host.needs_transcode || s.host.guest_layout_differs) {
        Unsupported("CopyTextureToBuffer",
                    s.host.needs_transcode
                        ? "the texture holds decoded texels of a compressed format; they cannot "
                          "be encoded back"
                        : "depth/stencil texel layout differs between guest and host");
        return;
    }
    Pool* pool{};
    u64 offset{};
    if (!ResolvePoolRange(p.buffer, 1, pool, offset, "CopyTextureToBuffer")) {
        return;
    }
    CopyGeometry geometry;
    if (!MakeCopyGeometry(s, level, layer, p.region, p.row_stride, p.image_stride, offset,
                          geometry)) {
        Unsupported("CopyTextureToBuffer", "region/stride/alignment cannot be copied");
        return;
    }
    if (offset + geometry.span > pool->size) {
        Failed("CopyTextureToBuffer", "destination range leaves the memory pool");
        return;
    }
    if (HasLinearWritebackOverlapping(*pool, offset, offset + geometry.span)) {
        EndDrawPass();
        FlushLinearWritebacks(pool, offset, offset + geometry.span);
    }
    EnsureContents(s, false);
    if (geometry.padded) {
        // Bytes between rows are not written: the mirror must hold the guest's version of them.
        UploadRange(*pool, offset, geometry.span);
    }
    PrepareImages({{&s, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL}});
    Cmd().CopyImageToBuffer(*s.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pool->Handle(),
                            geometry.regions);
    NotePoolWrite(*pool, offset, geometry.span);
    barrier_pending = true;
}

void Recorder::CopyTextureToTexture(const VC::CopyTextureToTexturePayload& p) {
    Storage* src{};
    Storage* dst{};
    std::shared_ptr<Storage> src_owner;
    std::shared_ptr<Storage> dst_owner;
    u32 src_level{};
    u32 src_layer{};
    u32 dst_level{};
    u32 dst_layer{};
    if (!ResolveRef(p.src, src, src_owner, src_level, src_layer, "CopyTextureToTexture") ||
        !ResolveRef(p.dst, dst, dst_owner, dst_level, dst_layer, "CopyTextureToTexture")) {
        return;
    }
    // Smart stand-ins of two classes hold different BC formats: copies are between BC3 ones.
    if ((src->host.bc_standin && !PinStandin(*src, "CopyTextureToTexture")) ||
        (dst->host.bc_standin && !PinStandin(*dst, "CopyTextureToTexture"))) {
        Failed("CopyTextureToTexture", "a BC stand-in could not be recreated as BC3");
        return;
    }
    // MKW can stream between the linear and sRGB versions of the atlas format. Copy the
    // decoded texel bits without a colour conversion, as NVN does for the ASTC blocks.
    const bool mkw_atlas_copy = impl.mario_kart_world && src->host.needs_transcode &&
                                dst->host.needs_transcode && IsAstc8x8(src->host.pixel_format) &&
                                IsAstc8x8(dst->host.pixel_format) &&
                                ((src->host.bc_standin && dst->host.bc_standin &&
                                  ViewsBcStandin(src->host.format, dst->host.format)) ||
                                 SrgbStorageViewFormat(src->host.format) == dst->host.format ||
                                 SrgbStorageViewFormat(dst->host.format) == src->host.format);
    if (src->host.needs_transcode != dst->host.needs_transcode ||
        (src->host.needs_transcode && src->host.format != dst->host.format && !mkw_atlas_copy)) {
        // Both sides decoded to the same stand-in copy texel for texel; a decoded texture and a
        // native one hold different bytes per texel (as do an RGBA8 and a BC stand-in).
        Unsupported("CopyTextureToTexture",
                    src->host.needs_transcode && dst->host.needs_transcode
                        ? "the stand-ins of the two compressed formats differ"
                        : "only one side is a decoded stand-in of a compressed format");
        return;
    }
    const BlockInfo src_block = FormatBlockInfo(src->desc.format);
    const BlockInfo dst_block = FormatBlockInfo(dst->desc.format);
    if (src_block.bytes != dst_block.bytes || src_block.width != dst_block.width ||
        src_block.height != dst_block.height) {
        Unsupported("CopyTextureToTexture", "source and destination formats are not compatible");
        return;
    }
    const bool resolve =
        src->samples != VK_SAMPLE_COUNT_1_BIT && dst->samples == VK_SAMPLE_COUNT_1_BIT;
    if (src->samples != dst->samples && !resolve) {
        Unsupported("CopyTextureToTexture", "sample counts differ");
        return;
    }
    if (resolve && (src->host.depth || src->host.stencil || dst->host.depth || dst->host.stencil)) {
        // vkCmdResolveImage resolves color only. A depth/stencil downsample needs a shader pass.
        Unsupported("Downsample", "depth/stencil multisample resolve");
        return;
    }
    if (resolve && src->host.format != dst->host.format) {
        Unsupported("Downsample", "resolve between different host formats");
        return;
    }
    const LevelExtent src_extent = ExtentOfLevel(*src, src_level);
    const LevelExtent dst_extent = ExtentOfLevel(*dst, dst_level);
    const auto clamp_region = [](const VC::CopyRegion& r, const LevelExtent& e, u32& x, u32& y,
                                 u32& z, u32& w, u32& h) {
        x = static_cast<u32>((std::max)(r.x, 0));
        y = static_cast<u32>((std::max)(r.y, 0));
        z = static_cast<u32>((std::max)(r.z, 0));
        if (x >= e.width || y >= e.height) {
            return false;
        }
        w = r.width > 0 ? (std::min)(static_cast<u32>(r.width), e.width - x) : e.width - x;
        h = r.height > 0 ? (std::min)(static_cast<u32>(r.height), e.height - y) : e.height - y;
        return true;
    };
    u32 sx, sy, sz, sw, sh, dx, dy, dz, dw, dh;
    if (!clamp_region(p.src_region, src_extent, sx, sy, sz, sw, sh) ||
        !clamp_region(p.dst_region, dst_extent, dx, dy, dz, dw, dh)) {
        Failed("CopyTextureToTexture", "region outside the texture");
        return;
    }
    const u32 width = (std::min)(sw, dst_extent.width - dx);
    const u32 height = (std::min)(sh, dst_extent.height - dy);
    const bool partial_bc_copy =
        src->host.bc_standin &&
        (!BcRegionAligned(sx, sy, width, height, src_extent.width, src_extent.height) ||
         !BcRegionAligned(dx, dy, width, height, dst_extent.width, dst_extent.height));
    const u32 count = p.src_region.depth > 0 ? static_cast<u32>(p.src_region.depth) : 1u;
    const bool src_3d = src->type == VK_IMAGE_TYPE_3D;
    const bool dst_3d = dst->type == VK_IMAGE_TYPE_3D;
    if (src_3d != dst_3d) {
        Unsupported("CopyTextureToTexture", "copy between 3D and layered textures");
        return;
    }
    const u32 src_first = src_3d ? src_layer : src_layer + sz;
    const u32 dst_first = dst_3d ? dst_layer : dst_layer + dz;
    const u32 layers =
        src_3d ? 1u
               : (std::min)({count, src->layers - (std::min)(src_first, src->layers - 1),
                             dst->layers - (std::min)(dst_first, dst->layers - 1)});
    const u32 depth =
        src_3d ? (std::min)({count, src_extent.depth - (std::min)(sz, src_extent.depth - 1),
                             dst_extent.depth - (std::min)(dz, dst_extent.depth - 1)})
               : 1u;

    if (partial_bc_copy && !src_3d && !dst_3d) {
        if (impl.logged_messages.insert("partial-bc-copy").second) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] partial BC copy: {}x{} texels to mip {} layer {}; preserving block "
                     "neighbors",
                     width, height, dst_level, dst_first);
        }
        EnsureContents(*src, false);
        EnsureContents(*dst, false);
        BcGpuEncoder* encoder = impl.BcEncoder();
        if (!encoder || impl.device.MaxPushDescriptors() < 3) {
            Failed("CopyTextureToTexture", "partial BC copy needs three push descriptors");
            return;
        }
        const u32 bx = dx / 4 * 4;
        const u32 by = dy / 4 * 4;
        const u32 bw = (std::min)((dx + width + 3) / 4 * 4, dst_extent.width) - bx;
        const u32 bh = (std::min)((dy + height + 3) / 4 * 4, dst_extent.height) - by;
        const auto chunks = SplitBcLayers(dst_level, dst_first, layers, bw, bh, bx, by);
        const bool recorded =
            RecordBcUpload(*dst, chunks, false, [&](const BcChunk& chunk, BcScratch& scratch) {
                const bool saved = track_writes;
                track_writes = false;
                PrepareImages({{src, VK_IMAGE_LAYOUT_GENERAL}, {dst, VK_IMAGE_LAYOUT_GENERAL}});
                track_writes = saved;
                auto views = std::make_shared<std::vector<vk::ImageView>>();
                const auto view = [&](Storage& storage, u32 level, u32 first) {
                    views->push_back(impl.device.GetLogical().CreateImageView({
                        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                        .image = *storage.image,
                        .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                        .format = storage.host.format == VK_FORMAT_BC3_SRGB_BLOCK
                                      ? VK_FORMAT_BC3_UNORM_BLOCK
                                      : storage.host.format,
                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, 1, first,
                                             chunk.layers},
                    }));
                    return **views->rbegin();
                };
                const VkImageView from =
                    view(*src, src_level, src_first + chunk.first_layer - dst_first);
                const VkImageView to = view(*dst, dst_level, chunk.first_layer);
                encoder->RecordCopy(Cmd(), from, to, *scratch.view,
                                    {static_cast<s32>(sx), static_cast<s32>(sy)},
                                    {static_cast<s32>(dx), static_cast<s32>(dy)},
                                    {static_cast<s32>(bx), static_cast<s32>(by)},
                                    {static_cast<s32>(width), static_cast<s32>(height)},
                                    {static_cast<s32>(bw), static_cast<s32>(bh)}, chunk.layers);
                flight.keepalive.push_back(std::move(views));
                PrepareImages({{dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
            });
        if (!recorded) {
            Failed("CopyTextureToTexture", "partial BC copy could not be encoded");
        }
        return;
    }
    if (partial_bc_copy) {
        Unsupported("CopyTextureToTexture", "partial BC copy of a 3D texture");
        return;
    }

    const bool same = src == dst;
    const bool dst_covered = !same && !dst_3d && dx == 0 && dy == 0 && width == dst_extent.width &&
                             height == dst_extent.height && dst->levels == 1 &&
                             layers == dst->layers && dst->samples == VK_SAMPLE_COUNT_1_BIT;
    if (!resolve && !same && !src_3d && src_level == 0 && dst_level == 0 &&
        CopyTextureScaled(*src, *dst, src_first, dst_first, layers,
                          {static_cast<s32>(sx), static_cast<s32>(sy), static_cast<s32>(dx),
                           static_cast<s32>(dy), width, height},
                          dst_covered)) {
        return; // both images are scaled: the copy ran on their twins
    }
    EnsureContents(*src, false);
    EnsureContents(*dst, dst_covered);
    if (same) {
        PrepareImages({{src, VK_IMAGE_LAYOUT_GENERAL}});
    } else {
        PrepareImages({{src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
                       {dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}});
    }
    const VkImageLayout src_layout =
        same ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    const VkImageLayout dst_layout =
        same ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    const VkImageSubresourceLayers src_sub{
        .aspectMask = CopyAspect(*src),
        .mipLevel = src_level,
        .baseArrayLayer = src_3d ? 0u : src_first,
        .layerCount = layers,
    };
    const VkImageSubresourceLayers dst_sub{
        .aspectMask = CopyAspect(*dst),
        .mipLevel = dst_level,
        .baseArrayLayer = dst_3d ? 0u : dst_first,
        .layerCount = layers,
    };
    if (resolve) {
        const VkImageResolve region{
            .srcSubresource = src_sub,
            .srcOffset = {static_cast<s32>(sx), static_cast<s32>(sy),
                          static_cast<s32>(src_3d ? sz : 0)},
            .dstSubresource = dst_sub,
            .dstOffset = {static_cast<s32>(dx), static_cast<s32>(dy),
                          static_cast<s32>(dst_3d ? dz : 0)},
            .extent = {width, height, depth},
        };
        Cmd().ResolveImage(*src->image, src_layout, *dst->image, dst_layout, region);
    } else {
        const VkImageCopy region{
            .srcSubresource = src_sub,
            .srcOffset = {static_cast<s32>(sx), static_cast<s32>(sy),
                          static_cast<s32>(src_3d ? sz : 0)},
            .dstSubresource = dst_sub,
            .dstOffset = {static_cast<s32>(dx), static_cast<s32>(dy),
                          static_cast<s32>(dst_3d ? dz : 0)},
            .extent = {width, height, depth},
        };
        Cmd().CopyImage(*src->image, src_layout, *dst->image, dst_layout, region);
    }
    barrier_pending = true;
}

void Recorder::FenceSync(const VC::FenceSyncPayload& p) {
    impl.NoteUse(p.sync_serial, flight.seq);
    impl.syncs[p.sync_serial] =
        SyncPoint{*queue.timeline, segments.back().signal_value, queue_index};
    segment_signaled = true;
}

void Recorder::WaitSync(const VC::WaitSyncPayload& p) {
    impl.NoteUse(p.sync_serial, flight.seq);
    const auto it = impl.syncs.find(p.sync_serial);
    if (it == impl.syncs.end()) {
        Failed("WaitSync", fmt::format("sync {:#x} was never fenced", p.sync_serial));
        return;
    }
    if (it->second.queue == queue_index) {
        return; // one NVN queue is one ordered stream
    }
    const bool flush = segment_signaled;
    EndSegment(flush);
    pending_waits.emplace_back(it->second.semaphore, it->second.value);
    BeginSegment(!flush);
}

void Recorder::SignalEvent(const VC::SignalEventPayload& p) {
    // The guest word is written by the completion action of the recorder, when the GPU is done; the
    // model has the value as of this record, which is what later waits of the replay are checked
    // against. The signal executes with the segment that contains it.
    impl.events.Signal(p.event_address, p.mode, p.value,
                       VC::SignalPoint{queue_index, segments.back().signal_value});
    segment_signaled = true;
    impl.stats.event_signals.fetch_add(1, std::memory_order_relaxed);
    if (Debug::Timeline()) {
        Debug::TimelineEvent("signal", flight.seq,
                             fmt::format("q={} addr={:#x} mode={} value={} tv={}", queue_index,
                                         p.event_address, p.mode, p.value,
                                         segments.back().signal_value));
    }
}

bool Recorder::WaitEvent(const VC::WaitEventPayload& p, bool& release) {
    event_deps.clear();
    const VC::EventModel::WaitCheck check =
        impl.events.Wait(p.event_address, p.mode, p.value, event_deps);
    const auto note = [&](const char* what) {
        std::string deps;
        for (const VC::SignalPoint& dep : event_deps) {
            deps += fmt::format("{}{}:{}", deps.empty() ? "" : ",", dep.queue, dep.value);
        }
        Debug::TimelineEvent(what, flight.seq,
                             fmt::format("q={} addr={:#x} mode={} want={} cur={} deps={}",
                                         queue_index, p.event_address, p.mode, p.value,
                                         check.current, deps.empty() ? "-" : deps));
    };
    if (!check.satisfied) {
        if (!release) {
            if (Debug::Timeline() && !wait_block_noted) {
                wait_block_noted = true;
                note("wait_blocked");
            }
            return false;
        }
        release = false;
        wait_block_noted = false;
        impl.stats.waits_unsatisfied.fetch_add(1, std::memory_order_relaxed);
        if (impl.unsatisfied_logged.emplace(p.event_address, p.mode, p.value).second) {
            LOG_ERROR(
                Render_Vulkan,
                "[nvn] queue {} submission {} waited {} ms for event {:#x} to satisfy {} {:#x}, "
                "but its word is {:#x} and nothing signalled it; passing the wait (later "
                "releases of this wait are only counted in waits_unsatisfied)",
                queue_index, flight.seq, impl.options.parked_timeout.count(), p.event_address,
                p.mode == VC::EVENT_WAIT_EQUAL         ? "=="
                : p.mode == VC::EVENT_WAIT_GEQUAL_WRAP ? ">= (wrapping)"
                                                       : "(unknown mode)",
                p.value, check.current);
        }
        return true;
    }
    impl.stats.event_waits.fetch_add(1, std::memory_order_relaxed);
    if (Debug::Timeline()) {
        note(wait_block_noted ? "wait_resumed" : "wait_ok");
        wait_block_noted = false;
    }
    // The signals the word was built from execute on the GPU: wait for them. One segment break
    // covers all of them; signals of this queue need nothing (one NVN queue is one ordered stream).
    const bool foreign = std::ranges::any_of(
        event_deps, [this](const VC::SignalPoint& dep) { return dep.queue != queue_index; });
    if (!foreign) {
        return true;
    }
    const bool flush = segment_signaled;
    EndSegment(flush);
    for (const VC::SignalPoint& dep : event_deps) {
        if (dep.queue == queue_index) {
            continue;
        }
        const VkSemaphore semaphore = *impl.GetQueue(dep.queue).timeline;
        const auto known = std::ranges::find_if(
            pending_waits, [semaphore](const auto& wait) { return wait.first == semaphore; });
        if (known != pending_waits.end()) {
            known->second = (std::max)(known->second, dep.value);
        } else {
            pending_waits.emplace_back(semaphore, dep.value);
        }
    }
    BeginSegment(!flush);
    return true;
}

void Recorder::Suspend() {
    if (records_replayed == records_at_piece) {
        return;
    }
    SubmitPiece();
    if (Debug::Timeline()) {
        Debug::TimelineEvent("event_wait", flight.seq);
    }
}

CounterFlight& Recorder::Counters() {
    if (!flight.counters) {
        flight.counters = std::make_unique<CounterFlight>();
    }
    return *flight.counters;
}

// The draw path (merged from NvnGraphics) is the caller; until it lands nothing references this.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 5245)
#endif
bool Recorder::BeginRenderCondition() {
    if (!queue.occlusion_counting) {
        ++queue.uncounted_draws;
    }
    if (render_state == RenderState::Enabled && !queue.occlusion_counting) {
        return true;
    }
    CounterEngine* const engine = impl.counters_engine.get();
    if (!engine) {
        return true;
    }
    if (render_state == RenderState::Disabled) {
        engine->Stats().ops_skipped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (queue.occlusion_counting && !occlusion && !engine->InPassQueries()) {
        // The draw about to be recorded counts into the query: begun here, outside any render pass
        // (a pass cannot be open: the segment is new, or a counter op ended it). In-pass queries
        // begin inside the pass instead (HostBeforeDraw).
        occlusion = engine->BeginOcclusion(Cmd(), Counters());
    }
    if (render_state == RenderState::Conditional) {
        engine->BeginConditional(Cmd(), condition_slot);
        condition_open = true;
        engine->Stats().ops_gated.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

void Recorder::EndRenderCondition() {
    if (condition_open) {
        impl.counters_engine->EndConditional(Cmd());
        condition_open = false;
    }
}

void Recorder::FoldOcclusion() {
    if (!occlusion) {
        return;
    }
    impl.counters_engine->FoldOcclusion(Cmd(), Counters(), *occlusion, queue_index, flight.seq);
    occlusion.reset();
    barrier_pending = true;
}

void Recorder::ResolveCounterBatch() {
    if (counter_batch.Empty()) {
        return;
    }
    impl.counters_engine->ResolveBatch(Cmd(), Counters(), counter_batch, queue_index, flight.seq);
    barrier_pending = true;
}

void Recorder::ReserveCounterEvents(BreakCause cause) {
    if (counter_batch.Room(3)) {
        return;
    }
    // The pass end writes a batch this full; outside a pass it is written right here.
    EndDrawPass(cause);
    ResolveCounterBatch();
}

bool Recorder::CounterOpKeepsPass(VC::Op op, const void* payload, size_t payload_size) {
    if (op == VC::Op::ReportCounter && payload_size >= sizeof(VC::ReportCounterPayload)) {
        const auto& report = *static_cast<const VC::ReportCounterPayload*>(payload);
        const CounterEngine* const engine = impl.Counters();
        if (!engine) {
            return false;
        }
        return IsTimestampReport(report)
                   ? engine->TimestampInPass()
                   : report.counter_type == CounterType::SamplesPassed && engine->InPassQueries();
    }
    if (op == VC::Op::ResetCounter && payload_size >= sizeof(VC::ResetCounterPayload)) {
        // Resets of the other counter types record nothing.
        const CounterEngine* const engine = impl.Counters();
        return engine && (engine->InPassQueries() ||
                          static_cast<const VC::ResetCounterPayload*>(payload)->counter_type !=
                              CounterType::SamplesPassed);
    }
    return false;
}

void Recorder::ReportValue(const VC::ReportValuePayload& p) {
    Pool* pool{};
    u64 offset{};
    if (!ResolvePoolRange(p.address, sizeof(u32), pool, offset, "ReportValue")) {
        return;
    }
    CounterEngine* const engine = impl.Counters();
    if (offset % 4 != 0) {
        Failed("ReportValue", fmt::format("address {:#x} is not 4-byte aligned", p.address));
        return;
    }
    // The value is also written by the recorder once the submission completed; writing it on the
    // GPU as well keeps it ordered for conditions and shaders that read it in this submission.
    EndDrawPass(BreakCause::ReportValue);
    FlushPending();
    Cmd().FillBuffer(pool->Handle(), offset, sizeof(u32), p.value);
    NotePoolWrite(*pool, offset, sizeof(u32));
    barrier_pending = true;
    if (engine) {
        engine->Stats().reports_value.fetch_add(1, std::memory_order_relaxed);
    }
}

void Recorder::ReportCounter(const VC::ReportCounterPayload& p) {
    Pool* pool{};
    u64 offset{};
    if (!ResolvePoolRange(p.address, 2 * sizeof(u64), pool, offset, "ReportCounter")) {
        return;
    }
    if (offset % 4 != 0) {
        Failed("ReportCounter", fmt::format("address {:#x} is not 4-byte aligned", p.address));
        return;
    }
    CounterEngine* const engine = impl.Counters();
    if (!engine) {
        Unsupported("ReportCounter", "the counter engine could not be created");
        return;
    }
    if (p.counter_type == CounterType::SamplesPassed && Debug::Get().draw_log != 0 &&
        Debug::FrameSelected(impl.presents_recorded + 1)) {
        LOG_INFO(Render_Vulkan, "[nvn] draw-marker f{} report samples address={:#x} counting={}",
                 impl.presents_recorded + 1, p.address, queue.occlusion_counting);
    }
    const bool timestamp = IsTimestampReport(p);
    const bool in_pass = p.counter_type == CounterType::SamplesPassed && engine->InPassQueries();
    const VkBuffer destination = pool->Handle();
    if (in_pass) {
        // An earlier timestamp report's zero-fill of these bytes is still deferred (recorded at
        // the next EndDrawPass, after the batch): recorded first, so this report wins.
        const bool zero_pending =
            std::ranges::any_of(deferred_zero, [&](const std::pair<VkBuffer, u64>& zero) {
                return zero.first == destination && zero.second < offset + sizeof(u64) &&
                       offset < zero.second + sizeof(u64);
            });
        if (zero_pending) {
            EndDrawPass(BreakCause::ReportCounter);
        }
        ReserveCounterEvents(BreakCause::ReportCounter);
    }
    if (!(timestamp && engine->TimestampInPass()) && !in_pass) {
        EndDrawPass(BreakCause::ReportCounter);
    }
    CounterFlight& counters = Counters();
    switch (p.counter_type) {
    case CounterType::SamplesPassed:
        // Cumulative since the last ResetCounter.
        if (in_pass) {
            // The open query (begun in this pass) ends here; the value is written when the pass
            // ends, right away when none is open.
            if (occlusion) {
                engine->EndOcclusion(Cmd(), counters, *occlusion);
                counter_batch.AddQuery(*occlusion);
                occlusion.reset();
            }
            counter_batch.AddReport(destination, offset, pool);
            if (!draw.PassOpen()) {
                ResolveCounterBatch();
            }
        } else {
            // Fold the open query, then copy the total.
            FoldOcclusion();
            engine->CopyAccumulator(Cmd(), queue_index, destination, offset);
        }
        engine->Stats().reports_samples.fetch_add(1, std::memory_order_relaxed);
        if (queue.uncounted_draws != 0) {
            engine->Stats().reports_samples_uncounted.fetch_add(1, std::memory_order_relaxed);
            LogOnce("DIAG SAMPLES_PASSED report after draws that were not counted (no report was "
                    "visible after the reset when they were recorded)");
        }
        // Draws after the last report of a reset period count for nobody: stop paying for them.
        queue.occlusion_counting = SamplesReportFollows();
        break;
    case CounterType::Timestamp:
    case CounterType::TimestampTop:
        // A timestamp report has no count: the counter field is 0, the time goes in the second
        // field (filled in when the submission completed).
        if (draw.PassOpen()) {
            deferred_zero.emplace_back(destination, offset);
        } else {
            FlushPending();
            Cmd().FillBuffer(destination, offset, sizeof(u64), 0);
        }
        engine->Stats().reports_timestamp.fetch_add(1, std::memory_order_relaxed);
        break;
    default:
        FlushPending();
        Cmd().FillBuffer(destination, offset, sizeof(u64), 0);
        engine->Stats().reports_unsupported_type.fetch_add(1, std::memory_order_relaxed);
        LogOnce(fmt::format("counter type {:#x} is not counted by the Vulkan replay; its reports "
                            "carry 0",
                            p.counter_type));
        break;
    }
    NotePoolWrite(*pool, offset, sizeof(u64));
    TimestampReport report;
    report.address = p.address;
    report.host = pool->host ? pool->host + offset : nullptr;
    report.owner = impl.pools.at(pool->serial);
    if (p.counter_type == CounterType::SamplesPassed && VideoCommon::Nvn::TimestampTraceEnabled()) {
        counters.traced_samples.push_back(report);
    }
    engine->WriteTimestamp(Cmd(), counters, p.counter_type == CounterType::TimestampTop,
                           std::move(report));
    barrier_pending = true;
}

void Recorder::FlushDeferredReports() {
    if (deferred_zero.empty() || draw.PassOpen()) {
        return;
    }
    FlushPending();
    for (const auto& [buffer, offset] : deferred_zero) {
        Cmd().FillBuffer(buffer, offset, sizeof(u64), 0);
    }
    deferred_zero.clear();
    barrier_pending = true;
}

bool Recorder::SamplesReportFollows() const {
    const auto scan = [](std::span<const u64> list, size_t from, bool& decided) {
        size_t index = from;
        while (index < list.size()) {
            const auto* header = reinterpret_cast<const VC::RecordHeader*>(&list[index]);
            if (header->op == static_cast<u16>(VC::Op::End) ||
                header->size < sizeof(VC::RecordHeader) || header->size % 8 != 0 ||
                index + header->size / 8 > list.size()) {
                break;
            }
            const auto* payload = reinterpret_cast<const u32*>(header + 1);
            if (header->op == static_cast<u16>(VC::Op::ReportCounter) &&
                header->size >= sizeof(VC::RecordHeader) + sizeof(VC::ReportCounterPayload) &&
                reinterpret_cast<const VC::ReportCounterPayload*>(header + 1)->counter_type ==
                    CounterType::SamplesPassed) {
                decided = true;
                return true;
            }
            if (header->op == static_cast<u16>(VC::Op::ResetCounter) &&
                payload[0] == CounterType::SamplesPassed) {
                decided = true;
                return false;
            }
            index += header->size / 8;
        }
        return false;
    };
    bool decided = false;
    const bool here = scan(current_list, next_record, decided);
    if (decided) {
        return here;
    }
    if (lookahead_submission != nullptr) {
        for (size_t i = lookahead_list + 1; i < lookahead_submission->lists.size(); ++i) {
            const bool later = scan(lookahead_submission->lists[i], 0, decided);
            if (decided) {
                return later;
            }
        }
    }
    // Neither a report nor a reset in the rest of the submission: a report of a later submission
    // may still read these samples (counted since the last reset, whatever the submission).
    return true;
}

void Recorder::ResetCounter(const VC::ResetCounterPayload& p) {
    CounterEngine* const engine = impl.Counters();
    if (!engine) {
        Unsupported("ResetCounter", "the counter engine could not be created");
        return;
    }
    engine->Stats().resets.fetch_add(1, std::memory_order_relaxed);
    if (p.counter_type != CounterType::SamplesPassed) {
        return; // no other counter accumulates
    }
    if (Debug::Get().draw_log != 0 && Debug::FrameSelected(impl.presents_recorded + 1)) {
        LOG_INFO(Render_Vulkan, "[nvn] draw-marker f{} reset samples", impl.presents_recorded + 1);
    }
    if (engine->InPassQueries()) {
        ReserveCounterEvents(BreakCause::ResetCounter);
        // Samples since the last report count for nobody; the accumulator is zeroed in order with
        // the reports around it (ResolveCounterBatch).
        if (occlusion) {
            engine->EndOcclusion(Cmd(), Counters(), *occlusion);
            occlusion.reset();
        }
        counter_batch.AddReset();
    } else {
        EndDrawPass(BreakCause::ResetCounter);
        if (occlusion) {
            engine->EndOcclusion(Cmd(), Counters(), *occlusion);
            occlusion.reset();
        }
        engine->ResetAccumulator(Cmd(), queue_index);
    }
    queue.uncounted_draws = 0;
    queue.occlusion_counting = SamplesReportFollows();
    barrier_pending = true;
}

void Recorder::SetRenderCondition(const VC::SetRenderConditionPayload& p) {
    CounterEngine* const engine = impl.Counters();
    if (!engine) {
        Unsupported("SetRenderCondition", "the counter engine could not be created; drawing");
        render_state = RenderState::Enabled;
        return;
    }
    EndRenderCondition();
    switch (p.kind) {
    case 1:
        render_state = RenderState::Enabled;
        return;
    case 2:
        render_state = RenderState::Disabled;
        return;
    case 0:
        break;
    default:
        Failed("SetRenderCondition", fmt::format("unknown kind {}", p.kind));
        render_state = RenderState::Enabled;
        return;
    }
    // NVNcounterData is 16 bytes: the counters compared are the first u64 of the report at
    // `address` and of the one at `address + 16`.
    Pool* pool_a{};
    Pool* pool_b{};
    u64 offset_a{};
    u64 offset_b{};
    if (!ResolvePoolRange(p.address, sizeof(u64), pool_a, offset_a, "SetRenderEnableConditional") ||
        !ResolvePoolRange(p.address + 16, sizeof(u64), pool_b, offset_b,
                          "SetRenderEnableConditional")) {
        engine->Stats().conditions_unresolved.fetch_add(1, std::memory_order_relaxed);
        render_state = RenderState::Enabled;
        return;
    }
    if (engine->GpuConditions()) {
        const u32 mode = p.mode & 1;
        const ConditionPrefetch::Entry* const ahead =
            condition_prefetch.Find(p.address, mode, pool_a, offset_a, pool_b, offset_b);
        // An UpdateUniformBuffer on the pair lands when the pass ends (EndDrawPass below): not
        // noted as a write yet.
        if (ahead != nullptr &&
            engine->SlotsAllocated() <= ahead->position + CounterEngine::SLOT_COUNT &&
            !draw.InlineUpdatePending(p.address, CounterEngine::REPORT_PAIR_SPAN) &&
            !HasLinearWritebackOverlapping(*pool_a, offset_a, offset_a + sizeof(u64)) &&
            !HasLinearWritebackOverlapping(*pool_b, offset_b, offset_b + sizeof(u64))) {
            // Computed ahead, nothing wrote its report pair since and its slot still holds it: the
            // predicate this point would compute, without a pass break or any command.
            condition_slot = ahead->slot;
            render_state = RenderState::Conditional;
            engine->Stats().conditions_prefetched.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        EndDrawPass(BreakCause::RenderCondition);
        if (!linear_writebacks.empty()) {
            FlushLinearWritebacks(pool_a, offset_a, offset_a + sizeof(u64));
            FlushLinearWritebacks(pool_b, offset_b, offset_b + sizeof(u64));
        }
        // CPU-written counters of mirrored pools must reach the mirror first.
        UploadRange(*pool_a, offset_a, sizeof(u64));
        UploadRange(*pool_b, offset_b, sizeof(u64));
        PrefetchConditions(p.address, mode, *pool_a, offset_a, *pool_b, offset_b);
        return;
    }
    EndDrawPass(BreakCause::RenderCondition);
    if (!linear_writebacks.empty()) {
        FlushLinearWritebacks(pool_a, offset_a, offset_a + sizeof(u64));
        FlushLinearWritebacks(pool_b, offset_b, offset_b + sizeof(u64));
    }
    UploadRange(*pool_a, offset_a, sizeof(u64));
    UploadRange(*pool_b, offset_b, sizeof(u64));
    EvaluateConditionOnCpu(*pool_a, offset_a, *pool_b, offset_b, p.mode & 1);
}

void Recorder::PrefetchConditions(u64 address, u32 mode, Pool& pool_a, u64 offset_a, Pool& pool_b,
                                  u64 offset_b) {
    CounterEngine& engine = *impl.counters_engine;
    prefetch_batch.clear();
    prefetch_batch.push_back({.address = address,
                              .mode = mode,
                              .pool_a = &pool_a,
                              .offset_a = offset_a,
                              .pool_b = &pool_b,
                              .offset_b = offset_b});
    // OoT sets ~1,200 conditions a frame, mostly inside render passes, on report pairs written
    // long before (earlier passes): each one ending the pass to compute its predicate cost more
    // GPU time than the draws. The conditions ahead in this list are computed here as well, unless
    // a report ahead of them writes their pair (the lookahead sees those). Any other write that
    // reaches the pair before the condition drops the predicate (NotePoolWrite, shadowed and raw
    // global stores), and so do a wait for another queue and the end of the command buffer: the
    // condition then computes it at its own point as before.
    prefetch_written.clear();
    const auto mark = [&](u64 at, u64 size) {
        for (u64 granule = at >> 3; granule <= (at + size - 1) >> 3; ++granule) {
            prefetch_written.insert(granule);
        }
    };
    const auto written = [&](u64 at) {
        return prefetch_written.contains(at >> 3) || prefetch_written.contains((at + 7) >> 3);
    };
    // A condition whose predicate is dropped computes a batch again: the walk is bounded so that a
    // list with few conditions far apart costs no more than a few thousand records per batch.
    // EDEN_NX2_NVN_CONDITION_PREFETCH=0: the condition's own predicate only, nothing kept.
    const bool ahead = Debug::Get().condition_prefetch;
    constexpr size_t MAX_RECORDS_AHEAD = 4096;
    size_t index = next_record;
    for (size_t records = 0; ahead && index < current_list.size() && records < MAX_RECORDS_AHEAD &&
                             prefetch_batch.size() < CounterEngine::MAX_CONDITION_BATCH;
         ++records) {
        const auto* header = reinterpret_cast<const VC::RecordHeader*>(&current_list[index]);
        if (header->size < sizeof(VC::RecordHeader) || header->size % 8 != 0 ||
            index + header->size / 8 > current_list.size()) {
            break;
        }
        const auto op = static_cast<VC::Op>(header->op);
        const size_t payload_size = header->size - sizeof(VC::RecordHeader);
        if (op == VC::Op::End || op == VC::Op::WaitSync || op == VC::Op::WaitEvent) {
            break;
        }
        if (op == VC::Op::ReportCounter && payload_size >= sizeof(VC::ReportCounterPayload)) {
            mark(reinterpret_cast<const VC::ReportCounterPayload*>(header + 1)->address,
                 2 * sizeof(u64));
        } else if (op == VC::Op::ReportValue && payload_size >= sizeof(VC::ReportValuePayload)) {
            mark(reinterpret_cast<const VC::ReportValuePayload*>(header + 1)->address, sizeof(u32));
        } else if (op == VC::Op::SetRenderCondition &&
                   payload_size >= sizeof(VC::SetRenderConditionPayload)) {
            const auto& next = *reinterpret_cast<const VC::SetRenderConditionPayload*>(header + 1);
            const u32 next_mode = next.mode & 1;
            const auto same = [&](const ConditionPrefetch::Entry& c) {
                return c.address == next.address && c.mode == next_mode;
            };
            Pool* next_a{};
            Pool* next_b{};
            u64 next_offset_a{};
            u64 next_offset_b{};
            if (next.kind == 0 && !written(next.address) && !written(next.address + 16) &&
                std::ranges::none_of(prefetch_batch, same) &&
                !condition_prefetch.Contains(next.address, next_mode) &&
                ResolveReportPairQuietly(next.address, next_a, next_offset_a, next_b,
                                         next_offset_b)) {
                prefetch_batch.push_back({.address = next.address,
                                          .mode = next_mode,
                                          .pool_a = next_a,
                                          .offset_a = next_offset_a,
                                          .pool_b = next_b,
                                          .offset_b = next_offset_b});
            }
        }
        index += header->size / 8;
    }
    const u32 count = static_cast<u32>(prefetch_batch.size());
    const u32 first_slot = engine.AllocateSlots(flight.seq, count);
    const u64 first_position = engine.SlotsAllocated() - count;
    // The ring may have come back to the slots of earlier predicates of this command buffer.
    condition_prefetch.DropRecycled(engine.SlotsAllocated(), CounterEngine::SLOT_COUNT);
    prefetch_inputs.clear();
    for (u32 i = 0; i < count; ++i) {
        ConditionPrefetch::Entry& c = prefetch_batch[i];
        c.slot = first_slot + i * CounterEngine::SLOT_SIZE;
        c.position = first_position + i;
        prefetch_inputs.push_back(
            {c.pool_a->Handle(), c.offset_a, c.pool_b->Handle(), c.offset_b, c.mode});
    }
    engine.RecordConditions(Cmd(), prefetch_inputs, first_slot);
    condition_slot = prefetch_batch.front().slot;
    render_state = RenderState::Conditional;
    barrier_pending = true;
    if (!ahead) {
        return;
    }
    // Mirrored pools take CPU writes at the next upload, which a predicate computed ahead would
    // miss: those are used at their own point only (the lookahead never takes them).
    for (const ConditionPrefetch::Entry& c : prefetch_batch) {
        if (c.pool_a->backing != Pool::Backing::Mirror &&
            c.pool_b->backing != Pool::Backing::Mirror) {
            condition_prefetch.Add(c);
        }
    }
}

bool Recorder::ResolveReportPairQuietly(u64 address, Pool*& pool_a, u64& offset_a, Pool*& pool_b,
                                        u64& offset_b) {
    const auto resolve = [&](u64 at, Pool*& pool, u64& offset) {
        if (!impl.ResolveBuffer(at, sizeof(u64), pool, offset)) {
            return false;
        }
        impl.EnsureBacking(*pool);
        if (pool->backing == Pool::Backing::None || pool->backing == Pool::Backing::Mirror) {
            return false;
        }
        impl.NoteUse(*pool, flight.seq);
        if (draw.HasShadows()) {
            // The copies read the bytes through host memory.
            draw.FlushShadowsOverlapping(*pool, offset, sizeof(u64));
        }
        return true;
    };
    return resolve(address, pool_a, offset_a) && resolve(address + 16, pool_b, offset_b);
}

void Recorder::EvaluateConditionOnCpu(Pool& pool_a, u64 offset_a, Pool& pool_b, u64 offset_b,
                                      u32 mode) {
    // No predicated rendering: copy both counters to the host, run everything recorded so far,
    // and decide here. Correct but synchronous.
    auto readback = std::make_shared<vk::Buffer>(impl.CreateHostBuffer(
        2 * sizeof(u64), VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::Download));
    flight.keepalive.push_back(readback);
    MemoryBarrier();
    const VkBufferCopy copy_a{.srcOffset = offset_a, .dstOffset = 0, .size = sizeof(u64)};
    const VkBufferCopy copy_b{.srcOffset = offset_b, .dstOffset = sizeof(u64), .size = sizeof(u64)};
    Cmd().CopyBuffer(pool_a.Handle(), **readback, copy_a);
    Cmd().CopyBuffer(pool_b.Handle(), **readback, copy_b);
    host_barrier_needed = true;
    FlushAndWait();
    readback->Invalidate();
    u64 values[2]{};
    std::memcpy(values, readback->Mapped().data(), sizeof(values));
    const bool equal = values[0] == values[1];
    const bool render = mode == ConditionMode::RenderIfNotEqual ? !equal : equal;
    CounterEngineStats& stats = impl.counters_engine->Stats();
    stats.conditions_cpu.fetch_add(1, std::memory_order_relaxed);
    if (stats.conditions_cpu.load(std::memory_order_relaxed) <= 64) {
        LOG_INFO(Render_Vulkan, "[nvn] render condition #{}: {} counters {:#x} / {:#x} -> {}",
                 stats.conditions_cpu.load(std::memory_order_relaxed),
                 mode == ConditionMode::RenderIfNotEqual ? "render if not equal"
                                                         : "render if equal",
                 values[0], values[1], render ? "render" : "skip");
    }
    if (!render) {
        stats.conditions_false.fetch_add(1, std::memory_order_relaxed);
    }
    render_state = render ? RenderState::Enabled : RenderState::Disabled;
}

void Recorder::LogOp(VC::Op op, const void* payload, size_t payload_size) {
    const u64 frame = impl.presents_recorded + 1;
    const auto line = [&](const std::string& text) {
        LOG_INFO(Render_Vulkan, "[nvn] op f{} #{} {}", frame, ++op_log_count, text);
    };
    switch (op) {
    case VC::Op::ClearBuffer:
        if (payload_size >= sizeof(VC::ClearBufferPayload)) {
            const auto* p = static_cast<const VC::ClearBufferPayload*>(payload);
            line(fmt::format("ClearBuffer {:#x}+{:#x} value {:#x}", p->address, p->size, p->value));
        }
        break;
    case VC::Op::CopyBufferToBuffer:
        if (payload_size >= sizeof(VC::CopyBufferToBufferPayload)) {
            const auto* p = static_cast<const VC::CopyBufferToBufferPayload*>(payload);
            line(fmt::format("CopyBufferToBuffer {:#x} -> {:#x} +{:#x}", p->src, p->dst, p->size));
        }
        break;
    case VC::Op::UpdateUniformBuffer:
        if (payload_size >= sizeof(VC::UpdateUniformBufferPayload)) {
            const auto* p = static_cast<const VC::UpdateUniformBufferPayload*>(payload);
            u32 words[5]{};
            std::memcpy(
                words, p + 1,
                (std::min)(sizeof(words), payload_size - sizeof(VC::UpdateUniformBufferPayload)));
            line(fmt::format(
                "UpdateUniformBuffer {:#x}+{:#x} first words {:#x} {:#x} {:#x} {:#x} {:#x}",
                p->address, p->size, words[0], words[1], words[2], words[3], words[4]));
        }
        break;
    case VC::Op::Dispatch:
        if (payload_size >= sizeof(VC::DispatchPayload)) {
            const auto* p = static_cast<const VC::DispatchPayload*>(payload);
            line(fmt::format("Dispatch {}x{}x{} indirect {:#x}", p->x, p->y, p->z, p->indirect));
        }
        break;
    case VC::Op::EvaluateDlss:
        if (payload_size >= sizeof(VC::EvaluateDlssPayload)) {
            const auto* p = static_cast<const VC::EvaluateDlssPayload*>(payload);
            line(fmt::format("EvaluateDlss {}x{} -> {}x{} flags {:#x} reset {}", p->input_width,
                             p->input_height, p->output_width, p->output_height, p->flags,
                             p->history_reset));
        }
        break;
    case VC::Op::ReportValue:
        if (payload_size >= sizeof(VC::ReportValuePayload)) {
            const auto* p = static_cast<const VC::ReportValuePayload*>(payload);
            line(fmt::format("ReportValue {:#x} = {:#x}", p->address, p->value));
        }
        break;
    case VC::Op::Barrier:
        if (payload_size >= sizeof(VC::BarrierPayload)) {
            line(fmt::format("Barrier bits {:#x}",
                             static_cast<const VC::BarrierPayload*>(payload)->bits));
        }
        break;
    case VC::Op::SetRenderTargets:
        if (payload_size >= sizeof(VC::SetRenderTargetsPayload)) {
            const auto* p = static_cast<const VC::SetRenderTargetsPayload*>(payload);
            line(fmt::format("SetRenderTargets colors {} depth {}", p->color_count, p->has_depth));
        }
        break;
    case VC::Op::ClearTexture:
        line("ClearTexture");
        break;
    case VC::Op::CopyBufferToTexture:
        line("CopyBufferToTexture");
        break;
    case VC::Op::CopyTextureToBuffer:
        line("CopyTextureToBuffer");
        break;
    case VC::Op::CopyTextureToTexture:
        line("CopyTextureToTexture");
        break;
    case VC::Op::ClearColor:
        line("ClearColor");
        break;
    case VC::Op::ClearDepthStencil:
        line("ClearDepthStencil");
        break;
    case VC::Op::Draw:
    case VC::Op::DrawIndexed:
        line(op == VC::Op::Draw ? "Draw" : "DrawIndexed");
        break;
    case VC::Op::ReportCounter:
        if (payload_size >= sizeof(VC::ReportCounterPayload)) {
            const auto* p = static_cast<const VC::ReportCounterPayload*>(payload);
            line(fmt::format("ReportCounter type {} -> {:#x}", p->counter_type, p->address));
        }
        break;
    case VC::Op::ResetCounter:
        if (payload_size >= sizeof(VC::ResetCounterPayload)) {
            line(fmt::format("ResetCounter type {}",
                             static_cast<const VC::ResetCounterPayload*>(payload)->counter_type));
        }
        break;
    case VC::Op::SetRenderCondition:
        if (payload_size >= sizeof(VC::SetRenderConditionPayload)) {
            const auto* p = static_cast<const VC::SetRenderConditionPayload*>(payload);
            line(fmt::format("SetRenderCondition kind {} mode {} at {:#x}", p->kind, p->mode,
                             p->address));
        }
        break;
    default:
        break;
    }
}

void Recorder::WatchSnapshot(std::string tag, u64 peek_address, u32 size_override) {
    const Debug::Config& config = Debug::Get();
    Pool* pool{};
    u64 offset{};
    const u64 address = config.watch_address == 1   ? draw.WatchHint()
                        : config.watch_address == 2 ? draw.WatchIndirectHint()
                        : config.watch_address == 3 || config.watch_address == 4
                            ? peek_address
                            : config.watch_address;
    const u32 size = size_override != 0          ? size_override
                     : config.watch_address == 3 ? config.peek_words * 4
                                                 : config.watch_size;
    if (address == 0 || !ResolvePoolRange(address, size, pool, offset, "Watch")) {
        return;
    }
    if (impl.watch_pool_logged != pool->serial) {
        impl.watch_pool_logged = pool->serial;
        LOG_INFO(Render_Vulkan,
                 "[nvn] watch address {:#x}: pool {:#x} (+{:#x} bytes) flags {:#x} backing {} {}",
                 address, pool->address, pool->size, pool->flags, static_cast<int>(pool->backing),
                 pool->chunk ? "in a chunk" : "own buffer");
    }
    UploadRange(*pool, offset, size);
    FlushPending();
    InFlight::Watch entry;
    entry.size = size;
    entry.tag = std::move(tag);
    entry.buffer =
        impl.CreateHostBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::Download);
    const VkBufferCopy region{.srcOffset = offset, .dstOffset = 0, .size = size};
    Cmd().CopyBuffer(pool->Handle(), *entry.buffer, region);
    barrier_pending = true;
    host_barrier_needed = true;
    flight.watches.push_back(std::move(entry));
}

bool Recorder::RecordList(std::span<const u64> list, size_t start, size_t& stop, bool& release) {
    const bool resuming = std::exchange(resumed_wait, false);
    if (resuming) {
        // The time other streams ran is not this piece's recording time.
        piece_start = std::chrono::steady_clock::now();
    }
    size_t index = start;
    while (index < list.size()) {
        if (++ops_since_check >= 32) {
            ops_since_check = 0;
            current_list = list;
            next_record = index;
            MaybeSubmitEarly();
        }
        const auto* header = reinterpret_cast<const VC::RecordHeader*>(&list[index]);
        const size_t words = header->size / sizeof(u64);
        if (header->size < sizeof(VC::RecordHeader) || header->size % 8 != 0 ||
            index + words > list.size()) {
            Failed("record", fmt::format("malformed record (op {}, size {}) at word {}", header->op,
                                         header->size, index));
            return true;
        }
        ++records_replayed;
        const auto op = static_cast<VC::Op>(header->op);
        const void* payload = header + 1;
        const size_t payload_size = header->size - sizeof(VC::RecordHeader);
        if (flight.gpu_prof) {
            flight.gpu_prof->MarkOp(Cmd(), header->op, draw.PassOpen());
        }
        const auto require = [&](size_t size) {
            if (payload_size < size) {
                Failed("record", fmt::format("op {} payload is {} bytes, expected {}", header->op,
                                             payload_size, size));
                return false;
            }
            return true;
        };
        if (Debug::Get().op_log && Debug::FrameSelected(impl.presents_recorded + 1) &&
            !(resuming && index == start)) {
            LogOp(op, payload, payload_size);
        }
        // Draw-path ops run inside the pass the draw context keeps open; everything else is
        // recorded outside of it.
        // A timestamp report can be written inside a pass (its zero-fill is deferred); with in-pass
        // queries SAMPLES_PASSED reports and resets are too.
        if (!DrawContext::OpKeepsPass(op) && !CounterOpKeepsPass(op, payload, payload_size)) {
            switch (op) {
            case VC::Op::Dispatch:
                EndDrawPass(BreakCause::Dispatch);
                break;
            case VC::Op::ReportValue:
                EndDrawPass(BreakCause::ReportValue);
                break;
            case VC::Op::ReportCounter:
                EndDrawPass(BreakCause::ReportCounter);
                break;
            case VC::Op::ResetCounter:
                EndDrawPass(BreakCause::ResetCounter);
                break;
            default:
                EndDrawPass(BreakCause::OtherOp);
                break;
            }
        }
        if (Debug::Get().watch_address == 3) {
            // The GPU view of the peeked range, after the peeked dispatch and before what follows.
            if (const u64 peek = draw.TakePeekAddress(); peek != 0) {
                WatchSnapshot(fmt::format("f{} peek {:#x} after op #{}", impl.presents_recorded + 1,
                                          peek, op_log_count),
                              peek);
                if (const u64 chase = std::exchange(draw.peek_chase_address, 0); chase != 0) {
                    WatchSnapshot(fmt::format("f{} chase {:#x} after op #{}",
                                              impl.presents_recorded + 1, chase, op_log_count),
                                  chase, Debug::Get().chase_words * 4);
                }
            }
        } else if (Debug::Get().watch_address == 4) {
            // The GPU view of each indirect dispatch's arguments, right before it.
            if (op == VC::Op::Dispatch && payload_size >= sizeof(VC::DispatchPayload) &&
                Debug::Get().op_log && Debug::FrameSelected(impl.presents_recorded + 1)) {
                const auto* p = static_cast<const VC::DispatchPayload*>(payload);
                if (p->indirect != 0) {
                    WatchSnapshot(fmt::format("f{} disp {:#x} op #{}", impl.presents_recorded + 1,
                                              p->indirect, op_log_count),
                                  p->indirect);
                }
            }
        } else if (Debug::Get().watch_address != 0 && Debug::Get().op_log &&
                   Debug::FrameSelected(impl.presents_recorded + 1) &&
                   (op == VC::Op::Dispatch || op == VC::Op::Barrier ||
                    op == VC::Op::SetRenderTargets)) {
            WatchSnapshot(
                fmt::format("f{} before op #{}", impl.presents_recorded + 1, op_log_count));
        }
        current_list = list;
        next_record = index + words;
        if (static_cast<u16>(op) >= static_cast<u16>(VC::Op::SetRenderTargets) &&
            draw.Handle(header)) {
            index += words;
            continue;
        }
        switch (op) {
        case VC::Op::End:
            return true;
        case VC::Op::CallList:
            // The recorder expands call lists before submitting.
            break;
        case VC::Op::ClearTexture:
            if (require(sizeof(VC::ClearTexturePayload))) {
                if ((header->flags & 0xF) != 0xF) {
                    // The recorder keeps the NVN channel write mask in the record flags.
                    Unsupported("ClearTexture", "channel write mask is not applied; clearing all "
                                                "channels");
                }
                const auto& clear = *static_cast<const VC::ClearTexturePayload*>(payload);
                const GpuProf::Scope prof{
                    flight.gpu_prof.get(), Cmd(),
                    flight.gpu_prof ? ProfTexture(GpuProf::Kind::Clear, clear.texture.serial)
                                    : GpuProf::Tag{}};
                ClearTexture(clear);
            }
            break;
        case VC::Op::ClearBuffer:
            if (require(sizeof(VC::ClearBufferPayload))) {
                const auto& clear = *static_cast<const VC::ClearBufferPayload*>(payload);
                const GpuProf::Scope prof{flight.gpu_prof.get(), Cmd(),
                                          GpuProf::Tag{.kind = GpuProf::Kind::Clear,
                                                       .count = static_cast<u32>(clear.size)}};
                ClearBuffer(clear);
            }
            break;
        case VC::Op::CopyBufferToBuffer:
            if (require(sizeof(VC::CopyBufferToBufferPayload))) {
                const auto& copy = *static_cast<const VC::CopyBufferToBufferPayload*>(payload);
                const GpuProf::Scope prof{flight.gpu_prof.get(), Cmd(),
                                          GpuProf::Tag{.kind = GpuProf::Kind::Copy,
                                                       .count = static_cast<u32>(copy.size)}};
                CopyBufferToBuffer(copy);
            }
            break;
        case VC::Op::CopyBufferToTexture:
            if (require(sizeof(VC::CopyBufferTexturePayload))) {
                const auto& copy = *static_cast<const VC::CopyBufferTexturePayload*>(payload);
                const GpuProf::Scope prof{
                    flight.gpu_prof.get(), Cmd(),
                    flight.gpu_prof ? ProfTexture(GpuProf::Kind::Copy, copy.texture.serial)
                                    : GpuProf::Tag{}};
                CopyBufferToTexture(copy);
            }
            break;
        case VC::Op::CopyTextureToBuffer:
            if (require(sizeof(VC::CopyBufferTexturePayload))) {
                const auto& copy = *static_cast<const VC::CopyBufferTexturePayload*>(payload);
                const GpuProf::Scope prof{
                    flight.gpu_prof.get(), Cmd(),
                    flight.gpu_prof ? ProfTexture(GpuProf::Kind::Copy, copy.texture.serial)
                                    : GpuProf::Tag{}};
                CopyTextureToBuffer(copy);
            }
            break;
        case VC::Op::CopyTextureToTexture:
            if (require(sizeof(VC::CopyTextureToTexturePayload))) {
                const auto& copy = *static_cast<const VC::CopyTextureToTexturePayload*>(payload);
                const GpuProf::Scope prof{flight.gpu_prof.get(), Cmd(),
                                          flight.gpu_prof
                                              ? ProfTexture(GpuProf::Kind::Copy, copy.dst.serial)
                                              : GpuProf::Tag{}};
                CopyTextureToTexture(copy);
            }
            break;
        case VC::Op::FenceSync:
            if (require(sizeof(VC::FenceSyncPayload))) {
                FenceSync(*static_cast<const VC::FenceSyncPayload*>(payload));
            }
            break;
        case VC::Op::SignalEvent:
            if (require(sizeof(VC::SignalEventPayload))) {
                SignalEvent(*static_cast<const VC::SignalEventPayload*>(payload));
            }
            break;
        case VC::Op::WaitEvent:
            // Another queue's writes may reach report pairs after this point.
            condition_prefetch.Clear();
            if (require(sizeof(VC::WaitEventPayload))) {
                const auto& wait = *static_cast<const VC::WaitEventPayload*>(payload);
                if (!WaitEvent(wait, release)) {
                    stop = index;
                    --records_replayed;
                    resumed_wait = true;
                    return false;
                }
                draw.OnEventWaitPassed(wait.event_address);
            }
            break;
        case VC::Op::WaitSync:
            condition_prefetch.Clear();
            if (require(sizeof(VC::WaitSyncPayload))) {
                WaitSync(*static_cast<const VC::WaitSyncPayload*>(payload));
            }
            break;
        case VC::Op::ReportValue:
            if (require(sizeof(VC::ReportValuePayload))) {
                ReportValue(*static_cast<const VC::ReportValuePayload*>(payload));
            }
            break;
        case VC::Op::ReportCounter:
            if (require(sizeof(VC::ReportCounterPayload))) {
                ReportCounter(*static_cast<const VC::ReportCounterPayload*>(payload));
            }
            break;
        case VC::Op::ResetCounter:
            if (require(sizeof(VC::ResetCounterPayload))) {
                ResetCounter(*static_cast<const VC::ResetCounterPayload*>(payload));
            }
            break;
        case VC::Op::SetRenderCondition:
            if (require(sizeof(VC::SetRenderConditionPayload))) {
                SetRenderCondition(*static_cast<const VC::SetRenderConditionPayload*>(payload));
            }
            break;
        default:
            // Draw-path ops belong to later phases.
            Unsupported(fmt::format("op {}", header->op), "draw-path ops are not replayed yet");
            break;
        }
        index += words;
    }
    return true;
}

namespace {

/// The UNORM format with the same channel layout, or `format` itself when it is not sRGB.
VkFormat UnormOfSrgb(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8G8B8A8_SRGB:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB:
        return VK_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
        return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    default:
        return format;
    }
}

/// What the values of a presented frame mean: Eglantine has no HDR output, so the legacy sRGB
/// interpretation.
Common::Hdr::Encoding PresentEncoding(VkFormat texture_format) {
    static_cast<void>(texture_format);
    return Common::Hdr::Encoding::Srgb;
}

} // namespace

VkImage Recorder::RawPresentCopy(Storage& s, VkFormat raw_format, bool sampled) {
    Context::Impl::PresentScratch& scratch = impl.present_scratch;
    if (!scratch.image || scratch.format != raw_format || scratch.width != s.extent.width ||
        scratch.height != s.extent.height) {
        if (scratch.image) {
            // Earlier presents may still read the old image: release it once this one is done.
            auto old_image = std::make_shared<vk::Image>(std::move(scratch.image));
            auto old_view = std::make_shared<vk::ImageView>(std::move(scratch.view));
            impl.retired.push_back({flight.seq, [old_image, old_view]() mutable {
                                        *old_view = vk::ImageView{};
                                        *old_image = vk::Image{};
                                    }});
        }
        scratch.image = impl.allocator->CreateImage({
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = raw_format,
            .extent = {s.extent.width, s.extent.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
        scratch.view = impl.device.GetLogical().CreateImageView({
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = *scratch.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = raw_format,
            .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        });
        scratch.format = raw_format;
        scratch.width = s.extent.width;
        scratch.height = s.extent.height;
    }
    const VkImageSubresourceRange range{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = ALL_ACCESS,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = *scratch.image,
        .subresourceRange = range,
    };
    Cmd().PipelineBarrier(ALL_STAGES, ALL_STAGES, 0, barrier);
    const VkImageCopy copy{
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffset = {0, 0, 0},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffset = {0, 0, 0},
        .extent = {s.extent.width, s.extent.height, 1},
    };
    Cmd().CopyImage(*s.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *scratch.image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copy);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    // The sampled copy is also what the capture / screenshot readback blits from (a transfer read).
    barrier.dstAccessMask = sampled ? (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT)
                                    : VK_ACCESS_TRANSFER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = sampled ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    Cmd().PipelineBarrier(ALL_STAGES, ALL_STAGES, 0, barrier);
    return *scratch.image;
}

bool Recorder::RecordWindowFilter(Texture& texture, VkFormat source_format, const Frame& frame,
                                  const PixelRect& crop, const PixelRect& picture) {
    Storage& source = *texture.storage;
    using Settings::ScalingFilter;
    const auto filter = Settings::values.scaling_filter.GetValue();
    // ScaleForce shifts sampling positions even at native size. Use the plain blit when it
    // preserves the source bits without an intermediate sRGB-to-UNORM copy.
    const bool native_size = crop.width == picture.width && crop.height == picture.height;
    if ((filter == ScalingFilter::ScaleForce && native_size && source_format == source.host.format) ||
        filter == ScalingFilter::NearestNeighbor || filter == ScalingFilter::Bilinear ||
        !frame.framebuffer || source.type != VK_IMAGE_TYPE_2D ||
        !impl.device.IsFormatSupported(source_format,
                                       VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT,
                                       FormatType::Optimal)) {
        return false;
    }
    const VkExtent2D extent{picture.width, picture.height};
    if (impl.window_filter_setting != filter || impl.window_filter_format != frame.format ||
        impl.window_filter_extent.width != extent.width ||
        impl.window_filter_extent.height != extent.height) {
        impl.window_filter.reset();
        impl.window_fsr.reset();
        impl.window_fsr3.reset();
        impl.window_sgsr.reset();
        impl.window_filter_failed = false;
        impl.window_filter_setting = filter;
        impl.window_filter_format = frame.format;
        impl.window_filter_extent = extent;
    }
    if (impl.window_filter_failed) {
        return false;
    }
    try {
        if (!impl.window_filter) {
            const bool spatial = filter == ScalingFilter::Fsr || filter == ScalingFilter::Fsr3 ||
                                 filter == ScalingFilter::Sgsr || filter == ScalingFilter::SgsrEdge;
            impl.window_filter = MakeWindowAdaptPass(impl.device, frame.format,
                                                     spatial ? ScalingFilter::Bilinear : filter,
                                                     impl.device.IsKhrPushDescriptorSupported());
            if (filter == ScalingFilter::Fsr) {
                impl.window_fsr = std::make_shared<FSR>(impl.device, *impl.allocator, 1, extent);
            } else if (filter == ScalingFilter::Fsr3) {
                impl.window_fsr3 = std::make_shared<FSR3>(impl.device, *impl.allocator, 1, extent);
            } else if (filter == ScalingFilter::Sgsr || filter == ScalingFilter::SgsrEdge) {
                impl.window_sgsr = std::make_shared<SGSR>(impl.device, *impl.allocator, 1, extent,
                                                          filter == ScalingFilter::SgsrEdge);
            }
            LOG_INFO(Render_Vulkan,
                     "[nvn] window filter {} active (source format {}, frame format {})",
                     static_cast<u32>(filter), static_cast<u32>(source_format),
                     static_cast<u32>(frame.format));
        }
    } catch (const vk::Exception& exception) {
        impl.window_filter_failed = true;
        LOG_WARNING(Render_Vulkan, "[nvn] window filter {} unavailable: {}",
                    static_cast<u32>(filter), exception.what());
        return false;
    }
    const VkImageView source_view =
        impl.draw_resources->GetView(texture, {
                                                  .format = source_format,
                                                  .type = VK_IMAGE_VIEW_TYPE_2D,
                                                  .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                                  .levels = 1,
                                                  .layers = 1,
                                              });
    if (source_view == VK_NULL_HANDLE) {
        return false;
    }
    const bool saved = track_writes;
    track_writes = false;
    PrepareImages({{&source, VK_IMAGE_LAYOUT_GENERAL}});
    track_writes = saved;
    VkImageView view = source_view;
    Common::Rectangle<f32> uv{
        static_cast<f32>(crop.x) / static_cast<f32>(source.extent.width),
        static_cast<f32>(crop.y) / static_cast<f32>(source.extent.height),
        static_cast<f32>(crop.x + crop.width) / static_cast<f32>(source.extent.width),
        static_cast<f32>(crop.y + crop.height) / static_cast<f32>(source.extent.height),
    };
    const auto upscale = [&](auto& pass) {
        if (pass) {
            view = pass->Record(impl.device, Cmd(), 0, *source.image, view,
                                {source.extent.width, source.extent.height}, uv);
            flight.keepalive.push_back(pass);
            flight.keepalive.push_back(pass->TakeRecordResources());
            uv = {0.0f, 0.0f, 1.0f, 1.0f};
        }
    };
    upscale(impl.window_fsr);
    upscale(impl.window_fsr3);
    upscale(impl.window_sgsr);
    auto descriptors = impl.window_filter->RecordTexture(
        Cmd(), frame, view, uv,
        {picture.x, picture.y, picture.x + picture.width, picture.y + picture.height});
    if (descriptors) {
        flight.keepalive.push_back(std::move(descriptors));
    }
    flight.keepalive.push_back(impl.window_filter);
    return true;
}

void Recorder::WarnUnsupportedPresentPasses(bool applied_filter) {
    auto settings = CurrentPresentSettings();
    if (applied_filter) {
        settings.scaling_filter = Settings::ScalingFilter::Bilinear;
    }
    if (PresentNeedsPasses(settings) && impl.logged_messages.insert("present-passes").second) {
        LOG_WARNING(Render_Vulkan, "[nvn] the presentation passes (window filter, anti-aliasing, "
                                   "post-processing) requested here are not available on VTX-HLE");
    }
}

void Recorder::GenerateFrames(size_t count, const Frame& frame, std::vector<Frame*>& generated) {
    // The target cannot hand out more than its capacity; the pacer planned within it. The present
    // submission signals every frame it displays (MaxPresentSignals).
    count = (std::min)({count, impl.present_target->MaxExtraFrames(), MaxPresentSignals - 1});
    for (size_t generation = 0; generation < count; ++generation) {
        Frame* extra = impl.present_target->AcquireFrame();
        if (extra->width != frame.width || extra->height != frame.height ||
            extra->format != frame.format || !extra->storage_view) {
            // The window changed size between the two acquisitions: the chain's output does not
            // fit this frame. The next present generates again at the new size.
            impl.present_target->ReleaseFrame(extra);
            break;
        }
        generated.push_back(extra);
        // The generated picture is shown through the same output pass as the real one.
        extra->source_encoding = frame.source_encoding;
        // The generated picture overwrites every texel; the frame's contents (an earlier picture)
        // are not needed, only its layout is made what the compute shader writes through.
        const VkImageMemoryBarrier to_general{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = ALL_ACCESS,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *extra->image,
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        Cmd().PipelineBarrier(ALL_STAGES, ALL_STAGES, 0, to_general);
        impl.present_frame_gen->GenerateInto(impl.device, *extra, generation);
    }
}

GpuProf::Tag Recorder::ProfStorage(GpuProf::Kind kind, const Storage& s, u64 sub) {
    return GpuProf::Tag{.kind = kind,
                        .hash_a = sub,
                        .format = static_cast<s32>(s.host.format),
                        .width = s.extent.width,
                        .height = s.extent.height};
}

GpuProf::Tag Recorder::ProfTexture(GpuProf::Kind kind, u64 serial, u32 count) const {
    GpuProf::Tag tag{.kind = kind, .count = count};
    if (const auto it = impl.textures.find(serial);
        it != impl.textures.end() && it->second->storage) {
        tag = ProfStorage(kind, *it->second->storage);
        tag.count = count;
    }
    return tag;
}

bool Recorder::RecordPresent(Texture& texture, Frame& frame, std::vector<Frame*>& generated,
                             const VideoCommon::Nvn::PresentCrop& window_crop) {
    VideoCommon::NvnDrawProbe::OnPresent();
    if (impl.vtxcov_diag) {
        impl.VtxCovOnPresent();
    }
    Storage& guest = *texture.storage;
    const bool own_upscaled = impl.ndsr_upscaled_since_present;
    const auto own_stage = impl.upscaled_stage;
    impl.ndsr_upscaled_since_present = false;
    impl.dlss_upscaled_since_present = false;
    impl.upscaled_stage = VideoCore::GameFrameSize::Stage::Game;
    impl.compute_chain_count = 0;
    impl.compute_chain_next = 0;
    impl.pass_depth_of_color.clear();
    GpuProf::Scope prof{flight.gpu_prof.get(), Cmd(),
                        flight.gpu_prof ? ProfStorage(GpuProf::Kind::Present, guest)
                                        : GpuProf::Tag{}};
    NoteTouched(texture.storage);
    // The picture is the scaled twin when the Resolution setting drew it (and nothing wrote the
    // guest-sized image since); everything below reads `s` and sizes in its pixels.
    const bool twin_only = !guest.image && guest.scaled &&
                           Rescale::NativeIsStale(guest.scale_state) && guest.scaled->image;
    const bool image_ready = twin_only || EnsureContents(guest, false, true);
    Storage& s = image_ready ? PresentSource(guest) : guest;
    const Rescale::Factor present_scale = &s != &guest ? impl.RescaleFactor() : Rescale::Factor{};
    if (!image_ready || !s.image) {
        Unsupported("Present", "the source image could not be created");
        return false;
    }
    if (s.samples != VK_SAMPLE_COUNT_1_BIT || s.host.depth || s.host.stencil) {
        Unsupported("Present", "multisampled or depth/stencil source");
        return false;
    }
    if (!s.sparse && s.origin == StorageOrigin::None) {
        // A failed initial upload can leave an allocated image with no pixels. Try the upload
        // again on the next use, and leave the last complete picture on screen in the meantime.
        s.contents_valid = false;
        Unsupported("Present", "the source image has no contents");
        return false;
    }
    // For the status bar: the stage that actually wrote the frame and its real input/output sizes
    // (a rejected frame leaves the last published ones).
    const Rescale::Factor setting = impl.RescaleFactor();
    VideoCore::GameFrameSize::Publish({
        .render_width = own_upscaled ? impl.upscaled_input_width : 0,
        .render_height = own_upscaled ? impl.upscaled_input_height : 0,
        .frame_width = own_upscaled ? impl.upscaled_output_width : texture.desc.width,
        .frame_height = own_upscaled ? impl.upscaled_output_height : texture.desc.height,
        .scale_percent = static_cast<u32>(setting.UpFactor() * 100.0f + 0.5f),
        .stage = own_upscaled ? own_stage : VideoCore::GameFrameSize::Stage::Game,
    });
    const VkImageMemoryBarrier frame_in{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = *frame.image,
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    // Captures and screenshots read the present's source back (RecordPictureReadback): the guest
    // picture, not the frame the chain above draws into.
    VkImage picture_source = VK_NULL_HANDLE;
    VkImageLayout picture_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    const VkFilter present_filter =
        Settings::values.scaling_filter.GetValue() == Settings::ScalingFilter::NearestNeighbor
            ? VK_FILTER_NEAREST
            : VK_FILTER_LINEAR;
    // The window's layout: the screen rectangle the picture goes into (aspect ratio setting,
    // as the emulated GPU's presentation) and the size of the frame.
    const Layout::FramebufferLayout layout = impl.present_target->ScreenLayout();
    // The part of the texture the game's window crop shows is the picture.
    const PixelRect guest_crop =
        CropRectOf(window_crop.x, window_crop.y, window_crop.width, window_crop.height,
                   guest.extent.width, guest.extent.height);
    // The same part in the pixels of the image that is read (the twin of a scaled texture).
    PixelRect crop_rect = guest_crop;
    if (&s != &guest) {
        const Rescale::Rect scaled = Rescale::ScaleRect(
            present_scale, {static_cast<s32>(guest_crop.x), static_cast<s32>(guest_crop.y),
                            guest_crop.width, guest_crop.height});
        crop_rect.x = (std::min)(static_cast<u32>(scaled.x), s.extent.width - 1);
        crop_rect.y = (std::min)(static_cast<u32>(scaled.y), s.extent.height - 1);
        crop_rect.width = (std::min)(scaled.width, s.extent.width - crop_rect.x);
        crop_rect.height = (std::min)(scaled.height, s.extent.height - crop_rect.y);
    }
    // The picture's aspect ratio is the guest's, whatever size the pixels are read at.
    const PixelRect picture_rect =
        PresentRectOf(layout.screen, layout.width, layout.height, guest_crop.width,
                      guest_crop.height, frame.width, frame.height);
    bool through_passes = false;
    const VkFormat texture_format = texture.host.format;
    const VkFormat raw_format = UnormOfSrgb(texture_format);
    const VkFormat frame_format = frame.format;
    const bool raw_copy = raw_format != s.host.format && UnormOfSrgb(frame_format) == frame_format;
    // Sample encoded values into UNORM frames; sRGB attachments require a decoding source
    // view so their automatic encode preserves the colours, just as the plain blit does.
    const VkFormat filter_format = raw_copy ? raw_format : texture_format;
    const auto filter_twin = &s != &guest ? impl.ScaledTextureOf(texture) : nullptr;
    Texture& filter_texture = filter_twin ? *filter_twin : texture;
    if (RecordWindowFilter(filter_texture, filter_format, frame, crop_rect, picture_rect)) {
        through_passes = true;
        picture_source = *s.image;
        picture_layout = VK_IMAGE_LAYOUT_GENERAL;
        frame.source_encoding = PresentEncoding(texture_format);
    }
    WarnUnsupportedPresentPasses(through_passes);
    if (!through_passes) {
        // The game renders to an sRGB texture, whose raw bytes are already encoded. A blit reads
        // an sRGB source as linear and writes the (UNORM) frame as is, which darkens the picture:
        // blit from a UNORM copy of the bytes instead. A texture can also be a different
        // mutable-format view of its cached storage (DK reuses its display-buffer bytes for the
        // RGBA8 -> RGB10A2 switch), so the copy must use the texture's format, not the storage's.
        frame.source_encoding = PresentEncoding(texture_format);
        PrepareImages({{&s, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL}}, std::span{&frame_in, 1});
        VkImage source = *s.image;
        if (raw_copy) {
            source = RawPresentCopy(s, raw_format);
        }
        const s32 blit_width = static_cast<s32>(picture_rect.width);
        const s32 blit_height = static_cast<s32>(picture_rect.height);
        const s32 blit_x = static_cast<s32>(picture_rect.x);
        const s32 blit_y = static_cast<s32>(picture_rect.y);
        if (blit_width != static_cast<s32>(frame.width) ||
            blit_height != static_cast<s32>(frame.height)) {
            const VkImageSubresourceRange range{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            };
            // The rest of the frame is the background colour setting.
            const f32 background[3]{
                static_cast<f32>(Settings::values.bg_red.GetValue()) / 255.0f,
                static_cast<f32>(Settings::values.bg_green.GetValue()) / 255.0f,
                static_cast<f32>(Settings::values.bg_blue.GetValue()) / 255.0f,
            };
            Cmd().ClearColorImage(
                *frame.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VkClearColorValue{.float32 = {background[0], background[1], background[2], 1.0f}},
                range);
            // The blit writes the same image: order it after the clear.
            const VkImageMemoryBarrier clear_done{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = *frame.image,
                .subresourceRange = range,
            };
            Cmd().PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                  clear_done);
        }
        const VkImageBlit blit{
            .srcSubresource{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .srcOffsets = {{static_cast<s32>(crop_rect.x), static_cast<s32>(crop_rect.y), 0},
                           {static_cast<s32>(crop_rect.x + crop_rect.width),
                            static_cast<s32>(crop_rect.y + crop_rect.height), 1}},
            .dstSubresource{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .dstOffsets = {{blit_x, blit_y, 0}, {blit_x + blit_width, blit_y + blit_height, 1}},
        };
        Cmd().BlitImage(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *frame.image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, blit, present_filter);
        picture_source = source;
    }
    // The present manager expects the frame image in GENERAL.
    const VkImageMemoryBarrier frame_out{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = *frame.image,
        .subresourceRange = frame_in.subresourceRange,
    };
    if (through_passes) {
        // The render pass of the window adapting filter leaves the frame in GENERAL.
        const VkMemoryBarrier passes_done{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        Cmd().PipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, ALL_STAGES, 0,
                              passes_done);
    } else {
        Cmd().PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, ALL_STAGES, 0, frame_out);
    }
    const auto requested_upscaler = VideoCore::Upscaler::Effective();
    static Settings::Upscaler last_upscaler = static_cast<Settings::Upscaler>(~0u);
    if (requested_upscaler != last_upscaler) {
        LOG_INFO(Render_Vulkan, "[dlss] HLE present upscaler setting {}",
                 static_cast<u32>(requested_upscaler));
        last_upscaler = requested_upscaler;
    }
    if (requested_upscaler == Settings::Upscaler::Fsr && !own_upscaled) {
        const VkExtent2D extent{frame.width, frame.height};
        if (!impl.final_fsr || !impl.final_fsr_extent ||
            impl.final_fsr_extent->width != extent.width ||
            impl.final_fsr_extent->height != extent.height) {
            if (impl.final_fsr) {
                flight.keepalive.push_back(std::shared_ptr<FSR>(std::move(impl.final_fsr)));
            }
            impl.final_fsr = std::make_unique<FSR>(impl.device, *impl.allocator, 1, extent);
            impl.final_fsr_extent = extent;
        }
        const Common::Rectangle<f32> crop{0.0f, 0.0f, 1.0f, 1.0f};
        impl.final_fsr->Record(impl.device, Cmd(), 0, *frame.image, *frame.image_view, extent,
                               crop);
        const std::array fsr_barriers{
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = *frame.image,
                .subresourceRange = frame_out.subresourceRange,
            },
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = impl.final_fsr->ResultImage(0),
                .subresourceRange = frame_out.subresourceRange,
            },
        };
        Cmd().PipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, {}, {}, fsr_barriers);
        const VkImageBlit fsr_blit{
            .srcSubresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets = {{0, 0, 0},
                           {static_cast<s32>(extent.width), static_cast<s32>(extent.height), 1}},
            .dstSubresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffsets = {{0, 0, 0},
                           {static_cast<s32>(extent.width), static_cast<s32>(extent.height), 1}},
        };
        Cmd().BlitImage(impl.final_fsr->ResultImage(0), VK_IMAGE_LAYOUT_GENERAL, *frame.image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, fsr_blit, VK_FILTER_LINEAR);
        const VkImageMemoryBarrier fsr_done{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *frame.image,
            .subresourceRange = frame_out.subresourceRange,
        };
        Cmd().PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, ALL_STAGES, 0, fsr_done);
        static Common::BootCounter final_fsr_evaluations;
        if (final_fsr_evaluations.Next() % 600 == 0) {
            LOG_INFO(Render_Vulkan, "[upscaler] final FSR evaluated {}x{}", extent.width,
                     extent.height);
        }
    }

    // Frame generation. The frame holds the final picture here (blit or passes, DLAA, neural
    // rendering and final FSR are recorded), and the caller hands it to the PresentManager only
    // after this command buffer is submitted, so the optical flow recorded now is on the queue
    // before the Streamline-hooked present that generates the extra frames reads it. FSR3 and
    // Lossless Scaling record their chain here and write the generated frames, which the caller
    // presents before this one.
    if (impl.present_frame_gen) {
        PresentRecording recording{
            .cmdbuf = Cmd(),
            .seq = flight.seq,
            .is_complete =
                [this](u64 token) {
                    return impl.completed_seq.load(std::memory_order_acquire) >= token;
                },
            .keepalive = &flight.keepalive,
        };
        const size_t generated_count = impl.present_frame_gen->Process(
            impl.device, frame, static_cast<VkFormat>(impl.present_target->SwapchainImageFormat()),
            VkExtent2D{guest.extent.width, guest.extent.height},
            impl.present_target->UsesPresentThread(), recording,
            impl.present_target->MaxExtraFrames());
        GenerateFrames(generated_count, frame, generated);
    }

    // EDEN_NX2_FRAME_CAPTURE: every 120th present is read back for a screenshot (completion
    // thread writes the file). EDEN_NX2_FRAME_CAPTURE_EVERY=<n> and EDEN_NX2_FRAME_CAPTURE_RANGE=
    // <first>-<last> (present numbers) change the schedule, e.g. every 4th present of a window.
    // EDEN_NX2_FRAME_CAPTURE_SKIPS=<n> captures instead the first n presents (inside the range)
    // whose frame had a draw skipped awaiting an asynchronous pipeline.
    struct CaptureSchedule {
        u64 every{120};
        u64 first{0};
        u64 last{~u64{0}};
        u64 skip_frames{0};
    };
    static const CaptureSchedule schedule = [] {
        CaptureSchedule result;
        if (const char* every = std::getenv("EDEN_NX2_FRAME_CAPTURE_EVERY"); every != nullptr) {
            result.every = (std::max)(u64{1}, static_cast<u64>(std::strtoull(every, nullptr, 10)));
        }
        if (const char* range = std::getenv("EDEN_NX2_FRAME_CAPTURE_RANGE"); range != nullptr) {
            char* end = nullptr;
            result.first = std::strtoull(range, &end, 10);
            if (end != nullptr && *end == '-') {
                result.last = std::strtoull(end + 1, nullptr, 10);
            }
        }
        if (const char* skips = std::getenv("EDEN_NX2_FRAME_CAPTURE_SKIPS"); skips != nullptr) {
            result.skip_frames = std::strtoull(skips, nullptr, 10);
        }
        return result;
    }();
    const u64 present_index = ++impl.presents_recorded;
    if (impl.costsplit) {
        impl.costsplit->OnPresent(present_index);
    }
    if (Debug::OnPresent(present_index)) {
        // A diag request (nvn-pass / nvn-draw) bound to the frame that just ended is done.
        for (auto& [index, context] : impl.draw_contexts) {
            context->DebugEndFrame();
        }
    }
    bool capture_due = present_index >= schedule.first && present_index <= schedule.last;
    if (schedule.skip_frames != 0) {
        const DrawCounters* counters =
            impl.draw_resources ? &impl.draw_resources->counters : nullptr;
        const bool skipped = counters != nullptr && counters->draws_skipped_async.load() !=
                                                        impl.draw_resources->noted_draw_skips;
        const auto after = std::find(impl.capture_after_frames.begin(),
                                     impl.capture_after_frames.end(), present_index);
        const bool is_after = after != impl.capture_after_frames.end();
        if (is_after) {
            impl.capture_after_frames.erase(after);
        }
        const bool is_skip =
            capture_due && skipped && impl.capture_skip_frames_captured < schedule.skip_frames;
        if (is_skip) {
            ++impl.capture_skip_frames_captured;
            impl.capture_after_frames.push_back(present_index + 8);
        }
        capture_due = is_skip || is_after;
    } else {
        capture_due = (Common::FrameCapture::Periodic() && capture_due &&
                       (present_index - 1) % schedule.every == 0) ||
                      Common::FrameCapture::Listed(present_index);
    }
    // Captures and screenshots are the guest's picture, rendered from the present blit's source
    // by the same blit and filter into an image of their own size: what the emulated path's
    // RequestScreenshot renders from the guest framebuffer (the layer through the scaling filter,
    // stretched into the layout's screen rectangle). The frame is not read: it has the window's
    // size, and DLAA, DLSS 5 neural rendering and the final FSR above change it, none of which
    // the emulated path's captures and screenshots contain.
    // `frame [N]` diag request: every present that claims a unit of the budget is saved too,
    // into the request's folder and independent of EDEN_NX2_FRAME_CAPTURE.
    const bool environment_due = !impl.capture_directory.empty() && capture_due;
    std::filesystem::path request_directory;
    Common::DiagRequest::Budget& frame_budget = Common::DiagRequest::FrameBudget();
    const bool request_due = frame_budget.Active() && frame_budget.Claim(request_directory);
    // The GPU command capture (video_core/gpu_capture.h) wants this present as its frame.png: the
    // emulated path serves it through RequestScreenshot at DefaultFrameLayout(1280, 720), so it
    // is the same guest picture as the file captures.
    const auto prepare_readback_source = [&] {
        if (through_passes && raw_copy && picture_source == *s.image) {
            PrepareImages({{&s, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL}});
            picture_source = RawPresentCopy(s, raw_format);
            picture_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        }
    };
    const bool gpu_capture_due = !gpu_capture_png.empty();
    if (environment_due || request_due || gpu_capture_due) {
        prepare_readback_source();
        // The nx2:vi2 path's capture: Layout::DefaultFrameLayout(1280, 720).
        const Layout::FramebufferLayout capture_layout = Layout::DefaultFrameLayout(1280, 720);
        PictureReadback readback =
            CreatePictureReadback(frame, capture_layout.width, capture_layout.height);
        RecordPictureReadback(
            readback, picture_source, picture_layout, crop_rect, present_filter,
            ScreenRect(capture_layout.screen, capture_layout.width, capture_layout.height));
        flight.capture = InFlight::Capture{
            .buffer = std::move(readback.buffer),
            .picture = readback.picture,
            .index = present_index,
            .to_environment_directory = environment_due,
            .request_directory = std::move(request_directory),
            .gpu_capture_png = std::move(gpu_capture_png),
        };
    }

    // A pending RendererBase::RequestScreenshot is answered from this present the same way, at
    // the request's layout, and converted into the request's buffer by the completion thread
    // (Impl::DeliverScreenshot). The request is claimed only once the host buffer exists, so a
    // failure here leaves it pending.
    VideoCore::RendererSettings* const requests =
        impl.screenshot_requests.load(std::memory_order_acquire);
    if (requests != nullptr && requests->screenshot_requested.load()) {
        const Layout::FramebufferLayout wanted = requests->screenshot_framebuffer_layout;
        if (wanted.width != 0 && wanted.height != 0) {
            PictureReadback readback = CreatePictureReadback(frame, wanted.width, wanted.height);
            if (requests->ClaimScreenshot()) {
                const Layout::FramebufferLayout& claimed = requests->screenshot_framebuffer_layout;
                if (claimed.width != wanted.width || claimed.height != wanted.height) {
                    // A new request replaced the one the readback was sized for: next present.
                    requests->ReleaseScreenshotClaim();
                } else {
                    prepare_readback_source();
                    RecordPictureReadback(
                        readback, picture_source, picture_layout, crop_rect, present_filter,
                        ScreenRect(claimed.screen, claimed.width, claimed.height));
                    flight.screenshot = InFlight::Screenshot{
                        .buffer = std::move(readback.buffer),
                        .picture = readback.picture,
                        .destination = static_cast<u8*>(requests->screenshot_bits),
                        .complete = requests->screenshot_complete_callback,
                        .settings = requests,
                    };
                }
            }
        }
    }
    return true;
}

Recorder::PictureReadback Recorder::CreatePictureReadback(const Frame& frame, u32 width,
                                                          u32 height) {
    // The frame's format in HDR (fp16) and for the 8-bit swapchain formats; any other frame
    // format is read back as B8G8R8A8 (the blit converts).
    const auto frame_format = static_cast<VkFormat>(frame.format);
    VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
    FramePixelFormat pixel_format = FramePixelFormat::Bgra8;
    switch (frame_format) {
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        format = frame_format;
        pixel_format = FramePixelFormat::Rgba16F;
        break;
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
        format = frame_format;
        break;
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
        format = frame_format;
        pixel_format = FramePixelFormat::Rgba8;
        break;
    default:
        break;
    }
    PictureReadback readback;
    readback.picture = Picture{
        .width = width,
        .height = height,
        .format = pixel_format,
        .encoding = frame.source_encoding,
        .paper_white_nits = static_cast<f32>(Common::Hdr::GetHostDisplay().paper_white_nits),
    };
    readback.image = impl.allocator->CreateImage({
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    readback.buffer = impl.CreateHostBuffer(
        static_cast<VkDeviceSize>(width) * height * BytesPerPixel(pixel_format),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::Download);
    return readback;
}

void Recorder::RecordPictureReadback(PictureReadback& readback, VkImage source_image,
                                     VkImageLayout source_layout, const PixelRect& source,
                                     VkFilter filter, const PixelRect& target) {
    const VkImageSubresourceRange range{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = *readback.image,
        .subresourceRange = range,
    };
    Cmd().PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                          barrier);
    Cmd().ClearColorImage(*readback.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VkClearColorValue{.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}, range);
    if (source_image != VK_NULL_HANDLE && target.width != 0 && target.height != 0 &&
        source.width != 0 && source.height != 0) {
        // The blit writes the cleared image: order it after the clear.
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        Cmd().PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                              barrier);
        const VkImageBlit blit{
            .srcSubresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets = {{static_cast<s32>(source.x), static_cast<s32>(source.y), 0},
                           {static_cast<s32>(source.x + source.width),
                            static_cast<s32>(source.y + source.height), 1}},
            .dstSubresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffsets = {{static_cast<s32>(target.x), static_cast<s32>(target.y), 0},
                           {static_cast<s32>(target.x + target.width),
                            static_cast<s32>(target.y + target.height), 1}},
        };
        Cmd().BlitImage(source_image, source_layout, *readback.image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, blit, filter);
    }
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    Cmd().PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                          barrier);
    const VkBufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {readback.picture.width, readback.picture.height, 1},
    };
    Cmd().CopyImageToBuffer(*readback.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *readback.buffer,
                            region);
    host_barrier_needed = true;
    flight.keepalive.push_back(std::make_shared<vk::Image>(std::move(readback.image)));
}

void Recorder::Finish(std::span<const VkSemaphore> binary_signals) {
    // Deferred report zero-fills first: the write-back copies below must see them.
    EndRenderCondition();
    EndDrawPass();
    FlushLinearWritebacks();
    WriteReadbacks();
    EndSegment();
    draw.EndSubmission();
    SubmitSegments(binary_signals);
    finished = true;
    if (impl.submit_batch.batching && binary_signals.empty() && !impl.submit_batch.entries.empty()) {
        impl.submit_batch.marks.push_back({&flight, *queue.timeline, segments.back().signal_value});
    } else {
        impl.MarkSubmitted(flight, *queue.timeline, segments.back().signal_value);
    }
}

void Recorder::SubmitSegments(std::span<const VkSemaphore> binary_signals) {
    // Chunks before the last one are full (only the last is still filled); an early submission
    // flushed them already.
    for (size_t i = flushed_arena; i < flight.arena.size(); ++i) {
        flight.arena[i]->buffer.Flush();
    }
    flushed_arena = flight.arena.empty() ? 0 : flight.arena.size() - 1;
    for (size_t i = flushed_staging; i < flight.staging.size(); ++i) {
        flight.staging[i].Flush();
    }
    flushed_staging = flight.staging.empty() ? 0 : flight.staging.size() - 1;

    constexpr size_t MaxSignals = MaxPresentSignals + 1; // + the queue's timeline semaphore
    ASSERT(binary_signals.size() <= MaxPresentSignals);
    const size_t first = submitted_segments;
    const size_t count = segments.size() - first;
    if (impl.submit_batch.batching && !Debug::Get().upload_log) {
        for (size_t k = 0; k < count; ++k) {
            const Segment& segment = segments[first + k];
            const bool last = first + k + 1 == segments.size();
            Context::Impl::SubmitBatch::Entry entry;
            entry.signals[0] = *queue.timeline;
            entry.signal_values[0] = segment.signal_value;
            u32 signal_count = 1;
            if (last) {
                for (const VkSemaphore semaphore : binary_signals) {
                    if (semaphore != VK_NULL_HANDLE) {
                        entry.signals[signal_count] = semaphore;
                        entry.signal_values[signal_count] = 0;
                        ++signal_count;
                    }
                }
            }
            entry.signal_count = signal_count;
            entry.wait_semaphores = segment.wait_semaphores;
            entry.wait_values = segment.wait_values;
            entry.wait_stages.assign(segment.wait_semaphores.size(), ALL_STAGES);
            entry.cmd = segment.raw;
            entry.flight_seq = flight.seq;
            entry.flight = &flight;
            entry.queue_timeline = *queue.timeline;
            entry.queue_signal_value = segment.signal_value;
            impl.submit_batch.entries.push_back(std::move(entry));
        }
        submitted_segments = segments.size();
        if (flight.gpu_prof && Debug::Timeline()) {
            flight.gpu_prof->NoteSubmitted(
                static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count()),
                queue_index);
        }
        if (!binary_signals.empty() || impl.submit_batch.entries.size() >= 64) {
            impl.FlushSubmitBatch();
        }
        return;
    }
    impl.FlushSubmitBatch();
    struct Info {
        VkTimelineSemaphoreSubmitInfo timeline;
        std::array<VkSemaphore, MaxSignals> signals;
        std::array<u64, MaxSignals> signal_values;
        std::vector<VkPipelineStageFlags> wait_stages;
    };
    std::vector<Info> infos(count);
    std::vector<VkSubmitInfo> submits(count);
    for (size_t k = 0; k < count; ++k) {
        const Segment& segment = segments[first + k];
        Info& info = infos[k];
        const bool last = first + k + 1 == segments.size();
        info.signals[0] = *queue.timeline;
        info.signal_values[0] = segment.signal_value;
        u32 signal_count = 1;
        if (last) {
            for (const VkSemaphore semaphore : binary_signals) {
                if (semaphore != VK_NULL_HANDLE) {
                    info.signals[signal_count] = semaphore;
                    info.signal_values[signal_count] = 0;
                    ++signal_count;
                }
            }
        }
        info.wait_stages.assign(segment.wait_semaphores.size(), ALL_STAGES);
        info.timeline = {
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreValueCount = static_cast<u32>(segment.wait_values.size()),
            .pWaitSemaphoreValues = segment.wait_values.data(),
            .signalSemaphoreValueCount = signal_count,
            .pSignalSemaphoreValues = info.signal_values.data(),
        };
        submits[k] = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &info.timeline,
            .waitSemaphoreCount = static_cast<u32>(segment.wait_semaphores.size()),
            .pWaitSemaphores = segment.wait_semaphores.data(),
            .pWaitDstStageMask = info.wait_stages.data(),
            .commandBufferCount = 1,
            .pCommandBuffers = &segment.raw,
            .signalSemaphoreCount = signal_count,
            .pSignalSemaphores = info.signals.data(),
        };
    }
    {
        const auto submit_begin = std::chrono::steady_clock::now();
        const Common::FrameStats::Scope submit_stall{Common::FrameStats::Cause::QueueSubmit};
        std::scoped_lock lock{impl.submit_mutex};
        const auto submit_locked = std::chrono::steady_clock::now();
        VkResult result = impl.queue.Submit(submits);
        if (const u64 from = impl.options.simulate_device_loss_from_seq;
            from != 0 && flight.seq >= from) {
            result = VK_ERROR_DEVICE_LOST;
        }
        if (Debug::Get().upload_log) {
            const auto submit_end = std::chrono::steady_clock::now();
            const double total =
                std::chrono::duration<double, std::milli>(submit_end - submit_begin).count();
            if (total > 10.0) {
                u64 staged = 0;
                for (const vk::Buffer& chunk : flight.staging) {
                    staged += chunk.Mapped().size();
                }
                LOG_INFO(
                    Render_Vulkan,
                    "[nvn] slow submit: present {} seq {} {:.1f} ms (lock wait {:.1f}), {} "
                    "segments, staging {:.1f} MB, storages created {}, chunk imports {}, "
                    "pools backed {}",
                    impl.stats.presents.load(std::memory_order_relaxed), flight.seq, total,
                    std::chrono::duration<double, std::milli>(submit_locked - submit_begin).count(),
                    count, staged / 1048576.0,
                    impl.stats.storages_created.load(std::memory_order_relaxed),
                    impl.chunk_imports.load(std::memory_order_relaxed),
                    impl.pools_backed_background.load(std::memory_order_relaxed));
            }
        }
        if (result == VK_ERROR_DEVICE_LOST) {
            impl.NoteDeviceLost("queue submit");
        }
        vk::Check(result);
    }
    if (count != 0) {
        flight.gpu_submitted_value =
            (std::max)(flight.gpu_submitted_value, segments.back().signal_value);
    }
    if (impl.mario_kart_world && count != 0) {
        impl.mkw_last_submit_semaphore = *queue.timeline;
        impl.mkw_last_submit_value = segments.back().signal_value;
    }
    if (impl.options.on_queue_submit) {
        // Force maximum clocks (renderer_force_max_clock): the renderer's TurboMode keeps the GPU
        // busy for 100 ms after each queue submission, and this queue is not the scheduler's.
        impl.options.on_queue_submit();
    }
    submitted_segments = segments.size();
    if (flight.gpu_prof && Debug::Timeline()) {
        flight.gpu_prof->NoteSubmitted(
            static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now().time_since_epoch())
                                 .count()),
            queue_index);
    }
}

void Recorder::FlushAndWait() {
    EndSegment();
    SubmitSegments({});
    impl.FlushSubmitBatch();
    const VkSemaphore semaphore = *queue.timeline;
    const u64 value = segments.back().signal_value;
    const VkSemaphoreWaitInfo wait_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext = nullptr,
        .flags = 0,
        .semaphoreCount = 1,
        .pSemaphores = &semaphore,
        .pValues = &value,
    };
    for (;;) {
        const VkResult result = impl.dld.vkWaitSemaphores(impl.logical, &wait_info, 20'000'000);
        if (result == VK_TIMEOUT) {
            continue;
        }
        vk::Check(result);
        break;
    }
    BeginSegment();
}

void Recorder::MaybeSubmitEarly() {
    if (impl.submit_batch.batching && !Debug::Get().upload_log) {
        return;
    }
    const Debug::Config& config = Debug::Get();
    if (config.early_interval_us == 0 || !impl.options.early_submit || finished || condition_open ||
        (!draw.PassOpen() && draw.HasInlineUpdates())) {
        return;
    }
    if (impl.mkw_coalesce_early_submit && early_pieces >= 2) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const u64 elapsed = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::microseconds>(now - piece_start).count());
    const u64 normal_us = early_pieces == 0 ? config.early_first_us : config.early_interval_us;
    if (impl.mkw_coalesce_early_submit) {
        if (!MkwEarlySubmitDue(early_pieces, elapsed, normal_us, config.early_pass_us,
                               draw.PassOpen(), draw.HasShadows())) {
            return;
        }
    } else {
        if (elapsed < normal_us) {
            return;
        }
        if (draw.PassOpen() && (config.early_pass_us == 0 || elapsed < config.early_pass_us)) {
            return; // a natural pass end comes soon: do not split the pass
        }
    }
    SubmitPiece(false);
    ++early_pieces;
    impl.stats.early_submits.fetch_add(1, std::memory_order_relaxed);
    if (Debug::Timeline()) {
        Debug::TimelineEvent("piece", flight.seq);
    }
}

void Recorder::SubmitPiece(bool flush_shadows) {
    const bool flush = flush_shadows || segment_signaled;
    EndSegment(flush);
    SubmitSegments({});
    BeginSegment(!flush);
    records_at_piece = records_replayed;
    piece_start = std::chrono::steady_clock::now();
}

/// A submission replayed as a stream: it runs up to the first WaitEvent the event model does not
/// satisfy, then suspends (what it recorded is on the GPU) until another stream or the guest CPU
/// signals the event.
class SubmissionItem final : public VC::StreamItem {
public:
    SubmissionItem(Context::Impl& impl_, CmdSubmit&& submit)
        : impl{impl_}, submission{std::move(submit.submission)}, flight{std::move(submit.flight)} {
        if (submission.uniform_snapshot) {
            impl.uniform_snapshots.Arrive(submission.seq,
                                          submission.uniform_snapshot->completed_seq);
        }
    }
    ~SubmissionItem() override {
        // The recorder (and the draw context's view of the snapshot) goes first.
        recorder.reset();
        if (submission.uniform_snapshot) {
            impl.uniform_snapshots.Leave(submission.seq,
                                         submission.uniform_snapshot->completed_seq);
        }
    }

    Step Advance(bool release) override;

    [[nodiscard]] u64 Order() const override {
        return submission.seq;
    }

private:
    Context::Impl& impl;
    VC::Submission submission;
    std::shared_ptr<InFlight> flight;
    /// Created at the first Advance that runs, so that the check of the fences it waits for keeps
    /// its meaning (nothing is allocated for a submission held back).
    std::unique_ptr<Recorder> recorder;
    size_t list_index{0};
    size_t word{0};         ///< where the replay of lists[list_index] goes on
    bool list_noted{false}; ///< the GPU capture has seen lists[list_index]
    bool held_counted{false};
    /// The wait the last suspension was counted for (event_waits_blocked counts a wait once).
    size_t blocked_list{~size_t{0}};
    size_t blocked_word{~size_t{0}};
};

VC::StreamItem::Step SubmissionItem::Advance(bool release) {
    if (impl.device_lost.load(std::memory_order_acquire)) {
        // The device is gone: complete the work without running it (a suspended submission
        // included), so no guest wait hangs.
        recorder.reset();
        impl.stats.dropped_after_device_loss.fetch_add(1, std::memory_order_relaxed);
        impl.FailFlight(*flight);
        return {.done = true, .progressed = true};
    }
    if (flight->failed) {
        // An earlier batched piece of this parked submission failed to submit: abort rather than
        // resuming past GPU commands that never ran.
        recorder.reset();
        impl.FailFlight(*flight);
        return {.done = true, .progressed = true};
    }
    const Common::FrameStats::Scope stall{Common::FrameStats::Cause::ReplayExecute};
    Step step;
    bool failed = false;
    try {
        bool released_here = false;
        if (!recorder) {
            u64 missing = 0;
            if (!impl.DependenciesKnown(submission, &missing)) {
                if (!held_counted) {
                    held_counted = true;
                    impl.stats.held_back_submissions.fetch_add(1, std::memory_order_relaxed);
                }
                if (!release) {
                    return step;
                }
                release = false;
                released_here = true;
                impl.stats.parked_timeouts.fetch_add(1, std::memory_order_relaxed);
                if (impl.parked_timeout_logged.insert(missing).second) {
                    LOG_ERROR(Render_Vulkan,
                              "[nvn] submission {} waited {} ms for sync {:#x}, which nobody "
                              "submitted a fence for; running it without the wait",
                              submission.seq, impl.options.parked_timeout.count(), missing);
                }
            }
            impl.replay_stage.store("submission: record", std::memory_order_relaxed);
            if (Debug::Timeline()) {
                Debug::TimelineEvent("rec_begin", submission.seq,
                                     fmt::format("q={}", submission.queue));
            }
            recorder = std::make_unique<Recorder>(impl, submission.queue, *flight);
            recorder->BindSubmission(submission);
        }
        impl.replay_seq.store(submission.seq, std::memory_order_relaxed);
        const u64 replayed_before = recorder->RecordsReplayed();
        while (list_index < submission.lists.size()) {
            const std::vector<u64>& list = submission.lists[list_index];
            recorder->SetLookahead(submission, list_index);
            if (!list_noted) {
                list_noted = true;
            }
            size_t stop = 0;
            if (!recorder->RecordList(list, word, stop, release)) {
                word = stop;
                recorder->Suspend();
                if (blocked_list != list_index || blocked_word != word) {
                    blocked_list = list_index;
                    blocked_word = word;
                    impl.stats.event_waits_blocked.fetch_add(1, std::memory_order_relaxed);
                }
                impl.replay_stage.store("submission: waiting for an event",
                                        std::memory_order_relaxed);
                step.progressed = released_here || recorder->RecordsReplayed() != replayed_before;
                return step;
            }
            word = 0;
            list_noted = false;
            ++list_index;
        }
        impl.replay_stage.store("submission: finish", std::memory_order_relaxed);
        recorder->Finish({});
        impl.replay_stage.store("submission: done", std::memory_order_relaxed);
        if (Debug::Timeline()) {
            Debug::TimelineEvent("rec_end", submission.seq);
        }
        recorder.reset();
    } catch (const std::exception& exception) {
        impl.stats.ops_failed.fetch_add(1, std::memory_order_relaxed);
        LOG_ERROR(Render_Vulkan, "[nvn] submission {} failed: {}", submission.seq,
                  exception.what());
        impl.NoteIfDeviceLost("a submission");
        failed = true;
    }
    if (failed) {
        // Outside the handler: the recorder's abort catches (and swallows) exceptions of its own,
        // which must not run while this one is being handled.
        recorder.reset();
        impl.FailFlight(*flight);
    }
    step.done = true;
    step.progressed = true;
    return step;
}

/// A present: ordered behind the earlier work of its queue, replayed whole.
class PresentItem final : public VC::StreamItem {
public:
    PresentItem(Context::Impl& impl_, CmdPresent&& present)
        : impl{impl_}, request{present.request}, flight{std::move(present.flight)} {}

    Step Advance(bool) override {
        try {
            impl.ExecutePresent(request, *flight);
        } catch (const std::exception& exception) {
            impl.stats.ops_failed.fetch_add(1, std::memory_order_relaxed);
            LOG_ERROR(Render_Vulkan, "[nvn] present {} failed: {}", request.seq, exception.what());
            impl.FailFlight(*flight);
        }
        return {.done = true, .progressed = true};
    }

    [[nodiscard]] u64 Order() const override {
        return request.seq;
    }

private:
    Context::Impl& impl;
    VC::PresentRequest request;
    std::shared_ptr<InFlight> flight;
};

} // Anonymous namespace

void Context::Impl::FlushSubmitBatch() {
    if (submit_batch.entries.empty() && submit_batch.marks.empty()) {
        return;
    }
    auto entries = std::move(submit_batch.entries);
    auto marks = std::move(submit_batch.marks);
    submit_batch.entries.clear();
    submit_batch.marks.clear();
    try {
        if (!entries.empty()) {
            std::vector<VkCommandBuffer> all_cmds;
            all_cmds.reserve(entries.size());
            std::vector<VkSubmitInfo> submits;
            submits.reserve(entries.size());
            u64 max_seq = 0;
            size_t first = 0;
            while (first < entries.size()) {
                size_t last = first;
                max_seq = (std::max)(max_seq, entries[first].flight_seq);
                all_cmds.push_back(entries[first].cmd);
                while (last + 1 < entries.size()) {
                    const SubmitBatch::Entry& prev = entries[last];
                    const SubmitBatch::Entry& next = entries[last + 1];
                    if (prev.queue_timeline == next.queue_timeline && prev.signal_count == 1 &&
                        next.wait_semaphores.empty() &&
                        next.queue_signal_value >= prev.queue_signal_value) {
                        ++last;
                        max_seq = (std::max)(max_seq, next.flight_seq);
                        all_cmds.push_back(next.cmd);
                    } else {
                        break;
                    }
                }
                SubmitBatch::Entry& head = entries[first];
                const SubmitBatch::Entry& tail = entries[last];
                head.timeline = {
                    .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                    .pNext = nullptr,
                    .waitSemaphoreValueCount = static_cast<u32>(head.wait_values.size()),
                    .pWaitSemaphoreValues = head.wait_values.data(),
                    .signalSemaphoreValueCount = tail.signal_count,
                    .pSignalSemaphoreValues = tail.signal_values.data(),
                };
                submits.push_back(VkSubmitInfo{
                    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .pNext = &head.timeline,
                    .waitSemaphoreCount = static_cast<u32>(head.wait_semaphores.size()),
                    .pWaitSemaphores = head.wait_semaphores.data(),
                    .pWaitDstStageMask = head.wait_stages.data(),
                    .commandBufferCount = static_cast<u32>(last - first + 1),
                    .pCommandBuffers = &all_cmds[first],
                    .signalSemaphoreCount = tail.signal_count,
                    .pSignalSemaphores = tail.signals.data(),
                });
                first = last + 1;
            }
            {
                const Common::FrameStats::Scope submit_stall{
                    Common::FrameStats::Cause::QueueSubmit};
                std::scoped_lock lock{submit_mutex};
                VkResult result = queue.Submit(submits);
                if (const u64 from = options.simulate_device_loss_from_seq;
                    from != 0 && max_seq >= from) {
                    result = VK_ERROR_DEVICE_LOST;
                }
                if (result == VK_ERROR_DEVICE_LOST) {
                    NoteDeviceLost("queue submit");
                }
                vk::Check(result);
            }
            for (const SubmitBatch::Entry& entry : entries) {
                if (entry.flight) {
                    entry.flight->gpu_submitted_value =
                        (std::max)(entry.flight->gpu_submitted_value, entry.queue_signal_value);
                }
            }
            if (mario_kart_world) {
                mkw_last_submit_semaphore = entries.back().queue_timeline;
                mkw_last_submit_value = entries.back().queue_signal_value;
            }
            if (options.on_queue_submit) {
                options.on_queue_submit();
            }
        }
    } catch (...) {
        for (const SubmitBatch::Entry& entry : entries) {
            if (entry.flight) {
                entry.flight->failed = true;
            }
        }
        for (const SubmitBatch::Mark& mark : marks) {
            if (mark.flight) {
                FailFlight(*mark.flight);
            }
        }
        throw;
    }
    if (!marks.empty()) {
        {
            std::scoped_lock lock{state_mutex};
            for (const SubmitBatch::Mark& mark : marks) {
                if (mark.flight) {
                    mark.flight->semaphore = mark.semaphore;
                    mark.flight->value = mark.value;
                    mark.flight->submitted = true;
                }
            }
        }
        complete_cv.notify_all();
    }
}

bool Context::Impl::DependenciesKnown(const VC::Submission& submission, u64* missing) const {
    std::unordered_set<u64> local;
    for (const std::vector<u64>& list : submission.lists) {
        size_t index = 0;
        while (index < list.size()) {
            const auto* header = reinterpret_cast<const VC::RecordHeader*>(&list[index]);
            const size_t words = header->size / sizeof(u64);
            if (header->size < sizeof(VC::RecordHeader) || header->size % 8 != 0 ||
                index + words > list.size() || header->op == static_cast<u16>(VC::Op::End)) {
                break;
            }
            if (header->op == static_cast<u16>(VC::Op::FenceSync) &&
                header->size >= sizeof(VC::RecordHeader) + sizeof(VC::FenceSyncPayload)) {
                local.insert(
                    reinterpret_cast<const VC::FenceSyncPayload*>(header + 1)->sync_serial);
            } else if (header->op == static_cast<u16>(VC::Op::WaitSync) &&
                       header->size >= sizeof(VC::RecordHeader) + sizeof(VC::WaitSyncPayload)) {
                const u64 serial =
                    reinterpret_cast<const VC::WaitSyncPayload*>(header + 1)->sync_serial;
                if (!syncs.contains(serial) && !local.contains(serial)) {
                    if (missing) {
                        *missing = serial;
                    }
                    return false;
                }
            }
            index += words;
        }
    }
    return true;
}

void Context::Impl::PumpStreams(bool release_old) {
    streams.Pump(release_old, VC::StreamScheduler::Clock::now());
    streams_pending.store(streams.Pending(), std::memory_order_relaxed);
}

void Context::Impl::ProcessSubmit(CmdSubmit&& submit) {
    stats.submissions.fetch_add(1, std::memory_order_relaxed);
    for (const u64 serial : submit.submission.retained_serials) {
        NoteUse(serial, submit.submission.seq);
    }
    // Held-back work must not outlive its timeout just because the guest keeps submitting.
    PumpStreams(true);
    const u32 queue_index = submit.submission.queue;
    streams.Push(queue_index, std::make_unique<SubmissionItem>(*this, std::move(submit)),
                 VC::StreamScheduler::Clock::now());
    PumpStreams(false);
}

void Context::Impl::ProcessPresent(CmdPresent&& present) {
    const u64 presented = stats.presents.fetch_add(1, std::memory_order_relaxed);
    if (presented % 900 == 899 && Debug::Get().storage_census) {
        LogStorageCensus();
    }
    if ((bc_timing || smart_bc.enabled) && presented % 60 == 59) {
        LogBcTimeWindow(presented);
    }
    // Textures die and storages age without a new image being made: the trim looks at them here.
    TrimStorageCache(true, present.request.seq);
    NoteUse(present.request.texture_serial, present.request.seq);
    PumpStreams(true);
    // A present is ordered after everything its queue submitted before it.
    const u32 queue_index = present.request.queue;
    streams.Push(queue_index, std::make_unique<PresentItem>(*this, std::move(present)),
                 VC::StreamScheduler::Clock::now());
    PumpStreams(false);
}

void Context::Impl::NotifyEventCpuNow(const CmdEventCpu& command) {
    if (Debug::Timeline()) {
        Debug::TimelineEvent("eventcpu", 0, fmt::format("addr={:#x}", command.address));
    }
    events.CpuChange(command.address, command.change, command.value);
    uniform_snapshots.NoteCpuEvent(command.address);
    // A CPU signal can unblock a suspended stream.
    PumpStreams(false);
}

void Context::Impl::ExecutePresent(const VC::PresentRequest& request, InFlight& flight) {
    if (device_lost.load(std::memory_order_acquire)) {
        stats.dropped_after_device_loss.fetch_add(1, std::memory_order_relaxed);
        FailFlight(flight);
        return;
    }
    const Common::FrameStats::Scope stall{Common::FrameStats::Cause::PresentBlit};
    Texture* texture = nullptr;
    if (const auto it = textures.find(request.texture_serial); it != textures.end()) {
        texture = it->second.get();
    }
    const bool displayable = texture && texture->storage && present_target;
    Frame* frame = nullptr;
    /// Frames FSR3 / Lossless Scaling generated, presented before `frame`.
    std::vector<Frame*> generated;
    bool work_submitted = false;
    // GPU command capture: one frame boundary per present request, on the replay thread that
    // also recorded the frame's NVN lists and draw bindings. The screenshot of this present (if
    // a capture ends here) is read back by RecordPresent.
    std::filesystem::path gpu_capture_png;
    if (VideoCommon::ShaderLog::Enabled()) {
        // Shader log: one presented frame of the HLE path (the emulated Composite is not used).
        VideoCommon::ShaderLog::NoteFrame();
    }
    try {
        replay_seq.store(request.seq, std::memory_order_relaxed);
        replay_stage.store("present: acquire frame", std::memory_order_relaxed);
        if (Debug::Timeline()) {
            Debug::TimelineEvent("pres_begin", request.seq);
        }
        if (displayable) {
            const Common::FrameStats::Scope acquire{Common::FrameStats::Cause::PresentAcquire};
            frame = present_target->AcquireFrame();
        }
        if (Debug::Timeline()) {
            Debug::TimelineEvent("pres_acquired", request.seq);
        }
        Recorder recorder(*this, request.queue, flight, true);
        recorder.gpu_capture_png = std::move(gpu_capture_png);
        replay_stage.store("present: record", std::memory_order_relaxed);
        if (frame && !recorder.RecordPresent(*texture, *frame, generated, request.crop)) {
            present_target->ReleaseFrame(std::exchange(frame, nullptr));
        }
        replay_stage.store("present: finish", std::memory_order_relaxed);
        {
            // Streamline's render-submit markers bracket the queue submission of the frame.
            const bool mark_submit = present_frame_gen && frame;
            if (mark_submit) {
                present_frame_gen->RenderSubmitStart();
            }
            SCOPE_EXIT {
                if (mark_submit) {
                    present_frame_gen->RenderSubmitEnd();
                }
            };
            // Every frame this submission displays is signalled by it (GenerateFrames keeps the
            // generated frames below MaxPresentSignals).
            std::array<VkSemaphore, MaxPresentSignals> signals;
            size_t signal_count = 0;
            for (Frame* extra : generated) {
                signals[signal_count++] = *extra->render_ready;
            }
            if (frame) {
                signals[signal_count++] = *frame->render_ready;
            }
            recorder.Finish(std::span<const VkSemaphore>{signals.data(), signal_count});
        }
        FlushSubmitBatch();
        work_submitted = true;
        replay_stage.store("present: hand over", std::memory_order_relaxed);
        if (frame) {
            const Common::FrameStats::Scope hand_over{Common::FrameStats::Cause::PresentHandOver};
            // The generated frames lie between the previous picture and this one: they come first.
            for (Frame* extra : generated) {
                present_target->PresentFrame(extra);
            }
            present_target->PresentFrame(frame);
        }
        if (draw_resources) {
            draw_resources->NoteAsyncSkips(presents_recorded);
            draw_resources->CheckSink(presents_recorded);
            draw_resources->MaybeRefreshShaders();
            if (VideoCommon::Nvn::TimestampTraceEnabled()) {
                draw_resources->TracePresent(presents_recorded);
            }
        }
        replay_stage.store("present: done", std::memory_order_relaxed);
        if (Debug::Timeline()) {
            Debug::TimelineEvent("pres_end", request.seq);
        }
    } catch (const std::exception& exception) {
        if (!work_submitted) {
            // Nothing uses the frames: give them back, or the replay runs out of frames after
            // `image_count` failures and blocks in AcquireFrame forever.
            if (frame != nullptr) {
                present_target->ReleaseFrame(frame);
            }
            for (Frame* extra : generated) {
                present_target->ReleaseFrame(extra);
            }
        }
        stats.ops_failed.fetch_add(1, std::memory_order_relaxed);
        LOG_ERROR(Render_Vulkan, "[nvn] present {} failed: {}", request.seq, exception.what());
        NoteIfDeviceLost("a present");
        FailFlight(flight);
    }
}

} // namespace Vulkan::Nvn
