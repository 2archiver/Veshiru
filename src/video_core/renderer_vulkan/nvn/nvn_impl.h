// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Internal types of the NVN Vulkan replay. Everything here belongs to nvn_context.cpp,
// nvn_resources.cpp and nvn_exec.cpp; nothing outside renderer_vulkan/nvn includes it.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include "common/hdr.h"
#include "video_core/game_frame_size.h"
#include "video_core/nvn/record.h"
#include "video_core/nvn/queue_sync.h"
#include "video_core/nvn/sink.h"
#include "video_core/pathdiff.h"
#include "video_core/upscaler.h"
#include "video_core/renderer_vulkan/nvn/nvn_debug.h"
#include "video_core/renderer_vulkan/nvn/nvn_alias_gpu.h"
#include "video_core/renderer_vulkan/nvn/nvn_astc_gpu.h"
#include "video_core/renderer_vulkan/nvn/nvn_bc_encode.h"
#include "video_core/renderer_vulkan/nvn/nvn_mesh_indirect.h"
#include "video_core/renderer_vulkan/nvn/nvn_quad_indirect.h"
#include "video_core/renderer_vulkan/nvn/nvn_untile_gpu.h"
#include "video_core/renderer_vulkan/nvn/nvn_vertex_stage_gpu.h"
#include "video_core/renderer_vulkan/nvn/nvn_diag_textures.h"
#include "video_core/renderer_vulkan/nvn/nvn_context.h"
#include "video_core/renderer_vulkan/nvn/nvn_counters.h"
#include "video_core/renderer_vulkan/nvn/nvn_costsplit.h"
#include "video_core/renderer_vulkan/nvn/nvn_format.h"
#include "video_core/renderer_vulkan/nvn/nvn_gpu_writes.h"
#include "video_core/renderer_vulkan/nvn/nvn_gpu_profile.h"
#include "video_core/renderer_vulkan/nvn/nvn_frame_gen.h"
#include "video_core/renderer_vulkan/nvn/nvn_host_import.h"
#include "video_core/renderer_vulkan/nvn/nvn_pool_mirror.h"
#include "video_core/renderer_vulkan/nvn/nvn_screenshot.h"
#include "video_core/renderer_vulkan/nvn/nvn_rescale.h"
#include "video_core/renderer_vulkan/nvn/nvn_texture_layout.h"
#include "video_core/renderer_vulkan/nvn/nvn_uniform_snapshot.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/buffer_cache/raw_coverage.h"
#include "video_core/renderer_vulkan/vk_raw_coverage.h"

namespace Vulkan {
class FSR;
class FSR3;
class SGSR;
class WindowAdaptPass;
namespace Dlss {
class Loader;
}
}


namespace Vulkan::Nvn {

struct ArenaChunk;
struct ShadowChunk;
class WriteBackMerge;
class SparseBacking;
class SparsePagePool;
struct DrawResources;
class DrawContext;
class PresentChain;

namespace VC = VideoCommon::Nvn;

/// NVNmemoryPoolFlags.
namespace PoolFlag {
constexpr u32 CpuNoAccess = 1u << 0;
constexpr u32 CpuUncached = 1u << 1;
constexpr u32 CpuCached = 1u << 2;
constexpr u32 GpuNoAccess = 1u << 3;
constexpr u32 ShaderCode = 1u << 6;
constexpr u32 Physical = 1u << 8;
constexpr u32 Virtual = 1u << 9;
} // namespace PoolFlag

/// A guest memory pool as the GPU sees it.
struct Pool {
    enum class Backing {
        None,        ///< Unsupported pool (allocation failure); ops on it are dropped.
        Imported,    ///< Guest memory imported with VK_EXT_external_memory_host.
        Mirror,      ///< Device-local mirror of CPU-accessible guest memory.
        DeviceLocal, ///< GPU-only memory (CPU_NO_ACCESS).
        Virtual,     ///< Virtual pool: a mapping table onto physical pools, no buffer of its own.
    };
    struct Dirty {
        u64 begin;
        u64 end;
        u64 seq;
    };
    /// One MapVirtual range of a virtual pool.
    struct VirtualRun {
        u64 size;
        u64 physical_serial;
        u64 physical_offset;
    };

    u64 serial{};
    u64 address{};
    u64 size{};
    u32 flags{};
    Backing backing{Backing::None};
    /// Created but not backed yet: the Vulkan buffer is made by the backing thread soon after
    /// creation, or by EnsureBacking at the first use, whichever comes first (under
    /// Impl::backing_mutex; the other fields are written before this is cleared with release).
    std::atomic<bool> pending{false};
    /// The pool was destroyed: backing it is pointless (the backing thread skips it).
    std::atomic<bool> released{false};
    /// A GPU-only pool whose buffer exists but whose guest contents are not uploaded yet (done at
    /// the first use, as late as possible: the game may still fill the storage after creating
    /// the pool). Replay thread only once `pending` is false.
    bool upload_pending{false};
    /// The chunk an imported pool buffer lives in (declared before `imported`: the buffer goes
    /// first).
    std::shared_ptr<ImportedChunk> chunk;
    ImportedBuffer imported;
    vk::Buffer buffer;
    /// Contiguous host view of the guest storage (Imported and Mirror pools), null otherwise.
    u8* host{};
    /// VkBuffer device address of the pool's buffer (raw global memory); 0 until first needed.
    u64 device_address{};
    /// Last submission seq that retained this pool (avoids hashing into `Impl::last_use`).
    u64 last_use_seq{0};
    /// GuestMemory::Track registration over the whole pool for Recorder::ResolveOverlaps: the pages
    /// of a storage are armed when the GPU takes it over (NoteGpuWrite), so a later CPU write to
    /// them shows. 0: not registered (yet, or tracking unavailable: `takeover_track_tried`).
    u64 takeover_track{};
    bool takeover_track_tried{false};
    /// Ranges a not yet completed submission writes on the mirror; the mirror is authoritative
    /// there until the write-back at completion. Guarded by dirty_mutex.
    std::mutex dirty_mutex;
    std::vector<Dirty> gpu_dirty;
    /// Write-backs that reached guest memory, {offset, end}: added at completion, taken by
    /// Impl::TakePoolCpuWrites (texture trust). Guarded by dirty_mutex.
    std::vector<std::pair<u64, u64>> landed_writebacks;
    /// Mirror pools: what the mirror holds, per MIRROR_PAGE bytes of the pool. Replay thread
    /// only once `pending` is false (sized before that).
    struct MirrorPage {
        u64 hash{};     ///< CityHash64 of the guest page the mirror was last made equal to
        u32 checked{};  ///< submission (seq + 1) that last found the guest page unchanged
        bool valid{};   ///< the mirror holds a guest page at all (else the first use uploads it)
        /// Hashed while `mirror_track` watched the page, with no GPU write of it pending, and
        /// nothing wrote it since that the tracker does not see: unchanged unless the tracker
        /// reports a write, so it is not hashed again (Recorder::MirrorTakeCpuWrites).
        bool trusted{};
        u64 raw_epoch{}; ///< MKW: raw completion epoch of this page's last comparison
    };
    static constexpr u64 MIRROR_PAGE = 4096;
    std::vector<MirrorPage> mirror_pages;
    /// Mirror pools: GuestMemory::Track registration over the whole pool, every page armed, so
    /// that unchanged pages need no hash (Impl::mirror_hash_tracked). 0: not registered (yet, or
    /// tracking unavailable: `mirror_track_tried`). Replay thread.
    u64 mirror_track{};
    bool mirror_track_tried{false};
    bool raw_written_through{false};
    u64 split_raw_epoch_unaliased_seen{0};
    std::shared_ptr<const std::atomic<u64>> mirror_write_epoch;
    u64 mirror_epoch_clean{}; ///< `mirror_write_epoch` before the last TakeWritten
    /// Per MIRROR_PAGE of the pool: `cpu_write_gen` of the last Impl::TakePoolCpuWrites that found
    /// the page written (0: not since the registration). Sized when `mirror_track` registers.
    /// Lets a CPU-sourced texture skip its hash (Storage::cpu_trusted_gen). Replay thread.
    std::vector<u32> page_write_gen;
    u32 cpu_write_gen{1};
    /// Split pools: `cpu_write_gen` when a completed raw store last may have changed any guest
    /// byte of the pool past the tracker (0: never). Names no page, so it is kept apart from
    /// `page_write_gen`: a texture full-compared before it is checked as an untracked one until
    /// its next full compare (Storage::cpu_full_gen).
    u32 raw_write_gen{0};
    /// Imported pools: the persistent device-local copy that storage bindings and indirect
    /// arguments use (EDEN_NX2_NVN_POOL_MIRROR, nvn_pool_mirror.h; not Backing::Mirror). Null until
    /// the first use that wants one. Replay thread only.
    std::unique_ptr<PoolMirror> vram_mirror;
    /// A VRAM mirror was refused (budget, allocation, no CPU write tracking): not tried again.
    bool vram_mirror_refused{false};
    /// Virtual pools: runs by virtual offset, never overlapping.
    std::map<u64, VirtualRun> virtual_runs;
    /// Mirror pools whose guest storage is several host runs (knob raw_split_import): each run
    /// imported on its own, sorted by offset, covering the pool. Draws keep binding the mirror;
    /// raw global memory reaches the guest bytes through these. Written under
    /// Impl::backing_mutex before `pending` is cleared, never changed after.
    struct SplitRun {
        u64 offset{};
        u64 size{};
        ImportedBuffer imported;
        u64 device_address{}; ///< 0 until the raw global table first needs it
    };
    std::vector<SplitRun> split_runs;
    /// Impl::split_raw_epoch when this split pool's mirror pages were last made untrusted (a
    /// completed raw store may have changed the guest bytes untracked). Replay thread.
    u64 split_raw_epoch_seen{};

    [[nodiscard]] VkBuffer Handle() const noexcept {
        return backing == Backing::Imported ? *imported.buffer : *buffer;
    }
    [[nodiscard]] bool IsVirtual() const noexcept {
        return backing == Backing::Virtual;
    }
};

/// Where a storage's bytes came from last, which decides how it is kept coherent.
enum class StorageOrigin : u8 {
    None, ///< Never uploaded or written.
    Cpu,  ///< Holds the guest bytes of `guest_hash`; the replay checks the guest memory for changes.
    Gpu,  ///< The GPU wrote it last; guest memory is not looked at again.
};

/// One VkImage. Textures at the same (pool, offset, shape) share it, and it outlives the textures
/// that used it (Context::Impl::storage_cache): DK re-creates its transient render targets over the
/// same pool bytes every frame, which would otherwise be a vkCreateImage per texture and frame.
///
/// Contract for code that records GPU work (draws, dispatches, copies): the image does not exist
/// until the first `Recorder::EnsureContents`. Call it before the GPU reads or writes the storage
/// (`will_overwrite` = the whole texture is overwritten without being read). It creates the image,
/// uploads or refreshes the contents from guest memory, and copies in what overlapping textures
/// wrote. Call `NoteGpuWrite` when the GPU writes the storage (`PrepareImages` does it for the
/// layouts that write).
/// Presents to wait before looking at an unchanged CPU-sourced storage again. The look is cheap (a
/// 16 KiB signature): every present for the first uses (a video frame the CPU writes changes every
/// frame or two), then less and less often up to every 30 presents (about half a second). The
/// full hash of the bytes is compared every FULL_VERIFY_PRESENTS and when the game flushed the
/// pool.
inline constexpr u64 FULL_VERIFY_PRESENTS = 90;
/// Bytes of scheduled full compares (not the every-check ones, see DetectGuestChange) one present
/// may start; the rest wait for a later check.
inline constexpr u64 FULL_HASH_BYTES_PER_PRESENT = u64{4} << 20;
inline u64 CheckBackoff(u32 unchanged_checks) {
    return unchanged_checks < 8 ? 0 : (std::min)(u64{1} << (unchanged_checks - 8), u64{30});
}

/// Image bytes the storage cache may hold before TrimStorageCache drops storages nobody uses
/// (ContextOptions::storage_cache_budget_bytes overrides it). DK's pools hold about 2 GB of
/// textures; 6 GB leaves room for decoded stand-ins.
inline constexpr u64 DEFAULT_STORAGE_CACHE_BUDGET = u64{6} << 30;
/// Presents an idle render target that guest bytes or newer storages can rebuild stays cached
/// under the budget (ContextOptions::storage_trim_age_presents overrides it; 0 never ages out).
inline constexpr u64 DEFAULT_STORAGE_TRIM_AGE_PRESENTS = 0;

/// True when the half-open ranges, sorted by start, leave no byte of [begin, end) uncovered.
[[nodiscard]] inline bool RangesCover(std::span<const std::pair<u64, u64>> ranges, u64 begin,
                                      u64 end) {
    u64 reached = begin;
    for (const auto& [from, to] : ranges) {
        if (reached >= end) {
            break;
        }
        if (from > reached) {
            return false;
        }
        reached = (std::max)(reached, to);
    }
    return reached >= end;
}

struct Storage : std::enable_shared_from_this<Storage> {
    vk::Image image;
    /// Counts the images made for the storage (EnsureImage): what was derived from `image` (the
    /// bindless table's entries) is stale when it moved.
    u32 image_gen{0};
    HostFormat host;
    TextureLayoutDesc desc;
    TextureLayout layout;
    VkImageType type{VK_IMAGE_TYPE_2D};
    VkExtent3D extent{};
    VkSampleCountFlagBits samples{VK_SAMPLE_COUNT_1_BIT};
    VkImageAspectFlags aspect{VK_IMAGE_ASPECT_COLOR_BIT};
    u32 levels{1};
    u32 layers{1};
    u64 pool_serial{};
    u64 pool_offset{};
    VkImageLayout current_layout{VK_IMAGE_LAYOUT_UNDEFINED};
    /// The image holds meaningful data: uploaded from the guest pool, or written by the GPU.
    bool contents_valid{false};

    // Creation parameters (the image is created on first use).
    VkImageCreateFlags create_flags{};
    VkImageUsageFlags usage{};
    bool image_failed{false};
    u64 vram_bytes{}; ///< Estimated size of the image once created.

    // Coherency with guest memory and with overlapping storages.
    StorageOrigin origin{StorageOrigin::None};
    /// Clock value of the last GPU write / of the last time the contents were made current.
    u64 gpu_write_gen{0};
    u64 sync_gen{0};
    /// DrawContext::render_target_epoch of the last feedback copy of this storage.
    u64 feedback_epoch{0};
    /// Seq of the last submission that recorded a use of this storage (Recorder::NoteTouched).
    u64 touched_seq{~u64{0}};
    /// PoolIndex::write_gen seen at the last overlap resolution.
    u64 pool_gen_seen{~u64{0}};
    /// CPU origin: hash of the guest bytes behind the contents, the check schedule. A check
    /// compares a cheap signature of ~16 KiB spread over the storage (any change that touches a
    /// sampled line shows at once, which is what a decoded video frame does); the full hash is
    /// compared every `next_full_tick` and when the game flushed the pool.
    u64 guest_hash{0};
    u64 sample_hash{0};
    /// SampleGuestBytes of the guest bytes under the storage when the GPU took it over (the first
    /// GPU write after an upload or creation); valid only for CPU-readable pools. The GPU never
    /// writes guest memory here, so a different signature later means the CPU rewrote the bytes
    /// after that write and this storage's pixels are older than them (Recorder::ResolveOverlaps;
    /// a fallback for pools without CPU write tracking, see `gpu_takeover_armed`).
    u64 gpu_guest_sig{0};
    bool gpu_guest_sig_valid{false};
    /// The storage's pages were armed in its pool's `takeover_track` when the GPU took it over:
    /// GuestMemory::AnyWritten over them tells whether the CPU wrote them since (any bytes, also
    /// the same ones again, which the signature cannot see).
    bool gpu_takeover_armed{false};
    u32 unchanged_checks{0};
    u64 next_check_tick{0};
    u64 next_full_tick{0};
    u64 checked_seq{~u64{0}};
    bool force_check{false};
    /// GPU origin: `sample_hash` is the guest signature taken at the first check after the last
    /// GPU write (the baseline a CPU rewrite is compared with).
    bool watch_valid{false};
    /// GPU origin on a linear storage in a CPU-readable pool: `seq` of the last submission that
    /// queued a write-back of the GPU's linear texels to guest memory (0: none, or settled).
    u64 linear_writeback_seq{0};
    /// The CPU changed the bytes behind this storage at least once (checked in full from then on).
    bool dynamic{false};
    /// CPU origin: the pool's `cpu_write_gen` when a full hash last found the guest bytes unchanged
    /// while the page tracker watched the pool (0: never). While no page under the storage has a
    /// later `page_write_gen`, the CPU did not write it and a check needs no hash
    /// (Impl::tex_hash_tracked).
    u64 cpu_trusted_pool{0};
    u32 cpu_trusted_gen{0};
    /// The pool's `cpu_write_gen` at the full hash behind `cpu_trusted_gen` (the page compares
    /// move only that one on). A later Pool::raw_write_gen: bytes no page write names may differ.
    u32 cpu_full_gen{0};
    /// CityHash64 of the storage's part of each pool page (Pool::MIRROR_PAGE grid, first page at
    /// pool_offset / MIRROR_PAGE), taken with the full hash that set `cpu_trusted_gen`: a check
    /// after a tracked write rehashes only the written pages. Empty: not taken.
    std::vector<u64> page_hashes;
    /// A tracked write once came after the trust: only such storages keep page hashes.
    bool wants_page_hashes{false};
    /// The guest wrote the storage itself (WriteTexels): upload again whatever the origin.
    bool force_upload{false};
    u64 last_used_seq{0};
    /// Context::Impl::stats.presents at the last use (the trim's age clock).
    u64 last_used_present{0};
    /// PoolIndex::write_gen at which TrimStorageCache found this (GPU-written) storage not
    /// superseded: it is not tested again before another GPU write in its pool (the test scans
    /// every storage over its bytes).
    u64 trim_checked_gen{~u64{0}};
    /// Host pointer to the guest bytes (null when not contiguous or not resolved yet).
    const u8* guest_pointer{};
    bool guest_pointer_resolved{false};
    bool cached{false}; ///< Registered in Context::Impl::storage_cache / storage_index.
    /// Replay thread: the PoolIndex of `pool_serial` (storage_index node, found once) and the
    /// Context::Impl::storage_index_epoch it was found at.
    const void* pool_index{};
    u64 pool_index_epoch{0};
    /// The image has STORAGE usage and extended usage so that `AstcGpuDecoder` can write its
    /// levels (ASTC stand-ins only). Whether it is used is decided per upload.
    bool astc_gpu{false};
    /// A BC stand-in of the smart selection (nvn_format.h SmartBc): its format is chosen when the
    /// image is created (EnsureImage classifies the guest's ASTC bytes; BC3 until then), and each
    /// later whole upload classifies its bytes again (Recorder::RefitStandin).
    bool smart_standin{false};
    /// Kept at BC3 for good: a copy or partial upload into or out of it, or a re-upload whose
    /// bytes did not fit the format chosen first. Nothing classifies it again.
    bool standin_pinned{false};
    /// The bytes classified when the image was created were all zero (not loaded yet): it stays
    /// BC3 until a whole upload brings blocks, whose class then chooses the format.
    bool standin_empty{false};
    /// Classified by EnsureImage and not uploaded since: the first upload reads the same bytes.
    bool standin_fresh{false};
    /// An sRGB texture view of it exists: no BC4/BC5 (luminance to BC1, luminance+alpha to BC3).
    bool standin_srgb_view{false};
    /// A stand-in format change asked for during a submission (smart BC's classification at an
    /// upload). Recreating the image now would leave the bindings already recorded this frame with
    /// a view key of the old BC format (invalid usage: a view of another BC class than its image's
    /// loses the device on NVIDIA), so the change is applied at the start of the next submission.
    std::optional<AstcStandin> pending_standin;
    /// The class the format was chosen for (Rgba when kept at BC3).
    Tegra::Texture::ASTC::Channels standin_class{Tegra::Texture::ASTC::Channels::Rgba};
    /// Tile-granular backing of a large sparse texture in a virtual pool (nvn_sparse.h): the image
    /// is a sparse residency image bound per mapped tile, and the whole-storage upload and
    /// change-check paths do not apply. Null for ordinary storages.
    std::shared_ptr<SparseBacking> sparse;
    /// Resolution scaling (nvn_rescale.h). A guest storage the Resolution setting scales owns a
    /// host-only twin: a Storage of the same format at the scaled size (`scaled`, created with the
    /// first scaled pass over it; its image on first use). `scale_state` says which image holds
    /// the newest pixels. On the twin `scale_owner` names the guest storage (which owns the twin,
    /// so the pointer outlives every recording that reaches the twin); a twin has no guest bytes:
    /// it never uploads, is never checked against memory and takes part in no overlap.
    std::shared_ptr<Storage> scaled;
    Storage* scale_owner{};
    Rescale::State scale_state{Rescale::State::Native};
};

struct Texture {
    u64 serial{};
    VC::TextureDesc desc{};
    std::shared_ptr<Storage> storage; ///< Null when the image could not be created.
    HostFormat host;                  ///< Format of this texture object (views).
    /// DrawResources::GetView: this texture's TextureViews (a texture_views node) and the
    /// DrawResources::views_epoch it was found at. Replay thread only.
    void* view_slot{};
    u64 view_slot_epoch{0};
    /// The scaled twin of this texture object (Impl::ScaledTextureOf), for views of the twin
    /// storage; null until a scaled pass or sampled read needs it. It goes with the texture.
    std::shared_ptr<Texture> scaled;
    /// Last submission seq that retained this texture (avoids hashing into `Impl::last_use`).
    u64 last_use_seq{0};
    bool live{true};
};

/// The scaled twin of `native` (nvn_rescale.cpp): no image yet, the guest description of `native`
/// at the scaled size, owned by `native` through `Storage::scaled`.
[[nodiscard]] std::shared_ptr<Storage> MakeScaledTwin(Storage& native,
                                                      const Rescale::Factor& factor);

/// The NVNformat a render target `ref` of `texture` renders in: the target's view format, else
/// the view object's, else the texture's.
[[nodiscard]] inline s32 RenderTargetFormat(const Texture& texture, const VC::RenderTargetRef& ref) {
    if (ref.view.set_format) {
        return ref.view.format;
    }
    if (texture.desc.view_of != 0 && texture.desc.view.set_format) {
        return texture.desc.view.format;
    }
    return texture.desc.format;
}

struct RawTableBuffer;

/// Everything a submission or present holds on to until the GPU is done with it. Resource members
/// are released by the completion thread.
struct InFlight {
    struct Readback {
        std::shared_ptr<Pool> pool;
        u64 pool_offset;
        u64 size;
        u64 download_offset;
        u64 row_bytes{0};
        u64 row_stride{0};
        u32 row_count{0};
    };

    u64 seq{};
    /// Set by the replay thread once the work is on the GPU; guarded by Impl::state_mutex.
    bool submitted{false};
    VkSemaphore semaphore{VK_NULL_HANDLE}; ///< Null: nothing was submitted, complete at once.
    u64 value{};
    /// Replay thread only: highest timeline value actually accepted by vkQueueSubmit for this
    /// flight, and whether a batched submit containing a piece of this flight failed.
    u64 gpu_submitted_value{0};
    bool failed{false};

    std::vector<vk::Buffer> staging;
    vk::Buffer download;
    std::vector<Readback> readbacks;
    std::vector<std::shared_ptr<Pool>> dirty_pools;
    std::vector<VkCommandBuffer> command_buffers;
    std::vector<std::shared_ptr<void>> keepalive;
    /// The raw global table the draws of this submission use (kept alive once through keepalive).
    const RawTableBuffer* raw_table_kept{nullptr};
    /// A draw or dispatch of this submission stored to raw global memory (Impl::split_raw_epoch).
    bool raw_writes{false};
    /// Same, without full write-through aliases (`Impl::split_raw_epoch_unaliased`).
    bool raw_writes_unaliased{false};
    /// Host buffer chunks the draw path carved this submission's dynamic data from; returned to
    /// the free list by the completion thread.
    std::vector<std::shared_ptr<ArenaChunk>> arena;
    /// Device-local chunks the storage shadows of this submission live in (returned to the free
    /// list by the completion thread).
    std::vector<std::shared_ptr<ShadowChunk>> shadow;
    /// Query slots and timestamp reports (NvnIndirect); null until the submission reports a counter.
    std::unique_ptr<CounterFlight> counters;
    /// GPU timestamps of this submission (EDEN_NX2_NVN_GPU_PROFILE); null when the profiler is off.
    std::unique_ptr<GpuProf::Flight> gpu_prof;
    /// GPU time of this submission's BC stand-in uploads; null until one is recorded.
    std::unique_ptr<BcUploadTimes> bc_times;

    /// A capture of the presented picture (EDEN_NX2_FRAME_CAPTURE), written at completion.
    struct Capture {
        vk::Buffer buffer;
        /// 1280x720, the nx2:vi2 capture size (WriteCapture tone maps HDR values to 8-bit sRGB).
        Picture picture;
        u64 index{}; ///< 1-based present number (the nx2:vi2 path's file name)
        /// EDEN_NX2_FRAME_CAPTURE is on and this present is scheduled: write to capture_directory.
        bool to_environment_directory{false};
        /// A `frame` diag request claimed this present: also write into this folder (empty: no).
        std::filesystem::path request_directory;
        /// The GPU command capture wants this present as its frame.png (video_core/gpu_capture.h):
        /// the picture is handed over at its size, 1280x720 (empty: no).
        std::filesystem::path gpu_capture_png;
    };
    std::optional<Capture> capture;

    /// The presented picture read back for a pending RendererBase::RequestScreenshot, at the
    /// request's layout size: converted into the request's buffer and answered at completion
    /// (Impl::DeliverScreenshot).
    struct Screenshot {
        vk::Buffer buffer;
        Picture picture;
        u8* destination{}; ///< RendererSettings::screenshot_bits
        std::function<void(bool)> complete;
        VideoCore::RendererSettings* settings{};
    };
    std::optional<Screenshot> screenshot;

    /// Render target dumps (EDEN_NX2_NVN_DUMP_TARGETS), written at completion.
    struct Dump {
        vk::Buffer buffer;
        Debug::PendingDump info;
    };
    std::vector<Dump> dumps;
    /// Buffer range snapshots (EDEN_NX2_NVN_WATCH), logged at completion.
    struct Watch {
        vk::Buffer buffer;
        std::string tag;
        u32 size{};
    };
    std::vector<Watch> watches;
    /// Texture copies of a `textures` diag request, converted and written at completion.
    struct TextureDump {
        vk::Buffer buffer;
        DiagTextures::Readback info;
    };
    std::vector<TextureDump> texture_dumps;
    u64 texture_dump_bytes{}; ///< bytes the texture_dumps of this submission hold (bounded)
    /// EDEN_NX2_PATHDIFF captures (video_core/pathdiff.h): the GPU copies the ranges into
    /// `buffer`; at completion the bytes go into the capture, which is written.
    struct PathDiff {
        VideoCommon::PathDiff::Capture capture;
        struct Copy {
            u32 range{};      ///< index into capture.ranges
            bool after{};     ///< the copy made after the dispatch/draw
            u64 offset{};     ///< offset in `buffer`
            u64 size{};
            VkBuffer source{};
            u64 source_offset{};
        };
        std::vector<Copy> copies;
        vk::Buffer buffer;
    };
    std::vector<PathDiff> pathdiffs;
};

struct CmdCreatePool {
    VC::PoolDesc desc;
};
struct CmdCreateTexture {
    VC::TextureDesc desc;
};
struct CmdCreateSampler {
    VC::SamplerDesc desc;
};
struct CmdCreateProgram {
    VC::ProgramDesc desc;
};
struct CmdDestroy {
    u64 serial;
};
struct CmdSubmit {
    VC::Submission submission;
    std::shared_ptr<InFlight> flight;
};
struct CmdPresent {
    VC::PresentRequest request;
    std::shared_ptr<InFlight> flight;
};
struct CmdSetTitle {
    u64 title_id;
};
struct CmdMapVirtual {
    u64 pool_serial;
    std::vector<VC::VirtualMapping> mappings;
};
struct CmdCpuWrite {
    u64 address;
    u64 size;
    bool host{}; ///< Sink::NotifyHostWrite rather than NotifyCpuWrite
};
struct CmdTextureWritten {
    u64 serial;
};
/// A blocking readback of one texture into guest memory (Sink::FlushTextureToGuest).
struct CmdFlushTexture {
    u64 serial;
    std::shared_ptr<std::promise<bool>> done;
};
/// The guest CPU changed an event's word (Sink::NotifyEventCpu).
struct CmdEventCpu {
    u64 address;
    VC::EventCpuChange change;
    u32 value;
};
/// The recorder's report of who keeps textures alive (Sink::NoteTextureHolders).
struct CmdTextureHolders {
    VC::TextureHolderReport report;
};
using Command = std::variant<CmdCreatePool, CmdCreateTexture, CmdCreateSampler, CmdCreateProgram,
                             CmdDestroy, CmdSubmit, CmdPresent, CmdSetTitle, CmdMapVirtual,
                             CmdCpuWrite, CmdTextureWritten, CmdFlushTexture, CmdEventCpu,
                             CmdTextureHolders>;

struct QueueState {
    vk::Semaphore timeline;
    u64 next_value{1};
    /// SAMPLES_PASSED is being counted on this queue (set by its first ResetCounter/ReportCounter).
    bool occlusion_counting{false};
    /// Draws recorded on this queue since its last SAMPLES_PASSED reset while counting was off.
    u64 uncounted_draws{0};
};

struct SyncPoint {
    VkSemaphore semaphore{VK_NULL_HANDLE};
    u64 value{};
    u32 queue{};
};

struct Retired {
    u64 after_seq;
    std::function<void()> release;
};

/// One range of the raw global memory table: guest (GPU) addresses [guest, guest + size) are the
/// device addresses [device, device + size). Layout fixed by Shader::Profile::RAW_GLOBAL_TABLE_*.
struct RawGlobalRange {
    u64 guest;
    u64 device;
    u64 size;
};
static_assert(sizeof(RawGlobalRange) == 24);

/// A candidate range for the table; where sources overlap in guest space the higher priority
/// wins (ties: the earlier source).
struct RawGlobalSource {
    u64 guest;
    u64 device;
    u64 size;
    u32 priority;
};

/// The table in device memory (a sorted array of RawGlobalRange) the shaders binary-search.
struct RawTableBuffer {
    vk::Buffer buffer;
    u64 address{};
    u32 count{};
};

/// Adds a range, extending the last one when both address spaces continue it.
void AppendRawGlobalRange(std::vector<RawGlobalRange>& ranges, u64 guest, u64 device, u64 size);
/// Sorted, non-overlapping ranges covering exactly the union of the sources, each part mapped by
/// the winning source; neighbours continuing in both address spaces are merged.
[[nodiscard]] std::vector<RawGlobalRange> FlattenRawGlobalRanges(
    std::vector<RawGlobalSource> sources);
/// The raw-store mirror alias table's header (Shader::Profile::RAW_GLOBAL_ALIAS_*): zeros when
/// there are no eligible read-only mirror or imported storage aliases.
struct RawAliasHeader {
    const RawTableBuffer* table{};
    u64 low{};  ///< guest bounds [low, high) of the entries
    u64 high{};
};
/// Writes the table header into a driver bank: the entry count at RAW_GLOBAL_TABLE_OFFSET, the
/// table's device address at RAW_GLOBAL_TABLE_ADDRESS_OFFSET (zeros without a table), the raw
/// coverage diagnostic's miss ring at Shader::RawCoverage::BANK_RING_OFFSET (0: off) and the
/// mirror alias table for written-through raw stores at RAW_GLOBAL_ALIAS_* (zeros: none).
void WriteRawGlobalTable(const RawTableBuffer* table, u64 coverage_ring, std::span<u8> bank,
                         const RawAliasHeader& alias = {});
/// The raw coverage class (VideoCommon::ClassifyRawMiss) of a guest address the table `ranges`
/// missed on the NVN HLE path: mapped = a pool holds it (a virtual pool only where one of its runs
/// maps it, elsewhere the GPU would fault); a mapped gap names the pool and why the table leaves
/// it out (its backing; a virtual pool's: the physical pool's, or "virtual->destroyed").
[[nodiscard]] VideoCommon::RawMissVerdict ClassifyHleRawMiss(
    u64 guest, std::span<const VideoCommon::RawGlobalRange> ranges,
    const std::map<u64, std::shared_ptr<Pool>>& pools_by_address,
    const std::unordered_map<u64, std::shared_ptr<Pool>>& pools);

struct Context::Impl {
    Impl(const Device& device, std::mutex& submit_mutex, std::unique_ptr<PresentTarget> present,
         ContextOptions options, std::unique_ptr<PresentFrameGen> present_frame_gen = {});
    ~Impl();

    // --- construction-time constants ---
    const Device& device;
    const vk::DeviceDispatch& dld;
    VkDevice logical;
    std::mutex& submit_mutex;
    std::unique_ptr<PresentTarget> present_target;
    ContextOptions options;
    VmaAllocator vma{};
    /// What `allocator` placed outside device-local memory (`[nvn] memory:` line).
    AllocationSpills spills;
    /// EDEN_NX2_TRY_VMA_DEVICE_STRICT (MemoryAllocator::TrackSpills).
    bool vma_device_strict{false};
    std::unique_ptr<MemoryAllocator> allocator;
    std::shared_ptr<::Vulkan::FSR> upscaler_fsr;
    std::unique_ptr<::Vulkan::FSR> final_fsr;
    /// DLSS frame generation of the presented frames; null without a renderer (headless
    /// contexts). Replay thread only; released before `allocator` (it owns images from it).
    std::unique_ptr<PresentFrameGen> present_frame_gen;
    std::optional<VkExtent2D> upscaler_fsr_extent;
    std::optional<VkExtent2D> final_fsr_extent;
    std::optional<HostImporter> importer;
    bool dlss_depth_missing_logged{};
    /// The image DLSS writes for one game network (Streamline viewport) before it is copied into
    /// the game's output rectangle. Replay thread only.
    struct GameDlssTarget {
        vk::Image image;
        vk::ImageView view;
    };
    struct GameDlssViewport {
        std::shared_ptr<GameDlssTarget> target;
        VkExtent2D render{};
        VkExtent2D output{};
        u32 last_evaluation{};
    };
    std::unordered_map<u32, GameDlssViewport> game_dlss;
    u32 game_dlss_evaluations{};
    bool game_dlss_failed{};
    /// Replay thread only: an own-upscale dispatch (DK's NDSR) or the game's own DLSS evaluation
    /// already replaced this frame's upscale stage.
    bool ndsr_upscaled_since_present{};
    /// True only when DLSS/DLAA (not FSR/bilinear fallback) wrote this frame's upscale.
    bool dlss_upscaled_since_present{};
    VideoCore::GameFrameSize::Stage upscaled_stage{VideoCore::GameFrameSize::Stage::Game};
    /// Replay thread only: the input and output sizes of that upscale.
    u32 upscaled_input_width{};
    u32 upscaled_input_height{};
    u32 upscaled_output_width{};
    u32 upscaled_output_height{};
    /// Replay thread only, kept while a game's upscaling network may be replaced
    /// (DrawContext::NetworkReplacementWanted): the last compute dispatches (a ring, slot
    /// `compute_chain_next` is the oldest once it is full) with the textures they sample and write,
    /// so the network's packed input can be traced back to the scene (FindNetworkColour). Emptied
    /// by every present.
    std::array<VideoCore::Upscaler::ComputeChainEntry, 16> compute_chain{};
    u32 compute_chain_count{};
    u32 compute_chain_next{};
    /// The depth texture serial of the last render pass drawn into each colour texture serial
    /// (the scene's depth for the scene colour), and whether a compute program holds tensor math
    /// (by program hash; the code never changes).
    std::unordered_map<u64, u64> pass_depth_of_color;
    std::unordered_map<u64, bool> program_tensor_math;
    vk::Queue queue;
    vk::CommandPool command_pool;

    std::atomic<VC::GuestMemory*> guest{nullptr};
    /// The device's window origin / depth mode as the game set them (Sink::SetDeviceModes).
    std::atomic<u32> device_window_origin{0};
    std::atomic<u32> device_depth_mode{0};

    struct AtomicStats {
        std::atomic<u64> pools_imported{};
        std::atomic<u64> pools_mirrored{};
        std::atomic<u64> pools_mirrored_discontiguous{}; ///< ... guest storage is several host runs
        std::atomic<u64> pools_mirrored_no_import{};     ///< ... host import unavailable or failed
        std::atomic<u64> pools_split_imported{};         ///< ... several host runs, each imported
        std::atomic<u64> split_runs{};                   ///< ... in this many runs
        std::atomic<u64> mirror_upload_bytes{};          ///< guest bytes copied into mirrors
        std::atomic<u64> mirror_upload_runs{};           ///< ... in this many copies
        std::atomic<u64> mirror_uploads{};               ///< UploadRange calls that copied anything
        std::atomic<u64> staging_chunks_reused{};
        std::atomic<u64> mirror_pages_hashed{};          ///< guest pages compared with the mirror
        std::atomic<u64> mirror_pages_trusted{}; ///< ... not compared: the tracker saw no write
        std::atomic<u64>
            mirror_pages_gpu_owned{}; ///< ... not compared: pending GPU writes own the page
        std::atomic<u64> mirror_ranges_clean{}; ///< range uses that needed no copy
        /// VRAM mirrors of imported pools (EDEN_NX2_NVN_POOL_MIRROR).
        std::atomic<u64> vram_mirrors{};            ///< mirrors created
        std::atomic<u64> vram_mirror_refused{};     ///< ... refused (budget, allocation, tracker)
        std::atomic<u64> vram_copy_in_runs{};       ///< page runs copied guest -> mirror
        std::atomic<u64> vram_copy_in_bytes{};      ///< ... their bytes
        std::atomic<u64> vram_bindings{};           ///< storage bindings / indirect reads served
        std::atomic<u64> vram_bindings_host{};      ///< ... kept in guest memory in an open pass
        std::atomic<u64> vram_write_ranges{};       ///< written ranges snapshotted and merged back
        std::atomic<u64> vram_write_bytes{};        ///< ... their bytes
        std::atomic<u64> vram_write_backs_plain{};  ///< ... merged by a plain copy (no merge shader)
        std::atomic<u64> vram_invalidate_all{};     ///< aborts invalidated every mirror
        std::atomic<u64> vram_mirrors_read_only{};  ///< mirrors made for vertex/index/uniform ranges
        std::atomic<u64> vram_mirrors_promoted{};   ///< ... of them that took a written storage range
        std::atomic<u64> vram_invalidate_read_only{}; ///< raw-writing draws that made them stale
        std::atomic<u64> vram_ro_written_through{}; ///< raw-writing draws that wrote through instead
        std::atomic<u64> vram_raw_alias_rebuilds{};  ///< raw store mirror alias tables built
        std::atomic<u64> vram_vertex_ranges{};      ///< vertex/index ranges served from mirrors
        std::atomic<u64> vram_vertex_budget_skips{}; ///< ... ranges left in guest memory (copy bound)
        std::atomic<u64> vram_flush_pages_kept{};   ///< guest-flushed pages a mirror kept (flush_tracked)
        std::atomic<u64> vram_flush_pages_taken{};  ///< ... pages it had to copy in again
        std::atomic<u64> pools_device_local{};
        std::atomic<u64> pools_unsupported{};
        std::atomic<u64> host_import_failures{};
        std::atomic<u64> textures_created{};
        std::atomic<u64> textures_failed{};
        std::atomic<u64> textures_transcode_unsupported{};
        std::atomic<u64> alias_unsupported{};
        std::atomic<u64> alias_events{}; ///< every unresolvable overlap read (alias_unsupported counts pairs)
        std::atomic<u64> alias_events_stale{};   ///< ... of a reader the GPU had written
        std::atomic<u64> alias_events_initial{}; ///< ... of a reader holding the guest's bytes
        /// Overlapping GPU contents not copied over a guest upload: the CPU rewrote the bytes
        /// after the GPU write (Recorder::ResolveOverlaps).
        std::atomic<u64> overlaps_skipped_cpu_rewrite{};
        /// Counters of the memory work (nvn_storage.cpp / nvn_upload.cpp).
        std::atomic<u64> storages_created{};     ///< VkImages created (first use of a storage)
        std::atomic<u64> storages_recycled{};    ///< VkImages taken from the exact-shape recycle pool
        std::atomic<u64> storages_reused{};      ///< textures that found a cached storage
        std::atomic<u64> storages_evicted{};
        std::atomic<u64> storages_superseded_evicted{}; ///< ... GPU-written ones a later GPU write replaced
        std::atomic<u64> storages_aged_evicted{};       ///< ... idle render targets past the age limit
        std::atomic<u64> storages_pressure_evicted{};   ///< ... under the budget, for the device budget
        std::atomic<u64> storage_trim_pressure_passes{}; ///< trim passes that found the device over budget
        std::atomic<u64> storage_cache_bytes{};         ///< storage_cache_bytes of the replay thread, as a gauge
        std::atomic<u64> alias_transfers{};      ///< overlapping textures' contents copied over
        std::atomic<u64> alias_byte_transfers{}; ///< ... through the pool's bytes (alias shader)
        std::atomic<u64> alias_byte_bytes{};     ///< bytes of window those moved
        std::atomic<u64> alias_overlaps_seen{};  ///< textures created over another's storage
        /// Resolution scaling (nvn_rescale.h): passes drawn at the scaled size, scaled passes that
        /// fell back to guest size (an ineligible attachment), downscale / upscale blits recorded
        /// (a guest-sized image made from its twin, and the reverse), scaled copies and draws /
        /// dispatches that sampled a twin.
        std::atomic<u64> rescale_passes{};
        std::atomic<u64> rescale_native_passes{};
        std::atomic<u64> rescale_downscales{};
        std::atomic<u64> rescale_upscales{};
        std::atomic<u64> rescale_copies{};
        std::atomic<u64> rescale_twin_samples{};
        std::atomic<u64> dispatch_barriers_left{}; ///< dispatches after a dispatch the game did not order (no barrier)
        std::atomic<u64> memory_barriers{};        ///< full memory barriers the recorder emitted (MemoryBarrier, PrepareImages)
        // EDEN_NX2_TRY_BARRIER_MERGE: replay commands with known buffer ranges (HostBeginBufferOp).
        std::atomic<u64> barrier_merge_ops_clean{};          ///< nothing wrote since the last barrier: none needed
        std::atomic<u64> barrier_merge_ops_left_out{};       ///< earlier buffer commands since the last barrier, no overlap: barrier left out
        std::atomic<u64> barrier_merge_ops_hazard{};         ///< overlap (RAW/WAR/WAW) with an earlier buffer command: barrier recorded
        std::atomic<u64> barrier_merge_ops_unknown_writes{}; ///< a draw/dispatch/image copy wrote since: barrier recorded
        std::atomic<u64> barrier_merge_folded{};             ///< barrier requests folded into one still waiting for its next command
        std::atomic<u64> barrier_merge_emitted{};            ///< barrier commands recorded for waiting requests (each stands for one or more)
        std::atomic<u64> uploads{};              ///< storage uploads from guest memory
        std::atomic<u64> astc_gpu_uploads{};     ///< ASTC storages decoded on the GPU (no CPU work)
        std::atomic<u64> astc_gpu_bytes{};       ///< ... guest bytes they read
        std::atomic<u64> astc_cpu_fallbacks{};   ///< ASTC storages with GPU decode enabled that took the CPU path
        std::atomic<u64> astc_bc_encodes{};      ///< ASTC uploads re-encoded into a BC stand-in
        std::atomic<u64> untile_gpu_uploads{};   ///< block-linear storages untiled on the GPU
        std::atomic<u64> untile_gpu_direct{};    ///< ... read in place (imported / GPU-only pool)
        std::atomic<u64> untile_gpu_bytes{};     ///< ... linear bytes they produced
        std::atomic<u64> untile_cpu_fallbacks{}; ///< tiled storages the GPU untile could not take
        std::atomic<u64> upload_bytes{};
        std::atomic<u64> cpu_checks{};           ///< hash checks of CPU-sourced storages
        std::atomic<u64> cpu_checks_trusted{};   ///< checks the page tracker answered (no hash)
        std::atomic<u64> cpu_checks_pages{};     ///< checks answered by rehashing written pages
        std::atomic<u64> cpu_pages_rehashed{};   ///< pages those checks hashed
        std::atomic<u64> cpu_changes{};          ///< ... that found changed guest bytes
        std::atomic<u64> host_write_uploads{};   ///< GPU-only pool ranges a host write re-uploaded
        std::atomic<u64> decode_ns{};            ///< ASTC/BCn decode time
        std::atomic<u64> decode_bytes{};         ///< ... output bytes
        std::atomic<u64> hash_ns{};
        std::atomic<u64> hash_bytes{};
        std::atomic<u64> hash_bytes_compressed{}; ///< ... of block-compressed storages
        std::atomic<u64> hash_bytes_dynamic{};    ///< ... of storages the CPU changed before
        std::atomic<u64> depth_upload_unsupported{};
        std::atomic<u64> virtual_maps{};
        std::atomic<u64> virtual_unresolved{};
        std::atomic<u64> virtual_sparse_binds{};
        std::atomic<u64> sparse_write_pool_checks{};
        std::atomic<u64> texels_flushed{};
        std::atomic<u64> image_transitions{}; ///< image layout transitions recorded (PrepareImagesSpan)
        std::atomic<u64> ops_unsupported{};
        std::atomic<u64> ops_failed{};
        std::atomic<u64> dropped_after_device_loss{};
        std::atomic<u64> submissions{};
        std::atomic<u64> presents{};
        std::atomic<u64> early_submits{}; ///< Command buffers submitted before their submission was recorded.
        std::atomic<u64> held_back_submissions{}; ///< Submissions that waited for a sync nobody had fenced yet.
        std::atomic<u64> parked_timeouts{};
        std::atomic<u64> event_signals{};       ///< SignalEvent records replayed.
        std::atomic<u64> event_waits{};         ///< WaitEvent records the model satisfied.
        std::atomic<u64> event_waits_blocked{}; ///< ... that suspended their stream at least once.
        std::atomic<u64> waits_unsatisfied{};   ///< WaitEvent records released by the timeout.
        std::atomic<u64> indirect_draws{};         ///< Indirect draw commands recorded.
        std::atomic<u64> indirect_counted_draws{}; ///< ... of which with a GPU draw count.
        std::atomic<u64> indirect_unsupported{};   ///< Indirect draws dropped (counted, logged once).
        std::atomic<u64> raw_global_draws{};
        std::atomic<u64> raw_global_pools_skipped{};
        std::atomic<u64> raw_global_ranges_dropped{};
        std::atomic<u64> raw_global_rebuilds{};
        std::atomic<u64> mesh_task_draws{}; ///< vkCmdDrawMeshTasks*EXT recorded.
    } stats;

    // --- work queue shared with the caller's thread and the completion thread ---
    std::mutex state_mutex;
    std::condition_variable work_cv;
    std::condition_variable complete_cv;
    std::deque<Command> commands;
    std::deque<std::shared_ptr<InFlight>> inflight;
    std::function<void(u64)> completion_callback;
    std::mutex callback_mutex; ///< Held while completion_callback runs (and while it is replaced).
    /// Set once when the Vulkan device was lost; never cleared. From then on the replay completes
    /// submissions and presents without running them.
    std::atomic<bool> device_lost{false};
    std::function<void(std::string)> failure_callback;
    std::mutex failure_mutex; ///< Held while failure_callback runs (and while it is replaced).
    bool stopping{false};
    std::atomic<u64> completed_seq{0};
    /// Newest submission whose GPU commands and mirror readbacks have finished (published right
    /// before `completion_callback`, whereas `completed_seq` is published right after it).
    std::atomic<u64> gpu_completed_seq{0};
    /// What the replay thread is doing (diagnostics only).
    std::atomic<const char*> replay_stage{"starting"};
    std::atomic<u64> replay_seq{0};
    /// Per command kind (variant index): count and total nanoseconds in Dispatch.
    std::array<std::atomic<u64>, 14> command_count{};
    std::array<std::atomic<u64>, 14> command_ns{};
    /// CreatePoolNow phase totals in nanoseconds: pointer lookup, host import, buffer creation.
    std::atomic<u64> pool_ns_pointer{0};
    std::atomic<u64> pool_ns_import{0};
    std::atomic<u64> pool_ns_buffer{0};
    std::atomic<u64> pool_bytes{0};
    std::atomic<u64> pool_slow_logged{0};
    /// Large host-memory chunks pools are bound into and their cost. Guarded by backing_mutex.
    std::vector<std::shared_ptr<ImportedChunk>> import_chunks;
    /// Serializes everything that backs pools: the replay thread (EnsureBacking, the raw global
    /// table) and the backing thread. Held while a chunk is imported (up to ~250 ms).
    std::mutex backing_mutex;
    std::mutex backing_queue_mutex;
    std::condition_variable backing_cv;
    std::deque<std::shared_ptr<Pool>> backing_queue;
    bool backing_stop{false};
    std::thread backing_thread;
    std::atomic<u64> pools_backed_background{0};
    void BackingMain();
    /// Backs the pool (backing_mutex held): host import / mirror / device-local buffer.
    void BackPoolLocked(Pool& pool);
    /// True when [host, host + size) lies in an imported chunk (backing_mutex held).
    [[nodiscard]] bool InImportedChunk(const u8* host, u64 size) const;
    std::atomic<u64> chunk_imports{0};
    std::atomic<u64> chunk_import_ns{0};
    std::atomic<u64> chunk_import_bytes{0};
    /// The chunks in `import_chunks` (kept until the context goes) and their bytes.
    std::atomic<u64> import_chunks_live{0};
    std::atomic<u64> import_chunks_live_bytes{0};
    std::atomic<u64> pools_in_chunks{0};
    std::atomic<u64> pool_import_fallbacks{0};
    /// EDEN_NX2_FRAME_CAPTURE directory (empty: off). Every 120th present is saved as
    /// nx2-frame-<n>.tga, the files the nx2:vi2 present path writes.
    std::string capture_directory;
    /// Presents recorded so far (replay thread): the capture schedule counts the presents that
    /// reach the GPU in order, not the ones queued (a held-back present runs later).
    u64 presents_recorded{0};
    /// EDEN_NX2_FRAME_CAPTURE_SKIPS state (replay thread): frames captured for a skipped draw so
    /// far, and the presents to capture 8 frames after each of them. Per context, like the
    /// presents they count.
    u64 capture_skip_frames_captured{0};
    std::vector<u64> capture_after_frames;
    /// The pool the EDEN_NX2_NVN_WATCH address was last logged for (replay thread).
    u64 watch_pool_logged{0};
    void WriteCapture(const InFlight::Capture& capture) const;
    /// The renderer's screenshot request slot (Context::SetScreenshotRequests); null: none.
    std::atomic<VideoCore::RendererSettings*> screenshot_requests{nullptr};
    /// Completion thread: answers `flight`'s screenshot, or gives its request back when the frame
    /// never reached the GPU. The conversion runs on `screenshot_thread`, not on the completion
    /// thread (a 4K HDR frame takes tens of milliseconds, and the guest waits on completions).
    void DeliverScreenshot(InFlight& flight);
    std::thread screenshot_thread;
    std::mutex wait_mutex;
    std::condition_variable wait_cv;
    std::thread replay_thread;
    std::thread complete_thread;
    /// Device::ReportLoss (it waits 15 s for the log and Nsight Aftermath) runs here, off the
    /// replay and completion threads, which must go on completing work after a loss.
    std::thread loss_report_thread;

    // Recycled command buffers (returned by the completion thread).
    std::mutex free_commands_mutex;
    std::vector<VkCommandBuffer> free_command_buffers;

    // The completion thread returns ordinary upload buffers after their final timeline signal.
    static constexpr u64 STAGING_CHUNK = u64{8} << 20;
    std::mutex staging_cache_mutex;
    std::array<vk::Buffer, 2> staging_cache;
    [[nodiscard]] vk::Buffer AcquireStaging(u64 size);

    /// Usage of pool buffers: POOL_BUFFER_USAGE, plus a device address when the device has
    /// buffer device address (raw global memory).
    VkBufferUsageFlags pool_usage{POOL_BUFFER_USAGE};

    // --- replay thread only ---
    bool mario_kart_world{};
    /// Sparse tiles mapped to the same guest bytes share one host page (SparseBacking). On by
    /// default for MKW where the device can alias sparse memory; EDEN_NX2_NVN_SPARSE_SHARE=0 turns
    /// it off, =1 on for every title.
    std::optional<bool> sparse_share_enabled;
    bool SparseShareEnabled();
    /// Counts SparseBacking::Prepare calls: a shared page allocated by an earlier call has its
    /// texels earlier in the command stream.
    u64 sparse_prepare_epoch{0};
    std::optional<bool> mkw_bc3_atlas_supported;
    bool MkwBc3AtlasSupported();
    bool mkw_coalesce_early_submit{};
    // Sparse binds on the Vulkan queue must wait for older command buffers using the image.
    VkSemaphore mkw_last_submit_semaphore{VK_NULL_HANDLE};
    u64 mkw_last_submit_value{};
    std::unordered_map<u64, std::shared_ptr<Pool>> pools;
    std::map<u64, std::shared_ptr<Pool>> pools_by_address;
    /// Bumped whenever a pool is created, destroyed or backed (the raw global table depends on
    /// the set; the backing thread bumps it under backing_mutex).
    std::atomic<u64> pool_generation{0};
    /// Bumped whenever a split-imported mirror pool is backed or destroyed.
    std::atomic<u64> split_alias_generation{0};

    /// Guest address -> device address ranges of every backed pool, for shaders that read and write
    /// raw global memory (Shader::Profile::RAW_GLOBAL_TABLE_*). Adjacent pools whose guest and
    /// device ranges are both contiguous share one range.
    struct RawGlobalTable {
        u64 generation{~u64{0}};
        /// Sorted by guest address, never overlapping (FlattenRawGlobalRanges).
        std::vector<RawGlobalRange> entries;
        /// The entries in device memory, read by the shaders (null when there are none). Draws
        /// keep it alive through InFlight::keepalive.
        std::shared_ptr<RawTableBuffer> device;
        /// Pools with a buffer of their own that the entries cover (a draw that uses the table
        /// retains them; chunks live as long as the context).
        std::vector<u64> pool_serials;
        std::vector<Pool*> pools;
        /// An imported pool: GPU writes through the table are host-visible there.
        Pool* imported_pool{nullptr};
    } raw_table;
    /// Raw stores may write through to read-only mirrors or the imported guest storage backing
    /// (EDEN_NX2_TRY_RO_MIRROR_EXACT / EDEN_NX2_TRY_STORAGE_RAW_WRITE_THROUGH).
    struct RawWriteThroughTable {
        /// Unique table generation, used to keep replaced buffers alive in submissions.
        u64 generation{0};
        /// Pool-set generation the entries were built from.
        u64 mirror_generation{~u64{0}};
        u64 split_generation{~u64{0}};
        bool read_only_enabled{};
        bool storage_enabled{};
        /// The entries in device memory (null when there are none); draws keep it alive through
        /// InFlight::keepalive.
        std::shared_ptr<RawTableBuffer> device;
        u64 low{};
        u64 high{};
        /// Pools whose current mirror is kept current by the aliases in this table.
        std::vector<u64> pool_serials;
        std::vector<Pool*> pools;
    } raw_write_through_table;
    /// Bumped whenever any mirror is made, promoted to a storage mirror or released.
    u64 ro_alias_generation{0};
    u64 raw_write_through_table_generation{0};
    /// Replay thread: rebuilt when the mirror set or either write-through switch changes.
    /// Overlapping pools are excluded so every address has one alias target.
    [[nodiscard]] const RawWriteThroughTable& RawStoreAliasTable();
    /// Replay thread: uploads the table entries to a device-local buffer (null when empty or the
    /// upload failed).
    [[nodiscard]] std::shared_ptr<RawTableBuffer> UploadRawTable(
        std::span<const RawGlobalRange> entries);
    /// Replay thread: backs every pool, then returns the table (rebuilt when the set of pools
    /// changed). Mirror and virtual pools are not in it (counted, logged once).
    [[nodiscard]] const RawGlobalTable& RawGlobalMemoryTable();
    /// Raw coverage diagnostic (VideoCommon::RawCoverageEnabled), replay thread only: the miss
    /// rings the programs report into (made on first use, released before `allocator`), the run
    /// report and the parse buffer of a finished window.
    std::optional<::Vulkan::RawCoverageRings> raw_coverage;
    VideoCommon::RawCoverageReport raw_coverage_report{"hle"};
    VideoCommon::RawCoverageRing raw_coverage_window;
    /// Replay thread, for a draw or dispatch of the flight `seq` that uses the raw table: logs a
    /// finished window, returns the device address of the ring the draw reports into (0 when the
    /// rings are unavailable). Every segment that used a ring must end with a host barrier.
    [[nodiscard]] u64 RawCoverageRingFor(u64 seq);
    std::unordered_map<u64, std::shared_ptr<Texture>> textures;
    std::unordered_map<u64, VC::SamplerDesc> samplers;
    std::unordered_map<u64, std::shared_ptr<VC::ProgramDesc>> programs;
    /// Pipeline cache, samplers, views, arena (nvn_draw_resources.h). Created with the context.
    std::unique_ptr<DrawResources> draw_resources;
    /// GPU writes against the submit-time uniform copies (nvn_uniform_snapshot.h). Replay thread
    /// only; declared before the draw contexts and streams that refer to it.
    UniformSnapshotTracker uniform_snapshots;
    /// Uncompleted GPU buffer writes recorded on the replay thread, kept until `gpu_completed_seq`
    /// passes their submission. When `inline_bytes` is non-empty, it holds the exact bytes copied
    /// by `DrawContext::FlushInlineUpdates` so same-submission in-pass private uniform copies can
    /// overlay them without breaking the render pass; when empty, the write is non-inline (or
    /// invalidated by a CPU write / WaitEvent) and overlapping uniform draws flush on the GPU.
    struct PendingPoolWrite {
        u64 begin;
        u64 end;
        u64 seq;
        std::vector<u8> inline_bytes;
    };
    /// In the order noted. A removed write stays as a dead entry (begin == end) until the list is
    /// compacted, so the positions the index holds stay valid.
    std::vector<PendingPoolWrite> pending_pool_writes;
    size_t pending_dead{};   ///< dead entries of the list
    size_t pending_inline{}; ///< live writes that hold inline bytes
    /// gpu_completed_seq at the last prune, and whether a write at or below it was noted since
    /// (nothing else can be pruned until it moves).
    u64 pending_pruned_done{~u64{0}};
    bool pending_prune_due{false};
    /// Positions of the live writes by 16 KiB block of their range, and of those spanning too
    /// many blocks. Kept current with every change: draws ask per uniform binding and note
    /// several writes each, with thousands of writes pending (a scan of the list per question
    /// was ~25% of the replay thread in Ocarina of Time).
    std::unordered_map<u64, std::vector<u32>> pending_blocks;
    std::vector<u32> pending_wide;
    bool noting_inline_pool_write{false};
    void IndexPendingPoolWrite(u32 index, bool add);
    void PushPendingPoolWrite(PendingPoolWrite&& write);
    void RemovePendingPoolWrite(u32 index);
    void RemovePendingPoolWritesWithin(u64 begin, u64 end);
    void SettlePendingPoolWrites();
    void RebuildPendingPoolIndex();
    void CollapsePendingPoolWrites();
    /// visit(position) for each live write that may overlap [begin, end), until it returns true
    /// (a write may come more than once). The visitor must not add or remove writes.
    template <typename Visitor>
    bool VisitPendingPoolWrites(u64 begin, u64 end, Visitor&& visit) const;
    void PrunePendingPoolWrites();
    void NoteFlushedInlinePoolWrite(u64 address, const u8* data, u64 size, u64 seq);
    void NoteNonInlinePoolWrite(u64 address, u64 size, u64 seq);
    void InvalidatePendingInlineBytes(u64 seq);
    void InvalidatePendingInlineBytesOverlapping(u64 address, u64 size);
    void ErasePendingPoolWritesForSeq(u64 seq);
    [[nodiscard]] bool NeedsGpuFlushForUniform(u64 begin, u64 end, u64 current_seq);
    void ApplyPendingInlineWrites(u64 base_address, u8* dst, u64 size, u64 current_seq,
                                  std::vector<std::pair<u64, u64>>* written = nullptr) const;
    /// Draw state per NVN queue index (it persists across submissions on a queue).
    std::map<u32, std::unique_ptr<DrawContext>> draw_contexts;
    std::map<u32, QueueState> queues;
    std::unordered_map<u64, SyncPoint> syncs;
    /// Per NVN queue: the submissions and presents waiting to be replayed, in order (the replay
    /// suspends a submission at a wait nothing satisfied and runs the other queues meanwhile).
    VC::StreamScheduler streams{options.parked_timeout};
    /// Items in `streams` as of the last pump (read by DescribeState from other threads).
    std::atomic<size_t> streams_pending{0};
    /// Value of every event the replay has seen, as of the replay position. Replay thread only.
    VC::EventModel events{[this](u64 address) -> std::optional<u32> {
        VC::GuestMemory* memory = guest.load(std::memory_order_acquire);
        u32 word = 0;
        if (memory && memory->Read(address, &word, sizeof(word))) {
            return word;
        }
        return std::nullopt;
    }};
    std::unordered_map<u64, u64> last_use; ///< serial -> last seq that retained it
    /// Bumped whenever an entry of `textures` is erased (cached node pointers die).
    u64 texture_epoch{1};
    /// Bumped whenever an entry of `textures` is created or erased.
    u64 texture_change_epoch{1};
    /// The serials erased last: erase number n (texture_epoch - 1 when it happened) is at
    /// n % size. Lets a holder of many node pointers drop only the dead ones.
    std::array<u64, 4096> texture_erased{};
    /// Bumped when a `storage_index` node is erased (Storage::pool_index).
    u64 storage_index_epoch{1};
    /// Replay thread: recently used (serial -> seq) so that NoteUse of an already noted serial does
    /// not touch `last_use`.
    struct UseSlot {
        u64 serial{~u64{0}};
        u64 seq{0};
    };
    std::array<UseSlot, 4096> use_cache{};
    /// Replay thread: recently resolved pool ranges, valid while `pool_generation` is unchanged.
    /// [begin, end) is the part of the pool no other pool starts in, so the answer is the same as
    /// the address map's for every address in it.
    struct ResolveSlot {
        u64 generation{~u64{0}};
        u64 begin{};
        u64 end{};
        Pool* pool{};
    };
    mutable std::array<ResolveSlot, 16> resolve_cache{};
    mutable u32 resolve_next{0};
    struct ExactResolveSlot {
        u64 generation{~u64{0}};
        u64 mappings{};
        u64 address{};
        u64 size{};
        Pool* pool{};
        u64 offset{};
    };
    mutable std::array<ExactResolveSlot, 4096> mkw_resolve_cache{};
    u64 mkw_mapping_generation{};
    mutable u64 mkw_resolve_hits{};
    mutable u64 mkw_resolve_misses{};
    std::vector<Retired> retired;
    /// Stand-ins whose format change waits for the next submission (Storage::pending_standin).
    std::vector<std::weak_ptr<Storage>> pending_standins;
    /// Storages of one pool, for overlap resolution.
    struct PoolIndex {
        std::multimap<u64, Storage*> spans; ///< begin offset -> cached storage
        u64 max_size{};
        /// Bumped at every GPU write to a storage of the pool (see Storage::pool_gen_seen).
        u64 write_gen{};
    };
    /// pool serial -> storages, for sharing and overlap detection.
    std::map<u64, PoolIndex> storage_index;
    struct StorageKey {
        u64 pool_serial;
        u64 offset;
        u64 size;
        s32 target;
        u32 width;
        u32 height;
        u32 depth;
        u32 levels;
        s32 samples;
        s32 flags_class;   ///< layout-relevant NVNtextureFlags (linear, sparse, minimal layout)
        u32 stride;
        s32 pkg_block_h;
        u32 bytes_per_block;
        u32 block_width;
        u32 block_height;
        s32 identity;      ///< NVNformat for transcoded textures (the decode depends on it), else 0
        u32 vk_format;     ///< host VkFormat for depth/stencil, else 0
        u32 host_caps;     ///< attachable | storage << 1
        auto operator<=>(const StorageKey&) const = default;
    };
    /// Every storage created, kept after its textures died (strong references) so that the next
    /// texture over the same bytes reuses the image. Replay thread only.
    std::map<StorageKey, std::shared_ptr<Storage>> storage_cache;
    u64 storage_cache_bytes{0};
    /// Exact-shape VkImage recycler for non-sparse storage images (avoids WDDM vmaCreateImage /
    /// vmaDestroyImage kernel overhead when storages churn). Replay thread only.
    struct ImageRecycleKey {
        VkImageCreateFlags flags{};
        VkImageType type{};
        VkFormat format{};
        u32 width{};
        u32 height{};
        u32 depth{};
        u32 levels{};
        u32 layers{};
        VkSampleCountFlagBits samples{};
        VkImageUsageFlags usage{};
        bool operator==(const ImageRecycleKey&) const = default;
        auto operator<=>(const ImageRecycleKey&) const = default;
    };
    struct RecycledImage {
        ImageRecycleKey key;
        vk::Image image;
        u64 vram_bytes{};
        u64 lru_tick{};
    };
    static constexpr u64 IMAGE_RECYCLE_MAX_BYTES = u64{256} << 20;
    static constexpr u64 IMAGE_RECYCLE_MAX_COUNT = 128;
    std::map<ImageRecycleKey, std::vector<RecycledImage>> image_recycle_pool;
    u64 image_recycle_bytes{0};
    u64 image_recycle_count{0};
    u64 image_recycle_tick{0};
    /// Trim threshold (ContextOptions::storage_cache_budget_bytes) and the throttle of the passes
    /// over the whole cache: the next one runs when the cache has grown past `trim_floor_bytes` or
    /// at present `trim_next_present`. Replay thread only.
    u64 storage_cache_budget{0};
    u64 trim_floor_bytes{0};
    u64 trim_next_present{0};
    /// EDEN_NX2_NVN_TRIM_LEGACY: only the over-budget pass at image creation, without the age and
    /// device budget rules or the passes at presents.
    bool trim_legacy{false};
    /// Idle render targets that can be rebuilt are dropped after this many presents without a use
    /// (0: never).
    u64 trim_age_presents{DEFAULT_STORAGE_TRIM_AGE_PRESENTS};
    /// Device usage over the device budget at the last pass (0: within it, or not reported).
    u64 trim_device_excess{0};
    /// Strict mode's over-budget allocations seen by the last pass: a new one brings the next
    /// pass at a present forward.
    u64 trim_seen_over_budget{0};
    /// Who keeps the textures the game no longer names alive (Sink::NoteTextureHolders, the
    /// recorder's last report), by serial; absent: the game's own texture. Replay thread only.
    std::unordered_map<u64, VC::TextureHolder> texture_holders;
    u64 texture_holders_present{0}; ///< present count when the report arrived
    /// Storages (and their scaled twins) by their strongest holder at the last report, for the
    /// `[nvn] memory:` line: count and bytes per VC::TextureHolder.
    std::array<std::atomic<u64>, VC::TEXTURE_HOLDER_KINDS> holder_storages{};
    std::array<std::atomic<u64>, VC::TEXTURE_HOLDER_KINDS> holder_storage_bytes{};
    std::atomic<bool> holder_report_seen{false};
    /// The strongest holder of each storage a texture (or a render-target alias) references.
    [[nodiscard]] std::unordered_map<const Storage*, VC::TextureHolder> StorageHolders() const;
    void NoteTextureHoldersNow(VC::TextureHolderReport& report);
    /// Device memory pages of the sparse storages (created with the first one) and the virtual
    /// pools that hold one. Replay thread (the pool is shared with the threads releasing storages).
    std::shared_ptr<SparsePagePool> sparse_pages;
    std::set<u64> sparse_virtual_pools;
    // MKW CPU flushes visit only virtual pools that map the physical pool being flushed.
    std::unordered_map<u64, std::vector<u64>> mkw_sparse_physical_users;
    bool mkw_sparse_users_dirty{true};
    [[nodiscard]] std::span<const u64> MkwSparseVirtualUsers(u64 physical_serial);
    /// Sparse tiles compared with their guest bytes in the present `sparse_verify_tick`.
    u64 sparse_verify_tick{~u64{0}};
    u32 sparse_verify_count{0};
    u32 virtual_storages_logged{0};
    u64 write_clock{0};
    /// Clock the CPU-change checks are scheduled by: the present count, plus 1 per 16 ms while no
    /// present happens (a loading screen), so checks never stall. Every new present moves it on,
    /// also after such a stretch moved it ahead of the present count. Replay thread only.
    u64 poll_tick{0};
    u64 poll_presents_seen{0};
    /// `poll_tick` at the last new present: the idle ticks of the current stretch count from here.
    u64 poll_tick_at_change{0};
    std::chrono::steady_clock::time_point poll_last_change{};
    /// Recent time between presents in eighths of a tick: what an ordinary frame is (PollTick).
    u64 poll_frame_ticks_x8{0};
    u64 full_hash_tick{~u64{0}};
    u64 full_hash_bytes{0};
    [[nodiscard]] u64 PollTick();
    std::set<std::pair<u64, u64>> alias_logged;
    std::set<s32> unsupported_formats_logged;
    std::set<std::string> logged_messages;
    std::set<u64> parked_timeout_logged;
    /// (event address, wait mode, value) of the waits released by the timeout that were logged.
    std::set<std::tuple<u64, u32, u32>> unsatisfied_logged;
    std::vector<vk::CommandBuffers> command_buffer_allocations;
    std::unordered_map<VkFormat, vk::RenderPass> clear_render_passes;
    /// UNORM copy of the present source: what the blit reads for an sRGB texture (it would
    /// otherwise decode it to linear) and what the presentation passes sample through `view`.
    struct PresentScratch {
        vk::Image image;
        vk::ImageView view;
        VkFormat format{VK_FORMAT_UNDEFINED};
        u32 width{};
        u32 height{};
    } present_scratch;
    std::shared_ptr<WindowAdaptPass> window_filter;
    Settings::ScalingFilter window_filter_setting{static_cast<Settings::ScalingFilter>(~0u)};
    VkFormat window_filter_format{VK_FORMAT_UNDEFINED};
    VkExtent2D window_filter_extent{};
    bool window_filter_failed{};
    std::shared_ptr<::Vulkan::FSR> window_fsr;
    std::shared_ptr<::Vulkan::FSR3> window_fsr3;
    std::shared_ptr<::Vulkan::SGSR> window_sgsr;


    // --- counters and render conditions (NvnIndirect; nvn_counters.cpp, nvn_exec.cpp) ---
    std::unique_ptr<CounterEngine> counters_engine;
    /// Set once the engine exists; read by Stats()/DescribeState from other threads.
    std::atomic<CounterEngine*> counters_published{nullptr};
    bool counters_failed{false};
    /// 1D indirect mesh-task argument conversion (nvn_mesh_indirect.h), created at the first
    /// such draw on the replay thread; null when the device cannot run it.
    std::unique_ptr<MeshIndirectGpu> mesh_indirect;
    bool mesh_indirect_failed{false};
    [[nodiscard]] MeshIndirectGpu* MeshIndirect();
    /// Indirect QUADS argument conversion (nvn_quad_indirect.h), created the same way.
    std::unique_ptr<QuadIndirectGpu> quad_indirect;
    bool quad_indirect_failed{false};
    [[nodiscard]] QuadIndirectGpu* QuadIndirect();
    /// Vertex staging copies by shader (nvn_vertex_stage_gpu.h), created at the first staged pass
    /// on the replay thread; null when the device cannot run it (vkCmdCopyBuffer then).
    std::unique_ptr<VertexStageGpu> vertex_stage;
    bool vertex_stage_failed{false};
    [[nodiscard]] VertexStageGpu* VertexStage();
    /// Word merge of GPU writes back into guest memory (nvn_write_back.h), created at the first
    /// pool mirror write-back on the replay thread; null when the device cannot run it (plain
    /// copies then).
    std::unique_ptr<WriteBackMerge> write_back_merge;
    bool write_back_merge_failed{false};
    [[nodiscard]] WriteBackMerge* WriteBackMerger();
    // --- pool mirrors (EDEN_NX2_NVN_POOL_MIRROR, nvn_pool_mirror.cpp; replay thread only) ---
    /// The VRAM mirror of an imported pool; created when `create` and none exists yet. Null when
    /// mirrors are off, the pool is not imported or not page aligned, guest memory cannot track
    /// CPU writes, the mirror budget is used up or the allocation failed (the pool stays in
    /// guest memory, as before).
    /// `storage`: the mirror is for storage bindings and the raw global table (the default); false
    /// makes a read-only one (vertex, index and uniform ranges: PoolMirror::storage), and a
    /// written storage range (`storage` with `create`) promotes a read-only mirror.
    PoolMirror* VramMirror(Pool& pool, bool create, bool storage = true);
    /// Turns a mirror into a storage mirror: write-back monitor and merge guard.
    void AttachStorageState(Pool& pool, PoolMirror& mirror);
    /// Moves the pages of [offset, offset + size) the guest CPU wrote since they were copied in
    /// into the mirror's invalid set (skips the tracker while its write epoch is unchanged since a
    /// whole-pool collect).
    void MirrorCollectCpuWrites(Pool& pool, PoolMirror& mirror, u64 offset, u64 size);
    void MirrorArmRuns(Pool& pool, PoolMirror& mirror, VC::GuestMemory* memory, size_t first_new,
                       const std::vector<std::pair<u64, u64>>& runs);
    /// MirrorTakeStale over the whole pool, for the pages uniform bindings used (`wanted`).
    void MirrorTakeStaleWanted(Pool& pool, std::vector<std::pair<u64, u64>>& runs);
    void MirrorMarkPrefetch(Pool& pool, u64 offset, u64 size);
    /// A draw stores through raw global memory: read-only mirrors do not hold the raw table's
    /// target, so every page of every one of them is copied in again at its next use. With
    /// `written_through` (the draw's stores also wrote the read-only mirror table) only the
    /// mirrors that table left out (PoolMirror::written_through false) are.
    void MirrorInvalidateReadOnly(bool written_through);
    /// The mirror holds the current bytes of [offset, offset + size): every page was copied in
    /// and the guest CPU wrote none since. Records nothing (safe inside a render pass).
    [[nodiscard]] bool MirrorCurrent(Pool& pool, u64 offset, u64 size);
    /// The page runs of [offset, offset + size) the mirror must copy in before the GPU reads the
    /// range through it (never copied, or written since by the guest CPU or the replay through the
    /// pool's own buffer). They count as current from here on and their CPU writes are tracked;
    /// the caller records the copies before any GPU work it records next.
    void MirrorTakeStale(Pool& pool, u64 offset, u64 size, std::vector<std::pair<u64, u64>>& runs);
    /// Something other than the mirror wrote the pool range: copy it in again at the next use.
    void MirrorInvalidate(Pool& pool, u64 offset, u64 size);
    /// Raw global stores may have written any pool.
    void MirrorInvalidateAll();
    /// The pool is destroyed: stop tracking its pages (its mirror goes with the pool).
    void ReleaseVramMirror(Pool& pool);
    /// Pools with a VRAM mirror (MirrorInvalidateAll) and their bytes (budget).
    std::vector<Pool*> vram_mirrored;
    /// Subset of `vram_mirrored` whose `prefetch` flag is set (`PrefetchMirrors`).
    struct PrefetchMirrorEntry {
        Pool* pool{};
        PoolMirror* mirror{};
        const std::atomic<u64>* write_epoch{};
    };
    std::vector<PrefetchMirrorEntry> vram_prefetch_mirrors;
    std::atomic<u64> vram_mirror_bytes{0}; ///< read by DescribeState
    bool vram_mirror_no_tracker_logged{false};
    /// EDEN_NX2_NVN_WB_MONITOR: counters of every mirror created (never removed; DescribeState).
    std::mutex mirror_monitors_mutex;
    std::vector<std::shared_ptr<MirrorMonitor>> mirror_monitors;
    /// Merge counters (2 words per MirrorMonitor::slot: changed words, dirty 32-word lines), host
    /// visible; created with the merge pipeline.
    vk::Buffer mirror_merge_stats;
    static constexpr u32 MIRROR_STAT_SLOTS = 256;
    /// Counters of the line-mode write-back guards of every mirror (DescribeState).
    std::shared_ptr<LineGuardStats> line_guard_stats{std::make_shared<LineGuardStats>()};
    /// ContextOptions::pool_mirror, else EDEN_NX2_NVN_POOL_MIRROR (set at construction): 0 off,
    /// 1 written storage ranges create mirrors, 2 every storage range does.
    u32 pool_mirror_mode{0};
    /// EDEN_NX2_TRY_UBO_MIRROR with pool mirrors on: read-only uniform ranges of imported pools
    /// create and use pool mirrors (DrawContext::PrefetchMirrors keeps them current).
    bool ubo_mirror{false};
    /// EDEN_NX2_TRY_VTX_MIRROR with pool mirrors on: the vertex and index ranges a pass stages
    /// are served from (read-only) pool mirrors, which StageVertexRanges creates and keeps
    /// current page by page.
    bool vtx_mirror{false};
    /// ContextOptions::vtxcov, else EDEN_NX2_TRY_VTXCOV_*: VtxCovSwitch bits (nvn_vtxcov.h); 0
    /// unless vtx_mirror.
    u32 vtxcov{0};
    /// ContextOptions::vtxcov_diag, else EDEN_NX2_NVN_VTXCOV_DIAG: vertex/index binds are counted
    /// by VtxCov (`vtxcov_counts`) and logged every 10 s of presents (VtxCovOnPresent).
    bool vtxcov_diag{false};
    std::array<std::array<std::atomic<u64>, VTXCOV_REASONS>, 2> vtxcov_binds{};
    std::array<std::array<std::atomic<u64>, VTXCOV_REASONS>, 2> vtxcov_bytes{};
    /// The counters, copy-in and staging bytes and presents at the last `[nvn] vtxcov:` line.
    VtxCovCounts vtxcov_logged{};
    u64 vtxcov_logged_copy_in{};
    u64 vtxcov_logged_staged{};
    u64 vtxcov_logged_presents{};
    std::chrono::steady_clock::time_point vtxcov_logged_at{};
    void NoteVtxCov(bool index, VtxCov reason, u64 bytes) {
        const size_t r = static_cast<size_t>(reason);
        vtxcov_binds[index ? 1 : 0][r].fetch_add(1, std::memory_order_relaxed);
        vtxcov_bytes[index ? 1 : 0][r].fetch_add(bytes, std::memory_order_relaxed);
    }
    [[nodiscard]] VtxCovCounts VtxCovSnapshot() const;
    /// Every present (replay thread): the `[nvn] vtxcov:` line once 10 s passed since the last.
    void VtxCovOnPresent();
    /// Why VramMirror(pool, true, false) makes no read-only mirror for `pool` (EDEN_NX2_NVN_VTXCOV_DIAG).
    [[nodiscard]] VtxCov ReadOnlyMirrorMiss(const Pool& pool) const;
    /// ContextOptions::try_storage_vram, else EDEN_NX2_TRY_STORAGE_VRAM.
    bool try_storage_vram{false};
    u64 try_storage_vram_max_bytes{u64{64} << 20};
    /// ContextOptions::dispatch_order_by_barriers, else EDEN_NX2_TRY_DISPATCH_ORDER (set at
    /// construction): a dispatch after a dispatch waits for its shader writes only when a game
    /// barrier (or a command of the replay) came in between (Recorder::HostMarkDispatchWritten).
    bool dispatch_order_by_barriers{false};
    /// EDEN_NX2_NVN_RAW_SPLIT_IMPORT (on unless =0; set at construction): a CPU-visible pool whose
    /// guest storage is several host runs imports each run (BackPoolLocked, also on the backing
    /// thread), so raw global memory reaches it.
    std::atomic<bool> raw_split_import{true};
    /// Bumped when a submission that stored to raw global memory completes: split pools' mirror
    /// pages trusted before may be stale (Recorder::MirrorTakeCpuWrites).
    std::atomic<u64> split_raw_epoch{0};
    /// Monotonic epoch captured from `split_raw_epoch` when a raw-writing submission without full
    /// write-through completes; split pools included in `raw_write_through_table` stay valid past
    /// write-through-only raw writes.
    std::atomic<u64> split_raw_epoch_unaliased{0};
    /// BarrierMergeMask of ContextOptions::barrier_merge, else EDEN_NX2_TRY_BARRIER_MERGE (set at
    /// construction); 0 = off.
    u32 barrier_merge{0};
    /// ContextOptions::flush_tracked, else EDEN_NX2_TRY_FLUSH_TRACKED (set at construction).
    bool flush_tracked{false};
    /// ContextOptions::mirror_hash_tracked, else on unless EDEN_NX2_NVN_MIRROR_HASH_ALWAYS (set at
    /// construction): mirror pools hash only the guest pages the page tracker saw written.
    bool mirror_hash_tracked{false};
    /// On with `mirror_hash_tracked` unless EDEN_NX2_NVN_TEX_HASH_ALWAYS (set at construction):
    /// a CPU-sourced texture whose pages the page tracker saw unwritten since its last full hash is
    /// not hashed again (DetectGuestChange).
    bool tex_hash_tracked{false};
    /// On with `tex_hash_tracked` unless EDEN_NX2_NVN_TEX_PAGE_HASH=0: a texture whose pages were
    /// written since it was trusted rehashes only those pages (Storage::page_hashes).
    bool tex_page_hash{false};
    /// A tracked guest flush lets CPU-sourced storages keep going by the page tracker.
    bool tex_flush_trusted{false};
    /// EDEN_NX2_TRY_INVALIDATE_WRITEBACK (ContextOptions / nvn_debug.h): milliseconds a guest
    /// invalidate waits for the GPU's unfinished writes to its range; 0 = off (nothing is noted).
    u32 invalidate_wait_ms{0};
    /// The GPU writes of unfinished submissions (Recorder::NotePoolWrite; only with the switch).
    GpuWriteLog gpu_writes;
    /// Commands enqueued / dispatched (Sink::GpuWriteMarker, PollGpuWrites).
    std::atomic<u64> commands_enqueued{0};
    std::atomic<u64> commands_dispatched{0};
    /// ContextOptions::raw_keep_entries, else EDEN_NX2_TRY_RAW_KEEP_ENTRIES (set at construction).
    bool raw_keep_entries{false};
    /// ContextOptions::refused_shadow, else EDEN_NX2_TRY_REFUSED_SHADOW (set at construction).
    bool refused_shadow{false};
    bool vram_mirror_budget_logged{false};
    /// EDEN_NX2_NVN_COSTSPLIT (measurement only): null when off. Created before the profiler that
    /// reports into it, and destroyed after it.
    std::unique_ptr<CostSplit::Recorder> costsplit;
    /// EDEN_NX2_NVN_GPU_PROFILE: null when off. Created with the context, destroyed after the flights.
    std::unique_ptr<GpuProf::Profiler> gpu_profiler;
    /// ASTC decode on the GPU (nvn_astc_gpu.h): enabled when the device can run it. The decoder is
    /// created at the first use (replay thread); null when creation failed.
    bool astc_gpu_enabled{false};
    std::unique_ptr<AstcGpuDecoder> astc_decoder;
    /// The decoder's pipeline compiles in ~400 ms: it is created on a worker at startup and a
    /// storage uploaded before it is ready decodes on the CPU.
    std::future<std::unique_ptr<AstcGpuDecoder>> astc_decoder_future;
    bool astc_decoder_failed{false};
    [[nodiscard]] AstcGpuDecoder* AstcDecoder();
    /// What ASTC stand-ins hold (nvn_format.h AstcStandin): EDEN_NX2_NVN_ASTC_STANDIN over the
    /// astc_recompression setting, Rgba8 on devices that hold ASTC or cannot run the BC encoder.
    AstcStandin astc_standin{AstcStandin::Rgba8};
    /// The BC encoder of the BC stand-ins (nvn_bc_encode.h), compiled on a worker at startup like
    /// the ASTC decoder; BcEncoder() waits for it (a BC stand-in has no other way to get texels).
    std::unique_ptr<BcGpuEncoder> bc_encoder;
    std::future<std::unique_ptr<BcGpuEncoder>> bc_encoder_future;
    bool bc_encoder_failed{false};
    [[nodiscard]] BcGpuEncoder* BcEncoder();
    /// Replay thread: LRU pool of scratch BC stand-in images (up to 4 shapes within BC_SCRATCH_BUDGET).
    std::vector<std::shared_ptr<BcScratch>> bc_scratches;
    bool bc_scratch_failed_logged{false};
    [[nodiscard]] std::shared_ptr<BcScratch> AcquireBcScratch(u32 width, u32 height, u32 layers);
    /// EDEN_NX2_NVN_BC7_QUALITY (nvn_bc_encode.h) for BC7 stand-ins.
    u32 bc7_quality{BC7_QUALITY_DEFAULT};
    /// The smart BC stand-in selection (EDEN_NX2_NVN_SMART_BC*, nvn_format.h SmartBc).
    SmartBc smart_bc{};
    /// Smart stand-ins: classifications by Tegra::Texture::ASTC::Channels (index 4: all-zero
    /// bytes), the guest bytes they scanned and the time it took, stand-ins recreated as BC3
    /// because new bytes did not fit (RefitStandin) or pinned there by a copy, and the same for
    /// the window ProcessPresent logs.
    struct SmartBcCounts {
        std::array<std::atomic<u64>, 5> classes{};
        std::atomic<u64> scan_bytes{};
        std::atomic<u64> scan_ns{};
        std::atomic<u64> refits{};
        std::atomic<u64> pinned{};
    };
    SmartBcCounts smart_total;
    SmartBcCounts smart_window;
    /// Classifies `size` guest bytes of smart stand-in `s` (counted, timed); `empty` when all
    /// were zero.
    Tegra::Texture::ASTC::Channels ClassifyStandin(const Storage& s, const u8* data, u64 size,
                                                   bool& empty);
    /// Chooses the format of smart stand-in `s` from the guest bytes under it before its image is
    /// created (EnsureImage); BC3 when they cannot be read.
    void ChooseSmartStandin(Storage& s);
    /// Stand-in `s` (no image) as `standin`, its memory estimate updated.
    void SetStandin(Storage& s, AstcStandin standin);
    /// GPU time of the BC stand-in uploads (BcUploadTimes, timestamps per submission): false when
    /// the graphics queue has no timestamps. CompleteBcTimes (completion thread) logs the uploads
    /// of BC_TIMED_LOG_BYTES or more and adds every one to the window ProcessPresent logs.
    bool bc_timing{false};
    double bc_timestamp_period{1.0};
    u64 bc_timestamp_mask{~u64{0}};
    struct BcTimeWindow {
        std::atomic<u64> uploads{};
        std::atomic<u64> untimed{};
        std::atomic<u64> bytes{};
        std::atomic<u64> gpu_ns{};
        std::atomic<u64> max_ns{};
    } bc_window;
    u64 bc_window_first_present{};
    void CompleteBcTimes(const BcUploadTimes& times);
    void LogBcTimeWindow(u64 present);
    /// Block-linear untiling on the GPU (nvn_untile_gpu.h), created like the ASTC decoder on a
    /// worker at startup; uploads before it is ready untile on the CPU. Null when creation failed.
    bool untile_gpu_enabled{false};
    std::unique_ptr<UntileGpu> untile_gpu;
    std::future<std::unique_ptr<UntileGpu>> untile_gpu_future;
    [[nodiscard]] UntileGpu* Untile();
    /// Alias transfers between textures of different layouts (nvn_alias_gpu.h): created on a
    /// worker at startup (the pipeline compiles slowly), waited for at the first use.
    std::unique_ptr<AliasTransferGpu> alias_gpu;
    std::future<std::unique_ptr<AliasTransferGpu>> alias_gpu_future;
    [[nodiscard]] AliasTransferGpu* AliasGpu();
    /// Replay thread: the engine, created on first use; null when it cannot be created.
    [[nodiscard]] CounterEngine* Counters();
    /// Blocks until every submission up to `seq` completed; false after `timeout` or when the
    /// context is stopping.
    bool WaitCompleted(u64 seq, std::chrono::milliseconds timeout);
    /// The presentation passes' waits (PassRecorder::ReplayHooks): the flight numbered `seq`, which
    /// the replay has submitted, has completed / every submitted flight except the open one
    /// numbered `seq` has completed. Replay thread only.
    void WaitFlight(u64 seq);
    void WaitPreviousFlights(u64 seq);
    /// Blocks until every timeline semaphore reached its value (returns early when stopping).
    void WaitTimelines(std::span<const VkSemaphore> semaphores, std::span<const u64> values);

    // --- Sink plumbing (nvn_context.cpp) ---
    void Enqueue(Command&& command, std::shared_ptr<InFlight> flight);
    void ReplayMain();
    void CompleteMain();
    void Dispatch(Command& command);
    void RetireCompleted();
    void FailFlight(InFlight& flight);
    void MarkSubmitted(InFlight& flight, VkSemaphore semaphore, u64 value);
    struct SubmitBatch {
        struct Entry {
            VkTimelineSemaphoreSubmitInfo timeline{};
            std::array<VkSemaphore, 16> signals{};
            std::array<u64, 16> signal_values{};
            std::vector<VkSemaphore> wait_semaphores;
            std::vector<u64> wait_values;
            std::vector<VkPipelineStageFlags> wait_stages;
            VkCommandBuffer cmd{VK_NULL_HANDLE};
            u32 signal_count{0};
            u64 flight_seq{0};
            InFlight* flight{nullptr};
            VkSemaphore queue_timeline{VK_NULL_HANDLE};
            u64 queue_signal_value{0};
        };
        struct Mark {
            InFlight* flight{};
            VkSemaphore semaphore{VK_NULL_HANDLE};
            u64 value{0};
        };
        std::vector<Entry> entries;
        std::vector<Mark> marks;
        bool batching{false};
    } submit_batch;
    void FlushSubmitBatch();
    /// The Vulkan device was lost at `where`: logs it (once), reports the loss to the device
    /// (Aftermath) and tells the owner to stop the emulation. Callable from any context thread.
    void NoteDeviceLost(const char* where);
    /// `exception` is a Vulkan error: a lost device is noted, any other error is left to the caller.
    /// Call from inside a catch handler: notes the loss when the exception being handled is a
    /// VK_ERROR_DEVICE_LOST.
    void NoteIfDeviceLost(const char* where);
    /// Replay thread, after a loss: names the pool, chunk or raw global range of ours that holds
    /// each address VK_EXT_device_fault reports (or the nearest ones).
    void DescribeDeviceFault();
    /// Pools with a device address that DestroyNow retired last (replay thread only): a fault
    /// address in one of them is a use after the pool's destruction.
    struct RetiredAddress {
        u64 serial;
        u64 guest;
        u64 device;
        u64 size;
        u64 last_use_seq;
        u64 destroyed_in_seq;
    };
    std::array<RetiredAddress, 64> retired_addresses{};
    u64 retired_address_count{0};

    QueueState& GetQueue(u32 index);
    /// The draw state of NVN queue `index` (created on first use; replay thread only).
    DrawContext& GetDrawContext(u32 index);
    VkCommandBuffer AcquireCommandBuffer();

    // --- resources (nvn_resources.cpp) ---
    void CreatePoolNow(const VC::PoolDesc& desc);
    /// Replay thread only: imports or allocates the pool's buffer the first time a command needs it.
    void EnsureBacking(Pool& pool);
    /// Uploads the guest contents of GPU-only pools (buffers exist, `upload_pending`) in batches:
    /// one staging buffer and one submission for many pools. Replay thread.
    void UploadWholePools(std::span<Pool* const> targets);
    /// The host range a pool's chunk import covers: guest [guest_low, guest_low + size) at `host`.
    struct ChunkWindow {
        u8* host;
        u64 guest_low;
        u64 size;
    };
    /// The chunk window around the pool at guest `address` (host pointer `host`), or none when
    /// the pool can only be imported on its own (guest and host addresses not congruent).
    [[nodiscard]] std::optional<ChunkWindow> ImportWindow(u64 address, u64 size, u8* host) const;
    /// One guest range of a pool's buffer to upload (UploadPoolRanges).
    struct PoolUpload {
        Pool* pool;
        u64 offset;
        u64 size;
    };
    /// Copies the guest bytes of the ranges into the pools' buffers, batched like
    /// UploadWholePools. `in_use`: the pools may be in use by earlier submissions (barriers around
    /// the copies). False when the upload failed. Replay thread.
    bool UploadPoolRanges(std::span<const PoolUpload> ranges, bool in_use);
    /// Binds the pool's buffer into a cached (or newly imported) large chunk of guest memory.
    VkResult ImportPoolBuffer(Pool& pool, u8* host, std::string& error);
    void CreateTextureNow(const VC::TextureDesc& desc);
    /// The texture object of `desc` with its storage (CreateTextureNow without registering it).
    [[nodiscard]] std::shared_ptr<Texture> BuildTexture(const VC::TextureDesc& desc);
    /// A render target `ref` names through a block view of a compressed texture (IsBlockView:
    /// a GPU block encoder writes whole blocks) is drawn into a texture of the view format over
    /// the level's bytes (BlockViewAliasOf); its contents reach the compressed texture through the
    /// overlap transfers. On success `ref` is rewritten to name the alias in its own format: level
    /// 0, layer 0 of a 2D level's alias, the same slices of a 3D level's. Null (and `ref` left
    /// alone) when `ref` is no such view, its level and layer have no alias, or it is a view of
    /// several layers of a 2D array (one alias per layer). Replay thread.
    [[nodiscard]] std::shared_ptr<Texture> TargetAlias(const Texture& texture,
                                                       VC::RenderTargetRef& ref);
    /// TargetAlias textures by (texture serial, level, layer (0 for 3D), view format); they go
    /// with the texture (DestroyNow). Their serials (for the view cache) count down from 2^64 - 1.
    std::map<std::tuple<u64, u32, u32, s32>, std::shared_ptr<Texture>> target_aliases;
    u64 next_alias_serial{~u64{0}};
    void DestroyNow(u64 serial);
    /// Pool range lookup: the pool containing [address, address + size) and the offset in it.
    bool ResolveBuffer(u64 address, u64 size, Pool*& pool, u64& offset) const;
    [[nodiscard]] vk::Buffer CreateHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                              MemoryUsage usage_hint,
                                              AllocationKind kind = AllocationKind::Other) const;
    /// Records and runs one-time commands on the graphics queue, waiting for them.
    void ImmediateSubmit(const std::function<void(vk::CommandBuffer)>& record);
    /// Host pointer for [address, address + size) when it is one contiguous mapping, else null.
    [[nodiscard]] const u8* GuestPointer(u64 address, u64 size) const;
    /// Number of host runs (maximal ranges of consecutive host pages) in the guest range, and the
    /// offset of the first break; diagnostics for pools that cannot be imported whole.
    void CountHostRuns(u64 address, u64 size, u64& runs, u64& first_break) const;
    /// Imports each host run of a CPU-visible pool's guest storage as its own buffer
    /// (Pool::split_runs). False, with no runs kept, when a run cannot be imported. Under
    /// backing_mutex.
    bool ImportSplitRuns(Pool& pool, std::string& error);

    // --- submissions (nvn_exec.cpp) ---
    void ProcessSubmit(CmdSubmit&& submit);
    void ProcessPresent(CmdPresent&& present);
    /// Runs everything the streams can run; with `release_old` the front items blocked for longer
    /// than the timeout go on without what blocks them.
    void PumpStreams(bool release_old);
    void NotifyEventCpuNow(const CmdEventCpu& command);
    void ExecutePresent(const VC::PresentRequest& request, InFlight& flight);
    /// `missing` receives the first waited-for sync nobody has fenced, when false is returned.
    [[nodiscard]] bool DependenciesKnown(const VC::Submission& submission,
                                         u64* missing = nullptr) const;
    void NoteUse(u64 serial, u64 seq);
    static void NoteUse(Pool& pool, u64 seq) noexcept {
        if (seq > pool.last_use_seq) {
            pool.last_use_seq = seq;
        }
    }
    static void NoteUse(Texture& texture, u64 seq) noexcept {
        if (seq > texture.last_use_seq) {
            texture.last_use_seq = seq;
        }
    }

    /// Virtual pool range -> the physical pool and offset backing it, when one physical range
    /// covers all of it.
    bool ResolveVirtualRange(const Pool& virtual_pool, u64 offset, u64 size, Pool*& physical,
                             u64& physical_offset) const;

    // --- storages, virtual pools, CPU coherency (nvn_storage.cpp) ---
    /// Takes a recycled non-sparse VkImage matching `info`, or an empty handle on miss.
    [[nodiscard]] vk::Image TakeRecycledImage(const VkImageCreateInfo& info);
    /// Moves a completed, non-sparse storage's VkImage into `image_recycle_pool`.
    void RecycleImage(Storage& s);
    /// Releases every VkImage held in `image_recycle_pool`.
    void ClearRecycledImages();
    /// Creates the VkImage of `s` on first use. False when it cannot be created (logged once).
    bool EnsureImage(Storage& s);
    /// Bytes [offset, offset + size) of `pool` as the guest sees them: a contiguous host pointer
    /// when there is one, else a copy in `scratch` (virtual pools are gathered from the physical
    /// pools; unmapped parts are zero). Null when guest memory cannot be read.
    [[nodiscard]] const u8* GuestBytes(const Pool& pool, u64 offset, u64 size,
                                       std::vector<u8>& scratch) const;
    /// Copies the bytes [offset, offset + size) of a virtual pool into `out` (already zeroed): the
    /// mapped parts come from the physical pools.
    void GatherVirtual(const Pool& pool, u64 offset, u64 size, u8* out) const;
    void MapVirtualNow(u64 pool_serial, const std::vector<VC::VirtualMapping>& mappings);
    /// True when a run of the virtual pool covers any byte of [offset, offset + size).
    [[nodiscard]] static bool VirtualRangeMapped(const Pool& virtual_pool, u64 offset, u64 size);
    /// Compares a few of the sparse storage's tiles with the bytes behind them (the CPU may have
    /// rewritten them without a flush) and marks the ones that changed. Replay thread.
    void VerifySparse(Storage& s);
    /// A guest flush (`host_write` false) or a host service's write (NotifyHostWriteNow) of
    /// [address, address + size). With `flush_tracked` a guest flush of a pool mirror only
    /// refreshes what the page tracker saw written or the mirror does not hold (FlushedRuns).
    void NotifyCpuWriteNow(u64 address, u64 size, bool host_write = false);
    /// The part of the pool range [begin, begin + size) a guest flush has to act on (pool offsets,
    /// byte runs clipped to the range): the pages of the range the guest wrote since the mirror
    /// last took them (taken here: the caller invalidates them) and the pages the mirror does not
    /// hold. Pages that are valid in the mirror and clean were not written by the guest CPU:
    /// its stores all go through the tracker.
    void FlushedRuns(Pool& pool, u64 begin, u64 size, std::vector<std::pair<u64, u64>>& runs);
    /// Sink::NotifyHostWrite: a GPU-only pool's buffer takes the range's guest bytes, a mirror
    /// compares them again, and every storage inside the range is uploaded again even over what
    /// the GPU wrote. Replay thread.
    void NotifyHostWriteNow(u64 address, u64 size);
    void NotifyTextureWrittenNow(u64 texture_serial);
    /// Settles a pending GPU write-back on a linear storage once its submission has completed,
    /// refreshing `gpu_guest_sig` and re-arming `takeover_track` against the written-back bytes.
    bool SettleLinearWriteback(Storage& s);
    /// Looks at a CPU-sourced storage's guest bytes when its check is due (a cheap signature, no
    /// commands recorded) and sets `force_upload` when they changed. At most once per `seq`.
    void DetectGuestChange(Storage& s, u64 seq);
    /// Registers `pool` with the page tracker at the first call (every page armed) and folds the
    /// guest CPU writes since the last call into `mirror_pages` (untrusted) and `page_write_gen`.
    /// False when the pool is not tracked. Replay thread.
    bool TakePoolCpuWrites(Pool& pool);
    /// Whether a page of [offset, offset + size) of `pool` was found written after `gen`.
    [[nodiscard]] static bool PoolWrittenSince(const Pool& pool, u64 offset, u64 size, u32 gen);
    /// True when `Recorder::EnsureContents(s, will_overwrite)` would record commands (a first
    /// upload, a CPU change, an overlapping texture written more recently). Creates the image and
    /// runs the change check, so draw code can decide before it begins a render pass. `seq` is the
    /// submission's seq. Replay thread only.
    /// `scaled_ok`: a guest storage whose twin is the newer image does not count as needing
    /// commands (the caller reads the twin); without it such a storage does (its guest-sized
    /// image has to be brought up to date: Recorder::EnsureContents downscales the twin).
    [[nodiscard]] bool StorageNeedsCommands(Storage& s, bool will_overwrite, u64 seq,
                                            bool scaled_ok = false);

    // --- Resolution scaling (nvn_rescale.h, nvn_rescale.cpp) ---
    /// The Resolution setting (Settings::values.resolution_info is fixed once per boot).
    [[nodiscard]] Rescale::Factor RescaleFactor() const noexcept {
        return Rescale::CurrentFactor();
    }
    /// What the eligibility rules see of a storage a render target draws into (`alias`: it is the
    /// block-view alias texture of a compressed one, not the game's own).
    [[nodiscard]] Rescale::Shape RescaleShapeOf(const Storage& s, bool alias) const;
    /// The twin of `s` (created on first call, without its image); null when `s` cannot have one.
    [[nodiscard]] Storage* ScaledTwinOf(Storage& s);
    /// The twin texture object of `texture` (a Texture over the twin storage, same description:
    /// guest sizes); null when its storage has no twin.
    [[nodiscard]] std::shared_ptr<Texture> ScaledTextureOf(Texture& texture);
    /// Scene sizes of the scaled passes so far (see Rescale::PassScaled). Replay thread only.
    Rescale::SceneSizes rescale_scene_sizes;
    /// The guest-sized image of `s` made current from its twin, outside a recording (ReadTexels).
    /// False when it could not be done.
    bool DownscaleNow(Storage& s);
    /// Records the blit that makes one image of a storage the other's scaled / guest-sized copy
    /// (`to_scaled`: guest -> twin). Both images exist; layouts are the caller's business (the
    /// source is read in TRANSFER_SRC_OPTIMAL, the destination written in TRANSFER_DST_OPTIMAL).
    void RecordScaleBlit(vk::CommandBuffer cmd, Storage& native, bool to_scaled) const;
    /// Replay thread: reads the texture's image back into guest memory (CmdFlushTexture).
    void FlushTextureNow(CmdFlushTexture& command);
    /// Drops cached storages nobody uses once the cache is over budget (least recently used first)
    /// or the device over its memory budget, and idle render targets past the age limit.
    /// `at_present`: the periodic pass of a present (the cache need not have grown).
    void TrimStorageCache(bool at_present = false, u64 retire_seq = 0);
    /// True when the GPU wrote every byte of the storage's span again after the storage's own last
    /// write, through other storages of the pool that took its contents over first (overlap
    /// resolution at their use): its image holds nothing the pool's bytes do not, so a later
    /// texture over the same bytes gets the same contents from those storages. Replay thread only.
    [[nodiscard]] bool StorageSuperseded(const Storage& s) const;
    /// Logs where the device memory of the storages goes (every 900 presents): per kind, and the
    /// largest ones.
    void LogStorageCensus();
    /// Removes every storage of the pool from the cache and the pool index.
    void EvictPoolStorages(u64 pool_serial);
    /// Why `address` does not resolve, for the log (virtual pools: unmapped or scattered).
    [[nodiscard]] std::string DescribeUnresolved(u64 address, u64 size) const;
};

} // namespace Vulkan::Nvn
