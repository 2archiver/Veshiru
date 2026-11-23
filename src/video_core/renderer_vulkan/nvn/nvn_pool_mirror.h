// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/nvn/nvn_line_guard.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::Nvn {

enum class MirrorWriteCoverage { None, Partial, Full };

/// Pending GPU writes own these mirror bytes until completion writes them back to the guest.
[[nodiscard]] inline MirrorWriteCoverage ClassifyMirrorWrites(
    u64 begin, u64 end, std::span<const std::pair<u64, u64>> pending) {
    if (begin >= end) {
        return MirrorWriteCoverage::None;
    }
    auto coverage = MirrorWriteCoverage::None;
    for (const auto& [lo, hi] : pending) {
        if (lo >= end || hi <= begin) {
            continue;
        }
        if (lo <= begin && hi >= end) {
            return MirrorWriteCoverage::Full;
        }
        coverage = MirrorWriteCoverage::Partial;
    }
    return coverage;
}

/// Which 4 KiB pages of a pool mirror hold the pool's bytes (DESIGN.md "Pool mirrors"). A valid
/// page holds what the guest page held when it was copied in, plus what the GPU wrote there since
/// through the mirror. A page is invalid before its first copy and after anything wrote the guest
/// page by another way than the mirror (guest CPU, replay writes through the pool's own buffer, raw
/// global memory). Replay thread only.
class MirrorPages {
public:
    static constexpr u64 PAGE = 4096;

    MirrorPages() = default;
    explicit MirrorPages(u64 pool_size);

    [[nodiscard]] u64 Size() const noexcept {
        return size;
    }
    /// Every page [offset, offset + length) touches is valid (false for an empty or out-of-pool
    /// range).
    [[nodiscard]] bool AllValid(u64 offset, u64 length) const;
    /// Marks the pages [offset, offset + length) touches invalid (clamped to the pool).
    void Invalidate(u64 offset, u64 length);
    void InvalidateAll();
    /// Marks the pages [offset, offset + length) touches valid (clamped to the pool).
    void Set(u64 offset, u64 length);
    /// As TakeInvalid over the whole pool, for the pages `wanted` has set (a Set-only bitmap of
    /// the same pool size) that are invalid here.
    void TakeInvalidWithin(const MirrorPages& wanted, std::vector<std::pair<u64, u64>>& out);
    /// Appends the runs [begin, end) of invalid pages that [offset, offset + length) touches (pool
    /// offsets, page aligned; a run at the pool's end stops at the pool size) and marks them valid.
    void TakeInvalid(u64 offset, u64 length, std::vector<std::pair<u64, u64>>& out);
    [[nodiscard]] u64 ValidPages() const;

private:
    /// Pages [first, last] the range touches; false when it is empty or outside the pool.
    [[nodiscard]] bool PageSpan(u64 offset, u64 length, u64& first, u64& last) const;

    std::vector<u64> valid; ///< one bit per page
    u64 size{};
    u64 pages{};
};

/// Counters of EDEN_NX2_NVN_WB_MONITOR for one mirrored pool (diagnostic). The access callback
/// runs on guest threads and holds the object alive, so it outlives the pool.
struct MirrorMonitor {
    u64 address{};
    u64 size{};
    u32 slot{0xFFFFFFFFu};            ///< merge counter slot (Impl::mirror_merge_stats)
    std::atomic<u64> armed_pages{};   ///< pages armed after a merge (replay thread)
    std::atomic<u64> armed_runs{};    ///< arm calls
    std::atomic<u64> touched_pages{}; ///< armed pages the guest accessed (guest threads)
    std::atomic<u64> whole_entries{}; ///< entries over the whole pool (not armed)
    std::atomic<u64> touch_events{};  ///< callback calls
    std::atomic<u64> entry_bytes{};   ///< bytes of the merged entries
    std::atomic<u64> entries{};
    std::atomic<u64> copy_in_bytes{}; ///< guest -> mirror page runs (CPU-written pages)
    std::atomic<u64> notify_calls{};  ///< FlushMappedRange notifications (NotifyCpuWriteNow)
    std::atomic<u64> notify_pages{};  ///< pages those notifications covered
    std::atomic<u64> notify_dirty{};  ///< ... of them written by the guest (page tracker)
    std::atomic<u64> notify_invalid{};///< ... of them already invalid in the mirror
    std::atomic<u64> copy_in_runs{};
};

/// The device-local copy of an imported pool (whole pool, same offsets) that storage bindings and
/// indirect arguments read and write instead of the guest memory over PCIe.
struct PoolMirror {
    vk::Buffer buffer;
    MirrorPages pages;
    /// GuestMemory page-tracker registration of the pool's range (pages are armed when copied in).
    u64 track_id{};
    /// The mirror serves storage bindings (reads and written entries) and the raw global table:
    /// its pool is part of the table and of every raw-writing draw's snapshot. A mirror made only
    /// for read-only uses (vertex, index and uniform ranges, EDEN_NX2_TRY_VTX_MIRROR /
    /// EDEN_NX2_TRY_UBO_MIRROR) is not: raw global pointers keep reaching the guest bytes, and a
    /// raw-writing draw makes all its pages stale instead (Impl::MirrorInvalidateReadOnly). A
    /// written storage range promotes it (Impl::VramMirror).
    bool storage{true};
    /// This mirror is in the current raw-store alias table: read-only mirrors are kept current,
    /// and storage mirrors also write to imported guest memory. False if excluded from the table.
    bool written_through{false};
    /// Made for uniform buffer bindings (EDEN_NX2_TRY_UBO_MIRROR), or a vertex/index range found it
    /// stale at its bind (EDEN_NX2_TRY_VTXCOV_INPASS): every stale page of `wanted` is copied in at
    /// a pass begin (DrawContext::PrefetchMirrors), so the draws of the pass find the pages earlier
    /// bindings used current instead of reading host memory.
    bool prefetch{false};
    /// The pages those bindings used so far (bits set by DrawContext::ApplyMirrors and
    /// VtxCovServeOrMiss): the prefetch's working set, which bounds its copies (null until the
    /// first use).
    std::unique_ptr<MirrorPages> wanted;
    bool wanted_dirty{true};
    /// The tracker's write counter of the pool's registration (GuestMemory::WriteEpoch) and its
    /// value from before the last whole-pool collect of CPU writes: while they are equal no page
    /// of the pool is dirty, and the currency checks skip the tracker.
    std::shared_ptr<const std::atomic<u64>> write_epoch;
    u64 epoch_clean{~u64{0}};
    /// Set with EDEN_NX2_NVN_WB_MONITOR.
    std::shared_ptr<MirrorMonitor> monitor;
    /// Guards the write-back merge's line mode (EDEN_NX2_TRY_WB_LINES); null: exact merge only.
    std::shared_ptr<MirrorLineGuard> line_guard;
};

} // namespace Vulkan::Nvn
