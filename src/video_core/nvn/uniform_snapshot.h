// SPDX-FileCopyrightText: Copyright (c) 2026 jhaizhou (d2r_gfx)
// SPDX-FileCopyrightText: Copyright (c) 2026 MacNativeGames contributors
// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Uniform snapshot at submit (EDEN_NX2_NVN_UBO_SNAPSHOT), the submit side. On the console the GPU
// reads a submission's uniform blocks soon after nvnQueueSubmitCommands returns; the replay runs a
// frame or more behind, while the game rewrites its per-object blocks (transforms, skinning, light
// constants) for the next frame right after submitting. So the submit call copies every uniform
// range the submission binds out of guest memory (nearby ranges of one pool merged) into the
// Submission, and the replay binds the copies (renderer_vulkan/nvn/nvn_uniform_snapshot.h).
//
// Only bytes the guest CPU owns are copied. MacNativeGames runs on a host where every GPU write
// lands in guest memory; here GPU results can sit in a VRAM mirror until a merge at the end of the
// command buffer, and raw global stores name no range at all. So a binding is copied only when
// every page it touches was written by the guest CPU in this frame or the previous one (the page
// tracker the pool mirrors use, armed per page) and no page of it has GPU write history
// (UniformSnapshotShared: every GPU write the replay noted, and in the strict mode every guest
// range a raw-writing draw could reach). Everything else is bound live, as without the snapshot.
//
// After MacNativeGames' TotK NVN layer (games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m at
// 9fe3fdc1d0: usnap_collect, rng_sort, mtl_sub_capture_ubos, usnap_take; MIT, jhaizhou and
// MacNativeGames contributors), re-implemented on ShiftX2's record format.

#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/nvn/sink.h"

namespace VideoCommon::Nvn {

/// EDEN_NX2_NVN_UBO_SNAPSHOT / game patch oot-ubo-snapshot.
enum class UniformSnapshotMode : u32 {
    Off = 0,
    /// Guest ranges a raw-writing draw could reach are GPU history (never copied).
    Strict = 1,
    /// Raw stores are left to the CPU-written test alone (A/B: a game whose every pool is in the
    /// raw table gets no copy at all in the strict mode).
    CpuOnly = 2,
};

/// State the submit threads and the replay share: the mode, and the guest pages the GPU has
/// written (sticky until their pool is finalized). Process-wide; the recorder resets it when it
/// is created. Thread safe.
class UniformSnapshotShared {
public:
    void Reset(UniformSnapshotMode mode);
    [[nodiscard]] UniformSnapshotMode Mode() const noexcept {
        return mode.load(std::memory_order_relaxed);
    }
    /// The GPU wrote [begin, end) (rounded out to pages).
    void NoteGpuWrite(u64 begin, u64 end);
    /// The pool [begin, end) is gone: its history goes with it.
    void Forget(u64 begin, u64 end);
    /// Removes the ranges (sorted by begin) that touch a page with GPU history; returns how many.
    u64 ExcludeGpuWritten(std::vector<UniformSnapshot::Range>& ranges) const;
    [[nodiscard]] size_t Runs() const;

private:
    std::atomic<UniformSnapshotMode> mode{UniformSnapshotMode::Off};
    mutable std::mutex mutex;
    std::map<u64, u64> written; ///< page-aligned begin -> end, disjoint, not touching
};

[[nodiscard]] UniformSnapshotShared& SharedUniformSnapshot();

/// A memory pool a snapshot may copy from: a physical pool the CPU can access (not
/// CPU_NO_ACCESS, whose bytes only the GPU holds, and not virtual, whose addresses are not guest
/// addresses), with its page tracker registration (made at its first capture; 0 = none).
struct UniformSnapshotPool {
    u64 end{};
    u64 track_id{};
    bool track_tried{};
    std::shared_ptr<const std::atomic<u64>> write_epoch;
    u64 epoch_clean{};
};
/// Guest VA of the storage -> pool. Owned by the recorder, under its sink mutex.
using UniformSnapshotPools = std::map<u64, UniformSnapshotPool>;

struct UniformSnapshotPageState {
    u64 written{}; ///< 1 + present count when last seen CPU-written; 0 = armed, not seen written
    u64 checked{}; ///< 1 + present count when last checked via TakeWritten
};

/// The guest CPU's writes to the pages of uniform bindings. Owned by the recorder, under its sink
/// mutex.
struct UniformSnapshotCpuPages {
    /// Page index -> written/checked timestamps. A page without an entry is not armed.
    std::unordered_map<u64, UniformSnapshotPageState> written;
    /// Presents so far (QueuePresentTexture).
    u64 presents{};
};

/// Reusable scratch of the capture (the submit thread's; the recorder keeps one).
struct UniformSnapshotScratch {
    std::vector<UniformSnapshot::Range> ranges;
    std::vector<UniformSnapshot::Range> sort_tmp;
    std::vector<std::pair<u64, u64>> runs;
};

/// What the captures did (counters for the log).
struct UniformSnapshotTotals {
    u64 bindings{};    ///< uniform buffer bindings looked at
    u64 not_cpu{};     ///< left live: a page the CPU did not write recently
    u64 gpu_written{}; ///< left live: a page with GPU write history
    u64 ranges{};      ///< ranges copied after the merge
    u64 bytes{};
};

/// Two bindings of one pool at most this far apart are copied as one range...
inline constexpr u64 UNIFORM_SNAPSHOT_MERGE_GAP = 256;
/// ...unless the merged range would grow past this (overlapping bindings are always merged, so a
/// binding never straddles two ranges).
inline constexpr u64 UNIFORM_SNAPSHOT_MERGE_LIMIT = 64 * 1024;
/// The largest uniform binding the replay binds (the rest of a longer one is not read).
inline constexpr u64 UNIFORM_SNAPSHOT_MAX_BINDING = 64 * 1024;

/// Copies the uniform ranges `lists` bind (BindUniformBuffers records with their deferred arrays
/// already resolved) that lie in `pools`, whose pages the CPU wrote recently and the GPU never.
/// `completed_seq` must be read before the call. Null when nothing was copied.
[[nodiscard]] std::shared_ptr<UniformSnapshot> CaptureUniformSnapshot(
    const std::vector<std::vector<u64>>& lists, UniformSnapshotPools& pools,
    UniformSnapshotCpuPages& cpu, const UniformSnapshotShared& shared, GuestMemory& memory,
    u64 completed_seq, UniformSnapshotScratch& scratch, UniformSnapshotTotals& totals);

} // namespace VideoCommon::Nvn
