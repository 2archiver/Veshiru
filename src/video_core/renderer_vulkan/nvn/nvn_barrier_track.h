// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// Which buffer ranges the replay's own commands (copies, write-back merges) touched since the last
// memory barrier, so that a barrier in front of the next such command is left out when it touches
// none of them (tests: src/tests/nvn/barrier_track.cpp; EDEN_NX2_TRY_BARRIER_MERGE).
//
// The replay orders every command with one lazy ALL_COMMANDS barrier (Recorder::MemoryBarrier):
// whoever writes marks it pending, whoever needs the writes flushes it. Two commands the replay
// records itself, whose buffers it knows exactly, need no barrier between them when no byte of one
// is read or written by the other: no RAW, WAR or WAW hazard, so Vulkan's missing order cannot
// change a result. The tracker keeps the ranges of the commands since the last emitted barrier
// (a barrier clears it: its first scope holds everything before, its second everything after) and
// answers "would this command conflict". Everything it does not know about (game dispatches,
// draws, images, counters) stays on the old path: the Recorder keeps its pending flag for those
// and flushes whenever the tracker holds anything.
//
// Reads never conflict with reads. A write conflicts with every overlapping access, a read with
// every overlapping write. A range is half open, [begin, end), in bytes of one VkBuffer; callers
// widen ranges to what the shader or copy can reach (whole words, the write-back merge's line
// units).

#pragma once

#include <algorithm>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::Nvn {

/// What EDEN_NX2_TRY_BARRIER_MERGE turns on (bit set, `BarrierMergeMask`).
inline constexpr u32 BARRIER_MERGE_BUFFERS = 1; ///< buffer commands of the replay: ranges decide
inline constexpr u32 BARRIER_MERGE_INLINE = 2;  ///< no barrier around inline updates that copy nothing
inline constexpr u32 BARRIER_MERGE_ADJACENT = 4; ///< requests with no command between them: one barrier

/// The switch's value: 0 off, 1 all of it, 2 buffer commands only, 3 inline updates only, 4 adjacent
/// barrier requests only (to find which part a difference comes from); any other value is all of it.
[[nodiscard]] constexpr u32 BarrierMergeMask(u64 mode) {
    switch (mode) {
    case 0:
        return 0;
    case 2:
        return BARRIER_MERGE_BUFFERS;
    case 3:
        return BARRIER_MERGE_INLINE;
    case 4:
        return BARRIER_MERGE_ADJACENT;
    default:
        return BARRIER_MERGE_BUFFERS | BARRIER_MERGE_INLINE | BARRIER_MERGE_ADJACENT;
    }
}

/// One range of one buffer that a command of the replay reads or writes.
struct BufferAccess {
    VkBuffer buffer{};
    u64 begin{};
    u64 end{};
    bool write{};
};

class BarrierTracker {
public:
    /// More ranges than this since the last barrier: the tracker stops recording and reports a
    /// conflict for everything until the next barrier (the cost of the check stays bounded).
    static constexpr size_t MAX_RANGES = 256;

    /// No command noted a buffer since the last barrier.
    [[nodiscard]] bool Empty() const noexcept {
        return ranges.empty() && !saturated;
    }

    /// True when a command with these accesses needs a barrier first: it conflicts with a range
    /// noted since the last barrier (or the tracker overflowed, or an access is malformed in the
    /// safe direction: a null buffer or an empty-by-mistake range never passes as disjoint).
    [[nodiscard]] bool Conflicts(std::span<const BufferAccess> accesses) const noexcept {
        if (saturated) {
            return !accesses.empty();
        }
        for (const BufferAccess& access : accesses) {
            if (access.buffer == VK_NULL_HANDLE || access.end < access.begin) {
                return true;
            }
            if (access.end == access.begin) {
                continue; // nothing is read or written
            }
            for (const BufferAccess& noted : ranges) {
                if (noted.buffer == access.buffer && noted.begin < access.end &&
                    access.begin < noted.end && (noted.write || access.write)) {
                    return true;
                }
            }
        }
        return false;
    }

    /// Records the accesses of a command about to be recorded. Touching or overlapping ranges of
    /// one buffer with the same direction share an entry. An access that names no buffer or ends
    /// before it begins cannot be compared: the tracker saturates (every later command conflicts
    /// until the next barrier).
    void Note(std::span<const BufferAccess> accesses) {
        for (const BufferAccess& access : accesses) {
            if (access.buffer == VK_NULL_HANDLE || access.end < access.begin) {
                saturated = true;
                continue;
            }
            if (access.end == access.begin) {
                continue;
            }
            Add(access);
        }
    }

    /// A barrier was emitted: everything before it is ordered before everything after it.
    void Clear() noexcept {
        ranges.clear();
        saturated = false;
    }

    [[nodiscard]] size_t Size() const noexcept {
        return ranges.size();
    }
    [[nodiscard]] bool Saturated() const noexcept {
        return saturated;
    }

private:
    void Add(BufferAccess access) {
        if (saturated) {
            return;
        }
        for (BufferAccess& noted : ranges) {
            if (noted.buffer == access.buffer && noted.write == access.write &&
                noted.begin <= access.end && access.begin <= noted.end) {
                noted.begin = (std::min)(noted.begin, access.begin);
                noted.end = (std::max)(noted.end, access.end);
                return;
            }
        }
        if (ranges.size() >= MAX_RANGES) {
            saturated = true;
            return;
        }
        ranges.push_back(access);
    }

    std::vector<BufferAccess> ranges;
    bool saturated{false};
};

} // namespace Vulkan::Nvn
