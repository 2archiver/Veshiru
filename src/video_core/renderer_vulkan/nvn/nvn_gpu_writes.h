// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (memulator)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <vector>

#include "common/common_types.h"

namespace Vulkan::Nvn {

/// Which submission last wrote which guest byte ranges, until that submission completes. The
/// replay thread notes a write when it records the command that makes it (the write-back of a
/// mirror entry, a shadow copy, a copy into a pool, a mirror pool's readback); a guest thread asks
/// for the newest submission that has not completed yet and writes a range (EDEN_NX2_TRY_
/// INVALIDATE_WRITEBACK: nvnMemoryPoolInvalidateMappedRange waits for it, so the CPU reads what
/// the GPU wrote). Addresses are guest virtual addresses, so one log serves every pool. Thread safe.
class GpuWriteLog {
public:
    /// A log that kept this many entries first drops the completed ones, then merges the entries
    /// of one submission into their hull (only ever a longer wait, never a missed one).
    static constexpr size_t PRUNE_AT = 64;

    /// Submission `seq` writes [begin, end). `completed_seq` is the newest completed submission
    /// (entries up to it are not needed any more).
    void Note(u64 begin, u64 end, u64 seq, u64 completed_seq) {
        if (end <= begin) {
            return;
        }
        std::scoped_lock lock{mutex};
        if (!entries.empty()) {
            Entry& last = entries.back();
            if (last.seq == seq && begin <= last.end && last.begin <= end) {
                last.begin = (std::min)(last.begin, begin);
                last.end = (std::max)(last.end, end);
                return;
            }
        }
        entries.push_back({begin, end, seq});
        if (entries.size() >= PRUNE_AT) {
            Prune(completed_seq);
        }
    }

    /// The newest submission <= `max_seq` and > `completed_seq` that writes a byte of
    /// [begin, end); 0 when there is none.
    [[nodiscard]] u64 Latest(u64 begin, u64 end, u64 max_seq, u64 completed_seq) const {
        if (end <= begin) {
            return 0;
        }
        u64 result = 0;
        std::scoped_lock lock{mutex};
        for (const Entry& entry : entries) {
            if (entry.seq > completed_seq && entry.seq <= max_seq && entry.begin < end &&
                begin < entry.end) {
                result = (std::max)(result, entry.seq);
            }
        }
        return result;
    }

    [[nodiscard]] size_t Size() const {
        std::scoped_lock lock{mutex};
        return entries.size();
    }

private:
    struct Entry {
        u64 begin;
        u64 end;
        u64 seq;
    };

    void Prune(u64 completed_seq) {
        std::erase_if(entries, [&](const Entry& entry) { return entry.seq <= completed_seq; });
        if (entries.size() < PRUNE_AT) {
            return;
        }
        // Still too many unfinished writes: one hull per submission.
        std::vector<Entry> merged;
        for (const Entry& entry : entries) {
            const auto it = std::ranges::find_if(
                merged, [&](const Entry& other) { return other.seq == entry.seq; });
            if (it == merged.end()) {
                merged.push_back(entry);
            } else {
                it->begin = (std::min)(it->begin, entry.begin);
                it->end = (std::max)(it->end, entry.end);
            }
        }
        entries = std::move(merged);
    }

    mutable std::mutex mutex;
    std::vector<Entry> entries;
};

} // namespace Vulkan::Nvn
