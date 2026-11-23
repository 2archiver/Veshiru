// SPDX-FileCopyrightText: Copyright (c) 2026 jhaizhou (d2r_gfx)
// SPDX-FileCopyrightText: Copyright (c) 2026 MacNativeGames contributors
// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>

#include "video_core/nvn/record.h"
#include "video_core/nvn/uniform_snapshot.h"

namespace VideoCommon::Nvn {
namespace {

using Range = UniformSnapshot::Range;

/// Snapshot objects are reused: their byte vectors are a few hundred KiB to a few MiB per
/// submission, and fresh pages made the copy about three times slower than memcpy
/// (MacNativeGames ucap_bytes_get). The replay thread returns them when a submission is gone.
class SnapshotRecycler {
public:
    std::shared_ptr<UniformSnapshot> Acquire() {
        std::unique_ptr<UniformSnapshot> object;
        {
            std::scoped_lock lock{mutex};
            if (!free_list.empty()) {
                object = std::move(free_list.back());
                free_list.pop_back();
            }
        }
        if (!object) {
            object = std::make_unique<UniformSnapshot>();
        }
        object->ranges.clear();
        object->bytes.clear();
        object->completed_seq = 0;
        return {object.release(), [this](UniformSnapshot* returned) { Release(returned); }};
    }

private:
    static constexpr size_t KEEP = 8;

    void Release(UniformSnapshot* returned) {
        std::unique_ptr<UniformSnapshot> object{returned};
        std::scoped_lock lock{mutex};
        if (free_list.size() < KEEP) {
            free_list.push_back(std::move(object));
        }
    }

    std::mutex mutex;
    std::vector<std::unique_ptr<UniformSnapshot>> free_list;
};

SnapshotRecycler& Recycler() {
    // Never destroyed: a submission may outlive static destruction order at exit.
    static auto* const recycler = new SnapshotRecycler;
    return *recycler;
}

/// The uniform ranges the lists bind (usnap_collect). Bindings without an address or a size read
/// zero and need no copy.
void Collect(const std::vector<std::vector<u64>>& lists, std::vector<Range>& out, u64& bindings) {
    for (const std::vector<u64>& list : lists) {
        RecordReader reader{list.data(), list.size()};
        while (const RecordHeader* header = reader.Next()) {
            if (RecordOp(header) != Op::BindUniformBuffers ||
                RecordPayloadSize(header) < sizeof(BindBuffersPayload)) {
                continue;
            }
            const auto* payload = RecordPayload<BindBuffersPayload>(header);
            const u32 available = static_cast<u32>(
                (RecordPayloadSize(header) - sizeof(BindBuffersPayload)) / sizeof(BufferRange));
            const BufferRange* ranges = TrailingArray<BufferRange>(payload);
            const u32 count = (std::min)(payload->count, available);
            bindings += count;
            for (u32 i = 0; i < count; ++i) {
                const u64 size = (std::min)(ranges[i].size, UNIFORM_SNAPSHOT_MAX_BINDING);
                if (ranges[i].address != 0 && size != 0) {
                    out.push_back({ranges[i].address, ranges[i].address + size, 0});
                }
            }
        }
    }
}

/// Sorts by start (rng_sort): LSD radix sort on the 64-bit start, 8 bits per pass, skipping the
/// bytes every key shares (the addresses of one frame differ only in their low bytes). Stable.
void SortRanges(std::vector<Range>& ranges, std::vector<Range>& tmp) {
    const size_t n = ranges.size();
    if (n <= 32) {
        std::sort(ranges.begin(), ranges.end(),
                  [](const Range& a, const Range& b) { return a.begin < b.begin; });
        return;
    }
    u64 any = 0;
    u64 all = ~u64{0};
    for (const Range& range : ranges) {
        any |= range.begin;
        all &= range.begin;
    }
    const u64 vary = any ^ all;
    tmp.resize(n);
    Range* src = ranges.data();
    Range* dst = tmp.data();
    for (u32 shift = 0; shift < 64; shift += 8) {
        if (((vary >> shift) & 0xFF) == 0) {
            continue;
        }
        std::array<size_t, 257> count{};
        for (size_t i = 0; i < n; ++i) {
            ++count[((src[i].begin >> shift) & 0xFF) + 1];
        }
        for (size_t b = 0; b < 256; ++b) {
            count[b + 1] += count[b];
        }
        for (size_t i = 0; i < n; ++i) {
            dst[count[(src[i].begin >> shift) & 0xFF]++] = src[i];
        }
        std::swap(src, dst);
    }
    if (src != ranges.data()) {
        std::memcpy(ranges.data(), src, n * sizeof(Range));
    }
}

constexpr u64 PAGE_BITS = 12;
constexpr u64 PAGE_SIZE = u64{1} << PAGE_BITS;

/// Keeps the ranges that start inside a snapshot pool, cut at the pool's end (a uniform binding
/// is 256-byte granular and may run past it). `Range::offset` holds the pool's start meanwhile.
void AssignPools(std::vector<Range>& ranges, const UniformSnapshotPools& pools) {
    size_t kept = 0;
    u64 pool_begin = 1; // the pool of the last lookup (sorted input: mostly the same pool)
    u64 pool_end = 0;
    for (const Range& range : ranges) {
        if (range.begin < pool_begin || range.begin >= pool_end) {
            auto it = pools.upper_bound(range.begin);
            if (it == pools.begin()) {
                continue;
            }
            --it;
            if (range.begin >= it->second.end) {
                continue;
            }
            pool_begin = it->first;
            pool_end = it->second.end;
        }
        ranges[kept++] = {range.begin, (std::min)(range.end, pool_end), pool_begin};
    }
    ranges.resize(kept);
}

/// Keeps the ranges whose every page the guest CPU wrote in this frame or the previous one.
/// Takes the pages written since the last capture from the page tracker (which arms them again)
/// and arms the pages seen for the first time (they qualify from their next write on). A pool the
/// tracker cannot follow keeps its ranges (nothing tells its CPU writes from its GPU writes there;
/// the GPU history still applies).
u64 KeepCpuWritten(std::vector<Range>& ranges, UniformSnapshotPools& pools,
                   UniformSnapshotCpuPages& cpu, GuestMemory& memory,
                   std::vector<std::pair<u64, u64>>& runs) {
    constexpr u64 RECENT_FRAMES = 16;
    const u64 current_present = cpu.presents + 1;
    const u64 recent = cpu.presents > RECENT_FRAMES ? cpu.presents - RECENT_FRAMES : 0;
    // One pass per run of touched pages of one pool: the tracker is asked once per run.
    size_t i = 0;
    while (i < ranges.size()) {
        const u64 pool_begin = ranges[i].offset;
        UniformSnapshotPool& pool = pools.find(pool_begin)->second;
        if (!pool.track_tried) {
            pool.track_tried = true;
            pool.track_id = memory.Track(pool_begin, pool.end - pool_begin);
            if (pool.track_id != 0) {
                pool.write_epoch = memory.WriteEpoch(pool.track_id);
                pool.epoch_clean =
                    pool.write_epoch ? pool.write_epoch->load(std::memory_order_acquire) : 0;
            }
        }
        const u64 first = ranges[i].begin >> PAGE_BITS;
        u64 last = (ranges[i].end - 1) >> PAGE_BITS;
        size_t j = i + 1;
        while (j < ranges.size() && ranges[j].offset == pool_begin &&
               (ranges[j].begin >> PAGE_BITS) <= last + 1) {
            last = (std::max)(last, (ranges[j].end - 1) >> PAGE_BITS);
            ++j;
        }
        if (pool.track_id != 0) {
            u64 arm_from = 0;
            u64 arm_count = 0;
            const auto flush_arm = [&] {
                if (arm_count != 0) {
                    memory.Arm(pool.track_id, arm_from << PAGE_BITS, arm_count << PAGE_BITS);
                    arm_count = 0;
                }
            };
            for (u64 page = first; page <= last; ++page) {
                if (cpu.written.try_emplace(page, UniformSnapshotPageState{}).second) {
                    if (arm_count == 0 || arm_from + arm_count != page) {
                        flush_arm();
                        arm_from = page;
                    }
                    ++arm_count;
                }
            }
            flush_arm();

            const u64 cur_epoch =
                pool.write_epoch ? pool.write_epoch->load(std::memory_order_acquire) : 0;
            const bool epoch_unchanged = pool.write_epoch && cur_epoch == pool.epoch_clean;
            if (!epoch_unchanged) {
                u64 sub_first = 0;
                bool in_sub = false;
                const auto flush_sub = [&](u64 sub_end_exclusive) {
                    runs.clear();
                    memory.TakeWritten(pool.track_id, sub_first << PAGE_BITS,
                                       (sub_end_exclusive - sub_first) << PAGE_BITS, runs);
                    for (const auto& [begin, end] : runs) {
                        for (u64 page = begin >> PAGE_BITS; page < (end >> PAGE_BITS); ++page) {
                            cpu.written[page].written = current_present;
                        }
                    }
                    in_sub = false;
                };
                for (u64 page = first; page <= last; ++page) {
                    UniformSnapshotPageState& state = cpu.written[page];
                    const bool stale = state.written == 0 || state.written < recent;
                    const bool needs_check =
                        stale && (!pool.write_epoch || state.checked != cur_epoch);
                    if (needs_check) {
                        state.checked = cur_epoch;
                        if (!in_sub) {
                            sub_first = page;
                            in_sub = true;
                        }
                    } else if (in_sub) {
                        flush_sub(page);
                    }
                }
                if (in_sub) {
                    flush_sub(last + 1);
                }
            }
        }
        i = j;
    }
    // Written within RECENT_FRAMES (or never yet checked cleanly for untracked pools).
    size_t kept = 0;
    u64 last_pool_begin = ~u64{0};
    bool last_tracked = false;
    for (const Range& range : ranges) {
        if (range.offset != last_pool_begin) {
            last_pool_begin = range.offset;
            last_tracked = pools.find(last_pool_begin)->second.track_id != 0;
        }
        if (last_tracked) {
            bool written = true;
            for (u64 page = range.begin >> PAGE_BITS; page <= (range.end - 1) >> PAGE_BITS;
                 ++page) {
                const auto it = cpu.written.find(page);
                if (it == cpu.written.end() || it->second.written == 0 ||
                    it->second.written < recent) {
                    written = false;
                    break;
                }
            }
            if (!written) {
                continue;
            }
        }
        ranges[kept++] = range;
    }
    const u64 dropped = ranges.size() - kept;
    ranges.resize(kept);
    return dropped;
}

/// Merges the sorted ranges in place (mtl_sub_capture_ubos / usnap_take): overlapping ranges of
/// one pool become one, and ranges at most UNIFORM_SNAPSHOT_MERGE_GAP apart merge while the
/// result stays within UNIFORM_SNAPSHOT_MERGE_LIMIT. `offset` still holds the pool's start.
void MergeRanges(std::vector<Range>& ranges) {
    size_t kept = 0;
    for (size_t i = 0; i < ranges.size(); ++i) {
        const Range range = ranges[i];
        if (kept != 0) {
            Range& last = ranges[kept - 1];
            if (last.offset == range.offset &&
                range.begin <= last.end + UNIFORM_SNAPSHOT_MERGE_GAP) {
                if (range.end <= last.end) {
                    continue;
                }
                if (range.begin < last.end ||
                    range.end - last.begin <= UNIFORM_SNAPSHOT_MERGE_LIMIT) {
                    last.end = range.end;
                    continue;
                }
            }
        }
        ranges[kept++] = range;
    }
    ranges.resize(kept);
}

} // namespace

// ---- UniformSnapshotShared ----------------------------------------------------------------

void UniformSnapshotShared::Reset(UniformSnapshotMode mode_) {
    std::scoped_lock lock{mutex};
    written.clear();
    mode.store(mode_, std::memory_order_relaxed);
}

void UniformSnapshotShared::NoteGpuWrite(u64 begin, u64 end) {
    if (end <= begin) {
        return;
    }
    begin &= ~(PAGE_SIZE - 1);
    end = (end + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    std::scoped_lock lock{mutex};
    auto it = written.upper_bound(begin);
    if (it != written.begin()) {
        const auto previous = std::prev(it);
        if (previous->second >= begin) {
            if (previous->second >= end) {
                return; // already there
            }
            begin = previous->first;
            it = written.erase(previous);
        }
    }
    while (it != written.end() && it->first <= end) {
        end = (std::max)(end, it->second);
        it = written.erase(it);
    }
    written.emplace(begin, end);
}

void UniformSnapshotShared::Forget(u64 begin, u64 end) {
    std::scoped_lock lock{mutex};
    std::vector<std::pair<u64, u64>> keep;
    auto it = written.upper_bound(begin);
    if (it != written.begin()) {
        --it;
    }
    while (it != written.end() && it->first < end) {
        if (it->second <= begin) {
            ++it;
            continue;
        }
        if (it->first < begin) {
            keep.emplace_back(it->first, begin);
        }
        if (it->second > end) {
            keep.emplace_back(end, it->second);
        }
        it = written.erase(it);
    }
    written.insert(keep.begin(), keep.end());
}

u64 UniformSnapshotShared::ExcludeGpuWritten(std::vector<UniformSnapshot::Range>& ranges) const {
    std::scoped_lock lock{mutex};
    if (written.empty()) {
        return 0;
    }
    size_t kept = 0;
    for (const Range& range : ranges) {
        const u64 begin = range.begin & ~(PAGE_SIZE - 1);
        const u64 end = (range.end + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
        const auto it = written.upper_bound(begin);
        bool hit = it != written.end() && it->first < end;
        if (!hit && it != written.begin()) {
            hit = std::prev(it)->second > begin;
        }
        if (!hit) {
            ranges[kept++] = range;
        }
    }
    const u64 dropped = ranges.size() - kept;
    ranges.resize(kept);
    return dropped;
}

size_t UniformSnapshotShared::Runs() const {
    std::scoped_lock lock{mutex};
    return written.size();
}

UniformSnapshotShared& SharedUniformSnapshot() {
    // Never destroyed: the replay thread may still note writes during static destruction.
    static auto* const shared = new UniformSnapshotShared;
    return *shared;
}

// ---- Capture ---------------------------------------------------------------------------------

std::shared_ptr<UniformSnapshot> CaptureUniformSnapshot(
    const std::vector<std::vector<u64>>& lists, UniformSnapshotPools& pools,
    UniformSnapshotCpuPages& cpu, const UniformSnapshotShared& shared, GuestMemory& memory,
    u64 completed_seq, UniformSnapshotScratch& scratch, UniformSnapshotTotals& totals) {
    std::vector<Range>& ranges = scratch.ranges;
    ranges.clear();
    Collect(lists, ranges, totals.bindings);
    if (ranges.empty() || pools.empty()) {
        return nullptr;
    }
    SortRanges(ranges, scratch.sort_tmp);
    AssignPools(ranges, pools);
    totals.not_cpu += KeepCpuWritten(ranges, pools, cpu, memory, scratch.runs);
    totals.gpu_written += shared.ExcludeGpuWritten(ranges);
    MergeRanges(ranges);
    if (ranges.empty()) {
        return nullptr;
    }
    std::shared_ptr<UniformSnapshot> snapshot = Recycler().Acquire();
    snapshot->completed_seq = completed_seq;
    u64 total = 0;
    for (const Range& range : ranges) {
        total += range.end - range.begin;
    }
    snapshot->bytes.resize(total);
    snapshot->ranges.reserve(ranges.size());
    u64 offset = 0;
    for (const Range& range : ranges) {
        const u64 size = range.end - range.begin;
        // A range the guest cannot read (unmapped) is left to the replay's live read.
        if (memory.Read(range.begin, snapshot->bytes.data() + offset, size)) {
            snapshot->ranges.push_back({range.begin, range.end, offset});
            offset += size;
        }
    }
    snapshot->bytes.resize(offset);
    if (snapshot->ranges.empty()) {
        return nullptr;
    }
    totals.ranges += snapshot->ranges.size();
    totals.bytes += offset;
    return snapshot;
}

} // namespace VideoCommon::Nvn
