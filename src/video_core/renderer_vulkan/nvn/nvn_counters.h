// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// GPU-side NVN counters, reports and render conditions. Behaviour follows nxnative's
// `nvn_mtl.m` (conditional rendering, occlusion queries) and the NVN counter data layout
// `NVNcounterData { u64 counter; u64 timestamp; }` (16 bytes):
//
//  - ReportCounter(TIMESTAMP / TIMESTAMP_TOP) writes counter = 0 and the time in the timestamp
//    field: EDEN_NX2_NVN_TIMESTAMPS=real (default) the GPU time of the report mapped onto the
//    guest clock, `legacy` the guest GPU clock as the GPU emulation path writes it; see
//    video_core/nvn/timestamp_mode.h.
//  - ReportCounter(SAMPLES_PASSED) writes the samples passed since the queue's last
//    ResetCounter(SAMPLES_PASSED) (occlusion queries, accumulated on the GPU), plus a timestamp.
//    With in-pass queries (InPassQueries) the queries live inside the render passes and the
//    reports of a pass are written when it ends (CounterBatch, ResolveBatch).
//  - Any other counter type is reported as 0 (counted and logged as unsupported).
//  - SetRenderEnableConditional compares the 64-bit counter fields of the two reports at `address`
//    and `address + 16`.
//
// Counter and predicate data live in one device-local scratch buffer, so everything a condition
// depends on is ordered on the GPU (a report written by an earlier submission is visible to a
// later condition without waiting for the CPU to see the completion).

#pragma once

#include <algorithm>
#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/nvn/timestamp_mode.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCommon::Nvn {
class GuestMemory;
}

namespace Vulkan::Nvn {

/// NVNcounterType values the replay gives a meaning to.
namespace CounterType {
constexpr u32 Timestamp = 0x0;
constexpr u32 SamplesPassed = 0x1;
constexpr u32 TimestampTop = 0x10;
} // namespace CounterType

/// NVNconditionalRenderMode.
namespace ConditionMode {
constexpr u32 RenderIfEqual = 0;
constexpr u32 RenderIfNotEqual = 1;
} // namespace ConditionMode

/// One query slot (an index into CounterFlight::blocks and a slot of that block).
struct QueryRef {
    u32 block{};
    u32 slot{};
};

/// SAMPLES_PASSED events recorded while a render pass was open (in-pass queries): the queries
/// that ended, the resets and the reports, in guest order. ResolveBatch (outside the pass) turns
/// them into the report values: each report is the queue's accumulator plus the queries before
/// it since the last reset.
struct CounterBatch {
    /// Events one resolve handles. A full batch ends the open pass (ReserveCounterEvents): OoT's
    /// main view reports and resets around ~700 draws per pass, which 256 events split into ~8
    /// passes. ResolveBatch walks them in dispatches of RESOLVE_CHUNK_EVENTS.
    static constexpr u32 MAX_EVENTS = 4096;
    enum class Event : u8 { Query = 0, Reset = 1, Report = 2 };
    struct Report {
        VkBuffer buffer{};
        u64 offset{};          ///< byte offset in `buffer` (= offset in the pool)
        const void* pool{};    ///< identity of the pool the report lands in (hazard checks)
    };
    std::vector<Event> events;
    std::vector<QueryRef> queries;
    std::vector<Report> reports;

    [[nodiscard]] bool Empty() const noexcept {
        return events.empty();
    }
    [[nodiscard]] bool HasReports() const noexcept {
        return !reports.empty();
    }
    /// Room for `count` more events.
    [[nodiscard]] bool Room(u32 count) const noexcept {
        return events.size() + count <= MAX_EVENTS;
    }
    void AddQuery(QueryRef query) {
        events.push_back(Event::Query);
        queries.push_back(query);
    }
    void AddReset() {
        // A reset right after a reset changes nothing (OoT issues thousands in a row).
        if (events.empty() || events.back() != Event::Reset) {
            events.push_back(Event::Reset);
        }
    }
    void AddReport(VkBuffer buffer, u64 offset, const void* pool) {
        events.push_back(Event::Report);
        reports.push_back({buffer, offset, pool});
        reports_begin = (std::min)(reports_begin, offset);
        reports_end = (std::max)(reports_end, offset + sizeof(u64));
    }
    /// A report still to be written overlaps [offset, offset + size) of `pool`.
    [[nodiscard]] bool Writes(const void* pool, u64 offset, u64 size) const noexcept {
        // Every draw of a pass with pending reports asks for each range it uses: most ranges are
        // nowhere near the reports (in any pool).
        if (offset >= reports_end || offset + size <= reports_begin) {
            return false;
        }
        for (const Report& report : reports) {
            if (report.pool == pool && report.offset < offset + size &&
                offset < report.offset + sizeof(u64)) {
                return true;
            }
        }
        return false;
    }
    void Clear() noexcept {
        events.clear();
        queries.clear();
        reports.clear();
        reports_begin = ~u64{0};
        reports_end = 0;
    }

private:
    /// Byte range of the pool offsets of `reports` (any pool).
    u64 reports_begin{~u64{0}};
    u64 reports_end{0};
};

/// Byte ring the resolve areas of CounterEngine::ResolveBatch come from. Space is handed out in
/// order and reused oldest first; an allocation that reaches an area another submission took
/// first waits for that submission (`wait(seq, begin)`; it may give up, the area is reused then).
/// Areas of the allocating submission are reused without a wait: its own commands are ordered.
class AreaRing {
public:
    explicit AreaRing(u32 size_) : size{size_} {}

    /// `bytes` (at least 16, rounded up to 16; at most the ring size); returns its ring offset.
    template <typename Wait>
    [[nodiscard]] u32 Allocate(u64 seq, u32 bytes, Wait&& wait) {
        bytes = (std::max)((bytes + 15u) & ~15u, 16u);
        if (head + bytes > size) {
            // The tail that does not fit is skipped: the areas in it are reused now as well.
            Reclaim(head, size, seq, wait);
            head = 0;
        }
        const u32 begin = head;
        Reclaim(begin, begin + bytes, seq, wait);
        head = begin + bytes;
        uses.push_back({begin, head, seq});
        return begin;
    }
    /// Areas still remembered (tests).
    [[nodiscard]] size_t Live() const noexcept {
        return uses.size();
    }

private:
    struct Use {
        u32 begin;
        u32 end;
        u64 seq;
    };
    template <typename Wait>
    void Reclaim(u32 begin, u32 end, u64 seq, Wait& wait) {
        // In ring order from the oldest area, the ones [begin, end) reaches are at the front.
        while (!uses.empty() && uses.front().begin < end && begin < uses.front().end) {
            const Use old = uses.front();
            uses.pop_front();
            if (old.seq != seq) {
                wait(old.seq, old.begin);
            }
        }
    }

    u32 size;
    u32 head{0};
    std::deque<Use> uses;
};

/// Query pools one submission draws its slots from; recycled when the submission completes.
struct QueryBlock {
    vk::QueryPool occlusion;
    vk::QueryPool timestamp;
    u32 occlusion_used{};
    u32 timestamp_used{};
};

/// A timestamp report the completion thread fills in: the GPU wrote the counter field, the time
/// is converted and stored in the timestamp field (+8) once the submission completed.
struct TimestampReport {
    QueryRef query;
    u64 address{};        ///< Guest address of the report (for GuestMemory::Write).
    u8* host{};           ///< Host pointer of the report when the pool is imported, else null.
    std::shared_ptr<void> owner; ///< Keeps `host` valid.
};

/// What one submission leaves for the completion thread.
struct CounterFlight {
    std::vector<QueryBlock> blocks;
    std::vector<TimestampReport> timestamps;
    /// EDEN_NX2_NVN_TSTRACE only: SAMPLES_PASSED reports whose value the completion thread logs.
    std::vector<TimestampReport> traced_samples;
};

/// Monotonic counters (read by DescribeState / the stats rows).
struct CounterEngineStats {
    std::atomic<u64> conditions_gpu{};        ///< Predicates evaluated on the GPU (batched ones too).
    /// SetRenderEnableConditional served by a predicate computed ahead (RecordConditions batch).
    std::atomic<u64> conditions_prefetched{};
    std::atomic<u64> conditions_cpu{};        ///< Conditions evaluated on the CPU (fallback).
    std::atomic<u64> conditions_unresolved{}; ///< Report addresses outside every pool: drew anyway.
    std::atomic<u64> conditions_false{};      ///< CPU-evaluated conditions that disabled rendering.
    std::atomic<u64> ops_gated{};             ///< Draws/clears wrapped in conditional rendering.
    std::atomic<u64> ops_skipped{};           ///< Draws/clears dropped by SetRenderEnable(false).
    std::atomic<u64> reports_timestamp{};
    std::atomic<u64> reports_samples{};
    std::atomic<u64> reports_value{};
    std::atomic<u64> reports_samples_uncounted{}; ///< SAMPLES_PASSED reports after draws counted by nobody.
    std::atomic<u64> reports_unsupported_type{}; ///< Reports of counter types written as 0.
    std::atomic<u64> resets{};
    /// Fold dispatches: one per report with whole-pass queries, one per resolved CounterBatch with
    /// in-pass queries.
    std::atomic<u64> occlusion_folds{};
};

class CounterEngine {
public:
    static constexpr u32 MAX_QUEUES = 64;
    static constexpr u32 ACCUMULATOR_STRIDE = 16;
    static constexpr u32 SLOT_SIZE = 64;
    static constexpr u32 SLOT_COUNT = 2048;
    static constexpr u32 BLOCK_QUERIES = 128;
    /// Bytes of a report pair the comparison reads (counter at +0 and counter at +16).
    static constexpr u32 REPORT_PAIR_SPAN = 24;
    /// Bytes of the ring ResolveBatch takes its areas from (query results, then report values; 8
    /// bytes each), in the order they are handed out (AreaRing).
    static constexpr u32 RESOLVE_RING_SIZE = 2u << 20;
    /// Events one resolve dispatch walks (2 bits each in the push constants); a larger batch takes
    /// several dispatches, each continuing from the accumulator the previous one stored.
    static constexpr u32 RESOLVE_CHUNK_EVENTS = 256;

    /// `immediate_submit` runs one-time commands and waits (zeroes the scratch buffer);
    /// `wait_completed` blocks until the replay's completed sequence reaches the argument (true) or
    /// gives up (false: the scratch slot is then reused, see AllocateSlot).
    /// `allow_in_pass_queries`: occlusion queries inside render passes (InPassQueries).
    CounterEngine(const Device& device, MemoryAllocator& allocator,
                  std::function<void(const std::function<void(vk::CommandBuffer)>&)> immediate_submit,
                  std::function<bool(u64)> wait_completed, bool allow_conditional_rendering,
                  bool allow_in_pass_queries);
    ~CounterEngine();

    /// VK_EXT_conditional_rendering is in use (else conditions are evaluated on the CPU).
    [[nodiscard]] bool GpuConditions() const noexcept {
        return gpu_conditions;
    }
    /// Query slots are reset on the host when a block is taken, so a timestamp can be written
    /// inside a render pass (a command-buffer reset must be outside one).
    [[nodiscard]] bool TimestampInPass() const noexcept {
        return host_reset;
    }
    /// SAMPLES_PASSED queries begin and end inside the render passes (one per counted stretch of a
    /// pass) and ReportCounter / ResetCounter keep the pass open; the reports of a pass are
    /// written by ResolveBatch once it ended. Needs the host query reset (a query begun inside a
    /// pass must have been reset outside one). Otherwise a query spans whole passes and every
    /// report ends the pass (FoldOcclusion).
    [[nodiscard]] bool InPassQueries() const noexcept {
        return in_pass_queries;
    }
    [[nodiscard]] VkBuffer Scratch() const noexcept {
        return *scratch;
    }
    [[nodiscard]] CounterEngineStats& Stats() noexcept {
        return stats;
    }

    /// A free 64-byte slot of the scratch ring (blocks until the GPU finished any earlier use).
    /// Returns its byte offset in Scratch().
    [[nodiscard]] u32 AllocateSlot(u64 seq);
    /// `count` (<= MAX_CONDITION_BATCH) consecutive slots, as AllocateSlot; the offset of the
    /// first, the others follow at SLOT_SIZE strides. The batch takes the last `count` positions
    /// of SlotsAllocated().
    [[nodiscard]] u32 AllocateSlots(u64 seq, u32 count);
    /// Slots handed out so far (monotonic, ring wraps included). The slot handed out at position
    /// `p` keeps what was written to it while SlotsAllocated() <= p + SLOT_COUNT: the ring has not
    /// come back to it (same submission or not).
    [[nodiscard]] u64 SlotsAllocated() const noexcept {
        return next_slot;
    }

    // ---- render conditions --------------------------------------------------------------------

    /// The report pair one SetRenderEnableConditional compares.
    struct ConditionInput {
        VkBuffer a_buffer;
        u64 a_offset;
        VkBuffer b_buffer;
        u64 b_offset;
        u32 mode; ///< ConditionMode
    };
    /// Predicates one RecordConditions call computes (one bit each in the push constants).
    static constexpr u32 MAX_CONDITION_BATCH = 256;

    /// Copies the counters at (a_buffer, a_offset) and (b_buffer, b_offset) next to each other,
    /// compares them (`mode` is a ConditionMode) and leaves the 32-bit predicate (1 = render) at
    /// `slot_offset + 32` of Scratch(), readable by conditional rendering. Both buffers must be
    /// transfer sources. Recorded outside a render pass.
    void RecordCondition(vk::CommandBuffer cmd, VkBuffer a_buffer, u64 a_offset, VkBuffer b_buffer,
                         u64 b_offset, u32 mode, u32 slot_offset);
    /// RecordCondition for up to MAX_CONDITION_BATCH pairs at once (one barrier set, one
    /// dispatch): the predicate of `conditions[i]` goes to the slot at `first_slot + i *
    /// SLOT_SIZE` (AllocateSlots).
    void RecordConditions(vk::CommandBuffer cmd, std::span<const ConditionInput> conditions,
                          u32 first_slot);

    /// Starts conditional rendering on the predicate of a slot RecordCondition filled in.
    void BeginConditional(vk::CommandBuffer cmd, u32 slot_offset) const;
    void EndConditional(vk::CommandBuffer cmd) const;

    // ---- counters -----------------------------------------------------------------------------

    /// Begins an occlusion query. Outside a render pass it may span whole passes; inside one
    /// (InPassQueries only) it must end before the pass does.
    [[nodiscard]] QueryRef BeginOcclusion(vk::CommandBuffer cmd, CounterFlight& flight);
    /// Ends `query` and discards its samples (ResetCounter).
    void EndOcclusion(vk::CommandBuffer cmd, CounterFlight& flight, QueryRef query);
    /// Ends `query` and adds its samples to the queue's accumulator.
    void FoldOcclusion(vk::CommandBuffer cmd, CounterFlight& flight, QueryRef query, u32 queue,
                       u64 seq);
    void ResetAccumulator(vk::CommandBuffer cmd, u32 queue);
    /// Copies the queue's accumulator (8 bytes) to `dst` + `dst_offset`.
    void CopyAccumulator(vk::CommandBuffer cmd, u32 queue, VkBuffer dst, u64 dst_offset);
    /// Writes the reports of `batch` (outside a render pass) and empties it: the queries' results
    /// are copied next to each other, one dispatch walks the events from the queue's accumulator
    /// (query: add, reset: zero, report: store), stores the accumulator back, and the report
    /// values are copied to their buffers. Every query of the batch has ended.
    void ResolveBatch(vk::CommandBuffer cmd, CounterFlight& flight, CounterBatch& batch, u32 queue,
                      u64 seq);

    /// Writes a GPU timestamp and registers `report` (its query field is filled in) for the
    /// completion thread.
    void WriteTimestamp(vk::CommandBuffer cmd, CounterFlight& flight, bool top_of_pipe,
                        TimestampReport report);

    /// Completion thread: stores the converted time of every timestamp report and recycles the
    /// query blocks.
    void Complete(CounterFlight& flight, VideoCommon::Nvn::GuestMemory* memory);

    /// Converts GPU ticks to nanoseconds on the guest clock (completion thread). Exposed for
    /// tests.
    [[nodiscard]] u64 TicksToNanoseconds(u64 ticks);

private:
    void Sync(vk::CommandBuffer cmd) const;
    void Dispatch(vk::CommandBuffer cmd, u32 base_word, u32 mode, u32 accumulator_word) const;
    /// `bytes` of the resolve ring (blocks until the GPU finished any earlier use of them); byte
    /// offset in Scratch().
    [[nodiscard]] u32 AllocateArea(u64 seq, u32 bytes);
    QueryBlock& BlockWithRoom(CounterFlight& flight, bool occlusion, u32& index);
    QueryBlock NewBlock();
    void ResetBlock(QueryBlock& block);

    const Device& device;
    const vk::DeviceDispatch& dld;
    std::function<bool(u64)> wait_completed;
    bool gpu_conditions{};
    bool precise_occlusion{};
    bool host_reset{};
    bool in_pass_queries{};
    double timestamp_period{1.0};
    u64 timestamp_mask{~u64{0}};

    vk::Buffer scratch;
    vk::DescriptorPool descriptor_pool;
    vk::DescriptorSetLayout set_layout;
    vk::DescriptorSets descriptor_sets;
    vk::PipelineLayout pipeline_layout;
    vk::ShaderModule module;
    vk::Pipeline pipeline;

    // Replay thread.
    u64 next_slot{0};
    std::vector<u64> slot_seq;
    /// RecordConditions: the counter copies of a batch, grouped by source buffer (reused).
    std::vector<std::pair<VkBuffer, VkBufferCopy>> condition_copies;
    std::vector<VkBufferCopy> condition_regions;
    /// The resolve areas (replay thread).
    AreaRing area_ring{RESOLVE_RING_SIZE};
    /// ResolveBatch: the report value copies of one destination buffer and their destination
    /// offsets (reused).
    std::vector<VkBufferCopy> report_regions;
    std::unordered_set<u64> region_offsets;

    // Shared with the completion thread.
    std::mutex block_mutex;
    std::vector<QueryBlock> free_blocks;

    // Completion thread.
    /// GPU clock -> guest clock (see GpuClockMap: a sliding minimum, so it follows clock drift).
    VideoCommon::Nvn::GpuClockMap clock_map{VideoCommon::Nvn::GpuClockMap::PolicyFromEnvironment()};
    u64 last_ticks{};
    u64 tick_epoch{};

    CounterEngineStats stats;
};

} // namespace Vulkan::Nvn
