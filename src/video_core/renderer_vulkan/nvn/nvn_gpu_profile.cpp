// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <utility>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/renderer_vulkan/nvn/nvn_costsplit.h"
#include "video_core/renderer_vulkan/nvn/nvn_debug.h"
#include "video_core/renderer_vulkan/nvn/nvn_gpu_profile.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan::Nvn::GpuProf {
namespace {

constexpr auto REPORT_INTERVAL = std::chrono::seconds(10);
constexpr size_t TOP_TAGS = 60;
constexpr size_t TOP_PROGRAMS = 40;

constexpr size_t KIND_COUNT = static_cast<size_t>(Kind::Count);

/// The formats DK renders to and copies; everything else prints as `f<number>`.
std::string FormatName(s32 format) {
    switch (format) {
    case 0:
        return "-";
    case VK_FORMAT_R8_UNORM:
        return "R8";
    case VK_FORMAT_R8G8_UNORM:
        return "RG8";
    case VK_FORMAT_R8G8B8A8_UNORM:
        return "RGBA8";
    case VK_FORMAT_R8G8B8A8_SRGB:
        return "RGBA8s";
    case VK_FORMAT_B8G8R8A8_UNORM:
        return "BGRA8";
    case VK_FORMAT_B8G8R8A8_SRGB:
        return "BGRA8s";
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
        return "ABGR8";
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
        return "ABGR8s";
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        return "A2B10G10R10";
    case VK_FORMAT_R16_SFLOAT:
        return "R16F";
    case VK_FORMAT_R16G16_SFLOAT:
        return "RG16F";
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return "RGBA16F";
    case VK_FORMAT_R16G16B16A16_UNORM:
        return "RGBA16";
    case VK_FORMAT_R32_SFLOAT:
        return "R32F";
    case VK_FORMAT_R32G32_SFLOAT:
        return "RG32F";
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return "RGBA32F";
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        return "B10G11R11";
    case VK_FORMAT_D16_UNORM:
        return "D16";
    case VK_FORMAT_D32_SFLOAT:
        return "D32F";
    case VK_FORMAT_D24_UNORM_S8_UINT:
        return "D24S8";
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return "D32S8";
    default:
        return fmt::format("f{}", format);
    }
}

std::string DescribeTag(const Tag& tag) {
    std::string text = fmt::format("{:<8}", KindName(tag.kind));
    switch (tag.kind) {
    case Kind::Pass:
    case Kind::Draw:
        text += fmt::format(" vs={:016x} fs={:016x}", tag.hash_a, tag.hash_b);
        if (tag.kind == Kind::Pass) {
            text += fmt::format(" draws={}", tag.count);
        }
        text += fmt::format(" rt={} {}x{} c{}{} l{} s{}", FormatName(tag.format), tag.width,
                            tag.height, tag.shape & 0xF, (tag.shape >> 4) & 1 ? "+z" : "",
                            (tag.shape >> 8) & 0xFFFF, 1u << (tag.shape >> 24));
        break;
    case Kind::Dispatch:
        text += fmt::format(" cs={:016x} groups={}x{}x{}", tag.hash_a, tag.width, tag.height,
                            tag.shape);
        break;
    case Kind::Segment:
        break;
    case Kind::Op:
        text += fmt::format(" op={}", tag.hash_a);
        break;
    default: {
        const char* sub = tag.hash_a == SUB_FEEDBACK  ? " feedback"
                          : tag.hash_a == SUB_INLINE  ? " inline-update"
                          : tag.hash_a == SUB_MIRROR  ? " mirror-upload"
                          : tag.hash_a == SUB_STORAGE ? " image-upload"
                          : tag.hash_a == SUB_SHADOW  ? " storage-shadow"
                          : tag.hash_a == SUB_VERTEX  ? " vertex-stage"
                                                      : "";
        text += fmt::format("{} fmt={} {}x{} n={}", sub, FormatName(tag.format), tag.width,
                            tag.height, tag.count);
        break;
    }
    }
    text += tag.zero_size ? " size0=1" : " size0=0";
    return text;
}

/// Intervals that contain the measured kinds (a command buffer, a pass's draws, a record of mode
/// `ops`), so they are not summed with them.
bool Overlaps(Kind kind) {
    return kind == Kind::Segment || kind == Kind::Draw || kind == Kind::Op;
}

} // namespace

const char* KindName(Kind kind) noexcept {
    switch (kind) {
    case Kind::Segment:
        return "segment";
    case Kind::Pass:
        return "pass";
    case Kind::Draw:
        return "draw";
    case Kind::Dispatch:
        return "dispatch";
    case Kind::Copy:
        return "copy";
    case Kind::Clear:
        return "clear";
    case Kind::Present:
        return "present";
    case Kind::Upload:
        return "upload";
    case Kind::Op:
        return "op";
    default:
        return "?";
    }
}

u32 ParseMode(const char* text) noexcept {
    if (text == nullptr || *text == '\0') {
        return 0;
    }
    if (std::strcmp(text, "frame") == 0) {
        return MODE_FRAME;
    }
    if (std::strcmp(text, "ops") == 0) {
        return MODE_OPS;
    }
    return static_cast<u32>(std::strtoul(text, nullptr, 0));
}

u32 PackShape(u32 colors, bool depth, u32 layers, u32 samples) noexcept {
    u32 log2_samples = 0;
    while ((1u << (log2_samples + 1)) <= samples && log2_samples < 6) {
        ++log2_samples;
    }
    return (colors & 0xF) | (depth ? 0x10u : 0u) | ((layers & 0xFFFF) << 8) | (log2_samples << 24);
}

// ---------------------------------------------------------------------------------------------
// Flight / Scope (replay thread)
// ---------------------------------------------------------------------------------------------

Flight::Flight(Profiler& owner_, vk::QueryPool pool_, bool present) noexcept
    : owner{owner_}, pool{std::move(pool_)}, is_present{present}, per_draw{owner_.PerDraw()},
      frame_only{owner_.FrameOnly()}, per_op{owner_.PerOp()} {}

Flight::~Flight() {
    if (pool) {
        // The flight never completed (a failed submission): the GPU may still own the queries, so
        // the pool is parked until the profiler dies instead of being destroyed or reused.
        owner.Retire(std::move(pool));
    }
}

u32 Flight::Stamp(vk::CommandBuffer cmd, bool inside_render_pass) noexcept {
    if (frame_only) {
        return NONE;
    }
    return StampSegment(cmd, inside_render_pass);
}

u32 Flight::StampSegment(vk::CommandBuffer cmd, bool inside_render_pass) noexcept {
    if (used >= Capacity(per_op)) {
        ++dropped;
        return NONE;
    }
    const u32 index = used++;
    if (!inside_render_pass) {
        // A BOTTOM_OF_PIPE timestamp alone does not wait for compute shaders on NVIDIA (RTX 5090,
        // driver 621.01: 8 dependent dispatches of 8.1 ms stamped 0.99 ms; a 1-group 1 ms
        // dispatch stamped 1 us; sync2 ALL_COMMANDS and an execution-only barrier behave the same).
        // The work lands on whatever really waits next (the following barrier, op or command
        // buffer end). A memory barrier makes the timestamp execute after the earlier work and its
        // writes completed: the interval is then the work's time. Only recorded when the
        // profiler is on; the barrier serializes what it separates (documented in DESIGN.md).
        constexpr VkMemoryBarrier DRAIN{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            0, DRAIN);
    }
    cmd.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *pool, index);
    return index;
}

void Flight::Add(u32 begin, u32 end, const Tag& tag, u64 units, u32 split) {
    if (begin == NONE || end == NONE) {
        return;
    }
    intervals.push_back({begin, end, units, tag, split});
    if (tag.kind == Kind::Segment) {
        ++segments_added;
    }
}

void Flight::NoteSubmitted(u64 cpu_us, u32 queue) {
    while (segment_submits.size() < segments_added) {
        segment_submits.push_back({cpu_us, queue});
    }
}

void Flight::MarkOp(vk::CommandBuffer cmd, u16 op, bool inside_render_pass) {
    if (!per_op) {
        return;
    }
    const u32 now = Stamp(cmd, inside_render_pass);
    Add(op_begin, now, Tag{.kind = Kind::Op, .hash_a = op_code});
    op_begin = now;
    op_code = op;
}

void Flight::EndOps(vk::CommandBuffer cmd) {
    if (!per_op || op_begin == NONE) {
        return;
    }
    Add(op_begin, Stamp(cmd), Tag{.kind = Kind::Op, .hash_a = op_code});
    op_begin = NONE;
}

void Scope::Begin(vk::CommandBuffer cmd, const Tag& tag, u64 units, u32 split) noexcept {
    outermost_ = flight_->depth++ == 0;
    if (!outermost_) {
        return;
    }
    cmd_ = cmd;
    tag_ = tag;
    units_ = units;
    split_ = split;
    begin_ = flight_->Stamp(cmd);
}

void Scope::End() noexcept {
    --flight_->depth;
    if (!outermost_ || begin_ == NONE) {
        return;
    }
    const u32 end = flight_->Stamp(cmd_);
    try {
        flight_->Add(begin_, end, tag_, units_, split_);
    } catch (...) {
    }
}

// ---------------------------------------------------------------------------------------------
// Profiler
// ---------------------------------------------------------------------------------------------

std::unique_ptr<Profiler> Profiler::Create(const Device& device, u32 mode, bool quiet,
                                           CostSplit::Recorder* costsplit) {
    if (mode == 0) {
        return nullptr;
    }
    if (!device.IsHostQueryResetSupported()) {
        LOG_ERROR(Render_Vulkan, "[nvn] gpuprof: hostQueryReset is not supported; profiler off");
        return nullptr;
    }
    const VkPhysicalDeviceProperties properties = device.GetPhysical().GetProperties();
    if (properties.limits.timestampPeriod <= 0.0f) {
        LOG_ERROR(Render_Vulkan, "[nvn] gpuprof: the device has no timestamps; profiler off");
        return nullptr;
    }
    u32 valid_bits = 64;
    const auto families = device.GetPhysical().GetQueueFamilyProperties();
    if (device.GetGraphicsFamily() < families.size()) {
        valid_bits = families[device.GetGraphicsFamily()].timestampValidBits;
    }
    if (valid_bits == 0) {
        LOG_ERROR(Render_Vulkan, "[nvn] gpuprof: the graphics queue has no timestamp bits; "
                                 "profiler off");
        return nullptr;
    }
    const u64 mask = valid_bits >= 64 ? ~u64{0} : (u64{1} << valid_bits) - 1;
    LOG_INFO(Render_Vulkan,
             "[nvn] gpuprof: on (mode {}{}), timestamp period {} ns, {} valid bits. {}",
             mode == MODE_FRAME ? "frame" : std::to_string(mode),
             mode == 2 ? ", per-draw intervals" : "", properties.limits.timestampPeriod, valid_bits,
             mode == MODE_FRAME
                 ? "Frame-only: one timestamp pair per command buffer, nothing inside it (the "
                   "passes keep their overlap); read gpu_busy / gpu_span per frame"
                 : "Timestamps drain the pipeline at every interval boundary: read relative "
                   "costs, not absolute FPS");
    return std::make_unique<Profiler>(device, mode, properties.limits.timestampPeriod, mask, quiet,
                                      costsplit);
}

Profiler::Profiler(const Device& device_, u32 mode_, double period_ns_, u64 mask_, bool quiet_,
                   CostSplit::Recorder* costsplit_)
    : device{device_}, mode{mode_}, period_ns{period_ns_}, mask{mask_}, quiet{quiet_},
      costsplit{costsplit_}, run_start{std::chrono::steady_clock::now()}, window_start{run_start} {}

Profiler::~Profiler() = default;

std::unique_ptr<Flight> Profiler::NewFlight(bool present) {
    vk::QueryPool pool;
    {
        std::scoped_lock lock{pool_mutex};
        if (!free_pools.empty()) {
            pool = std::move(free_pools.back());
            free_pools.pop_back();
        }
    }
    if (!pool) {
        try {
            pool = device.GetLogical().CreateQueryPool({
                .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queryType = VK_QUERY_TYPE_TIMESTAMP,
                .queryCount = Flight::Capacity(PerOp()),
                .pipelineStatistics = 0,
            });
            device.GetLogical().ResetQueryPool(*pool, 0, Flight::Capacity(PerOp()));
        } catch (const vk::Exception& exception) {
            if (!pool_failure_logged) {
                pool_failure_logged = true;
                LOG_ERROR(Render_Vulkan, "[nvn] gpuprof: query pool creation failed: {}",
                          exception.what());
            }
            return nullptr;
        }
    }
    return std::make_unique<Flight>(*this, std::move(pool), present);
}

void Profiler::Retire(vk::QueryPool&& pool) {
    std::scoped_lock lock{pool_mutex};
    retired_pools.push_back(std::move(pool));
}

void Profiler::Add(Window& w, const Tag& tag, double ns, u64 units) {
    Agg& agg = w.tags[tag];
    agg.ns += ns;
    ++agg.n;
    agg.units += units;
    const size_t kind = static_cast<size_t>(tag.kind);
    w.kind_ns[kind] += ns;
    ++w.kind_n[kind];
    if (!Overlaps(tag.kind)) {
        (tag.zero_size ? w.zero_ns : w.other_ns) += ns;
        ++(tag.zero_size ? w.zero_n : w.other_n);
    }
    if (tag.kind == Kind::Pass && (tag.shape & 0xF) == 0 && ((tag.shape >> 4) & 1) != 0) {
        w.depth_only_ns += ns;
        ++w.depth_only_n;
        w.depth_only_draws += tag.count;
        w.depth_only_units += units;
    }
}

void Profiler::AddIdle(Window& w, double ns, bool frame_start) {
    w.idle_ns += ns;
    ++w.idle_n;
    const size_t bucket = ns < 1e5 ? 0 : ns < 1e6 ? 1 : ns < 5e6 ? 2 : 3;
    w.idle_bucket_ns[bucket] += ns;
    ++w.idle_bucket_n[bucket];
    if (frame_start) {
        w.idle_frame_start_ns += ns;
    }
}

void Profiler::LogSegments(const Flight& flight) const {
    // Per command buffer: when the CPU submitted it, its GPU span (GPU clock, us), the measured
    // intervals inside it (passes, dispatches, copies, clears, presents, uploads) and, in mode
    // `ops`, the records that took the most of it. The GPU clock has its own epoch: the reader
    // aligns it with the smallest begin - submit.
    double work_ns = 0.0;
    u32 work_n = 0;
    std::array<double, 256> op_ns{};
    // Mode `ops`: the first record's start and the last record's end, so the command buffer's
    // head (before the first record) and tail (the end-of-segment work after the last one) show.
    u64 ops_first = 0;
    u64 ops_last = 0;
    bool have_ops = false;
    size_t segment = 0;
    for (const Flight::Interval& interval : flight.intervals) {
        const u64* begin = &results[size_t{interval.begin} * 2];
        const u64* end = &results[size_t{interval.end} * 2];
        if (begin[1] == 0 || end[1] == 0) {
            continue;
        }
        const double ns = static_cast<double>((end[0] - begin[0]) & mask) * period_ns;
        if (interval.tag.kind == Kind::Op) {
            op_ns[interval.tag.hash_a & 0xFF] += ns;
            if (!have_ops) {
                ops_first = begin[0] & mask;
                have_ops = true;
            }
            ops_last = end[0] & mask;
            continue;
        }
        if (interval.tag.kind != Kind::Segment) {
            if (!Overlaps(interval.tag.kind)) {
                work_ns += ns;
                ++work_n;
            }
            continue;
        }
        const Flight::SegmentSubmit submit = segment < flight.segment_submits.size()
                                                 ? flight.segment_submits[segment]
                                                 : Flight::SegmentSubmit{0, ~u32{0}};
        ++segment;
        const auto to_us = [this](u64 ticks) {
            return static_cast<u64>(static_cast<double>(ticks & mask) * period_ns / 1000.0);
        };
        std::string ops;
        for (int k = 0; k < 3; ++k) {
            const auto top = std::ranges::max_element(op_ns);
            if (*top < 1000.0) {
                break;
            }
            ops += fmt::format("{}{}:{:.0f}", ops.empty() ? "" : ",", top - op_ns.begin(),
                               *top / 1000.0);
            *top = 0.0;
        }
        const u64 seg_begin = begin[0] & mask;
        const u64 seg_end = end[0] & mask;
        const auto part_us = [&](u64 from, u64 to) {
            return to > from ? static_cast<double>(to - from) * period_ns / 1000.0 : 0.0;
        };
        Debug::TimelineEvent(
            "gseg", flight.seq,
            fmt::format("q={} sub={} gb={} ge={} work_us={:.0f} n={} ops={} head_us={:.0f} "
                        "tail_us={:.0f}",
                        submit.queue, submit.cpu_us, to_us(seg_begin), to_us(seg_end),
                        work_ns / 1000.0, work_n, ops.empty() ? "-" : ops,
                        have_ops ? part_us(seg_begin, ops_first) : 0.0,
                        have_ops ? part_us(ops_last, seg_end) : 0.0));
        work_ns = 0.0;
        work_n = 0;
        op_ns.fill(0.0);
        have_ops = false;
    }
}

void Profiler::Complete(Flight& flight) {
    if (!flight.pool) {
        return;
    }
    const vk::Device& logical = device.GetLogical();
    FlightLine line;
    line.seq = flight.seq;
    line.present = flight.is_present;
    bool have_line = false;
    // EDEN_NX2_NVN_COSTSPLIT: the GPU time of a submission recorded in a slot past its warm-up.
    const bool split =
        costsplit != nullptr && flight.costsplit_slot >= 0 && flight.costsplit_measured;
    double split_class_ns = 0.0;
    u64 split_passes = 0;
    if (flight.used > 0) {
        results.assign(size_t{flight.used} * 2, 0);
        // Value + availability per query. NOT_READY only says some are unavailable (work that
        // never ran); those intervals are skipped below.
        const VkResult result = logical.GetQueryResults(
            *flight.pool, 0, flight.used, results.size() * sizeof(u64), results.data(),
            2 * sizeof(u64), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result == VK_SUCCESS || result == VK_NOT_READY) {
            bool first_segment = true;
            for (const Flight::Interval& interval : flight.intervals) {
                const u64* begin = &results[size_t{interval.begin} * 2];
                const u64* end = &results[size_t{interval.end} * 2];
                if (begin[1] == 0 || end[1] == 0) {
                    ++window.unavailable;
                    ++total.unavailable;
                    continue;
                }
                const double ns = static_cast<double>((end[0] - begin[0]) & mask) * period_ns;
                line.kind_n[static_cast<size_t>(interval.tag.kind)]++;
                if (!Overlaps(interval.tag.kind) &&
                    ns > line.biggest_ns) {
                    line.biggest_ns = ns;
                    line.biggest = interval.tag;
                }
                Add(window, interval.tag, ns, interval.units);
                Add(total, interval.tag, ns, interval.units);
                if (split && interval.split != 0 && interval.tag.kind != Kind::Draw) {
                    split_class_ns += ns;
                    ++split_passes;
                }
                if (interval.tag.kind == Kind::Segment) {
                    const auto now = std::chrono::steady_clock::now();
                    const size_t second = static_cast<size_t>(
                        std::chrono::duration_cast<std::chrono::seconds>(now - window_start)
                            .count());
                    if (busy_ns_per_second.size() <= second) {
                        busy_ns_per_second.resize(second + 1, 0.0);
                    }
                    busy_ns_per_second[second] += ns;
                    line.busy_ns += ns;
                    if (!have_line) {
                        line.first_begin = begin[0] & mask;
                        have_line = true;
                    }
                    line.last_end = end[0] & mask;
                    // The queue is in order: the gap to the previous command buffer is time the
                    // GPU had nothing to run (or waited on a semaphore).
                    const u64 first = begin[0] & mask;
                    // Masked: the counter may wrap; a "gap" past half its range is overlap.
                    const u64 delta = (first - last_segment_end) & mask;
                    if (have_segment_end && delta != 0 && delta <= (mask >> 1)) {
                        const double gap = static_cast<double>(delta) * period_ns;
                        const bool frame_start = previous_was_present && first_segment;
                        AddIdle(window, gap, frame_start);
                        AddIdle(total, gap, frame_start);
                    }
                    last_segment_end = end[0] & mask;
                    have_segment_end = true;
                    first_segment = false;
                }
            }
            if (Debug::Timeline()) {
                LogSegments(flight);
            }
        }
        logical.ResetQueryPool(*flight.pool, 0, flight.used);
    }
    window.dropped += flight.dropped;
    total.dropped += flight.dropped;
    ++window.flights;
    ++total.flights;
    if (flight.is_present) {
        ++window.presents;
        ++total.presents;
    }
    if (split) {
        // Earlier slots are logged once a submission past this slot's warm-up completed: the
        // queues' late submissions of the slot before have completed by then.
        const u64 slot = static_cast<u64>(flight.costsplit_slot);
        costsplit->AddGpu(slot, flight.is_present, line.busy_ns, split_class_ns, split_passes);
        costsplit->ReportFinished(slot);
    }
    if (have_line) {
        frame_flights.push_back(line);
    }
    if (flight.is_present && !frame_flights.empty()) {
        // One frame: the command buffers since the previous present, this one's included.
        double busy_ns = frame_busy_carry_ns;
        for (const FlightLine& f : frame_flights) {
            busy_ns += f.busy_ns;
        }
        const u64 first = have_frame_first ? frame_first_begin : frame_flights.front().first_begin;
        const u64 last = frame_flights.back().last_end;
        const double span_ns = static_cast<double>((last - first) & mask) * period_ns;
        for (Window* w : {&window, &total}) {
            w->frame_busy_ms.push_back(static_cast<float>(busy_ns / 1e6));
            w->frame_span_ms.push_back(static_cast<float>(span_ns / 1e6));
        }
    }
    if (flight.is_present) {
        if (!quiet && total.presents % 200 == 100) {
            ReportFrame();
        }
        frame_flights.clear();
        frame_busy_carry_ns = 0.0;
        have_frame_first = false;
    } else if (frame_flights.size() > 256) {
        // Only the detailed lines are dropped: the frame's totals carry over to its present.
        if (!have_frame_first) {
            frame_first_begin = frame_flights.front().first_begin;
            have_frame_first = true;
        }
        for (const FlightLine& f : frame_flights) {
            frame_busy_carry_ns += f.busy_ns;
        }
        frame_flights.clear();
    }
    previous_was_present = flight.is_present;
    flight.intervals.clear();
    flight.segment_submits.clear();
    flight.segments_added = 0;
    flight.used = 0;
    flight.dropped = 0;
    {
        std::scoped_lock lock{pool_mutex};
        free_pools.push_back(std::move(flight.pool));
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - window_start >= REPORT_INTERVAL) {
        const double seconds = std::chrono::duration<double>(now - window_start).count();
        if (!quiet) {
            Report(window, "10 s window", seconds, &busy_ns_per_second);
        }
        window = {};
        busy_ns_per_second.clear();
        window_start = now;
    }
}

void Profiler::ReportFrame() {
    if (frame_flights.empty()) {
        return;
    }
    const u64 origin = frame_flights.front().first_begin;
    const auto ms = [&](u64 ticks) {
        return ticks >= origin ? static_cast<double>(ticks - origin) * period_ns / 1e6 : 0.0;
    };
    LOG_INFO(Render_Vulkan,
             "[nvn] gpuprof frame timeline (present #{}, {} submissions; ms from the first command "
             "buffer):",
             total.presents, frame_flights.size());
    u64 previous_end = origin;
    for (const FlightLine& f : frame_flights) {
        const double gap = f.first_begin > previous_end
                               ? static_cast<double>(f.first_begin - previous_end) * period_ns / 1e6
                               : 0.0;
        const auto count = [&](Kind kind) { return f.kind_n[static_cast<size_t>(kind)]; };
        LOG_INFO(Render_Vulkan,
                 "[nvn] gpuprof   seq {}{}: +{:.2f}..+{:.2f} ms busy {:.2f} gap before {:.2f} | "
                 "pass {} dispatch {} copy {} clear {} upload {} | biggest {:.2f} ms {}",
                 f.seq, f.present ? " (present)" : "", ms(f.first_begin), ms(f.last_end),
                 f.busy_ns / 1e6, gap, count(Kind::Pass), count(Kind::Dispatch), count(Kind::Copy),
                 count(Kind::Clear), count(Kind::Upload), f.biggest_ns / 1e6,
                 f.biggest_ns > 0.0 ? DescribeTag(f.biggest) : std::string{});
        previous_end = f.last_end;
    }
}

void Profiler::ReportTotals() {
    if (costsplit) {
        costsplit->ReportRun();
    }
    if (quiet) {
        return;
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count();
    Report(total, "whole run", seconds, nullptr);
}

void Profiler::Report(const Window& w, const char* label, double seconds,
                      const std::vector<double>* busy_per_second) {
    if (w.flights == 0) {
        return;
    }
    // Everything per present; a window without presents (loading) reports per window.
    const double frames = static_cast<double>((std::max<u64>)(w.presents, 1));
    const auto per_frame = [&](double ns) { return ns / frames / 1e6; };
    const double segment_ns = w.kind_ns[static_cast<size_t>(Kind::Segment)];
    double measured_ns = 0.0;
    std::string kinds;
    for (size_t k = 0; k < KIND_COUNT; ++k) {
        const Kind kind = static_cast<Kind>(k);
        if (Overlaps(kind)) {
            continue;
        }
        measured_ns += w.kind_ns[k];
        kinds += fmt::format("{}{} {:.3f}", kinds.empty() ? "" : ", ", KindName(kind),
                             per_frame(w.kind_ns[k]));
    }
    const double fps = static_cast<double>(w.presents) / seconds;
    LOG_INFO(Render_Vulkan,
             "[nvn] gpuprof {} ({:.1f} s): presents={} ({:.1f}/s) submissions={} "
             "gpu_busy={:.3f} ms/present ({:.1f}% of wall; frame time {:.2f} ms)",
             label, seconds, w.presents, fps, w.flights, per_frame(segment_ns),
             segment_ns / (seconds * 1e9) * 100.0, fps > 0.0 ? 1000.0 / fps : 0.0);
    if (!w.frame_span_ms.empty()) {
        const auto describe = [](std::vector<float> samples) {
            std::sort(samples.begin(), samples.end());
            double sum = 0.0;
            for (const float ms : samples) {
                sum += ms;
            }
            const auto at = [&](double q) {
                return samples[static_cast<size_t>(q * static_cast<double>(samples.size() - 1))];
            };
            return fmt::format("avg {:.3f} p50 {:.3f} p95 {:.3f} p99 {:.3f} max {:.3f}",
                               sum / static_cast<double>(samples.size()), at(0.5), at(0.95),
                               at(0.99), samples.back());
        };
        LOG_INFO(Render_Vulkan,
                 "[nvn] gpuprof {}: gpu ms per frame over {} frames{}: busy (sum of command "
                 "buffers) [{}]; span (first stamp to last, gaps included) [{}]; idle between "
                 "command buffers {:.3f} ms/frame",
                 label, w.frame_span_ms.size(), FrameOnly() ? " (frame-only timer)" : "",
                 describe(w.frame_busy_ms), describe(w.frame_span_ms), per_frame(w.idle_ns));
    }
    if (FrameOnly()) {
        return;
    }
    LOG_INFO(Render_Vulkan,
             "[nvn] gpuprof {}: in intervals {:.3f} ms/present [{}] unattributed {:.3f} ms/present "
             "(barriers, layout transitions, uncovered ops); size0 intervals {:.3f} ms/present ({} "
             "intervals) vs others {:.3f} ms/present ({}); dropped={} unavailable={}",
             label, per_frame(measured_ns), kinds, per_frame(segment_ns - measured_ns),
             per_frame(w.zero_ns), w.zero_n, per_frame(w.other_ns), w.other_n, w.dropped,
             w.unavailable);
    LOG_INFO(Render_Vulkan,
             "[nvn] gpuprof {}: gpu idle between command buffers {:.3f} ms/present ({:.1f}% of "
             "wall) in {:.1f} gaps/present; by gap size <0.1 ms {:.3f} ({:.1f}), <1 ms {:.3f} "
             "({:.1f}), <5 ms {:.3f} ({:.1f}), >=5 ms {:.3f} ({:.1f}) [ms/present (gaps/present)]; "
             "before the first command buffer after a present {:.3f} ms/present",
             label, per_frame(w.idle_ns), w.idle_ns / (seconds * 1e9) * 100.0,
             static_cast<double>(w.idle_n) / frames, per_frame(w.idle_bucket_ns[0]),
             static_cast<double>(w.idle_bucket_n[0]) / frames, per_frame(w.idle_bucket_ns[1]),
             static_cast<double>(w.idle_bucket_n[1]) / frames, per_frame(w.idle_bucket_ns[2]),
             static_cast<double>(w.idle_bucket_n[2]) / frames, per_frame(w.idle_bucket_ns[3]),
             static_cast<double>(w.idle_bucket_n[3]) / frames,
             per_frame(w.idle_frame_start_ns));
    if (busy_per_second != nullptr && !busy_per_second->empty()) {
        std::string line;
        for (const double ns : *busy_per_second) {
            line += fmt::format(" {:.0f}", ns / 1e7); // percent of one second
        }
        LOG_INFO(Render_Vulkan, "[nvn] gpuprof {}: gpu busy % per second:{}", label, line);
    }
    if (w.depth_only_n != 0) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] gpuprof {}: depth-only passes {:.3f} ms/present ({:.2f} passes, {:.1f} "
                 "draws, {:.0f} verts per present; {:.3f} us per draw)",
                 label, per_frame(w.depth_only_ns), static_cast<double>(w.depth_only_n) / frames,
                 static_cast<double>(w.depth_only_draws) / frames,
                 static_cast<double>(w.depth_only_units) / frames,
                 w.depth_only_draws != 0
                     ? w.depth_only_ns / static_cast<double>(w.depth_only_draws) / 1e3
                     : 0.0);
    }

    // Top tags; drawn intervals and records (mode `ops`) listed apart: they overlap the passes and
    // the other intervals that contain them or that they contain.
    // (`Kind::Segment` stands for the measured kinds: segments themselves are never listed.)
    for (const Kind apart : {Kind::Segment, Kind::Draw, Kind::Op}) {
        std::vector<std::pair<const Tag*, const Agg*>> rows;
        for (const auto& [tag, agg] : w.tags) {
            const bool listed = apart == Kind::Segment ? !Overlaps(tag.kind) : tag.kind == apart;
            if (!listed) {
                continue;
            }
            rows.emplace_back(&tag, &agg);
        }
        if (rows.empty()) {
            continue;
        }
        std::sort(rows.begin(), rows.end(),
                  [](const auto& a, const auto& b) { return a.second->ns > b.second->ns; });
        for (size_t i = 0; i < rows.size() && i < TOP_TAGS; ++i) {
            const Tag& tag = *rows[i].first;
            const Agg& agg = *rows[i].second;
            LOG_INFO(Render_Vulkan,
                     "[nvn] gpuprof {} top {:>2}: {:7.3f} ms/present ({:5.2f} per present, avg "
                     "{:.3f} ms{}) {}",
                     apart == Kind::Draw ? "draws" : apart == Kind::Op ? "ops" : "tags", i + 1,
                     per_frame(agg.ns),
                     static_cast<double>(agg.n) / frames,
                     agg.n != 0 ? agg.ns / static_cast<double>(agg.n) / 1e6 : 0.0,
                     agg.units != 0 ? fmt::format(", {:.0f} verts/present",
                                                  static_cast<double>(agg.units) / frames)
                                    : std::string{},
                     DescribeTag(tag));
        }
    }

    // The same by program, summing every shape / count of one kind and program pair.
    {
        struct Program {
            double ns{};
            u64 n{};
            bool zero{};
        };
        std::map<std::tuple<Kind, u64, u64>, Program> programs;
        for (const auto& [tag, agg] : w.tags) {
            if (tag.kind != Kind::Pass && tag.kind != Kind::Draw && tag.kind != Kind::Dispatch) {
                continue;
            }
            Program& program = programs[{tag.kind, tag.hash_a, tag.hash_b}];
            program.ns += agg.ns;
            program.n += agg.n;
            program.zero |= tag.zero_size;
        }
        std::vector<std::pair<std::tuple<Kind, u64, u64>, Program>> rows(programs.begin(),
                                                                          programs.end());
        std::sort(rows.begin(), rows.end(),
                  [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
        for (size_t i = 0; i < rows.size() && i < TOP_PROGRAMS; ++i) {
            const auto& [key, program] = rows[i];
            LOG_INFO(Render_Vulkan,
                     "[nvn] gpuprof by program {:>2}: {:7.3f} ms/present ({:5.2f} per present) "
                     "{:<8} {}={:016x}{} size0={}",
                     i + 1, per_frame(program.ns), static_cast<double>(program.n) / frames,
                     KindName(std::get<0>(key)),
                     std::get<0>(key) == Kind::Dispatch ? "cs" : "vs", std::get<1>(key),
                     std::get<0>(key) == Kind::Dispatch
                         ? std::string{}
                         : fmt::format(" fs={:016x}", std::get<2>(key)),
                     program.zero ? 1 : 0);
        }
    }
}

} // namespace Vulkan::Nvn::GpuProf
