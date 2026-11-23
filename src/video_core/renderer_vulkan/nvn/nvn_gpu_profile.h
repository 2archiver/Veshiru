// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// GPU time profiler of the NVN replay, `EDEN_NX2_NVN_GPU_PROFILE=1` (2: also one interval per draw;
// `frame`: only the command buffers, see `MODE_FRAME`). See DESIGN.md "GPU profiler". Nothing here
// runs when the switch is unset: `Context::Impl` holds no `Profiler`, no `Flight` is created and no
// query pool exists.
//
// The replay thread writes `vkCmdWriteTimestamp`s into a query pool owned by the submission's
// `Flight` and records one `Interval` (two query indices and a `Tag`) per measured region. The
// completion thread reads the timestamps once the submission's timeline value is reached
// (`Profiler::Complete`), aggregates them per tag and logs a report every 10 s.

#pragma once

#include <array>
#include <chrono>
#include <compare>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
}

namespace Vulkan::Nvn::CostSplit {
class Recorder;
}

namespace Vulkan::Nvn::GpuProf {

inline constexpr u32 NONE = ~0u;

/// `EDEN_NX2_NVN_GPU_PROFILE=frame`: one timestamp pair per command buffer and nothing inside it,
/// so the GPU keeps the overlap between passes that the per-interval modes remove (their
/// BOTTOM_OF_PIPE stamps drain the pipeline at every boundary). Judges barrier/pass changes.
inline constexpr u32 MODE_FRAME = 3;

/// `EDEN_NX2_NVN_GPU_PROFILE=ops`: mode 1 plus one `Kind::Op` interval per replayed record, from
/// its start to the next record's start (the pass end, barriers and counter work it causes
/// included), so the time outside every other interval is named by the op that spent it.
inline constexpr u32 MODE_OPS = 4;

/// The value of EDEN_NX2_NVN_GPU_PROFILE: `frame` is MODE_FRAME, `ops` MODE_OPS, anything else a
/// number (0 off).
[[nodiscard]] u32 ParseMode(const char* text) noexcept;

/// `Tag::hash_a` of the kinds that have no program: what the work was.
inline constexpr u64 SUB_FEEDBACK = 1; ///< copies of a render target the draw samples
inline constexpr u64 SUB_INLINE = 2;   ///< inline uniform updates written into pools
inline constexpr u64 SUB_MIRROR = 3;   ///< guest bytes copied into a mirror pool
inline constexpr u64 SUB_STORAGE = 4;  ///< guest bytes uploaded into an image
inline constexpr u64 SUB_SHADOW = 5;   ///< storage shadow copy-in / copy-back (nvn_draw_shadow.cpp)
inline constexpr u64 SUB_VERTEX = 6;   ///< vertex ranges staged for a pass (nvn_vertex_stage.h)

enum class Kind : u8 {
    Segment,  ///< one command buffer of a submission, start to end
    Pass,     ///< begin..end of a render pass (the draws and the pass's own load/store work)
    Draw,     ///< one draw of a pass (mode 2 only): previous timestamp .. after the draw
    Dispatch, ///< a compute dispatch
    Copy,     ///< buffer/texture copies, feedback copies, inline uniform updates
    Clear,    ///< ClearTexture / ClearBuffer (and their partial-clear passes)
    Present,  ///< the present blit of a frame
    Upload,   ///< guest contents uploaded to images / mirrors outside of any copy op
    Op,       ///< one replayed record (mode `ops` only; overlaps the other kinds, `Tag::hash_a` = op)
    Count,
};

[[nodiscard]] const char* KindName(Kind kind) noexcept;

/// What an interval was. The ordering is the aggregation key.
struct Tag {
    Kind kind{};
    bool zero_size{};  ///< a storage buffer bound with an address and size 0 was used inside
    u32 count{};       ///< draws in a pass; groups of a dispatch; bytes of a buffer copy/clear
    u64 hash_a{};      ///< vertex program hash of the first draw (draw/pass); compute program hash
    u64 hash_b{};      ///< fragment program hash of the first draw (draw/pass)
    s32 format{};      ///< VkFormat: first color target (pass/draw), texture format (copies)
    u32 width{};       ///< target width; dispatch: groups x
    u32 height{};      ///< target height; dispatch: groups y
    u32 shape{};       ///< pass: colors | depth << 4 | layers << 8 | samples << 24; dispatch: groups z

    auto operator<=>(const Tag&) const = default;
};

[[nodiscard]] u32 PackShape(u32 colors, bool depth, u32 layers, u32 samples) noexcept;

class Profiler;

/// The timestamps of one submission (or present). Created by the replay thread, read and recycled
/// by the completion thread after the work finished; the hand-over is `InFlight::submitted`.
class Flight {
public:
    /// Timestamps per submission. A busy frame records ~100-300 intervals (mode 1) or one more per
    /// draw (mode 2); intervals past the capacity are dropped and counted.
    static constexpr u32 CAPACITY = 16384;
    /// Mode `ops`: one stamp per replayed record as well (an OoT gameplay command buffer has
    /// ~10k records plus the scopes inside them).
    static constexpr u32 OPS_CAPACITY = 262144;
    [[nodiscard]] static constexpr u32 Capacity(bool per_op) noexcept {
        return per_op ? OPS_CAPACITY : CAPACITY;
    }

    struct Interval {
        u32 begin;
        u32 end;
        u64 units; ///< vertices x instances of a draw
        Tag tag;
        u32 split; ///< draws / dispatches of the cost-split class inside (nvn_costsplit.h)
    };

    Flight(Profiler& owner, vk::QueryPool pool, bool present) noexcept;
    ~Flight();
    Flight(const Flight&) = delete;
    Flight& operator=(const Flight&) = delete;

    /// Writes a timestamp after a full memory barrier (all earlier work and its writes finished:
    /// a BOTTOM_OF_PIPE stamp alone does not wait for compute on NVIDIA) and returns its index,
    /// NONE when full or in frame-only mode (only `StampSegment` stamps there).
    /// `inside_render_pass`: no barrier (invalid there); only the per-draw mode (2) stamps inside a
    /// pass and its intervals are launch-ordered, not completion-ordered.
    [[nodiscard]] u32 Stamp(vk::CommandBuffer cmd, bool inside_render_pass = false) noexcept;
    /// The stamp of a command buffer's first and last command: taken in every mode.
    [[nodiscard]] u32 StampSegment(vk::CommandBuffer cmd, bool inside_render_pass = false) noexcept;
    /// Records an interval between two stamps; ignored when either is NONE. `split`: the
    /// cost-split class work inside it.
    void Add(u32 begin, u32 end, const Tag& tag, u64 units = 0, u32 split = 0);
    [[nodiscard]] bool PerDraw() const noexcept {
        return per_draw;
    }
    /// Timestamps this flight can take (Capacity of its profiler's mode).
    [[nodiscard]] u32 StampCapacity() const noexcept {
        return Capacity(per_op);
    }
    /// Mode `ops`: a replayed record `op` starts here; ends the interval of the record before (in
    /// the same command buffer). Does nothing in the other modes.
    /// `inside_render_pass`: a pass is open, so no barrier can go before the stamp (the time of
    /// the pass lands on the record that ends it).
    void MarkOp(vk::CommandBuffer cmd, u16 op, bool inside_render_pass = false);
    /// Mode `ops`: the command buffer ends; ends the interval of its last record.
    void EndOps(vk::CommandBuffer cmd);
    /// EDEN_NX2_NVN_TIMELINE: the command buffers of NVN queue `queue` whose segment intervals were
    /// added since the last call were handed to vkQueueSubmit at `cpu_us` (steady-clock
    /// microseconds since its epoch); the completion logs each one's GPU span next to it (`gseg`
    /// timeline lines).
    void NoteSubmitted(u64 cpu_us, u32 queue);

    /// Nesting depth of `Scope`s (only the outermost one measures).
    u32 depth{0};
    /// Submission sequence number (the frame timeline lines name it).
    u64 seq{0};
    /// EDEN_NX2_NVN_COSTSPLIT: the schedule slot the submission was recorded in (-1: none) and
    /// whether that slot was past its warm-up.
    s64 costsplit_slot{-1};
    bool costsplit_measured{false};

private:
    friend class Profiler;
    Profiler& owner;
    vk::QueryPool pool;
    bool is_present;
    bool per_draw;
    bool frame_only;
    bool per_op;
    u32 op_begin{NONE}; ///< mode `ops`: stamp at the start of the current record
    u16 op_code{};
    u32 used{0};
    u32 dropped{0};
    std::vector<Interval> intervals;
    /// NoteSubmitted: per Segment interval in order, the submit time and queue (timeline only).
    struct SegmentSubmit {
        u64 cpu_us;
        u32 queue;
    };
    std::vector<SegmentSubmit> segment_submits;
    size_t segments_added{0}; ///< Segment intervals added (NoteSubmitted covers those up to here)
};

/// Measures the commands recorded in its lifetime into `cmd` (outside a render pass). Nested
/// scopes are ignored, so the outermost operation owns the interval. Does nothing for a null
/// flight.
class Scope {
public:
    Scope() = default;
    Scope(Flight* flight, vk::CommandBuffer cmd, const Tag& tag, u64 units = 0,
          u32 split = 0) noexcept
        : flight_{flight} {
        if (flight_ != nullptr) {
            Begin(cmd, tag, units, split);
        }
    }
    ~Scope() {
        if (flight_ != nullptr) {
            End();
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    void Begin(vk::CommandBuffer cmd, const Tag& tag, u64 units, u32 split) noexcept;
    void End() noexcept;

    Flight* flight_{};
    vk::CommandBuffer cmd_{};
    u32 begin_{NONE};
    bool outermost_{false};
    u64 units_{};
    u32 split_{};
    Tag tag_{};
};

class Profiler {
public:
    /// Null when `mode` is 0 or the device cannot do it (host query reset, timestamps). `quiet`
    /// (performance HUD only, no `EDEN_NX2_NVN_GPU_PROFILE`) keeps the 10 s log reports off.
    /// `costsplit` (EDEN_NX2_NVN_COSTSPLIT, may be null) gets the GPU time of each submission.
    [[nodiscard]] static std::unique_ptr<Profiler> Create(const Device& device, u32 mode,
                                                          bool quiet = false,
                                                          CostSplit::Recorder* costsplit = nullptr);

    Profiler(const Device& device, u32 mode, double period_ns, u64 mask, bool quiet = false,
             CostSplit::Recorder* costsplit = nullptr);
    ~Profiler();

    /// Replay thread: the timestamps of a new submission; null when no pool could be made.
    [[nodiscard]] std::unique_ptr<Flight> NewFlight(bool present);
    /// Completion thread, after the flight's timeline value was reached: reads the timestamps,
    /// aggregates them and logs the report every 10 s. The pool is recycled.
    void Complete(Flight& flight);
    /// Completion thread has stopped: logs the totals of the whole run.
    void ReportTotals();
    /// Intervals and nanoseconds of `kind` over the whole run so far (completion thread, or any
    /// thread once it is idle: tests).
    [[nodiscard]] std::pair<u64, double> KindTotal(Kind kind) const noexcept {
        return {total.kind_n[static_cast<size_t>(kind)], total.kind_ns[static_cast<size_t>(kind)]};
    }

    [[nodiscard]] bool PerDraw() const noexcept {
        return mode == 2;
    }
    /// `EDEN_NX2_NVN_GPU_PROFILE=frame`: only command-buffer stamps; the report is per frame.
    [[nodiscard]] bool FrameOnly() const noexcept {
        return mode == MODE_FRAME;
    }
    /// `EDEN_NX2_NVN_GPU_PROFILE=ops`: one `Kind::Op` interval per replayed record as well.
    [[nodiscard]] bool PerOp() const noexcept {
        return mode == MODE_OPS;
    }
    /// Frames (presents) measured over the whole run so far and the sum of their GPU span
    /// (first command buffer's first stamp .. last command buffer's last stamp), milliseconds.
    [[nodiscard]] std::pair<u64, double> FrameSpanTotal() const noexcept {
        double sum = 0.0;
        for (const float ms : total.frame_span_ms) {
            sum += ms;
        }
        return {total.frame_span_ms.size(), sum};
    }

private:
    friend class Flight;

    struct Agg {
        double ns{};
        u64 n{};
        u64 units{};
    };
    /// Everything aggregated over a window (10 s) or over the whole run.
    struct Window {
        std::map<Tag, Agg> tags;
        std::array<double, static_cast<size_t>(Kind::Count)> kind_ns{};
        std::array<u64, static_cast<size_t>(Kind::Count)> kind_n{};
        double zero_ns{};   ///< non-segment, non-draw intervals with the size-0 flag
        u64 zero_n{};
        double other_ns{};  ///< ... without it
        u64 other_n{};
        u64 presents{};
        u64 flights{};
        u64 dropped{};
        u64 unavailable{}; ///< intervals whose timestamps were not available (aborted work)
        /// Time the queue sat empty between two command buffers (the next one's first timestamp
        /// minus the previous one's last): total, count, and by gap size (<0.1, <1, <5, >=5 ms).
        double idle_ns{};
        u64 idle_n{};
        std::array<double, 4> idle_bucket_ns{};
        std::array<u64, 4> idle_bucket_n{};
        double idle_frame_start_ns{}; ///< ... of it, before the first command buffer after a present
        /// Passes with a depth target and no colour target (shadow maps): GPU time, passes, the
        /// draws in them and their vertices x instances (direct draws).
        double depth_only_ns{};
        u64 depth_only_n{};
        u64 depth_only_draws{};
        u64 depth_only_units{};
        /// One sample per frame (a present and the command buffers since the previous one): the
        /// sum of its command buffers' GPU time, and the time from the first one's start to the
        /// last one's end (busy plus the gaps the GPU waited in), milliseconds.
        std::vector<float> frame_busy_ms;
        std::vector<float> frame_span_ms;
    };

    void Retire(vk::QueryPool&& pool);
    void AddIdle(Window& window, double ns, bool frame_start);
    /// EDEN_NX2_NVN_TIMELINE: one `gseg` line per command buffer of `flight` (its query results
    /// are in `results`).
    void LogSegments(const Flight& flight) const;
    void Add(Window& window, const Tag& tag, double ns, u64 units);
    void Report(const Window& window, const char* label, double seconds,
                const std::vector<double>* busy_per_second);

    const Device& device;
    const u32 mode;
    const double period_ns;
    const u64 mask;
    const bool quiet; ///< performance HUD only: aggregate for it, log nothing every 10 s
    CostSplit::Recorder* const costsplit;

    std::mutex pool_mutex;
    std::vector<vk::QueryPool> free_pools;
    std::vector<vk::QueryPool> retired_pools; ///< pools of flights that never completed
    bool pool_failure_logged{false};

    // Completion thread only.
    std::vector<u64> results;
    Window window;
    Window total;
    std::chrono::steady_clock::time_point run_start;
    std::chrono::steady_clock::time_point window_start;
    std::vector<double> busy_ns_per_second; ///< segment ns completed in each second of the window
    /// End of the previous command buffer in GPU ticks, and whether the flight before was a present.
    u64 last_segment_end{0};
    bool have_segment_end{false};
    bool previous_was_present{false};
    /// The command buffers of the frame being completed (since the last present), for the
    /// timeline printed for every 200th frame.
    struct FlightLine {
        u64 seq{};
        bool present{};
        u64 first_begin{};
        u64 last_end{};
        double busy_ns{};
        std::array<u32, static_cast<size_t>(Kind::Count)> kind_n{};
        double biggest_ns{};
        Tag biggest;
    };
    std::vector<FlightLine> frame_flights;
    /// The frame's busy time and first timestamp from lines trimmed off frame_flights.
    double frame_busy_carry_ns{0.0};
    u64 frame_first_begin{0};
    bool have_frame_first{false};
    void ReportFrame();
};

} // namespace Vulkan::Nvn::GpuProf
