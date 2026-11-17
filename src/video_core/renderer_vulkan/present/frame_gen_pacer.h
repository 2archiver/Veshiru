// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <optional>

#include "common/common_types.h"

namespace Vulkan {

struct FrameGenPlan {
    size_t generations{};
    bool warm{};
    /// True when generation was skipped for a single isolated hitch frame while keeping optical
    /// flow history and pacing state intact.
    bool hitch_skipped{};
};

/// The generations a game frame can present without forcing the renderer to wait for a free
/// present frame: those free now, minus one kept in reserve for the next real game frame.
/// Because frame generation runs on FIFO, when the display accepts fewer presents than the game
/// and its generated frames produce (e.g. 2x at 60 Hz), waiting for a free frame would throttle
/// real game FPS below its native rate. Generation steps back instead.
[[nodiscard]] inline size_t GenerationsWithoutWaiting(size_t planned, size_t free_frames) {
    return (std::min)(planned, free_frames > 0 ? free_frames - 1 : 0);
}

class FrameGenPacer {
public:
    static constexpr size_t MAX_TRACKED_LEVELS = 8;

    /// blocked: of the time since the last call, how long the renderer waited for the present
    /// thread to free a frame (FIFO/pacing backpressure, excluding GPU execution fence waits).
    [[nodiscard]] FrameGenPlan Plan(size_t capacity,
                                    std::chrono::steady_clock::duration blocked = {});

    /// Deterministic clock/settings entry point used by Plan() and unit tests.
    [[nodiscard]] FrameGenPlan PlanAt(std::chrono::steady_clock::time_point now, size_t capacity,
                                      std::chrono::steady_clock::duration blocked, f32 target_rate,
                                      f32 guest_interval_seconds, size_t max_generations,
                                      size_t fixed_generations);

    void Reset();

    /// The smoothed time between real frames in seconds, or 0 while it is unknown.
    [[nodiscard]] f32 SmoothedInterval() const {
        return smoothed_interval;
    }

    /// Current generation cap chosen by the pacer (0 while idle or stabilizing).
    [[nodiscard]] size_t CurrentLimit() const {
        return limit;
    }

    /// Peak failure count across tracked generation levels (diagnostic / test accessor).
    [[nodiscard]] u32 ProbeFailures() const {
        u32 peak = 0;
        for (const u32 count : level_failures) {
            peak = (std::max)(peak, count);
        }
        return peak;
    }

    /// Failure count associated with probing a specific generation level.
    [[nodiscard]] u32 LevelFailures(size_t target_level) const {
        return target_level < MAX_TRACKED_LEVELS ? level_failures[target_level] : 0;
    }

private:
    using Clock = std::chrono::steady_clock;

    void Stabilize(Clock::time_point now, bool clear_backoffs);
    void RecordLevelFailure(size_t failed_level, Clock::time_point now);
    [[nodiscard]] f32 PlanningInterval(f32 interval_seconds, f32 blocked_seconds, f32 target_rate,
                                       f32 guest_interval_seconds, bool had_generations);
    void LogPlans(f32 interval_seconds, f32 blocked_seconds, size_t generations);
    void DeferEvaluations(Clock::duration amount);
    void UpdateLimit(Clock::time_point now, f32 base_rate, f32 real_rate, f32 target_rate,
                     size_t ceiling);

    std::optional<Clock::time_point> last_frame;
    std::optional<Clock::time_point> stable_until;
    std::optional<Clock::time_point> probe_until;
    std::optional<Clock::time_point> probe_step_until;
    std::optional<Clock::time_point> deficit_since;
    std::optional<Clock::time_point> surplus_since;
    std::optional<Clock::time_point> slowdown_since;
    std::array<std::optional<Clock::time_point>, MAX_TRACKED_LEVELS> level_next_probe{};
    std::array<u32, MAX_TRACKED_LEVELS> level_failures{};
    f32 smoothed_interval{};
    /// The interval generations are planned against (PlanningInterval), smoothed likewise.
    f32 planning_interval{};
    struct {
        u32 frames{};
        u32 generations{};
        f32 interval{};
        f32 blocked{};
    } plan_log;
    f32 output_credit{};
    f32 probe_base_rate{};
    f32 probe_real_rate{};
    f32 active_real_rate{};
    size_t probe_previous_limit{};
    size_t limit{};
    u32 consecutive_slow_frames{};
};

} // namespace Vulkan
