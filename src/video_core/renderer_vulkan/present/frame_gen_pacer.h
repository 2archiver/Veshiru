// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <optional>

#include "common/common_types.h"

namespace Vulkan {

struct FrameGenPlan {
    size_t generations{};
    bool warm{};
};

class FrameGenPacer {
public:
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

    /// Consecutive failed probes used for exponential backoff.
    [[nodiscard]] u32 ProbeFailures() const {
        return probe_failures;
    }

private:
    using Clock = std::chrono::steady_clock;

    void Stabilize(Clock::time_point now);
    [[nodiscard]] f32 PlanningInterval(f32 interval_seconds, f32 blocked_seconds, f32 target_rate,
                                       f32 guest_interval_seconds, bool had_generations);
    void LogPlans(f32 interval_seconds, f32 blocked_seconds, size_t generations);
    void DeferEvaluations(Clock::duration amount);
    void UpdateLimit(Clock::time_point now, f32 base_rate, f32 real_rate, f32 target_rate,
                     size_t ceiling);

    std::optional<Clock::time_point> last_frame;
    std::optional<Clock::time_point> stable_until;
    std::optional<Clock::time_point> probe_until;
    std::optional<Clock::time_point> next_probe;
    std::optional<Clock::time_point> deficit_since;
    std::optional<Clock::time_point> surplus_since;
    std::optional<Clock::time_point> slowdown_since;
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
    u32 probe_failures{};
};

} // namespace Vulkan
