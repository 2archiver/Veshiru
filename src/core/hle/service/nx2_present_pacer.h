// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>

#include "common/game_presets.h"

namespace Service {

struct PresentFenceWaitPlan {
    bool try_adaptive_current{};
    bool wait_previous{};
    bool wait_current{};
};

/**
 * Selects which syncpoint fence(s) nx2:vi2 PresentClientBuffer waits on before displaying.
 *
 * Mario Kart World's GPU channels wait on a CPU semaphore released only after
 * PresentClientBuffer returns, so the adaptive current-frame wait always expires its suspension
 * probe before falling back to the previous frame's fence. When its current fence is not already
 * signaled in adaptive mode, skip the current-frame probe and pace against the previous frame's
 * fence directly without stalling on the blocked current frame.
 */
[[nodiscard]] constexpr PresentFenceWaitPlan PlanPresentFenceWait(
    int frames_ahead, std::uint64_t app_program_id, bool current_fence_ready,
    bool previous_fence_ready) noexcept {
    if (frames_ahead == 0) {
        return {.wait_current = true};
    }
    if (frames_ahead == 1) {
        return {
            .wait_previous = true,
            .wait_current = !previous_fence_ready,
        };
    }
    if (app_program_id == Settings::MarioKartWorldProgramId) {
        if (current_fence_ready) {
            return {};
        }
        return {.wait_previous = true};
    }
    return {.try_adaptive_current = true};
}

/**
 * Returns the interval between paced presents: the vblank the game asked for, lengthened when an
 * explicit FPS limit is below the rate that vblank allows.
 *
 * @param vblank_interval The interval the game's present word asks for, already scaled by the
 *                        speed limit.
 * @param fps_limit       Frames per second to hold, or 0 to leave the game's interval alone.
 * @return The longer of the two intervals.
 */
[[nodiscard]] inline std::chrono::nanoseconds PacedInterval(std::chrono::nanoseconds vblank_interval,
                                                            std::uint32_t fps_limit) {
    if (fps_limit == 0) {
        return vblank_interval;
    }
    const auto limited_interval = std::chrono::nanoseconds{1'000'000'000LL / fps_limit};
    return (std::max)(vblank_interval, limited_interval);
}

class Nx2PresentPacer {
public:
    using Clock = std::chrono::steady_clock;

    struct Step {
        Clock::time_point deadline;
        bool reset;
    };

    /**
     * Returns the deadline for the present arriving at @p now: the previous deadline plus the
     * interval, re-anchored to @p now once the caller has fallen more than one interval behind.
     *
     * @param now      When the present reached the pacer.
     * @param interval Present interval, normally the value from PacedInterval.
     * @return The deadline to hold until, and whether the pacer re-anchored because it was late.
     */
    Step Advance(Clock::time_point now, std::chrono::nanoseconds interval) {
        const auto deadline = last_deadline ? *last_deadline + interval : now;
        const bool reset = !last_deadline || now > deadline + interval;
        last_deadline = reset ? now : deadline;
        return {*last_deadline, reset};
    }

private:
    std::optional<Clock::time_point> last_deadline;
};

} // namespace Service
