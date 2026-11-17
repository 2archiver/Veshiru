// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <optional>

#include "common/common_types.h"

namespace Vulkan {

/// The schedule Frame Pacing Mode holds presents to. Its targets count real game frames, not
/// generated presents: each game frame takes one 1/target_fps slot, and the presents that show it
/// (the generated frames plus the real game frame) share that slot evenly.
class FramePacer {
public:
    using Clock = std::chrono::steady_clock;

    /// Returns when the present may go, or std::nullopt when it is already late and goes at once.
    [[nodiscard]] std::optional<Clock::time_point> Next(Clock::time_point now, double target_fps,
                                                        u32 game_frame_presents) {
        if (last_target_fps != target_fps) {
            frame_interval = std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(1.0 / target_fps));
            max_frames = static_cast<u64>(0.1 * target_fps);
            last_target_fps = target_fps;
            frames = 0.0;
            start_time = now;
        }
        const u32 presents = game_frame_presents > 0 ? game_frame_presents : 1U;
        frames += 1.0 / static_cast<double>(presents);
        const Clock::time_point target_time =
            start_time + std::chrono::duration_cast<Clock::duration>(frame_interval * frames);
        if (target_time >= now) {
            return target_time;
        }
        if (frames > static_cast<double>(max_frames)) {
            frames = 0.0;
            start_time = now;
        }
        return std::nullopt;
    }

private:
    Clock::duration frame_interval{};
    Clock::time_point start_time{};
    double last_target_fps{};
    u64 max_frames{};
    double frames{};
};

} // namespace Vulkan
